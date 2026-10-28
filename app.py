import os
import re
import hashlib
from pathlib import Path

import streamlit as st

# Streamlit Community Cloud stores secrets in st.secrets. The existing backend
# reads GROQ_API_KEY from the environment, so bridge the two safely here.
try:
    if st.secrets.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

from backend import documents, topics, tutor, storage, lessons, usage
from backend.llm import LLMError
from backend.schemas import Course, StudentProfile

st.set_page_config(page_title="AI Tutor Agent", page_icon="🎓", layout="wide")
st.title("🎓 AI Tutor Agent")
st.caption("A document-grounded learning assistant that turns your study material into topics, lessons, and answers.")

with st.sidebar:
    st.header("Learner profile")
    student_name = st.text_input("Your name", value="Student")
    level = st.selectbox("Current level", ["beginner", "intermediate", "advanced"], index=0)
    goal = st.text_input("Short-term learning goal", value="Understand the uploaded material")
    long_goal = st.text_input("Long-term goal", value="Build strong subject knowledge")
    prior = st.text_area("What do you already know?", value="I am learning this topic.", height=75)
    daily_minutes = st.slider("Study time per day (minutes)", 5, 120, 30, 5)
    styles = st.multiselect("What helps you learn?", ["examples", "step-by-step explanations", "practice questions", "stories", "summaries"], default=["examples", "practice questions"])
    interests = st.text_input("Interests (comma-separated)", value="technology, everyday life")

profile = StudentProfile(
    student_id="streamlit_" + hashlib.sha1(student_name.strip().lower().encode()).hexdigest()[:10],
    name=student_name.strip() or "Student",
    short_term_goal=goal,
    long_term_goal=long_goal,
    level=level,
    prior_knowledge=prior,
    daily_minutes=daily_minutes,
    learning_styles=styles or ["examples"],
    interests=[x.strip() for x in interests.split(",") if x.strip()],
    notes="Created through the Streamlit interface",
)

if "course" not in st.session_state:
    st.session_state.course = None
if "spans" not in st.session_state:
    st.session_state.spans = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_lesson" not in st.session_state:
    st.session_state.active_lesson = None

st.subheader("1. Upload your study material")
uploaded = st.file_uploader("Choose a text-based PDF or TXT file", type=["pdf", "txt"])
if uploaded is not None:
    st.caption(f"Selected file: {uploaded.name} · {uploaded.size:,} bytes")
    if st.button("Analyze document and create course", type="primary"):
        if not os.environ.get("GROQ_API_KEY"):
            st.error("Groq API key is missing. Add GROQ_API_KEY in Streamlit app Settings → Secrets before using AI features.")
        else:
            try:
                with st.status("Preparing your course…", expanded=True) as status:
                    pages = documents.read_document(uploaded.getvalue(), uploaded.name)
                    documents.check_quality(pages)
                    spans = documents.make_spans(pages)
                    st.write(f"Extracted {len(spans)} source passages.")
                    progress = st.progress(0)
                    msg = st.empty()
                    def update(p, text):
                        progress.progress(min(1.0, max(0.0, float(p))))
                        msg.write(text)
                    found_topics = topics.find_topics(spans, progress_callback=update)
                    if not found_topics:
                        raise ValueError("No topics were found in this document. Try a PDF with selectable text or a TXT file.")
                    course_title = topics.make_course_title(found_topics)
                    raw_id = re.sub(r"[^a-z0-9]+", "-", course_title.lower()).strip("-")[:40] or "my-course"
                    course_id = raw_id + "-" + hashlib.sha1(uploaded.getvalue()).hexdigest()[:8]
                    course = Course(course_id=course_id, course_title=course_title, topics=found_topics)
                    cdir = storage.course_dir(course_id)
                    storage.save_json(cdir / "topics.json", course)
                    storage.save_json(cdir / "source_spans.json", spans)
                    st.session_state.course = course
                    st.session_state.spans = spans
                    st.session_state.chat_history = []
                    st.session_state.active_lesson = None
                    status.update(label="Course created successfully", state="complete", expanded=False)
            except (documents.DocumentError, LLMError, ValueError) as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Something went wrong ({type(e).__name__}). Check your app logs and configuration.")

course = st.session_state.course
spans = st.session_state.spans
if course:
    st.divider()
    st.subheader(f"2. Your course: {course.course_title}")
    st.write(f"Found **{len(course.topics)} topics** across **{len(spans)} source passages**.")
    with st.expander("View learning topics", expanded=True):
        for t in course.topics:
            st.markdown(f"**{t.id}. {t.title}** · {t.difficulty.title()}")
            st.write(t.summary)
            st.caption("Source passages: " + (", ".join(t.source_spans) if t.source_spans else "Not specified"))

    tab_chat, tab_lesson, tab_about = st.tabs(["Ask the tutor", "Generate a lesson", "Usage & notes"])
    with tab_chat:
        st.markdown("Ask questions about the uploaded material. Answers are intended to stay grounded in the document.")
        for m in st.session_state.chat_history:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        prompt = st.chat_input("Ask a question about your study material…")
        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Looking through your material…"):
                    try:
                        history = st.session_state.chat_history[:-1]
                        answer = tutor.answer_subject(prompt, spans, course.course_title, profile.level, history)
                    except LLMError as e:
                        answer = str(e)
                    except Exception as e:
                        answer = f"I couldn't answer that just now ({type(e).__name__}). Please try again."
                st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    with tab_lesson:
        st.markdown("Generate a structured lesson from a topic in your uploaded document. This may use several API calls.")
        topic_by_title = {f"{t.id} — {t.title}": t for t in course.topics}
        selected = st.selectbox("Choose a topic", list(topic_by_title.keys()))
        variant_label = st.selectbox("Lesson style", ["Full lesson", "Short lesson", "Misconception-focused", "Prerequisite bridge"])
        variant_map = {"Full lesson": "full", "Short lesson": "short", "Misconception-focused": "misconception", "Prerequisite bridge": "bridge"}
        if st.button("Generate lesson", type="primary"):
            if not os.environ.get("GROQ_API_KEY"):
                st.error("Groq API key is missing. Add it in Streamlit app Settings → Secrets.")
            else:
                try:
                    with st.spinner("Building your lesson and practice questions…"):
                        lesson, review_result = lessons.get_or_create_lesson(course.course_id, topic_by_title[selected], spans, profile, variant_map[variant_label])
                        st.session_state.active_lesson = (lesson, review_result)
                except LLMError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Lesson generation failed ({type(e).__name__}). Check the app logs.")
        if st.session_state.active_lesson:
            lesson, review_result = st.session_state.active_lesson
            st.markdown(f"### {topic_by_title[selected].title}")
            st.info(lesson.hook)
            st.markdown("**Learning objectives**")
            for obj in lesson.objectives:
                st.markdown(f"- {obj}")
            st.markdown("**Explanation**")
            for part in lesson.explanation:
                st.markdown(part)
            st.markdown("**Worked example**")
            st.write(lesson.worked_example)
            st.markdown("**Another example**")
            st.write(lesson.extra_worked_example)
            st.markdown("**Practice questions**")
            for i, q in enumerate(lesson.practice, 1):
                st.markdown(f"**{i}. {q.text}**")
                for j, option in enumerate(q.options):
                    st.write(f"{chr(65+j)}. {option}")
                with st.expander("Show answer and explanation"):
                    st.success(f"Correct answer: {q.options[q.correct_index]}")
                    if q.correct_index < len(q.option_explanations):
                        st.write(q.option_explanations[q.correct_index])
            st.markdown("**Summary**")
            st.write(lesson.summary)
            st.caption(f"Structural review status: {review_result.decision}. This checks structure, not guaranteed factual accuracy.")

    with tab_about:
        snap = usage.snapshot()
        a, b, c = st.columns(3)
        a.metric("Successful model calls", snap.get("calls", 0))
        b.metric("Input tokens", snap.get("input_tokens", 0))
        c.metric("Output tokens", snap.get("output_tokens", 0))
        st.warning("Streamlit Community Cloud's local filesystem may reset or be ephemeral. This prototype stores course data as JSON files; use a persistent database for reliable multi-user production use.")
else:
    st.info("Upload a PDF or TXT file above, then select **Analyze document and create course** to get started.")

st.divider()
st.caption("AI Tutor Agent · Learning support only. Verify important information with your instructor or trusted course materials.")
