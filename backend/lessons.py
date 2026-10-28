"""Stage 5: lesson maker with the nine-step blueprint, cached on disk."""
import random
from typing import List, Tuple

from pydantic import BaseModel

import config
from backend import llm, review, storage
from backend import questions as qgen
from backend.schemas import Lesson, Question, ReviewResult, Span, StudentProfile, Topic

VARIANT_FOR_STATE = {"full_lesson": "full", "short_path": "short", "skip_with_check": "short",
                     "misconception_lesson": "misconception", "bridge_first": "bridge"}

VARIANT_RULES = {
    "full": "A complete lesson from the beginning.",
    "short": "A short lesson. Skip the hook (use an empty string) and go straight to the key missing ideas.",
    "misconception": "Open the hook with an example that shows why a common wrong belief about this topic fails, then teach the correct idea.",
    "bridge": "A gentle bridge lesson: only the basic ideas a beginner needs first. Very small steps and very simple words.",
}


class _Core(BaseModel):
    hook: str
    objectives: List[str]
    explanation: List[str]
    worked_example: str
    extra_worked_example: str
    summary: str


class _Practice(BaseModel):
    warmup: List[Question] = []
    practice: List[Question]
    transfer_question: Question


def _context(profile: StudentProfile) -> str:
    return (f"Student level: {profile.level}. Interests: {', '.join(profile.interests) or 'not given'}. "
            f"Goal: {profile.short_term_goal or 'not given'}. Likes to learn by: {', '.join(profile.learning_styles) or 'not given'}.")


def generate_lesson(topic: Topic, spans: List[Span], profile: StudentProfile, variant: str = "full") -> Lesson:
    text, ctx = qgen.passages_text(topic, spans), _context(profile)
    n_obj = 2 if variant == "short" else config.OBJECTIVES_PER_LESSON
    per = config.VARIANTS_PER_OBJECTIVE

    core = llm.ask_json(
        f"Write the teaching part of a lesson on: {topic.title} - {topic.summary}\n{VARIANT_RULES[variant]}\n{ctx}\n"
        "Use simple words for this level and use the student's interests in examples where it fits.\n"
        f'Output {{"hook":"a real problem or curious question","objectives":["By the end you will be able to ..."] (exactly {n_obj}),'
        '"explanation":["short segment, one idea each, key terms in **bold**"] (3 to 5 segments),'
        '"worked_example":"solved step by step with the reasoning shown",'
        '"extra_worked_example":"a different example in a different context",'
        '"summary":"two or three sentences"}\n'
        f"Use ONLY the passages. If they do not say it, leave it out.\n\nPASSAGES:\n{text}",
        _Core, system=llm.TEACHER_SYSTEM, max_tokens=4000)
    objectives = core.objectives[:n_obj]
    obj_text = "; ".join(f"({i}) {o}" for i, o in enumerate(objectives))

    part = llm.ask_json(
        f"Write practice questions for a lesson on: {topic.title}. Objectives: {obj_text}\n{ctx}\n"
        f"Output {{\"warmup\":[{'2 easy recall questions on what the student may already know' if variant != 'short' else 'empty list'}],"
        '"practice":[4 questions from easy to harder; at least 2 must have kind "apply" using a NEW case],'
        '"transfer_question":one question in a brand new situation}\nEach question uses this shape: ' + qgen.Q_SHAPE +
        f"\nOptions must be 4 different answers; give an explanation for EVERY option. Use ONLY the passages.\n\nPASSAGES:\n{text}",
        _Practice, system=llm.TEACHER_SYSTEM, max_tokens=6000)
    rng = random.Random()

    def rename(qs: List[Question], prefix: str) -> List[Question]:
        return [qgen.finalize(q, topic, f"{topic.id}-{variant}-{prefix}{i + 1}", rng) for i, q in enumerate(qs)]

    quiz = qgen.ask_questions(f"Write {n_obj} quiz questions, one for each objective in order (objective_index 0 to {n_obj - 1}). "
                              f"Topic: {topic.title}. Objectives: {obj_text}", topic, spans, f"{topic.id}-{variant}-qz", n_obj)
    for i, q in enumerate(quiz):
        quiz[i] = q.model_copy(update={"objective_index": i})
    variants = qgen.ask_questions(
        f"Write {per * n_obj} extra quiz questions for retries: the first {per} for objective 0, the next {per} for objective 1"
        f"{', and so on' if n_obj > 2 else ''}. They must be different from each other. "
        f"Topic: {topic.title}. Objectives: {obj_text}", topic, spans, f"{topic.id}-{variant}-rt", per * n_obj)
    for i, q in enumerate(variants):
        variants[i] = q.model_copy(update={"objective_index": min(i // per, n_obj - 1)})

    return Lesson(
        topic_id=topic.id, variant=variant, hook=core.hook if variant != "short" else "", objectives=objectives,
        warmup=rename(part.warmup[:2], "wu") if variant != "short" else [], explanation=core.explanation,
        worked_example=core.worked_example, extra_worked_example=core.extra_worked_example,
        practice=rename(part.practice[:4], "pr"), quiz=quiz, quiz_variants=variants, summary=core.summary,
        transfer_question=rename([part.transfer_question], "tr")[0], source_spans=list(topic.source_spans))


def get_or_create_lesson(course_id: str, topic: Topic, spans: List[Span], profile: StudentProfile,
                         variant: str = "full") -> Tuple[Lesson, ReviewResult]:
    """Load the lesson from disk if it exists; otherwise generate, check, retry, and save."""
    base = storage.course_dir(course_id) / "lessons"
    lpath, rpath = base / f"{topic.id}_{variant}.json", base / f"{topic.id}_{variant}.review.json"
    if lpath.exists() and rpath.exists():
        return Lesson.model_validate(storage.load_json(lpath)), ReviewResult.model_validate(storage.load_json(rpath))
    lesson, result = None, None
    for attempt in range(config.MAX_REVISIONS + 1):
        lesson = generate_lesson(topic, spans, profile, variant)
        problems = review.structural_checks(lesson, topic)
        decision = "pass" if not problems else ("revise" if attempt < config.MAX_REVISIONS else "person_review")
        result = ReviewResult(topic_id=topic.id, structural_ok=not problems, structural_problems=problems,
                              decision=decision, attempt=attempt)
        if not problems:
            break
    storage.save_json(lpath, lesson)
    storage.save_json(rpath, result)
    return lesson, result
