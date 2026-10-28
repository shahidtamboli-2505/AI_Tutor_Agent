"""Shared question maker used by the pre-test (Stage 4) and lessons (Stage 5)."""
import random
from typing import Dict, List, Sequence, Tuple

from pydantic import BaseModel

import config
from backend import llm, retrieval
from backend.schemas import Question, Span, Topic

Q_SHAPE = ('{"id":"q1","topic_id":"T1","text":"question","options":["A","B","C","D"],"correct_index":0,'
           '"option_explanations":["why option A is right or wrong","...","...","..."],'
           '"kind":"recall|explain|apply|compare|choose_and_justify","hint":"a gentle nudge",'
           '"source_spans":["S1"],"objective_index":0}')


class QuestionList(BaseModel):
    questions: List[Question]


def passages_text(topic: Topic, spans: List[Span]) -> str:
    by_id = {s.id: s for s in spans}
    out, size = [], 0
    for sid in topic.source_spans:
        if sid in by_id:
            line = f"[{sid}] {by_id[sid].text}"
            if out and size + len(line) > config.PASSAGE_CHARS_PER_CALL:
                break
            out.append(line)
            size += len(line)
    return "\n\n".join(out)


def question_problems(q: Question) -> List[str]:
    problems = []
    opts = [o.strip().lower() for o in q.options]
    if len(set(opts)) != 4 or "" in opts:
        problems.append("options must be four different, non-empty answers")
    if any(not e.strip() for e in q.option_explanations):
        problems.append("every option needs an explanation")
    if not q.text.strip():
        problems.append("question text is empty")
    return problems


def finalize(q: Question, topic: Topic, qid: str, rng: random.Random) -> Question:
    """Set ids, keep only valid passage ids, and shuffle options so the answer is not always in the same place."""
    order = list(range(4))
    rng.shuffle(order)
    spans = [s for s in q.source_spans if s in topic.source_spans] or list(topic.source_spans)
    return q.model_copy(update={
        "id": qid, "topic_id": topic.id, "source_spans": spans,
        "options": [q.options[i] for i in order],
        "option_explanations": [q.option_explanations[i] for i in order],
        "correct_index": order.index(q.correct_index),
    })


def ask_questions(task: str, topic: Topic, spans: List[Span], prefix: str, count: int) -> List[Question]:
    """Ask the model for `count` questions on one topic, using only that topic's passages. Retries once."""
    rng = random.Random()
    prompt = (f"{task}\n\nEach question has exactly 4 options and ONE correct answer. Wrong options should reflect common "
              "mistakes. Give a short explanation for EVERY option (why it is right or why it is wrong). "
              "Use ONLY the passages. Question shape:\n" + Q_SHAPE +
              f'\nOutput {{"questions":[...]}} with exactly {count} questions.\n\nPASSAGES:\n{passages_text(topic, spans)}')
    good: List[Question] = []
    for _ in range(2):
        batch = llm.ask_json(prompt, QuestionList, system=llm.TEACHER_SYSTEM, max_tokens=5000).questions
        good = [q for q in batch if not question_problems(q)]
        if len(good) >= count:
            break
        prompt += "\nSome questions were invalid (options must be unique and every option needs an explanation). Try again."
    if not good:
        raise llm.LLMError("The teacher could not prepare questions this time. Please try again.")
    return [finalize(q, topic, f"{prefix}{i + 1}", rng) for i, q in enumerate(good[:count])]


# ---------- checks used by the pre-test ----------

def is_duplicate(text: str, others: Sequence[str]) -> bool:
    """True when the question is (nearly) the same as one we already have: same normalised text or high word overlap."""
    a = set(retrieval.tokens(text))
    norm = " ".join((text or "").lower().split())
    for o in others:
        if norm == " ".join((o or "").lower().split()):
            return True
        b = set(retrieval.tokens(o))
        if a and b and len(a & b) / len(a | b) >= config.DUPLICATE_SIMILARITY:
            return True
    return False


class Verdict(BaseModel):
    id: str
    answer_index: int = -1               # the option the checker finds correct (0 to 3), -1 if the passages do not settle it
    only_one_correct: bool = True        # False when more than one option could be defended
    supported: bool = True               # False when the passages cannot answer the question


class VerdictList(BaseModel):
    verdicts: List[Verdict]


def _question_spans(q: Question, topics: Dict[str, Topic], by_span: Dict[str, Span]) -> List[Span]:
    ids = [i for i in q.source_spans if i in by_span]
    if not ids and q.topic_id in topics:
        ids = [i for i in topics[q.topic_id].source_spans if i in by_span]
    return [by_span[i] for i in ids]


def _batches(questions: List[Question], topics: Dict[str, Topic], by_span: Dict[str, Span]) -> List[List[Question]]:
    """Group questions so the passages sent in one call stay within VERIFY_PASSAGE_CHARS."""
    out, cur, seen, size = [], [], set(), 0
    for q in questions:
        new = [s for s in _question_spans(q, topics, by_span) if s.id not in seen]
        add = sum(len(s.text) for s in new)
        if cur and size + add > config.VERIFY_PASSAGE_CHARS:
            out.append(cur)
            cur, seen, size = [], set(), 0
            new = _question_spans(q, topics, by_span)
            add = sum(len(s.text) for s in new)
        cur.append(q)
        seen.update(s.id for s in new)
        size += add
    if cur:
        out.append(cur)
    return out


def verify_questions(questions: List[Question], topics: Sequence[Topic], spans: Sequence[Span]) -> List[Question]:
    """Ask the model to answer each question WITHOUT seeing the answer key, using only the passages.
    Keep a question only if the checker picks the same option, finds only one defensible option, and says the passages
    support it. Kept questions are marked verified=True. Raises LLMError if the check cannot run (never keeps unchecked
    questions silently)."""
    if not questions:
        return []
    tmap, by_span = {t.id: t for t in topics}, {s.id: s for s in spans}
    kept: List[Question] = []
    for batch in _batches(questions, tmap, by_span):
        used: Dict[str, Span] = {}
        for q in batch:
            for sp in _question_spans(q, tmap, by_span):
                used[sp.id] = sp
        passages = "\n\n".join(f"[{sp.id}] {sp.text}" for sp in used.values())[: config.VERIFY_PASSAGE_CHARS * 2]
        listing = "\n\n".join(
            f"ID: {q.id}\nQuestion: {q.text}\n" + "\n".join(f"{i}. {o}" for i, o in enumerate(q.options)) for q in batch)
        prompt = (
            "You are checking quiz questions. For EACH question, using ONLY the passages, decide which option is correct.\n"
            "- answer_index: the number (0 to 3) of the correct option, or -1 if the passages do not settle it.\n"
            "- only_one_correct: false if a careful student could defend more than one option, or none.\n"
            "- supported: false if the passages do not contain what is needed to answer the question.\n"
            'Output {"verdicts":[{"id":"...","answer_index":0,"only_one_correct":true,"supported":true}]} with one entry '
            f"per question.\n\nPASSAGES:\n{passages}\n\nQUESTIONS:\n{listing}")
        got = {v.id: v for v in llm.ask_json(prompt, VerdictList, system=llm.TEACHER_SYSTEM, max_tokens=1500).verdicts}
        for q in batch:
            v = got.get(q.id)
            if v and v.supported and v.only_one_correct and v.answer_index == q.correct_index:
                kept.append(q.model_copy(update={"verified": True}))
    return kept
