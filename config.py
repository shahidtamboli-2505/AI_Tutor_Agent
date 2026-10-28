"""Settings, limits and thresholds. All values are provisional: tune after testing with real students."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Chat models only, best first. Filtered at startup against what your key can list.
PREFERRED_MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",   # extra fallbacks so the demo still runs if names change
    "llama-3.1-8b-instant",
]
BLOCKED_MODEL_WORDS = ("whisper", "orpheus", "prompt-guard", "safeguard", "guard", "tts", "playai")

# Documents and topics
MIN_TOTAL_CHARS = 300
MIN_AVG_CHARS = 80
SPAN_TARGET = 1200
SPAN_MAX = 1800
WINDOW_CHARS = 9000
MAX_WINDOWS = 6
MAX_TOPICS_SINGLE = 8
MAX_TOPICS_PER_WINDOW = 4
PASSAGE_CHARS_PER_CALL = 3500

# Pre-test: how many questions, and how the material limits it
PRETEST_MAX_QUESTIONS = 10
PRETEST_MIN_TARGET = 5            # aimed for whenever the material is long enough
PRETEST_CHARS_PER_QUESTION = 300  # roughly one question per this much passage text
PRETEST_DRAFT_BATCH = 5           # questions asked from the model per call
PRETEST_ROUNDS = 2                # draft + verify, then one top-up round for rejected questions
VERIFY_PASSAGE_CHARS = 9000       # passage text sent to the answer checker per call
DUPLICATE_SIMILARITY = 0.7        # two questions this similar (word overlap) count as duplicates

# Pre-test score bands: (upper limit, key). Accuracy is rounded to 1 decimal first, the same number the student sees.
SCORE_BANDS = [(40.0, "fundamentals"), (70.0, "core"), (90.0, "revise"), (100.1, "advanced")]

# Pre-test and path (minutes per lesson type)
PATH_MINUTES = {"full_lesson": 25, "short_path": 12, "skip_with_check": 3,
                "misconception_lesson": 25, "bridge_first": 15}

# Lessons
OBJECTIVES_PER_LESSON = 3
VARIANTS_PER_OBJECTIVE = 3
MAX_REVISIONS = 1            # spec says 2; kept at 1 for the MVP to save free-tier calls
QUIZ_PASS_RATIO = 0.66

# Daily targets
TARGET_STEP_MINUTES = 5
MIN_TARGET_MINUTES = 5
KIND_MESSAGE_AFTER_MISSED_DAYS = 3

# Support
RUSH_SECONDS_PER_QUESTION = 8
