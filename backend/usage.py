"""Token and character usage for the whole app. No Streamlit here.

Nothing is estimated. After every successful Groq answer, llm.ask() hands in the `usage` numbers Groq returned:
prompt_tokens are added to INPUT and completion_tokens are added to OUTPUT. Characters generated are counted
separately, from the length of the answer text. Failed or retried calls (rate limits, errors) return no answer
and are not counted. Totals live in data/usage.json, so they survive page reloads."""
import json
import threading
import time
from typing import Any, Dict, Optional

import config

MAX_LOG = 200                      # newest answers kept for the table and charts
_LOCK = threading.Lock()


def _path():
    return config.DATA_DIR / "usage.json"


def _empty() -> Dict[str, Any]:
    return {"input_tokens": 0, "output_tokens": 0, "chars": 0, "calls": 0,
            "since": time.time(), "by_model": {}, "log": []}


def _load() -> Dict[str, Any]:
    state = _empty()
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
        if isinstance(data, dict):
            state.update({k: v for k, v in data.items() if k in state})
    except (OSError, ValueError):
        pass
    return state


def _save(state: Dict[str, Any]) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    tmp.replace(p)


def _count(usage_obj: Any, name: str) -> int:
    value = usage_obj.get(name) if isinstance(usage_obj, dict) else getattr(usage_obj, name, 0)
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def record(model: str, usage_obj: Optional[Any], text: str) -> None:
    """Add ONE successful answer to the totals. Never raises: counting must not break the teacher."""
    try:
        prompt_tokens = _count(usage_obj, "prompt_tokens")           # -> input
        completion_tokens = _count(usage_obj, "completion_tokens")   # -> output
        chars = len(text or "")                                      # counted from the response text
        with _LOCK:
            s = _load()
            s["input_tokens"] += prompt_tokens
            s["output_tokens"] += completion_tokens
            s["chars"] += chars
            s["calls"] += 1
            m = s["by_model"].setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "chars": 0})
            m["calls"] += 1
            m["input_tokens"] += prompt_tokens
            m["output_tokens"] += completion_tokens
            m["chars"] += chars
            s["log"] = (s["log"] + [{"time": time.time(), "model": model, "input_tokens": prompt_tokens,
                                     "output_tokens": completion_tokens, "chars": chars}])[-MAX_LOG:]
            _save(s)
    except Exception:
        pass


def snapshot() -> Dict[str, Any]:
    """Current totals (read fresh from disk, so every page and process sees the same numbers)."""
    s = _load()
    s["total_tokens"] = s["input_tokens"] + s["output_tokens"]
    return s


def reset() -> None:
    with _LOCK:
        _save(_empty())
