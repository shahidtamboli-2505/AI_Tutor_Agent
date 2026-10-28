"""Groq wrapper: ask() and ask_json(). Reads the key from the environment only. No Streamlit here."""
import json
import os
import re
import time
from typing import List, Optional, Type, TypeVar

from groq import BadRequestError, Groq, NotFoundError, RateLimitError
from pydantic import BaseModel, ValidationError

import config
from backend import usage

T = TypeVar("T", bound=BaseModel)

TEACHER_SYSTEM = (
    "You are the best teacher in the world: warm, patient, and encouraging. You never scold. "
    "You say 'not yet' instead of 'failed'. Use simple words that fit the student's level. "
    "Use only the provided passages when teaching; if the passages do not say it, say you do not know."
)
JSON_RULE = "Reply with ONE valid JSON object only. No explanation and no markdown. Use exactly the keys given."


class LLMError(Exception):
    """Raised with a friendly message that is safe to show to the student."""


_client: Optional[Groq] = None
_models: List[str] = []


def clean_text(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in the reply.")
    return json.loads(text[start:end + 1])


def has_key() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


def get_client() -> Groq:
    global _client
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise LLMError("No API key found. Add GROQ_API_KEY to .streamlit/secrets.toml (or the app's Secrets).")
    if _client is None:
        _client = Groq(api_key=key)
    return _client


def available_chat_models() -> List[str]:
    global _models
    if _models:
        return _models
    try:
        ids = [m.id for m in get_client().models.list().data]
    except LLMError:
        raise
    except Exception as e:  # never include the key in messages
        raise LLMError(f"I could not reach the model service ({type(e).__name__}). Check your key and internet.")
    ids = [i for i in ids if not any(w in i.lower() for w in config.BLOCKED_MODEL_WORDS)]
    chosen = [m for m in config.PREFERRED_MODELS if m in ids] or sorted(ids)
    if not chosen:
        raise LLMError("No chat models are available for this key.")
    _models = chosen
    return _models


def ask(prompt: str, system: Optional[str] = None, history: Optional[list] = None,
        temperature: float = 0.7, json_mode: bool = False, max_tokens: int = 2000, retries: int = 6) -> str:
    models = available_chat_models()
    idx, use_json = 0, json_mode
    messages = ([{"role": "system", "content": system}] if system else []) + list(history or [])
    messages.append({"role": "user", "content": prompt})
    attempt = 0
    while attempt < retries:
        kwargs = dict(model=models[idx], messages=messages, temperature=temperature, max_tokens=max_tokens)
        if use_json:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = get_client().chat.completions.create(**kwargs)
            answer = clean_text(resp.choices[0].message.content)
            # Groq's own counts: prompt_tokens -> input, completion_tokens -> output; characters from the text
            usage.record(models[idx], getattr(resp, "usage", None), answer)
            return answer
        except RateLimitError:
            time.sleep(15 * (attempt + 1))      # "The teacher is thinking, please wait"
        except (NotFoundError, BadRequestError):
            if use_json:
                use_json = False
            elif idx + 1 < len(models):
                idx += 1
            else:
                raise LLMError("The model could not answer this request. Please try again.")
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"Something went wrong while talking to the model ({type(e).__name__}).")
        attempt += 1
    raise LLMError("The teacher is very busy right now. Please wait a minute and try again.")


def ask_json(prompt: str, model_cls: Type[T], system: Optional[str] = None,
             tries: int = 3, max_tokens: int = 3000) -> T:
    base = f"{prompt}\n\n{JSON_RULE}"
    last_error = ""
    for _ in range(tries):
        p = base if not last_error else f"{base}\n\nYour last reply was not valid: {last_error}\nFix it and reply again."
        raw = ask(p, system=system, temperature=0.2, json_mode=True, max_tokens=max_tokens)
        try:
            return model_cls.model_validate(extract_json(raw))
        except (ValueError, ValidationError) as e:
            last_error = str(e)[:600]
    raise LLMError("The teacher could not prepare a clean answer. Please try again.")
