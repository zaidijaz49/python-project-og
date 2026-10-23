"""Shared Groq helpers: build the SDK from .env, retry hiccups, fall back to
the next model. Used by the chat agent and by the RAG answer step."""

import logging
import os
from collections.abc import Callable
from typing import TypeVar

import groq
import httpx
from dotenv import load_dotenv
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
)
from tenacity.wait import wait_base

logger = logging.getLogger("assistant")

# Tried in this order. If one is busy or out of quota, the next one is used.
DEFAULT_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
REQUEST_TIMEOUT_SECONDS = 30
MAX_ATTEMPTS = 3

# Quick server hiccups that usually fix themselves: retry the SAME model.
TRANSIENT_STATUS_CODES = {500, 502, 503}

# Errors where retrying the same model doesn't help, so we move on to the
# NEXT model right away:
#   429 = this model's rate limit or quota is used up.
#   498 = Groq has no free capacity for this model right now.
#   504 = the model was too slow and timed out; it would be slow again.
NEXT_MODEL_STATUS_CODES = {429, 498, 504}

T = TypeVar("T")


def is_transient(exc: BaseException) -> bool:
    """True if the error is a quick hiccup worth retrying on the same model."""
    if isinstance(exc, groq.APIStatusError):
        return exc.status_code in TRANSIENT_STATUS_CODES
    if isinstance(exc, groq.APITimeoutError):
        return False  # too slow: handled by should_try_next_model
    return isinstance(
        exc, (groq.APIConnectionError, httpx.NetworkError, ConnectionError)
    )


def should_try_next_model(exc: BaseException) -> bool:
    """True if another model might work: quota used up, too slow, or still failing.

    Not for errors like 400 (bad request) or 401/403 (key problem):
    every model would fail the same way, so trying them only wastes requests.
    """
    if isinstance(exc, groq.APIStatusError):
        if exc.status_code in NEXT_MODEL_STATUS_CODES:
            return True
    if isinstance(exc, (groq.APITimeoutError, httpx.TimeoutException, TimeoutError)):
        return True  # too slow: don't wait again, try the next model
    return is_transient(exc)


def make_retryer(wait: wait_base, max_attempts: int) -> Retrying:
    """Retry rules shared by GroqAgent and GroqLLM."""
    return Retrying(
        retry=retry_if_exception(is_transient),
        stop=stop_after_attempt(max_attempts),
        wait=wait,
        reraise=True,  # after the last attempt, raise the real error
    )


def try_models(models: list[str], call: Callable[[str], T]) -> tuple[T, str]:
    """Run call(model) for each model in order until one works.

    Returns (result, model_that_worked). Raises the last error if all fail.
    """
    for i, model in enumerate(models):
        try:
            return call(model), model
        except Exception as exc:
            is_last_model = i == len(models) - 1
            if is_last_model or not should_try_next_model(exc):
                raise
            logger.warning(
                "Model %s failed (%s). Trying %s.", model, exc, models[i + 1]
            )
    raise ValueError("No models to try.")


def parse_models(raw: str | None) -> list[str]:
    """'a, b,c' -> ['a', 'b', 'c']. Empty or missing -> DEFAULT_MODELS."""
    models = [m.strip() for m in (raw or "").split(",") if m.strip()]
    return models or list(DEFAULT_MODELS)


def sdk_from_env() -> tuple[groq.Groq, list[str]]:
    """Read the key and model list from .env and build the Groq SDK.

    GROQ_MODELS=a,b,c sets the fallback order. GROQ_MODEL (one model) still
    works too. If neither is set, DEFAULT_MODELS is used.
    """
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
    models = parse_models(os.getenv("GROQ_MODELS") or os.getenv("GROQ_MODEL"))
    # max_retries=0: the SDK must not retry on its own, we do that ourselves.
    sdk = groq.Groq(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0)
    return sdk, models
