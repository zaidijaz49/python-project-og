"""Tests for GroqLLM in rag/answer.py. Groq is faked: no key, no network."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import groq
import httpx
from tenacity import wait_none

from assistant.rag.answer import GroqLLM


def fake_response(text):
    """Looks like a Groq response: response.choices[0].message.content."""
    message = SimpleNamespace(content=text)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def status_error(code):
    """A Groq error with this HTTP status code, e.g. 429 or 503."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(code, request=request)
    return groq.APIStatusError(f"HTTP {code}", response=response, body=None)


def make_llm(*outcomes, models=("test-model",)):
    """GroqLLM with a fake SDK. Each outcome is one call's result or error."""
    sdk = MagicMock()
    sdk.chat.completions.create.side_effect = list(outcomes)
    return GroqLLM(
        sdk, models=list(models), wait=wait_none()
    ), sdk.chat.completions.create


def test_complete_sends_system_and_user_prompt():
    llm, create = make_llm(fake_response("Refunds take 7 working days."))

    answer = llm.complete("Be helpful.", "<question>Refund time?</question>")

    assert answer == "Refunds take 7 working days."
    assert create.call_args.kwargs["messages"] == [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "<question>Refund time?</question>"},
    ]
    assert create.call_args.kwargs["temperature"] == 0


def test_empty_reply_becomes_empty_string():
    llm, _ = make_llm(fake_response(None))

    assert llm.complete("system", "user") == ""


def test_busy_groq_is_retried():
    llm, create = make_llm(status_error(503), fake_response("Hi!"))

    assert llm.complete("system", "user") == "Hi!"
    assert create.call_count == 2


def test_rate_limit_moves_to_next_model():
    llm, create = make_llm(
        status_error(429), fake_response("Hi!"), models=["model-a", "model-b"]
    )

    llm.complete("system", "user")

    assert [c.kwargs["model"] for c in create.call_args_list] == ["model-a", "model-b"]
