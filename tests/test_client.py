"""Tests for GroqClient and MockClient. Groq is faked: no key, no network."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import groq
import httpx
import pytest
from tenacity import wait_none

from assistant.client import (
    DEFAULT_MODELS,
    AIResponseError,
    GroqClient,
    MockClient,
    parse_models,
)
from assistant.models import Analysis

GOOD_JSON = '{"summary": "A good day.", "sentiment": "positive", "tags": ["life"]}'


def fake_response(text):
    """Looks like a Groq response: response.choices[0].message.content."""
    message = SimpleNamespace(content=text)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def status_error(code):
    """A Groq error with this HTTP status code, e.g. 429 or 503."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(code, request=request)
    return groq.APIStatusError(f"HTTP {code}", response=response, body=None)


def make_client(*outcomes, models=("test-model",)):
    """GroqClient with a fake SDK. Each outcome is one call's result or error."""
    sdk = MagicMock()
    sdk.chat.completions.create.side_effect = list(outcomes)
    # wait_none() = no sleeping between retries, so tests stay fast.
    client = GroqClient(sdk, models=list(models), wait=wait_none())
    return client, sdk.chat.completions.create


def models_called(generate):
    """Which model each Groq call used, in order."""
    return [c.kwargs["model"] for c in generate.call_args_list]


def test_success_returns_analysis():
    client, generate = make_client(fake_response(GOOD_JSON))

    result = client.analyze("I had a good day")

    assert result == Analysis(
        summary="A good day.", sentiment="positive", tags=["life"]
    )
    assert generate.call_count == 1
    # Groq's JSON mode: the reply is always JSON, then Pydantic checks it.
    assert generate.call_args.kwargs["response_format"] == {"type": "json_object"}
    # The user text is wrapped in tags so the AI treats it as data only.
    user_message = generate.call_args.kwargs["messages"][1]
    assert user_message == {
        "role": "user",
        "content": "<user_text>\nI had a good day\n</user_text>",
    }


def test_invalid_json_raises_error():
    client, _ = make_client(fake_response("this is not json"))

    with pytest.raises(AIResponseError):
        client.analyze("hello")


def test_empty_response_raises_error():
    client, _ = make_client(fake_response(None))

    with pytest.raises(AIResponseError):
        client.analyze("hello")


def test_retry_then_success():
    client, generate = make_client(
        status_error(503),
        fake_response(GOOD_JSON),
    )

    result = client.analyze("hello")

    assert result.sentiment == "positive"
    assert generate.call_count == 2


def test_retry_on_connection_error_then_success():
    client, generate = make_client(
        httpx.ConnectError("network hiccup"),
        fake_response(GOOD_JSON),
    )

    client.analyze("hello")

    assert generate.call_count == 2


def test_timeout_moves_to_next_model_without_retry():
    # A model that was too slow would be slow again, so don't wait twice.
    client, generate = make_client(
        httpx.ReadTimeout("too slow"),
        fake_response(GOOD_JSON),
        models=["slow-model", "fast-model"],
    )

    client.analyze("hello")

    assert models_called(generate) == ["slow-model", "fast-model"]


def test_504_moves_to_next_model_without_retry():
    client, generate = make_client(
        status_error(504),
        fake_response(GOOD_JSON),
        models=["slow-model", "fast-model"],
    )

    client.analyze("hello")

    assert models_called(generate) == ["slow-model", "fast-model"]


def test_groq_timeout_moves_to_next_model_without_retry():
    request = httpx.Request("POST", "https://api.groq.com")
    client, generate = make_client(
        groq.APITimeoutError(request=request),
        fake_response(GOOD_JSON),
        models=["slow-model", "fast-model"],
    )

    client.analyze("hello")

    assert models_called(generate) == ["slow-model", "fast-model"]


def test_retry_gives_up_after_3_attempts():
    client, generate = make_client(*[status_error(503)] * 3)

    with pytest.raises(groq.APIStatusError):
        client.analyze("hello")

    assert generate.call_count == 3


def test_no_retry_on_401():
    client, generate = make_client(status_error(401))

    with pytest.raises(groq.APIStatusError):
        client.analyze("hello")

    assert generate.call_count == 1


def test_no_retry_on_429_quota_error():
    # 429 = quota used up. Waiting won't help, so only 1 call (no retry).
    client, generate = make_client(status_error(429))

    with pytest.raises(groq.APIStatusError):
        client.analyze("hello")

    assert generate.call_count == 1


# ---------- fallback to the next model ----------


def test_quota_error_moves_to_next_model():
    client, generate = make_client(
        status_error(429),
        fake_response(GOOD_JSON),
        models=["model-a", "model-b"],
    )

    result = client.analyze("hello")

    assert result.sentiment == "positive"
    assert models_called(generate) == ["model-a", "model-b"]


def test_busy_model_is_retried_then_next_model():
    client, generate = make_client(
        *[status_error(503)] * 3,  # model-a: 3 tries, all busy
        fake_response(GOOD_JSON),  # model-b: works
        models=["model-a", "model-b"],
    )

    client.analyze("hello")

    assert models_called(generate) == ["model-a"] * 3 + ["model-b"]


def test_no_fallback_on_401():
    # A bad key fails on every model, so don't waste requests on the others.
    client, generate = make_client(
        status_error(401),
        models=["model-a", "model-b"],
    )

    with pytest.raises(groq.APIStatusError):
        client.analyze("hello")

    assert models_called(generate) == ["model-a"]


def test_all_models_out_of_quota_raises_last_error():
    client, generate = make_client(
        *[status_error(429)] * 3,
        models=["model-a", "model-b", "model-c"],
    )

    with pytest.raises(groq.APIStatusError):
        client.analyze("hello")

    assert models_called(generate) == ["model-a", "model-b", "model-c"]


def test_parse_models():
    assert parse_models("a, b,c") == ["a", "b", "c"]
    assert parse_models("only-one") == ["only-one"]
    assert parse_models("") == DEFAULT_MODELS
    assert parse_models(None) == DEFAULT_MODELS


def test_mock_client_returns_fixed_answer():
    mock = MockClient()

    result = mock.analyze("anything")

    assert result.sentiment == "neutral"
    assert mock.calls == ["anything"]
