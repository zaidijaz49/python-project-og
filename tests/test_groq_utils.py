"""Tests for the shared Groq helpers: retry rules and model fallback."""

from unittest.mock import MagicMock

import groq
import httpx
import pytest

from assistant.groq_utils import (
    DEFAULT_MODELS,
    is_transient,
    parse_models,
    should_try_next_model,
    try_models,
)


def status_error(code):
    """A Groq error with this HTTP status code, e.g. 429 or 503."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(code, request=request)
    return groq.APIStatusError(f"HTTP {code}", response=response, body=None)


def groq_timeout():
    return groq.APITimeoutError(request=httpx.Request("POST", "https://api.groq.com"))


# ---------- is_transient: retry the SAME model? ----------


@pytest.mark.parametrize("code", [500, 502, 503])
def test_server_hiccups_are_retried(code):
    assert is_transient(status_error(code))


@pytest.mark.parametrize("code", [400, 401, 403, 429])
def test_other_errors_are_not_retried(code):
    assert not is_transient(status_error(code))


def test_connection_error_is_retried():
    request = httpx.Request("POST", "https://api.groq.com")
    assert is_transient(groq.APIConnectionError(request=request))
    assert is_transient(httpx.ConnectError("network hiccup"))


def test_timeout_is_not_retried_on_same_model():
    # A model that was too slow would be slow again, so don't wait twice.
    assert not is_transient(groq_timeout())


# ---------- should_try_next_model: move to the NEXT model? ----------


@pytest.mark.parametrize("code", [429, 498, 504, 503])
def test_these_errors_move_to_next_model(code):
    assert should_try_next_model(status_error(code))


def test_timeout_moves_to_next_model():
    assert should_try_next_model(groq_timeout())
    assert should_try_next_model(httpx.ReadTimeout("too slow"))


@pytest.mark.parametrize("code", [400, 401, 403])
def test_key_or_request_errors_do_not_move_to_next_model(code):
    # A bad key fails on every model, so trying the others only wastes requests.
    assert not should_try_next_model(status_error(code))


# ---------- try_models ----------


def test_first_model_works():
    call = MagicMock(return_value="answer")

    assert try_models(["model-a", "model-b"], call) == ("answer", "model-a")
    call.assert_called_once_with("model-a")


def test_quota_error_moves_to_next_model():
    call = MagicMock(side_effect=[status_error(429), "answer"])

    assert try_models(["model-a", "model-b"], call) == ("answer", "model-b")


def test_no_fallback_on_401():
    call = MagicMock(side_effect=[status_error(401)])

    with pytest.raises(groq.APIStatusError):
        try_models(["model-a", "model-b"], call)

    assert call.call_count == 1


def test_all_models_failing_raises_last_error():
    call = MagicMock(side_effect=[status_error(429)] * 3)

    with pytest.raises(groq.APIStatusError):
        try_models(["model-a", "model-b", "model-c"], call)

    assert call.call_count == 3


def test_parse_models():
    assert parse_models("a, b,c") == ["a", "b", "c"]
    assert parse_models("only-one") == ["only-one"]
    assert parse_models("") == DEFAULT_MODELS
    assert parse_models(None) == DEFAULT_MODELS
