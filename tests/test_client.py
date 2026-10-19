"""Tests for GeminiClient and MockClient. Gemini is faked: no key, no network."""

from unittest.mock import MagicMock

import httpx
import pytest
from google.genai import errors
from tenacity import wait_none

from assistant.client import AIResponseError, GeminiClient, MockClient
from assistant.models import Analysis

GOOD_JSON = '{"summary": "A good day.", "sentiment": "positive", "tags": ["life"]}'


def fake_response(text):
    """Looks like a Gemini response: it only needs a .text attribute."""
    return MagicMock(text=text)


def make_client(*outcomes):
    """GeminiClient with a fake SDK. Each outcome is one call's result or error."""
    sdk = MagicMock()
    sdk.models.generate_content.side_effect = list(outcomes)
    # wait_none() = no sleeping between retries, so tests stay fast.
    client = GeminiClient(sdk, model="test-model", wait=wait_none())
    return client, sdk.models.generate_content


def test_success_returns_analysis():
    client, generate = make_client(fake_response(GOOD_JSON))

    result = client.analyze("I had a good day")

    assert result == Analysis(
        summary="A good day.", sentiment="positive", tags=["life"]
    )
    assert generate.call_count == 1
    assert generate.call_args.kwargs["config"].response_schema is Analysis


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
        errors.ServerError(503, {}),
        fake_response(GOOD_JSON),
    )

    result = client.analyze("hello")

    assert result.sentiment == "positive"
    assert generate.call_count == 2


def test_retry_on_timeout_then_success():
    client, generate = make_client(
        httpx.ReadTimeout("too slow"),
        fake_response(GOOD_JSON),
    )

    client.analyze("hello")

    assert generate.call_count == 2


def test_retry_gives_up_after_3_attempts():
    client, generate = make_client(*[errors.ServerError(503, {})] * 3)

    with pytest.raises(errors.ServerError):
        client.analyze("hello")

    assert generate.call_count == 3


def test_no_retry_on_401():
    client, generate = make_client(errors.ClientError(401, {}))

    with pytest.raises(errors.ClientError):
        client.analyze("hello")

    assert generate.call_count == 1


def test_mock_client_returns_fixed_answer():
    mock = MockClient()

    result = mock.analyze("anything")

    assert result.sentiment == "neutral"
    assert mock.calls == ["anything"]
