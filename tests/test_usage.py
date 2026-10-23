"""Tests for token counting and cost estimates in usage.py. No Groq involved."""

from types import SimpleNamespace

import pytest

from assistant.usage import TokenUsage, estimate_cost


def groq_usage(prompt_tokens, completion_tokens):
    """The `usage` part of a Groq response."""
    return SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )


def test_cost_of_one_million_tokens_is_the_listed_price():
    assert estimate_cost("openai/gpt-oss-120b", 1_000_000, 0) == pytest.approx(0.15)
    assert estimate_cost("openai/gpt-oss-120b", 0, 1_000_000) == pytest.approx(0.60)
    assert estimate_cost("openai/gpt-oss-20b", 1_000_000, 0) == pytest.approx(0.075)


def test_cost_adds_input_and_output():
    # 1900 x $0.15/M + 60 x $0.60/M
    assert estimate_cost("openai/gpt-oss-120b", 1900, 60) == pytest.approx(0.000321)


def test_unknown_model_has_no_cost():
    assert estimate_cost("some-new-model", 1000, 100) is None


def test_usage_adds_up_every_call():
    usage = TokenUsage()
    usage.add(groq_usage(900, 20))
    usage.add(groq_usage(1000, 40))

    assert usage.calls == 2
    assert usage.input_tokens == 1900
    assert usage.output_tokens == 60


def test_missing_usage_still_counts_the_call():
    usage = TokenUsage()
    usage.add(None)

    assert usage.calls == 1
    assert usage.input_tokens == 0


def test_log_line(caplog):
    caplog.set_level("INFO", logger="assistant")
    usage = TokenUsage()
    usage.add(groq_usage(1900, 60))

    usage.log("openai/gpt-oss-120b")

    assert caplog.messages == [
        "Tokens: 1900 in, 60 out (1 calls) | model: openai/gpt-oss-120b"
        " | cost: ~$0.000321"
    ]


def test_log_line_for_unknown_model(caplog):
    caplog.set_level("INFO", logger="assistant")
    usage = TokenUsage()
    usage.add(groq_usage(10, 5))

    usage.log("some-new-model")

    assert "cost: unknown" in caplog.text
