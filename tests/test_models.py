"""Tests for the Pydantic request schemas in models.py."""

import pytest
from pydantic import ValidationError

from assistant.models import ChatRequest


def test_chat_message_is_accepted():
    assert ChatRequest(message="Where is my order?").message == "Where is my order?"


@pytest.mark.parametrize("message", ["", "a" * 2_001])
def test_chat_message_empty_or_too_long_fails(message):
    with pytest.raises(ValidationError):
        ChatRequest(message=message)
