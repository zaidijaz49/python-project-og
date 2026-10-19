"""Tests for the Pydantic schemas in models.py."""

import pytest
from pydantic import ValidationError

from assistant.models import Analysis, AnalyzeRequest


def test_valid_analysis_passes():
    result = Analysis(
        summary="Pakistan won the match.",
        sentiment="positive",
        tags=["sports", "cricket"],
    )

    assert result.sentiment == "positive"
    assert result.tags == ["sports", "cricket"]


def test_bad_sentiment_fails():
    with pytest.raises(ValidationError):
        Analysis(summary="Some text.", sentiment="happy", tags=["life"])


def test_missing_summary_fails():
    with pytest.raises(ValidationError):
        Analysis(sentiment="neutral", tags=["life"])


@pytest.mark.parametrize("tags", [[], ["a", "b", "c", "d", "e", "f"]])
def test_wrong_number_of_tags_fails(tags):
    with pytest.raises(ValidationError):
        Analysis(summary="Some text.", sentiment="neutral", tags=tags)


def test_analysis_from_json_string():
    raw = '{"summary": "Ok.", "sentiment": "negative", "tags": ["work"]}'

    result = Analysis.model_validate_json(raw)

    assert result.sentiment == "negative"


def test_empty_request_text_fails():
    with pytest.raises(ValidationError):
        AnalyzeRequest(text="")
