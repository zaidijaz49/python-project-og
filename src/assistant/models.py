"""Pydantic schemas: the exact shape of data going in and coming out."""

from typing import Literal

from pydantic import BaseModel, Field

Sentiment = Literal["positive", "negative", "neutral"]


class AnalyzeRequest(BaseModel):
    """What the user sends to POST /analyze."""

    text: str = Field(min_length=1, max_length=10_000)


class Analysis(BaseModel):
    """What the AI must return. Anything else fails validation."""

    summary: str = Field(min_length=1, description="Short summary of the text")
    sentiment: Sentiment = Field(description="Overall tone of the text")
    tags: list[str] = Field(
        min_length=1,
        max_length=5,
        description="1 to 5 short category tags",
    )


class ChatRequest(BaseModel):
    """What the user sends to POST /chat."""

    message: str = Field(min_length=1, max_length=2_000)


class ChatResponse(BaseModel):
    """What POST /chat returns: the answer and which tools were used."""

    answer: str
    tools_used: list[str]
