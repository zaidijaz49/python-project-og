"""Pydantic schemas: the exact shape of data going in and coming out."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """What the user sends to POST /api/v1/ai/chat."""

    message: str = Field(min_length=1, max_length=2_000)


class ChatResponse(BaseModel):
    """What POST /api/v1/ai/chat returns: answer, tools used and the model used."""

    answer: str
    tools_used: list[str]
    model_used: str | None = None


class AskRequest(BaseModel):
    """What the user sends to POST /api/v1/ai/query."""

    question: str = Field(min_length=1, max_length=2_000)


class AskResponse(BaseModel):
    """What POST /api/v1/ai/query returns: the answer and its doc chunks."""

    answer: str
    sources: list[str]
    found: bool
    cached: bool = False  # True = came from the cache, the AI was not called
