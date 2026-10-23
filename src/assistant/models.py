"""Pydantic schemas: the exact shape of data going in and coming out."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """What the user sends to POST /api/v1/ai/chat."""

    message: str = Field(min_length=1, max_length=2_000)


class ChatResponse(BaseModel):
    """What POST /api/v1/ai/chat returns."""

    answer: str
    tools_used: list[str]
    sources: list[str] = []  # help doc sections used, e.g. "refunds.md#3"
    model_used: str | None = None
    cached: bool = False  # True = came from the cache, the AI was not called
