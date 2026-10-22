"""FastAPI app: the door through which users talk to our AI client."""

from functools import lru_cache

import groq
from fastapi import Depends, FastAPI, HTTPException
from google.genai import errors

from assistant.agent import AgentError, ChatAgent, GeminiAgent
from assistant.client import AIClient, AIResponseError, GeminiClient
from assistant.guardrails import GuardrailError, check_text
from assistant.models import (
    Analysis,
    AnalyzeRequest,
    AskRequest,
    AskResponse,
    ChatRequest,
    ChatResponse,
)
from assistant.rag.answer import RagAssistant

app = FastAPI(title="Customer Support Assistant")


def get_client() -> AIClient:
    """Give FastAPI the AI client. Tests replace this with MockClient."""
    return GeminiClient.from_env()


def get_agent() -> ChatAgent:
    """Give FastAPI the chat agent. Tests replace this with MockAgent."""
    return GeminiAgent.from_env()


@lru_cache
def get_rag() -> RagAssistant:
    """Give FastAPI the RAG assistant. Tests replace this with a mock version.

    lru_cache builds it on the first request and reuses it after that, so the
    index and the embedding model are not loaded again for every question.
    """
    return RagAssistant.from_env()


@app.get("/health")
def health():
    """Simple check: is the server running?"""
    return {"status": "ok"}


@app.post("/analyze")
def analyze(
    request: AnalyzeRequest, client: AIClient = Depends(get_client)
) -> Analysis:
    """Take user text, send it to the AI, return summary + sentiment + tags."""
    try:
        safe_text = check_text(request.text)  # first the guard
        return client.analyze(safe_text)  # then the AI

    except GuardrailError as exc:
        # The guard blocked the text. The AI was never called.
        raise HTTPException(status_code=400, detail=str(exc)) from None

    except AIResponseError:
        # Gemini answered, but the answer was broken.
        raise HTTPException(
            status_code=502, detail="AI returned invalid data"
        ) from None

    except errors.APIError:
        # Gemini is busy or down (even after 3 tries).
        raise HTTPException(
            status_code=503, detail="AI service is unavailable"
        ) from None


@app.post("/chat")
def chat(request: ChatRequest, agent: ChatAgent = Depends(get_agent)) -> ChatResponse:
    """Answer a customer question. The AI may look up accounts and orders."""
    try:
        safe_message = check_text(request.message)  # first the guard
        return agent.chat(safe_message)  # then the agent (Gemini + tools)

    except GuardrailError as exc:
        # The guard blocked the message. The AI was never called.
        raise HTTPException(status_code=400, detail=str(exc)) from None

    except AgentError:
        # Gemini looped on tools too long, or gave an empty answer.
        raise HTTPException(
            status_code=502, detail="AI could not finish the answer"
        ) from None

    except errors.APIError:
        # Gemini is busy or down (even after 3 tries).
        raise HTTPException(
            status_code=503, detail="AI service is unavailable"
        ) from None


@app.post("/ask")
def ask(request: AskRequest, rag: RagAssistant = Depends(get_rag)) -> AskResponse:
    """Answer a question using only the help docs in knowledge/."""
    try:
        safe_question = check_text(request.question)  # first the guard
        return AskResponse(**rag.ask(safe_question))  # then search + AI

    except GuardrailError as exc:
        # The guard blocked the question. Search and AI were never called.
        raise HTTPException(status_code=400, detail=str(exc)) from None

    except groq.APIError:
        # Groq is busy, down, or the key is wrong.
        raise HTTPException(
            status_code=503, detail="AI service is unavailable"
        ) from None
