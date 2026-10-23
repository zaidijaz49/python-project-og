"""FastAPI app: the door through which users talk to our AI client."""

from functools import lru_cache

import groq
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from google.genai import errors

from assistant.agent import AgentError, ChatAgent, GeminiAgent
from assistant.cache import SimpleCache
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

# Every endpoint lives under /api/v1. If we ever change the API in a way that
# breaks old apps, we add /api/v2 and v1 keeps working.
router = APIRouter(prefix="/api/v1")


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


# One cache for the whole server, shared by every user. That's safe here because
# /ai/query answers come only from the help docs, which are the same for everyone.
answer_cache = SimpleCache()


def get_cache() -> SimpleCache:
    """Give FastAPI the answer cache. Tests replace this with a fresh one."""
    return answer_cache


@router.get("/health")
def health():
    """Simple check: is the server running?"""
    return {"status": "ok"}


@router.post("/ai/analyze")
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


@router.post("/ai/chat")
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


@router.post("/ai/query")
def query(
    request: AskRequest,
    rag: RagAssistant = Depends(get_rag),
    cache: SimpleCache = Depends(get_cache),
) -> AskResponse:
    """Answer a question using only the help docs in knowledge/."""
    try:
        safe_question = check_text(request.question)  # first the guard

        # Asked before? Return the saved answer and skip search + AI.
        saved = cache.get(safe_question)
        if saved is not None:
            return AskResponse(**saved, cached=True)

        result = rag.ask(safe_question)  # then search + AI
        # Only reached when the AI answered. Errors jump to "except" below,
        # so they are never saved in the cache.
        cache.set(safe_question, result)
        return AskResponse(**result, cached=False)

    except GuardrailError as exc:
        # The guard blocked the question. Search and AI were never called.
        raise HTTPException(status_code=400, detail=str(exc)) from None

    except groq.APIError:
        # Groq is busy, down, or the key is wrong.
        raise HTTPException(
            status_code=503, detail="AI service is unavailable"
        ) from None


# Must come after all the @router endpoints above, so they all get added.
app.include_router(router)
