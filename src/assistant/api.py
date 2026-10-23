"""FastAPI app: the door through which users talk to our AI chatbot."""

import logging
from functools import lru_cache

import groq
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from assistant.agent import AgentError, ChatAgent, GroqAgent
from assistant.auth import AuthError, SupabaseAuth
from assistant.cache import SimpleCache
from assistant.guardrails import GuardrailError, check_text
from assistant.models import ChatRequest, ChatResponse
from assistant.tools import PERSONAL_TOOLS

# Show our INFO log lines (like the token usage) in the uvicorn terminal.
# Without this, Python only prints warnings and errors.
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")
logger = logging.getLogger("assistant")

app = FastAPI(title="Customer Support Assistant")

# Every endpoint lives under /api/v1. If we ever change the API in a way that
# breaks old apps, we add /api/v2 and v1 keeps working.
router = APIRouter(prefix="/api/v1")


@lru_cache
def get_agent() -> ChatAgent:
    """Give FastAPI the chat agent. Tests replace this with MockAgent.

    lru_cache builds it on the first request and reuses it after that, so the
    doc index and the embedding model are not loaded again for every message.
    """
    return GroqAgent.from_env()


# One cache for the whole server, shared by every user. We only save answers
# that used no personal tools (see PERSONAL_TOOLS), so nobody ever gets another
# customer's account or order details.
answer_cache = SimpleCache()


def get_cache() -> SimpleCache:
    """Give FastAPI the answer cache. Tests replace this with a fresh one."""
    return answer_cache


@lru_cache
def get_auth() -> SupabaseAuth:
    """Give FastAPI the login checker. Built once, so Supabase's public keys
    are downloaded only once. Tests replace this with a fake."""
    return SupabaseAuth.from_env()


# Reads the "Authorization: Bearer <token>" header. auto_error=False: if the
# header is missing we send our own 401 below instead of FastAPI's 403.
bearer = HTTPBearer(auto_error=False)


def get_current_customer(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    auth: SupabaseAuth = Depends(get_auth),
) -> str:
    """Who is asking? Returns their customer ID (e.g. "U001") or stops with 401.

    Like an Express auth middleware that sets req.user.
    """
    if credentials is None:
        raise HTTPException(status_code=401, detail="Please log in first.")
    try:
        return auth.customer_id(credentials.credentials)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from None


@router.get("/health")
def health():
    """Simple check: is the server running?"""
    return {"status": "ok"}


@router.post("/ai/chat")
def chat(
    request: ChatRequest,
    customer_id: str = Depends(get_current_customer),  # 401 if not logged in
    agent: ChatAgent = Depends(get_agent),
    cache: SimpleCache = Depends(get_cache),
) -> ChatResponse:
    """Answer a customer: orders and accounts (tools) and store policies (docs)."""
    try:
        safe_message = check_text(request.message)  # first the guard

        # Asked before? Return the saved answer and skip the AI.
        saved = cache.get(safe_message)
        if saved is not None:
            logger.info("Tokens: 0 (answered from cache) | cost: $0")
            return ChatResponse(**saved, cached=True)

        # Then the agent (Groq + tools), only allowed to see THIS customer's data.
        response = agent.chat(safe_message, customer_id)

        # Only answers without anyone's own data are safe to share. Errors jump
        # to "except" below, so they are never saved either.
        if not PERSONAL_TOOLS & set(response.tools_used):
            cache.set(safe_message, response.model_dump(exclude={"cached"}))
        return response

    except GuardrailError as exc:
        # The guard blocked the message. The AI was never called.
        raise HTTPException(status_code=400, detail=str(exc)) from None

    except AgentError:
        # The AI looped on tools too long, or gave an empty answer.
        raise HTTPException(
            status_code=502, detail="AI could not finish the answer"
        ) from None

    except groq.APIError:
        # Groq is busy or down (even after 3 tries), or the key is wrong.
        raise HTTPException(
            status_code=503, detail="AI service is unavailable"
        ) from None


# Must come after all the @router endpoints above, so they all get added.
app.include_router(router)
