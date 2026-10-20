"""AI clients: one real (Gemini) and one fake (Mock), both with the same interface."""

import os
from typing import Protocol

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import ValidationError
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from tenacity.wait import wait_base

from assistant.models import Analysis

DEFAULT_MODEL = "gemini-3.5-flash"
REQUEST_TIMEOUT_MS = 30_000
MAX_ATTEMPTS = 3

# Status codes that usually fix themselves if we wait and try again.
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}

SYSTEM_PROMPT = (
    "You analyze user text. Return a short summary, the overall sentiment "
    "(positive, negative or neutral) and 1 to 5 short category tags."
)


class AIResponseError(Exception):
    """The AI answered, but the answer was not valid Analysis data."""


class AIClient(Protocol):
    """Rule book: every AI client must have this method."""

    def analyze(self, text: str) -> Analysis: ...


def is_transient(exc: BaseException) -> bool:
    """True if the error is temporary and worth retrying."""
    if isinstance(exc, errors.APIError):
        return exc.code in TRANSIENT_STATUS_CODES
    return isinstance(
        exc,
        (httpx.TimeoutException, httpx.NetworkError, TimeoutError, ConnectionError),
    )


class GeminiClient:
    """Calls the real Gemini API, with retries on temporary errors."""

    def __init__(
        self,
        sdk_client: genai.Client,
        model: str = DEFAULT_MODEL,
        wait: wait_base | None = None,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        self._sdk = sdk_client
        self._model = model
        # Waits 1s, 2s, 4s... between attempts. Tests pass wait_none() instead.
        self._wait = wait or wait_exponential(multiplier=1, min=1, max=8)
        self._max_attempts = max_attempts

    @classmethod
    def from_env(cls) -> "GeminiClient":
        """Build a client using GEMINI_API_KEY and GEMINI_MODEL from .env."""
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
        model = os.getenv("GEMINI_MODEL") or DEFAULT_MODEL
        sdk = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        )
        return cls(sdk, model)

    def analyze(self, text: str) -> Analysis:
        retryer = Retrying(
            retry=retry_if_exception(is_transient),
            stop=stop_after_attempt(self._max_attempts),
            wait=self._wait,
            reraise=True,  # after the last attempt, raise the real error
        )
        raw = retryer(self._generate, text)
        return self._parse(raw)

    def _generate(self, text: str) -> str | None:
        """One single call to Gemini. Returns the raw JSON text."""
        response = self._sdk.models.generate_content(
            model=self._model,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=Analysis,
            ),
        )
        return response.text

    @staticmethod
    def _parse(raw: str | None) -> Analysis:
        """Check Gemini's JSON ourselves, even though we asked for a schema."""
        if not raw:
            raise AIResponseError("Gemini returned an empty response.")
        try:
            return Analysis.model_validate_json(raw)
        except ValidationError as exc:
            raise AIResponseError(f"Gemini returned invalid data: {exc}") from exc


class MockClient:
    """Fake client for tests: no network, always the same answer."""

    def __init__(self, result: Analysis | None = None) -> None:
        self.result = result or Analysis(
            summary="This is a mock summary.",
            sentiment="neutral",
            tags=["mock"],
        )
        self.calls: list[str] = []

    def analyze(self, text: str) -> Analysis:
        self.calls.append(text)
        return self.result
