"""FastAPI app: the door through which users talk to our AI client."""

from fastapi import Depends, FastAPI, HTTPException
from google.genai import errors

from assistant.client import AIClient, AIResponseError, GeminiClient
from assistant.models import Analysis, AnalyzeRequest

app = FastAPI(title="Text Analyzer")


def get_client() -> AIClient:
    """Give FastAPI the AI client. Tests replace this with MockClient."""
    return GeminiClient.from_env()


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
        return client.analyze(request.text)

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
