"""Tests for the FastAPI app. The real Gemini client is swapped for fakes."""

from fastapi.testclient import TestClient
from google.genai import errors

from assistant.api import app, get_client
from assistant.client import AIResponseError, MockClient

http = TestClient(app)


class BrokenClient:
    """Fake client whose AI always returns broken data."""

    def analyze(self, text):
        raise AIResponseError("bad data")


class BusyClient:
    """Fake client whose AI is always busy."""

    def analyze(self, text):
        raise errors.ServerError(503, {})


def use_client(fake):
    """Tell FastAPI: use this fake instead of the real GeminiClient."""
    app.dependency_overrides[get_client] = lambda: fake


def teardown_function():
    """Runs after every test: undo the swap so tests don't affect each other."""
    app.dependency_overrides.clear()


def test_health():
    response = http.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_returns_result():
    mock = MockClient()
    use_client(mock)

    response = http.post("/analyze", json={"text": "hello"})

    assert response.status_code == 200
    assert response.json() == {
        "summary": "This is a mock summary.",
        "sentiment": "neutral",
        "tags": ["mock"],
    }
    assert mock.calls == ["hello"]


def test_empty_text_returns_422():
    mock = MockClient()
    use_client(mock)

    response = http.post("/analyze", json={"text": ""})

    assert response.status_code == 422
    assert mock.calls == []  # bad request never reached the AI


def test_missing_text_field_returns_422():
    use_client(MockClient())

    response = http.post("/analyze", json={"message": "hello"})

    assert response.status_code == 422


def test_broken_ai_answer_returns_502():
    use_client(BrokenClient())

    response = http.post("/analyze", json={"text": "hello"})

    assert response.status_code == 502


def test_busy_ai_returns_503():
    use_client(BusyClient())

    response = http.post("/analyze", json={"text": "hello"})

    assert response.status_code == 503
