"""Tests for the search_help_docs tool in rag/docs.py. No real model, no Groq."""

from assistant.rag.docs import MIN_SCORE, HelpDocs


class FixedEmbedder:
    """Fake embedder: every question gets the same vector."""

    def __init__(self, vector):
        self.vector = vector

    def embed_query(self, text):
        return self.vector

    def embed_passages(self, texts):
        return [self.vector for _ in texts]


INDEX = {
    "model": "fake",
    "chunks": [
        {
            "id": "refunds.md#3",
            "source": "refunds.md",
            "heading": "Refund Time",
            "text": "Refunds take 7 working days.",
            "vector": [1.0, 0.0],
        },
        {
            "id": "shipping.md#2",
            "source": "shipping.md",
            "heading": "Charges",
            "text": "Shipping is Rs. 200.",
            "vector": [0.0, 1.0],
        },
    ],
}


def test_close_question_returns_sections_best_first():
    docs = HelpDocs(INDEX, FixedEmbedder([1.0, 0.0]), k=2)

    result = docs.search_help_docs("Refund time?")

    assert [section["id"] for section in result["sections"]] == [
        "refunds.md#3",
        "shipping.md#2",
    ]
    assert result["sections"][0]["text"] == "Refunds take 7 working days."


def test_far_question_returns_no_sections():
    # The best score is below MIN_SCORE, so the AI gets nothing to guess from.
    weak = MIN_SCORE - 0.1
    docs = HelpDocs(INDEX, FixedEmbedder([weak, 0.0]))

    result = docs.search_help_docs("Who is the CEO?")

    assert result == {"sections": [], "note": "No relevant help docs found."}


def test_vectors_are_not_sent_to_the_ai():
    docs = HelpDocs(INDEX, FixedEmbedder([1.0, 0.0]))

    section = docs.search_help_docs("Refund time?")["sections"][0]

    assert set(section) == {"id", "text"}
