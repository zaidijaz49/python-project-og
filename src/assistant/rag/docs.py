"""The help docs as a tool the chat agent can call: search_help_docs(question)."""

from assistant.rag.embeddings import Embedder, LocalEmbedder
from assistant.rag.store import load_index, search

# Picked with scripts/check_scores.py. Below this, no chunk is close enough, so
# the agent gets no doc sections at all. Re-check it if the embedding model or
# docs change.
MIN_SCORE = 0.62


class HelpDocs:
    """Keeps the index and the embedding model in memory, so they load only once."""

    def __init__(self, index: dict, embedder: Embedder, k: int = 3) -> None:
        self.index = index
        self.embedder = embedder
        self.k = k

    @classmethod
    def from_env(cls) -> "HelpDocs":
        """Load data/index.json and the local embedding model."""
        return cls(load_index(), LocalEmbedder())

    def search_help_docs(self, question: str) -> dict:
        """The agent's tool: the best doc sections for the question, or a note
        saying nothing relevant was found."""
        results = search(question, self.index, self.embedder, self.k)

        # Nothing close enough: tell the AI plainly, so it doesn't guess.
        if not results or results[0]["score"] < MIN_SCORE:
            return {"sections": [], "note": "No relevant help docs found."}

        return {
            "sections": [
                {"id": result["id"], "text": result["text"]} for result in results
            ]
        }
