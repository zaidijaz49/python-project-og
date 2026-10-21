"""Turn text into vectors (lists of numbers) so we can search by meaning."""

from typing import Protocol

MODEL_NAME = "BAAI/bge-small-en-v1.5"


class Embedder(Protocol):
    """Anything that can turn text into vectors: the real model or a fake one."""

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LocalEmbedder:
    """Runs the bge-small model on this computer with fastembed (no API key)."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        # Imported here so tests that use MockEmbedder never load the model.
        from fastembed import TextEmbedding

        # The first run downloads the model (~130 MB), after that it is cached.
        self.model = TextEmbedding(model_name=model_name)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Vectors for our doc chunks."""
        vectors = self.model.passage_embed(texts)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Vector for the user's question. bge adds a small search hint to queries."""
        vectors = self.model.query_embed(text)
        return next(iter(vectors)).tolist()


class MockEmbedder:
    """Fake embedder for tests: counts a few keywords instead of using a model."""

    KEYWORDS = ["refund", "shipping", "order", "payment", "password", "support"]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        text = text.lower()
        return [float(text.count(word)) for word in self.KEYWORDS]


if __name__ == "__main__":
    from assistant.rag.chunking import load_chunks

    chunks = load_chunks()
    embedder = LocalEmbedder()
    vectors = embedder.embed_passages([chunk["text"] for chunk in chunks])

    print(f"Chunks: {len(chunks)}")
    print(f"Vectors: {len(vectors)}")
    print(f"Numbers in one vector: {len(vectors[0])}")
    print(f"First 5 numbers of {chunks[0]['id']}: {vectors[0][:5]}")
