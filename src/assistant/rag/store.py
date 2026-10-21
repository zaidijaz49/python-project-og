"""Save and load the index: every chunk together with its vector, in one JSON file."""

import json
from pathlib import Path

from assistant.rag.embeddings import Embedder

# src/assistant/rag/store.py -> go up 3 folders to reach the project root.
INDEX_PATH = Path(__file__).parents[3] / "data" / "index.json"


def save_index(
    chunks: list[dict], vectors: list[list[float]], model: str, path: Path = INDEX_PATH
) -> None:
    """Write chunks + vectors to a JSON file. Each chunk gets its own vector."""
    items = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        items.append({**chunk, "vector": vector})

    data = {"model": model, "chunks": items}

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def load_index(path: Path = INDEX_PATH) -> dict:
    """Read the JSON file back. Run scripts/build_index.py first if it is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"No index at {path}. Run: python scripts/build_index.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def dot_product(a: list[float], b: list[float]) -> float:
    """Multiply the numbers at the same position, then add them all up."""
    total = 0.0
    for x, y in zip(a, b, strict=True):
        total += x * y
    return total


def search(question: str, index: dict, embedder: Embedder, k: int = 3) -> list[dict]:
    """Return the k chunks whose meaning is closest to the question, best first."""
    question_vector = embedder.embed_query(question)

    results = []
    for chunk in index["chunks"]:
        score = dot_product(question_vector, chunk["vector"])
        results.append(
            {
                "id": chunk["id"],
                "source": chunk["source"],
                "heading": chunk["heading"],
                "text": chunk["text"],
                "score": score,
            }
        )

    # Highest score first, then keep only the top k.
    results.sort(key=lambda result: result["score"], reverse=True)
    return results[:k]


if __name__ == "__main__":
    from assistant.rag.embeddings import LocalEmbedder

    index = load_index()
    embedder = LocalEmbedder()

    for question in ["How can I get my money back?", "Do you ship to Dubai?"]:
        print("=" * 60)
        print(f"Question: {question}")
        for result in search(question, index, embedder):
            print(f"  {result['score']:.3f}  {result['id']:<22} {result['heading']}")
