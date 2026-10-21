"""Build data/index.json from the docs in knowledge/.

Run this again every time a doc in knowledge/ changes:
    python scripts/build_index.py
"""

from assistant.rag.chunking import load_chunks
from assistant.rag.embeddings import MODEL_NAME, LocalEmbedder
from assistant.rag.store import INDEX_PATH, save_index


def main() -> None:
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from knowledge/")

    embedder = LocalEmbedder()
    vectors = embedder.embed_passages([chunk["text"] for chunk in chunks])
    print(f"Made {len(vectors)} vectors with {MODEL_NAME}")

    save_index(chunks, vectors, MODEL_NAME)
    print(f"Saved index to {INDEX_PATH}")


if __name__ == "__main__":
    main()
