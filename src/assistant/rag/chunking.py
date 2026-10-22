"""Split the help docs in knowledge/ into small chunks, one chunk per ## section."""

from pathlib import Path

# src/assistant/rag/chunking.py -> go up 3 folders to reach the project root.
KNOWLEDGE_DIR = Path(__file__).parents[3] / "knowledge"


def make_chunk(
    source: str, number: int, title: str, heading: str, lines: list[str]
) -> dict:
    """Build one chunk. The title and heading go on top so search knows the topic."""
    body = "\n".join(lines)
    return {
        "id": f"{source}#{number}",
        "source": source,
        "heading": heading,
        "text": f"{title} > {heading}\n{body}",
    }


def chunk_file(path: Path) -> list[dict]:
    """Turn one .md file into a list of chunks."""
    source = path.name
    chunks = []

    # The 3 "boxes" from the dry run. They start empty for every file.
    title = ""
    heading = ""
    lines = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if line == "":
            continue

        if line.startswith("## "):
            # A new section starts, so save the previous one first.
            if lines:
                chunks.append(make_chunk(source, len(chunks), title, heading, lines))
            heading = line[3:].strip()
            lines = []
        elif line.startswith("# "):
            title = line[2:].strip()
        else:
            lines.append(line)

    # The last section has no heading after it, so save it here.
    if lines:
        chunks.append(make_chunk(source, len(chunks), title, heading, lines))

    return chunks


def load_chunks(folder: Path = KNOWLEDGE_DIR) -> list[dict]:
    """Chunk every .md file in the folder and return all chunks in one list."""
    all_chunks = []
    for path in sorted(folder.glob("*.md")):
        all_chunks.extend(chunk_file(path))
    return all_chunks


if __name__ == "__main__":
    chunks = load_chunks()
    for chunk in chunks:
        print("=" * 60)
        print(chunk["id"])
        print(chunk["text"])
    print("=" * 60)
    print(f"Total chunks: {len(chunks)}")
