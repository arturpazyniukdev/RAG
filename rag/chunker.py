from dataclasses import dataclass

from rag.loader import Document


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: int
    source_name: str
    text: str


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


def chunk_documents(docs: list[Document], size: int = 80, overlap: int = 20) -> list[Chunk]:
    chunks = []
    for doc_id, doc in enumerate(docs):
        for i, text in enumerate(chunk_text(doc.text, size, overlap)):
            chunks.append(Chunk(f"{doc_id}-{i}", doc_id, doc.name, text))
    return chunks
