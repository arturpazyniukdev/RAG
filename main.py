import argparse
import logging
from collections import Counter
from pathlib import Path

from rag.chunker import chunk_documents
from rag.loader import load_documents

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG: load, clean and chunk knowledge base")
    parser.add_argument("--path", type=Path, default=Path("knowledge_base"))
    parser.add_argument("--chunk-size", type=int, default=80, help="words per chunk")
    parser.add_argument("--overlap", type=int, default=20, help="words shared by neighbours")
    parser.add_argument("--examples", type=int, default=3, help="chunks to print")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    docs = load_documents(args.path)
    for doc in docs:
        print(f"{doc.name:<24} {len(doc.text):>6}")
    print(f"{'total':<24} {len(docs):>6} docs")

    chunks = chunk_documents(docs, args.chunk_size, args.overlap)
    print(f"{'chunks':<24} {len(chunks):>6} (size={args.chunk_size}, overlap={args.overlap})")

    if chunks:
        longest = Counter(c.source_name for c in chunks).most_common(1)[0][0]
        examples = [c for c in chunks if c.source_name == longest][: args.examples]
        for chunk in examples:
            print(
                f"\n[{chunk.chunk_id}] doc={chunk.document_id} {chunk.source_name}"
                f" words={len(chunk.text.split())}"
            )
            print(f"  {chunk.text[:120]}...")
            print(f"  ...{chunk.text[-80:]}")
    logger.info("ready for embedding")


if __name__ == "__main__":
    main()
