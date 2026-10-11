import argparse
import logging
from pathlib import Path

from rag.loader import load_documents

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG day 1: load and clean knowledge base")
    parser.add_argument("--path", type=Path, default=Path("knowledge_base"))
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
    logger.info("ready for chunking")


if __name__ == "__main__":
    main()
