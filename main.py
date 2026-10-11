import argparse
import asyncio
import logging
from collections import Counter
from pathlib import Path

from rag.chunker import chunk_documents
from rag.embedder import embed_texts
from rag.loader import load_documents
from rag.retriever import search
from rag.store import connect, count, ensure_collection, upsert_chunks

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local RAG over knowledge_base/")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="load, chunk, embed, store")
    index.add_argument("--path", type=Path, default=Path("knowledge_base"))
    index.add_argument("--chunk-size", type=int, default=80, help="words per chunk")
    index.add_argument("--overlap", type=int, default=20, help="words shared by neighbours")
    index.add_argument("--examples", type=int, default=3, help="chunks to print")

    query = sub.add_parser("query", help="semantic search")
    query.add_argument("question")
    query.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def run_index(args: argparse.Namespace) -> None:
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

    vectors = asyncio.run(embed_texts([c.text for c in chunks]))
    logger.info("embedded %d chunks", len(vectors))
    client = connect()
    try:
        collection = ensure_collection(client)
        upsert_chunks(collection, chunks, vectors)
        print(f"{'in weaviate':<24} {count(collection):>6} objects")
    finally:
        client.close()
    logger.info("ready for retrieval")


def run_query(args: argparse.Namespace) -> None:
    [vector] = asyncio.run(embed_texts([args.question]))
    client = connect()
    try:
        hits = search(ensure_collection(client), vector, args.top_k)
    finally:
        client.close()
    print(f"Q: {args.question}")
    for hit in hits:
        print(f"{hit.distance:.3f}  {hit.chunk_id:<6} {hit.source_name}")
        print(f"       {hit.text[:160]}...")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    {"index": run_index, "query": run_query}[args.command](args)


if __name__ == "__main__":
    main()
