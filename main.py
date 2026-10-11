import argparse
import asyncio
import logging
from collections import Counter
from pathlib import Path

from rag.chunker import chunk_documents
from rag.embedder import embed_texts
from rag.generator import NO_CONTEXT, generate
from rag.loader import load_documents
from rag.retriever import Hit, search
from rag.store import connect, count, drop_collection, ensure_collection, upsert_chunks

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
    index.add_argument("--reset", action="store_true", help="drop collection before upload")

    query = sub.add_parser("query", help="semantic search")
    query.add_argument("question")
    query.add_argument("--top-k", type=int, default=3)

    ask = sub.add_parser("ask", help="retrieve context, then answer with LLM")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=3)
    ask.add_argument("--max-distance", type=float, default=0.40, help="drop farther hits")
    ask.add_argument("--no-rag", action="store_true", help="answer without retrieval")

    ev = sub.add_parser("eval", help="check retrieval against a question set")
    ev.add_argument("--file", type=Path, default=Path("eval/questions.txt"))
    ev.add_argument("--top-k", type=int, default=3)
    ev.add_argument("--max-distance", type=float, default=0.40)
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
        if args.reset:
            drop_collection(client)
        collection = ensure_collection(client)
        upsert_chunks(collection, chunks, vectors)
        print(f"{'in weaviate':<24} {count(collection):>6} objects")
    finally:
        client.close()
    logger.info("ready for retrieval")


def retrieve(question: str, k: int) -> list[Hit]:
    [vector] = asyncio.run(embed_texts([question]))
    client = connect()
    try:
        return search(ensure_collection(client), vector, k)
    finally:
        client.close()


def run_query(args: argparse.Namespace) -> None:
    hits = retrieve(args.question, args.top_k)
    print(f"Q: {args.question}")
    for hit in hits:
        print(f"{hit.distance:.3f}  {hit.chunk_id:<6} {hit.source_name}")
        print(f"       {hit.text[:160]}...")


def run_ask(args: argparse.Namespace) -> None:
    print(f"Q: {args.question}")
    if args.no_rag:
        print(f"A: {generate(args.question, [])}")
        return
    hits = [h for h in retrieve(args.question, args.top_k) if h.distance <= args.max_distance]
    if not hits:
        print(f"A: {NO_CONTEXT}\nSources: none above threshold")
        return
    print(f"A: {generate(args.question, hits)}")
    print("Sources:")
    for hit in hits:
        print(f"  {hit.distance:.3f}  {hit.chunk_id:<6} {hit.source_name}")


def run_eval(args: argparse.Namespace) -> None:
    lines = [line for line in args.file.read_text().splitlines() if line.strip()]
    correct = 0
    for line in lines:
        label, question = line.split("|", 1)
        hits = retrieve(question, args.top_k)
        kept = [h for h in hits if h.distance <= args.max_distance]
        ok = bool(kept) == (label == "answer")
        correct += ok
        best = hits[0].distance if hits else float("nan")
        print(
            f"{'OK  ' if ok else 'MISS'} {label:<6} {best:.3f} {len(kept)}/{len(hits)} {question}"
        )
        print("       " + "  ".join(f"{h.chunk_id}:{h.distance:.3f}" for h in hits))
    print(f"{correct}/{len(lines)} as expected")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for name in ("httpx", "httpx2"):
        logging.getLogger(name).setLevel(logging.WARNING)
    {"index": run_index, "query": run_query, "ask": run_ask, "eval": run_eval}[args.command](args)


if __name__ == "__main__":
    main()
