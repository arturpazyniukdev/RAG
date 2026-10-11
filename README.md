# Local RAG over a text knowledge base

Question → embed → nearest chunks from Weaviate → Claude answers only from those chunks, citing them.
Off-topic question → honest "I don't know", no hallucination.

Stack: Python 3.12, aiohttp, Ollama `nomic-embed-text` (embeddings), Weaviate 1.30 in Docker (vector DB),
Anthropic API `claude-haiku-5-5` (generation).

## Layout

```
knowledge_base/   source .txt documents
rag/
  loader.py       read + clean .txt files
  chunker.py      word-window chunks (80 words, 20 overlap)
  embedder.py     Ollama embeddings (batched, async)
  store.py        Weaviate collection, idempotent upsert
  retriever.py    near_vector search with distances
  generator.py    prompt + Claude call
main.py           CLI: index / query / ask / eval
eval/questions.txt  answer/refuse test set
NOTES.md          threshold and chunking findings
docker-compose.yml
```

## Setup

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

ollama pull nomic-embed-text           # Ollama must be running on :11434
docker compose up -d                   # Weaviate on :8080 / :50051, data in output/weaviate-docker
export ANTHROPIC_API_KEY=sk-ant-...    # key scoped to a workspace
```

Check Weaviate is up: `curl -s localhost:8080/v1/.well-known/ready` returns HTTP 200.

## Prepare the knowledge base

Drop plain `.txt` files into `knowledge_base/`. Anything goes; the loader collapses whitespace,
drops empty lines and skips files that are empty after cleaning. Keep each file on one topic:
retrieval returns chunks, and a chunk never spans two files.

## Load chunks into Weaviate

```sh
.venv/bin/python -W ignore main.py index
```

Prints per-document sizes, chunk count, sample chunks, then `in weaviate N objects`.
Re-running is safe: chunk ids map to deterministic UUIDs, so existing chunks are overwritten.
If documents shrank or chunk settings changed, use `--reset` to drop the collection first:

```sh
.venv/bin/python -W ignore main.py index --reset --chunk-size 80 --overlap 20
```

`-W ignore` only hides a deprecation warning from a weaviate-client dependency.

## Ask questions

```sh
.venv/bin/python -W ignore main.py ask "when should I create aiohttp ClientSession?"
.venv/bin/python -W ignore main.py ask "..." --top-k 5 --max-distance 0.40
.venv/bin/python -W ignore main.py ask "..." --no-rag      # plain LLM, for comparison
.venv/bin/python -W ignore main.py query "..."             # retrieval only, no LLM
```

Flags: `--top-k` chunks retrieved (default 3), `--max-distance` cutoff (default 0.40),
`--verbose` for debug logs including token usage.

## Reading the output

```
Q: how does asyncio.wait_for handle a timeout?
A: `asyncio.wait_for(aw, timeout)` cancels the awaitable if it runs longer than the
   timeout, and then raises `TimeoutError` [3-2].
Sources:
  0.235  3-2    asyncio_deep_dive.txt
  0.268  8-0    python_async.txt
  0.276  3-3    asyncio_deep_dive.txt
```

- `[3-2]` in the answer is a citation: chunk 2 of document 3.
- `Sources` lists every chunk the model saw, in rank order: cosine distance, chunk id, file.
- Distance scale for `nomic-embed-text`: below 0.30 strong match, 0.30–0.40 related,
  above 0.45 noise. Chunks above `--max-distance` are dropped before the model sees them.
- `Sources: none above threshold` means nothing relevant was found and the model was not called.
  The answer is always `I don't know based on the provided context.`
- The model can still say "I don't know" with sources present: the topic matched but the
  specific fact is not in the base. That is intended.

## Demo

Five answerable questions and one that must be refused. Full output: run `ask` for each.

```
$ python -W ignore main.py ask "when should I create aiohttp ClientSession?"
A: Create one `ClientSession` per application and reuse it. Creating a session for each
   request is slow [0-0].
Sources:
  0.240  0-0    aiohttp_client.txt
  0.360  1-0    aiohttp_server.txt
  0.365  1-2    aiohttp_server.txt

$ python -W ignore main.py ask "what does frozen=True do in a dataclass?"
A: According to the context, `frozen=True` makes dataclass instances immutable and hashable [4-0].
Sources:
  0.296  4-0    dataclasses.txt

$ python -W ignore main.py ask "why use logger.info with %s instead of f-strings?"
A: Because formatting is lazy: the context says to prefer `logger.info("x=%s", x)` over
   f-strings "so formatting is lazy" [7-0].
Sources:
  0.299  7-0    logging_guide.txt

$ python -W ignore main.py ask "how does asyncio.wait_for handle a timeout?"
A: `asyncio.wait_for(aw, timeout)` cancels the awaitable if it runs longer than the timeout,
   and then raises `TimeoutError` [3-2].
Sources:
  0.235  3-2    asyncio_deep_dive.txt
  0.268  8-0    python_async.txt
  0.276  3-3    asyncio_deep_dive.txt

$ python -W ignore main.py ask "how big are embedding vectors usually?"
A: Embedding vectors typically have 384, 768, or 1536 dimensions [5-0].
Sources:
  0.200  5-0    embeddings.txt

$ python -W ignore main.py ask "how do I configure nginx as a reverse proxy?"
A: I don't know based on the provided context.
Sources: none above threshold
```

With `--no-rag` the same questions get longer, generic answers with no sources and nothing
to check them against.

## Evaluate retrieval

```sh
.venv/bin/python -W ignore main.py eval
```

Runs every line of `eval/questions.txt` (`answer|...` or `refuse|...`) through retrieval only,
no LLM, and reports whether the threshold decision matches the label. Current: 12/13.
Findings and the chunk-size sweep are in `NOTES.md`.

## Stop

```sh
docker compose down          # data stays in output/weaviate-docker
```
