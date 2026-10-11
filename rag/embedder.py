import logging

import aiohttp

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"


async def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    vectors: list[list[float]] = []
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = {"model": MODEL, "input": batch}
            async with session.post(OLLAMA_URL, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
            vectors.extend(data["embeddings"])
            logger.debug("embedded %d/%d", len(vectors), len(texts))
    return vectors
