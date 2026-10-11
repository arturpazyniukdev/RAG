import logging

import anthropic

from rag.retriever import Hit

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-5-5"
NO_CONTEXT = "I don't know based on the provided context."
SYSTEM_RAG = (
    "Answer the question using ONLY the context blocks below. "
    "Cite chunk ids in square brackets, e.g. [1-0], after each fact. "
    f"If the context does not contain the answer, reply exactly: {NO_CONTEXT}"
)
SYSTEM_PLAIN = "Answer the question briefly from your own knowledge."


def build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = [f"[{h.chunk_id}] ({h.source_name})\n{h.text}" for h in hits]
    return "\n\n".join([*blocks, f"Question: {question}"])


def generate(question: str, hits: list[Hit]) -> str:
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=SYSTEM_RAG if hits else SYSTEM_PLAIN,
        messages=[{"role": "user", "content": build_prompt(question, hits)}],
    )
    logger.debug("tokens in=%d out=%d", resp.usage.input_tokens, resp.usage.output_tokens)
    return next(block.text for block in resp.content if block.type == "text")
