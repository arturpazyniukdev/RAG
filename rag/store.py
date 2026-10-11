import dataclasses
import logging

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.collections import Collection
from weaviate.util import generate_uuid5

from rag.chunker import Chunk

logger = logging.getLogger(__name__)

COLLECTION = "KnowledgeChunk"


def connect() -> weaviate.WeaviateClient:
    return weaviate.connect_to_local()


def ensure_collection(client: weaviate.WeaviateClient) -> Collection:
    if not client.collections.exists(COLLECTION):
        client.collections.create(
            COLLECTION,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="document_id", data_type=DataType.INT),
                Property(name="source_name", data_type=DataType.TEXT),
                Property(name="text", data_type=DataType.TEXT),
            ],
        )
        logger.info("created collection %s", COLLECTION)
    return client.collections.get(COLLECTION)


def upsert_chunks(collection: Collection, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    with collection.batch.fixed_size(batch_size=50) as batch:
        for chunk, vector in zip(chunks, vectors, strict=True):
            batch.add_object(
                properties=dataclasses.asdict(chunk),
                vector=vector,
                uuid=generate_uuid5(chunk.chunk_id),
            )
    failed = collection.batch.failed_objects
    if failed:
        raise RuntimeError(f"{len(failed)} objects failed: {failed[0].message}")


def count(collection: Collection) -> int:
    return collection.aggregate.over_all(total_count=True).total_count
