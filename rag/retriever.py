from dataclasses import dataclass

from weaviate.classes.query import MetadataQuery
from weaviate.collections import Collection


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    source_name: str
    text: str
    distance: float


def search(collection: Collection, vector: list[float], k: int = 3) -> list[Hit]:
    result = collection.query.near_vector(
        near_vector=vector, limit=k, return_metadata=MetadataQuery(distance=True)
    )
    return [
        Hit(
            chunk_id=obj.properties["chunk_id"],
            source_name=obj.properties["source_name"],
            text=obj.properties["text"],
            distance=obj.metadata.distance,
        )
        for obj in result.objects
    ]
