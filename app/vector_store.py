from dataclasses import dataclass
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.embeddings import cosine_similarity


@dataclass(slots=True)
class VectorRecord:
    id: str
    vector: list[float]
    text: str
    source: str
    metadata: dict[str, str]


@dataclass(slots=True)
class ScoredVectorRecord:
    record: VectorRecord
    score: float


class VectorStore(Protocol):
    backend_name: str

    def upsert(self, records: list[VectorRecord]) -> None: ...
    def search(self, vector: list[float], limit: int) -> list[ScoredVectorRecord]: ...
    def health(self) -> bool: ...


class InMemoryVectorStore:
    backend_name = "memory"

    def __init__(self) -> None:
        self.records: dict[str, VectorRecord] = {}

    def upsert(self, records: list[VectorRecord]) -> None:
        self.records.update({record.id: record for record in records})

    def search(self, vector: list[float], limit: int) -> list[ScoredVectorRecord]:
        scored = [
            ScoredVectorRecord(record=r, score=cosine_similarity(vector, r.vector))
            for r in self.records.values()
        ]
        return sorted(scored, key=lambda x: x.score, reverse=True)[:limit]

    def health(self) -> bool:
        return True


class QdrantVectorStore:
    backend_name = "qdrant"

    def __init__(self, url: str, collection: str, dimension: int, api_key: str | None = None):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection = collection
        self.dimension = dimension
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
            )

    def upsert(self, records: list[VectorRecord]) -> None:
        points = [
            PointStruct(
                id=record.id,
                vector=record.vector,
                payload={
                    "text": record.text,
                    "source": record.source,
                    "metadata": record.metadata,
                },
            )
            for record in records
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(self, vector: list[float], limit: int) -> list[ScoredVectorRecord]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        results: list[ScoredVectorRecord] = []
        for point in response.points:
            payload = point.payload or {}
            record = VectorRecord(
                id=str(point.id),
                vector=[],
                text=str(payload.get("text", "")),
                source=str(payload.get("source", "unknown")),
                metadata=dict(payload.get("metadata", {})),
            )
            results.append(ScoredVectorRecord(record=record, score=float(point.score)))
        return results

    def health(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
