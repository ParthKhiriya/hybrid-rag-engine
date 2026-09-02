import hashlib
import re
from dataclasses import dataclass

from app.embeddings import Embedder
from app.graph_store import GraphStore
from app.ingest import extract_entities
from app.models import SourceResult
from app.security import TRUST_WEIGHT
from app.vector_store import VectorStore


@dataclass(slots=True)
class HybridResult:
    sources: list[SourceResult]
    entities: list[str]


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def stable_key(source: str, text: str) -> str:
    """Process-stable dedup key. Python's built-in hash() is salted per
    interpreter run, so it must not be used for keys that need to be
    consistent across requests or restarts."""
    digest = hashlib.blake2b(text.encode(), digest_size=8).hexdigest()
    return f"{source}:{digest}"


def lexical_bonus(question: str, text: str) -> float:
    q = set(re.findall(r"[a-z0-9_-]+", question.lower()))
    t = set(re.findall(r"[a-z0-9_-]+", text.lower()))
    if not q:
        return 0.0
    return len(q & t) / len(q)


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        graph_store: GraphStore,
        vector_k: int = 8,
        graph_k: int = 8,
        rrf_k: int = 60,
        exclude_quarantined: bool = True,
        graph_max_hops: int = 2,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.vector_k = vector_k
        self.graph_k = graph_k
        self.rrf_k = rrf_k
        self.exclude_quarantined = exclude_quarantined
        self.graph_max_hops = graph_max_hops

    def retrieve(self, question: str, top_k: int = 5) -> HybridResult:
        entities = extract_entities(question)
        vector_hits = self.vector_store.search(self.embedder.embed(question), self.vector_k)
        graph_hits = self.graph_store.search(entities, self.graph_k, self.graph_max_hops)

        by_key: dict[str, SourceResult] = {}
        vector_rank: list[str] = []
        graph_rank: list[str] = []

        for hit in vector_hits:
            key = stable_key(hit.record.source, hit.record.text)
            vector_rank.append(key)
            by_key[key] = SourceResult(
                source=hit.record.source,
                text=hit.record.text,
                score=max(0.0, hit.score),
                retrieval="vector",
                metadata=hit.record.metadata,
                trust=hit.record.metadata.get("trust", "unverified"),
            )

        for hit in graph_hits:
            key = stable_key(hit.source, hit.text)
            graph_rank.append(key)
            relation_text = "\nRelationships: " + "; ".join(hit.relations) if hit.relations else ""
            if key in by_key:
                by_key[key].retrieval = "hybrid"
                by_key[key].text = by_key[key].text + relation_text
                by_key[key].graph_hop_distance = hit.hop_distance
            else:
                # Graph store doesn't carry per-document trust metadata today;
                # default new graph-only hits to "unverified" rather than
                # implicitly trusting them.
                by_key[key] = SourceResult(
                    source=hit.source,
                    text=hit.text + relation_text,
                    score=0.0,
                    retrieval="graph",
                    metadata={},
                    trust="unverified",
                    graph_hop_distance=hit.hop_distance,
                )

        if self.exclude_quarantined:
            by_key = {k: v for k, v in by_key.items() if v.trust != "quarantined"}

        fused = reciprocal_rank_fusion([vector_rank, graph_rank], self.rrf_k)
        for key, result in by_key.items():
            base = fused.get(key, 0.0) + 0.15 * lexical_bonus(question, result.text)
            result.score = base * TRUST_WEIGHT.get(result.trust, 0.6)

        ordered = sorted(by_key.values(), key=lambda x: x.score, reverse=True)[:top_k]
        return HybridResult(sources=ordered, entities=entities)
