from typing import Literal

from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    source: str
    text: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)
    trust: Literal["trusted", "unverified"] = "unverified"


class IngestRequest(BaseModel):
    documents: list[DocumentIn]


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    entities: int
    relations: int
    backends: dict[str, str]
    quarantined: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)


class SourceResult(BaseModel):
    source: str
    text: str
    score: float
    retrieval: Literal["vector", "graph", "hybrid"]
    metadata: dict[str, str] = Field(default_factory=dict)
    trust: Literal["trusted", "unverified", "quarantined"] = "unverified"
    graph_hop_distance: int | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResult]
    entities: list[str]
    latency_ms: float
    provider: str
    output_flagged: bool = False


class EvalCase(BaseModel):
    question: str
    expected_sources: list[str] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)
    requires_hops: int = 0
    """Minimum graph hop distance a correct retrieval must traverse to
    answer this question. 0 means the question doesn't specifically test
    multi-hop graph reasoning (single-hop / vector retrieval is enough)."""


class EvaluateRequest(BaseModel):
    cases: list[EvalCase] | None = None


class EvalSummary(BaseModel):
    cases: int
    retrieval_hit_rate: float
    answer_keyword_coverage: float
    groundedness_proxy: float
    multi_hop_cases: int
    multi_hop_hit_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
