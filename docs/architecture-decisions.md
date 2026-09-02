# Architecture decisions

## ADR-001 — Hybrid vector + graph retrieval

**Decision:** Use Qdrant for semantic retrieval and Neo4j for explicit relationships.

**Reason:** Enterprise questions often combine fuzzy intent with exact topology/ownership/dependency relationships. One retrieval method is not optimal for both.

## ADR-002 — Reciprocal Rank Fusion

**Decision:** Use RRF as the first fusion strategy.

**Reason:** Vector and graph systems expose scores on different scales. RRF combines rank positions without fragile score normalization and is easy to explain and benchmark.

## ADR-003 — Deterministic local fallback

**Decision:** Keep an offline hashing embedder and extractive answer provider.

**Reason:** The service should be testable locally and in CI without API keys. Hosted models remain pluggable for higher quality.

## ADR-004 — Explicit LangGraph workflow

**Decision:** Use an explicit retrieval → generation graph rather than an opaque agent executor.

**Reason:** The workflow is easier to test, trace, extend with routing/tool nodes, and reason about during incidents.
