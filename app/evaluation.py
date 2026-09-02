import math
import re
from statistics import median

from app.agent import RAGAgent
from app.models import EvalCase, EvalSummary


def _pct(num: float) -> float:
    return round(100.0 * num, 2)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1)
    return ordered[max(0, idx)]


def groundedness_proxy(answer: str, contexts: list[str]) -> float:
    a = set(re.findall(r"[a-z0-9_-]+", answer.lower()))
    c = set(re.findall(r"[a-z0-9_-]+", " ".join(contexts).lower()))
    if not a:
        return 0.0
    return len(a & c) / len(a)


def evaluate(agent: RAGAgent, cases: list[EvalCase]) -> EvalSummary:
    retrieval_hits = []
    keyword_coverages = []
    grounded = []
    latencies = []
    multi_hop_hits = []

    for case in cases:
        result = agent.invoke(case.question, top_k=5)
        sources = {s.source for s in result.sources}
        expected_sources = set(case.expected_sources)
        retrieval_hits.append(1.0 if not expected_sources or sources & expected_sources else 0.0)

        answer_lower = result.answer.lower()
        if case.expected_keywords:
            hits = sum(1 for kw in case.expected_keywords if kw.lower() in answer_lower)
            keyword_coverages.append(hits / len(case.expected_keywords))
        else:
            keyword_coverages.append(1.0)

        grounded.append(groundedness_proxy(result.answer, [s.text for s in result.sources]))
        latencies.append(result.latency_ms)

        if case.requires_hops > 0:
            # A "hit" means the correct source was surfaced AND it was
            # reached via genuine multi-hop graph traversal (hop_distance
            # >= 1), not just because it also happened to match the vector
            # search. hop_distance == 0 means the entity was a direct seed
            # match, which single-hop retrieval could already do.
            hit = any(
                s.source in expected_sources
                and s.graph_hop_distance is not None
                and s.graph_hop_distance >= 1
                for s in result.sources
            )
            multi_hop_hits.append(1.0 if hit else 0.0)

    n = max(1, len(cases))
    multi_hop_n = max(1, len(multi_hop_hits))
    return EvalSummary(
        cases=len(cases),
        retrieval_hit_rate=_pct(sum(retrieval_hits) / n),
        answer_keyword_coverage=_pct(sum(keyword_coverages) / n),
        groundedness_proxy=_pct(sum(grounded) / n),
        multi_hop_cases=len(multi_hop_hits),
        multi_hop_hit_rate=_pct(sum(multi_hop_hits) / multi_hop_n) if multi_hop_hits else 0.0,
        p50_latency_ms=round(median(latencies), 2) if latencies else 0.0,
        p95_latency_ms=round(percentile(latencies, 0.95), 2),
    )
