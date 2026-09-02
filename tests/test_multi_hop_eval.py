from app.agent import RAGAgent
from app.embeddings import HashingEmbedder
from app.graph_store import InMemoryGraphStore
from app.ingest import IngestionService
from app.llm import LocalExtractiveProvider
from app.models import DocumentIn, EvalCase
from app.retriever import HybridRetriever
from app.vector_store import InMemoryVectorStore


def build_multi_hop_agent():
    embedder = HashingEmbedder(128)
    vector = InMemoryVectorStore()
    graph = InMemoryGraphStore()
    ingestion = IngestionService(embedder, vector, graph)
    retriever = HybridRetriever(embedder, vector, graph, vector_k=4, graph_k=4, graph_max_hops=3)
    agent = RAGAgent(retriever, LocalExtractiveProvider())
    ingestion.ingest(
        [
            DocumentIn(
                source="chain.md",
                text=(
                    "The Embedding Cache depends on the Redis Cluster. "
                    "The Redis Cluster runs on the Cloud Provider. "
                    "The Cloud Provider is monitored by Datadog."
                ),
                trust="trusted",
            )
        ]
    )
    return agent


def test_multi_hop_eval_metric_credits_genuine_traversal():
    from app.evaluation import evaluate

    agent = build_multi_hop_agent()
    cases = [
        EvalCase(
            question="Which tool monitors the infrastructure that powers the Embedding Cache?",
            expected_sources=["chain.md"],
            expected_keywords=["Datadog"],
            requires_hops=3,
        )
    ]
    summary = evaluate(agent, cases)
    assert summary.multi_hop_cases == 1
    assert summary.multi_hop_hit_rate == 100.0


def test_multi_hop_metric_zero_when_no_multi_hop_cases():
    from app.evaluation import evaluate

    agent = build_multi_hop_agent()
    cases = [
        EvalCase(
            question="What does the Embedding Cache depend on?",
            expected_sources=["chain.md"],
            expected_keywords=["Redis Cluster"],
        )
    ]
    summary = evaluate(agent, cases)
    assert summary.multi_hop_cases == 0
    assert summary.multi_hop_hit_rate == 0.0
