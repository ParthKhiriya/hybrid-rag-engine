from app.agent import RAGAgent
from app.embeddings import HashingEmbedder
from app.graph_store import InMemoryGraphStore
from app.ingest import IngestionService
from app.llm import LocalExtractiveProvider
from app.models import DocumentIn
from app.retriever import HybridRetriever
from app.vector_store import InMemoryVectorStore


def build_local_agent():
    embedder = HashingEmbedder(128)
    vector = InMemoryVectorStore()
    graph = InMemoryGraphStore()
    ingestion = IngestionService(embedder, vector, graph)
    retriever = HybridRetriever(embedder, vector, graph, vector_k=4, graph_k=4)
    agent = RAGAgent(retriever, LocalExtractiveProvider())
    return ingestion, agent


def test_end_to_end_local_pipeline():
    ingestion, agent = build_local_agent()
    ingestion.ingest([
        DocumentIn(
            source="demo.md",
            text=(
                "The Inference Gateway depends on GPU Cluster. "
                "The Inference Gateway is monitored by Prometheus."
            ),
        )
    ])
    result = agent.invoke("How is the Inference Gateway monitored?", top_k=3)
    assert result.sources
    assert any(s.source == "demo.md" for s in result.sources)
    assert "Inference Gateway" in result.answer
