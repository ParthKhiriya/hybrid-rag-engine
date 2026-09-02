from dataclasses import dataclass

from app.agent import RAGAgent
from app.config import Settings, get_settings
from app.embeddings import HashingEmbedder
from app.graph_store import GraphStore, InMemoryGraphStore, Neo4jGraphStore
from app.ingest import IngestionService
from app.llm import AnthropicProvider, LLMProvider, LocalExtractiveProvider, OpenAIProvider
from app.retriever import HybridRetriever
from app.vector_store import InMemoryVectorStore, QdrantVectorStore, VectorStore


@dataclass(slots=True)
class Services:
    settings: Settings
    vector_store: VectorStore
    graph_store: GraphStore
    ingestion: IngestionService
    agent: RAGAgent


def _vector_store(settings: Settings, embedder: HashingEmbedder) -> VectorStore:
    try:
        return QdrantVectorStore(
            settings.qdrant_url,
            settings.qdrant_collection,
            embedder.dimension,
            settings.qdrant_api_key,
        )
    except Exception:
        if not settings.allow_in_memory_fallback:
            raise
        return InMemoryVectorStore()


def _graph_store(settings: Settings) -> GraphStore:
    try:
        return Neo4jGraphStore(
            settings.neo4j_uri,
            settings.neo4j_user,
            settings.neo4j_password,
            settings.neo4j_database,
        )
    except Exception:
        if not settings.allow_in_memory_fallback:
            raise
        return InMemoryGraphStore()


def _llm(settings: Settings) -> LLMProvider:
    provider = settings.llm_provider.lower()
    if provider == "openai" and settings.openai_api_key:
        return OpenAIProvider(settings.openai_api_key, settings.openai_model)
    if provider == "anthropic" and settings.anthropic_api_key:
        return AnthropicProvider(settings.anthropic_api_key, settings.anthropic_model)
    return LocalExtractiveProvider()


def build_services(settings: Settings | None = None) -> Services:
    settings = settings or get_settings()
    embedder = HashingEmbedder(settings.embedding_dim)
    vector = _vector_store(settings, embedder)
    graph = _graph_store(settings)
    ingestion = IngestionService(embedder, vector, graph, settings.block_quarantined_ingest)
    retriever = HybridRetriever(
        embedder,
        vector,
        graph,
        settings.top_k_vector,
        settings.top_k_graph,
        settings.rrf_k,
        settings.exclude_quarantined_from_context,
        settings.graph_max_hops,
    )
    agent = RAGAgent(retriever, _llm(settings))
    return Services(settings, vector, graph, ingestion, agent)
