# Retrieval and Data Pipeline

The Retrieval Service depends on Qdrant. The Retrieval Service depends on Neo4j. The Retrieval Service is monitored by Prometheus. Qdrant stores vector embeddings and chunk metadata. Neo4j stores entities and explicit relationships between platform components, owners, services, and dependencies.

The Ingestion Worker depends on the Object Store. The Ingestion Worker is owned by Data Platform. Documents are normalized, chunked, embedded, and indexed into Qdrant. Entity and relationship extraction populates Neo4j so the query path can combine semantic similarity with graph traversal.

Hybrid search merges vector and graph rankings with Reciprocal Rank Fusion before lightweight reranking.

The Embedding Cache depends on the Redis Cluster. The Redis Cluster runs on the Cloud Provider. The Cloud Provider is monitored by Datadog.
