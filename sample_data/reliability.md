# Reliability and Operations

The AI Platform is owned by Platform Engineering. The AI Platform is monitored by Prometheus. Production objectives include availability, p95 latency, successful-token rate, and retrieval quality on a fixed evaluation set.

Incident Response depends on Runbooks. Runbooks are owned by SRE. Major incidents require trace correlation across the API, retriever, vector database, graph database, and model provider. Recovery procedures include stateless API restart, Qdrant snapshot restoration, Neo4j backup recovery, and model-serving rollback.

Changes to prompts, retrieval parameters, embedding versions, and model versions are evaluated against regression cases before promotion.
