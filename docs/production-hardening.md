# Production hardening checklist

- Replace development credentials and enforce secret-manager integration.
- Put API, Qdrant, and Neo4j on private networks; enforce TLS/mTLS where appropriate.
- ~~Add authentication and authorization~~ — implemented: RBAC via scoped API keys (`app/rbac.py`); still needed: tenant-level retrieval filters, key rotation, and a real identity provider instead of static keys.
- Run ingestion asynchronously through a queue and make all writes idempotent.
- Use production embedding models and benchmark dimensionality/latency/cost.
- Add a cross-encoder or LLM reranker only after measuring its quality lift and latency cost.
- Enable Qdrant replication/sharding and snapshot policies based on RPO/RTO.
- Configure Neo4j backups, indexes, least-privilege roles, and query timeouts. **Least privilege specifically:** the app currently connects with one Neo4j user that can read and write; production should split into a write-scoped ingestion role and a read-only query role, and likewise use a scoped/read-only Qdrant API key for the query path.
- Add distributed tracing exporter, log correlation IDs, dashboards, and SLO alerts.
- Version prompts, embeddings, graph schema, retrieval settings, and eval datasets.
- Run regression evaluation in CI/CD and block promotion on material quality regressions.
- ~~Add prompt-injection defenses~~ — implemented at the heuristic level: input scanning at ingest (`app/security.py`), source trust scoring that fences untrusted content out of the instruction channel, output validation, and a human-approval gate that separates requesting a destructive action from executing it (`app/rbac.py`). Still needed for a real production system: a trained classifier (regex/heuristics alone are bypassable), tool-call allowlisting with argument-level schema validation, sandboxed execution for any tool with side effects, and document/data classification beyond the trust label used here.
- Generate an SBOM, scan dependencies/images, sign images, and verify provenance during deployment.
- Add rate limiting per API key and audit logging of every write/admin action.
