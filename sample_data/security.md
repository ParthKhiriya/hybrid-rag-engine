# Security Model

The API Gateway is owned by Platform Engineering. The API Gateway depends on Identity Provider. The API Gateway is monitored by Prometheus.

Production deployment uses TLS, workload identity, least-privilege database roles, private networking, tenant metadata filters, audit logging, input validation, rate limiting, and a managed secrets service. Tool-enabled agents must authorize every side-effecting tool call independently from the model decision.

Sensitive documents should be classified before indexing and retrieval responses should retain source metadata for audit and access-control enforcement.
