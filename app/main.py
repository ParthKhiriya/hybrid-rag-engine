from fastapi import Depends, FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.bootstrap import build_services
from app.evaluation import evaluate
from app.ingest import load_sample_documents
from app.models import (
    EvalSummary,
    EvaluateRequest,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.observability import REQUESTS, configure_tracing
from app.rbac import approval_registry, require_approval, require_role

app = FastAPI(
    title="Hybrid RAG Engine",
    version="0.2.0",
    description="Hybrid vector + knowledge-graph retrieval engine with LangGraph orchestration.",
)
services = build_services()
configure_tracing(app, services.settings.otel_enabled, services.settings.otel_service_name)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "vector": {
            "backend": services.vector_store.backend_name,
            "healthy": services.vector_store.health(),
        },
        "graph": {
            "backend": services.graph_store.backend_name,
            "healthy": services.graph_store.health(),
        },
        "provider": services.agent.llm.name,
    }


@app.get("/metrics", dependencies=[Depends(require_role("admin"))])
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/v1/ingest/sample",
    response_model=IngestResponse,
    dependencies=[Depends(require_role("writer"))],
)
def ingest_sample() -> IngestResponse:
    try:
        result = services.ingestion.ingest(load_sample_documents())
        REQUESTS.labels(endpoint="ingest_sample", status="ok").inc()
        return result
    except Exception:
        REQUESTS.labels(endpoint="ingest_sample", status="error").inc()
        raise


@app.post(
    "/v1/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(require_role("writer"))],
)
def ingest(request: IngestRequest) -> IngestResponse:
    try:
        result = services.ingestion.ingest(request.documents)
        REQUESTS.labels(endpoint="ingest", status="ok").inc()
        return result
    except Exception:
        REQUESTS.labels(endpoint="ingest", status="error").inc()
        raise


@app.post(
    "/v1/query",
    response_model=QueryResponse,
    dependencies=[Depends(require_role("reader"))],
)
def query(request: QueryRequest) -> QueryResponse:
    try:
        result = services.agent.invoke(request.question, request.top_k)
        REQUESTS.labels(endpoint="query", status="ok").inc()
        return result
    except Exception:
        REQUESTS.labels(endpoint="query", status="error").inc()
        raise


@app.post(
    "/v1/evaluate",
    response_model=EvalSummary,
    dependencies=[Depends(require_role("admin"))],
)
def run_evaluation(request: EvaluateRequest) -> EvalSummary:
    from app.eval_runner import load_cases

    cases = request.cases or load_cases()
    return evaluate(services.agent, cases)


# --- Admin / destructive actions ------------------------------------------
#
# Any state-changing action beyond ingest (which is additive) is both
# admin-only AND gated behind a human-approval token minted by a separate
# call. This prevents a single request -- including one carrying an
# injected instruction -- from both requesting and approving its own
# destructive action.


@app.post("/v1/admin/approvals", dependencies=[Depends(require_role("admin"))])
def request_approval(action: str) -> dict:
    """Mint a short-lived, single-use approval token for a named destructive
    action. In production this issuance would be triggered by a human
    (chat-ops approval, ticketing system) rather than an API call an admin
    key alone can make."""
    token = approval_registry.issue(action)
    return {"action": action, "approval_token": token, "expires_in_seconds": 300}


@app.post(
    "/v1/admin/purge",
    dependencies=[Depends(require_role("admin")), Depends(require_approval("purge"))],
)
def purge_index() -> dict:
    """Example destructive action: clears the in-memory fallback stores.
    Requires admin role AND a valid, unused approval token for the 'purge'
    action (see /v1/admin/approvals)."""
    if hasattr(services.vector_store, "records"):
        services.vector_store.records.clear()
    if hasattr(services.graph_store, "docs"):
        services.graph_store.docs.clear()
        services.graph_store.doc_entities.clear()
        services.graph_store.relations.clear()
    return {"status": "purged"}
