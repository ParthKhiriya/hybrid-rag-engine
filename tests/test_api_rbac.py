import os

os.environ.setdefault("ALLOW_IN_MEMORY_FALLBACK", "true")
os.environ.setdefault("LLM_PROVIDER", "local")
os.environ["API_KEYS_READER"] = "test-reader"
os.environ["API_KEYS_WRITER"] = "test-writer"
os.environ["API_KEYS_ADMIN"] = "test-admin"

from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.main import app  # noqa: E402

get_settings.cache_clear()
client = TestClient(app)


def test_health_is_public():
    response = client.get("/health")
    assert response.status_code == 200


def test_query_without_key_is_unauthorized():
    response = client.post("/v1/query", json={"question": "What depends on the gateway?"})
    assert response.status_code == 401


def test_query_with_reader_key_succeeds():
    response = client.post(
        "/v1/query",
        json={"question": "What depends on the gateway?"},
        headers={"X-API-Key": "test-reader"},
    )
    assert response.status_code == 200


def test_ingest_with_reader_key_is_forbidden():
    response = client.post(
        "/v1/ingest/sample",
        headers={"X-API-Key": "test-reader"},
    )
    assert response.status_code == 403


def test_ingest_with_writer_key_succeeds():
    response = client.post(
        "/v1/ingest/sample",
        headers={"X-API-Key": "test-writer"},
    )
    assert response.status_code == 200


def test_metrics_requires_admin():
    reader_response = client.get("/metrics", headers={"X-API-Key": "test-reader"})
    admin_response = client.get("/metrics", headers={"X-API-Key": "test-admin"})
    assert reader_response.status_code == 403
    assert admin_response.status_code == 200


def test_purge_requires_admin_and_approval_token():
    no_approval = client.post("/v1/admin/purge", headers={"X-API-Key": "test-admin"})
    assert no_approval.status_code == 412

    issued = client.post(
        "/v1/admin/approvals",
        params={"action": "purge"},
        headers={"X-API-Key": "test-admin"},
    )
    assert issued.status_code == 200
    token = issued.json()["approval_token"]

    approved = client.post(
        "/v1/admin/purge",
        headers={"X-API-Key": "test-admin", "X-Approval-Token": token},
    )
    assert approved.status_code == 200

    reused = client.post(
        "/v1/admin/purge",
        headers={"X-API-Key": "test-admin", "X-Approval-Token": token},
    )
    assert reused.status_code == 412
