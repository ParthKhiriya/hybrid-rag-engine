from app.config import Settings
from app.rbac import ApprovalRegistry, resolve_role


def _settings(**overrides) -> Settings:
    return Settings(
        api_keys_reader="reader-key",
        api_keys_writer="writer-key",
        api_keys_admin="admin-key",
        **overrides,
    )


def test_resolve_role_reader_key():
    assert resolve_role("reader-key", _settings()) == "reader"


def test_resolve_role_writer_key():
    assert resolve_role("writer-key", _settings()) == "writer"


def test_resolve_role_admin_key_outranks_others():
    assert resolve_role("admin-key", _settings()) == "admin"


def test_resolve_role_unknown_key_is_none():
    assert resolve_role("not-a-real-key", _settings()) is None


def test_resolve_role_missing_key_is_none():
    assert resolve_role(None, _settings()) is None


def test_approval_registry_issue_and_consume():
    registry = ApprovalRegistry()
    token = registry.issue("purge")
    assert registry.consume(token, "purge") is True


def test_approval_registry_token_is_single_use():
    registry = ApprovalRegistry()
    token = registry.issue("purge")
    assert registry.consume(token, "purge") is True
    assert registry.consume(token, "purge") is False


def test_approval_registry_wrong_action_rejected():
    registry = ApprovalRegistry()
    token = registry.issue("purge")
    assert registry.consume(token, "delete_all") is False


def test_approval_registry_unknown_token_rejected():
    registry = ApprovalRegistry()
    assert registry.consume("not-a-real-token", "purge") is False
