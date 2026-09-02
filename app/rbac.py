"""Role-based access control and human-approval gating.

Design:
- Three roles, least privilege by default: reader < writer < admin.
- Roles are bound to API keys via environment configuration — never
  hardcoded, never derived from anything the caller controls beyond the
  key itself.
- Endpoints declare the *minimum* role they need via `require_role(...)`.
- Destructive/irreversible actions additionally require a short-lived
  approval token minted by a separate call (`/v1/admin/approvals`), so a
  single compromised or injected request cannot both request and approve
  its own destructive action. In a real deployment that second call would
  be triggered by a human (Slack/PagerDuty approval flow), not by the
  agent itself.
"""

import secrets
import time
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import Settings, get_settings

ROLE_RANK = {"reader": 0, "writer": 1, "admin": 2}

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _keys_for_role(settings: Settings, role: str) -> set[str]:
    raw = {
        "reader": settings.api_keys_reader,
        "writer": settings.api_keys_writer,
        "admin": settings.api_keys_admin,
    }[role]
    return {k.strip() for k in raw.split(",") if k.strip()}


def resolve_role(api_key: str | None, settings: Settings) -> str | None:
    if not api_key:
        return None
    # Check highest privilege first so an admin key also satisfies lower checks.
    for role in ("admin", "writer", "reader"):
        if api_key in _keys_for_role(settings, role):
            return role
    return None


def get_current_role(
    api_key: str | None = Security(_api_key_header),
    settings: Settings = Depends(get_settings),  # noqa: B008 -- required FastAPI DI pattern
) -> str:
    role = resolve_role(api_key, settings)
    if role is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
        )
    return role


def require_role(minimum: str):
    def _check(role: str = Depends(get_current_role)) -> str:
        if ROLE_RANK[role] < ROLE_RANK[minimum]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires '{minimum}' role or higher; caller has '{role}'.",
            )
        return role

    return _check


# --- Human-approval gate for destructive actions --------------------------

APPROVAL_TTL_SECONDS = 300


@dataclass(slots=True)
class Approval:
    action: str
    issued_at: float
    used: bool = False


class ApprovalRegistry:
    """In-memory approval-token store.

    A real deployment would back this with a durable store and route
    issuance through an actual human approval channel (chat-ops button,
    ticketing system) rather than an API call any admin key can make.
    """

    def __init__(self) -> None:
        self._approvals: dict[str, Approval] = {}

    def issue(self, action: str) -> str:
        token = secrets.token_urlsafe(24)
        self._approvals[token] = Approval(action=action, issued_at=time.time())
        return token

    def consume(self, token: str, action: str) -> bool:
        approval = self._approvals.get(token)
        if approval is None or approval.used or approval.action != action:
            return False
        if time.time() - approval.issued_at > APPROVAL_TTL_SECONDS:
            return False
        approval.used = True
        return True


approval_registry = ApprovalRegistry()


def require_approval(action: str):
    def _check(x_approval_token: str | None = Header(default=None)) -> None:
        if not x_approval_token or not approval_registry.consume(x_approval_token, action):
            raise HTTPException(
                status_code=status.HTTP_412_PRECONDITION_FAILED,
                detail=(
                    f"This action ('{action}') requires a valid, unused approval token. "
                    "Request one via POST /v1/admin/approvals and supply it as "
                    "X-Approval-Token."
                ),
            )

    return _check
