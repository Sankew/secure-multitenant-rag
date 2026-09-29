"""Authorization for retrieved documents.

Vector metadata is a useful prefilter, but the ACL store is authoritative. A
revocation therefore takes effect before an asynchronous vector update lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from dataclasses import replace
from threading import RLock
from typing import Callable, Iterable

from .domain import ACL, Chunk, Principal


def can_read(principal: Principal, acl: ACL) -> bool:
    """Return whether this principal may read content under ``acl``."""
    return (
        principal.tenant_id == acl.tenant_id
        and principal.clearance >= acl.classification
        and (
            principal.user_id in acl.allowed_users
            or bool(principal.roles & acl.allowed_roles)
        )
    )


def current_acl_prefilter(principal: Principal, acl_store: ACLStore) -> Callable[[Chunk], bool]:
    """Filter the local index with current permissions before ranking."""

    def allowed(chunk: Chunk) -> bool:
        current = acl_store.get(chunk.document_id)
        return current is not None and can_read(principal, current)

    return allowed


def qdrant_payload_filter(principal: Principal) -> dict[str, object]:
    """Qdrant filter for flattened ACL payload fields on each vector point."""
    grants = [{"key": "allowed_users", "match": {"value": principal.user_id}}]
    grants.extend({"key": "allowed_roles", "match": {"value": role}} for role in sorted(principal.roles))
    return {"must": [
        {"key": "tenant_id", "match": {"value": principal.tenant_id}},
        {"key": "classification", "range": {"lte": principal.clearance}},
        {"should": grants},
    ]}


class ACLStore:
    """Thread-safe source of truth for current document ACLs.

    The in-memory implementation keeps examples and tests self-contained. A
    persistent deployment can replace it while retaining the same interface.
    """

    def __init__(self) -> None:
        self._acls: dict[str, ACL] = {}
        self._lock = RLock()

    def get(self, document_id: str) -> ACL | None:
        with self._lock:
            return self._acls.get(document_id)

    def update(self, document_id: str, acl: ACL) -> None:
        """Create or replace an ACL, rejecting stale and cross-tenant writes."""
        with self._lock:
            prior = self._acls.get(document_id)
            if prior is not None:
                if prior.tenant_id != acl.tenant_id:
                    raise ValueError("document tenant cannot change")
                if acl.revision <= prior.revision:
                    raise ValueError("ACL revision must increase")
            self._acls[document_id] = acl

    def delete(self, document_id: str) -> None:
        """Immediately revoke access to a removed document."""
        with self._lock:
            self._acls.pop(document_id, None)


@dataclass(frozen=True)
class AuditRow:
    request_id: str
    timestamp_utc: str
    user_id: str
    tenant_id: str
    query: str
    model_id: str
    returned_chunk_ids: tuple[str, ...]
    rejected_chunk_ids: tuple[str, ...]


class AuditLog:
    def __init__(self) -> None:
        self._rows: list[AuditRow] = []
        self._lock = RLock()

    def record(self, row: AuditRow) -> None:
        with self._lock:
            self._rows.append(row)

    def rows(self) -> tuple[AuditRow, ...]:
        with self._lock:
            return tuple(self._rows)

    def set_result(self, request_id: str, model_id: str, returned_chunk_ids: tuple[str, ...]) -> None:
        with self._lock:
            for index, row in enumerate(self._rows):
                if row.request_id == request_id:
                    self._rows[index] = replace(row, model_id=model_id, returned_chunk_ids=returned_chunk_ids)
                    break


def secure_retrieve(
    principal: Principal,
    candidates: Iterable[Chunk],
    acl_store: ACLStore,
    *,
    limit: int = 5,
    audit: AuditLog | None = None,
    request_id: str = "",
    query: str = "",
) -> tuple[Chunk, ...]:
    """Return up to ``limit`` authorized candidates in their supplied rank order.

    The caller supplies a vector-ranked candidate stream. We use the current
    ACL from the authoritative store. Rejected candidates never reach generation
    or citation code. The stream is consumed until ``limit``
    authorized chunks are found, so revoked candidates do not shrink top-k.
    """
    if limit < 0:
        raise ValueError("limit must be non-negative")

    returned: list[Chunk] = []
    rejected: list[str] = []
    if limit:
        for chunk in candidates:
            current = acl_store.get(chunk.document_id)
            if current is None or not can_read(principal, current):
                rejected.append(chunk.chunk_id)
                continue
            returned.append(chunk)
            if len(returned) == limit:
                break

    if audit is not None:
        audit.record(
            AuditRow(
                request_id=request_id,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                user_id=principal.user_id,
                tenant_id=principal.tenant_id,
                query=query,
                model_id="",
                returned_chunk_ids=tuple(c.chunk_id for c in returned),
                rejected_chunk_ids=tuple(rejected),
            )
        )
    return tuple(returned)
