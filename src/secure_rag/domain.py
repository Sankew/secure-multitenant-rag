"""Immutable identities, access rules, and indexed document chunks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Principal:
    user_id: str
    tenant_id: str
    roles: frozenset[str]
    clearance: int = 0


@dataclass(frozen=True)
class ACL:
    tenant_id: str
    allowed_roles: frozenset[str]
    allowed_users: frozenset[str]
    classification: int = 0
    revision: int = 1


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    acl: ACL
    embedding: tuple[float, ...]
