"""JWT-protected HTTP interface for the permission-aware demo."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import jwt
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from .core import SecureService
from .domain import ACL, Principal


CHALLENGE = {"WWW-Authenticate": "Bearer"}


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(max_length=4000, pattern=r"\S")


class AnswerBody(BaseModel):
    answer: str
    citations: list[str]
    request_id: str


class CorpusDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    allowed_roles: list[str]
    allowed_users: list[str]
    classification: int = Field(ge=0)


def principal_from_claims(claims: dict) -> Principal:
    roles, clearance = claims.get("roles"), claims.get("clearance")
    if not isinstance(claims.get("sub"), str) or not claims["sub"]:
        raise ValueError("invalid subject")
    if not isinstance(claims.get("tenant_id"), str) or not claims["tenant_id"]:
        raise ValueError("invalid tenant")
    if not isinstance(roles, list) or not all(isinstance(role, str) for role in roles):
        raise ValueError("invalid roles")
    if type(clearance) is not int or clearance < 0:
        raise ValueError("invalid clearance")
    return Principal(claims["sub"], claims["tenant_id"], frozenset(roles), clearance)


def load_service(path: Path) -> SecureService:
    service = SecureService()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = CorpusDocument.model_validate(json.loads(line))
        policy = ACL(
            item.tenant_id, frozenset(item.allowed_roles),
            frozenset(item.allowed_users), item.classification,
        )
        service.add_document(item.document_id, item.text, policy)
    return service


def create_app(service: SecureService, *, secret: str, issuer: str, audience: str) -> FastAPI:
    if len(secret.encode()) < 32 or not issuer or not audience:
        raise ValueError("a 32-byte JWT secret, issuer, and audience are required")
    bearer = HTTPBearer(auto_error=False)

    def current_principal(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> Principal:
        if credentials is None:
            raise HTTPException(401, "invalid credentials", headers=CHALLENGE)
        try:
            claims = jwt.decode(
                credentials.credentials, secret, algorithms=["HS256"], issuer=issuer,
                audience=audience, options={"require": ["sub", "tenant_id", "roles", "clearance", "iss", "aud", "exp"]},
            )
            return principal_from_claims(claims)
        except (jwt.PyJWTError, ValueError):
            raise HTTPException(401, "invalid credentials", headers=CHALLENGE) from None

    app = FastAPI(title="Secure Multi-tenant RAG Demo")

    @app.post("/answer")
    def answer(body: Question, principal: Principal = Depends(current_principal)) -> AnswerBody:
        result = service.answer(principal, body.question, str(uuid4()))
        return AnswerBody(answer=result.answer, citations=list(result.citations), request_id=result.request_id)

    return app


def build_app() -> FastAPI:
    corpus = (
        Path(os.environ["RAG_DOCUMENTS"])
        if "RAG_DOCUMENTS" in os.environ
        else Path(__file__).resolve().parents[2] / "data" / "fictional_documents.jsonl"
    )
    secret = os.getenv("RAG_JWT_SECRET")
    if not secret:
        raise RuntimeError("Set RAG_JWT_SECRET to a random secret of at least 32 bytes")
    return create_app(
        load_service(corpus), secret=secret,
        issuer=os.getenv("RAG_JWT_ISSUER", "secure-rag-local"),
        audience=os.getenv("RAG_JWT_AUDIENCE", "secure-rag-api"),
    )
