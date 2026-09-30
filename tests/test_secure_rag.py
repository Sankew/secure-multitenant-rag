from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
import pytest
from fastapi.testclient import TestClient

from secure_rag.access import qdrant_payload_filter
from secure_rag.api import build_app, create_app, load_service
from secure_rag.core import SecureService
from secure_rag.domain import ACL, Principal


DATA = Path(__file__).resolve().parents[1] / "data" / "fictional_documents.jsonl"
SECRET = "local-test-secret-with-at-least-thirty-two-bytes"


def token(user: str, tenant: str, roles: list[str], clearance: int = 0) -> str:
    return jwt.encode(
        {"sub": user, "tenant_id": tenant, "roles": roles, "clearance": clearance,
         "iss": "test", "aud": "test-api", "exp": datetime.now(timezone.utc) + timedelta(minutes=5)},
        SECRET, algorithm="HS256",
    )


def client() -> tuple[TestClient, SecureService]:
    service = load_service(DATA)
    return TestClient(create_app(service, secret=SECRET, issuer="test", audience="test-api")), service


def test_tenant_role_and_classification_boundaries() -> None:
    api, _ = client()
    headers = {"Authorization": f"Bearer {token('alice', 'acme', ['staff'])}"}
    response = api.post("/answer", json={"question": "annual leave"}, headers=headers)
    assert response.status_code == 200
    assert response.json()["citations"] == ["acme:leave"]

    pay = api.post("/answer", json={"question": "pay bands"}, headers=headers)
    assert pay.json()["citations"] == []
    assert "70000" not in pay.json()["answer"]
    assert "60000" not in pay.json()["answer"]

    hr = {"Authorization": f"Bearer {token('helen', 'acme', ['hr'], 2)}"}
    permitted = api.post("/answer", json={"question": "pay bands"}, headers=hr)
    assert permitted.json()["citations"] == ["acme:pay"]
    assert "70000" in permitted.json()["answer"]


def test_tenants_can_reuse_local_document_ids() -> None:
    _, service = client()
    assert set(service.chunks) == {"acme:leave", "acme:pay", "beacon:leave", "beacon:pay"}
    assert service.acls.get("acme:leave").tenant_id == "acme"
    assert service.acls.get("beacon:leave").tenant_id == "beacon"


def test_revocation_immediately_blocks_stale_index_entry() -> None:
    api, service = client()
    current = service.acls.get("acme:leave")
    assert current is not None
    service.update_acl("acme", "leave", replace(current, allowed_roles=frozenset(), revision=2))
    headers = {"Authorization": f"Bearer {token('alice', 'acme', ['staff'])}"}
    response = api.post("/answer", json={"question": "annual leave"}, headers=headers)
    assert response.status_code == 200
    assert response.json()["citations"] == []
    assert "twenty" not in response.json()["answer"]


def test_new_grant_is_visible_without_reindexing() -> None:
    api, service = client()
    current = service.acls.get("acme:pay")
    assert current is not None
    service.update_acl("acme", "pay", replace(current, allowed_roles=frozenset({"hr", "staff"}), classification=0, revision=2))
    headers = {"Authorization": f"Bearer {token('alice', 'acme', ['staff'])}"}
    response = api.post("/answer", json={"question": "pay bands"}, headers=headers)
    assert response.status_code == 200
    assert response.json()["citations"] == ["acme:pay"]


def test_question_uses_matching_sentence_and_ignores_stopwords() -> None:
    api, _ = client()
    headers = {"Authorization": f"Bearer {token('alice', 'acme', ['staff'])}"}
    portal = api.post("/answer", json={"question": "HR portal"}, headers=headers)
    assert portal.json()["citations"] == ["acme:leave"]
    assert portal.json()["answer"] == "Requests go through the HR portal."
    unrelated = api.post("/answer", json={"question": "What are the pay bands?"}, headers=headers)
    assert unrelated.json()["citations"] == []


def test_corpus_rejects_role_string_that_would_split_into_characters(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text('{"document_id":"pay","text":"Salary bands.","tenant_id":"acme","allowed_roles":"hr","allowed_users":[],"classification":0}\n')
    with pytest.raises(ValueError):
        load_service(invalid)


def test_default_corpus_loads_outside_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RAG_JWT_SECRET", SECRET)
    monkeypatch.delenv("RAG_DOCUMENTS", raising=False)
    assert any(route.path == "/answer" for route in build_app().routes)


def test_user_grant_respects_tenant_clearance_and_revocation() -> None:
    api, service = client()
    service.add_document(
        "launch", "The amber phase begins next month.",
        ACL("acme", frozenset(), frozenset({"alice"}), classification=2),
    )

    def citations(user: str, tenant: str, clearance: int) -> list[str]:
        headers = {"Authorization": f"Bearer {token(user, tenant, [], clearance)}"}
        response = api.post("/answer", json={"question": "amber phase"}, headers=headers)
        assert response.status_code == 200
        return response.json()["citations"]

    assert citations("alice", "acme", 2) == ["acme:launch"]
    assert citations("alice", "acme", 0) == []
    assert citations("alice", "beacon", 2) == []
    assert citations("bob", "acme", 2) == []

    current = service.acls.get("acme:launch")
    assert current is not None
    service.update_acl("acme", "launch", replace(current, allowed_users=frozenset(), revision=2))
    assert citations("alice", "acme", 2) == []


def test_invalid_token_and_body_auth_fields_are_rejected() -> None:
    api, _ = client()
    missing = api.post("/answer", json={"question": "leave"})
    assert missing.status_code == 401
    assert missing.headers["WWW-Authenticate"] == "Bearer"
    headers = {"Authorization": f"Bearer {token('alice', 'acme', ['staff'])}"}
    assert api.post("/answer", json={"question": "leave", "tenant_id": "beacon"}, headers=headers).status_code == 422
    for blank in ("", "   ", "\n\t"):
        assert api.post("/answer", json={"question": blank}, headers=headers).status_code == 422


def test_wrong_signature_expiry_issuer_audience_algorithm_and_missing_claims() -> None:
    api, _ = client()
    valid = {
        "sub": "alice", "tenant_id": "acme", "roles": ["staff"], "clearance": 0,
        "iss": "test", "aud": "test-api", "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
    }
    bad_tokens = [
        jwt.encode(valid, "different-signing-secret-that-is-long-enough", algorithm="HS256"),
        jwt.encode({**valid, "exp": datetime.now(timezone.utc) - timedelta(minutes=5)}, SECRET, algorithm="HS256"),
        jwt.encode({**valid, "iss": "other"}, SECRET, algorithm="HS256"),
        jwt.encode({**valid, "aud": "other"}, SECRET, algorithm="HS256"),
        jwt.encode(valid, key="", algorithm="none"),
        jwt.encode({key: value for key, value in valid.items() if key != "roles"}, SECRET, algorithm="HS256"),
        jwt.encode({key: value for key, value in valid.items() if key != "clearance"}, SECRET, algorithm="HS256"),
        jwt.encode(valid, SECRET + "-padding-for-hs512-key-length-requirements", algorithm="HS512"),
    ]
    for bad in bad_tokens:
        response = api.post("/answer", json={"question": "leave"}, headers={"Authorization": f"Bearer {bad}"})
        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize("claims", [
    {"sub": ""}, {"tenant_id": ""}, {"roles": "staff"}, {"roles": ["staff", 1]},
    {"clearance": True}, {"clearance": 2.0}, {"clearance": "2"}, {"clearance": -1},
])
def test_malformed_identity_claims_are_rejected(claims: dict) -> None:
    api, _ = client()
    valid = {
        "sub": "alice", "tenant_id": "acme", "roles": ["staff"], "clearance": 0,
        "iss": "test", "aud": "test-api", "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
    }
    bad = jwt.encode({**valid, **claims}, SECRET, algorithm="HS256")
    response = api.post("/answer", json={"question": "leave"}, headers={"Authorization": f"Bearer {bad}"})
    assert response.status_code == 401


def test_openapi_documents_the_answer_schema() -> None:
    api, _ = client()
    schema = api.get("/openapi.json").json()
    response = schema["paths"]["/answer"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert response == {"$ref": "#/components/schemas/AnswerBody"}
    assert set(schema["components"]["schemas"]["AnswerBody"]["required"]) == {"answer", "citations", "request_id"}


def test_irrelevant_accessible_document_is_not_cited() -> None:
    api, _ = client()
    headers = {"Authorization": f"Bearer {token('alice', 'acme', ['staff'])}"}
    response = api.post("/answer", json={"question": "pay bands"}, headers=headers)
    assert response.status_code == 200
    assert response.json()["citations"] == []


def test_wrong_tenant_acl_update_is_rejected() -> None:
    _, service = client()
    beacon = service.acls.get("beacon:leave")
    assert beacon is not None
    try:
        service.update_acl("acme", "leave", replace(beacon, revision=2))
    except ValueError as error:
        assert "tenant" in str(error)
    else:
        raise AssertionError("cross-tenant update was accepted")
    assert service.acls.get("acme:leave").revision == 1


def test_audit_has_request_metadata_without_source_text() -> None:
    api, service = client()
    headers = {"Authorization": f"Bearer {token('alice', 'acme', ['staff'])}"}
    response = api.post("/answer", json={"question": "annual leave"}, headers=headers)
    row = service.audit.rows()[-1]
    assert row.request_id == response.json()["request_id"]
    assert row.timestamp_utc.endswith("+00:00")
    assert row.query == "annual leave"
    assert row.model_id == "extractive-demo"
    assert "twenty days" not in repr(row)


def test_qdrant_filter_has_tenant_classification_and_grant_branches() -> None:
    principal = Principal("alice", "acme", frozenset({"staff", "auditor"}), 2)
    clauses = qdrant_payload_filter(principal)["must"]
    assert clauses[0] == {"key": "tenant_id", "match": {"value": "acme"}}
    assert clauses[1] == {"key": "classification", "range": {"lte": 2}}
    assert {"key": "allowed_users", "match": {"value": "alice"}} in clauses[2]["should"]
    assert {"key": "allowed_roles", "match": {"value": "staff"}} in clauses[2]["should"]
