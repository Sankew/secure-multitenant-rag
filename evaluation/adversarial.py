"""Repeatable end-to-end authorization probes against the fictional corpus."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
from fastapi.testclient import TestClient

from secure_rag.api import create_app, load_service


ROOT = Path(__file__).resolve().parents[1]
SECRET = "adversarial-evaluation-secret-longer-than-32-bytes"
QUESTIONS = (
    "annual leave", "pay bands", "twenty days", "twelve days",
    "70000 dollars", "60000 dollars", "HR portal", "operations lead",
    "Level A", "Grade 1", "staff requests", "fictional policy",
)


def bearer(user: str, tenant: str, role: str, clearance: int) -> dict[str, str]:
    claims = {
        "sub": user, "tenant_id": tenant, "roles": [role] if role else [],
        "clearance": clearance, "iss": "eval", "aud": "eval-api",
        "exp": datetime.now(timezone.utc) + timedelta(minutes=10),
    }
    return {"Authorization": f"Bearer {jwt.encode(claims, SECRET, algorithm='HS256')}"}


def run() -> dict[str, int]:
    service = load_service(ROOT / "data" / "fictional_documents.jsonl")
    api = TestClient(create_app(service, secret=SECRET, issuer="eval", audience="eval-api"))
    source_text = {chunk.chunk_id: chunk.text for chunk in service.chunks.values()}
    requests = 0
    authorization_violations = 0
    answer_failures = 0
    false_denials = 0
    positive_probes = 0

    # Each query passes through JWT verification, search, ACL prefilter,
    # authoritative recheck, answer generation, and citation formatting.
    for tenant in ("acme", "beacon", "outsider"):
        for role, clearance in (("staff", 0), ("hr", 0), ("hr", 2), ("", 2)):
            permitted = set()
            if tenant in ("acme", "beacon") and role == "staff":
                permitted.add(f"{tenant}:leave")
            if tenant in ("acme", "beacon") and role == "hr" and clearance >= 2:
                permitted.add(f"{tenant}:pay")
            headers = bearer(f"{tenant}-{role or 'none'}", tenant, role, clearance)
            for question in QUESTIONS:
                response = api.post("/answer", json={"question": question}, headers=headers)
                requests += 1
                if response.status_code != 200:
                    answer_failures += 1
                    continue
                body = response.json()
                citations = set(body["citations"])
                if not citations <= permitted:
                    authorization_violations += 1
                if citations and not any(c in source_text and body["answer"] in source_text[c] for c in citations):
                    answer_failures += 1
                if not citations and body["answer"] != "I don't know based on the documents you can access.":
                    answer_failures += 1

    # Fixed expected citations catch an implementation that refuses every query.
    for tenant, role, clearance, question, expected in (
        ("acme", "staff", 0, "annual leave", "acme:leave"),
        ("beacon", "staff", 0, "annual leave", "beacon:leave"),
        ("acme", "hr", 2, "pay bands", "acme:pay"),
        ("beacon", "hr", 2, "pay bands", "beacon:pay"),
    ):
        response = api.post("/answer", json={"question": question}, headers=bearer("positive", tenant, role, clearance))
        requests += 1
        positive_probes += 1
        if response.status_code != 200 or response.json()["citations"] != [expected]:
            false_denials += 1

    # Search metadata still grants staff, while the authoritative ACL revokes it.
    old = service.acls.get("acme:leave")
    assert old is not None
    pre = api.post("/answer", json={"question": "annual leave"}, headers=bearer("revoked", "acme", "staff", 0))
    requests += 1
    positive_probes += 1
    if pre.status_code != 200 or pre.json()["citations"] != ["acme:leave"]:
        false_denials += 1
    service.update_acl("acme", "leave", replace(old, allowed_roles=frozenset(), revision=2))
    for question in QUESTIONS:
        response = api.post("/answer", json={"question": question}, headers=bearer("revoked", "acme", "staff", 0))
        requests += 1
        if response.status_code != 200:
            answer_failures += 1
        elif response.json()["citations"]:
            authorization_violations += 1

    return {
        "requests": requests,
        "positive_probes": positive_probes,
        "authorization_violations": authorization_violations,
        "false_denials": false_denials,
        "answer_failures": answer_failures,
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, sort_keys=True))
    if any(result[key] for key in ("authorization_violations", "false_denials", "answer_failures")):
        raise SystemExit(1)
