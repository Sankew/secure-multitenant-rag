"""Small deterministic retrieval engine used to demonstrate access controls."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from .access import ACLStore, AuditLog, current_acl_prefilter, secure_retrieve
from .domain import ACL, Chunk, Principal


def embed(text: str, dimensions: int = 256) -> tuple[float, ...]:
    values = [0.0] * dimensions
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        index = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big") % dimensions
        values[index] += 1.0
    norm = math.sqrt(sum(value * value for value in values))
    return tuple(value / norm for value in values) if norm else tuple(values)


def terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


STOPWORDS = {
    "a", "an", "are", "do", "does", "for", "how", "in", "is", "of", "on",
    "the", "to", "what", "when", "where", "who",
}


@dataclass(frozen=True)
class Result:
    answer: str
    citations: tuple[str, ...]
    request_id: str


class SecureService:
    def __init__(self, audit: AuditLog | None = None) -> None:
        self.chunks: dict[str, Chunk] = {}
        self.acls = ACLStore()
        self.audit = audit or AuditLog()

    def add_document(self, document_id: str, text: str, acl: ACL) -> None:
        if not document_id or not text.strip() or not acl.tenant_id:
            raise ValueError("document ID, text, and tenant are required")
        if ":" in document_id or ":" in acl.tenant_id:
            raise ValueError("document ID and tenant ID cannot contain ':'")
        scoped_id = f"{acl.tenant_id}:{document_id}"
        if self.acls.get(scoped_id) is not None:
            raise ValueError("document already exists")
        self.acls.update(scoped_id, acl)
        self.chunks[scoped_id] = Chunk(scoped_id, scoped_id, text, acl, embed(text))

    def update_acl(self, tenant_id: str, document_id: str, acl: ACL) -> None:
        if tenant_id != acl.tenant_id:
            raise ValueError("ACL tenant does not match requested tenant")
        scoped_id = f"{tenant_id}:{document_id}"
        if scoped_id not in self.chunks:
            raise KeyError(document_id)
        self.acls.update(scoped_id, acl)

    def answer(self, principal: Principal, question: str, request_id: str) -> Result:
        if not question.strip():
            raise ValueError("question is required")
        vector = embed(question)
        query_terms = terms(question) - STOPWORDS
        allowed = current_acl_prefilter(principal, self.acls)
        ranked = sorted(
            (chunk for chunk in self.chunks.values()
             if allowed(chunk) and query_terms & terms(chunk.text)),
            key=lambda chunk: (-sum(a * b for a, b in zip(vector, chunk.embedding)), chunk.chunk_id),
        )
        evidence = secure_retrieve(
            principal, ranked, self.acls, limit=3, audit=self.audit,
            request_id=request_id, query=question,
        )
        if not evidence:
            self.audit.set_result(request_id, "extractive-demo", ())
            return Result("I don't know based on the documents you can access.", (), request_id)
        # This local fixture quotes the most relevant accessible sentence; it is not an LLM.
        required = {term for term in query_terms if any(char.isdigit() for char in term)}
        best: tuple[int, Chunk, str] | None = None
        for chunk in evidence:
            for sentence in re.split(r"(?<=[.!?])\s+", chunk.text):
                sentence_terms = terms(sentence)
                overlap = len(query_terms & sentence_terms)
                if required <= sentence_terms and (best is None or overlap > best[0]):
                    best = (overlap, chunk, sentence)
        threshold = 1 if len(query_terms) == 1 else 2
        if best is None or best[0] < threshold:
            self.audit.set_result(request_id, "extractive-demo", ())
            return Result("I don't know based on the documents you can access.", (), request_id)
        _, chunk, sentence = best
        self.audit.set_result(request_id, "extractive-demo", (chunk.chunk_id,))
        return Result(sentence, (chunk.chunk_id,), request_id)
