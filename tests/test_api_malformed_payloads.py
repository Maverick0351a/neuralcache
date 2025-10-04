import json

import pytest
from fastapi.testclient import TestClient

from neuralcache.api.server import app

client = TestClient(app)


def test_rerank_missing_required_field():
    # 'query' is required; omit it
    payload = {"documents": [{"id": "a", "text": "hello"}], "top_k": 1}
    resp = client.post("/rerank", json=payload)
    assert resp.status_code == 422  # validation error from pydantic
    body = resp.json()
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert isinstance(body["error"]["detail"], list)


def test_rerank_invalid_json():
    resp = client.post("/rerank", data="{not json", headers={"Content-Type": "application/json"})
    # FastAPI surfaces this as 422 validation
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "VALIDATION_ERROR"


def test_rerank_top_k_exceeds_limit():
    payload = {
        "query": "test",
        "documents": [{"id": "a", "text": "hi"}],
        "top_k": 10_000,
    }
    resp = client.post("/rerank", json=payload)
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "BAD_REQUEST"
    assert "top_k exceeds" in body["error"]["message"].lower()


def test_rerank_document_too_long():
    long_text = "x" * 9000  # exceeds default max_text_length=8192
    payload = {"query": "q", "documents": [{"id": "a", "text": long_text}], "top_k": 1}
    resp = client.post("/rerank", json=payload)
    # The validation of doc length occurs inside handler -> 413 mapped to ENTITY_TOO_LARGE
    assert resp.status_code == 413 or resp.status_code == 422
    body = resp.json()
    # Accept either direct validation or our explicit size check
    assert body["error"]["code"] in {"ENTITY_TOO_LARGE", "VALIDATION_ERROR"}


def test_feedback_unknown_ids_structure():
    payload = {"query": "q", "selected_ids": ["zzz"], "success": 1.0}
    resp = client.post("/feedback", json=payload)
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "NOT_FOUND"
    assert "unknown" in body["error"]["message"].lower()
