import json
from fastapi.testclient import TestClient

from neuralcache.api.server import app

client = TestClient(app)


def test_rerank_schema_basic():
    payload = {
        "query": "What is stigmergy?",
        "documents": [
            {"id": "a", "text": "Stigmergy is indirect coordination."},
            {"id": "b", "text": "Vector DBs store embeddings."},
        ],
        "top_k": 2,
    }
    r = client.post("/rerank", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "results" in body
    assert isinstance(body["results"], list)
    assert len(body["results"]) <= 2
    assert "debug" in body
    # Ensure scored document structure
    first = body["results"][0]
    assert {"id", "text", "score"}.issubset(first.keys())


def test_error_envelope_top_k_exceeds():
    payload = {
        "query": "X",
        "documents": [{"id": "a", "text": "x"}],
        "top_k": 999999,
    }
    r = client.post("/rerank", json=payload)
    assert r.status_code == 400
    body = r.json()
    assert body.get("error", {}).get("code") == "BAD_REQUEST"
    assert "top_k" in body.get("error", {}).get("message", "")


def test_batch_schema():
    payload = [
        {
            "query": "What is stigmergy?",
            "documents": [
                {"id": "a", "text": "Stigmergy is indirect coordination."},
                {"id": "b", "text": "Vector DBs store embeddings."},
            ],
            "top_k": 2,
        }
    ]
    r = client.post("/rerank/batch", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert "results" in body[0]
    assert "debug" in body[0]


def test_feedback_not_found():
    # Selected id not present in feedback cache should yield NOT_FOUND
    payload = {"query": "q", "selected_ids": ["missing"], "success": 1.0}
    r = client.post("/feedback", json=payload)
    assert r.status_code in (400, 404)  # 400 if selected_ids empty; 404 if unknown
    body = r.json()
    assert "error" in body
    assert body["error"]["code"] in {"BAD_REQUEST", "NOT_FOUND"}
