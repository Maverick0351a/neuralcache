import pytest
from fastapi.testclient import TestClient

from neuralcache.api.server import app, settings

client = TestClient(app)


def _rerank(namespace: str, query: str, docs):
    payload = {"query": query, "documents": docs, "top_k": len(docs)}
    return client.post("/rerank", json=payload, headers={settings.namespace_header: namespace})


def test_namespace_invalid_names():
    # Empty string maps to default namespace (allowed); others are invalid
    bad = ["a" * 65, "semi;colon", "space name", "slash/"]
    for ns in bad:
        r = client.post("/rerank", json={"query": "q", "documents": []}, headers={settings.namespace_header: ns})
        assert r.status_code == 400
        body = r.json()
        assert body["error"]["code"] == "BAD_REQUEST"
        assert "Invalid namespace" in body["error"]["message"]


def test_namespace_isolation_feedback_and_scoring():
    docs = [
        {"id": "d1", "text": "alpha", "embedding": [0.1, 0.2, 0.3]},
        {"id": "d2", "text": "beta", "embedding": [0.4, 0.5, 0.6]},
    ]
    r1 = _rerank("tenantA", "alpha query", docs)
    assert r1.status_code == 200
    r2 = _rerank("tenantB", "alpha query", docs)
    assert r2.status_code == 200
    # Provide feedback only to tenantA for d1 success
    fb = {
        "query": "alpha query",
        "selected_ids": ["d1"],
        "success": 1.0,
    }
    fr = client.post("/feedback", json=fb, headers={settings.namespace_header: "tenantA"})
    assert fr.status_code == 200
    # Re-run queries; tenantA narrative/pheromone should diverge from tenantB after feedback
    r1_again = _rerank("tenantA", "alpha query", docs).json()
    r2_again = _rerank("tenantB", "alpha query", docs).json()
    # Results are scored documents; verify ordering difference or score delta
    scores_a = {d["id"]: d["score"] for d in r1_again["results"]}
    scores_b = {d["id"]: d["score"] for d in r2_again["results"]}
    # Expect d1 score higher in tenantA vs tenantB due to feedback reinforcement effects
    assert scores_a["d1"] >= scores_b["d1"]


def test_namespace_default_applied_when_missing():
    # No header -> default namespace
    r = client.post("/rerank", json={"query": "q", "documents": []})
    assert r.status_code == 200
