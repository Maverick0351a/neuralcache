from __future__ import annotations

import os
from copy import deepcopy
from fastapi.testclient import TestClient

from neuralcache.api.server import app, settings


def test_epsilon_env_override(monkeypatch):
    monkeypatch.setenv("NEURALCACHE_EPSILON", "0.33")
    monkeypatch.setattr(settings, "deterministic", False)
    client = TestClient(app)
    payload = {
        "query": "epsilon test",
        "documents": [{"id": str(i), "text": f"Doc {i}"} for i in range(6)],
        "top_k": 5,
    }
    r = client.post("/rerank?return_debug=true", json=payload)
    assert r.status_code == 200
    debug = r.json()["debug"]
    assert abs(debug["epsilon_used"] - 0.33) < 1e-6


def test_epsilon_ignored_when_deterministic(monkeypatch):
    monkeypatch.setenv("NEURALCACHE_EPSILON", "0.9")
    monkeypatch.setattr(settings, "deterministic", True)
    client = TestClient(app)
    payload = {
        "query": "epsilon test",
        "documents": [{"id": str(i), "text": f"Doc {i}"} for i in range(4)],
        "top_k": 4,
    }
    r = client.post("/rerank?return_debug=true", json=payload)
    assert r.status_code == 200
    debug = r.json()["debug"]
    assert debug["deterministic"] is True
    assert debug["epsilon_used"] == 0.0
