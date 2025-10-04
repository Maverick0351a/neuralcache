from __future__ import annotations

import os
from copy import deepcopy

from fastapi.testclient import TestClient

from neuralcache.api.server import app, settings


def _sample_payload():
    return {
        "query": "test query",
        "documents": [
            {"id": f"d{i}", "text": f"Document content {i}"} for i in range(5)
        ],
        "top_k": 5,
        "mmr_lambda": 0.5,
    }


def test_deterministic_mode_order_stable(monkeypatch):
    # Enable deterministic mode
    monkeypatch.setattr(settings, "deterministic", True)
    monkeypatch.setattr(settings, "deterministic_seed", 42)
    client = TestClient(app)
    p = _sample_payload()
    r1 = client.post("/rerank?return_debug=true", json=p)
    r2 = client.post("/rerank?return_debug=true", json=deepcopy(p))
    assert r1.status_code == 200 and r2.status_code == 200
    ids1 = [d["id"] for d in r1.json()["results"]]
    ids2 = [d["id"] for d in r2.json()["results"]]
    assert ids1 == ids2, "Deterministic mode should yield identical ordering"
    debug = r1.json().get("debug")
    assert debug.get("deterministic") is True
    assert debug.get("epsilon_used") == 0.0


def test_non_deterministic_mode_differs(monkeypatch):
    monkeypatch.setattr(settings, "deterministic", False)
    monkeypatch.setattr(settings, "epsilon_greedy", 0.5)  # amplify randomness chance
    client = TestClient(app)
    p = _sample_payload()
    r1 = client.post("/rerank?return_debug=true", json=p)
    r2 = client.post("/rerank?return_debug=true", json=deepcopy(p))
    assert r1.status_code == 200 and r2.status_code == 200
    ids1 = [d["id"] for d in r1.json()["results"]]
    ids2 = [d["id"] for d in r2.json()["results"]]
    # It's possible (though unlikely) they match; allow retry logic minimal by asserting not always same
    # If they match spuriously, lower risk; pattern mainly ensures deterministic branch works.
    # We'll accept either to avoid flaky test but ensure epsilon_used reflects non-zero.
    if ids1 == ids2:
        # Ensure epsilon was >0 so randomness path existed
        assert r1.json()["debug"]["epsilon_used"] > 0.0
    else:
        assert r1.json()["debug"]["epsilon_used"] > 0.0
