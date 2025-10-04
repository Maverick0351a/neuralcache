import os
import time
import json
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from neuralcache.api.server import app, reranker, settings
from neuralcache.narrative import NarrativeTracker
from neuralcache.types import Document, RerankRequest
from neuralcache.rerank import Reranker
from neuralcache.config import Settings


def test_narrative_purge_if_stale_json(tmp_path):
    """Purge uses internal _updated_ts, so set that stale and confirm JSON file removal."""
    narr = NarrativeTracker(dim=4, alpha=0.5, success_gate=0.0, backend="json", path="n.json", storage_dir=str(tmp_path))
    emb = np.ones((4,), dtype=np.float32)
    narr.update(emb, success=1.0)
    file_path = Path(tmp_path)/"n.json"
    assert file_path.exists()
    # Make internal timestamp stale
    narr._updated_ts = time.time() - 10_000  # type: ignore[attr-defined]
    narr.purge_if_stale(retention_seconds=3600)
    assert not file_path.exists(), "Narrative JSON should be deleted when stale"
    assert np.allclose(narr.v, np.zeros_like(narr.v))


def test_cr_empty_candidates_fallback(monkeypatch):
    # Force CR on but monkeypatch hierarchical_candidates to return empty
    s = Settings()
    s.cr.on = True
    rr = Reranker(settings=s)
    docs = [Document(id=str(i), text=f"text {i}") for i in range(5)]
    q = rr.encode_query("query")
    import neuralcache.cr.search as crsearch
    monkeypatch.setattr(crsearch, "hierarchical_candidates", lambda **kwargs: [])
    scored = rr.score(q, docs, query_text="query")
    ids = {d.id for d in scored}
    assert ids == {d.id for d in docs}, "Fallback should revert to all documents when CR yields none"


def test_encoder_unknown_backend_warning(caplog):
    caplog.set_level("WARNING")
    from neuralcache.encoder import create_encoder
    enc = create_encoder("unknown-backend-xyz", dim=8)
    assert enc.encode("hello").shape[0] == 8
    assert any("Unknown embedding backend" in rec.message for rec in caplog.records)


def test_rate_limit_enforced(monkeypatch):
    client = TestClient(app)
    # Patch settings directly for test isolation
    orig = settings.rate_limit_per_minute
    settings.rate_limit_per_minute = 2
    try:
        payload = {"query":"q", "documents":[{"id":"a","text":"A"},{"id":"b","text":"B"}], "top_k":1}
        # First two should pass
        for _ in range(2):
            r = client.post("/rerank", json=payload)
            assert r.status_code == 200
        # Third should rate limit
        r3 = client.post("/rerank", json=payload)
        assert r3.status_code == 429
        body = r3.json()
        assert body["error"]["code"] == "RATE_LIMITED"
    finally:
        settings.rate_limit_per_minute = orig


def test_api_key_authentication(monkeypatch):
    client = TestClient(app)
    # Inject an API token
    orig_tokens = list(settings.api_tokens)
    settings.api_tokens = ["secret123"]
    try:
        payload = {"query":"q", "documents":[{"id":"a","text":"A"}], "top_k":1}
        # Missing token -> 401
        r = client.post("/rerank", json=payload)
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "UNAUTHORIZED"
        # Provide token via x-api-key
        r2 = client.post("/rerank", json=payload, headers={"x-api-key":"secret123"})
        assert r2.status_code == 200
    finally:
        settings.api_tokens = orig_tokens


def test_batch_rerank_gating_debug():
    client = TestClient(app)
    # Use small document lists for clarity
    batch = [
        {"query":"q1","documents":[{"id":"a","text":"A"},{"id":"b","text":"B"}],"top_k":1},
        {"query":"q2","documents":[{"id":"c","text":"C"},{"id":"d","text":"D"}],"top_k":1}
    ]
    # Override gating to 'on' via query param use_cr left default
    resp = client.post("/rerank/batch?use_cr=false", json=batch)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list) and len(data) == 2
    # Ensure debug present and has gating keys
    for item in data:
        debug = item.get("debug")
        assert debug is not None
        gating_dbg = debug.get("gating")
        assert gating_dbg is not None
        assert set(["mode","uncertainty","use_gating","candidate_count","effective_candidate_count","total_candidates"]).issubset(gating_dbg.keys())
