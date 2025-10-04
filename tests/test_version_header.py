from __future__ import annotations

from fastapi.testclient import TestClient

from neuralcache.api.server import app, settings


def test_version_header_present():
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.headers.get("X-NeuralCache-API-Version") == settings.api_version
    # Alias retained for now
    assert r.headers.get("X-API-Version") == settings.api_version
