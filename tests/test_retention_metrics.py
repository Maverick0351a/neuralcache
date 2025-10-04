import os
from fastapi.testclient import TestClient
from neuralcache.api.server import app, settings


def test_retention_metrics_disabled(monkeypatch):
    # Ensure disabled when retention_days unset
    monkeypatch.setattr(settings, "storage_retention_days", None)
    client = TestClient(app)
    resp = client.get("/metrics/retention")
    assert resp.status_code == 200
    data = resp.json()
    assert data["retention_enabled"] is False


def test_retention_metrics_enabled(monkeypatch):
    monkeypatch.setattr(settings, "storage_retention_days", 0.00001)  # ~immediate expiry window
    monkeypatch.setattr(settings, "storage_retention_sweep_on_start", True)
    client = TestClient(app)
    resp = client.get("/metrics/retention")
    assert resp.status_code == 200
    data = resp.json()
    assert data["retention_enabled"] is True
    assert "retention_days" in data
    # Fields may be None initially but must exist
    assert "last_startup_purge_ts" in data
    assert "sweep_count" in data
