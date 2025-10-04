from fastapi.testclient import TestClient

from neuralcache.api.server import app, settings, _rerankers, settings as global_settings

client = TestClient(app)


def _ping(ns: str):
    r = client.post(
        "/rerank",
        json={"query": "q", "documents": []},
        headers={settings.namespace_header: ns},
    )
    assert r.status_code == 200


def test_lru_eviction():
    # Skip if already constrained externally
    # Force max namespaces to 2 (including default) for this test using monkeypatch-like direct attr
    global_settings.max_namespaces = 2

    default_ns = global_settings.default_namespace

    # Touch two tenant namespaces beyond default; with max=2 total, only one non-default can coexist.
    # Sequence: create tenant1 (default + tenant1). Then create tenant2 -> should evict tenant1.
    _ping("tenant1")
    assert default_ns in _rerankers
    assert "tenant1" in _rerankers

    _ping("tenant2")
    # After creating tenant2 with max=2, tenant1 should be evicted (LRU) keeping default + tenant2
    assert default_ns in _rerankers
    assert "tenant2" in _rerankers
    assert "tenant1" not in _rerankers

    # Access default then create tenant3; tenant2 should be LRU and evicted
    _ping(default_ns)  # updates recency for default (though it's never evicted)
    _ping("tenant3")
    assert default_ns in _rerankers
    assert "tenant3" in _rerankers
    assert "tenant2" not in _rerankers
