import re
import pytest
from fastapi.testclient import TestClient

from neuralcache.api.server import app, settings
from neuralcache.metrics import metrics_enabled

client = TestClient(app)


def test_metrics_namespace_labeling_toggle():
    if not settings.metrics_enabled:
        pytest.skip("metrics disabled in settings")
    if not metrics_enabled():
        pytest.skip("prometheus_client not installed")

    # First ensure label feature is off by default (unless already enabled in env)
    # We can't modify the already-imported settings instance safely here if user enabled it.
    # If it's already on, we still perform assertions accordingly.
    want_label = settings.metrics_namespace_label

    # Hit two namespaces to generate metrics
    docs = [{"id": "x1", "text": "hello", "embedding": [0.1, 0.2, 0.3]}]
    for ns in ["tenantX", "tenantY"]:
        r = client.post(
            "/rerank",
            json={"query": "q", "documents": docs, "top_k": 1},
            headers={settings.namespace_header: ns},
        )
        assert r.status_code == 200

    m = client.get("/metrics")
    assert m.status_code == 200, m.text
    body = m.text.splitlines()

    # Find rerank request counter lines
    counter_lines = [ln for ln in body if ln.startswith("nc_rerank_requests_total")]
    assert counter_lines, "Expected rerank request counter metrics present"

    # If labeling disabled, none of these lines should contain 'namespace=' label.
    # If enabled, we expect at least tenantX and tenantY labels present.
    if not want_label:
        assert all("namespace=" not in ln for ln in counter_lines), "Namespace label unexpectedly present when disabled"
    else:
        # Build a map of namespaces observed
        seen = set()
        for ln in counter_lines:
            m = re.search(r'namespace="([^"]+)"', ln)
            if m:
                seen.add(m.group(1))
        # If labeling is enabled we expect both namespaces plus possibly default if earlier tests executed
        assert {"tenantX", "tenantY"}.issubset(seen), f"Expected tenantX and tenantY in metrics labels, saw: {seen}"
