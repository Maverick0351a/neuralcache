from fastapi.testclient import TestClient
from neuralcache.api.server import app


def test_feedback_unknown_ids():
    client = TestClient(app)
    # Send feedback with ID not seen (no prior rerank) should 404 per server logic
    resp = client.post("/feedback", json={"query":"q","selected_ids":["missing"],"success":1.0})
    assert resp.status_code == 404
    data = resp.json()
    # Structured error envelope: {"error": {"code":..., "message":...}}
    assert "error" in data
    err = data["error"]
    assert err.get("code") == "NOT_FOUND"
