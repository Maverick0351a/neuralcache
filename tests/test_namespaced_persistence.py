import os
from pathlib import Path
from fastapi.testclient import TestClient

from neuralcache.api.server import app, settings, _rerankers, settings as global_settings

client = TestClient(app)


def test_namespaced_persistence_file_creation(tmp_path):
    # Enable namespaced persistence and point storage_dir to tmp_path for isolation
    global_settings.namespaced_persistence = True
    global_settings.storage_dir = str(tmp_path)

    # Narrative/pheromone templates already defined; trigger two namespaces
    docs = [{"id": "a", "text": "alpha", "embedding": [0.1, 0.2, 0.3]}]
    for ns in ["persistA", "persistB"]:
        r = client.post(
            "/rerank",
            json={"query": "q", "documents": docs, "top_k": 1},
            headers={settings.namespace_header: ns},
        )
        assert r.status_code == 200
        # Provide feedback to trigger narrative update (persistence happens on update)
        fb = {
            "query": "q",
            "selected_ids": ["a"],
            "success": 1.0,
        }
        fr = client.post(
            "/feedback",
            json=fb,
            headers={settings.namespace_header: ns},
        )
        assert fr.status_code == 200

    # Files should exist under storage_dir for each namespace
    for ns in ["persistA", "persistB"]:
        narr = Path(global_settings.storage_dir) / global_settings.narrative_store_template.format(namespace=ns)
        pher = Path(global_settings.storage_dir) / global_settings.pheromone_store_template.format(namespace=ns)
        assert narr.exists(), f"Expected narrative store for {ns} at {narr}"
        assert pher.exists(), f"Expected pheromone store for {ns} at {pher}"
