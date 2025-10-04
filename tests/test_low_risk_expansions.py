import os
import time
import tempfile
import json
import numpy as np
from pathlib import Path
from neuralcache.rerank import Reranker
from neuralcache.types import Document
from neuralcache.config import Settings
from neuralcache.narrative import NarrativeTracker
from neuralcache.pheromone import PheromoneStore


def test_narrative_update_skips_below_success_gate(tmp_path):
    narr = NarrativeTracker(dim=4, alpha=0.5, success_gate=0.8, backend="memory")
    original = narr.v.copy()
    emb = np.ones((4,), dtype=np.float32)
    narr.update(emb, success=0.5)  # below gate
    assert np.allclose(narr.v, original), "Narrative should not update when success below gate"


def test_narrative_resize_on_mismatched_dim(tmp_path):
    narr = NarrativeTracker(dim=4, alpha=0.5, success_gate=0.0, backend="memory")
    emb = np.arange(6, dtype=np.float32)
    narr.update(emb, success=1.0)  # triggers resize
    assert narr.v.shape[0] == 6, "Narrative vector should resize to embedding dimension"


def test_pheromone_json_purge(tmp_path):
    store = PheromoneStore(half_life_s=10, backend="json", path="pher.json", storage_dir=str(tmp_path))
    # Manually insert old record
    now = time.time()
    store.data["old"] = {"value": 1.0, "t": now - 10_000, "exposures": 0.0}
    store.data["fresh"] = {"value": 1.0, "t": now, "exposures": 0.0}
    store._save()
    store.purge_older_than(3600)  # 1 hour retention => purge "old"
    assert "old" not in store.data and "fresh" in store.data
    # Ensure file persisted the purge
    with open(Path(tmp_path)/"pher.json", "r", encoding="utf-8") as f:
        persisted = json.load(f)
    assert "old" not in persisted


def test_epsilon_override_env_controls_exploration(monkeypatch):
    settings = Settings()
    settings.deterministic = False
    settings.epsilon_greedy = 0.0  # base = 0
    rr = Reranker(settings=settings)
    docs = [Document(id=str(i), text=f"doc {i}") for i in range(6)]
    q = rr.encode_query("q")
    # Without override, order should be deterministic given epsilon=0
    first_run = [d.id for d in rr.score(q, docs, query_text="q")]
    monkeypatch.setenv("NEURALCACHE_EPSILON", "1.0")  # force pure exploration
    second_run = [d.id for d in rr.score(q, docs, query_text="q")]
    # Highly likely to differ (pure random vs greedy). If it does not, accept but ensure override applied via debug
    if first_run == second_run:
        # fallback assertion: epsilon override recorded in debug
        dbg = {}
        rr.score(q, docs, query_text="q", debug=dbg)
        assert dbg.get("epsilon_used") == 1.0
    else:
        # exploration path hit
        pass


def test_mmr_lambda_default_and_override():
    settings = Settings()
    settings.mmr_lambda_default = 0.7
    rr = Reranker(settings=settings)
    docs = [Document(id="a", text="alpha"), Document(id="b", text="beta")] 
    q = rr.encode_query("alpha")
    dbg1 = {}
    rr.score(q, docs, debug=dbg1, query_text="alpha")
    assert abs(dbg1.get("mmr_lambda_used") - 0.7) < 1e-6
    dbg2 = {}
    rr.score(q, docs, mmr_lambda=0.2, debug=dbg2, query_text="alpha")
    assert abs(dbg2.get("mmr_lambda_used") - 0.2) < 1e-6


def test_gating_overrides_debug_fields():
    settings = Settings()
    settings.gating_mode = "auto"
    rr = Reranker(settings=settings)
    docs = [Document(id=str(i), text=f"text {i}") for i in range(12)]
    q = rr.encode_query("query")
    debug = {}
    overrides = {
        "gating_mode": "on",
        "gating_threshold": 0.99,  # likely triggers gating
        "gating_min_candidates": 2,
        "gating_max_candidates": 4,
        "gating_entropy_temp": 0.5,
    }
    rr.score(q, docs, overrides=overrides, debug=debug, query_text="query")
    gating_dbg = debug.get("gating")
    assert gating_dbg is not None
    assert gating_dbg.get("mode") == "on"
    assert gating_dbg.get("total_candidates") == len(docs) or gating_dbg.get("total_candidates") == len(docs)  # consistency
    assert 0 <= gating_dbg.get("effective_candidate_count") <= gating_dbg.get("total_candidates")
