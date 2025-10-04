from __future__ import annotations

import time
from fastapi.testclient import TestClient

from neuralcache.api.server import app, settings, reranker


def test_retention_sweep_manual_purge():
    # Force old timestamps by directly manipulating underlying stores (sqlite path independent)
    # Insert a pheromone and narrative, then set retention to 0 to ensure purge clears immediately.
    # Narrative: simulate update then purge
    reranker.narr.v[:] = 1.0
    reranker.narr._updated_ts = time.time() - 999999  # type: ignore[attr-defined]
    reranker.pher.reinforce(["doc-old"], reward=1.0)
    # Manually age pheromone entry if using sqlite
    if reranker.pher.backend == "sqlite" and reranker._sqlite_state is not None:  # type: ignore[attr-defined]
        conn = reranker._sqlite_state._conn  # type: ignore[attr-defined]
        with conn:  # set ts far in past
            conn.execute("UPDATE pheromones SET ts = ? WHERE doc_id = ?", (time.time() - 999999, "doc-old"))
    elif reranker.pher.backend == "json":
        rec = reranker.pher.data.get("doc-old")  # type: ignore[attr-defined]
        if rec:
            rec["t"] = time.time() - 999999
    # Overwrite settings for test
    settings.storage_retention_days = 0.000001  # ~0.0864 seconds
    # Run purge explicitly via API startup hook logic (call the functions directly)
    retention_seconds = settings.storage_retention_days * 86400.0
    reranker.narr.purge_if_stale(retention_seconds)
    reranker.pher.purge_older_than(retention_seconds)
    # Narrative should be zeroed
    assert reranker.narr.v.sum() == 0.0
    # Pheromone removed
    dump = reranker.pher.bulk_bonus(["doc-old"])  # triggers load; should be zero or absent after purge
    assert dump[0] == 0.0


def test_retention_sweeper_thread_runs():
    # Configure a short sweep interval and a retention window that purges immediately
    settings.storage_retention_days = 0.000001
    settings.storage_retention_sweep_interval_s = 0.1
    # Seed old data
    reranker.narr.v[:] = 1.0
    reranker.narr._updated_ts = time.time() - 999999  # type: ignore[attr-defined]
    reranker.pher.reinforce(["doc-expire"], reward=1.0)
    client = TestClient(app)
    # Trigger startup (launch context manager)
    with client:
        # Allow at least one sweep cycle
        time.sleep(0.25)
        assert reranker.narr.v.sum() == 0.0  # purged
        assert reranker.pher.bulk_bonus(["doc-expire"])[0] == 0.0
