import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest

# Import the core SQLite implementation
try:  # pragma: no cover - import guard
    from neuralcache.storage.sqlite_state import SQLiteState  # type: ignore
except Exception as exc:  # pragma: no cover
    pytest.skip(f"SQLiteState not available: {exc}")


def test_sqlite_narrative_persistence(tmp_path: Path):
    db_path = tmp_path / "state.db"
    st = SQLiteState(path=db_path)
    vec = np.random.RandomState(0).randn(8).astype(float)
    st.save_narrative(vec)
    # Reload via new instance
    st2 = SQLiteState(path=db_path)
    loaded = st2.load_narrative()
    assert loaded is not None
    assert np.allclose(vec, loaded)


def test_sqlite_pheromone_upsert_and_exposures(tmp_path: Path):
    st = SQLiteState(path=tmp_path / "state.db")
    st.upsert_pheromone("action:A", 1.0, add_exposure=1)
    st.upsert_pheromone("action:A", 2.0, add_exposure=2)  # exposures accumulate
    rows = st.get_pheromones(["action:A"])  # value should be last write (2.0) and exposures 3
    rec = rows["action:A"]
    value = rec["value"]
    exposures = rec["exposures"]
    assert value == pytest.approx(2.0)
    assert exposures == pytest.approx(3.0)


def test_sqlite_evaporate_decay(tmp_path: Path):
    st = SQLiteState(path=tmp_path / "state.db")
    # Insert two pheromones with an old timestamp so decay is significant
    past = time.time() - 10.0
    st.upsert_pheromone("a", 10.0, timestamp=past, add_exposure=1)
    st.upsert_pheromone("b", 20.0, timestamp=past, add_exposure=2)
    before = st.get_pheromones(["a", "b"])
    # Apply decay with half-life = 1s at current time
    st.evaporate(half_life_s=1.0, now=time.time())
    after = st.get_pheromones(["a", "b"])
    for k in ["a", "b"]:
        assert after[k]["value"] < before[k]["value"]
        # exposures untouched by evaporate
        assert after[k]["exposures"] == before[k]["exposures"]


def test_sqlite_narrative_none_when_empty(tmp_path: Path):
    st = SQLiteState(path=tmp_path / "state.db")
    assert st.load_narrative() is None


def test_sqlite_get_pheromones_empty_input(tmp_path: Path):
    st = SQLiteState(path=tmp_path / "state.db")
    assert st.get_pheromones([]) == {}
