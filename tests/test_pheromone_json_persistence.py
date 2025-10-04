import json
import tempfile
from pathlib import Path
from neuralcache.pheromone import PheromoneStore


def test_pheromone_json_persistence_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        store = PheromoneStore(half_life_s=1000.0, backend="json", storage_dir=td, path="pher.json")
        store.reinforce(["x"], reward=2.0)
        first = store.get_bonus("x")
        assert first > 0
        # Create new instance pointing at same file
        store2 = PheromoneStore(half_life_s=1000.0, backend="json", storage_dir=td, path="pher.json")
        second = store2.get_bonus("x")
    # Allow small floating differences due to decay math / timestamp differences
    assert abs(first - second) < 5e-5
