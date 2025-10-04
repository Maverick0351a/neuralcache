import time
from neuralcache.pheromone import PheromoneStore


def test_pheromone_reinforce_and_decay():
    store = PheromoneStore(half_life_s=1.0, exposure_penalty=0.0, backend="memory")
    store.reinforce(["a"], reward=1.0)
    v_initial = store.get_bonus("a")
    assert v_initial > 0.0
    # Instead of sleeping, directly modify the timestamp to simulate elapsed time
    rec = store.data["a"]
    past = rec["t"] - 1.1  # pretend last update was 1.1s earlier
    rec["t"] = past
    store.data["a"] = rec
    v_later = store.get_bonus("a")
    assert v_later < v_initial  # decayed


def test_pheromone_exposure_penalty():
    store = PheromoneStore(half_life_s=1000.0, exposure_penalty=0.5, backend="memory")
    store.reinforce(["b"], reward=1.0)
    v0 = store.get_bonus("b")
    store.record_exposure(["b"])  # one exposure halves remaining multiplier (1 - 0.5*1)
    v1 = store.get_bonus("b")
    assert v1 <= v0
