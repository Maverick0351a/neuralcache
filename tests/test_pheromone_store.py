import time
from neuralcache.pheromone import PheromoneStore


def test_pheromone_reinforce_and_decay():
    store = PheromoneStore(half_life_s=1.0, exposure_penalty=0.0, backend="memory")
    store.reinforce(["a"], reward=1.0)
    v_initial = store.get_bonus("a")
    assert v_initial > 0.0
    time.sleep(1.1)
    v_later = store.get_bonus("a")
    assert v_later < v_initial  # decayed


def test_pheromone_exposure_penalty():
    store = PheromoneStore(half_life_s=1000.0, exposure_penalty=0.5, backend="memory")
    store.reinforce(["b"], reward=1.0)
    v0 = store.get_bonus("b")
    store.record_exposure(["b"])  # one exposure halves remaining multiplier (1 - 0.5*1)
    v1 = store.get_bonus("b")
    assert v1 <= v0
