import numpy as np
from neuralcache.similarity import safe_normalize, cosine_sim, batched_cosine_sims


def test_safe_normalize_zero_vector():
    z = np.zeros((3,), dtype=np.float32)
    normed = safe_normalize(z)
    # All zeros divided by eps => still zeros
    assert np.allclose(normed, 0.0)


def test_safe_normalize_mixed_rows():
    m = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0]], dtype=np.float32)
    normed = safe_normalize(m)
    assert np.allclose(normed[1], [1.0,0.0,0.0])


def test_cosine_and_batched():
    a = np.array([1.0,0.0,0.0], dtype=np.float32)
    b = np.array([0.0,1.0,0.0], dtype=np.float32)
    c = np.array([1.0,1.0,0.0], dtype=np.float32)
    sim_ab = cosine_sim(a,b)
    assert abs(sim_ab) < 1e-6
    sims = batched_cosine_sims(a, np.stack([a,b,c]))
    assert sims.shape == (3,)
    # Highest should be self-similarity
    assert sims[0] >= sims[2] >= sims[1]
