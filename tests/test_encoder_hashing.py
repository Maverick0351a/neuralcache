import numpy as np
from neuralcache.encoder import create_encoder, _hash_to_vector  # type: ignore


def test_hashing_encoder_deterministic():
    enc = create_encoder("hash", dim=32, model=None)
    v1 = enc.encode("Hello World")
    v2 = enc.encode("Hello World")
    assert np.allclose(v1, v2)
    assert v1.shape == (32,)


def test_hashing_encoder_batch_empty():
    enc = create_encoder("hash", dim=16, model=None)
    batch = enc.encode_batch([])
    assert batch.shape == (0, 16)


def test_hash_to_vector_non_empty():
    vec = _hash_to_vector("", 8)
    # empty string should still produce a non-all-zero vector (fallback sets index 0)
    assert vec.shape == (8,)
    assert vec[0] != 0.0
