import importlib
import types
import os

import numpy as np
import pytest

from neuralcache.encoder import create_encoder, HashingEncoder


@pytest.fixture
def reset_env(monkeypatch):
    # ensure no backend leakage
    for var in ["NEURALCACHE_ENCODER_BACKEND", "NEURALCACHE_ENCODER_MODEL"]:
        if var in os.environ:
            monkeypatch.delenv(var, raising=False)
    yield


def test_hashing_encoder_deterministic(reset_env):
    enc = create_encoder("hash", dim=64)
    v1 = enc.encode("The Quick Brown Fox")
    v2 = enc.encode("the quick brown fox")  # case insensitivity in tokenisation
    assert np.allclose(v1, v2)
    # Different text changes distribution
    v3 = enc.encode("jumps over the lazy dog")
    assert not np.allclose(v1, v3)


def test_hashing_encoder_empty_batch(reset_env):
    enc = create_encoder("hash", dim=32)
    batch = enc.encode_batch([])  # type: ignore[arg-type]
    assert batch.shape == (0, 32)


def test_openai_backend_missing_dependency_falls_back(monkeypatch, reset_env, caplog):
    caplog.set_level("WARNING")
    # Simulate ImportError when trying to import openai
    real_import = importlib.import_module

    def fake_import(name, *a, **k):  # pragma: no cover - executed inside test
        if name == "openai":
            raise ImportError("mock missing openai")
        return real_import(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    enc = create_encoder("openai", dim=16, model="test-model")
    # Should fall back to hashing encoder
    assert isinstance(enc, HashingEncoder)
    assert any("openai backend requested" in r.message for r in caplog.records)


def test_sentence_transformer_backend_missing_dependency_falls_back(monkeypatch, reset_env, caplog):
    caplog.set_level("WARNING")
    real_import = importlib.import_module

    def fake_import(name, *a, **k):  # pragma: no cover - executed inside test
        if name == "sentence_transformers":
            raise ImportError("mock missing st")
        return real_import(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    enc = create_encoder("sentence-transformer", dim=24, model="all-MiniLM-L6-v2")
    assert isinstance(enc, HashingEncoder)
    assert any("sentence-transformer backend requested" in r.message for r in caplog.records)


def test_unknown_backend_warning(reset_env, caplog):
    caplog.set_level("WARNING")
    enc = create_encoder("nonexistent-backend", dim=8)
    assert isinstance(enc, HashingEncoder)
    assert any("Unknown embedding backend" in r.message for r in caplog.records)
