import os
import numpy as np
from neuralcache.config import Settings
from neuralcache.rerank import Reranker
from neuralcache.types import Document


def test_rerank_env_epsilon_and_mmr_lambda_override(monkeypatch):
    monkeypatch.setenv("NEURALCACHE_EPSILON", "0.25")
    settings = Settings(deterministic=False)
    r = Reranker(settings=settings)

    docs = [
        Document(id="a", text="alpha beta"),
        Document(id="b", text="beta gamma"),
        Document(id="c", text="gamma delta"),
    ]
    q = r.encode_query("alpha")
    debug = {}
    # Supply explicit mmr_lambda that's out of range to trigger default fallback
    scored = r.score(q, docs, mmr_lambda=1.5, query_text="alpha", debug=debug)
    assert len(scored) == 3
    assert debug["epsilon_used"] == 0.25
    # 1.5 invalid => fallback to settings.mmr_lambda_default (0.5 unless changed)
    assert abs(debug["mmr_lambda_used"] - settings.mmr_lambda_default) < 1e-9


def test_rerank_mmr_lambda_explicit_valid(monkeypatch):
    monkeypatch.delenv("NEURALCACHE_EPSILON", raising=False)
    settings = Settings(deterministic=True)  # deterministic => epsilon suppressed to 0.0
    r = Reranker(settings=settings)
    docs = [Document(id="x", text="one two"), Document(id="y", text="two three")]
    q = r.encode_query("one")
    debug = {}
    scored = r.score(q, docs, mmr_lambda=0.8, query_text="one", debug=debug)
    assert len(scored) == 2
    assert debug["epsilon_used"] == 0.0  # deterministic suppresses exploration
    assert abs(debug["mmr_lambda_used"] - 0.8) < 1e-9
