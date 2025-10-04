import numpy as np
from neuralcache.config import Settings
from neuralcache.rerank import Reranker
from neuralcache.types import Document


def _make_docs(n=12):
    return [Document(id=f"d{i}", text=f"token {i}") for i in range(n)]


def test_gating_off_mode():
    s = Settings(gating_mode="off")
    r = Reranker(settings=s)
    q = r.encode_query("query")
    debug = {}
    docs = _make_docs(10)
    scored = r.score(q, docs, query_text="query", debug=debug)
    assert len(scored) == 10
    assert debug["gating"]["use_gating"] is False
    assert debug["gating"]["effective_candidate_count"] == 10


def test_gating_on_mode():
    s = Settings(gating_mode="on", gating_min_candidates=3, gating_max_candidates=5)
    r = Reranker(settings=s)
    q = r.encode_query("query")
    debug = {}
    docs = _make_docs(12)
    scored = r.score(q, docs, query_text="query", debug=debug)
    assert debug["gating"]["use_gating"] is True
    assert 3 <= debug["gating"]["effective_candidate_count"] <= 5
    # All returned docs should be subset of originals
    ids = {d.id for d in scored}
    assert ids.issubset({d.id for d in docs})


def test_gating_auto_threshold_behavior():
    # Force threshold very low so gating triggers
    s = Settings(gating_mode="auto", gating_threshold=0.0, gating_min_candidates=2, gating_max_candidates=4)
    r = Reranker(settings=s)
    q = r.encode_query("query")
    debug = {}
    docs = _make_docs(8)
    r.score(q, docs, query_text="query", debug=debug)
    assert debug["gating"]["use_gating"] is True
    assert 2 <= debug["gating"]["effective_candidate_count"] <= 4
