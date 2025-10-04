import numpy as np
from neuralcache.config import Settings
from neuralcache.rerank import Reranker
from neuralcache.types import Document
from types import SimpleNamespace

class DummyCR:
    def __init__(self, dim0, buckets):
        self.meta = SimpleNamespace(d0=dim0)
        self.buckets = buckets


def test_cr_candidate_path(monkeypatch):
    settings = Settings()
    settings.cr.on = True
    settings.cr.top_coarse = 1
    settings.cr.top_topics_per_coarse = 1
    settings.cr.max_candidates = 2
    r = Reranker(settings=settings)

    # Monkeypatch load to return None index but force _ensure_cr_loaded path to None then fallback
    # Instead patch hierarchical_candidates indirectly by returning empty so fallback triggers.
    monkeypatch.setattr("neuralcache.rerank.hierarchical_candidates", lambda **kwargs: [])

    docs = [Document(id=f"d{i}", text=f"text {i}") for i in range(5)]
    q = r.encode_query("text")
    debug = {}
    scored = r.score(q, docs, query_text="text", debug=debug)
    # Should fallback to full docs since hierarchical returned empty
    assert len(scored) == 5
