import numpy as np
from neuralcache.config import Settings
from neuralcache.rerank import Reranker
from neuralcache.types import Document


def test_update_feedback_with_doc_map():
    settings = Settings(deterministic=True)
    r = Reranker(settings=settings)
    docs = [Document(id="a", text="alpha"), Document(id="b", text="beta")]
    q = r.encode_query("alpha")
    scored = r.score(q, docs, query_text="alpha")
    doc_map = {d.id: d for d in docs}
    old_vector = r.narr.v.copy()
    r.update_feedback([scored[0].id], doc_map, success=1.0)
    assert not (r.narr.v == old_vector).all()


def test_update_feedback_with_best_doc_text_only():
    settings = Settings(deterministic=True)
    r = Reranker(settings=settings)
    old_vector = r.narr.v.copy()
    r.update_feedback([], doc_map=None, success=1.0, best_doc_text="some helpful answer")
    assert not (r.narr.v == old_vector).all()


def test_update_feedback_ignores_when_no_signal():
    settings = Settings(deterministic=True)
    r = Reranker(settings=settings)
    old_vector = r.narr.v.copy()
    r.update_feedback([], doc_map=None, success=1.0)
    assert (r.narr.v == old_vector).all()
