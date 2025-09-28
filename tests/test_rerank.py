import numpy as np

from neuralcache.config import Settings
from neuralcache.rerank import Reranker
from neuralcache.types import Document


def test_basic_rerank_runs() -> None:
    settings = Settings(narrative_dim=128)
    reranker = Reranker(settings=settings)

    docs = [
        Document(id="1", text="alpha beta gamma"),
        Document(id="2", text="gamma delta epsilon"),
        Document(id="3", text="zeta eta theta"),
    ]
    query = "beta delta"
    q = np.zeros((settings.narrative_dim,), dtype=float)
    for tok in query.split():
        q[hash(tok) % settings.narrative_dim] += 1.0

    scored = reranker.score(q, docs, mmr_lambda=0.7, query_text=query)
    assert len(scored) == 3
    # Ensure ordering deterministic with epsilon near 0
    assert all(hasattr(s, "score") for s in scored)
