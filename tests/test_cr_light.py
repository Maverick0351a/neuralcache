from __future__ import annotations

from neuralcache.cr.index import build_cr_index
from neuralcache.cr.search import hierarchical_candidates
from neuralcache.embedding import encode_texts


def test_cr_candidate_selection_basic() -> None:
    docs = [f"alpha {i}" for i in range(60)] + [f"beta {i}" for i in range(60)]
    embeddings = encode_texts(docs, dim=128)
    index = build_cr_index(embeddings, d1=64, d2=16, k2=4, k1_per_bucket=4)

    q_alpha = encode_texts(["alpha query"], dim=128)[0]
    cand_alpha = hierarchical_candidates(
        q_alpha,
        embeddings,
        index,
        top_coarse=2,
        top_topics_per_coarse=2,
        max_candidates=64,
    )
    alpha_ratio = sum(doc_id < 60 for doc_id in cand_alpha) / max(1, len(cand_alpha))
    assert alpha_ratio > 0.55

    q_beta = encode_texts(["beta query"], dim=128)[0]
    cand_beta = hierarchical_candidates(
        q_beta,
        embeddings,
        index,
        top_coarse=2,
        top_topics_per_coarse=2,
        max_candidates=64,
    )
    beta_ratio = sum(doc_id >= 60 for doc_id in cand_beta) / max(1, len(cand_beta))
    assert beta_ratio > 0.55
