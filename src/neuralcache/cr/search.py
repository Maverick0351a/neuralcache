from __future__ import annotations

import numpy as np

from neuralcache.cr.index import CRIndex
from neuralcache.cr.utils import pca_transform


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + eps)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + eps)
    return a_norm @ b_norm.T


def hierarchical_candidates(
    q0_query: np.ndarray,
    doc_embeddings_q0: np.ndarray,
    cr: CRIndex,
    top_coarse: int = 3,
    top_topics_per_coarse: int = 2,
    max_candidates: int = 256,
) -> list[int]:
    q1 = pca_transform(q0_query[None, :], cr.proj1_components, cr.proj1_mean)
    q2 = pca_transform(q1, cr.proj2_components, cr.proj2_mean)

    s2 = _cosine(q2, cr.coarse_centroids).ravel()
    coarse_boost = np.full_like(s2, fill_value=-1.0)
    for idx, bucket in enumerate(cr.coarse_buckets):
        if not bucket:
            continue
        mean_vec = doc_embeddings_q0[np.asarray(bucket)].mean(axis=0, keepdims=True)
        coarse_boost[idx] = float(_cosine(q0_query[None, :], mean_vec).ravel()[0])
    coarse_scores = 0.5 * s2 + 0.5 * coarse_boost
    coarse_ids = np.argsort(-coarse_scores)[:top_coarse]

    candidate_ids: list[int] = []
    topics_total = max(1, top_coarse * top_topics_per_coarse)
    docs_per_topic_cap = max(1, max_candidates // topics_total)
    for cid in coarse_ids:
        topic_centroids = cr.topic_centroids_per_coarse[cid]
        if topic_centroids.shape[0] == 0:
            continue
        s1 = _cosine(q1, topic_centroids).ravel()
        topic_boost = np.full_like(s1, fill_value=-1.0)
        for tid, bucket in enumerate(cr.topic_buckets_per_coarse[cid]):
            if not bucket:
                continue
            mean_vec = doc_embeddings_q0[np.asarray(bucket)].mean(axis=0, keepdims=True)
            topic_boost[tid] = float(_cosine(q0_query[None, :], mean_vec).ravel()[0])
        topic_scores = 0.5 * s1 + 0.5 * topic_boost
        topic_ids = np.argsort(-topic_scores)[:top_topics_per_coarse]
        for tid in topic_ids:
            doc_bucket = cr.topic_buckets_per_coarse[cid][tid]
            if not doc_bucket:
                continue
            doc_indices = np.asarray(doc_bucket)
            topic_doc_scores = _cosine(q0_query[None, :], doc_embeddings_q0[doc_indices]).ravel()
            keep = np.argsort(-topic_doc_scores)[
                : min(docs_per_topic_cap, doc_indices.size)
            ]
            candidate_ids.extend(doc_indices[keep].tolist())

    candidate_ids = list(dict.fromkeys(candidate_ids))

    if len(candidate_ids) > max_candidates:
        candidate_vectors = doc_embeddings_q0[np.array(candidate_ids)]
        s0 = _cosine(q0_query[None, :], candidate_vectors).ravel()
        order = np.argsort(-s0)[:max_candidates]
        candidate_ids = [candidate_ids[i] for i in order.tolist()]

    return candidate_ids
