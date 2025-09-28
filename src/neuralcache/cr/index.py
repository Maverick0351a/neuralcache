from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass

import numpy as np

from neuralcache.cr.utils import kmeans_lloyd, pca_fit, pca_transform


@dataclass
class CRIndexMeta:
    d0: int
    d1: int
    d2: int
    k1: int
    k2: int
    doc_count: int


@dataclass
class CRIndex:
    meta: CRIndexMeta
    proj1_components: np.ndarray
    proj1_mean: np.ndarray
    proj2_components: np.ndarray
    proj2_mean: np.ndarray
    coarse_centroids: np.ndarray
    coarse_buckets: list[list[int]]
    topic_centroids_per_coarse: list[np.ndarray]
    topic_buckets_per_coarse: list[list[list[int]]]


def build_cr_index(
    embeddings_q0: np.ndarray,
    d1: int = 256,
    d2: int = 64,
    k2: int = 16,
    k1_per_bucket: int = 12,
    seed: int = 42,
) -> CRIndex:
    doc_count, dim0 = embeddings_q0.shape
    proj1_components, proj1_mean = pca_fit(embeddings_q0, out_dim=min(d1, dim0))
    q1 = pca_transform(embeddings_q0, proj1_components, proj1_mean)
    dim1_eff = q1.shape[1]

    proj2_components, proj2_mean = pca_fit(q1, out_dim=min(d2, dim1_eff))
    q2 = pca_transform(q1, proj2_components, proj2_mean)
    dim2_eff = q2.shape[1]

    k2_eff = min(k2, max(2, int(np.sqrt(doc_count))))
    km_level2 = kmeans_lloyd(q2, k=k2_eff, iters=30, seed=seed)
    coarse_buckets = [
        np.where(km_level2.labels == coarse_idx)[0].tolist()
        for coarse_idx in range(km_level2.centroids.shape[0])
    ]

    topic_centroids: list[np.ndarray] = []
    topic_buckets: list[list[list[int]]] = []
    rng = np.random.default_rng(seed)
    for doc_bucket in coarse_buckets:
        if len(doc_bucket) == 0:
            topic_centroids.append(np.zeros((0, dim1_eff), dtype=np.float32))
            topic_buckets.append([])
            continue
        q1_bucket = q1[doc_bucket]
        k1_bucket = min(k1_per_bucket, max(2, int(np.sqrt(len(doc_bucket)))))
        sub_seed = int(rng.integers(0, 1_000_000))
        km_level1 = kmeans_lloyd(q1_bucket, k=k1_bucket, iters=25, seed=sub_seed)
        topic_centroids.append(km_level1.centroids)
        sub_buckets: list[list[int]] = []
        for topic_idx in range(km_level1.centroids.shape[0]):
            mask = km_level1.labels == topic_idx
            sub_buckets.append([int(doc_bucket[i]) for i in np.where(mask)[0]])
        topic_buckets.append(sub_buckets)

    meta = CRIndexMeta(
        d0=dim0,
        d1=dim1_eff,
        d2=dim2_eff,
        k1=k1_per_bucket,
        k2=km_level2.centroids.shape[0],
        doc_count=doc_count,
    )
    return CRIndex(
        meta=meta,
        proj1_components=proj1_components.astype(np.float32),
        proj1_mean=proj1_mean.astype(np.float32),
        proj2_components=proj2_components.astype(np.float32),
        proj2_mean=proj2_mean.astype(np.float32),
        coarse_centroids=km_level2.centroids.astype(np.float32),
        coarse_buckets=coarse_buckets,
        topic_centroids_per_coarse=[c.astype(np.float32) for c in topic_centroids],
        topic_buckets_per_coarse=topic_buckets,
    )


def save_cr_index(idx: CRIndex, path_npz: str, path_meta_json: str) -> None:
    np.savez_compressed(
        path_npz,
        proj1_components=idx.proj1_components,
        proj1_mean=idx.proj1_mean,
        proj2_components=idx.proj2_components,
        proj2_mean=idx.proj2_mean,
        coarse_centroids=idx.coarse_centroids,
        coarse_buckets=np.array(idx.coarse_buckets, dtype=object),
        topic_centroids_per_coarse=np.array(idx.topic_centroids_per_coarse, dtype=object),
        topic_buckets_per_coarse=np.array(idx.topic_buckets_per_coarse, dtype=object),
    )
    with pathlib.Path(path_meta_json).open("w", encoding="utf-8") as f:
        json.dump(asdict(idx.meta), f, indent=2)


def load_cr_index(path_npz: str, path_meta_json: str) -> CRIndex:
    with pathlib.Path(path_meta_json).open("r", encoding="utf-8") as f:
        meta = CRIndexMeta(**json.load(f))
    z = np.load(path_npz, allow_pickle=True)

    def _pick(key: str, legacy_key: str) -> np.ndarray:
        if key in z:
            return z[key]
        return z[legacy_key]

    def _pick_list(key: str, legacy_key: str) -> list:
        if key in z:
            return z[key].tolist()
        return z[legacy_key].tolist()

    return CRIndex(
        meta=meta,
        proj1_components=_pick("proj1_components", "W1"),
        proj1_mean=_pick("proj1_mean", "mu1"),
        proj2_components=_pick("proj2_components", "W2"),
        proj2_mean=_pick("proj2_mean", "mu2"),
        coarse_centroids=_pick("coarse_centroids", "C2"),
        coarse_buckets=_pick_list("coarse_buckets", "buckets2"),
        topic_centroids_per_coarse=_pick_list("topic_centroids_per_coarse", "C1_per_C2"),
        topic_buckets_per_coarse=_pick_list("topic_buckets_per_coarse", "buckets1_per_C2"),
    )
