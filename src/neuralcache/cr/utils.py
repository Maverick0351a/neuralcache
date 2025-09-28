from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def pca_fit(x: np.ndarray, out_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute PCA projection via SVD, returning (components, mean)."""
    mu = x.mean(axis=0, keepdims=True)
    centered = x - mu
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:out_dim, :]
    return components, mu.squeeze(0)


def pca_transform(x: np.ndarray, components: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return (x - mu[None, :]) @ components.T


def pca_backproject(z: np.ndarray, components: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return z @ components + mu[None, :]


@dataclass
class KMeansResult:
    centroids: np.ndarray
    labels: np.ndarray


def kmeans_lloyd(x: np.ndarray, k: int, iters: int = 25, seed: int = 42) -> KMeansResult:
    """Simple Lloyd's algorithm with random-sample initialisation."""
    rng = np.random.default_rng(seed)
    n_samples = x.shape[0]
    if k > n_samples:
        raise ValueError("k cannot exceed number of samples")
    idx = rng.choice(n_samples, size=k, replace=False)
    centroids = x[idx].copy()
    labels = np.zeros(n_samples, dtype=np.int32)
    for _ in range(iters):
        distances = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        for centroid_idx in range(k):
            mask = labels == centroid_idx
            if mask.any():
                centroids[centroid_idx] = x[mask].mean(axis=0)
    return KMeansResult(centroids=centroids, labels=labels)
