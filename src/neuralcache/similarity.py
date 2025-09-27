from __future__ import annotations

import numpy as np


def safe_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + eps
    return x / n


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a2 = a.reshape(-1)
    b2 = b.reshape(-1)
    denom = (np.linalg.norm(a2) * np.linalg.norm(b2)) or 1e-9
    return float(np.dot(a2, b2) / denom)


def batched_cosine_sims(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    # query shape: (D,), docs: (N,D)
    q = query.reshape(1, -1)
    q = safe_normalize(q)
    docs_norm = safe_normalize(docs)
    return (q @ docs_norm.T).reshape(-1)
