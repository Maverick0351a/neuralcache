from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class GatingDecision:
    use_gating: bool
    uncertainty: float  # normalized entropy in [0,1]
    candidate_count: int


def _softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """Numerically stable softmax."""
    z = (x / max(temp, _EPS)) - np.max(x)
    e = np.exp(z)
    return e / (np.sum(e) + _EPS)


def normalized_entropy(probs: np.ndarray) -> float:
    """Entropy(p)/log(N) in [0,1]."""
    n = probs.shape[0]
    if n <= 1:
        return 0.0
    h = -np.sum(probs * np.log(probs + _EPS))
    return float(h / np.log(n))


def compute_uncertainty(similarities: np.ndarray, temp: float = 1.0) -> float:
    """
    Turn dense similarity scores into a normalized uncertainty:
    - Convert sims -> p via softmax (temperature controls peaking).
    - Return entropy(p)/log(N).
    Larger => more ambiguous query (flatter distribution).
    """
    if similarities.ndim != 1:
        raise ValueError("compute_uncertainty expects a 1D array of similarities.")
    probs = _softmax(similarities.astype(np.float64), temp=temp)
    return normalized_entropy(probs)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def decide_candidate_count(
    uncertainty: float,
    *,
    min_candidates: int,
    max_candidates: int,
    threshold: float,
    k: float = 8.0,
) -> int:
    """
    Map uncertainty to a candidate count in [min_candidates, max_candidates]
    using a logistic gate centered at `threshold` with slope `k`.
    """
    min_c = max(1, int(min_candidates))
    max_c = max(min_c, int(max_candidates))
    # Logit centered at threshold
    gate = _sigmoid(k * (uncertainty - threshold))  # in (0,1)
    count = int(round(min_c + gate * (max_c - min_c)))
    return max(min_c, min(count, max_c))


def make_decision(
    similarities: np.ndarray,
    *,
    mode: str = "auto",  # 'off' | 'auto' | 'on'
    threshold: float = 0.7,
    min_candidates: int = 100,
    max_candidates: int = 400,
    entropy_temp: float = 1.0,
) -> GatingDecision:
    """
    Decide whether to use gating and how many candidates to score.
    - mode='off': never gate.
    - mode='on' : always gate (use uncertainty only to set candidate_count).
    - mode='auto': gate iff uncertainty >= threshold.
    """
    u = compute_uncertainty(similarities, temp=entropy_temp)

    if mode == "off":
        return GatingDecision(use_gating=False, uncertainty=u, candidate_count=similarities.size)

    candidate_count = decide_candidate_count(
        u,
        min_candidates=min_candidates,
        max_candidates=max_candidates,
        threshold=threshold,
        k=8.0,
    )

    if mode == "on":
        return GatingDecision(use_gating=True, uncertainty=u, candidate_count=candidate_count)

    # auto
    use = u >= threshold
    return GatingDecision(
        use_gating=use,
        uncertainty=u,
        candidate_count=(candidate_count if use else similarities.size),
    )


def top_indices_by_similarity(similarities: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top-k similarities (descending)."""
    k = max(0, min(int(k), similarities.size))
    if k == 0:
        return np.empty(0, dtype=int)
    # Argpartition for speed, then sort the top-k slice
    idx = np.argpartition(-similarities, kth=k - 1)[:k]
    sub = similarities[idx]
    order = np.argsort(-sub, kind="mergesort")  # stable
    return idx[order]
