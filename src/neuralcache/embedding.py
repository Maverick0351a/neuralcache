from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable

import numpy as np


def _unit(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x) + eps
    return x / norm


def _token_hash(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.sha256(f"token::{token}".encode()).digest()
    idx = int.from_bytes(digest[:4], "big", signed=False) % dim
    sign = 1.0 if digest[4] & 1 else -1.0
    magnitude = 1.0 + (int(digest[5]) / 255.0)
    return idx, sign * magnitude


def stable_embed_text(text: str, dim: int = 384) -> np.ndarray:
    """Deterministic text embedding using SHA-256 with token-aware boosts."""
    values = np.empty(dim, dtype=np.float32)
    for i in range(dim):
        digest = hashlib.sha256(f"{text}::{i}".encode()).digest()
        unsigned = int.from_bytes(digest[:8], "big", signed=False)
        values[i] = (unsigned / 2**63) - 1.0

    # Token-based reinforcement to improve lexical discrimination
    tokens = [tok for tok in text.lower().split() if tok]
    if tokens:
        boost = np.zeros(dim, dtype=np.float32)
        scale = 1.0 / math.sqrt(len(tokens))
        for token in tokens:
            idx, weight = _token_hash(token, dim)
            boost[idx] += weight
    values += 1.2 * scale * boost

    return _unit(values)


def stable_embed_texts(texts: Iterable[str], dim: int = 384) -> np.ndarray:
    return np.vstack([stable_embed_text(text, dim=dim) for text in texts])


# Hook so callers can swap to a real model without touching call-sites.
def encode_texts(texts: Iterable[str], dim: int = 384) -> np.ndarray:
    """Default encoder: deterministic hash embedding.

    Replace this with a production embedding model while maintaining the shape (N, dim).
    """
    return stable_embed_texts(texts, dim=dim)
