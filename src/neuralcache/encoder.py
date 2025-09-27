"""Embedding helpers for NeuralCache.

Provides optional integrations with external embedding providers while
falling back to a deterministic hashing encoder that requires no extra
dependencies. All third-party imports are guarded so the module can be
used in lightweight environments without the optional packages
installed.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

import numpy as np

logger = logging.getLogger(__name__)


class SupportsEncode(Protocol):
    """Protocol describing the encode interface expected by callers."""

    # pragma: no cover - protocol definition
    def encode(self, text: str) -> np.ndarray: ...

    # pragma: no cover - protocol definition
    def encode_batch(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(slots=True)
class HashingEncoder:
    """Deterministic hashing encoder using MD5 scatter."""

    dim: int

    def encode(self, text: str) -> np.ndarray:
        return _hash_to_vector(text, self.dim)

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.encode(text) for text in texts], axis=0)


class OpenAIEncoder:
    """Thin wrapper around the OpenAI embeddings API."""

    def __init__(self, model: str) -> None:
        try:
            module = importlib.import_module("openai")
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise ImportError(
                "openai package is required for the 'openai' embedding backend"
            ) from exc
        client_factory: Any = module.OpenAI
        self._client = client_factory()
        self._model = model

    def encode(self, text: str) -> np.ndarray:
        response = self._client.embeddings.create(model=self._model, input=[text])
        return _to_float32(response.data[0].embedding)

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        response = self._client.embeddings.create(model=self._model, input=list(texts))
        vectors = [_to_float32(item.embedding) for item in response.data]
        return _ensure_matrix(vectors)


class SentenceTransformerEncoder:
    """Wrapper for sentence-transformers models."""

    def __init__(self, model_name: str) -> None:
        try:
            module = importlib.import_module("sentence_transformers")
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise ImportError(
                "sentence-transformers package is required for the 'sentence-transformer' backend"
            ) from exc
        encoder_factory: Any = module.SentenceTransformer
        self._model = encoder_factory(model_name)

    def encode(self, text: str) -> np.ndarray:
        vector = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return _to_float32(vector)

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = self._model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        return _ensure_matrix(vectors)


def _hash_to_vector(text: str, dim: int) -> np.ndarray:
    vec = np.zeros((dim,), dtype=np.float32)
    for token in _tokenise(text):
        digest = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).hexdigest()
        val = int(digest, 16)
        idx = val % dim
        sign = 1.0 if (val & 1) == 0 else -1.0
        vec[idx] += sign
    if not np.any(vec):
        vec[0] = 1.0
    return vec


def _tokenise(text: str) -> Iterable[str]:
    return text.lower().split()


def _to_float32(array_like: Sequence[float]) -> np.ndarray:
    return np.asarray(array_like, dtype=np.float32)


VectorInput: TypeAlias = Sequence[float] | np.ndarray


def _ensure_matrix(vectors: Sequence[VectorInput]) -> np.ndarray:
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray(vectors, dtype=np.float32)


def create_encoder(name: str, *, dim: int, model: str | None = None) -> SupportsEncode:
    """Factory returning an encoder implementation based on configuration."""

    backend = (name or "hash").lower()
    if backend == "hash":
        return HashingEncoder(dim=dim)
    if backend in {"openai", "openai-api"}:
        chosen_model = model or "text-embedding-3-small"
        try:
            return OpenAIEncoder(model=chosen_model)
        except ImportError:
            logger.warning(
                "openai backend requested but 'openai' package not installed; "
                "falling back to hashing encoder"
            )
            return HashingEncoder(dim=dim)
    if backend in {"sentence-transformer", "sentence_transformer", "hf"}:
        chosen_model = model or "all-MiniLM-L6-v2"
        try:
            return SentenceTransformerEncoder(model_name=chosen_model)
        except ImportError:
            logger.warning(
                "sentence-transformer backend requested but dependency missing; "
                "falling back to hashing encoder"
            )
            return HashingEncoder(dim=dim)

    logger.warning("Unknown embedding backend '%s'; defaulting to hashing encoder", name)
    return HashingEncoder(dim=dim)


__all__ = [
    "SupportsEncode",
    "HashingEncoder",
    "OpenAIEncoder",
    "SentenceTransformerEncoder",
    "create_encoder",
]
