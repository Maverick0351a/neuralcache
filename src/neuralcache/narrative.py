from __future__ import annotations

import json
import pathlib
from contextlib import suppress

import numpy as np

from .similarity import safe_normalize


class NarrativeTracker:
    def __init__(
        self,
        dim: int = 768,
        alpha: float = 0.01,
        success_gate: float = 0.5,
        path: str = "narrative.json",
    ) -> None:
        self.alpha = float(alpha)
        self.success_gate = float(success_gate)
        self.path = path
        self.v = np.zeros((dim,), dtype=np.float32)
        self._load()

    def _load(self) -> None:
        path = pathlib.Path(self.path)
        if not path.exists():
            return

        with suppress(Exception):
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            arr = np.array(data.get("v", []), dtype=np.float32)
            if arr.size == self.v.size:
                self.v = arr

    def _save(self) -> None:
        path = pathlib.Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with suppress(Exception), path.open("w", encoding="utf-8") as handle:
            json.dump({"v": self.v.tolist()}, handle)

    def update(self, doc_embedding: np.ndarray, success: float) -> None:
        if success < self.success_gate:
            return
        doc_embedding = doc_embedding.astype(np.float32).reshape(-1)
        if doc_embedding.size != self.v.size:
            # Resize narrative to match embedding dim if needed
            self.v = np.zeros_like(doc_embedding, dtype=np.float32)
        self.v = (1 - self.alpha) * self.v + self.alpha * doc_embedding
        self.v = safe_normalize(self.v)
        self._save()

    def coherence(self, doc_embeddings: np.ndarray) -> np.ndarray:
        # Returns cosine similarity with narrative vector
        if self.v.size == 0:
            return np.zeros((doc_embeddings.shape[0],), dtype=np.float32)
        v = self.v.reshape(1, -1)
        v = safe_normalize(v)
        docs_norm = safe_normalize(doc_embeddings)
        sims = (v @ docs_norm.T).reshape(-1)
        return sims.astype(np.float32)
