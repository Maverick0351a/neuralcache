from __future__ import annotations

import json
import pathlib
import time
from contextlib import suppress

import numpy as np

from .similarity import safe_normalize
from .storage.sqlite_state import SQLiteState


class NarrativeTracker:
    def __init__(
        self,
        dim: int = 768,
        alpha: float = 0.01,
        success_gate: float = 0.5,
        path: str = "narrative.json",
        backend: str = "sqlite",
        storage_dir: str | None = None,
        sqlite_state: SQLiteState | None = None,
    ) -> None:
        self.alpha = float(alpha)
        self.success_gate = float(success_gate)
        base_path = pathlib.Path(storage_dir or ".")
        self.backend = backend.lower()
        self.path = (base_path / path).as_posix()
        self._sqlite = sqlite_state
        self.v = np.zeros((dim,), dtype=np.float32)
        self._updated_ts = 0.0
        self._load()

    def _load(self) -> None:
        if self.backend == "memory":
            self._updated_ts = 0.0
            return
        if self.backend == "sqlite" and self._sqlite is not None:
            with suppress(Exception):
                stored, updated_ts = self._sqlite.load_narrative_record()
                if stored is not None and stored.size == self.v.size:
                    self.v = stored.astype(np.float32)
                    self._updated_ts = float(updated_ts or 0.0)
            return

        path = pathlib.Path(self.path)
        if not path.exists():
            return

        with suppress(Exception):
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            arr = np.array(data.get("v", []), dtype=np.float32)
            if arr.size == self.v.size:
                self.v = arr
                self._updated_ts = float(data.get("updated_ts", 0.0))

    def _save(self) -> None:
        now = time.time()
        self._updated_ts = now
        if self.backend == "memory":
            return
        if self.backend == "sqlite" and self._sqlite is not None:
            with suppress(Exception):
                self._sqlite.save_narrative(self.v)
            return

        path = pathlib.Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with suppress(Exception), path.open("w", encoding="utf-8") as handle:
            json.dump({"v": self.v.tolist(), "updated_ts": now}, handle)
        with suppress(Exception):
            path.chmod(0o600)

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

    def purge_if_stale(self, retention_seconds: float) -> None:
        if retention_seconds <= 0:
            return
        cutoff = time.time() - retention_seconds
        if self._updated_ts == 0 or self._updated_ts >= cutoff:
            return
        self.v = np.zeros_like(self.v, dtype=np.float32)
        self._updated_ts = 0.0
        if self.backend == "sqlite" and self._sqlite is not None:
            with suppress(Exception):
                self._sqlite.clear_narrative()
        elif self.backend == "json":
            path = pathlib.Path(self.path)
            with suppress(FileNotFoundError):
                path.unlink()

    def refresh(self) -> None:
        self._load()
