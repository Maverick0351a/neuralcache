from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from .config import Settings
from .encoder import create_encoder
from .narrative import NarrativeTracker
from .pheromone import PheromoneStore
from .similarity import batched_cosine_sims, safe_normalize
from .storage.sqlite_state import SQLiteState
from .types import Document, ScoredDocument


class Reranker:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.encoder = create_encoder(
            self.settings.embedding_backend,
            dim=self.settings.narrative_dim,
            model=self.settings.embedding_model,
        )
        storage_backend = (self.settings.storage_backend or "sqlite").lower()
        storage_dir = Path(self.settings.storage_dir or ".")
        storage_dir.mkdir(parents=True, exist_ok=True)
        sqlite_state: SQLiteState | None = None
        if storage_backend == "sqlite":
            db_path = storage_dir / self.settings.storage_db_name
            sqlite_state = SQLiteState(path=str(db_path))
        self._sqlite_state = sqlite_state
        self.narr = NarrativeTracker(
            dim=self.settings.narrative_dim,
            alpha=self.settings.narrative_ema_alpha,
            success_gate=self.settings.narrative_success_gate,
            path=self.settings.narrative_store_path,
            backend=storage_backend,
            storage_dir=str(storage_dir),
            sqlite_state=sqlite_state,
        )
        self.pher = PheromoneStore(
            half_life_s=self.settings.pheromone_decay_half_life_s,
            exposure_penalty=self.settings.pheromone_exposure_penalty,
            path=self.settings.pheromone_store_path,
            backend=storage_backend,
            storage_dir=str(storage_dir),
            sqlite_state=sqlite_state,
        )

    def _ensure_embeddings(self, docs: list[Document]) -> np.ndarray:
        # Expect embeddings to be provided; otherwise fallback to simple bag-of-words hashing.
        # In production you would plug a real embedding model here.
        if len(docs) == 0:
            return np.zeros((0, self.settings.narrative_dim), dtype=np.float32)
        embeddings: list[np.ndarray | None] = []
        missing_texts: list[str] = []
        missing_indices: list[int] = []

        for idx, doc in enumerate(docs):
            if doc.embedding:
                embeddings.append(np.asarray(doc.embedding, dtype=np.float32))
            else:
                embeddings.append(None)
                missing_texts.append(doc.text)
                missing_indices.append(idx)

        if missing_texts:
            encoded = self.encoder.encode_batch(missing_texts)
            encoded = np.atleast_2d(np.asarray(encoded, dtype=np.float32))
            for offset, vec in zip(missing_indices, encoded, strict=False):
                embeddings[offset] = np.asarray(vec, dtype=np.float32)

        target_dim = self.settings.narrative_dim
        adjusted: list[np.ndarray] = []
        for emb in embeddings:
            vec = (
                np.zeros((target_dim,), dtype=np.float32)
                if emb is None
                else np.asarray(emb, dtype=np.float32).reshape(-1)
            )
            if vec.size > target_dim:
                vec = vec[:target_dim]
            elif vec.size < target_dim:
                vec = np.pad(vec, (0, target_dim - vec.size))
            adjusted.append(vec.astype(np.float32))

        mat = np.stack(adjusted, axis=0)
        return safe_normalize(mat)

    def encode_query(self, query: str) -> np.ndarray:
        vec = self.encoder.encode(query)
        return safe_normalize(np.asarray(vec, dtype=np.float32).reshape(1, -1)).reshape(-1)

    def score(
        self, query_embedding: np.ndarray, docs: list[Document], mmr_lambda: float = 0.5
    ) -> list[ScoredDocument]:
        if len(docs) == 0:
            return []

        doc_embeddings = self._ensure_embeddings(docs)
        q = query_embedding.astype(np.float32).reshape(-1)
        if q.size != doc_embeddings.shape[1]:
            # resize query vector via simple pad/truncate for compatibility
            target_dim = doc_embeddings.shape[1]
            q = q[:target_dim] if q.size > target_dim else np.pad(q, (0, target_dim - q.size))
        dense = batched_cosine_sims(q, doc_embeddings)

        narr = self.narr.coherence(doc_embeddings)
        pher = np.array(self.pher.bulk_bonus([d.id for d in docs]), dtype=np.float32)

        base = (
            self.settings.weight_dense * dense
            + self.settings.weight_narrative * narr
            + self.settings.weight_pheromone * pher
        )

        # MMR diversity — greedy re-ranking
        doc_count = len(docs)
        selected: list[int] = []
        remaining = set(range(doc_count))

        # ε-greedy exploration: occasionally pick a random item
        epsilon = self.settings.epsilon_greedy
        mmr_lam = float(mmr_lambda if 0.0 <= mmr_lambda <= 1.0 else 0.5)

        def mmr_gain(idx: int) -> float:
            if not selected:
                return float(base[idx])
            sim_to_selected = max(
                float(np.dot(doc_embeddings[idx], doc_embeddings[j])) for j in selected
            )
            return float(mmr_lam * base[idx] - (1.0 - mmr_lam) * sim_to_selected)

        order: list[int] = []
        while remaining:
            if random.random() < epsilon:
                pick = random.choice(list(remaining))
            else:
                pick = max(remaining, key=mmr_gain)
            order.append(pick)
            selected.append(pick)
            remaining.remove(pick)

        scored = [
            ScoredDocument(
                id=docs[i].id,
                text=docs[i].text,
                metadata=docs[i].metadata,
                embedding=docs[i].embedding,
                score=float(base[i]),
                components={
                    "dense": float(dense[i]),
                    "narrative": float(narr[i]),
                    "pheromone": float(pher[i]),
                },
            )
            for i in order
        ]

        # Record exposure for top-K
        self.pher.record_exposure([sd.id for sd in scored[: min(len(scored), 10)]])

        return scored

    def update_feedback(
        self,
        selected_ids: list[str],
        doc_map: dict[str, Document] | None,
        success: float,
        *,
        best_doc_embedding: list[float] | None = None,
        best_doc_text: str | None = None,
    ) -> None:
        # Update narrative and pheromones with feedback signal
        self.pher.reinforce(selected_ids, reward=success)
        if not selected_ids and best_doc_embedding is None and not best_doc_text:
            return

        selected_docs: list[Document] = []
        if doc_map:
            selected_docs = [doc_map[sid] for sid in selected_ids if sid in doc_map]

        if selected_docs:
            doc_embeddings = self._ensure_embeddings(selected_docs)
            emb = doc_embeddings.mean(axis=0)
            self.narr.update(emb, success=success)
            return

        emb = None
        if best_doc_embedding is not None:
            emb = np.asarray(best_doc_embedding, dtype=np.float32)
        elif best_doc_text:
            emb = self.encoder.encode(best_doc_text)
        if emb is not None:
            self.narr.update(np.asarray(emb, dtype=np.float32), success=success)

    def feedback(
        self,
        selected_ids: list[str],
        success: float,
        best_doc_embedding: list[float] | None = None,
        best_doc_text: str | None = None,
    ) -> None:
        self.update_feedback(
            selected_ids,
            doc_map=None,
            success=success,
            best_doc_embedding=best_doc_embedding,
            best_doc_text=best_doc_text,
        )
