from __future__ import annotations

import random

import numpy as np

from .config import Settings
from .narrative import NarrativeTracker
from .pheromone import PheromoneStore
from .similarity import batched_cosine_sims, safe_normalize
from .types import Document, ScoredDocument


class Reranker:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.narr = NarrativeTracker(
            dim=self.settings.narrative_dim,
            alpha=self.settings.narrative_ema_alpha,
            success_gate=self.settings.narrative_success_gate,
        )
        self.pher = PheromoneStore(
            half_life_s=self.settings.pheromone_decay_half_life_s,
            exposure_penalty=self.settings.pheromone_exposure_penalty,
        )

    def _ensure_embeddings(self, docs: list[Document]) -> np.ndarray:
        # Expect embeddings to be provided; otherwise fallback to simple bag-of-words hashing.
        # In production you would plug a real embedding model here.
        if len(docs) == 0:
            return np.zeros((0, self.settings.narrative_dim), dtype=np.float32)
        if docs[0].embedding is not None:
            return np.array([d.embedding or [] for d in docs], dtype=np.float32)
        # Fallback: hashing trick to fixed dim
        dim = self.settings.narrative_dim
        mat = np.zeros((len(docs), dim), dtype=np.float32)
        for i, d in enumerate(docs):
            for tok in d.text.lower().split():
                h = hash(tok) % dim
                mat[i, h] += 1.0
        return safe_normalize(mat)

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
            q = (
                q[:target_dim]
                if q.size > target_dim
                else np.pad(q, (0, target_dim - q.size))
            )
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
                float(np.dot(doc_embeddings[idx], doc_embeddings[j]))
                for j in selected
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
        doc_map: dict[str, Document],
        success: float,
    ) -> None:
        # Update narrative and pheromones with feedback signal
        self.pher.reinforce(selected_ids, reward=success)
        # For narrative, average embeddings of selected docs
        if not selected_ids:
            return
        selected_docs = [doc_map[sid] for sid in selected_ids if sid in doc_map]
        if not selected_docs:
            return
        doc_embeddings = self._ensure_embeddings(selected_docs)
        emb = doc_embeddings.mean(axis=0)
        self.narr.update(emb, success=success)
