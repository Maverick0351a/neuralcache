from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

from . import gating
from .config import Settings
from .cr.index import CRIndex, load_cr_index
from .cr.search import hierarchical_candidates
from .encoder import create_encoder
from .narrative import NarrativeTracker
from .pheromone import PheromoneStore
from .similarity import batched_cosine_sims, embed_corpus, safe_normalize
from .storage.sqlite_state import SQLiteState
from .types import Document, ScoredDocument
import os


def _safe_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return float(default)
    return float(default)


def _safe_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return int(default)
    return int(default)


class Reranker:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        if self.settings.deterministic:
            try:  # pragma: no cover
                random.seed(int(self.settings.deterministic_seed))
                np.random.seed(int(self.settings.deterministic_seed))
            except Exception:  # pragma: no cover
                pass
        self.encoder = create_encoder(
            self.settings.embedding_backend,
            dim=self.settings.narrative_dim,
            model=self.settings.embedding_model,
        )
        self._cr_index: CRIndex | None = None
        storage_backend = (self.settings.storage_backend or "sqlite").lower()
        if not self.settings.storage_persistence_enabled:
            storage_backend = "memory"
        storage_dir = Path(self.settings.storage_dir or ".")
        storage_dir.mkdir(parents=True, exist_ok=True)
        sqlite_state: SQLiteState | None = None
        retention_seconds: float | None = None
        if self.settings.storage_retention_days is not None:
            retention_seconds = max(0.0, float(self.settings.storage_retention_days) * 86400.0)
        if storage_backend == "sqlite":
            db_path = storage_dir / self.settings.storage_db_name
            sqlite_state = SQLiteState(path=str(db_path))
            if retention_seconds:
                sqlite_state.purge_older_than(retention_seconds)
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
        if retention_seconds:
            self.narr.purge_if_stale(retention_seconds)
            self.pher.purge_older_than(retention_seconds)

    def _ensure_cr_loaded(self) -> CRIndex | None:
        if not self.settings.cr.on:
            return None
        if self._cr_index is None:
            try:
                self._cr_index = load_cr_index(
                    self.settings.cr.index_npz_path,
                    self.settings.cr.index_meta_path,
                )
            except FileNotFoundError:
                self._cr_index = None
        return self._cr_index

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
        self,
        query_embedding: np.ndarray,
        docs: list[Document],
        mmr_lambda: float = 0.5,
        *,
        query_text: str | None = None,
        overrides: dict[str, object] | None = None,
        debug: dict[str, object] | None = None,
    ) -> list[ScoredDocument]:
        if len(docs) == 0:
            if debug is not None:
                debug["gating"] = {
                    "mode": (overrides or {}).get("gating_mode", self.settings.gating_mode),
                    "uncertainty": 0.0,
                    "use_gating": False,
                    "candidate_count": 0,
                    "effective_candidate_count": 0,
                    "total_candidates": 0,
                }
            return []

        doc_texts = [d.text for d in docs]
        doc_embeddings = self._ensure_embeddings(docs)
        q = query_embedding.astype(np.float32).reshape(-1)
        if q.size != doc_embeddings.shape[1]:
            # resize query vector via simple pad/truncate for compatibility
            target_dim = doc_embeddings.shape[1]
            q = q[:target_dim] if q.size > target_dim else np.pad(q, (0, target_dim - q.size))

        candidates = list(range(len(docs)))
        cr = self._ensure_cr_loaded()
        if cr is not None and query_text:
            dim0 = int(cr.meta.d0)
            doc_embeddings_q0 = embed_corpus(doc_texts, dim=dim0)
            q0 = embed_corpus([query_text], dim=dim0)[0]
            candidates = hierarchical_candidates(
                q0_query=q0,
                doc_embeddings_q0=doc_embeddings_q0,
                cr=cr,
                top_coarse=self.settings.cr.top_coarse,
                top_topics_per_coarse=self.settings.cr.top_topics_per_coarse,
                max_candidates=min(self.settings.cr.max_candidates, len(docs)),
            )
            if not candidates:
                candidates = list(range(len(docs)))

        if not candidates:
            return []

        dense = batched_cosine_sims(q, doc_embeddings)

        overrides = overrides or {}
        mode_override = overrides.get("gating_mode")
        mode = mode_override if isinstance(mode_override, str) else self.settings.gating_mode
        threshold = _safe_float(
            overrides.get("gating_threshold"),
            self.settings.gating_threshold,
        )
        min_c = _safe_int(
            overrides.get("gating_min_candidates"),
            self.settings.gating_min_candidates,
        )
        max_c = _safe_int(
            overrides.get("gating_max_candidates"),
            self.settings.gating_max_candidates,
        )
        temp = _safe_float(
            overrides.get("gating_entropy_temp"),
            self.settings.gating_entropy_temp,
        )

        candidate_indices = np.array(candidates, dtype=int)
        sims_for_gate = dense[candidate_indices]
        total_candidates = int(candidate_indices.size)

        decision = gating.make_decision(
            similarities=sims_for_gate,
            mode=str(mode),
            threshold=threshold,
            min_candidates=min_c,
            max_candidates=max_c,
            entropy_temp=temp,
        )

        if decision.use_gating:
            gating_positions = gating.top_indices_by_similarity(
                sims_for_gate, decision.candidate_count
            )
            candidate_indices = candidate_indices[gating_positions]
            sims_for_gate = sims_for_gate[gating_positions]

        effective_candidate_count = int(candidate_indices.size)

        debug_gating = {
            "mode": mode,
            "uncertainty": decision.uncertainty,
            "use_gating": decision.use_gating,
            "candidate_count": int(decision.candidate_count),
            "effective_candidate_count": effective_candidate_count,
            "total_candidates": total_candidates,
        }

        if debug is not None:
            debug["gating"] = debug_gating
            debug["deterministic"] = bool(self.settings.deterministic)

        if effective_candidate_count == 0:
            return []

        docs_subset = [docs[i] for i in candidate_indices]
        doc_embeddings_subset = doc_embeddings[candidate_indices]
        dense_subset = dense[candidate_indices]

        narr = self.narr.coherence(doc_embeddings_subset)
        pher = np.array(
            self.pher.bulk_bonus([docs[i].id for i in candidate_indices]),
            dtype=np.float32,
        )

        base = (
            self.settings.weight_dense * dense_subset
            + self.settings.weight_narrative * narr
            + self.settings.weight_pheromone * pher
        )

        # MMR diversity — greedy re-ranking
        mmr_selected_positions: list[int] = []
        remaining_positions = set(range(effective_candidate_count))

        # ε-greedy exploration: occasionally pick a random item
        override = os.getenv("NEURALCACHE_EPSILON")
        epsilon = 0.0 if self.settings.deterministic else self.settings.epsilon_greedy
        if override is not None and not self.settings.deterministic:
            try:
                val = float(override)
                if 0.0 <= val <= 1.0:
                    epsilon = val
            except ValueError:  # pragma: no cover
                pass
        mmr_lam = float(mmr_lambda if 0.0 <= mmr_lambda <= 1.0 else 0.5)

        def mmr_gain(pos: int) -> float:
            if not mmr_selected_positions:
                return float(base[pos])
            sim_to_selected = max(
                float(np.dot(doc_embeddings_subset[pos], doc_embeddings_subset[j]))
                for j in mmr_selected_positions
            )
            return float(mmr_lam * base[pos] - (1.0 - mmr_lam) * sim_to_selected)

        order_positions: list[int] = []
        while remaining_positions:
            if random.random() < epsilon:
                pick = random.choice(list(remaining_positions))
            else:
                pick = max(remaining_positions, key=mmr_gain)
            order_positions.append(pick)
            mmr_selected_positions.append(pick)
            remaining_positions.remove(pick)

        scored = [
            ScoredDocument(
                id=docs_subset[pos].id,
                text=docs_subset[pos].text,
                metadata=docs_subset[pos].metadata,
                embedding=docs_subset[pos].embedding,
                score=float(base[pos]),
                components={
                    "dense": float(dense_subset[pos]),
                    "narrative": float(narr[pos]),
                    "pheromone": float(pher[pos]),
                },
            )
            for pos in order_positions
        ]
        if debug is not None:
            debug["epsilon_used"] = float(epsilon)

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
