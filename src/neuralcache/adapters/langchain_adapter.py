from __future__ import annotations

from typing import Any

try:
    from langchain_core.documents import Document as LCDocument
except Exception:  # pragma: no cover - optional dependency
    LCDocument = Any  # type: ignore

import numpy as np

from ..config import Settings
from ..rerank import Reranker
from ..types import Document as NC_Document


class NeuralCacheLangChainReranker:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.reranker = Reranker(self.settings)

    def __call__(self, query: str, documents: list[LCDocument]) -> list[LCDocument]:
        # Convert to NeuralCache docs
        nc_docs = [
            NC_Document(
                id=str(index),
                text=doc.page_content,
                metadata=doc.metadata,
            )
            for index, doc in enumerate(documents)
        ]
        # Hash-based query embedding for demo
        dim = self.settings.narrative_dim
        q = np.zeros((dim,), dtype=np.float32)
        for tok in query.lower().split():
            q[hash(tok) % dim] += 1.0

        scored = self.reranker.score(q, nc_docs)
        # Map back to LC docs preserving metadata
        return [documents[int(sd.id)] for sd in scored]
