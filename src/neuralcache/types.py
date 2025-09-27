from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None  # optional precomputed embedding


class RerankRequest(BaseModel):
    query: str
    documents: list[Document]
    query_embedding: list[float] | None = None
    top_k: int = 10
    mmr_lambda: float = 0.5


class ScoredDocument(Document):
    score: float = 0.0
    components: dict[str, float] = Field(default_factory=dict)


class Feedback(BaseModel):
    query: str
    selected_ids: list[str]
    success: float = 1.0  # [0,1], quality of result for narrative gating
