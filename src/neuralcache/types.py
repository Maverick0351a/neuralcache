from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from .config import Settings

_settings = Settings()


class Document(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None  # optional precomputed embedding

    @field_validator("id")
    @classmethod
    def _ensure_id(cls, value: str) -> str:
        if not value:
            raise ValueError("Document id must be non-empty")
        return value

    @field_validator("text")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        if len(value) > _settings.max_text_length:
            raise ValueError("Document text exceeds configured maximum length")
        return value


class RerankRequest(BaseModel):
    query: str
    documents: list[Document]
    query_embedding: list[float] | None = None
    top_k: int = 10
    mmr_lambda: float = 0.5

    @field_validator("top_k")
    @classmethod
    def _validate_top_k(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("top_k must be greater than zero")
        return value

    @field_validator("mmr_lambda")
    @classmethod
    def _clamp_mmr(cls, value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value


class ScoredDocument(Document):
    score: float = 0.0
    components: dict[str, float] = Field(default_factory=dict)


class Feedback(BaseModel):
    query: str
    selected_ids: list[str]
    success: float = 1.0  # [0,1], quality of result for narrative gating
