from __future__ import annotations

from typing import Any, Literal

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
    gating_mode: Literal["off", "auto", "on"] | None = None
    gating_threshold: float | None = None
    gating_min_candidates: int | None = None
    gating_max_candidates: int | None = None
    gating_entropy_temp: float | None = None

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


class RerankDebug(BaseModel):
    gating: dict[str, Any] | None = None
    deterministic: bool | None = None
    epsilon_used: float | None = None


class ErrorInfo(BaseModel):
    code: str
    message: str
    detail: Any | None = None


class ErrorResponse(BaseModel):
    error: ErrorInfo


class RerankResponse(BaseModel):
    results: list[ScoredDocument]
    debug: RerankDebug | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class BatchRerankResponseItem(RerankResponse):
    pass


class Feedback(BaseModel):
    query: str
    selected_ids: list[str]
    success: float = 1.0  # [0,1], quality of result for narrative gating
