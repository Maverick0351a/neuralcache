from __future__ import annotations

import time

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from ..config import Settings
from ..metrics import latest_metrics, metrics_enabled, observe_rerank
from ..rerank import Reranker
from ..types import RerankRequest, ScoredDocument
from .server import app as legacy_app

settings = Settings()
app = FastAPI(title=f"{settings.api_title} Plus", version=settings.api_version)
app.mount("/v1", legacy_app)
reranker = Reranker(settings=settings)


class BatchRerankRequest(BaseModel):
    requests: list[RerankRequest] = Field(default_factory=list)


def _resolve_query_embedding(req: RerankRequest) -> np.ndarray:
    if req.query_embedding is not None:
        return np.array(req.query_embedding, dtype=np.float32)
    return reranker.encode_query(req.query)


def _score_documents(req: RerankRequest) -> list[ScoredDocument]:
    docs = list(req.documents)
    query_embedding = _resolve_query_embedding(req)
    scored = reranker.score(query_embedding, docs, mmr_lambda=req.mmr_lambda)
    return scored[: min(req.top_k, len(scored))]


def _observe(endpoint: str, status: str, duration: float, doc_count: int) -> None:
    observe_rerank(endpoint=endpoint, status=status, duration=duration, doc_count=doc_count)


@app.post("/rerank")
def rerank_endpoint(req: RerankRequest) -> JSONResponse:
    start = time.perf_counter()
    status = "success"
    scope_docs: list[ScoredDocument] = []
    try:
        scope_docs = _score_documents(req)
        payload = [doc.model_dump() for doc in scope_docs]
        return JSONResponse(payload)
    except Exception as exc:  # pragma: no cover - FastAPI handles traceback
        status = "error"
        raise exc
    finally:
        duration = time.perf_counter() - start
        _observe("/rerank", status, duration, len(scope_docs))


@app.post("/rerank/batch")
def rerank_batch_endpoint(batch: BatchRerankRequest) -> JSONResponse:
    start = time.perf_counter()
    status = "success"
    scored_batches: list[list[ScoredDocument]] = []
    try:
        for request in batch.requests:
            scored = _score_documents(request)
            scored_batches.append(scored)
        payload = [[doc.model_dump() for doc in scored] for scored in scored_batches]
        return JSONResponse(payload)
    except Exception as exc:  # pragma: no cover
        status = "error"
        raise exc
    finally:
        duration = time.perf_counter() - start
        doc_count = sum(len(scored) for scored in scored_batches)
        _observe("/rerank/batch", status, duration, doc_count)


@app.get("/metrics")
def metrics_endpoint() -> Response:
    rendered = latest_metrics()
    if rendered is None:
        raise HTTPException(status_code=503, detail="prometheus_client is not installed.")
    content_type, payload = rendered
    return Response(content=payload, media_type=content_type)


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "metrics_enabled": str(metrics_enabled()).lower()}
