from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from fastapi.responses import JSONResponse

from ..config import Settings
from ..metrics import latest_metrics, metrics_enabled, observe_rerank, record_feedback
from ..rerank import Reranker
from ..types import Document, Feedback, RerankRequest, ScoredDocument

settings = Settings()
app = FastAPI(title=settings.api_title, version=settings.api_version)
reranker = Reranker(settings=settings)

_feedback_cache: OrderedDict[str, list[ScoredDocument]] = OrderedDict()
_feedback_lock = threading.Lock()
_rate_lock = threading.Lock()
_request_times: deque[float] = deque()


def _require_api_key(
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
    authorization: str | None = Header(default=None),
) -> None:
    tokens = {token.strip() for token in settings.api_tokens if token.strip()}
    if not tokens:
        return
    presented: set[str] = set()
    if x_api_key:
        presented.add(x_api_key.strip())
    if authorization and authorization.lower().startswith("bearer "):
        presented.add(authorization.split(" ", 1)[1].strip())
    if tokens.isdisjoint(presented):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API token")


def _rate_limit() -> None:
    limit = settings.rate_limit_per_minute
    if not limit or limit <= 0:
        return
    now = time.time()
    with _rate_lock:
        window_start = now - 60.0
        while _request_times and _request_times[0] < window_start:
            _request_times.popleft()
        if len(_request_times) >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )
        _request_times.append(now)


def _validate_request(req: RerankRequest) -> None:
    if req.top_k > settings.max_top_k:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="top_k exceeds configured maximum",
        )
    if len(req.documents) > settings.max_documents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document count exceeds configured maximum",
        )
    for doc in req.documents:
        if len(doc.text) > settings.max_text_length:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Document {doc.id} text length exceeds limit",
            )


def _remember_scored(docs: list[ScoredDocument]) -> None:
    if settings.feedback_cache_size <= 0:
        return
    with _feedback_lock:
        for doc in docs:
            _feedback_cache[doc.id] = docs
        while len(_feedback_cache) > settings.feedback_cache_size:
            _feedback_cache.popitem(last=False)


def _documents_for_ids(ids: list[str]) -> dict[str, Document]:
    result: dict[str, Document] = {}
    with _feedback_lock:
        for doc_id in ids:
            scored = _feedback_cache.get(doc_id)
            if not scored:
                continue
            for item in scored:
                if item.id == doc_id:
                    result[doc_id] = Document.model_validate(item.model_dump())
                    break
    return result


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/rerank")
async def rerank(
    req: RerankRequest,
    api_ok: None = Depends(_require_api_key),
    rate_ok: None = Depends(_rate_limit),
) -> JSONResponse:
    _validate_request(req)
    start = time.perf_counter()
    status_label = "success"
    try:
        if req.query_embedding is not None:
            q = np.array(req.query_embedding, dtype=np.float32)
        else:
            q = reranker.encode_query(req.query)
        scored = reranker.score(q, list(req.documents), mmr_lambda=req.mmr_lambda)
        limited = scored[: min(req.top_k, len(scored))]
        _remember_scored(limited)
        payload = [doc.model_dump() for doc in limited]
        return JSONResponse(payload)
    except HTTPException:
        status_label = "error"
        raise
    except Exception as exc:  # pragma: no cover - FastAPI will log the stack trace
        status_label = "error"
        raise exc
    finally:
        observe_rerank(
            endpoint="/rerank",
            status=status_label,
            duration=time.perf_counter() - start,
            doc_count=len(req.documents),
        )


@app.post("/rerank/batch")
async def rerank_batch(
    batch: list[RerankRequest],
    api_ok: None = Depends(_require_api_key),
    rate_ok: None = Depends(_rate_limit),
) -> JSONResponse:
    if len(batch) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size exceeds configured maximum",
        )
    results: list[list[dict[str, object]]] = []
    start = time.perf_counter()
    status_label = "success"
    total_docs = 0
    try:
        for req in batch:
            _validate_request(req)
            total_docs += len(req.documents)
            if req.query_embedding is not None:
                q = np.array(req.query_embedding, dtype=np.float32)
            else:
                q = reranker.encode_query(req.query)
            scored = reranker.score(q, list(req.documents), mmr_lambda=req.mmr_lambda)
            limited = scored[: min(req.top_k, len(scored))]
            _remember_scored(limited)
            results.append([doc.model_dump() for doc in limited])
        return JSONResponse(results)
    except HTTPException:
        status_label = "error"
        raise
    except Exception as exc:  # pragma: no cover
        status_label = "error"
        raise exc
    finally:
        observe_rerank(
            endpoint="/rerank/batch",
            status=status_label,
            duration=time.perf_counter() - start,
            doc_count=total_docs,
        )


class FeedbackRequest(Feedback):
    best_doc_text: str | None = None
    best_doc_embedding: list[float] | None = None


@app.post("/feedback")
async def feedback(
    fb: FeedbackRequest,
    api_ok: None = Depends(_require_api_key),
    rate_ok: None = Depends(_rate_limit),
) -> dict[str, str]:
    if not fb.selected_ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="selected_ids required")
    doc_map = _documents_for_ids(fb.selected_ids)
    if len(doc_map) != len(fb.selected_ids):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="One or more selected_ids are unknown or expired",
        )
    reranker.update_feedback(
        fb.selected_ids,
        doc_map=doc_map,
        success=fb.success,
        best_doc_embedding=fb.best_doc_embedding,
        best_doc_text=fb.best_doc_text,
    )
    record_feedback(fb.success >= settings.narrative_success_gate)
    return {"status": "ok"}


@app.get("/metrics")
async def metrics(
    api_ok: None = Depends(_require_api_key),
) -> Response:
    if not settings.metrics_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Metrics are disabled")
    rendered = latest_metrics()
    if rendered is None or not metrics_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="prometheus_client not installed",
        )
    content_type, payload = rendered
    return Response(content=payload, media_type=content_type)
