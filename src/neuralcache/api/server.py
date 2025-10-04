from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response, status, Request
from fastapi.responses import JSONResponse

from ..config import Settings
from ..metrics import latest_metrics, metrics_enabled, observe_rerank, record_feedback
from ..rerank import Reranker
from ..types import (
    BatchRerankResponseItem,
    Document,
    ErrorInfo,
    ErrorResponse,
    Feedback,
    RerankDebug,
    RerankRequest,
    RerankResponse,
    ScoredDocument,
)

settings = Settings()
app = FastAPI(title=settings.api_title, version=settings.api_version)


@app.middleware("http")
async def add_version_header(request: Request, call_next):  # type: ignore[override]
    response = await call_next(request)
    # Canonical header
    response.headers["X-NeuralCache-API-Version"] = settings.api_version
    # Back-compat alias (documented as deprecated once versioning policy matures)
    response.headers["X-API-Version"] = settings.api_version
    return response
reranker = Reranker(settings=settings)

_feedback_cache: OrderedDict[str, list[ScoredDocument]] = OrderedDict()
_feedback_lock = threading.Lock()
_rate_lock = threading.Lock()
_request_times: deque[float] = deque()

_sweeper_stop = threading.Event()
_sweeper_thread: threading.Thread | None = None


def _retention_sweep_loop() -> None:
    interval = max(0.0, settings.storage_retention_sweep_interval_s)
    if interval <= 0:
        return
    while not _sweeper_stop.wait(interval):
        try:
            retention_days = settings.storage_retention_days
            if retention_days is None or retention_days <= 0:
                continue
            retention_seconds = retention_days * 86400.0
            # Purge stale narrative & pheromones via underlying stores
            reranker.narr.purge_if_stale(retention_seconds)
            reranker.pher.purge_older_than(retention_seconds)
        except Exception:  # pragma: no cover - defensive
            pass


@app.on_event("startup")
def _start_retention_sweeper() -> None:
    if settings.storage_retention_sweep_on_start:
        # Execute one purge cycle synchronously if configured
        try:
            if settings.storage_retention_days and settings.storage_retention_days > 0:
                retention_seconds = settings.storage_retention_days * 86400.0
                reranker.narr.purge_if_stale(retention_seconds)
                reranker.pher.purge_older_than(retention_seconds)
        except Exception:
            pass
    if settings.storage_retention_sweep_interval_s > 0:
        global _sweeper_thread
        _sweeper_thread = threading.Thread(
            target=_retention_sweep_loop, name="nc-retention-sweeper", daemon=True
        )
        _sweeper_thread.start()


@app.on_event("shutdown")
def _stop_retention_sweeper() -> None:  # pragma: no cover
    _sweeper_stop.set()
    if _sweeper_thread and _sweeper_thread.is_alive():
        _sweeper_thread.join(timeout=1.0)


def _build_gating_overrides(req: RerankRequest) -> dict[str, object] | None:
    overrides: dict[str, object] = {}
    if req.gating_mode is not None:
        overrides["gating_mode"] = req.gating_mode
    if req.gating_threshold is not None:
        overrides["gating_threshold"] = float(req.gating_threshold)
    if req.gating_min_candidates is not None:
        overrides["gating_min_candidates"] = int(req.gating_min_candidates)
    if req.gating_max_candidates is not None:
        overrides["gating_max_candidates"] = int(req.gating_max_candidates)
    if req.gating_entropy_temp is not None:
        overrides["gating_entropy_temp"] = float(req.gating_entropy_temp)
    return overrides or None


def _extract_gating_debug(payload: dict[str, Any]) -> dict[str, Any] | None:
    gating_debug = payload.get("gating")
    return gating_debug if isinstance(gating_debug, dict) else None


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


@app.post("/rerank", response_model=RerankResponse)
async def rerank(
    req: RerankRequest,
    api_ok: None = Depends(_require_api_key),
    rate_ok: None = Depends(_rate_limit),
    use_cr: bool | None = Query(default=None, description="Override CR toggle"),
) -> JSONResponse:
    previous_cr = reranker.settings.cr.on
    if use_cr is not None:
        reranker.settings.cr.on = use_cr
    _validate_request(req)
    start = time.perf_counter()
    status_label = "success"
    try:
        overrides = _build_gating_overrides(req)
        debug_payload: dict[str, Any] = {}
        if req.query_embedding is not None:
            q = np.array(req.query_embedding, dtype=np.float32)
        else:
            q = reranker.encode_query(req.query)
        scored = reranker.score(
            q,
            list(req.documents),
            mmr_lambda=req.mmr_lambda,
            query_text=req.query,
            overrides=overrides,
            debug=debug_payload,
        )
        limited = scored[: min(req.top_k, len(scored))]
        _remember_scored(limited)
        payload = [doc.model_dump() for doc in limited]
        debug_model = RerankDebug(
            gating=_extract_gating_debug(debug_payload),
            deterministic=debug_payload.get("deterministic"),
            epsilon_used=debug_payload.get("epsilon_used"),
        )
        return JSONResponse(
            RerankResponse(results=[ScoredDocument(**doc) for doc in payload], debug=debug_model).model_dump()
        )
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
        reranker.settings.cr.on = previous_cr


@app.post("/rerank/batch", response_model=list[BatchRerankResponseItem])
async def rerank_batch(
    batch: list[RerankRequest],
    api_ok: None = Depends(_require_api_key),
    rate_ok: None = Depends(_rate_limit),
    use_cr: bool | None = Query(default=None, description="Override CR toggle"),
) -> JSONResponse:
    previous_cr = reranker.settings.cr.on
    if use_cr is not None:
        reranker.settings.cr.on = use_cr
    if len(batch) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size exceeds configured maximum",
        )
    results: list[dict[str, object]] = []
    start = time.perf_counter()
    status_label = "success"
    total_docs = 0
    try:
        for req in batch:
            _validate_request(req)
            total_docs += len(req.documents)
            overrides = _build_gating_overrides(req)
            debug_payload: dict[str, Any] = {}
            if req.query_embedding is not None:
                q = np.array(req.query_embedding, dtype=np.float32)
            else:
                q = reranker.encode_query(req.query)
            scored = reranker.score(
                q,
                list(req.documents),
                mmr_lambda=req.mmr_lambda,
                query_text=req.query,
                overrides=overrides,
                debug=debug_payload,
            )
            limited = scored[: min(req.top_k, len(scored))]
            _remember_scored(limited)
            payload = [doc.model_dump() for doc in limited]
            debug_model = RerankDebug(
                gating=_extract_gating_debug(debug_payload),
                deterministic=debug_payload.get("deterministic"),
                epsilon_used=debug_payload.get("epsilon_used"),
            )
            results.append(
                BatchRerankResponseItem(
                    results=[ScoredDocument(**doc) for doc in payload], debug=debug_model
                ).model_dump()
            )
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
        reranker.settings.cr.on = previous_cr


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


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:  # type: ignore[override]
    # Map status_code to stable error code strings
    code_map = {
        status.HTTP_400_BAD_REQUEST: "BAD_REQUEST",
        status.HTTP_401_UNAUTHORIZED: "UNAUTHORIZED",
        status.HTTP_404_NOT_FOUND: "NOT_FOUND",
        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: "ENTITY_TOO_LARGE",
        status.HTTP_422_UNPROCESSABLE_ENTITY: "VALIDATION_ERROR",
        status.HTTP_429_TOO_MANY_REQUESTS: "RATE_LIMITED",
        status.HTTP_500_INTERNAL_SERVER_ERROR: "INTERNAL_ERROR",
    }
    code = code_map.get(exc.status_code, "ERROR")
    body = ErrorResponse(error=ErrorInfo(code=code, message=str(exc.detail), detail=None))
    return JSONResponse(status_code=exc.status_code, content=body.model_dump())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
    body = ErrorResponse(error=ErrorInfo(code="INTERNAL_ERROR", message="Unhandled error", detail=None))
    return JSONResponse(status_code=500, content=body.model_dump())
