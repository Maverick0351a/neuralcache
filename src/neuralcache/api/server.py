from __future__ import annotations

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ..config import Settings
from ..rerank import Reranker
from ..types import Document, Feedback, RerankRequest

settings = Settings()
app = FastAPI(title=settings.api_title, version=settings.api_version)
reranker = Reranker(settings=settings)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/rerank")
def rerank(req: RerankRequest) -> JSONResponse:
    # Build query embedding (fallback to hashing if not provided)
    if req.query_embedding is not None:
        q = np.array(req.query_embedding, dtype=np.float32)
    else:
        # simple hashing trick
        dim = settings.narrative_dim
        q = np.zeros((dim,), dtype=np.float32)
        for tok in req.query.lower().split():
            q[hash(tok) % dim] += 1.0

    docs = [Document(**d.model_dump()) for d in req.documents]
    scored = reranker.score(q, docs, mmr_lambda=req.mmr_lambda)
    topk = scored[: min(req.top_k, len(scored))]
    return JSONResponse([s.model_dump() for s in topk])


@app.post("/feedback")
def feedback(fb: Feedback) -> dict[str, str]:
    doc_map = {d.id: d for d in []}
    # In production, fetch documents by ID or reuse the last rerank batch
    # so narrative updates apply.
    reranker.update_feedback(fb.selected_ids, doc_map=doc_map, success=fb.success)
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict[str, float]:
    # Placeholder metrics. Extend with Prometheus exporter if desired.
    return {"requests_total": 0.0, "uptime_s": 0.0}
