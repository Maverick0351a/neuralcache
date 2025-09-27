from __future__ import annotations

try:  # pragma: no cover - optional dependency
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Histogram,
        generate_latest,
    )
except Exception:  # pragma: no cover - prefer graceful fallback
    PROMETHEUS_AVAILABLE = False

    def metrics_enabled() -> bool:
        return False

    def observe_rerank(
        endpoint: str,
        status: str,
        duration: float,
        doc_count: int | None = None,
    ) -> None:
        return None

    def record_context_use(endpoint: str, hits: int, total: int) -> None:
        return None

    def latest_metrics() -> tuple[str, bytes] | None:
        return None

    def record_feedback(success: bool) -> None:
        return None

else:  # pragma: no cover - exercised when prometheus_client is installed
    PROMETHEUS_AVAILABLE = True
    _REGISTRY = CollectorRegistry()
    _RERANK_LATENCY = Histogram(
        "neuralcache_rerank_latency_seconds",
        "Latency (in seconds) spent handling rerank endpoints.",
        labelnames=("endpoint",),
        registry=_REGISTRY,
        buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
    )
    _RERANK_REQUESTS = Counter(
        "neuralcache_rerank_requests_total",
        "Total rerank requests processed, partitioned by status.",
        labelnames=("endpoint", "status"),
        registry=_REGISTRY,
    )
    _DOCS_PER_REQUEST = Histogram(
        "neuralcache_rerank_documents_per_request",
        "Documents supplied in each rerank invocation.",
        labelnames=("endpoint",),
        registry=_REGISTRY,
        buckets=(1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0),
    )
    _CONTEXT_USE = Counter(
        "neuralcache_context_use_outcomes_total",
        "Context-Use outcomes recorded by evaluation tooling.",
        labelnames=("endpoint", "outcome"),
        registry=_REGISTRY,
    )
    _FEEDBACK_EVENTS = Counter(
        "neuralcache_feedback_events_total",
        "Feedback outcomes by success flag.",
        labelnames=("outcome",),
        registry=_REGISTRY,
    )

    def metrics_enabled() -> bool:
        return True

    def observe_rerank(
        endpoint: str,
        status: str,
        duration: float,
        doc_count: int | None = None,
    ) -> None:
        _RERANK_LATENCY.labels(endpoint=endpoint).observe(max(duration, 0.0))
        _RERANK_REQUESTS.labels(endpoint=endpoint, status=status).inc()
        if doc_count is not None:
            _DOCS_PER_REQUEST.labels(endpoint=endpoint).observe(float(max(doc_count, 0)))

    def record_context_use(endpoint: str, hits: int, total: int) -> None:
        safe_total = max(total, hits, 0)
        hit_count = max(hits, 0)
        miss_count = max(safe_total - hit_count, 0)
        if hit_count:
            _CONTEXT_USE.labels(endpoint=endpoint, outcome="hit").inc(hit_count)
        if miss_count:
            _CONTEXT_USE.labels(endpoint=endpoint, outcome="miss").inc(miss_count)

    def latest_metrics() -> tuple[str, bytes] | None:
        return CONTENT_TYPE_LATEST, generate_latest(_REGISTRY)

    def record_feedback(success: bool) -> None:
        outcome = "success" if success else "failure"
        _FEEDBACK_EVENTS.labels(outcome=outcome).inc()


__all__ = [
    "latest_metrics",
    "metrics_enabled",
    "observe_rerank",
    "record_context_use",
    "record_feedback",
]
