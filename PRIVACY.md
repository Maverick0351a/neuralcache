# Privacy & Data Handling

This document describes how NeuralCache processes, stores, and retains data so
operators can make informed decisions about deployment, compliance, and risk.

## Data Classes

| Data Type | Source | Stored? | Purpose | Notes |
|-----------|--------|--------|---------|-------|
| Query text | Client requests | No (transient) | Embedding & scoring | Only used in-memory; not persisted |
| Document text | Client payloads | No (transient) | Scoring / feature extraction | Not written to disk unless embedding backend logs externally |
| Embeddings | Generated (hash or model) | Narrative vector + pheromone features derived | Adaptive scoring | Embedding arrays reduced into aggregate state (EMA narrative) |
| Feedback signals | /feedback endpoint | Narrative + pheromone updates | Improve personalization & success weighting | Stored indirectly in adaptive state, not raw payload |
| API tokens | Environment variable | No | AuthZ | Not logged by default |
| Metrics counters | In-process | Optional (Prometheus scrape) | Observability | No PII; aggregated counts & durations |

## What Is Persisted

If `storage_persistence_enabled = true` (default) the following artifacts may be
written inside `storage_dir`:

| File | Contents | Privacy Considerations |
|------|----------|------------------------|
| `neuralcache.db` | SQLite metadata (if future features persist more) | Currently minimal; avoid storing PII here |
| `narrative.json` | Narrative EMA vector(s) (currently global, roadmap: namespaced) | Numerical floats only; cannot reconstruct raw text |
| `pheromones.json` | Document exposure/decay metadata (doc IDs & timestamps) | Ensure doc IDs are non-sensitive (hash if needed) |

Embeddings for documents are NOT persisted. Only aggregate statistical summaries
and per-doc exposure timestamps are stored.

## Namespaces & Isolation

Namespaces (see `MULTITENANCY.md`) isolate adaptive state so feedback in one
namespace does not influence another. Namespaces do not constitute a security
boundary; run separate deployments for strict data isolation.

## Retention Controls

- `NEURALCACHE_STORAGE_RETENTION_DAYS`: When set (>0), periodic sweeping removes
  pheromone entries and purges stale narrative participation.
- Setting it to `None` disables retention sweeping.
- Set `storage_retention_sweep_interval_s` > 0 to enable continuous purging.

## Data Deletion Procedure (Operator Playbook)

1. Identify target namespace(s) requiring deletion.
2. If persistence is enabled, stop the service to prevent new writes.
3. Remove or archive `narrative.json` and `pheromones.json`.
4. (If future namespaced persistence) remove only the namespace block / file.
5. Restart the service; new state will be rebuilt lazily.

For immediate in-memory purge without restart (future feature), an admin endpoint
could invalidate a namespace reranker—open an issue if needed.

## Minimizing Sensitive Exposure

- Use hashing or irreversible ID surrogates for document identifiers if they encode user info.
- Pre-redact or classify text prior to sending; large language model hallucination
  risk is not increased since raw text isn't persisted.
- Avoid embedding direct PII; upstream embedding providers may log inputs.
- Consider pairwise differential privacy or noise injection if narrative vectors
  might leak per-user contribution (threat model dependent).

## Embedding Backends

| Backend | Privacy Characteristics | Guidance |
|---------|------------------------|----------|
| `hash` | Fully local, deterministic hashing | Safe for local dev; no external calls |
| OpenAI / remote | Leaves host boundary | Ensure DPA with vendor; do not send prohibited data |
| Sentence-transformers (local) | Local inference | Keep model updated for security patches |

## Logging & Observability

By default, application logs should not emit document text or queries. If you
add custom logging, avoid raw payload dumps. Metrics endpoints expose only counts,
latencies, and success/error labels.

## Threat Model Notes

Out-of-scope for current version:
- Hard multi-tenant memory isolation (single process design)
- Cryptographic deletion guarantees
- Differential privacy enforcement

## Roadmap

| Area | Planned | Priority |
|------|---------|----------|
| Namespaced persistence | Yes | High |
| Namespace eviction / purge endpoint | Yes | Medium |
| Per-namespace metrics labeling | Yes | Medium |
| Configurable doc ID hashing strategy | Investigate | Medium |
| Differential privacy narrative updates | Investigate | Low |

## Operator Checklist Before Production

- [ ] Decide namespace strategy & expected cardinality
- [ ] Set API tokens (`NEURALCACHE_API_TOKENS`)
- [ ] Enable TLS termination at ingress / proxy
- [ ] Review embedding provider data policies
- [ ] Configure retention if required (`NEURALCACHE_STORAGE_RETENTION_DAYS`)
- [ ] Validate no PII in document IDs
- [ ] Monitor memory growth vs namespace count

## Questions / Contributions
Open an issue for clarifications, or propose enhancements to improve privacy guarantees.
