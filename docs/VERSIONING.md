# API Versioning Policy

NeuralCache aims to provide a stable, predictable API surface for production RAG pipelines. This document defines how we version and evolve HTTP endpoints and schemas.

## Current status (0.3.x)

- The public HTTP API (e.g. `/rerank`, `/rerank/batch`, `/feedback`, `/metrics`, `/healthz`) emits an `X-NeuralCache-API-Version` header on every response. For 0.3.x this matches the library package version (e.g. `0.3.1`).
- A compatibility alias header `X-API-Version` is also returned. This alias may be deprecated in a future minor release; rely on `X-NeuralCache-API-Version`.
- Legacy routes may also appear under `/v1/...` for a limited deprecation window when new structural changes land.

## SemVer alignment

We follow semantic versioning for the PyPI package. API changes are classified using the same MAJOR.MINOR.PATCH:

| Change type | SemVer impact | Examples |
|-------------|---------------|----------|
| Backwards compatible additions | MINOR | New optional field in response, new endpoint |
| Backwards incompatible change | MAJOR (or guarded under new path like `/v2`) | Removing/renaming response field, changing required parameter |
| Patch / bug fix | PATCH | Correcting error message text, performance improvements, internal refactors |

## Stability guarantees

Within a given MINOR series (e.g. `0.3.*`):
- Response envelope structure (`RerankResponse`, `ErrorResponse`) will not remove or rename existing fields.
- Error code strings (`BAD_REQUEST`, `UNAUTHORIZED`, etc.) remain stable.
- New fields, if added, are optional (nullable or have safe defaults) and documented in the CHANGELOG.

## Planned evolution

| Planned feature | Approach | Notes |
|-----------------|----------|-------|
| Deterministic mode flag exposure | Add optional field `debug.deterministic` | Non-breaking addition |
| Epsilon configuration surface | Add `debug.exploration` metadata | Non-breaking addition |
| API v2 (future) | Introduce `/v2` path + negotiated Accept header | Only if envelope needs a breaking change |

## Compatibility strategies

1. Additive first: prefer adding new optional fields over mutating existing ones.
2. Dual route window: when a breaking change is unavoidable, ship new behavior under `/vN` while keeping prior version for at least one MINOR release cycle.
3. Explicit deprecations: mark deprecated fields in docs + CHANGELOG and only remove in next MAJOR.
4. Header negotiation (future): we may introduce an `Accept: application/vnd.neuralcache.v2+json` media type to negotiate structural changes without path proliferation.

## Client guidance

- Always read the `X-NeuralCache-API-Version` header; log it for debugging.
- Tolerate unknown fields in JSON responses (forward compatibility).
- Treat missing optional fields as unset rather than error conditions.
- Pin to a tested MINOR series when deploying (`neuralcache>=0.3,<0.4`).

## Deprecation process

1. Mark upcoming removal in `CHANGELOG.md` under a "Deprecated" subsection.
2. Add runtime warning log (where feasible) when deprecated surface is invoked.
3. Remove no sooner than the next MAJOR.

## FAQ

**Why not start at `/v1`?** 0.x line allows us to refine ergonomics quickly while still offering stability guarantees documented here. A `/v1` stable base path will be introduced before 1.0.

**How do I request a new field?** Open a GitHub issue describing the use case; we prefer additive changes bundled with clear tests.

---
Last updated: 2025-10-03
