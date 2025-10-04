# Changelog

All notable changes to this project will be documented in this file. The format roughly follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.2] - Unreleased
### Added
- Retention telemetry endpoint `/metrics/retention` exposing sweep counters & timestamps
- `mmr_lambda_default` setting (`NEURALCACHE_MMR_LAMBDA_DEFAULT`) and debug field `mmr_lambda_used`
- Expanded test suite (gating modes, CR path fallback, narrative success gate, feedback error envelope, pheromone JSON persistence, similarity utilities, rerank feedback flows, encoder backend fallbacks, SQLite persistence) raising coverage to 84% and CI gate accordingly.

### Changed
- Migrated startup/shutdown hooks to FastAPI lifespan context (removes deprecation warnings)

### Fixed
- N/A

### Security
- N/A

## [0.3.1] - 2025-10-03
### Added
- Structured API success + error envelopes with standardized error codes
- Scoring pipeline specification (`docs/SCORING_MODEL.md`)
- Sample evaluation dataset (`data/sample_eval.jsonl`) + smoke eval harness
- Response versioning headers + `docs/VERSIONING.md`
- Background retention sweep + startup purge controls
- Deterministic mode flags (`NEURALCACHE_DETERMINISTIC`, `NEURALCACHE_DETERMINISTIC_SEED`) with debug exposure of `deterministic` + `epsilon_used`
- Configurable epsilon override via `NEURALCACHE_EPSILON` environment variable (validated range 0-1); ignored when deterministic mode is enabled. Exposed effective value in debug envelope as `epsilon_used`.

### Changed
- Build bootstrap hardening: enforce safe pip range excluding 25.2 (GHSA-4xh5-x5gv-qwph) and updated setuptools minimum
- Broadened dependency version ranges (FastAPI, Starlette, Uvicorn) reducing upgrade churn
- README enhancements (positioning, retention sweep, deterministic mode)

### Fixed
- Prevent accidental install of vulnerable pip version 25.2 by pinning `<25.2`
- Improved alignment between PyPI metadata and README summary

## [0.3.0] - 2025-09-28
### Added
- Cognitive gating layer with entropy-aware candidate trimming and configuration overrides
- Expanded API surface via `server_plus` with batch reranking and Prometheus metrics endpoints
- SQLite-backed persistence helpers, CR tooling, evaluation scripts, and tests covering gating behavior

### Changed
- Updated documentation to highlight the Plus API, ops extra, and new operational workflows

### Fixed
- Resolved lint issues in retired synthetic demo scaffold

## [0.2.1] - 2025-09-27
### Changed
- Inlined the GitHub CodeQL workflow configuration to restore scanning
- Removed the scheduled tests status badge from the README

## [0.2.0] - 2025-09-24
### Added
- Initial public release with narrative- and stigmergy-aware reranker, API, CLI, and adapters
