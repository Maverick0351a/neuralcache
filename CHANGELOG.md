# Changelog

All notable changes to this project will be documented in this file. The format roughly follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0] - 2025-09-28
## [0.3.1] - 2025-10-03
### Added

### Changed
- Configurable epsilon override via `NEURALCACHE_EPSILON` environment variable (validated range 0-1); ignored when deterministic mode is enabled. Exposed effective value in debug envelope as `epsilon_used`.
- Build bootstrap hardening: enforce safe pip range excluding 25.2 (GHSA-4xh5-x5gv-qwph) and updated setuptools minimum
### Fixed
- Prevent accidental install of vulnerable pip version 25.2 by pinning `<25.2`
- Improved alignment between PyPI metadata and README summary

### Notes
- Next minor (0.4.x) will introduce formal API versioning header, deterministic mode, standardized error envelopes, and scoring spec documentation (tracked in issues)
	(Structured envelopes + scoring spec landed early in 0.3.1; versioning header + deterministic mode still pending.)

### Added
- Cognitive gating layer with entropy-aware candidate trimming and configuration overrides
- Expanded API surface via `server_plus` with batch reranking and Prometheus metrics endpoints
- SQLite-backed persistence helpers, CR tooling, evaluation scripts, and tests covering gating behavior

### Changed
- Updated documentation to highlight the Plus API, ops extra, and new operational workflows

### Fixed
- Resolved outstanding lint issues in the retired synthetic demo scaffold

## [0.2.1] - 2025-09-27
### Changed
- Inlined the GitHub CodeQL workflow configuration to restore scanning
- Removed the scheduled tests status badge from the README

## [0.2.0] - 2025-09-24
### Added
- Initial public release with narrative- and stigmergy-aware reranker, API, CLI, and adapters
