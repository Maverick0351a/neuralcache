# Changelog

All notable changes to this project will be documented in this file. The format roughly follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0] - 2025-09-28
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
