# NeuralCache Documentation Index

Central entry point for the extended documentation set. Each section links to deeper design or operator guidance.

## Core Guides
- [README (Overview & Quickstart)](README.md)
- [Upgrading Notes](README.md#upgrading)
- [Scoring Model Specification](docs/SCORING_MODEL.md)
- [Versioning & API Stability](docs/VERSIONING.md)
- [Error Envelopes](docs/ERROR_ENVELOPES.md)

## Multi-Tenancy & Privacy
- [Multi-Tenancy & Namespaces](MULTITENANCY.md)
- [Privacy & Data Handling](PRIVACY.md)
- [Security Considerations](SECURITY.md)

## Operations & Observability
- Metrics (see Prometheus section in `README.md` and optional namespace labeling)
- Retention sweeper (documented in README retention + gating sections)

## Configuration Reference
All runtime tunables live in `src/neuralcache/config.py` and are surfaced via environment variables (`NEURALCACHE_*`). The README tables summarize the most important knobs; consult the source for authoritative field names and defaults.

## Release History
- [CHANGELOG](CHANGELOG.md)

## Contributing
See sections in `README.md` for development environment, coverage expectations, and roadmap.

---
This index will evolve as new docs land. Open an issue if a link is broken or a conceptual area needs a dedicated page.
