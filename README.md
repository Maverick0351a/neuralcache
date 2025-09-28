<p align="center">
	<img src="assets/Carnot.svg" alt="Carnot Engine" width="280" />
</p>

# NeuralCache — Narrative- & Stigmergy-Aware Reranker for RAG

[![PyPI](https://img.shields.io/pypi/v/neuralcache.svg)](https://pypi.org/project/neuralcache/)
[![CI](https://github.com/Maverick0351a/neuralcache/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Maverick0351a/neuralcache/actions/workflows/ci.yml)
[![Lint](https://github.com/Maverick0351a/neuralcache/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/Maverick0351a/neuralcache/actions/workflows/lint.yml)
[![Docker](https://github.com/Maverick0351a/neuralcache/actions/workflows/docker.yml/badge.svg?branch=main)](https://github.com/Maverick0351a/neuralcache/actions/workflows/docker.yml)
[![CodeQL](https://github.com/Maverick0351a/neuralcache/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/Maverick0351a/neuralcache/actions/workflows/codeql.yml)

NeuralCache is a drop-in reranker for Retrieval-Augmented Generation (RAG) that learns which context the model actually uses—then steers future retrieval toward that context using:

- Dense similarity (semantic closeness)
- Narrative coherence (EMA of successful context; continuity across turns)
- Stigmergic pheromones (exposure-aware reinforcement with decay)
- MMR diversity and ε-greedy exploration to avoid filter bubbles

It plugs into your existing stack (Pinecone/Weaviate/Qdrant, LangChain/LlamaIndex, any embedding model) and improves Context-Use@K with single-digit millisecond overhead at small K.

> This repository open-sources the reranker. The broader “Cognitive Tetrad” engine remains proprietary IP and is not included here.

## Contents

- [Features](#features)
- [Quickstart](#quickstart)
- [API](#api)
- [CLI](#cli)
- [Adapters](#adapters)
- [Configuration](#configuration)
- [Persistence](#persistence)
- [Metrics](#metrics)
- [Repository Layout](#repository-layout)
- [Performance Goals](#performance-goals)
- [Security & Privacy](#security--privacy)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Changelog](#changelog)

## Features

- Rerank API & CLI with clean Pydantic v2 models
- NarrativeTracker (success-gated EMA) for long-horizon coherence
- PheromoneStore with exponential decay & exposure penalty
- MMR + ε-greedy to balance relevance, diversity, and exploration
- Adapters for LangChain & LlamaIndex (import-guarded)
- Ruff (lint+format), mypy, pytest, pre-commit
- Dockerfile and GitHub Actions for CI, tests, releases, and images

## Quickstart

### 1) Install (dev)

```powershell
# Windows PowerShell (VS Code terminal)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .[dev,test]
pre-commit install
pytest
```

### 2) Run the API

```powershell
uvicorn neuralcache.api.server:app --reload --port 8080
```

- Health: <http://localhost:8080/healthz>
- Rerank: `POST /rerank`
- Feedback: `POST /feedback`

### 3) Try the CLI

```powershell
# Prepare a JSONL corpus (one doc per line)
echo {"id":"1","text":"Neural networks learn patterns"} > docs.jsonl
echo {"id":"2","text":"Stigmergy is indirect coordination"} >> docs.jsonl
echo {"id":"3","text":"Vector DBs store embeddings"} >> docs.jsonl

# Rerank
neuralcache "What is stigmergy?" docs.jsonl --top-k 3
```

## API

### `POST /rerank`

Request (excerpt):

```json
{
	"query": "How do stigmergic systems coordinate?",
	"documents": [
		{"id": "d1", "text": "Stigmergy is indirect coordination via shared environment."},
		{"id": "d2", "text": "A vector DB stores embeddings for retrieval."}
	],
	"top_k": 3
}
```

Response (excerpt):

```json
{
	"results": [
		{"id": "d1", "score": 0.87},
		{"id": "d2", "score": 0.32}
	],
	"meta": {
		"strategy": {"mmr": true, "epsilon_greedy": 0.05},
		"weights": {"dense": 0.6, "narrative": 0.3, "pheromone": 0.1}
	}
}
```

### `POST /feedback`

Provide reinforcement signals (e.g., user selected doc contributed to a correct answer):

```json
{
	"query": "What is stigmergy?",
	"selected_doc_ids": ["d1"],
	"success": 1.0
}
```

### `GET /metrics`

Lightweight counters (requests, successes) and placeholder Context-Use@K.

### `GET /healthz`

Liveness check.

## CLI

```powershell
neuralcache "<query>" docs.jsonl --top-k 5 --epsilon 0.05 --mmr-lambda 0.3
```

- Inputs: JSONL with `{"id": "...", "text": "..."}` per line (optional embedding)
- Outputs: Ranked JSON to stdout

## Adapters

Adapters are optional (import-guarded). Install their libs only if you use them.

- LangChain: `NeuralCacheLangChainReranker`
- LlamaIndex: `NeuralCacheLlamaIndexReranker`

## Configuration

All knobs are managed via `pydantic-settings` and/or environment variables (prefix `NC_`). Defaults are in `neuralcache/config.py`.

| Variable | Meaning | Default |
| --- | --- | --- |
| `NC_WEIGHT_DENSE_SIM` | Weight for dense similarity term | `0.6` |
| `NC_WEIGHT_NARRATIVE` | Weight for narrative coherence | `0.3` |
| `NC_WEIGHT_PHEROMONE` | Weight for pheromone bonus | `0.1` |
| `NC_EPSILON_GREEDY` | Exploration probability | `0.05` |
| `NC_MMR_LAMBDA` | MMR trade-off (relevance vs diversity) | `0.3` |
| `NC_NARRATIVE_ALPHA` | EMA update rate (narrative) | `0.005` |
| `NC_PHEROMONE_HALF_LIFE_S` | Decay half-life (seconds) | `86400` |
| `NC_STORAGE_DIR` | Persistence directory | `./` |

Embeddings: By default we include a hashing-trick placeholder for easy local demos. In production, pass real embeddings via your retriever, or integrate a model call in your pipeline.

## Persistence

Narrative and Pheromones persist as SQLite databases or JSON files (fallback) in `NC_STORAGE_DIR`.

For multi-process or multi-instance deployments, keep the shared SQLite directory on durable storage with proper file permissions.

## Metrics

- Context-Use@K proxy (lexical overlap) included
- Prometheus metrics for rerank latency, request rates, and feedback outcomes
- Extend with semantic overlap or OpenTelemetry exporters in production

## Repository Layout

```
neuralcache/
├─ pyproject.toml
├─ README.md
├─ Dockerfile
├─ Makefile
├─ .pre-commit-config.yaml
├─ src/
│  └─ neuralcache/
│     ├─ config.py
│     ├─ types.py
│     ├─ similarity.py
│     ├─ narrative.py
│     ├─ pheromone.py
│     ├─ rerank.py
│     ├─ encoder.py
│     ├─ api/
│     │  └─ server.py
│     ├─ adapters/
│     │  ├─ langchain_adapter.py
│     │  └─ llamaindex_adapter.py
│     └─ cli.py
├─ tests/
│  └─ test_rerank.py
├─ examples/
│  └─ quickstart.py
└─ .github/
	 └─ workflows/
			└─ (CI, lint, tests, release, docker, codeql)
```

## Performance Goals

- p50 < 10 ms for `K ≤ 32` in Python (achievable with vectorized NumPy)
- Optional: move hot loops to Numba or Rust (FFI) if needed

## Security & Privacy

- No documents persist beyond transient rerank inputs by default
- Narrative/pheromone stores contain aggregate signals only (no raw content)
- Run behind your gateway; enable HTTPS and per-tenant isolation for multi-tenant hosting

## Roadmap

- ✅ SQLite persistence (drop-in)
- ✅ Batch `/rerank` endpoint
- ☐ Semantic Context-Use@K metric
- ☐ Prometheus/OpenTelemetry exporter extensions
- ☐ Rust/Numba core (optional acceleration)

## Contributing

```powershell
pip install -e .[dev,test]
pre-commit install
ruff check && mypy && pytest
```

Open a PR with a clear description, tests, and passing CI.

## License

Apache-2.0 for this repository. The Cognitive Tetrad architecture and related research remain proprietary and are not part of this codebase.

## Changelog

Release history and notable changes are tracked in [CHANGELOG.md](CHANGELOG.md).

## CI & Workflows

Copy the workflow files from this README into `.github/workflows/` (create the folder if it does not exist) to enable CI, linting, scheduled tests, releases, and Docker image publishing.

### `ci.yml` — Build, Lint, Type-check, Test (matrix)

```yaml
name: CI

on:
	pull_request:
	push:
		branches: [ main ]

jobs:
	ci:
		runs-on: ubuntu-latest
		strategy:
			matrix:
				python-version: ['3.11', '3.12']
		steps:
			- uses: actions/checkout@v4

			- name: Setup Python
				uses: actions/setup-python@v5
				with:
					python-version: ${{ matrix.python-version }}

			- name: Cache pip
				uses: actions/cache@v4
				with:
					path: ~/.cache/pip
					key: pip-${{ runner.os }}-${{ matrix.python-version }}-${{ hashFiles('pyproject.toml') }}
					restore-keys: |
						pip-${{ runner.os }}-${{ matrix.python-version }}-

			- name: Install
				run: |
					python -m pip install --upgrade pip
					pip install -e .[dev,test]

			- name: Ruff (lint + format check)
				run: ruff check .

			- name: Type-check (mypy)
				run: mypy src

			- name: Pytest
				run: pytest -q --maxfail=1 --disable-warnings --cov=neuralcache --cov-report=xml

			- name: Upload coverage artifact
				uses: actions/upload-artifact@v4
				with:
					name: coverage-xml
					path: coverage.xml
```

### `lint.yml` — Pre-commit on all files (fast feedback)

```yaml
name: Lint

on:
	pull_request:
	push:
		branches: [ main ]

jobs:
	precommit:
		runs-on: ubuntu-latest
		steps:
			- uses: actions/checkout@v4

			- uses: actions/setup-python@v5
				with:
					python-version: '3.11'

			- name: Install
				run: |
					python -m pip install --upgrade pip
					pip install -e .[dev]

			- name: Run pre-commit
				run: |
					pre-commit run --all-files
```

### `tests.yml` — Focused test job with coverage badge

```yaml
name: Tests

on:
	workflow_dispatch:
	schedule:
		- cron: '0 7 * * *'  # daily @ 07:00 UTC

jobs:
	tests:
		runs-on: ubuntu-latest
		steps:
			- uses: actions/checkout@v4

			- uses: actions/setup-python@v5
				with:
					python-version: '3.11'

			- name: Install
				run: |
					python -m pip install --upgrade pip
					pip install -e .[test]

			- name: Pytest
				run: pytest -q --maxfail=1 --disable-warnings --cov=neuralcache --cov-report=xml
```

### `release.yml` — Build & publish to PyPI on tagged release

> Requires `PYPI_API_TOKEN` secret.

```yaml
name: Release

on:
	push:
		tags:
			- 'v*.*.*'

jobs:
	pypi:
		runs-on: ubuntu-latest
		steps:
			- uses: actions/checkout@v4
			- uses: actions/setup-python@v5
				with:
					python-version: '3.11'

			- name: Build sdist & wheel
				run: |
					python -m pip install --upgrade pip build
					python -m build

			- name: Publish to PyPI
				uses: pypa/gh-action-pypi-publish@release/v1
				with:
					password: ${{ secrets.PYPI_API_TOKEN }}
```

### `docker.yml` — Build & push to GHCR (tags & main)

> Ensure workflow permissions grant `packages: write`.

```yaml
name: Docker

on:
	push:
		branches: [ main ]
		tags:
			- 'v*.*.*'

jobs:
	docker:
		runs-on: ubuntu-latest
		permissions:
			contents: read
			packages: write
		steps:
			- uses: actions/checkout@v4

			- name: Login to GHCR
				uses: docker/login-action@v3
				with:
					registry: ghcr.io
					username: ${{ github.actor }}
					password: ${{ secrets.GITHUB_TOKEN }}

			- name: Extract version
				id: meta
				run: |
					REF="${GITHUB_REF##*/}"
					if [[ "$GITHUB_REF" == refs/tags/* ]]; then
						echo "tag=$REF" >> $GITHUB_OUTPUT
					else
						echo "tag=latest" >> $GITHUB_OUTPUT
					fi

			- name: Build & push
				uses: docker/build-push-action@v6
				with:
					context: .
					push: true
					tags: |
						ghcr.io/${{ github.repository_owner }}/neuralcache:${{ steps.meta.outputs.tag }}
						ghcr.io/${{ github.repository_owner }}/neuralcache:latest
```

### `codeql.yml` — Code scanning (optional)

```yaml
name: CodeQL

on:
	push:
		branches: [ main ]
	pull_request:
	schedule:
		- cron: '0 3 * * 6'

jobs:
	analyze:
		permissions:
			security-events: write
			actions: read
			contents: read
		uses: github/codeql-action/.github/workflows/codeql.yml@v3
		with:
			languages: python
```

### `.github/dependabot.yml`

```yaml
version: 2
updates:
	- package-ecosystem: "pip"
		directory: "/"
		schedule:
			interval: "weekly"
		open-pull-requests-limit: 5
	- package-ecosystem: "github-actions"
		directory: "/"
		schedule:
			interval: "weekly"
```

## Repo Setup Checklist

- [ ] Create the workflow YAMLs above in `.github/workflows/`
- [ ] Add `.github/dependabot.yml`
- [ ] Ensure `.pre-commit-config.yaml` exists (Ruff, mypy, EOF fixer, trailing whitespace)
- [ ] Run once locally: `pre-commit run --all-files`
- [ ] Add README badges (they turn green after the first successful runs)
- [ ] Configure secrets: `PYPI_API_TOKEN` (if publishing to PyPI)
- [ ] Optionally enable branch protection requiring CI/Lint/Tests
- [ ] Tag releases `vX.Y.Z` to trigger PyPI & Docker workflows
- [ ] Install dev tooling locally: `pip install -e .[dev,test]`
- [ ] Enable hooks via `pre-commit install`
- [ ] Validate locally with `ruff check && mypy && pytest`
