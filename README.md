<p align="center">
	<img src="assets/Carnot.svg" alt="Carnot Engine" width="280" />
</p>

# NeuralCache 🧠⚡
*Adaptive reranker for Retrieval-Augmented Generation (RAG)*

[![PyPI](https://img.shields.io/pypi/v/neuralcache.svg)](https://pypi.org/project/neuralcache/)
[![CI](https://github.com/Maverick0351a/neuralcache/actions/workflows/ci.yml/badge.svg)](https://github.com/Maverick0351a/neuralcache/actions/workflows/ci.yml)
[![Docker](https://github.com/Maverick0351a/neuralcache/actions/workflows/docker.yml/badge.svg)](https://github.com/Maverick0351a/neuralcache/actions/workflows/docker.yml)
[![CodeQL](https://github.com/Maverick0351a/neuralcache/actions/workflows/codeql.yml/badge.svg)](https://github.com/Maverick0351a/neuralcache/actions/workflows/codeql.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/Maverick0351a/neuralcache?style=social)](https://github.com/Maverick0351a/neuralcache/stargazers)

NeuralCache is a lightweight reranker for RAG pipelines that *actually remembers what helped*. It blends dense semantic similarity with a narrative memory of past wins and stigmergic pheromones that reward helpful passages while decaying stale ones—then spices in MMR diversity and ε-greedy exploration. The result: more relevant context for your LLM without rebuilding your stack.

> This repository open-sources the NeuralCache reranker. The broader “Cognitive Tetrad” engine remains proprietary IP and is not included here.

---

## ⚡ 60-second quickstart

```bash
# 1. Install
pip install neuralcache

# 2. Launch the API (Ctrl+C to stop)
uvicorn neuralcache.api.server:app --port 8080 --reload

# 3. Hit the reranker
curl -s -X POST http://127.0.0.1:8080/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query":"What is stigmergy?",
    "documents":[
      {"id":"a","text":"Stigmergy is indirect coordination via shared context."},
      {"id":"b","text":"Vector DBs store embeddings for retrieval."}
    ],
    "top_k":2
  }' | python -m json.tool
```

Prefer a single command? 👇

```bash
pip install neuralcache && \
uvicorn neuralcache.api.server:app --port 8080 --reload & \
server_pid=$! && sleep 3 && \
curl -s -X POST http://127.0.0.1:8080/rerank -H "Content-Type: application/json" \
     -d '{"query":"What is stigmergy?","documents":[{"id":"a","text":"Stigmergy is indirect coordination."},{"id":"b","text":"Vector DBs store embeddings."}],"top_k":2}' | python -m json.tool && \
kill $server_pid
```

---

## Why teams choose NeuralCache

- **Drop-in reranker** for any retriever that can send JSON. Works with Pinecone, Weaviate, Qdrant, Chroma—or your own Postgres table.
- **Narrative memory (EMA)** keeps track of passages that consistently helped users, biasing future reranks toward them.
- **Stigmergic pheromones** reward useful documents but decay over time, preventing filter bubbles.
- **MMR + ε-greedy** introduces diversity without tanking relevance.
- **Zero external dependencies by default.** Uses a hashing trick for embeddings so you can see results instantly, but slots in any vector model when you’re ready.
- **Adapters included.** LangChain and LlamaIndex adapters ship in `neuralcache.adapters` and only import their extras when you use them.
- **CLI + REST API + FastAPI docs** give you multiple ways to integrate and debug.

### Use cases

- *Customer support copilots* → surface articles with the exact resolution steps.
- *Internal knowledge bases* → highlight documents that past agents actually referenced.
- *Vertical SaaS (legal/health/finance)* → pair compliance-ready snippets with LLM summaries.
- *Evaluation harnesses* → measure and tune Context-Use@K uplift before going live.

---

## How it works

| Signal | What it captures | Why it matters |
| --- | --- | --- |
| **Dense similarity** | Cosine distance over embeddings (hash-based fallback out of the box) | Makes sure obviously relevant passages rank high. |
| **Narrative EMA** | Exponential moving average of successful context windows | Remembers story arcs across multi-turn conversations. |
| **Stigmergic pheromones** | Exposure-aware reinforcement with decay | Rewards docs that helped *recently* while fading stale ones. |
| **MMR diversity** | Maximal Marginal Relevance | Reduces redundancy and surfaces complementary evidence. |
| **ε-greedy exploration** | Occasional exploration of long-tail docs | Keeps fresh signals flowing so the model doesn’t get stuck. |

All of this is orchestrated by `neuralcache.rerank.Reranker`, configurable through [`Settings`](src/neuralcache/config.py) or environment variables (`NEURALCACHE_*`).

---

## Integrations & interfaces

- **REST API** (`uvicorn neuralcache.api.server:app`) with `/rerank`, `/feedback`, `/metrics`, and `/healthz` endpoints.
- **CLI** (`neuralcache "<query>" docs.jsonl --top-k 5`) for quick experiments and scripting.
- **LangChain adapter**: `from neuralcache.adapters import NeuralCacheLangChainReranker`
- **LlamaIndex adapter**: `from neuralcache.adapters import NeuralCacheLlamaIndexReranker`

See [`examples/quickstart.py`](examples/quickstart.py) for an end-to-end script.

---

## Configuration essentials

| Env var | Purpose | Default |
| --- | --- | --- |
| `NEURALCACHE_WEIGHT_DENSE` | Weight on dense similarity | `1.0` |
| `NEURALCACHE_WEIGHT_NARRATIVE` | Weight on narrative memory | `0.6` |
| `NEURALCACHE_WEIGHT_PHEROMONE` | Weight on pheromone signal | `0.3` |
| `NEURALCACHE_MAX_DOCUMENTS` | Safety cap on rerank set size | `128` |
| `NEURALCACHE_MAX_TEXT_LENGTH` | Hard limit on document length (characters) | `8192` |
| `NEURALCACHE_STORAGE_DIR` | Where SQLite + JSON state is stored | `storage/` |

Adjust everything via `.env`, environment variables, or direct `Settings(...)` instantiation.

Persistence happens automatically using SQLite (or JSON fallback) so narrative and pheromone stores survive restarts. Point `NEURALCACHE_STORAGE_DIR` at shared storage for multi-worker deployments.

---

## Evaluation: prove the uplift

We ship `scripts/eval_context_use.py` to measure Context-Use@K on any JSONL dataset (query, docs, answer). It can compare a baseline retriever with a NeuralCache-powered candidate.

```bash
python scripts/eval_context_use.py \
  --api http://localhost:8080 \
  --data data/sample_rag.jsonl \
  --out reports/neuralcache_eval.csv \
  --top-k 5

# Optional: compare against another API host
python scripts/eval_context_use.py \
  --api http://localhost:8000 --data data/sample_rag.jsonl \
  --compare-api http://localhost:8080 --out reports/compare.csv
```

Example output (toy dataset):

```
Eval complete in 4.82s | Baseline Context-Use@5: 9/20 | NeuralCache: 13/20
```

Use the generated CSV to inspect which queries improved, regressions, and latency statistics.

---

## Project layout

```
neuralcache/
├─ assets/                # Logos, diagrams, and other static media
├─ examples/              # Quickstart notebooks and scripts
├─ scripts/               # Evaluation + operational tooling
├─ src/neuralcache/
│  ├─ api/                # FastAPI app exposing REST endpoints
│  ├─ adapters/           # LangChain + LlamaIndex integrations
│  ├─ metrics/            # Context-Use@K helpers & Prometheus hooks
│  ├─ narrative.py        # Narrative memory tracker
│  ├─ pheromone.py        # Pheromone store with decay/exposure logic
│  ├─ rerank.py           # Core reranking orchestrator
│  └─ config.py           # Pydantic Settings (env + .env aware)
├─ tests/                 # Pytest suite (unit + adapter sanity)
└─ .github/workflows/     # CI, lint, release, docker, code scanning
```

---

## Metrics & observability

- `/metrics` exposes Prometheus counters for request volume, success rate, and Context-Use@K proxy.
- Structured logging (via `rich` + standard logging) shows rerank decisions with scores.
- Extend telemetry by dropping in OpenTelemetry exporters or shipping events to your own observability stack.

---

## Roadmap

- ✅ SQLite persistence (drop-in)
- ✅ Batch `/rerank` endpoint
- ✅ LangChain + LlamaIndex adapters
- ☐ Semantic Context-Use@K metric
- ☐ Prometheus/OpenTelemetry exporters
- ☐ Optional Rust / Numba core for hot loops

Have ideas? [Open an issue](https://github.com/Maverick0351a/neuralcache/issues/new/choose) or grab a ticket.

---

## Contributing & community

```bash
pip install -e .[dev,test]
pre-commit install
ruff check && mypy && pytest
```

- Look for [good first issues](https://github.com/Maverick0351a/neuralcache/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
- Add test coverage for user-visible changes.
- PRs with docs, demos, and eval improvements are extra appreciated.

Optionally, join the discussion in **#neuralcache** on Discord (coming soon—watch this space).

---

## License

Apache-2.0. The NeuralCache reranker is open source; the broader Cognitive Tetrad engine remains proprietary.

---

## Automation details

Need to replicate our CI? Expand the sections below for workflow templates.

<details>
<summary><code>.github/workflows/ci.yml</code> — lint, type-check, test</summary>

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
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: pip-${{ runner.os }}-${{ matrix.python-version }}-${{ hashFiles('pyproject.toml') }}
          restore-keys: pip-${{ runner.os }}-${{ matrix.python-version }}-
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

</details>

<details>
<summary><code>.github/workflows/lint.yml</code> — pre-commit</summary>

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
          python-version: "3.11"
      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev]
      - name: Run pre-commit
        run: pre-commit run --all-files
```

</details>

<details>
<summary><code>.github/workflows/tests.yml</code> — scheduled coverage</summary>

```yaml
name: Tests

on:
  workflow_dispatch:
  schedule:
    - cron: "0 7 * * *"  # daily @ 07:00 UTC

jobs:
  tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install -e .[test]
      - name: Pytest
        run: pytest -q --maxfail=1 --disable-warnings --cov=neuralcache --cov-report=xml
```

</details>

<details>
<summary><code>.github/workflows/release.yml</code> — PyPI publish</summary>

```yaml
name: Release

on:
  push:
    tags:
      - "v*.*.*"

jobs:
  pypi:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Build sdist & wheel
        run: |
          python -m pip install --upgrade pip build
          python -m build
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```

</details>

<details>
<summary><code>.github/workflows/docker.yml</code> — GHCR images</summary>

```yaml
name: Docker

on:
  push:
    branches: [ main ]
    tags:
      - "v*.*.*"

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

</details>

<details>
<summary><code>.github/dependabot.yml</code></summary>

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

</details>

---

## Support the project

If NeuralCache saves you time, consider starring the repo or sharing a demo with the community. Contributions, bug reports, and evaluation results are the best way to help the project grow.
