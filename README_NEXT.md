# NeuralCache Add-ons

This drop extends the base NeuralCache scaffold with batch reranking, Prometheus-ready
metrics, durable SQLite persistence, and an evaluation tool for Context-Use@K uplift.
All additions are optional and ship as new modules so your existing integrations remain
unchanged.

## What’s included

- `src/neuralcache/api/server_plus.py` – FastAPI application that mounts the original API
  under `/v1`, adds `POST /rerank/batch`, re-exports the single rerank endpoint, and
  exposes `GET /metrics` (Prometheus text format) when `prometheus-client` is available.
- `src/neuralcache/metrics/` – Prometheus helpers (`prom.py`) plus the lexical metrics
  (`text.py`) previously provided in `metrics.py`.
- `src/neuralcache/storage/sqlite_state.py` – Small, threadsafe SQLite backend for
  persisting the narrative vector and pheromone store across processes.
- `scripts/eval_context_use.py` – Command-line tool that calls a running API, evaluates
  Context-Use@K, and emits a CSV you can share with stakeholders.
- `README_NEXT.md` – This document.

## Setup (Windows PowerShell)

```powershell
# Activate the project environment
.\.venv\Scripts\Activate.ps1

# Install optional extras for metrics and evaluation
pip install prometheus-client requests
```

## Run the enhanced API

```powershell
uvicorn neuralcache.api.server_plus:app --reload --port 8081
```

Endpoints:

- `POST /rerank` – Same semantics as the original endpoint.
- `POST /rerank/batch` – Accepts `{ "requests": [RerankRequest, ...] }`.
- `GET /metrics` – Prometheus exposition format (503 if the client library is missing).
- Legacy endpoints remain available under `/v1`, e.g. `/v1/rerank` and `/v1/healthz`.

## Evaluate Context-Use@K uplift

Prepare a JSONL dataset:

```jsonl
{"query":"What is stigmergy?","docs":[{"id":"1","text":"Stigmergy is indirect coordination."},{"id":"2","text":"Vector DBs store embeddings."}],"answer":"Stigmergy is a mechanism of indirect coordination."}
```

Run the evaluator against your baseline server:

```powershell
python scripts/eval_context_use.py --api http://localhost:8080 --data .\dataset.jsonl --out results_base.csv --top-k 5
```

Compare the enhanced server in one step (baseline + candidate):

```powershell
python scripts/eval_context_use.py --api http://localhost:8080 --compare-api http://localhost:8081 --data .\dataset.jsonl --out results_compare.csv --top-k 5 --compare-endpoint /rerank
```

The script prints an aggregate summary and writes per-query hits (`base_hit`, `nc_hit`) to
CSV so you can benchmark uplift over time.

## SQLite persistence quickstart

```python
from neuralcache.storage.sqlite_state import SQLiteState
import numpy as np

state = SQLiteState("neuralcache.db")

# Narrative
vector = state.load_narrative() or np.zeros(384, dtype=float)
vector[:5] = 0.5
state.save_narrative(vector)

# Pheromones
state.upsert_pheromone("doc:1", value=0.42, add_exposure=1)
state.increment_exposures(["doc:1", "doc:2"])  # record exposures
state.evaporate(half_life_s=3600)
print(state.get_pheromones(["doc:1"]))
```

Wrap your existing `NarrativeTracker` / `PheromoneStore` implementations around
`SQLiteState` to enable multi-process safety without changing public APIs.

## Next steps

1. Swap your in-memory stores for the SQLite backend to share state across processes.
2. Publish `/metrics` to a Prometheus server and visualise request rates / latencies in Grafana.
3. Integrate real embeddings and feed Context-Use@K reports into CI for regression tracking.
