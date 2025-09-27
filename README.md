# NeuralCache

NeuralCache scaffold and examples initialized via automated assistant.

**Narrative + Stigmergic Reranker for RAG** — a lightweight, drop‑in module that improves
Context‑Use@K and Top‑1 retrieval quality by combining dense similarity, narrative coherence
(EMA of successful context), and a stigmergic pheromone field (exposure‑aware, time‑decayed).
Includes an optional MMR diversity term and ε‑greedy exploration to avoid filter bubbles.

> Built for Python 3.11+. API server delivered via FastAPI. Adapters for LangChain/LlamaIndex.
> This scaffold is production‑ready: typing, linting (ruff), tests (pytest), and Docker support.

## Why NeuralCache?
- **Plug‑in** to existing RAG stacks (Pinecone/Weaviate/Qdrant/FAISS) without replacing them.
- **Narrative coherence** helps the model pick context that actually gets used in the answer.
- **Stigmergy** amplifies documents that proved useful recently while decaying stale ones.
- **MMR + ε‑greedy** keep diversity and exploration on by default.

## Quickstart

```bash
# 1) Create a virtual env (Windows PowerShell)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install (dev tools included)
pip install -U pip
pip install -e .[dev,test]

# 3) Set up pre-commit
pre-commit install

# 4) Run tests
pytest

# 5) Launch API
uvicorn neuralcache.api.server:app --reload --port 8080
```

## REST API

- `POST /rerank` — Rerank a list of documents for a query.
- `POST /feedback` — Send success/failure signals to update narrative/pheromones.
- `GET /healthz` — Health probe.
- `GET /metrics` — Basic counters.

See `examples/quickstart.py` for a Python usage example.

## VS Code Tips
- Use the recommended extensions: Python, Ruff, Pylance.
- Enable `black` or `ruff format` on save (we ship `ruff`).

---

© 2025 Carnot Engine — Apache License 2.0.
