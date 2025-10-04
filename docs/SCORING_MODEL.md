# NeuralCache Scoring Model

This document specifies the end‑to‑end reranking pipeline used by NeuralCache so results are explainable, reproducible, and externally auditable.

## Pipeline Overview

1. Input Validation
2. Candidate Gating (optional / adaptive)
3. Embedding / Dense Similarity
4. Narrative Memory Update & Contribution
5. Pheromone (Stigmergic) Reinforcement & Decay
6. Linear Score Fusion
7. Exploration Adjustment (ε-greedy)
8. Maximal Marginal Relevance (MMR) Diversification
9. Top-K Truncation / Return Envelope
10. (Later) Feedback Integration for learning step

## Notation
- Let D = {d_i} be the candidate document set after gating.
- Query embedding q ∈ R^n.
- Dense similarity: s_i = cosine(q, emb(d_i)).
- Narrative memory weight for doc i: N_i (EMA of past successes; 0 if unseen).
- Pheromone weight for doc i: P_i (stigmergic signal accumulating with decay).
- Fusion weights (configurable): w_d (dense), w_n (narrative), w_p (pheromone).
- ε ∈ [0,1]: exploration rate (future env var `NEURALCACHE_EPSILON`).

## 1. Candidate Gating
Heuristic / entropy-aware reduction before expensive downstream processing.

Supported modes:
- off: use all provided documents.
- auto (default): adapt candidate count based on gating entropy from similarity distribution.
- on: enforce strict thresholds.

Parameters:
- threshold: minimum preliminary score to retain.
- min_candidates / max_candidates: bounding box for retained set.
- entropy_temp: temperature scaling used in softmax entropy heuristic.

Result: Reduced ordered subset D' used for subsequent stages.

## 2. Dense Similarity
If user provided `query_embedding`, reuse; otherwise encode text. Each doc embedding is computed (or re-used if provided) and cosine similarity s_i is produced. Values may be normalized to [0,1] range internally; the raw value is retained in component breakdown.

## 3. Narrative Memory (N_i)
An exponential moving average updated only after explicit feedback successes. For inference, current N_i is retrieved (0 if absent). This biases toward historically helpful passages.

## 4. Pheromone Signal (P_i)
A stigmergic reinforcement key:
- Incremented when a document is part of a successful context window.
- Decays over wall-clock time \( P_i \leftarrow P_i * e^{-\lambda \Delta t} \) with configurable decay constant.
- Prevents permanent lock-in by decaying stale signals.

## 5. Linear Fusion
Raw pre-exploration score:

```
raw_i = w_d * s_i + w_n * N_i + w_p * P_i
```

All three component contributions are exposed in `components` for transparency.

## 6. Exploration Adjustment (ε-Greedy)
With probability ε a small subset of mid-ranked documents may be promoted (swap with an adjacent higher rank) to gather exploitation vs exploration signal. Deterministic mode (future `NEURALCACHE_DETERMINISTIC=true`) will set ε=0 and disable randomness.

## 7. MMR Diversification
Apply Maximal Marginal Relevance to reorder top candidates balancing relevance vs novelty:

```
MMR = argmax_{d_j ∈ R \ S} [ λ * rel(d_j) - (1-λ) * max_{d_k ∈ S} sim(d_j, d_k) ]
```

where rel(d_j) uses `raw_i` (possibly after exploration), `sim` is cosine between doc embeddings, S is the growing selected set, and λ = `mmr_lambda` (clamped to [0,1]).

## 8. Top-K Truncation
Return first K (user `top_k`) documents after MMR ordering.

## 9. Feedback Loop (Out-of-band)
After user feedback, narrative EMA and pheromones are updated; subsequent queries observe the adjusted N_i and P_i.

## 10. Deterministic Mode (Planned)
When enabled:
- ε forced to 0.
- RNG seeds fixed.
- Time-dependent decay evaluated against a frozen reference timestamp.

This enables reproducible benchmarks.

## Component Exports
`/rerank` debug envelope includes `gating` block and will later expose aggregated weights and exploration actions.

## Future Enhancements
- Export full fusion weights & ε in debug meta.
- Schema-stable telemetry for gating decision rationale.
- Pluggable decay strategies.

## Versioning
This spec corresponds to implementation version 0.3.1+ (post structured response & error codes introduction). Changes will be tracked here and summarized in CHANGELOG entries.
