# NeuralCache Architectural Cross-Links

This document summarizes how the major adaptive components relate to each other, their governing parameters, and operational concerns so you can reason about trade‑offs when tuning or extending the system.

## Overview Diagram (Conceptual)
```
Incoming Query + Candidates
           │
           ▼
  Dense Similarity (vector encoder)
           │  (similarity.py)
           ▼
  Narrative Memory Adjustment (EMA of scores over time)
           │  (narrative.py)
           ▼
  Pheromone Reinforcement (stigmergic boosts + decay)
           │  (pheromone.py)
           ▼
  Diversity / MMR Re‑ranking (rerank.py)
           │
           ▼
  ε-Greedy Exploration (rerank.py) ── influenced by NEURALCACHE_EPSILON / deterministic
           │
           ▼
        Final Ordered Results
```

## Component Interactions

| Component | Inputs | Mutates State? | Key Params / Env | Downstream Influence | Debug Fields |
|-----------|--------|----------------|------------------|----------------------|--------------|
| Encoder / Similarity | Query + candidate texts | No | Model choice (future), normalization flags | Provides base similarity scores | (future) |
| Narrative Memory | Prior smoothed scores, current similarities | Yes (SQLite / memory) | `narrative_decay`, retention sweep interval | Smooths temporal noise; stabilizes ranks | `narrative_weight` (planned) |
| Pheromones | Feedback events, decay clock | Yes (SQLite / memory) | `pheromone_decay`, feedback weight, retention sweep | Reinforces historically useful docs | `pheromone_boost` (planned) |
| MMR Diversity | Adjusted scores + candidate embeddings | No | `mmr_lambda` (future exposure) | Promotes diversity, combats redundancy | `mmr_lambda_used` (future) |
| ε-Greedy Exploration | Ordered list post-fusion, RNG | No (but uses global RNG) | `epsilon_greedy`, `NEURALCACHE_EPSILON`, `NEURALCACHE_DETERMINISTIC` | Injects stochastic exploration to surface new docs | `epsilon_used`, `deterministic` |
| Retention Sweep | Storage layer timestamps | Yes (prunes) | `NEURALCACHE_STORAGE_RETENTION_SWEEP_INTERVAL_S`, `NEURALCACHE_STORAGE_RETENTION_SWEEP_ON_START` | Prevents unbounded state growth; affects narrative + pheromones | `retention_purged` (future) |

## Cross-Cutting Concerns

### Deterministic Mode
When `NEURALCACHE_DETERMINISTIC=1`, a fixed seed (`NEURALCACHE_DETERMINISTIC_SEED`) initializes RNG used by exploration. Additionally exploration epsilon is forced to 0 regardless of configured or overridden value to guarantee reproducible ordering. Debug output includes `deterministic: true` and `epsilon_used: 0.0`.

### Epsilon Override Precedence
`epsilon_greedy` setting (config / .env) establishes a default. At runtime, if `NEURALCACHE_EPSILON` is set (0–1) AND deterministic mode is off, that value replaces the configured epsilon for that request lifecycle. Invalid values are ignored (fallback to configured). Effective value is surfaced via `epsilon_used` in the debug envelope for transparency.

### Retention Sweep Effects
The background purge thread + optional startup sweep remove aged rows from narrative and pheromone tables. This indirectly alters downstream scoring (older reinforcement and memory contributions vanish). Tune sweep interval to balance freshness vs. historical leverage.

### Feedback Loop Timing
Pheromone reinforcement depends on feedback endpoint updates; narrative memory reflects query frequency. Fast sweeps with slow feedback may emphasize exploration (lower historical boosts). High epsilon further increases chance of discovering fresh items that can later accumulate pheromone reinforcement.

### Diversity vs. Reinforcement
Increasing diversity pressure (future exposed param) can counteract strong pheromone reinforcement to prevent convergence on a narrow set. Monitoring context-use metrics helps calibrate.

## Tuning Strategy Cheat Sheet

| Goal | Increase | Decrease | Notes |
|------|----------|----------|-------|
| Faster adaptation to new relevant docs | epsilon (explore), pheromone learning rate | narrative decay half-life | Raises volatility; watch stability |
| More stable / reproducible rankings | enable deterministic mode, lower epsilon | pheromone decay speed | Determinism disables exploration entirely |
| Prevent stale dominance | pheromone decay rate, sweep frequency | narrative smoothing window | Combine decay + exploration |
| Improve diversity | (future) lower MMR lambda | pheromone weight | Pending explicit parameterization |
| Minimize noise during eval | deterministic mode, strict epsilon=0 | exploration rate | Use for regression tests |

## Future Exposure / Telemetry Wishlist
- Surface MMR lambda & narrative/phero weight contributions in debug
- Optional percentile-based gating stats
- Retention sweep metrics endpoint (purged rows, last sweep timestamp)

## Implementation References
- Re-ranking pipeline orchestration: `src/neuralcache/rerank.py`
- API orchestration & debug envelope: `src/neuralcache/api/server.py`
- Narrative memory logic: `src/neuralcache/narrative.py`
- Pheromone reinforcement logic: `src/neuralcache/pheromone.py`
- Similarity calculations: `src/neuralcache/similarity.py`
- Types / debug schema: `src/neuralcache/types.py`

## Maintenance Notes
When adding a new adaptive component:
1. Define its state handling & retention strategy alignment.
2. Expose any tunable parameters via config + env (document precedence rules).
3. Add debug surface fields for transparency.
4. Update cross-link table above.
5. Provide at least one metric or test capturing its contribution.
