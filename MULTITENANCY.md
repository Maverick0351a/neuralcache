# Multi-Tenancy & Namespace Isolation

NeuralCache provides lightweight logical isolation between tenants ("namespaces") so
feedback signals (narrative updates, pheromone exposure, and future adaptive state)
do not bleed across customers or applications sharing a single API deployment.

## Overview

Each request may specify a namespace via the HTTP header:

```
X-NeuralCache-Namespace: <name>
```

If omitted, the `default` namespace is used. A per-namespace `Reranker` instance
is created on first use and retained in-memory for subsequent requests.

| Aspect              | Isolated Per Namespace | Notes |
|---------------------|------------------------|-------|
| Narrative state     | ✅                     | EMA vector & gating success accrual |
| Pheromone tracking  | ✅                     | Exposure/decay history separate |
| Feedback updates    | ✅                     | Selected doc reinforcement scoped |
| CR Index (future)   | ⏳ Planned             | Currently shared; future path: per-namespace index selection |
| Metrics aggregation | 🚧 Partial             | Global metrics currently aggregate all namespaces |

## Namespace Semantics

Namespaces are labels (up to 64 chars) restricted by regex:

```
^[a-zA-Z0-9_.-]{1,64}$
```

Invalid names result in `400 BAD_REQUEST` with error code `BAD_REQUEST` and message
`Invalid namespace`.

## Lifecycle & Memory Considerations

A namespace is instantiated lazily and kept indefinitely. There is no automatic
LRU eviction today. Deployments with highly cardinal tenant identifiers should
introduce an external lifecycle controller or run separate service instances.

### Potential Memory Growth

Each namespace maintains:
- Narrative EMA vector (float32 length = `narrative_dim`)
- Pheromone map (per document ID touched in that namespace)
- Feedback cache references (shared global LRU mapping doc IDs to last rerank batch)

If you expect thousands of ephemeral namespaces, consider:
- Fronting the service with a routing layer that shards namespaces across pods.
- Periodically restarting pods (stateless if persistence disabled) and relying on retention.
- Contributing an eviction strategy (e.g., size + inactivity timeout) upstream.

## Persistence & Retention

Retention sweeping currently scans all instantiated namespace rerankers and applies
narrative/pheromone purging according to `NEURALCACHE_STORAGE_RETENTION_DAYS`.

Persistence paths (`narrative_store_path`, `pheromone_store_path`) remain *shared*.
If persistence is enabled, serialization merges namespace state is NOT yet implemented.
For multi-tenant deployments needing persisted isolation, run separate processes
or disable persistence until namespaced persistence lands.

## Security & Isolation Guarantees

Logical isolation only—data is resident in the same process address space:
- No cross-namespace feedback influence (narrative & pheromones).
- No access control boundary: API tokens/IP ACLs still required if separation of customers matters.
- A memory disclosure vulnerability could still reveal multi-tenant state; use distinct deployments for strong isolation.

## Roadmap

| Feature | Status | Planned Improvement |
|---------|--------|--------------------|
| Namespaced reranker registry | ✅ | Add LRU eviction / idle culling |
| Namespaced persistence | ❌ | Persist per-namespace narrative & pheromones |
| Namespaced metrics | 🚧 | Per-namespace metrics labels / filtering |
| Namespaced CR index | ❌ | Optional per-namespace index load & selection |
| Hard quotas (max namespaces) | ❌ | Enforce upper bound via config |

## Configuration Summary

| Setting | Description | Default |
|---------|-------------|---------|
| `NEURALCACHE_NAMESPACE_HEADER` | Header key to select namespace | `X-NeuralCache-Namespace` |
| `NEURALCACHE_DEFAULT_NAMESPACE` | Namespace when header omitted | `default` |
| `NEURALCACHE_NAMESPACE_PATTERN` | Regex for validation | `^[a-zA-Z0-9_.-]{1,64}$` |

## Examples

### Basic Request

```
POST /rerank
X-NeuralCache-Namespace: tenantA
{
  "query": "improve latency",
  "documents": [{"id":"d1","text":"..."}]
}
```

### Error: Invalid Namespace
```
X-NeuralCache-Namespace: bad/tenant
-> 400 {"error":{"code":"BAD_REQUEST","message":"Invalid namespace"}}
```

## When to Use Separate Deployments Instead
- Regulatory or contractual data isolation requirements.
- Strong security boundaries (e.g., handling confidential embeddings per customer).
- Extremely large or unpredictable namespace cardinality.

## Contributing
Have a need for eviction strategies, metrics partitioning, or persisted isolation? Open an issue or PR with design context.
