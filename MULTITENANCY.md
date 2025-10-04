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
is created on first use and retained in-memory for subsequent requests. Optional
LRU eviction and namespaced persistence can now constrain memory growth and
provide file-level state isolation.

| Aspect              | Isolated Per Namespace | Notes |
|---------------------|------------------------|-------|
| Narrative state     | ✅                     | EMA vector & gating success accrual |
| Pheromone tracking  | ✅                     | Exposure/decay history separate |
| Feedback updates    | ✅                     | Selected doc reinforcement scoped |
| CR Index (future)   | ⏳ Planned             | Currently shared; future path: per-namespace index selection |
| Metrics aggregation | ✅ (opt-in labels)      | Enable `NEURALCACHE_METRICS_NAMESPACE_LABEL` for per-namespace label |

## Namespace Semantics

Namespaces are labels (up to 64 chars) restricted by regex:

```
^[a-zA-Z0-9_.-]{1,64}$
```

Invalid names result in `400 BAD_REQUEST` with error code `BAD_REQUEST` and message
`Invalid namespace`.

## Lifecycle & Memory Considerations

Namespaces are instantiated lazily. When `NEURALCACHE_MAX_NAMESPACES` is set,
an LRU policy (`NEURALCACHE_NAMESPACE_EVICTION_POLICY=lru`) evicts the least
recently used non-default namespace once the cap is reached. The default
namespace is never evicted.

### Memory Footprint

Each active namespace maintains:
- Narrative EMA vector (float32 length = `narrative_dim`)
- Pheromone map (per document ID touched in that namespace)
- Feedback cache references (shared global LRU mapping doc IDs to last rerank batch)

If you expect very high namespace churn:
- Set `NEURALCACHE_MAX_NAMESPACES` to a reasonable ceiling.
- Tune retention sweep to keep adaptive state trimmed.
- Consider sharding namespaces across pods if cardinality still exceeds operational limits.

## Persistence & Retention

Retention sweeping scans all live namespace rerankers and applies purge rules
based on `NEURALCACHE_STORAGE_RETENTION_DAYS`.

When `NEURALCACHE_NAMESPACED_PERSISTENCE=true`, per-namespace JSON files are
written using templates:
```
NEURALCACHE_NARRATIVE_STORE_TEMPLATE=narrative.{namespace}.json
NEURALCACHE_PHEROMONE_STORE_TEMPLATE=pheromones.{namespace}.json
```
These enable selective archival or deletion of a single tenant's adaptive state.
SQLite mode still provides a shared durable store; namespaced JSON is most useful
when operating in lightweight file persistence or needing file-level separation.

## Security & Isolation Guarantees

Logical isolation only—data is resident in the same process address space:
- No cross-namespace feedback influence (narrative & pheromones).
- No access control boundary: API tokens/IP ACLs still required if separation of customers matters.
- A memory disclosure vulnerability could still reveal multi-tenant state; use distinct deployments for strong isolation.

## Roadmap

| Feature | Status | Planned Improvement |
|---------|--------|--------------------|
| Namespaced reranker registry | ✅ | Potential future: idle timeout eviction |
| Namespaced persistence | ✅ | Potential future: SQLite partitioning per namespace |
| Namespaced metrics | ✅ (labels) | Future: cardinality safeguards & sampling |
| Namespaced CR index | ⏳ Planned | Lazy load CR index per namespace if needed |
| Hard quotas (max namespaces) | ✅ | Future: dual criteria (count + memory) |

## Configuration Summary

| Setting | Description | Default |
|---------|-------------|---------|
| `NEURALCACHE_NAMESPACE_HEADER` | Header key to select namespace | `X-NeuralCache-Namespace` |
| `NEURALCACHE_DEFAULT_NAMESPACE` | Namespace when header omitted | `default` |
| `NEURALCACHE_NAMESPACE_PATTERN` | Regex for validation | `^[a-zA-Z0-9_.-]{1,64}$` |
| `NEURALCACHE_MAX_NAMESPACES` | Max in-memory namespaces (including default) | _unset_ |
| `NEURALCACHE_NAMESPACE_EVICTION_POLICY` | Eviction policy (`lru`) | `lru` |
| `NEURALCACHE_NAMESPACED_PERSISTENCE` | Enable per-namespace JSON stores | `false` |
| `NEURALCACHE_NARRATIVE_STORE_TEMPLATE` | Narrative file template | `narrative.{namespace}.json` |
| `NEURALCACHE_PHEROMONE_STORE_TEMPLATE` | Pheromone file template | `pheromones.{namespace}.json` |
| `NEURALCACHE_METRICS_NAMESPACE_LABEL` | Add `namespace` label to rerank metrics | `false` |

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
