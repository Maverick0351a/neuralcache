# API Error Envelopes

NeuralCache returns all non-2xx responses in a structured JSON envelope so callers can implement
reliable, forward-compatible error handling. The schema is intentionally small:

```jsonc
{
  "error": {
    "code": "STRING",      // Stable machine-readable code
    "message": "STRING",   // Human readable summary
    "detail": { ... }       // Optional structured context (may be null)
  }
}
```

## Codes

| Code | Typical HTTP | Meaning | Notes |
|------|--------------|---------|-------|
| `BAD_REQUEST` | 400 | Input failed domain validation *after* basic JSON / pydantic parsing | e.g. `top_k` exceeds configured limit |
| `UNAUTHORIZED` | 401 | API token missing/invalid | Only emitted when `NEURALCACHE_API_TOKENS` configured |
| `NOT_FOUND` | 404 | Resource / endpoint disabled | `/metrics` when metrics disabled; feedback docs missing |
| `ENTITY_TOO_LARGE` | 413 | Document text length exceeded configured limit | Uses new `HTTP_413_CONTENT_TOO_LARGE` constant |
| `VALIDATION_ERROR` | 422 | Request failed pydantic model validation | `detail` contains a sanitized list of validation errors |
| `RATE_LIMITED` | 429 | Per-minute request ceiling reached | Driven by `NEURALCACHE_RATE_LIMIT_PER_MINUTE` |
| `INTERNAL_ERROR` | 500 | Unexpected server error | Stack trace logged server-side |

Codes are stable within minor versions. New codes may be added; existing codes will not change
semantic meaning without a major version bump.

## Validation error detail shape

`VALIDATION_ERROR` responses include `detail` as a list of objects derived from FastAPI / pydantic
validation error entries. All nested non-JSON-serializable values are coerced to strings to guarantee
safe serialization (e.g., raw `ValueError` instances).

Example:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Validation failed",
    "detail": [
      {"type": "missing", "loc": ["body", "query"], "msg": "Field required"}
    ]
  }
}
```

## Client handling recommendations

1. Inspect `error.code` first; prefer code over status text matching.
2. Treat unknown codes as retriable only if status is >=500.
3. For `VALIDATION_ERROR`, surface a user-friendly list derived from `detail` entries. Avoid depending
   on internal keys not documented here.
4. For `RATE_LIMITED`, implement exponential backoff capped by a sane upper bound.
5. For `INTERNAL_ERROR`, consider capturing the original request payload (scrub PII) for diagnostics.

## Versioning

The presence and structure of this envelope is guaranteed within a major version. Additional top-level
fields (e.g., `meta`) may appear in the future; clients should ignore unknown keys.

## Related files

- Implementation: `src/neuralcache/api/server.py` (exception handlers)
- Types: `src/neuralcache/types.py` (`ErrorInfo`, `ErrorResponse`)
- Tests exercising envelopes: `tests/test_api_malformed_payloads.py`, `tests/test_api_schema.py`
