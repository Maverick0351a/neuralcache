# Security Policy

## Supported Versions

We support the most recent minor release line of `neuralcache`. Security fixes are
backported to the latest published version on PyPI. Older minors may receive fixes
on a best-effort basis only if they are no more than one release behind.

| Version | Supported | Notes |
| ------- | --------- | ----- |
| 0.3.x   | ✅        | Receive fixes & security patches |
| < 0.3   | ⚠️       | Best-effort only; please upgrade |

## Reporting a Vulnerability

If you discover a security issue, please email **security@carnotengine.com** with the
following details:

- A clear description of the vulnerability.
- Steps or proof-of-concept required to reproduce the issue.
- The impact you believe the vulnerability has.
- Any suggested fixes or mitigations.

We aim to acknowledge new reports within **3 business days** and will keep you updated
on progress. Please do not open public GitHub issues for potential vulnerabilities.

### Encryption / Sensitive Artifact Guidance

Do not include production embeddings, proprietary documents, or user PII in issue attachments or
test cases. Provide hashes or synthetic surrogates where possible.

## Handling Sensitive Data

NeuralCache can store reranking telemetry in SQLite. When deploying to production:

- Place the SQLite database on encrypted storage.
- Rotate API tokens stored in environment variables regularly.
- Run the API behind TLS (e.g., via a reverse proxy such as Nginx or Caddy).
- Set `NEURALCACHE_API_TOKENS` to enforce bearer-token authentication.

## Dependency Management

- Constrained version ranges (FastAPI, Starlette, Uvicorn) balance stability with timely security
  patch adoption. We widen ranges when upstream APIs stabilize and tests pass.
- Check the [Python Packaging Advisory Database](https://github.com/pypa/advisory-database) before
  broadening constraints.
- The CI workflow runs [`pip-audit`](https://github.com/pypa/pip-audit) on every pull
  request and push to `main` to detect vulnerable dependencies early.
- Use Dependabot updates to stay ahead of transitive dependency advisories.
- Builds harden bootstrap tooling by pinning `pip` away from known vulnerable releases (see
  CHANGELOG for currently excluded versions) and upgrading `setuptools` proactively.

## Secure Development Checklist

- Run `ruff`, `mypy`, and `pytest` locally before sending a pull request.
- Avoid storing secrets in the repository or sample configuration files.
- Keep container builds based on the published Dockerfile up to date with the
  latest security patches from the base image.
- Avoid adding new runtime dependencies without evaluating: supply chain risk, maintenance cadence,
  license compatibility, and CVE history.

## Coordinated Disclosure

We prefer coordinated disclosure. After we release a fix, we'll work with you
on appropriate public communication and attribution if desired. Thank you for
helping us keep NeuralCache safe for everyone.

---
For questions not involving sensitive security matters, open a standard GitHub issue.
