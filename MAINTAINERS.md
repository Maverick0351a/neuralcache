# Maintainers Guide

## Manual Release Process (PyPI)
Automated PyPI workflow has been removed. Follow these steps to cut a release:

1. Update version in `pyproject.toml` and `src/neuralcache/version.py`.
2. Update `CHANGELOG.md` (date + entries) and create a `RELEASE_NOTES_<version>.md`.
3. Commit and tag:
   ```bash
   git add .
   git commit -m "release: vX.Y.Z <summary>"
   git tag vX.Y.Z
   git push origin main --follow-tags
   ```
4. Build artifacts:
   ```bash
   python -m pip install --upgrade pip build twine
   python -m build
   ```
5. Upload:
   ```bash
   export TWINE_USERNAME=__token__
   export TWINE_PASSWORD=$PYPI_API_TOKEN  # or set in shell profile
   python -m twine upload dist/neuralcache-X.Y.Z*
   ```
6. Verify:
   ```bash
   pip install --upgrade neuralcache==X.Y.Z
   python -c "import neuralcache, neuralcache.version; print(neuralcache.version.__version__)"
   ```
7. Draft GitHub release using the release notes.

## Notes
- Keep coverage gate updates incremental to avoid large unreviewable jumps.
- Avoid pushing build artifacts; only ship source + wheels via PyPI.
- For security sensitive dependency bumps, reference advisories in commit messages.

## Contacts
Open a GitHub issue for questions or email hello@carnotengine.com.
