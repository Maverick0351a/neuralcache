from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REFERENCES = {
    "ERROR_ENVELOPES.md": ["docs/ERROR_ENVELOPES.md"],  # README references docs/ERROR_ENVELOPES.md explicitly
    "MULTITENANCY.md": ["MULTITENANCY.md"],
    "PRIVACY.md": ["PRIVACY.md"],
    "SECURITY.md": ["SECURITY.md"],
}

def test_referenced_docs_exist():
    missing: list[str] = []
    for label, rel_paths in REFERENCES.items():
        for rel in rel_paths:
            if not (ROOT / rel).exists():
                missing.append(rel)
    assert not missing, f"Missing referenced docs: {missing}"
