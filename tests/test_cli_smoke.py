import json
import tempfile
from pathlib import Path
from neuralcache.cli import rerank as cli_rerank


def test_cli_rerank_basic():
    # Create a temporary JSONL docs file
    docs = [
        {"id": "d1", "text": "Stigmergy enables indirect coordination"},
        {"id": "d2", "text": "Vector databases store embeddings for retrieval"},
        {"id": "d3", "text": "Pheromones reinforce helpful documents"},
    ]
    with tempfile.TemporaryDirectory() as td:
        docs_path = Path(td) / "docs.jsonl"
        with docs_path.open("w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d) + "\n")

        # Directly invoke the command function to avoid Typer CLI parsing edge cases in test env
        # Capture stdout by temporarily redirecting
        import io, sys as _sys
        buf = io.StringIO()
        old_stdout = _sys.stdout
        _sys.stdout = buf
        try:
            cli_rerank("What is stigmergy?", str(docs_path), top_k=2, use_cr=False)
        finally:
            _sys.stdout = old_stdout
        output = buf.getvalue()
        lines = [ln for ln in output.strip().splitlines() if ln]
        assert len(lines) == 2  # requested top_k=2
        # Each line should be valid JSON with required fields
        first = json.loads(lines[0])
        assert {"id", "score", "text"}.issubset(first.keys())
