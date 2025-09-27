from __future__ import annotations

import json
import pathlib

import numpy as np
import typer

from .config import Settings
from .rerank import Reranker
from .types import Document

app = typer.Typer(help="NeuralCache CLI")


@app.command()
def rerank(
    query: str = typer.Argument(..., help="User query"),
    docs_file: str = typer.Argument(
        ...,
        help="Path to a JSONL file of documents with fields: id,text",
    ),
    top_k: int = typer.Option(5, help="Top-K results to print"),
) -> None:
    settings = Settings()
    r = Reranker(settings=settings)

    # Build query embedding (hash trick for demo)
    dim = settings.narrative_dim
    q = np.zeros((dim,), dtype=np.float32)
    for tok in query.lower().split():
        q[hash(tok) % dim] += 1.0

    docs: list[Document] = []
    with pathlib.Path(docs_file).open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append(
                Document(
                    id=obj["id"],
                    text=obj["text"],
                    metadata=obj.get("metadata", {}),
                )
            )

    scored = r.score(q, docs)
    for sd in scored[:top_k]:
        typer.echo(json.dumps(sd.model_dump(), ensure_ascii=False))


if __name__ == "__main__":
    app()
