from __future__ import annotations

import json
import pathlib

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
    use_cr: bool = typer.Option(
        False,
        "--use-cr/--no-cr",
        help="Toggle hierarchical Cognitive Renormalization candidate selection",
    ),
) -> None:
    settings = Settings()
    r = Reranker(settings=settings)
    r.settings.cr.on = use_cr

    # Build query embedding via configured encoder (hashing fallback)
    q = r.encode_query(query)

    docs: list[Document] = []
    with pathlib.Path(docs_file).open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append(
                Document(
                    id=obj["id"],
                    text=obj["text"],
                    metadata=obj.get("metadata", {}),
                    embedding=obj.get("embedding"),
                )
            )

    if len(docs) > settings.max_documents:
        raise typer.BadParameter(
            f"Document count {len(docs)} exceeds limit of {settings.max_documents}",
            param_hint="docs_file",
        )

    if top_k > settings.max_top_k:
        raise typer.BadParameter(
            f"top_k {top_k} exceeds limit of {settings.max_top_k}",
            param_hint="top_k",
        )

    scored = r.score(q, docs, query_text=query)
    for sd in scored[:top_k]:
        typer.echo(json.dumps(sd.model_dump(), ensure_ascii=False))


if __name__ == "__main__":
    app()
