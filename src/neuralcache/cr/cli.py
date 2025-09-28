from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuralcache.cr.index import build_cr_index, save_cr_index
from neuralcache.embedding import encode_texts


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def build_cr_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a Cognitive Renormalization index")
    parser.add_argument("docs_jsonl", help="Path to JSONL with fields 'id' and 'text'")
    parser.add_argument("--dim", type=int, default=384, help="Base embedding dimension")
    parser.add_argument("--d1", type=int, default=256, help="First PCA dimension")
    parser.add_argument("--d2", type=int, default=64, help="Second PCA dimension")
    parser.add_argument("--k2", type=int, default=16, help="Coarse bucket count")
    parser.add_argument("--k1", type=int, default=12, help="Topic buckets per coarse bucket")
    parser.add_argument("--npz", default="cr_index.npz", help="Output NPZ path")
    parser.add_argument("--meta", default="cr_index.meta.json", help="Output metadata JSON path")
    args = parser.parse_args(argv)

    docs_path = Path(args.docs_jsonl)
    if not docs_path.exists():
        raise SystemExit(f"Docs file not found: {docs_path}")

    docs = _load_jsonl(docs_path)
    if not docs:
        raise SystemExit("Docs JSONL contained no usable records")

    texts = [str(doc.get("text", "")) for doc in docs]
    embeddings = encode_texts(texts, dim=args.dim)
    index = build_cr_index(
        embeddings_q0=embeddings,
        d1=args.d1,
        d2=args.d2,
        k2=args.k2,
        k1_per_bucket=args.k1,
    )
    save_cr_index(index, args.npz, args.meta)
    print(f"Built CR index for {len(texts)} docs \u2192 {args.npz}, {args.meta}")


__all__ = ["build_cr_main"]
