from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import requests

from neuralcache.metrics import context_used


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Context-Use@K uplift via the NeuralCache API.",
    )
    parser.add_argument(
        "--api",
        required=True,
        help="Base URL for the target API, e.g. http://localhost:8080",
    )
    parser.add_argument(
        "--endpoint",
        default="/rerank",
        help="Relative endpoint to call for reranking (defaults to /rerank).",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to a JSONL dataset with query, docs, and answer fields.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Destination CSV file for detailed results.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K documents to evaluate for Context-Use@K.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Timeout (seconds) for each HTTP request to the API.",
    )
    parser.add_argument(
        "--compare-api",
        default=None,
        help="Optional second API host to compare (treated as NeuralCache candidate).",
    )
    parser.add_argument(
        "--compare-endpoint",
        default=None,
        help="Optional endpoint override for the comparison API.",
    )
    return parser.parse_args()


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _call_api(
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    timeout: float,
) -> list[dict[str, Any]]:
    url = f"{base_url.rstrip('/')}{endpoint}"
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, list):
        raise ValueError(f"Expected list response, received: {type(body).__name__}")
    return body


def _context_hits(answer: str, docs: list[dict[str, Any]], top_k: int) -> list[bool]:
    top_docs = [doc.get("text", "") for doc in docs[:top_k]]
    return context_used(answer, top_docs)


def main() -> None:
    args = _parse_args()
    dataset = _load_dataset(Path(args.data))
    compare_endpoint = args.compare_endpoint or args.endpoint

    start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    base_hits = 0
    nc_hits = 0

    for idx, record in enumerate(dataset):
        docs = record.get("docs", [])
        if not docs:
            continue
        top_k = min(args.top_k, len(docs))
        payload: dict[str, Any] = {
            "query": record.get("query", ""),
            "documents": docs,
            "top_k": top_k,
        }
        if "query_embedding" in record:
            payload["query_embedding"] = record["query_embedding"]
        answer = record.get("answer", "")

        baseline = _call_api(args.api, args.endpoint, payload, args.timeout)
        baseline_hits = _context_hits(answer, baseline, top_k)
        base_hit = int(any(baseline_hits))
        base_hits += base_hit

        candidate_hit: int | None = None
        candidate_used: int | None = None
        if args.compare_api:
            candidate = _call_api(args.compare_api, compare_endpoint, payload, args.timeout)
            candidate_hits = _context_hits(answer, candidate, top_k)
            candidate_hit = int(any(candidate_hits))
            candidate_used = sum(candidate_hits)
            nc_hits += candidate_hit

        rows.append(
            {
                "idx": idx,
                "query": record.get("query", ""),
                "base_hit": base_hit,
                "base_used": sum(baseline_hits),
                "nc_hit": candidate_hit if candidate_hit is not None else "",
                "nc_used": candidate_used if candidate_used is not None else "",
                "top_k": top_k,
            }
        )

    if not rows:
        print("No evaluable records found in dataset.")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.perf_counter() - start
    total = len(rows)
    summary = (
        f"Eval complete in {elapsed:.2f}s | "
        f"Baseline Context-Use@{args.top_k}: {base_hits}/{total}"
    )
    if args.compare_api:
        summary += f" | NeuralCache: {nc_hits}/{total}"
    print(summary)


if __name__ == "__main__":
    main()
