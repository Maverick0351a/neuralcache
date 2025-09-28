from __future__ import annotations

import csv
import pathlib
import random
from dataclasses import dataclass

import numpy as np

from neuralcache.gating import make_decision, top_indices_by_similarity


@dataclass
class Row:
    query_id: int
    difficulty: str
    mode: str
    uncertainty: float
    total_candidates: int
    candidate_count: int
    hit_at_10: int
    ndcg_at_10: float


def synthetic_run(n_queries: int = 300, seed: int = 13) -> list[Row]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    rows: list[Row] = []
    for qid in range(n_queries):
        # Difficulty drives distribution sharpness
        difficulty = rng.choice(["easy", "medium", "hard"], weights=[0.4, 0.4, 0.2])
        n_docs = 400

        if difficulty == "easy":
            # clear peak
            sims = np.array(
                [3.0] + [np.abs(np_rng.normal(0.2, 0.2)) for _ in range(n_docs - 1)],
                dtype=np.float64,
            )
        elif difficulty == "medium":
            # semi-peaked
            sims = np.array(
                [1.5] + [np.abs(np_rng.normal(0.8, 0.4)) for _ in range(n_docs - 1)],
                dtype=np.float64,
            )
        else:
            # flat-ish
            sims = np.array([1.0] * n_docs, dtype=np.float64)
            sims += np_rng.normal(0.0, 0.05, size=n_docs)

        # synthetic relevance vector (top doc is "relevant", a few neighbors too)
        rel = np.zeros(n_docs, dtype=float)
        rel[0] = 1.0
        rel[1:6] = 0.5  # neighbors

        # Compare OFF vs AUTO (threshold=0.7)
        for mode in ["off", "auto"]:
            if mode == "off":
                unc = float("nan")
                cand_count = n_docs
                idx = np.argsort(-sims)
            else:
                decision = make_decision(
                    similarities=sims,
                    mode="auto",
                    threshold=0.7,
                    min_candidates=100,
                    max_candidates=400,
                    entropy_temp=1.0,
                )
                unc = decision.uncertainty
                cand_count = decision.candidate_count
                idx_cand = top_indices_by_similarity(sims, cand_count)
                cand_sims = sims[idx_cand]
                order = np.argsort(-cand_sims)
                idx = idx_cand[order]

            top10 = idx[:10]
            hit_at_10 = int(rel[top10].max() > 0.99)
            gains = (2 ** rel[top10] - 1.0)
            discounts = 1.0 / np.log2(np.arange(2, 12))
            ideal = np.sort(rel)[-10:][::-1]
            ideal_gains = 2 ** ideal - 1.0
            ndcg = float(np.sum(gains * discounts) / (np.sum(ideal_gains * discounts) + 1e-9))

            rows.append(
                Row(
                    query_id=qid,
                    difficulty=difficulty,
                    mode=mode,
                    uncertainty=unc,
                    total_candidates=n_docs,
                    candidate_count=cand_count,
                    hit_at_10=hit_at_10,
                    ndcg_at_10=ndcg,
                )
            )
    return rows


def main() -> None:
    reports_dir = pathlib.Path("reports")
    reports_dir.mkdir(exist_ok=True, parents=True)
    rows = synthetic_run(n_queries=600, seed=7)
    out = reports_dir / "gating_eval_synth.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "query_id",
                "difficulty",
                "mode",
                "uncertainty",
                "total_candidates",
                "candidate_count",
                "hit_at_10",
                "ndcg_at_10",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.query_id,
                    r.difficulty,
                    r.mode,
                    f"{r.uncertainty:.4f}" if not np.isnan(r.uncertainty) else "nan",
                    r.total_candidates,
                    r.candidate_count,
                    r.hit_at_10,
                    f"{r.ndcg_at_10:.6f}",
                ]
            )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
