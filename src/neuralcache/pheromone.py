from __future__ import annotations

import json
import pathlib
import time
from contextlib import suppress


class PheromoneStore:
    """Simple JSON-backed pheromone store with exponential decay and exposure penalty."""

    def __init__(
        self,
        half_life_s: float = 1800.0,
        exposure_penalty: float = 0.1,
        path: str = "pheromones.json",
    ) -> None:
        self.half_life_s = float(half_life_s)
        self.exposure_penalty = float(exposure_penalty)
        self.path = path
        self.data: dict[str, dict[str, float]] = {}  # id -> {value, t, exposures}
        self._load()

    def _load(self) -> None:
        path = pathlib.Path(self.path)
        if not path.exists():
            return

        with suppress(Exception), path.open(encoding="utf-8") as handle:
            self.data = json.load(handle)
        if not isinstance(self.data, dict):
            self.data = {}

    def _save(self) -> None:
        path = pathlib.Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with suppress(Exception), path.open("w", encoding="utf-8") as handle:
            json.dump(self.data, handle)

    def _decay_factor(self, dt: float) -> float:
        if self.half_life_s <= 0:
            return 0.0
        return 0.5 ** (dt / self.half_life_s)

    def get_bonus(self, doc_id: str, now: float | None = None) -> float:
        if doc_id not in self.data:
            return 0.0
        now = time.time() if now is None else now
        rec = self.data[doc_id]
        value = float(rec.get("value", 0.0))
        t = float(rec.get("t", now))
        exposures = float(rec.get("exposures", 0.0))
        # apply time decay
        value *= self._decay_factor(now - t)
        # exposure penalty
        value *= max(0.0, 1.0 - self.exposure_penalty * exposures)
        return value

    def bulk_bonus(self, ids: list[str]) -> list[float]:
        now = time.time()
        return [self.get_bonus(i, now=now) for i in ids]

    def reinforce(self, ids: list[str], reward: float) -> None:
        now = time.time()
        for doc_id in ids:
            rec = self.data.get(doc_id, {"value": 0.0, "t": now, "exposures": 0.0})
            # decay existing value to now
            rec["value"] = float(rec["value"]) * self._decay_factor(now - float(rec["t"]))
            # add reward
            rec["value"] += float(reward)
            rec["t"] = now
            self.data[doc_id] = rec
        self._save()

    def record_exposure(self, ids: list[str]) -> None:
        for doc_id in ids:
            rec = self.data.get(doc_id, {"value": 0.0, "t": time.time(), "exposures": 0.0})
            rec["exposures"] = float(rec.get("exposures", 0.0)) + 1.0
            self.data[doc_id] = rec
        self._save()
