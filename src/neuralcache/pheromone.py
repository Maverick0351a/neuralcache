from __future__ import annotations

import json
import pathlib
import time
from contextlib import suppress

from .storage.sqlite_state import SQLiteState


class PheromoneStore:
    """Durable pheromone store supporting JSON or SQLite backends."""

    def __init__(
        self,
        half_life_s: float = 1800.0,
        exposure_penalty: float = 0.1,
        path: str = "pheromones.json",
        backend: str = "sqlite",
        storage_dir: str | None = None,
        sqlite_state: SQLiteState | None = None,
    ) -> None:
        self.half_life_s = float(half_life_s)
        self.exposure_penalty = float(exposure_penalty)
        base_path = pathlib.Path(storage_dir or ".")
        self.backend = backend.lower()
        self.path = (base_path / path).as_posix()
        self._sqlite = sqlite_state
        self.data: dict[str, dict[str, float]] = {}  # id -> {value, t, exposures}
        if self.backend == "json":
            self._load()

    def _load(self) -> None:
        if self.backend == "memory":
            return
        path = pathlib.Path(self.path)
        if not path.exists():
            return

        with suppress(Exception), path.open(encoding="utf-8") as handle:
            self.data = json.load(handle)
        if not isinstance(self.data, dict):
            self.data = {}

    def _save(self) -> None:
        if self.backend == "sqlite" and self._sqlite is not None:
            return
        if self.backend == "memory":
            return

        path = pathlib.Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with suppress(Exception), path.open("w", encoding="utf-8") as handle:
            json.dump(self.data, handle)
        with suppress(Exception):
            path.chmod(0o600)

    def _decay_factor(self, dt: float) -> float:
        if self.half_life_s <= 0:
            return 0.0
        return 0.5 ** (dt / self.half_life_s)

    def get_bonus(self, doc_id: str, now: float | None = None) -> float:
        now = time.time() if now is None else now
        if self.backend == "sqlite" and self._sqlite is not None:
            rec = self._sqlite.get_pheromones([doc_id]).get(
                doc_id,
                {"value": 0.0, "t": now, "exposures": 0.0},
            )
            value = float(rec.get("value", 0.0))
            t = float(rec.get("t", now))
            exposures = float(rec.get("exposures", 0.0))
            value *= self._decay_factor(now - t)
            value *= max(0.0, 1.0 - self.exposure_penalty * exposures)
            self._sqlite.upsert_pheromone(doc_id, value, timestamp=now, add_exposure=0.0)
            return value

        if doc_id not in self.data:
            return 0.0
        rec = self.data[doc_id]
        value = float(rec.get("value", 0.0))
        t = float(rec.get("t", now))
        exposures = float(rec.get("exposures", 0.0))
        value *= self._decay_factor(now - t)
        value *= max(0.0, 1.0 - self.exposure_penalty * exposures)
        rec["value"] = value
        rec["t"] = now
        self.data[doc_id] = rec
        self._save()
        return value

    def bulk_bonus(self, ids: list[str]) -> list[float]:
        now = time.time()
        if self.backend == "sqlite" and self._sqlite is not None:
            recs = self._sqlite.get_pheromones(ids)
            bonuses: list[float] = []
            for doc_id in ids:
                rec = recs.get(doc_id, {"value": 0.0, "t": now, "exposures": 0.0})
                value = float(rec.get("value", 0.0))
                t = float(rec.get("t", now))
                exposures = float(rec.get("exposures", 0.0))
                value *= self._decay_factor(now - t)
                value *= max(0.0, 1.0 - self.exposure_penalty * exposures)
                self._sqlite.upsert_pheromone(doc_id, value, timestamp=now, add_exposure=0.0)
                bonuses.append(value)
            return bonuses
        return [self.get_bonus(i, now=now) for i in ids]

    def reinforce(self, ids: list[str], reward: float) -> None:
        now = time.time()
        for doc_id in ids:
            if self.backend == "sqlite" and self._sqlite is not None:
                rec = self._sqlite.get_pheromones([doc_id]).get(
                    doc_id,
                    {"value": 0.0, "t": now, "exposures": 0.0},
                )
                value = float(rec.get("value", 0.0))
                t = float(rec.get("t", now))
                value = value * self._decay_factor(now - t) + float(reward)
                self._sqlite.upsert_pheromone(doc_id, value, timestamp=now, add_exposure=0.0)
            else:
                rec = self.data.get(doc_id, {"value": 0.0, "t": now, "exposures": 0.0})
                rec["value"] = float(rec["value"]) * self._decay_factor(now - float(rec["t"]))
                rec["value"] += float(reward)
                rec["t"] = now
                self.data[doc_id] = rec
        if self.backend == "json":
            self._save()

    def record_exposure(self, ids: list[str]) -> None:
        for doc_id in ids:
            if self.backend == "sqlite" and self._sqlite is not None:
                self._sqlite.increment_exposures([doc_id], step=1.0)
            else:
                rec = self.data.get(doc_id, {"value": 0.0, "t": time.time(), "exposures": 0.0})
                rec["exposures"] = float(rec.get("exposures", 0.0)) + 1.0
                self.data[doc_id] = rec
        if self.backend == "json":
            self._save()

    def purge_older_than(self, retention_seconds: float) -> None:
        if retention_seconds <= 0:
            return
        if self.backend == "sqlite" and self._sqlite is not None:
            self._sqlite.purge_older_than(retention_seconds)
            return
        cutoff = time.time() - retention_seconds
        stale_keys = [
            doc_id
            for doc_id, rec in self.data.items()
            if float(rec.get("t", 0.0)) < cutoff
        ]
        for key in stale_keys:
            self.data.pop(key, None)
        if self.backend == "json":
            self._save()
