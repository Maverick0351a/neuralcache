from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

SqliteValue = dict[str, float]


class SQLiteState:
    """Thread-safe persistence for narrative vectors and pheromone values."""

    def __init__(self, path: str = "neuralcache.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._initialise_schema()

    def _initialise_schema(self) -> None:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS narrative (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    vector TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    updated_ts REAL NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS pheromones (
                    doc_id TEXT PRIMARY KEY,
                    value REAL NOT NULL,
                    ts REAL NOT NULL,
                    exposures REAL NOT NULL
                )
                """
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> SQLiteState:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def save_narrative(self, vector: np.ndarray | list[float]) -> None:
        arr = np.asarray(vector, dtype=float).reshape(-1)
        payload = json.dumps(arr.tolist())
        ts = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO narrative (id, vector, dim, updated_ts)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    vector = excluded.vector,
                    dim = excluded.dim,
                    updated_ts = excluded.updated_ts
                """,
                (payload, int(arr.size), ts),
            )
            self._conn.commit()

    def load_narrative(self) -> np.ndarray | None:
        with self._lock:
            cursor = self._conn.execute("SELECT vector FROM narrative WHERE id = 1")
            row = cursor.fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        return np.array(data, dtype=np.float32)

    def clear_narrative(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM narrative WHERE id = 1")
            self._conn.commit()

    def get_pheromones(self, ids: list[str]) -> dict[str, SqliteValue]:
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        query = (
            "SELECT doc_id, value, ts, exposures FROM pheromones "
            f"WHERE doc_id IN ({placeholders})"
        )
        with self._lock:
            cursor = self._conn.execute(query, ids)
            rows = cursor.fetchall()
        return {
            doc_id: {"value": float(value), "t": float(ts), "exposures": float(exposures)}
            for doc_id, value, ts, exposures in rows
        }

    def upsert_pheromone(
        self,
        doc_id: str,
        value: float,
        timestamp: float | None = None,
        add_exposure: float = 0.0,
    ) -> None:
        ts = time.time() if timestamp is None else float(timestamp)
        with self._lock:
            cursor = self._conn.execute(
                "SELECT value, ts, exposures FROM pheromones WHERE doc_id = ?",
                (doc_id,),
            )
            row = cursor.fetchone()
            exposures = add_exposure
            if row is not None:
                exposures += float(row[2])
            self._conn.execute(
                """
                INSERT INTO pheromones (doc_id, value, ts, exposures)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    value = excluded.value,
                    ts = excluded.ts,
                    exposures = excluded.exposures
                """,
                (doc_id, float(value), ts, float(exposures)),
            )
            self._conn.commit()

    def evaporate(self, half_life_s: float, now: float | None = None) -> None:
        if half_life_s <= 0:
            return
        current_time = time.time() if now is None else float(now)
        with self._lock:
            cursor = self._conn.execute("SELECT doc_id, value, ts FROM pheromones")
            rows = cursor.fetchall()
            for doc_id, value, ts in rows:
                decay = 0.5 ** ((current_time - float(ts)) / half_life_s)
                new_value = float(value) * decay
                self._conn.execute(
                    "UPDATE pheromones SET value = ?, ts = ? WHERE doc_id = ?",
                    (new_value, current_time, doc_id),
                )
            self._conn.commit()

    def increment_exposures(self, ids: list[str], step: float = 1.0) -> None:
        if not ids:
            return
        now = time.time()
        with self._lock:
            for doc_id in ids:
                cursor = self._conn.execute(
                    "SELECT value, ts, exposures FROM pheromones WHERE doc_id = ?",
                    (doc_id,),
                )
                row = cursor.fetchone()
                exposures = float(row[2]) + float(step) if row is not None else float(step)
                value = float(row[0]) if row is not None else 0.0
                self._conn.execute(
                    """
                    INSERT INTO pheromones (doc_id, value, ts, exposures)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(doc_id) DO UPDATE SET
                        value = excluded.value,
                        ts = excluded.ts,
                        exposures = excluded.exposures
                    """,
                    (doc_id, value, now, exposures),
                )
            self._conn.commit()

    def dump_pheromones(self) -> dict[str, SqliteValue]:
        with self._lock:
            cursor = self._conn.execute("SELECT doc_id, value, ts, exposures FROM pheromones")
            rows = cursor.fetchall()
        return {
            doc_id: {"value": float(value), "t": float(ts), "exposures": float(exposures)}
            for doc_id, value, ts, exposures in rows
        }


__all__ = ["SQLiteState"]
