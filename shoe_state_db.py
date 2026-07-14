"""Read and query the 5M baccarat remaining-shoe state database.

No sequence, streak, road, Markov, or momentum feature is used. Queries are
based only on remaining card composition and shoe depth.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import json
import os
import sqlite3
import threading

import numpy as np

DEFAULT_BASELINE = np.asarray([0.458838, 0.4461478, 0.0950142], dtype=float)
DEFAULT_DRAW = np.asarray([0.3785964, 0.1858848, 0.1179186, 0.3176002], dtype=float)


def _normalize(values: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    values = np.maximum(0.0, np.asarray(values, dtype=float))
    total = float(values.sum())
    return values / total if total > 0 else fallback.copy()


def _posterior(counts: np.ndarray, prior: np.ndarray, strength: float) -> np.ndarray:
    prior = _normalize(prior, prior)
    return _normalize(np.asarray(counts, dtype=float) + prior * max(0.0, strength), prior)


def _composition_bucket(actual: int, total: int, baseline_ratio: float) -> int:
    if total <= 0:
        return 2
    expected = total * baseline_ratio
    if expected <= 0:
        return 2
    relative = actual / expected - 1.0
    if relative < -0.08:
        return 0
    if relative < -0.025:
        return 1
    if relative <= 0.025:
        return 2
    if relative <= 0.08:
        return 3
    return 4


def state_key_from_counts(counts: np.ndarray, decks: int = 8) -> tuple[int, int, int, int, int]:
    counts = np.asarray(counts, dtype=int)
    total = int(counts.sum())
    removed = 52 * decks - total
    depth = min(7, max(0, removed // max(1, 6 * decks)))
    zero = int(counts[0])
    low = int(counts[1:4].sum())
    mid = int(counts[4:7].sum())
    high = int(counts[7:10].sum())
    return (
        depth,
        _composition_bucket(zero, total, 16.0 / 52.0),
        _composition_bucket(low, total, 12.0 / 52.0),
        _composition_bucket(mid, total, 12.0 / 52.0),
        _composition_bucket(high, total, 12.0 / 52.0),
    )


@dataclass(frozen=True)
class ShoeStateEstimate:
    probabilities: Dict[str, float]
    draw_paths: Dict[str, float]
    samples: int
    depth_samples: int
    reliability: float
    level: str
    key: tuple[int, int, int, int, int]
    database_available: bool
    hands: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "probabilities": dict(self.probabilities),
            "draw_paths": dict(self.draw_paths),
            "samples": self.samples,
            "depth_samples": self.depth_samples,
            "reliability": self.reliability,
            "level": self.level,
            "key": list(self.key),
            "database_available": self.database_available,
            "hands": self.hands,
        }


class ShoeStateDatabase:
    def __init__(self, path: str | Path | None = None) -> None:
        configured = path or os.getenv("PF_SHOE_DB_PATH", "").strip()
        self.path = Path(configured) if configured else Path(__file__).with_name("shoe_state_5m.sqlite3")
        self.lock = threading.RLock()
        self.loaded = False
        self.available = False
        self.meta: Dict[str, str] = {}
        self.baseline = DEFAULT_BASELINE.copy()
        self.draw_baseline = DEFAULT_DRAW.copy()
        self.outcome_counts = np.zeros((8, 5, 5, 5, 5, 3), dtype=np.int64)
        self.draw_counts = np.zeros((8, 5, 5, 5, 5, 4), dtype=np.int64)
        self.depth_outcome = np.zeros((8, 3), dtype=np.int64)
        self.depth_draw = np.zeros((8, 4), dtype=np.int64)
        self.global_outcome = np.zeros(3, dtype=np.int64)
        self.global_draw = np.zeros(4, dtype=np.int64)

    def _load_array(self, connection: sqlite3.Connection, name: str) -> np.ndarray:
        row = connection.execute(
            "SELECT dtype, shape, data FROM arrays WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            raise KeyError(name)
        dtype, shape_json, blob = row
        shape = tuple(int(v) for v in json.loads(shape_json))
        return np.frombuffer(blob, dtype=np.dtype(dtype)).reshape(shape).copy()

    def load(self) -> None:
        with self.lock:
            if self.loaded:
                return
            self.loaded = True
            if not self.path.exists():
                return
            try:
                connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
                try:
                    self.meta = {str(k): str(v) for k, v in connection.execute("SELECT key,value FROM meta")}
                    self.outcome_counts = self._load_array(connection, "outcome_counts")
                    self.draw_counts = self._load_array(connection, "draw_counts")
                    self.depth_outcome = self._load_array(connection, "depth_outcome")
                    self.depth_draw = self._load_array(connection, "depth_draw")
                    self.global_outcome = self._load_array(connection, "global_outcome")
                    self.global_draw = self._load_array(connection, "global_draw")
                finally:
                    connection.close()
                self.baseline = _normalize(self.global_outcome, DEFAULT_BASELINE)
                self.draw_baseline = _normalize(self.global_draw, DEFAULT_DRAW)
                self.available = True
            except Exception:
                self.available = False

    @property
    def hands(self) -> int:
        self.load()
        try:
            return int(self.meta.get("hands", "0"))
        except Exception:
            return 0

    def database_info(self) -> Dict[str, Any]:
        self.load()
        return {
            "available": self.available,
            "path": str(self.path),
            "hands": self.hands,
            "shoes": int(self.meta.get("shoes", "0") or 0),
            "decks": int(self.meta.get("decks", "0") or 0),
            "description": self.meta.get("description", ""),
            "baseline": {"B": float(self.baseline[0]), "P": float(self.baseline[1]), "T": float(self.baseline[2])},
            "draw_path_baseline": {
                "none": float(self.draw_baseline[0]),
                "player_only": float(self.draw_baseline[1]),
                "banker_only": float(self.draw_baseline[2]),
                "both": float(self.draw_baseline[3]),
            },
        }

    def estimate(self, counts: np.ndarray, decks: int = 8) -> ShoeStateEstimate:
        self.load()
        key = state_key_from_counts(counts, decks)
        depth = key[0]
        if not self.available:
            return ShoeStateEstimate(
                probabilities={"B": float(self.baseline[0]), "P": float(self.baseline[1]), "T": float(self.baseline[2])},
                draw_paths={"none": float(self.draw_baseline[0]), "player_only": float(self.draw_baseline[1]), "banker_only": float(self.draw_baseline[2]), "both": float(self.draw_baseline[3])},
                samples=0,
                depth_samples=0,
                reliability=0.0,
                level="baseline",
                key=key,
                database_available=False,
                hands=self.hands,
            )

        exact_outcome = self.outcome_counts[key]
        exact_draw = self.draw_counts[key]
        exact_samples = int(exact_outcome.sum())
        depth_samples = int(self.depth_outcome[depth].sum())

        outcome = _posterior(self.depth_outcome[depth], self.baseline, 120_000.0)
        draw = _posterior(self.depth_draw[depth], self.draw_baseline, 120_000.0)
        level = "depth"
        if exact_samples > 0:
            outcome = _posterior(exact_outcome, outcome, 700.0)
            draw = _posterior(exact_draw, draw, 700.0)
            level = "exact_shoe_state"
        reliability = exact_samples / (exact_samples + 900.0) if exact_samples > 0 else 0.0

        return ShoeStateEstimate(
            probabilities={"B": float(outcome[0]), "P": float(outcome[1]), "T": float(outcome[2])},
            draw_paths={"none": float(draw[0]), "player_only": float(draw[1]), "banker_only": float(draw[2]), "both": float(draw[3])},
            samples=exact_samples,
            depth_samples=depth_samples,
            reliability=float(max(0.0, min(1.0, reliability))),
            level=level,
            key=key,
            database_available=True,
            hands=self.hands,
        )


_DEFAULT_DB: ShoeStateDatabase | None = None
_DEFAULT_LOCK = threading.RLock()


def get_shoe_state_database() -> ShoeStateDatabase:
    global _DEFAULT_DB
    with _DEFAULT_LOCK:
        if _DEFAULT_DB is None:
            _DEFAULT_DB = ShoeStateDatabase()
        return _DEFAULT_DB
