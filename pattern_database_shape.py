"""Side-invariant baccarat shape database with B/P continuation, switch and tie.

Schema v3 stores, for every S/C/T shape suffix:
- continue_count: next non-tie result repeated the last non-tie side
- switch_count: next non-tie result changed side
- tie_count: next result was a tie

The runtime never scans millions of rows. It performs indexed suffix lookups,
then Monte Carlo sampling is done from the returned conditional distribution.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ALLOWED = {"B", "P", "T"}


def clean_sequence(value: str) -> List[str]:
    return [ch for ch in str(value).upper() if ch in _ALLOWED]


def last_bp_side(history: Sequence[str]) -> Optional[str]:
    for value in reversed(history):
        value = str(value).upper()
        if value in {"B", "P"}:
            return value
    return None


def encode_shape(history: Sequence[str]) -> str:
    shape: List[str] = []
    previous: Optional[str] = None
    for raw in history:
        value = str(raw).upper()
        if value == "T":
            if previous is not None:
                shape.append("T")
            continue
        if value not in {"B", "P"}:
            continue
        if previous is None:
            previous = value
            continue
        shape.append("S" if value == previous else "C")
        previous = value
    return "".join(shape)


@dataclass(frozen=True)
class PatternLookup:
    available: bool
    status: str
    context: str
    order: int
    matches: int
    continue_count: int
    switch_count: int
    tie_count: int
    continue_prob: float
    switch_prob: float
    tie_prob: float
    b_prob: float
    p_prob: float
    last_side: str


class PatternDatabase:
    def __init__(
        self,
        path: str,
        max_order: int = 16,
        min_matches: int = 20,
        smoothing: float = 6.0,
        b_prior: float = 0.4586,
        p_prior: float = 0.4462,
        t_prior: float = 0.0952,
    ) -> None:
        self.path = str(path or "")
        self.max_order = max(1, int(max_order))
        self.min_matches = max(1, int(min_matches))
        self.smoothing = max(0.0, float(smoothing))
        total = max(1e-12, b_prior + p_prior + t_prior)
        self.b_prior = b_prior / total
        self.p_prior = p_prior / total
        self.t_prior = t_prior / total
        self._local = threading.local()

    @property
    def exists(self) -> bool:
        return Path(self.path).is_file()

    def _connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "connection", None)
        if conn is None:
            if not self.exists:
                raise FileNotFoundError(self.path)
            uri = f"file:{Path(self.path).resolve()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=2.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only=ON")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA cache_size=-16000")
            self._local.connection = conn
        return conn

    def lookup(self, history: Sequence[str]) -> PatternLookup:
        cleaned = [str(x).upper() for x in history if str(x).upper() in _ALLOWED]
        last_side = last_bp_side(cleaned) or ""
        if not self.exists:
            return self._fallback("database_missing", last_side)

        shape = encode_shape(cleaned)
        contexts = [
            (shape[-order:], order)
            for order in range(min(self.max_order, len(shape)), 0, -1)
        ]
        contexts.append(("", 0))

        try:
            conn = self._connection()
            for context, order in contexts:
                row = conn.execute(
                    """
                    SELECT continue_count, switch_count, tie_count
                    FROM patterns WHERE context=?
                    """,
                    (context,),
                ).fetchone()
                if row is None:
                    continue
                c = int(row["continue_count"] or 0)
                s = int(row["switch_count"] or 0)
                t = int(row["tie_count"] or 0)
                matches = c + s + t
                if order > 0 and matches < self.min_matches:
                    continue

                # Symmetric prior for continue/switch; empirical baccarat prior for tie.
                alpha_t = self.smoothing * self.t_prior
                alpha_non_tie = self.smoothing - alpha_t
                alpha_c = alpha_non_tie * 0.5
                alpha_s = alpha_non_tie * 0.5
                denominator = matches + self.smoothing
                cp = (c + alpha_c) / denominator
                sp = (s + alpha_s) / denominator
                tp = (t + alpha_t) / denominator

                if last_side == "B":
                    bp, pp = cp, sp
                elif last_side == "P":
                    bp, pp = sp, cp
                else:
                    non_tie = max(0.0, 1.0 - tp)
                    ratio = self.b_prior / max(1e-12, self.b_prior + self.p_prior)
                    bp, pp = non_tie * ratio, non_tie * (1.0 - ratio)

                return PatternLookup(
                    True, "ready_shape_v3", context, order, matches,
                    c, s, t, cp, sp, tp, bp, pp, last_side
                )
        except sqlite3.OperationalError as exc:
            status = (
                "legacy_schema_rebuild_required"
                if "tie_count" in str(exc)
                else f"sqlite_error:{exc}"
            )
            return self._fallback(status, last_side)
        except sqlite3.Error as exc:
            return self._fallback(f"sqlite_error:{exc}", last_side)

        return self._fallback("no_match", last_side)

    def _fallback(self, status: str, last_side: str) -> PatternLookup:
        return PatternLookup(
            False, status, "", 0, 0, 0, 0, 0,
            0.4524, 0.4524, self.t_prior,
            self.b_prior, self.p_prior, last_side
        )


def initialize_database(path: str, replace_legacy: bool = False) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(patterns)")}
    required = {"context", "continue_count", "switch_count", "tie_count"}
    if columns and not required.issubset(columns):
        if not replace_legacy:
            conn.close()
            raise RuntimeError("Legacy pattern database detected; rebuild required.")
        conn.execute("DROP TABLE IF EXISTS patterns")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS patterns (
            context TEXT PRIMARY KEY,
            continue_count INTEGER NOT NULL DEFAULT 0,
            switch_count INTEGER NOT NULL DEFAULT 0,
            tie_count INTEGER NOT NULL DEFAULT 0
        ) WITHOUT ROWID
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID
        """
    )
    return conn


def database_info(path: str) -> Dict[str, Any]:
    conn = sqlite3.connect(path)
    columns = [r[1] for r in conn.execute("PRAGMA table_info(patterns)")]
    metadata = dict(conn.execute("SELECT key,value FROM metadata").fetchall())
    contexts = int(conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0])
    conn.close()
    return {"path": str(Path(path).resolve()), "columns": columns, "contexts": contexts, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["info"])
    parser.add_argument("--db", required=True)
    args = parser.parse_args()
    print(json.dumps(database_info(args.db), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
