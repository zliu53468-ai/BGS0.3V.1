"""Side-invariant baccarat road-shape pattern database.

The database learns road *shape*, not absolute Banker/Player dominance.

Context encoding
----------------
S = current non-tie result is the same as the previous non-tie result
C = current non-tie result changed from the previous non-tie result
T = tie (does not replace the previous non-tie side)

Targets
-------
continue_count = next B/P continues the latest non-tie side
switch_count   = next B/P switches away from the latest non-tie side

Examples such as BBBPPBP and PPPBBPB therefore share the same shape context.
This removes direct Banker/Player majority bias from database matching.
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
        side = str(value).upper()
        if side in {"B", "P"}:
            return side
    return None


def encode_shape(history: Sequence[str]) -> str:
    """Encode a B/P/T history into side-invariant S/C/T road shape."""
    output: List[str] = []
    previous_bp: Optional[str] = None
    for raw in history:
        value = str(raw).upper()
        if value == "T":
            if previous_bp is not None:
                output.append("T")
            continue
        if value not in {"B", "P"}:
            continue
        if previous_bp is None:
            previous_bp = value
            continue
        output.append("S" if value == previous_bp else "C")
        previous_bp = value
    return "".join(output)


def continuation_to_bp(
    continue_prob: float,
    switch_prob: float,
    history: Sequence[str],
    first_b_prior: float = 0.5,
) -> Tuple[float, float]:
    side = last_bp_side(history)
    if side is None:
        b = min(0.999, max(0.001, float(first_b_prior)))
        return b, 1.0 - b
    total = max(1e-12, float(continue_prob) + float(switch_prob))
    cont = float(continue_prob) / total
    switch = float(switch_prob) / total
    return (cont, switch) if side == "B" else (switch, cont)


@dataclass(frozen=True)
class PatternLookup:
    probs: Tuple[float, float]  # Runtime B/P probabilities.
    available: bool
    context: str
    order: int
    matches: int
    b_count: int  # Compatibility: mapped B count for the current last side.
    p_count: int  # Compatibility: mapped P count for the current last side.
    status: str
    continue_count: int = 0
    switch_count: int = 0
    continue_prob: float = 0.5
    switch_prob: float = 0.5
    last_side: str = ""
    shape_context: str = ""


class PatternDatabase:
    """Read-only shape lookup over aggregated continuation/switch counts."""

    def __init__(
        self,
        path: str,
        max_order: int = 24,
        min_matches: int = 8,
        smoothing: float = 4.0,
        b_prior: float = 0.5,
    ) -> None:
        self.path = str(path or "").strip()
        self.max_order = max(1, int(max_order))
        self.min_matches = max(1, int(min_matches))
        self.smoothing = max(0.0, float(smoothing))
        self.b_prior = min(0.999, max(0.001, float(b_prior)))
        self._local = threading.local()

    @property
    def exists(self) -> bool:
        return bool(self.path and Path(self.path).is_file())

    def _connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            if not self.exists:
                raise FileNotFoundError(self.path)
            uri = f"file:{Path(self.path).resolve()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=3.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only=ON")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA cache_size=-32000")
            self._local.conn = conn
        return conn

    def lookup(self, history: Sequence[str]) -> PatternLookup:
        fallback = (self.b_prior, 1.0 - self.b_prior)
        if not self.exists:
            return PatternLookup(fallback, False, "", 0, 0, 0, 0, "database_missing")

        cleaned = [str(x).upper() for x in history if str(x).upper() in _ALLOWED]
        shape = encode_shape(cleaned)
        last_side = last_bp_side(cleaned) or ""
        conn = self._connection()
        max_order = min(self.max_order, len(shape))
        contexts = [(shape[-order:], order) for order in range(max_order, 0, -1)]
        contexts.append(("", 0))

        try:
            for context, order in contexts:
                row = conn.execute(
                    "SELECT continue_count, switch_count FROM patterns WHERE context=?",
                    (context,),
                ).fetchone()
                if row is None:
                    continue
                continue_count = int(row["continue_count"] or 0)
                switch_count = int(row["switch_count"] or 0)
                matches = continue_count + switch_count
                if order > 0 and matches < self.min_matches:
                    continue

                alpha = self.smoothing * 0.5
                total = matches + self.smoothing
                continue_prob = (
                    (continue_count + alpha) / total if total > 0 else 0.5
                )
                switch_prob = 1.0 - continue_prob
                b_prob, p_prob = continuation_to_bp(
                    continue_prob, switch_prob, cleaned, self.b_prior
                )
                if last_side == "B":
                    b_count, p_count = continue_count, switch_count
                elif last_side == "P":
                    b_count, p_count = switch_count, continue_count
                else:
                    b_count = p_count = matches // 2

                return PatternLookup(
                    probs=(float(b_prob), float(p_prob)),
                    available=True,
                    context=context,
                    order=order,
                    matches=matches,
                    b_count=b_count,
                    p_count=p_count,
                    status="ready_shape",
                    continue_count=continue_count,
                    switch_count=switch_count,
                    continue_prob=float(continue_prob),
                    switch_prob=float(switch_prob),
                    last_side=last_side,
                    shape_context=context,
                )
        except sqlite3.OperationalError as exc:
            message = str(exc)
            status = (
                "legacy_database_schema_rebuild_required"
                if "continue_count" in message
                else f"sqlite_error:{message}"
            )
            return PatternLookup(fallback, False, "", 0, 0, 0, 0, status)
        except sqlite3.Error as exc:
            return PatternLookup(
                fallback, False, "", 0, 0, 0, 0, f"sqlite_error:{exc}"
            )

        return PatternLookup(fallback, False, "", 0, 0, 0, 0, "no_shape_match")


def initialize_database(path: str, replace_legacy: bool = False) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")

    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(patterns)").fetchall()
    }
    if columns and {"continue_count", "switch_count"} - columns:
        if not replace_legacy:
            conn.close()
            raise RuntimeError(
                "Legacy B/P pattern database detected. Rebuild pattern_10m.sqlite3 "
                "with generate_simulated_baccarat_patterns.py."
            )
        conn.execute("DROP TABLE IF EXISTS patterns")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS patterns (
            context TEXT PRIMARY KEY,
            continue_count INTEGER NOT NULL DEFAULT 0,
            switch_count INTEGER NOT NULL DEFAULT 0
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


def build_database(
    input_path: str,
    db_path: str,
    max_order: int = 24,
    commit_every: int = 5000,
) -> Dict[str, Any]:
    """Build a shape database from real or supplied shoe histories."""
    max_order = max(1, int(max_order))
    commit_every = max(1, int(commit_every))
    conn = initialize_database(db_path, replace_legacy=True)
    upsert = (
        "INSERT INTO patterns(context,continue_count,switch_count) VALUES(?,?,?) "
        "ON CONFLICT(context) DO UPDATE SET "
        "continue_count=continue_count+excluded.continue_count,"
        "switch_count=switch_count+excluded.switch_count"
    )
    shoes = transitions = updates = 0
    batch: Dict[str, List[int]] = {}

    def flush() -> None:
        nonlocal updates
        if not batch:
            return
        conn.executemany(
            upsert,
            ((ctx, counts[0], counts[1]) for ctx, counts in batch.items()),
        )
        updates += len(batch)
        batch.clear()
        conn.commit()

    with open(input_path, "r", encoding="utf-8", errors="ignore") as source:
        for line in source:
            seq = clean_sequence(line)
            if not seq:
                continue
            shoes += 1
            previous_bp: Optional[str] = None
            shape_history = ""
            for target in seq:
                if target == "T":
                    if previous_bp is not None:
                        shape_history += "T"
                    continue
                if target not in {"B", "P"}:
                    continue
                if previous_bp is None:
                    previous_bp = target
                    continue

                is_continue = target == previous_bp
                transitions += 1
                max_here = min(max_order, len(shape_history))
                for order in range(max_here + 1):
                    context = "" if order == 0 else shape_history[-order:]
                    counts = batch.setdefault(context, [0, 0])
                    counts[0 if is_continue else 1] += 1

                shape_history += "S" if is_continue else "C"
                previous_bp = target

            if shoes % commit_every == 0:
                flush()

    flush()
    metadata = {
        "schema": "shape_continue_switch_v2",
        "source_shoes": str(shoes),
        "transitions": str(transitions),
        "max_order": str(max_order),
        "side_invariant": "1",
    }
    conn.executemany(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
        metadata.items(),
    )
    conn.commit()
    contexts = int(conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0])
    conn.close()
    return {
        "ok": True,
        "schema": "shape_continue_switch_v2",
        "source_shoes": shoes,
        "transitions": transitions,
        "contexts": contexts,
        "max_order": max_order,
        "db_path": str(Path(db_path).resolve()),
        "flush_updates": updates,
    }


def database_info(db_path: str) -> Dict[str, Any]:
    conn = sqlite3.connect(db_path)
    metadata = dict(conn.execute("SELECT key,value FROM metadata").fetchall())
    columns = [row[1] for row in conn.execute("PRAGMA table_info(patterns)")]
    contexts = int(conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0])
    root = None
    if {"continue_count", "switch_count"}.issubset(columns):
        root = conn.execute(
            "SELECT continue_count,switch_count FROM patterns WHERE context=''"
        ).fetchone()
    conn.close()
    return {
        "db_path": str(Path(db_path).resolve()),
        "schema_columns": columns,
        "contexts": contexts,
        "root_counts": (
            {"continue": int(root[0]), "switch": int(root[1])} if root else None
        ),
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build/query side-invariant baccarat shape DB"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--input", required=True)
    build.add_argument("--db", required=True)
    build.add_argument("--max-order", type=int, default=24)
    build.add_argument("--commit-every", type=int, default=5000)
    info = sub.add_parser("info")
    info.add_argument("--db", required=True)
    args = parser.parse_args()
    result = (
        build_database(args.input, args.db, args.max_order, args.commit_every)
        if args.command == "build"
        else database_info(args.db)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
