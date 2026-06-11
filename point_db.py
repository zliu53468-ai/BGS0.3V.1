import json
import os
from functools import lru_cache
from typing import Dict, Any

from config import POINT_DB_PATH


@lru_cache(maxsize=1)
def load_point_db() -> Dict[str, Any]:
    path = POINT_DB_PATH

    if not os.path.exists(path):
        path = os.path.join(os.getcwd(), path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Point DB file not found: {POINT_DB_PATH}")

    with open(path, "r", encoding="utf-8") as f:
        db = json.load(f)

    if "records" not in db:
        raise ValueError("Point DB format error: missing 'records' field")

    return db


def get_point_record(player_point: int, banker_point: int) -> Dict[str, Any]:
    db = load_point_db()
    key = f"P{player_point}_B{banker_point}"

    rec = db.get("records", {}).get(key)

    if not rec:
        raise KeyError(f"Point record not found: {key}")

    rec = rec.copy()

    if "sample" not in rec and "sample_count" in rec:
        rec["sample"] = rec["sample_count"]

    return rec


def point_db_meta() -> Dict[str, Any]:
    return load_point_db().get("meta", {})
