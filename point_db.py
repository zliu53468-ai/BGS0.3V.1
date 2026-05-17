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
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_point_record(player_point: int, banker_point: int) -> Dict[str, Any]:
    db = load_point_db()
    key = f"P{player_point}_B{banker_point}"
    rec = db.get("records", {}).get(key)
    if not rec:
        raise KeyError(f"Point record not found: {key}")
    return rec

def point_db_meta() -> Dict[str, Any]:
    return load_point_db().get("meta", {})
