import json
import os
from functools import lru_cache
from typing import Dict, Any, Optional

try:
    from config import POINT_DB_PATH
except Exception:
    POINT_DB_PATH = os.getenv("POINT_DB_PATH", "data/point_db_3m.json")


BASE_BANKER_NO_TIE = 0.5068


def _resolve_path(path: str) -> str:
    """
    支援 Render / 本機兩種路徑：
    1. 直接存在的路徑
    2. 以目前工作目錄為基準的相對路徑
    """
    if os.path.exists(path):
        return path

    cwd_path = os.path.join(os.getcwd(), path)

    if os.path.exists(cwd_path):
        return cwd_path

    return path


@lru_cache(maxsize=1)
def load_point_db() -> Dict[str, Any]:
    path = _resolve_path(POINT_DB_PATH)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Point DB file not found: {POINT_DB_PATH}")

    with open(path, "r", encoding="utf-8") as f:
        db = json.load(f)

    if not isinstance(db, dict):
        raise ValueError("Point DB format error: root must be dict")

    if "records" not in db:
        # 兼容格式：
        # {
        #   "P6_B5": {...},
        #   "P8_B9": {...}
        # }
        db = {
            "meta": db.get("meta", {}) if isinstance(db.get("meta", {}), dict) else {},
            "records": db,
        }

    if not isinstance(db.get("records"), (dict, list)):
        raise ValueError("Point DB format error: records must be dict or list")

    return db


def point_key(player_point: int, banker_point: int) -> str:
    return f"P{int(player_point)}_B{int(banker_point)}"


def normalize_prob_pair(banker: float, player: float):
    banker = float(banker)
    player = float(player)

    # 支援 53 / 47 或 0.53 / 0.47
    if banker > 1:
        banker = banker / 100.0

    if player > 1:
        player = player / 100.0

    total = banker + player

    if total <= 0:
        banker = BASE_BANKER_NO_TIE
        player = 1.0 - banker
    else:
        banker = banker / total
        player = player / total

    return banker, player


def extract_point_record(rec: Dict[str, Any], key: str) -> Dict[str, Any]:
    rec = dict(rec)

    banker = (
        rec.get("next_banker_rate")
        if rec.get("next_banker_rate") is not None
        else rec.get("banker_prob")
        if rec.get("banker_prob") is not None
        else rec.get("banker_rate")
        if rec.get("banker_rate") is not None
        else rec.get("next_banker_prob")
        if rec.get("next_banker_prob") is not None
        else rec.get("banker")
    )

    player = (
        rec.get("next_player_rate")
        if rec.get("next_player_rate") is not None
        else rec.get("player_prob")
        if rec.get("player_prob") is not None
        else rec.get("player_rate")
        if rec.get("player_rate") is not None
        else rec.get("next_player_prob")
        if rec.get("next_player_prob") is not None
        else rec.get("player")
    )

    if banker is None or player is None:
        raise ValueError(f"Point DB record missing banker/player probability: {key}")

    banker_prob, player_prob = normalize_prob_pair(float(banker), float(player))

    sample = (
        rec.get("sample")
        if rec.get("sample") is not None
        else rec.get("sample_count")
        if rec.get("sample_count") is not None
        else rec.get("sample_size")
        if rec.get("sample_size") is not None
        else 0
    )

    try:
        sample = int(sample)
    except Exception:
        sample = 0

    rec["feature_key"] = key
    rec["next_banker_rate"] = banker_prob
    rec["next_player_rate"] = player_prob
    rec["banker_prob"] = banker_prob
    rec["player_prob"] = player_prob
    rec["sample"] = sample
    rec.setdefault("source", "POINT_DB")

    return rec


def find_point_record(player_point: int, banker_point: int) -> Optional[Dict[str, Any]]:
    db = load_point_db()
    records = db.get("records", {})

    key = point_key(player_point, banker_point)

    keys_to_try = [
        key,
        f"{int(player_point)}_{int(banker_point)}",
        f"{int(player_point)}{int(banker_point)}",
    ]

    # 格式 A：
    # records = {
    #   "P6_B5": {...}
    # }
    if isinstance(records, dict):
        for k in keys_to_try:
            rec = records.get(k)

            if isinstance(rec, dict):
                return extract_point_record(rec, k)

    # 格式 B：
    # records = [
    #   {"player_point": 6, "banker_point": 5, ...}
    # ]
    if isinstance(records, list):
        for rec in records:
            if not isinstance(rec, dict):
                continue

            rec_key = str(
                rec.get("feature_key")
                or rec.get("key")
                or rec.get("point_key")
                or ""
            )

            if rec_key in keys_to_try:
                return extract_point_record(rec, rec_key)

            rp = rec.get("player_point", rec.get("p", None))
            rb = rec.get("banker_point", rec.get("b", None))

            try:
                if int(rp) == int(player_point) and int(rb) == int(banker_point):
                    return extract_point_record(rec, key)
            except Exception:
                pass

    return None


def get_point_record(player_point: int, banker_point: int) -> Dict[str, Any]:
    rec = find_point_record(player_point, banker_point)

    if not rec:
        key = point_key(player_point, banker_point)
        raise KeyError(f"Point record not found: {key}")

    return rec


def point_db_meta() -> Dict[str, Any]:
    db = load_point_db()
    meta = db.get("meta", {})

    if not isinstance(meta, dict):
        meta = {}

    records = db.get("records", {})

    if isinstance(records, dict):
        meta.setdefault("record_count", len(records))
    elif isinstance(records, list):
        meta.setdefault("record_count", len(records))
    else:
        meta.setdefault("record_count", 0)

    meta.setdefault("total_simulated_samples", 0)
    meta.setdefault("source", "POINT_DB")

    return meta


def clear_point_db_cache():
    load_point_db.cache_clear()
