import json
import os
from functools import lru_cache
from typing import Dict, Any, Optional

PATTERN_FEATURE_DB_PATH = os.getenv("PATTERN_FEATURE_DB_PATH", "pattern_feature_db.json").strip()


def _point_zone(point: int) -> str:
    if point <= 2:
        return "LOW"
    if point <= 5:
        return "MID"
    if point <= 7:
        return "HIGH"
    return "TOP"


def _diff_zone(diff: int) -> str:
    ad = abs(diff)
    if ad == 0:
        return "Z"
    if ad <= 2:
        return "S"
    if ad <= 5:
        return "M"
    return "L"


def _last_result(player_point: int, banker_point: int) -> str:
    if player_point > banker_point:
        return "閒"
    if banker_point > player_point:
        return "莊"
    return "和"


def pattern_feature_key(player_point: int, banker_point: int) -> str:
    pp = int(player_point)
    bp = int(banker_point)
    diff = pp - bp
    return (
        f"P{pp}_B{bp}"
        f"_R{_last_result(pp, bp)}"
        f"_D{diff}"
        f"_Z{_diff_zone(diff)}"
        f"_PZ{_point_zone(pp)}"
        f"_BZ{_point_zone(bp)}"
    )


def simple_key(player_point: int, banker_point: int) -> str:
    return f"P{int(player_point)}_B{int(banker_point)}"


@lru_cache(maxsize=1)
def load_pattern_feature_db() -> Dict[str, Any]:
    path = PATTERN_FEATURE_DB_PATH

    if not os.path.exists(path):
        path = os.path.join(os.getcwd(), path)

    if not os.path.exists(path):
        return {
            "meta": {
                "source": "PATTERN_FEATURE_DB_FILE_NOT_FOUND",
                "path": PATTERN_FEATURE_DB_PATH,
                "record_count": 0,
                "total_simulated_samples": 0,
            },
            "records": {},
        }

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("pattern_feature_db root must be dict")

        data.setdefault("records", {})
        data.setdefault("meta", {})
        return data
    except Exception as e:
        return {
            "meta": {
                "source": f"PATTERN_FEATURE_DB_LOAD_ERROR:{e}",
                "path": PATTERN_FEATURE_DB_PATH,
                "record_count": 0,
                "total_simulated_samples": 0,
            },
            "records": {},
        }


def pattern_feature_db_meta() -> Dict[str, Any]:
    data = load_pattern_feature_db()
    meta = data.get("meta", {})

    if not isinstance(meta, dict):
        meta = {}

    records = data.get("records", {})
    count = len(records) if isinstance(records, dict) else 0

    meta.setdefault("source", "PATTERN_FEATURE_DB")
    meta.setdefault("record_count", count)
    meta.setdefault("total_simulated_samples", 0)
    meta.setdefault("path", PATTERN_FEATURE_DB_PATH)
    return meta


def get_pattern_feature_record(player_point: int, banker_point: int) -> Optional[Dict[str, Any]]:
    data = load_pattern_feature_db()
    records = data.get("records", {})

    if not isinstance(records, dict):
        return None

    pp = int(player_point)
    bp = int(banker_point)
    full_key = pattern_feature_key(pp, bp)
    short_key = simple_key(pp, bp)

    for key in (full_key, short_key, f"{pp}_{bp}", f"{pp}{bp}"):
        rec = records.get(key)
        if isinstance(rec, dict):
            out = dict(rec)
            out.setdefault("feature_key", key)
            out.setdefault("source", "PATTERN_FEATURE_DB")
            return out

    return None
