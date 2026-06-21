import json
import os
import time
from typing import Dict, Any, Optional

AI_POINT_FEATURE_DB_PATH = os.getenv("AI_POINT_FEATURE_DB_PATH", "ai_point_feature_db.json").strip()

# 快取控制
_last_load_time: float = 0.0
_cached_data: Optional[Dict[str, Any]] = None
_CACHE_TTL: float = 30.0  # 30 秒後重新讀檔


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


def ai_point_feature_key(player_point: int, banker_point: int) -> str:
    pp = int(player_point)
    bp = int(banker_point)
    diff = pp - bp
    return (
        f"P{pp}_B{bp}"
        f"_D{diff}"
        f"_Z{_diff_zone(diff)}"
        f"_PZ{_point_zone(pp)}"
        f"_BZ{_point_zone(bp)}"
    )


def simple_key(player_point: int, banker_point: int) -> str:
    return f"P{int(player_point)}_B{int(banker_point)}"


def load_ai_point_feature_db() -> Dict[str, Any]:
    """載入 AI 點數特徵資料庫 (含 TTL 快取)"""
    global _last_load_time, _cached_data
    now = time.time()

    if _cached_data is not None and (now - _last_load_time) < _CACHE_TTL:
        return _cached_data

    path = AI_POINT_FEATURE_DB_PATH

    if not os.path.exists(path):
        path = os.path.join(os.getcwd(), path)

    if not os.path.exists(path):
        data = {
            "meta": {
                "source": "AI_POINT_FEATURE_DB_FILE_NOT_FOUND",
                "path": AI_POINT_FEATURE_DB_PATH,
                "record_count": 0,
                "total_simulated_samples": 0,
            },
            "records": {},
        }
        _last_load_time = now
        _cached_data = data
        return data

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("ai_point_feature_db root must be dict")

        data.setdefault("records", {})
        data.setdefault("meta", {})

        # 自動統計 record_count
        records = data.get("records", {})
        if isinstance(records, dict):
            data["meta"]["record_count"] = len(records)

        _last_load_time = now
        _cached_data = data
        return data
    except Exception as e:
        data = {
            "meta": {
                "source": f"AI_POINT_FEATURE_DB_LOAD_ERROR:{e}",
                "path": AI_POINT_FEATURE_DB_PATH,
                "record_count": 0,
                "total_simulated_samples": 0,
            },
            "records": {},
        }
        _last_load_time = now
        _cached_data = data
        return data


def ai_point_feature_db_meta() -> Dict[str, Any]:
    """取得 AI 點數特徵資料庫 meta 資訊"""
    data = load_ai_point_feature_db()
    meta = data.get("meta", {})

    if not isinstance(meta, dict):
        meta = {}

    records = data.get("records", {})
    count = len(records) if isinstance(records, dict) else 0

    meta.setdefault("source", "AI_POINT_FEATURE_DB")
    meta.setdefault("record_count", count)
    meta.setdefault("total_simulated_samples", 0)
    meta.setdefault("path", AI_POINT_FEATURE_DB_PATH)
    return meta


def get_ai_point_feature_record(player_point: int, banker_point: int) -> Optional[Dict[str, Any]]:
    """查詢 AI 點數特徵記錄 (支援多種 key 格式)"""
    data = load_ai_point_feature_db()
    records = data.get("records", {})

    if not isinstance(records, dict):
        return None

    pp = int(player_point)
    bp = int(banker_point)
    full_key = ai_point_feature_key(pp, bp)
    short_key = simple_key(pp, bp)

    for key in (full_key, short_key, f"{pp}_{bp}", f"{pp}{bp}"):
        rec = records.get(key)
        if isinstance(rec, dict):
            out = dict(rec)
            out.setdefault("feature_key", key)
            out.setdefault("source", "AI_POINT_FEATURE_DB")
            return out

    return None


def save_ai_point_feature_record(
    player_point: int,
    banker_point: int,
    record: Dict[str, Any],
    use_full_key: bool = True,
) -> bool:
    """
    寫入或更新一筆 AI 點數特徵記錄。

    Args:
        player_point: 閒家點數 (0-9)
        banker_point: 莊家點數 (0-9)
        record: 記錄內容 (需含 banker_prob, player_prob 等)
        use_full_key: True 使用完整特徵 key, False 使用簡易 key

    Returns:
        寫入成功與否
    """
    data = load_ai_point_feature_db()
    records = data.get("records", {})

    if not isinstance(records, dict):
        records = {}
        data["records"] = records

    pp = int(player_point)
    bp = int(banker_point)

    if use_full_key:
        key = ai_point_feature_key(pp, bp)
    else:
        key = simple_key(pp, bp)

    record["feature_key"] = key
    record.setdefault("source", "AI_POINT_FEATURE_DB")
    records[key] = record

    # 更新 meta
    data["meta"]["record_count"] = len(records)

    try:
        with open(AI_POINT_FEATURE_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 清除全域快取
        global _last_load_time, _cached_data
        _last_load_time = 0.0
        _cached_data = None

        return True
    except Exception:
        return False


def save_ai_point_feature_batch(records_batch: Dict[str, Dict[str, Any]]) -> bool:
    """
    批次寫入多筆記錄。

    Args:
        records_batch: {feature_key: record, ...}

    Returns:
        寫入成功與否
    """
    data = load_ai_point_feature_db()
    records = data.get("records", {})

    if not isinstance(records, dict):
        records = {}
        data["records"] = records

    for key, rec in records_batch.items():
        rec["feature_key"] = key
        rec.setdefault("source", "AI_POINT_FEATURE_DB")
        records[key] = rec

    data["meta"]["record_count"] = len(records)

    try:
        with open(AI_POINT_FEATURE_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        global _last_load_time, _cached_data
        _last_load_time = 0.0
        _cached_data = None

        return True
    except Exception:
        return False


def clear_ai_point_feature_db() -> bool:
    """清空所有記錄 (保留 meta)"""
    data = load_ai_point_feature_db()
    data["records"] = {}
    data["meta"]["record_count"] = 0

    try:
        with open(AI_POINT_FEATURE_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        global _last_load_time, _cached_data
        _last_load_time = 0.0
        _cached_data = None

        return True
    except Exception:
        return False
