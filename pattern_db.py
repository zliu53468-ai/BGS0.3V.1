import hashlib
import json
import math
import os
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple

import config
from point_db import get_point_record, point_db_meta


# ============================================================
# 環境變數 / config 參數讀取
# ============================================================

def env_bool(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return float(default)


def env_int(name: str, default: str) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return int(default)


POINT_WEIGHT = env_float("POINT_WEIGHT", str(getattr(config, "POINT_WEIGHT", 0.72)))
PATTERN_WEIGHT = env_float("PATTERN_WEIGHT", str(getattr(config, "PATTERN_WEIGHT", 0.18)))
SIM_WEIGHT = env_float("SIM_WEIGHT", str(getattr(config, "SIM_WEIGHT", 0.10)))

MIN_OUTPUT_PROB = env_float("MIN_OUTPUT_PROB", str(getattr(config, "MIN_OUTPUT_PROB", 0.38)))
MAX_OUTPUT_PROB = env_float("MAX_OUTPUT_PROB", str(getattr(config, "MAX_OUTPUT_PROB", 0.62)))
PERCENT_DECIMALS = env_int("PERCENT_DECIMALS", str(getattr(config, "PERCENT_DECIMALS", 2)))

USE_POINT_DB = env_bool("USE_POINT_DB", "1" if getattr(config, "USE_POINT_DB", True) else "0")
USE_PATTERN_DB = env_bool("USE_PATTERN_DB", "1")

POINT_DB_PATH = os.getenv("POINT_DB_PATH", getattr(config, "POINT_DB_PATH", "point_db.json")).strip()
PATTERN_DB_PATH = os.getenv("PATTERN_DB_PATH", getattr(config, "PATTERN_DB_PATH", "pattern_db.json")).strip()


# ============================================================
# 模型基準參數
# ============================================================

# 不含和局時，莊家的長期期望略高於閒
BASE_BANKER_NO_TIE = env_float("BASE_BANKER_NO_TIE", "0.5068")

# 一般局進場門檻
# gap = abs(banker - player)
# 0.035 = 莊閒差距小於 3.5% 不建議進場
MIN_GAP_FOR_ENTRY = env_float("MIN_GAP_FOR_ENTRY", "0.035")

# 強勢局門檻，給前端或文字判斷用
STRONG_GAP_FOR_ENTRY = env_float("STRONG_GAP_FOR_ENTRY", "0.065")


# ============================================================
# 和局點數保護參數
# ============================================================

TIE_AI_MAX_WEIGHT = env_float("TIE_AI_MAX_WEIGHT", "0.02")
TIE_SHRINK = env_float("TIE_SHRINK", "0.30")
TIE_MIN_GAP_FOR_ENTRY = env_float("TIE_MIN_GAP_FOR_ENTRY", "0.08")


# ============================================================
# 工具函式
# ============================================================

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def stable_noise(key: str, scale: float = 0.035) -> float:
    """
    穩定雜訊：
    同一組 key 永遠得到同一個偏移值。
    不是隨機亂數，不會每次跳不同結果。
    """
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    raw = int(digest[:8], 16) / 0xFFFFFFFF
    return (raw - 0.5) * 2 * scale


def get_last_result(player_point: int, banker_point: int) -> str:
    if player_point > banker_point:
        return "閒"

    if banker_point > player_point:
        return "莊"

    return "和"


def validate_point(v: int) -> int:
    iv = int(v)

    if iv < 0 or iv > 9:
        raise ValueError("point must be 0-9")

    return iv


def point_zone(point: int) -> str:
    if point <= 2:
        return "LOW"

    if point <= 5:
        return "MID"

    if point <= 7:
        return "HIGH"

    return "TOP"


def diff_zone(diff: int) -> str:
    ad = abs(diff)

    if ad == 0:
        return "Z"

    if ad <= 2:
        return "S"

    if ad <= 5:
        return "M"

    return "L"


def feature_key(player_point: int, banker_point: int) -> str:
    """
    規律特徵 key。
    這個 key 是用來查 pattern_db 的核心。

    例：
    閒 6、莊 5
    P6_B5_R閒_D1_ZS_PZHIGH_BZMID
    """
    diff = player_point - banker_point

    return (
        f"P{player_point}_B{banker_point}"
        f"_R{get_last_result(player_point, banker_point)}"
        f"_D{diff}"
        f"_Z{diff_zone(diff)}"
        f"_PZ{point_zone(player_point)}"
        f"_BZ{point_zone(banker_point)}"
    )


def simple_point_key(player_point: int, banker_point: int) -> str:
    return f"P{player_point}_B{banker_point}"


def normalize_prob_pair(banker: float, player: float) -> Tuple[float, float]:
    """
    將 banker / player 機率轉成 0~1，並正規化。
    支援資料庫裡存 0.53 或 53 兩種格式。
    """

    banker = float(banker)
    player = float(player)

    if banker > 1.0:
        banker = banker / 100.0

    if player > 1.0:
        player = player / 100.0

    total = banker + player

    if total <= 0:
        banker = BASE_BANKER_NO_TIE
        player = 1.0 - banker
    else:
        banker = banker / total
        player = player / total

    banker = clamp(banker, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)
    player = 1.0 - banker

    return banker, player


def neutral_record(source: str = "NEUTRAL_FALLBACK") -> Dict[str, Any]:
    banker = clamp(BASE_BANKER_NO_TIE, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)

    return {
        "available": False,
        "feature_key": "NEUTRAL",
        "banker_prob": banker,
        "player_prob": 1.0 - banker,
        "source": source,
        "sample_size": 0,
        "total_simulated_samples": 0,
    }


# ============================================================
# POINT DB 查詢
# ============================================================

def fallback_point_lookup(player_point: int, banker_point: int) -> Dict[str, Any]:
    """
    point_db 不可用時的保底規則。
    這層只負責避免系統掛掉，不建議當正式主模型。
    """

    diff = player_point - banker_point
    key = feature_key(player_point, banker_point)

    banker = BASE_BANKER_NO_TIE

    if diff == 0:
        banker += stable_noise(key + ":tie", 0.018)
    elif 1 <= diff <= 2:
        banker -= 0.185
    elif 3 <= diff <= 5:
        banker += 0.185
    elif diff >= 6:
        banker += 0.115
    elif -2 <= diff <= -1:
        banker += 0.185
    elif -5 <= diff <= -3:
        banker -= 0.185
    elif diff <= -6:
        banker -= 0.115

    banker += stable_noise(key + ":fallback", 0.045)
    banker = clamp(banker, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)

    return {
        "available": False,
        "feature_key": key,
        "banker_prob": banker,
        "player_prob": 1.0 - banker,
        "source": "FALLBACK_POINT_RULE_ONLY",
        "sample_size": 0,
        "total_simulated_samples": 0,
    }


def point_db_lookup(player_point: int, banker_point: int) -> Dict[str, Any]:
    """
    主要點數資料庫查詢：
    根據輸入點數 P/B 直接查 point_db。
    """

    if not USE_POINT_DB:
        return fallback_point_lookup(player_point, banker_point)

    try:
        rec = get_point_record(player_point, banker_point)

        banker = rec.get("next_banker_rate", rec.get("banker_prob", rec.get("banker_rate", None)))
        player = rec.get("next_player_rate", rec.get("player_prob", rec.get("player_rate", None)))

        if banker is None or player is None:
            return fallback_point_lookup(player_point, banker_point)

        banker, player = normalize_prob_pair(float(banker), float(player))

        meta = point_db_meta()

        return {
            "available": True,
            "feature_key": simple_point_key(player_point, banker_point),
            "banker_prob": banker,
            "player_prob": player,
            "source": rec.get("source", "POINT_DB"),
            "sample_size": int(rec.get("sample", rec.get("sample_size", 0)) or 0),
            "total_simulated_samples": int(meta.get("total_simulated_samples", 0) or 0),
        }

    except Exception:
        return fallback_point_lookup(player_point, banker_point)


# ============================================================
# PATTERN DB 查詢
# ============================================================

@lru_cache(maxsize=1)
def load_pattern_db_file() -> Dict[str, Any]:
    """
    支援 pattern_db.json 多種格式：

    格式 A：
    {
      "records": {
        "P6_B5_R閒_D1_ZS_PZHIGH_BZMID": {
          "next_banker_rate": 0.53,
          "next_player_rate": 0.47,
          "sample": 1200
        }
      }
    }

    格式 B：
    {
      "records": [
        {
          "feature_key": "P6_B5_R閒_D1_ZS_PZHIGH_BZMID",
          "next_banker_rate": 0.53,
          "next_player_rate": 0.47,
          "sample": 1200
        }
      ]
    }

    格式 C：
    {
      "P6_B5_R閒_D1_ZS_PZHIGH_BZMID": {
        "banker_prob": 53,
        "player_prob": 47
      }
    }
    """

    path = PATTERN_DB_PATH

    if not os.path.exists(path):
        path = os.path.join(os.getcwd(), path)

    if not os.path.exists(path):
        return {
            "records": {},
            "meta": {
                "source": "PATTERN_DB_FILE_NOT_FOUND",
                "path": PATTERN_DB_PATH,
                "total_simulated_samples": 0,
            }
        }

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return {
                "records": {},
                "meta": {
                    "source": "PATTERN_DB_FORMAT_ERROR",
                    "path": PATTERN_DB_PATH,
                    "total_simulated_samples": 0,
                }
            }

        return data

    except Exception as e:
        return {
            "records": {},
            "meta": {
                "source": f"PATTERN_DB_LOAD_ERROR:{e}",
                "path": PATTERN_DB_PATH,
                "total_simulated_samples": 0,
            }
        }


def try_import_pattern_db_record(player_point: int, banker_point: int, fkey: str) -> Optional[Dict[str, Any]]:
    """
    優先嘗試使用 pattern_db.py 裡面的 get_pattern_record。
    為了兼容你現有專案，這裡支援多種函式寫法：

    get_pattern_record(player_point, banker_point)
    get_pattern_record(feature_key)
    get_pattern_record(player_point=..., banker_point=..., feature_key=...)
    """

    try:
        import pattern_db
    except Exception:
        return None

    fn = getattr(pattern_db, "get_pattern_record", None)

    if not callable(fn):
        return None

    call_styles = [
        lambda: fn(player_point, banker_point),
        lambda: fn(fkey),
        lambda: fn(player_point=player_point, banker_point=banker_point, feature_key=fkey),
        lambda: fn(feature_key=fkey),
    ]

    for call in call_styles:
        try:
            rec = call()

            if isinstance(rec, dict):
                return rec

        except Exception:
            continue

    return None


def find_record_in_pattern_json(player_point: int, banker_point: int, fkey: str) -> Optional[Dict[str, Any]]:
    data = load_pattern_db_file()

    records = data.get("records", data)

    keys_to_try = [
        fkey,
        simple_point_key(player_point, banker_point),
        f"P{player_point}_B{banker_point}",
        f"{player_point}_{banker_point}",
        f"{player_point}{banker_point}",
    ]

    # records 是 dict 格式
    if isinstance(records, dict):
        for key in keys_to_try:
            rec = records.get(key)

            if isinstance(rec, dict):
                rec = dict(rec)
                rec.setdefault("feature_key", key)
                return rec

    # records 是 list 格式
    if isinstance(records, list):
        for rec in records:
            if not isinstance(rec, dict):
                continue

            rec_key = str(
                rec.get("feature_key")
                or rec.get("key")
                or rec.get("pattern_key")
                or rec.get("point_key")
                or ""
            )

            if rec_key in keys_to_try:
                return rec

            rp = rec.get("player_point", rec.get("p", None))
            rb = rec.get("banker_point", rec.get("b", None))

            try:
                if int(rp) == int(player_point) and int(rb) == int(banker_point):
                    return rec
            except Exception:
                pass

    return None


def extract_prob_from_pattern_record(rec: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """
    從 pattern_db record 取出 banker/player 機率。
    支援多種欄位名稱。
    """

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
        return None

    try:
        return normalize_prob_pair(float(banker), float(player))
    except Exception:
        return None


def pattern_db_meta_safe() -> Dict[str, Any]:
    try:
        import pattern_db
        fn = getattr(pattern_db, "pattern_db_meta", None)

        if callable(fn):
            meta = fn()

            if isinstance(meta, dict):
                return meta
    except Exception:
        pass

    data = load_pattern_db_file()
    meta = data.get("meta", {})

    if not isinstance(meta, dict):
        meta = {}

    records = data.get("records", {})

    if isinstance(records, dict):
        count = len(records)
    elif isinstance(records, list):
        count = len(records)
    else:
        count = 0

    meta.setdefault("total_simulated_samples", 0)
    meta.setdefault("record_count", count)
    meta.setdefault("source", "PATTERN_DB_JSON")

    return meta


def pattern_db_lookup(player_point: int, banker_point: int) -> Dict[str, Any]:
    """
    真正的 pattern_db 查詢。

    重點：
    這裡不再借用 point_db。
    找不到 pattern_db 時，回中性值 BASE_BANKER_NO_TIE。
    """

    fkey = feature_key(player_point, banker_point)

    if not USE_PATTERN_DB:
        rec = neutral_record("PATTERN_DB_DISABLED")
        rec["feature_key"] = fkey
        return rec

    rec = try_import_pattern_db_record(player_point, banker_point, fkey)

    if rec is None:
        rec = find_record_in_pattern_json(player_point, banker_point, fkey)

    if not isinstance(rec, dict):
        neutral = neutral_record("PATTERN_DB_NEUTRAL_FALLBACK")
        neutral["feature_key"] = fkey
        return neutral

    probs = extract_prob_from_pattern_record(rec)

    if probs is None:
        neutral = neutral_record("PATTERN_DB_RECORD_NO_PROB_FALLBACK")
        neutral["feature_key"] = fkey
        return neutral

    banker, player = probs
    meta = pattern_db_meta_safe()

    return {
        "available": True,
        "feature_key": rec.get("feature_key", rec.get("key", fkey)),
        "banker_prob": banker,
        "player_prob": player,
        "source": rec.get("source", "PATTERN_DB"),
        "sample_size": int(rec.get("sample", rec.get("sample_size", 0)) or 0),
        "total_simulated_samples": int(meta.get("total_simulated_samples", 0) or 0),
    }


# ============================================================
# AI 模擬層
# ============================================================

def ai_simulation_layer(player_point: int, banker_point: int) -> Dict[str, Any]:
    """
    本地 AI 修正層。
    注意：這層不是主模型，只做小幅修正。
    """

    diff = player_point - banker_point
    key = feature_key(player_point, banker_point)

    x = 0.0

    x += -0.055 * diff

    if abs(diff) in {1, 2}:
        x += -0.16 if diff > 0 else 0.16
    elif abs(diff) in {3, 4, 5}:
        x += 0.16 if diff > 0 else -0.16
    elif abs(diff) >= 6:
        x += 0.09 if diff > 0 else -0.09

    x += stable_noise(key + ":ai", 0.11)

    banker = 1.0 / (1.0 + math.exp(-x))
    banker = 0.15 + banker * 0.70
    banker = clamp(banker, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)

    return {
        "banker_prob": banker,
        "player_prob": 1.0 - banker,
        "source": "LOCAL_AI_SIMULATION_POINT_FEATURE",
    }


# ============================================================
# 和局保護
# ============================================================

def apply_tie_point_protection(banker: float, is_tie_point: bool) -> float:
    """
    上一局點數為和局時，把結果往基準值收斂。
    避免 66 / 77 / 88 這種點數被 AI 或 DB 偏移拉太誇張。
    """

    if not is_tie_point:
        return banker

    return BASE_BANKER_NO_TIE + (banker - BASE_BANKER_NO_TIE) * TIE_SHRINK


# ============================================================
# 進場判斷
# ============================================================

def build_entry_decision(is_tie_point: bool, gap: float, recommend: str) -> Tuple[bool, str, str]:
    """
    回傳：
    entry_allowed: 是否建議進場
    entry_level: no_entry / normal / strong
    weak_reason: 原因文字
    """

    if is_tie_point and gap < TIE_MIN_GAP_FOR_ENTRY:
        return (
            False,
            "no_entry",
            "上一局為和局點數，莊閒優勢不足，建議觀察一局"
        )

    if gap < MIN_GAP_FOR_ENTRY:
        return (
            False,
            "no_entry",
            "莊閒機率差距不足，建議觀察一局"
        )

    if gap >= STRONG_GAP_FOR_ENTRY:
        return (
            True,
            "strong",
            ""
        )

    return (
        True,
        "normal",
        ""
    )


# ============================================================
# 主預測函式
# ============================================================

def predict(player_point: int, banker_point: int, rounds: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    主模型：

    不吃使用者前端歷史路紙。
    只使用當前輸入點數：
    1. 查 point_db
    2. 查 pattern_db
    3. AI 模擬微調
    4. 和局保護
    5. 進場判斷
    """

    player_point = validate_point(player_point)
    banker_point = validate_point(banker_point)

    last_result = get_last_result(player_point, banker_point)
    is_tie_point = player_point == banker_point

    point = point_db_lookup(player_point, banker_point)
    pattern = pattern_db_lookup(player_point, banker_point)
    ai = ai_simulation_layer(player_point, banker_point)

    p_w = float(POINT_WEIGHT)
    pat_w = float(PATTERN_WEIGHT)
    sim_w = float(SIM_WEIGHT)

    if not USE_POINT_DB:
        p_w = 0.0

    if not USE_PATTERN_DB:
        pat_w = 0.0

    # pattern_db 找不到時，不要把 pattern 權重偷塞給 point_db。
    # 這裡保留 pattern 權重，但 pattern 給中性值。
    # 這樣輸出會比較保守，不會假裝 pattern 有命中。
    if is_tie_point:
        sim_w = min(sim_w, TIE_AI_MAX_WEIGHT)

    total_weight = max(p_w + pat_w + sim_w, 0.0001)

    banker = (
        point["banker_prob"] * p_w +
        pattern["banker_prob"] * pat_w +
        ai["banker_prob"] * sim_w
    ) / total_weight

    banker = apply_tie_point_protection(banker, is_tie_point)
    banker = clamp(banker, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)

    player = 1.0 - banker
    gap = abs(banker - player)

    recommend = "莊" if banker >= player else "閒"

    entry_allowed, entry_level, weak_reason = build_entry_decision(
        is_tie_point=is_tie_point,
        gap=gap,
        recommend=recommend,
    )

    return {
        "ok": True,

        "player_point": player_point,
        "banker_point": banker_point,
        "last_result": last_result,

        "recommend": recommend,
        "player_prob": round(player * 100, PERCENT_DECIMALS),
        "banker_prob": round(banker * 100, PERCENT_DECIMALS),
        "player_prob_raw": player,
        "banker_prob_raw": banker,

        "confidence_gap": round(gap * 100, PERCENT_DECIMALS),
        "confidence_gap_raw": gap,

        "entry_allowed": entry_allowed,
        "entry_level": entry_level,
        "weak_reason": weak_reason,
        "no_observe": not entry_allowed,

        "tie_point_mode": is_tie_point,
        "tie_ai_max_weight": TIE_AI_MAX_WEIGHT if is_tie_point else None,
        "tie_shrink": TIE_SHRINK if is_tie_point else None,
        "tie_min_gap_for_entry": TIE_MIN_GAP_FOR_ENTRY if is_tie_point else None,

        "min_gap_for_entry": MIN_GAP_FOR_ENTRY,
        "strong_gap_for_entry": STRONG_GAP_FOR_ENTRY,

        "feature_key": feature_key(player_point, banker_point),

        "point_feature_key": point["feature_key"],
        "pattern_feature_key": pattern["feature_key"],

        "point_source": point["source"],
        "pattern_source": pattern["source"],
        "ai_source": ai["source"],

        "point_available": point["available"],
        "pattern_available": pattern["available"],

        "point_sample_size": point["sample_size"],
        "pattern_sample_size": pattern["sample_size"],

        "point_total_samples": point["total_simulated_samples"],
        "pattern_total_samples": pattern["total_simulated_samples"],

        "matched_patterns": [pattern["feature_key"]] if pattern["available"] else [],

        "weights": {
            "point": p_w,
            "pattern": pat_w,
            "simulation": sim_w,
            "total": total_weight,
        },

        "raw_layers": {
            "point_banker_prob": point["banker_prob"],
            "pattern_banker_prob": pattern["banker_prob"],
            "ai_banker_prob": ai["banker_prob"],
            "point_player_prob": point["player_prob"],
            "pattern_player_prob": pattern["player_prob"],
            "ai_player_prob": ai["player_prob"],
        },

        # 保留這兩個欄位，明確表示不使用前端歷史路紙
        "history_used": False,
        "rounds_ignored": True,

        "mode": "POINT_DB_PLUS_PATTERN_DB_PLUS_AI_NO_HISTORY",
    }
