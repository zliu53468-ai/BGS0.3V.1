import hashlib
import json
import math
import os
import random
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple

import config
from point_db import get_point_record, point_db_meta

try:
    from point_composition_mc import composition_mc_lookup
except Exception:
    composition_mc_lookup = None


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

# ============================================================
# Feature DB 模式參數
# ============================================================
# hybrid  = 優先查 feature_db，查不到退回原本 history rounds 模式
# feature = 只用 feature_db，查不到走中性 fallback
# history = 完全使用原本 rounds 模式
PREDICT_MODE = os.getenv("PREDICT_MODE", "hybrid").strip().lower()

USE_PATTERN_FEATURE_DB = env_bool("USE_PATTERN_FEATURE_DB", "1")
USE_AI_POINT_FEATURE_DB = env_bool("USE_AI_POINT_FEATURE_DB", "1")
FEATURE_DB_FALLBACK_TO_HISTORY = env_bool("FEATURE_DB_FALLBACK_TO_HISTORY", "1")
FEATURE_DB_MIN_SAMPLE = env_int("FEATURE_DB_MIN_SAMPLE", "80")

PATTERN_FEATURE_DB_PATH = os.getenv("PATTERN_FEATURE_DB_PATH", "pattern_feature_db.json").strip()
AI_POINT_FEATURE_DB_PATH = os.getenv("AI_POINT_FEATURE_DB_PATH", "ai_point_feature_db.json").strip()

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

BASE_BANKER_NO_TIE = env_float("BASE_BANKER_NO_TIE", "0.5068")

MIN_GAP_FOR_ENTRY = env_float("MIN_GAP_FOR_ENTRY", "0.035")
STRONG_GAP_FOR_ENTRY = env_float("STRONG_GAP_FOR_ENTRY", "0.065")


# ============================================================
# 和局點數保護參數
# ============================================================

TIE_AI_MAX_WEIGHT = env_float("TIE_AI_MAX_WEIGHT", "0.02")
TIE_SHRINK = env_float("TIE_SHRINK", "0.30")
TIE_MIN_GAP_FOR_ENTRY = env_float("TIE_MIN_GAP_FOR_ENTRY", "0.08")


# ============================================================
# AI 微調參數
# ============================================================

AI_NOISE_SCALE = env_float("AI_NOISE_SCALE", "0.018")


# ============================================================
# Monte Carlo 風控驗證參數
# ============================================================

USE_MONTE_CARLO = env_bool("USE_MONTE_CARLO", "1")
MC_SIMULATIONS = env_int("MC_SIMULATIONS", "300")
MC_MIN_SIMULATIONS = env_int("MC_MIN_SIMULATIONS", "80")
MC_MAX_SIMULATIONS = env_int("MC_MAX_SIMULATIONS", "800")
MC_SEED = env_int("MC_SEED", "42")
MC_MAX_NOISE = env_float("MC_MAX_NOISE", "0.018")
MC_BLOCK_LOW_GAP = env_bool("MC_BLOCK_LOW_GAP", "1")
MC_MIN_GAP_FOR_ENTRY = env_float("MC_MIN_GAP_FOR_ENTRY", "0.035")
MC_DIRECTION_MISMATCH_BLOCK = env_bool("MC_DIRECTION_MISMATCH_BLOCK", "0")


# ============================================================
# 點數組成 / 補牌可能性 MC 輔助層參數
# ============================================================
# 這層不需要使用者輸入實際牌面，仍然只吃 player_point / banker_point。
# 它會反推「不補牌 / 閒補 / 莊補 / 雙方補」可能組合，當成輔助修正。
USE_COMPOSITION_MC = env_bool("USE_COMPOSITION_MC", "1")
COMPOSITION_MC_WEIGHT = env_float("COMPOSITION_MC_WEIGHT", "0.10")
COMPOSITION_MC_SIMULATIONS = env_int("COMPOSITION_MC_SIMULATIONS", "500")
COMPOSITION_MC_MAX_COMBOS = env_int("COMPOSITION_MC_MAX_COMBOS", "160")


# ============================================================
# 工具函式
# ============================================================

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def stable_noise(key: str, scale: float = 0.035) -> float:
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
    diff = player_point - banker_point
    key = feature_key(player_point, banker_point)

    banker = BASE_BANKER_NO_TIE

    if diff == 0:
        banker += stable_noise(key + ":tie", 0.018)
    elif 1 <= diff <= 2:
        # diff = player_point - banker_point；diff > 0 代表閒點較高，banker 應下修。
        banker -= 0.185
    elif 3 <= diff <= 5:
        # 閒中高點優勢，fallback 不可反向偏莊。
        banker -= 0.185
    elif diff >= 6:
        # 閒大點差優勢，仍只做溫和下修，避免 fallback 過度極端。
        banker -= 0.115
    elif -2 <= diff <= -1:
        # diff < 0 代表莊點較高，banker 應上修。
        banker += 0.185
    elif -5 <= diff <= -3:
        # 莊中高點優勢，banker 上修。
        banker += 0.185
    elif diff <= -6:
        # 莊大點差優勢，溫和上修。
        banker += 0.115

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
    if not USE_POINT_DB:
        return fallback_point_lookup(player_point, banker_point)

    try:
        rec = get_point_record(player_point, banker_point)

        banker = rec.get(
            "next_banker_rate",
            rec.get("banker_prob", rec.get("banker_rate", None))
        )
        player = rec.get(
            "next_player_rate",
            rec.get("player_prob", rec.get("player_rate", None))
        )

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
    備援用：如果 pattern_db.py 沒有成功 import / 呼叫，
    才直接讀 PATTERN_DB_PATH 的 json。
    主要邏輯仍以 pattern_db.py 的 pattern_lookup(rounds) / get_pattern_record(pattern, window) 為優先。
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
            },
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
                },
            }

        return data

    except Exception as e:
        return {
            "records": {},
            "meta": {
                "source": f"PATTERN_DB_LOAD_ERROR:{e}",
                "path": PATTERN_DB_PATH,
                "total_simulated_samples": 0,
            },
        }


def normalize_round_result(value: Any) -> Optional[str]:
    """
    將各種可能的歷史資料格式統一成 B / P / T。
    支援：
    - "B" / "P" / "T"
    - "莊" / "閒" / "和"
    - "banker" / "player" / "tie"
    - {"result": "..."}
    - {"winner": "..."}
    - {"last_result": "..."}
    - {"player_point": 6, "banker_point": 5}
    - {"player": 6, "banker": 5}
    """

    if value is None:
        return None

    if isinstance(value, dict):
        raw = (
            value.get("result")
            or value.get("winner")
            or value.get("last_result")
            or value.get("outcome")
            or value.get("side")
        )

        if raw is not None:
            return normalize_round_result(raw)

        pp = (
            value.get("player_point")
            if value.get("player_point") is not None
            else value.get("player")
            if value.get("player") is not None
            else value.get("p")
        )
        bp = (
            value.get("banker_point")
            if value.get("banker_point") is not None
            else value.get("banker")
            if value.get("banker") is not None
            else value.get("b")
        )

        try:
            if pp is None or bp is None:
                return None

            pp = int(pp)
            bp = int(bp)

            if pp > bp:
                return "P"
            if bp > pp:
                return "B"
            return "T"
        except Exception:
            return None

    s = str(value).strip().upper()

    if s in {"B", "BANKER", "庄", "莊"}:
        return "B"

    if s in {"P", "PLAYER", "闲", "閒", "閑"}:
        return "P"

    if s in {"T", "TIE", "和", "和局"}:
        return "T"

    return None


def rounds_to_pattern_string(rounds: Optional[List[Any]]) -> str:
    """
    將 rounds 轉成 pattern_db 真正吃的牌路字串，例如：BPBPP。
    無法辨識的資料會自動略過。
    """
    if not rounds:
        return ""

    chars: List[str] = []

    for r in rounds:
        ch = normalize_round_result(r)
        if ch in {"B", "P", "T"}:
            chars.append(ch)

    return "".join(chars)


def symbol_to_last_result(symbol: Optional[str]) -> Optional[str]:
    """
    將 B / P / T 轉成 pattern_db.py 常見的中文 last_result。
    pattern_db.py 若只讀 r.get("last_result")，就需要這個欄位。
    """
    if symbol == "B":
        return "莊"
    if symbol == "P":
        return "閒"
    if symbol == "T":
        return "和"
    return None


def enrich_rounds_for_pattern_db(rounds: Optional[List[Any]]) -> List[Dict[str, Any]]:
    """
    補齊 pattern_db.pattern_lookup(rounds) 需要的 last_result。

    你的實際 rounds 多半是：
        {"player_point": 6, "banker_point": 5}

    但 pattern_db.py 可能只讀：
        r.get("last_result")

    所以這裡會在不破壞原資料的前提下，自動補：
        player_point > banker_point => last_result = "閒"
        banker_point > player_point => last_result = "莊"
        相等 => last_result = "和"
    """
    if not rounds:
        return []

    enriched: List[Dict[str, Any]] = []

    for r in rounds:
        symbol = normalize_round_result(r)
        last_result = symbol_to_last_result(symbol)

        if isinstance(r, dict):
            item = dict(r)
            if not item.get("last_result") and last_result:
                item["last_result"] = last_result
            enriched.append(item)
        else:
            # 非 dict 的歷史資料也轉成 pattern_lookup 可讀的格式。
            if last_result:
                enriched.append({"last_result": last_result, "raw": r})

    return enriched


def extract_prob_from_pattern_record(rec: Dict[str, Any]) -> Optional[Tuple[float, float]]:
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
        if rec.get("banker") is not None
        else rec.get("B")
        if rec.get("B") is not None
        else rec.get("b")
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
        if rec.get("player") is not None
        else rec.get("P")
        if rec.get("P") is not None
        else rec.get("p")
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


def try_pattern_lookup_from_module(rounds: Optional[List[Any]], pattern: str) -> Optional[Dict[str, Any]]:
    """
    優先呼叫 pattern_db.py 裡真正的 pattern_lookup。
    Claude 指出的核心問題就是這裡：不能再拿 P6_B5 這種點數 key 去查規律。
    """
    try:
        import pattern_db
    except Exception:
        return None

    fn = getattr(pattern_db, "pattern_lookup", None)

    if not callable(fn):
        return None

    call_styles = [
        lambda: fn(rounds),
        lambda: fn(pattern),
        lambda: fn(rounds=rounds),
        lambda: fn(pattern=pattern),
        lambda: fn(history=rounds),
        lambda: fn(pattern_string=pattern),
    ]

    for call in call_styles:
        try:
            rec = call()
            if isinstance(rec, dict):
                rec = dict(rec)
                rec.setdefault("feature_key", pattern)
                rec.setdefault("pattern", pattern)
                return rec
        except Exception:
            continue

    return None


def try_get_pattern_record_from_module(pattern: str) -> Optional[Dict[str, Any]]:
    """
    若 pattern_db.py 沒有 pattern_lookup，就退而求其次呼叫：
    get_pattern_record(pattern: str, window: int)

    會從較長 window 往短 window 找，讓 300 萬組規律資料有機會命中。
    """
    try:
        import pattern_db
    except Exception:
        return None

    fn = getattr(pattern_db, "get_pattern_record", None)

    if not callable(fn):
        return None

    max_window = min(len(pattern), env_int("PATTERN_MAX_WINDOW", "12"))
    min_window = min(env_int("PATTERN_MIN_WINDOW", "3"), max_window)

    for window in range(max_window, min_window - 1, -1):
        sub_pattern = pattern[-window:]

        call_styles = [
            lambda sub_pattern=sub_pattern, window=window: fn(sub_pattern, window),
            lambda sub_pattern=sub_pattern, window=window: fn(pattern=sub_pattern, window=window),
            lambda sub_pattern=sub_pattern, window=window: fn(pattern=sub_pattern, win=window),
            lambda sub_pattern=sub_pattern, window=window: fn(sub_pattern),
        ]

        for call in call_styles:
            try:
                rec = call()
                if isinstance(rec, dict):
                    rec = dict(rec)
                    rec.setdefault("feature_key", sub_pattern)
                    rec.setdefault("pattern", sub_pattern)
                    rec.setdefault("window", window)
                    return rec
            except Exception:
                continue

    return None


def find_record_in_pattern_json(pattern: str) -> Optional[Dict[str, Any]]:
    """
    最後備援：直接讀 pattern_db.json。
    支援 records 為 dict 或 list 的格式。
    """
    data = load_pattern_db_file()
    records = data.get("records", data)

    max_window = min(len(pattern), env_int("PATTERN_MAX_WINDOW", "12"))
    min_window = min(env_int("PATTERN_MIN_WINDOW", "3"), max_window)

    keys_to_try: List[str] = []

    for window in range(max_window, min_window - 1, -1):
        sub_pattern = pattern[-window:]
        keys_to_try.extend([
            sub_pattern,
            f"{sub_pattern}:{window}",
            f"{window}:{sub_pattern}",
            f"W{window}_{sub_pattern}",
        ])

    if isinstance(records, dict):
        for key in keys_to_try:
            rec = records.get(key)
            if isinstance(rec, dict):
                rec = dict(rec)
                rec.setdefault("feature_key", key)
                rec.setdefault("pattern", key)
                return rec

    if isinstance(records, list):
        for key in keys_to_try:
            for rec in records:
                if not isinstance(rec, dict):
                    continue

                rec_key = str(
                    rec.get("feature_key")
                    or rec.get("key")
                    or rec.get("pattern_key")
                    or rec.get("pattern")
                    or ""
                )

                if rec_key == key:
                    rec = dict(rec)
                    rec.setdefault("feature_key", rec_key)
                    return rec

    return None


def pattern_db_lookup(
    player_point: int,
    banker_point: int,
    rounds: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    正確版 pattern 查詢：
    1. rounds 先轉成 B/P/T 歷史牌路字串
    2. 優先呼叫 pattern_db.pattern_lookup(rounds / pattern)
    3. 再退回 pattern_db.get_pattern_record(pattern, window)
    4. 最後才直接查 json
    """

    current_fkey = feature_key(player_point, banker_point)
    enriched_rounds = enrich_rounds_for_pattern_db(rounds)
    pattern = rounds_to_pattern_string(enriched_rounds)

    if not USE_PATTERN_DB:
        rec = neutral_record("PATTERN_DB_DISABLED")
        rec["feature_key"] = current_fkey
        rec["pattern"] = pattern
        return rec

    if not pattern or len(pattern) < env_int("PATTERN_MIN_HISTORY", "3"):
        rec = neutral_record("PATTERN_COLD_START")
        rec["feature_key"] = current_fkey
        rec["pattern"] = pattern
        return rec

    # 關鍵修正：pattern_lookup 要吃已補 last_result 的 rounds，
    # 不能直接把只有點數的 rounds 丟進去，否則 pattern_db.py 會讀不到牌路。
    rec = try_pattern_lookup_from_module(enriched_rounds, pattern)

    if rec is None:
        rec = try_get_pattern_record_from_module(pattern)

    if rec is None:
        rec = find_record_in_pattern_json(pattern)

    if not isinstance(rec, dict):
        neutral = neutral_record("PATTERN_DB_NEUTRAL_FALLBACK")
        neutral["feature_key"] = pattern
        neutral["pattern"] = pattern
        return neutral

    probs = extract_prob_from_pattern_record(rec)

    if probs is None:
        neutral = neutral_record("PATTERN_DB_RECORD_NO_PROB_FALLBACK")
        neutral["feature_key"] = rec.get("feature_key", pattern)
        neutral["pattern"] = pattern
        return neutral

    banker, player = probs
    meta = pattern_db_meta_safe()
    matched_pattern = str(
        rec.get("pattern")
        or rec.get("feature_key")
        or rec.get("key")
        or pattern
    )

    return {
        "available": True,
        "feature_key": matched_pattern,
        "pattern": pattern,
        "banker_prob": banker,
        "player_prob": player,
        "source": rec.get("source", "PATTERN_DB_HISTORY_LOOKUP"),
        "sample_size": int(rec.get("sample", rec.get("sample_size", rec.get("count", 0))) or 0),
        "total_simulated_samples": int(meta.get("total_simulated_samples", 0) or 0),
        "window": int(rec.get("window", len(matched_pattern)) or 0),
    }


# ============================================================
# FEATURE DB 查詢層：讓 pattern / AI 也可以像 point_db 一樣用當前點數獨立查表
# ============================================================

def _extract_prob_from_feature_record(rec: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """
    feature_db / ai_feature_db 共用的機率欄位解析。
    支援 banker_prob/player_prob、next_banker_rate/next_player_rate、B/P 等常見命名。
    """
    if not isinstance(rec, dict):
        return None

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
        if rec.get("banker") is not None
        else rec.get("B")
        if rec.get("B") is not None
        else rec.get("b")
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
        if rec.get("player") is not None
        else rec.get("P")
        if rec.get("P") is not None
        else rec.get("p")
    )

    if banker is None or player is None:
        return None

    try:
        return normalize_prob_pair(float(banker), float(player))
    except Exception:
        return None


def pattern_feature_db_lookup(player_point: int, banker_point: int) -> Dict[str, Any]:
    """
    方案一：pattern_feature_db。
    用當前點數特徵直接查 pattern_feature_db.json，讓 pattern 層可像 point_db 一樣獨立查表。
    查不到時回 neutral，是否退回原本 rounds pattern 由 predict() 控制。
    """
    fkey = feature_key(player_point, banker_point)

    if not USE_PATTERN_FEATURE_DB:
        rec = neutral_record("PATTERN_FEATURE_DB_DISABLED")
        rec["feature_key"] = fkey
        return rec

    try:
        import pattern_feature_db
        fn = getattr(pattern_feature_db, "get_pattern_feature_record", None)
        meta_fn = getattr(pattern_feature_db, "pattern_feature_db_meta", None)

        if not callable(fn):
            raise RuntimeError("get_pattern_feature_record not callable")

        raw = fn(player_point, banker_point)

        if not isinstance(raw, dict):
            raise RuntimeError("feature record not dict")

        sample_size = int(raw.get("sample", raw.get("sample_size", raw.get("count", 0))) or 0)

        if sample_size and sample_size < FEATURE_DB_MIN_SAMPLE:
            rec = neutral_record("PATTERN_FEATURE_DB_LOW_SAMPLE_FALLBACK")
            rec["feature_key"] = raw.get("feature_key", fkey)
            rec["sample_size"] = sample_size
            return rec

        probs = _extract_prob_from_feature_record(raw)

        if probs is None:
            rec = neutral_record("PATTERN_FEATURE_DB_NO_PROB_FALLBACK")
            rec["feature_key"] = raw.get("feature_key", fkey)
            rec["sample_size"] = sample_size
            return rec

        meta = meta_fn() if callable(meta_fn) else {}
        if not isinstance(meta, dict):
            meta = {}

        banker, player = probs

        return {
            "available": True,
            "feature_key": raw.get("feature_key", raw.get("key", fkey)),
            "pattern": raw.get("pattern", raw.get("feature_key", fkey)),
            "banker_prob": banker,
            "player_prob": player,
            "source": raw.get("source", "PATTERN_FEATURE_DB"),
            "sample_size": sample_size,
            "total_simulated_samples": int(meta.get("total_simulated_samples", 0) or 0),
            "window": 0,
        }

    except Exception as e:
        rec = neutral_record(f"PATTERN_FEATURE_DB_ERROR:{e}")
        rec["feature_key"] = fkey
        return rec


def ai_point_feature_db_lookup(player_point: int, banker_point: int) -> Dict[str, Any]:
    """
    方案二：ai_point_feature_db。
    用當前點數特徵直接查 ai_point_feature_db.json，讓 AI 層可像 point_db 一樣獨立查表。
    查不到時回 neutral，是否退回原本 rounds AI 由 predict() 控制。
    """
    fkey = feature_key(player_point, banker_point)

    if not USE_AI_POINT_FEATURE_DB:
        rec = neutral_record("AI_POINT_FEATURE_DB_DISABLED")
        rec["feature_key"] = fkey
        return {
            **rec,
            "history_points_used": 0,
            "history_adjust": 0.0,
            "history_reasons": ["ai_feature_db_disabled"],
        }

    try:
        import ai_point_feature_db
        fn = getattr(ai_point_feature_db, "get_ai_point_feature_record", None)
        meta_fn = getattr(ai_point_feature_db, "ai_point_feature_db_meta", None)

        if not callable(fn):
            raise RuntimeError("get_ai_point_feature_record not callable")

        raw = fn(player_point, banker_point)

        if not isinstance(raw, dict):
            raise RuntimeError("ai feature record not dict")

        sample_size = int(raw.get("sample", raw.get("sample_size", raw.get("count", 0))) or 0)

        if sample_size and sample_size < FEATURE_DB_MIN_SAMPLE:
            rec = neutral_record("AI_POINT_FEATURE_DB_LOW_SAMPLE_FALLBACK")
            rec["feature_key"] = raw.get("feature_key", fkey)
            rec["sample_size"] = sample_size
            return {
                **rec,
                "history_points_used": 0,
                "history_adjust": 0.0,
                "history_reasons": ["ai_feature_low_sample"],
            }

        probs = _extract_prob_from_feature_record(raw)

        if probs is None:
            rec = neutral_record("AI_POINT_FEATURE_DB_NO_PROB_FALLBACK")
            rec["feature_key"] = raw.get("feature_key", fkey)
            rec["sample_size"] = sample_size
            return {
                **rec,
                "history_points_used": 0,
                "history_adjust": 0.0,
                "history_reasons": ["ai_feature_no_prob"],
            }

        meta = meta_fn() if callable(meta_fn) else {}
        if not isinstance(meta, dict):
            meta = {}

        banker, player = probs

        return {
            "available": True,
            "feature_key": raw.get("feature_key", raw.get("key", fkey)),
            "banker_prob": banker,
            "player_prob": player,
            "source": raw.get("source", "AI_POINT_FEATURE_DB"),
            "sample_size": sample_size,
            "total_simulated_samples": int(meta.get("total_simulated_samples", 0) or 0),
            "history_points_used": 0,
            "history_adjust": float(raw.get("history_adjust", 0.0) or 0.0),
            "history_reasons": ["ai_point_feature_db_lookup"],
        }

    except Exception as e:
        rec = neutral_record(f"AI_POINT_FEATURE_DB_ERROR:{e}")
        rec["feature_key"] = fkey
        return {
            **rec,
            "history_points_used": 0,
            "history_adjust": 0.0,
            "history_reasons": ["ai_feature_db_error"],
        }


def choose_pattern_layer(player_point: int, banker_point: int, rounds: Optional[List[Any]]) -> Dict[str, Any]:
    mode = PREDICT_MODE if PREDICT_MODE in {"hybrid", "feature", "history"} else "hybrid"

    if mode == "history":
        rec = pattern_db_lookup(player_point, banker_point, rounds=rounds)
        rec["layer_mode"] = "history"
        return rec

    feature_rec = pattern_feature_db_lookup(player_point, banker_point)

    if feature_rec.get("available") or mode == "feature" or not FEATURE_DB_FALLBACK_TO_HISTORY:
        feature_rec["layer_mode"] = "feature"
        return feature_rec

    history_rec = pattern_db_lookup(player_point, banker_point, rounds=rounds)
    history_rec["layer_mode"] = "history_fallback"
    history_rec["feature_fallback_source"] = feature_rec.get("source")
    return history_rec


def choose_ai_layer(player_point: int, banker_point: int, rounds: Optional[List[Any]]) -> Dict[str, Any]:
    mode = PREDICT_MODE if PREDICT_MODE in {"hybrid", "feature", "history"} else "hybrid"

    if mode == "history":
        rec = ai_simulation_layer(player_point, banker_point, rounds=rounds)
        rec["layer_mode"] = "history"
        return rec

    feature_rec = ai_point_feature_db_lookup(player_point, banker_point)

    if feature_rec.get("available") or mode == "feature" or not FEATURE_DB_FALLBACK_TO_HISTORY:
        feature_rec["layer_mode"] = "feature"
        return feature_rec

    history_rec = ai_simulation_layer(player_point, banker_point, rounds=rounds)
    history_rec["layer_mode"] = "history_fallback"
    history_rec["feature_fallback_source"] = feature_rec.get("source")
    return history_rec


# ============================================================
# AI 模擬層
# ============================================================

AI_HISTORY_WINDOW = env_int("AI_HISTORY_WINDOW", "8")
AI_TREND_STRENGTH = env_float("AI_TREND_STRENGTH", "0.014")
AI_DIFF_MOMENTUM_STRENGTH = env_float("AI_DIFF_MOMENTUM_STRENGTH", "0.012")
AI_REVERSAL_STRENGTH = env_float("AI_REVERSAL_STRENGTH", "0.010")
AI_HISTORY_MAX_ADJUST = env_float("AI_HISTORY_MAX_ADJUST", "0.035")


def extract_round_points(rounds: Optional[List[Any]]) -> List[Tuple[int, int]]:
    """
    從 rounds 抽出歷史點數序列。
    支援格式：
    - {"player_point": 6, "banker_point": 5}
    - {"player": 6, "banker": 5}
    - {"p": 6, "b": 5}
    - (6, 5) / [6, 5]

    回傳格式固定為：[(player_point, banker_point), ...]
    """
    if not rounds:
        return []

    out: List[Tuple[int, int]] = []

    for r in rounds:
        pp = None
        bp = None

        if isinstance(r, dict):
            pp = (
                r.get("player_point")
                if r.get("player_point") is not None
                else r.get("player")
                if r.get("player") is not None
                else r.get("p")
            )
            bp = (
                r.get("banker_point")
                if r.get("banker_point") is not None
                else r.get("banker")
                if r.get("banker") is not None
                else r.get("b")
            )
        elif isinstance(r, (list, tuple)) and len(r) >= 2:
            pp, bp = r[0], r[1]

        try:
            if pp is None or bp is None:
                continue

            pp = int(pp)
            bp = int(bp)

            if 0 <= pp <= 9 and 0 <= bp <= 9:
                out.append((pp, bp))
        except Exception:
            continue

    return out


def trend_delta(values: List[int]) -> float:
    """
    用簡單線性趨勢看最近點數是往上還往下。
    正值代表後段偏高，負值代表後段偏低。
    """
    n = len(values)
    if n < 3:
        return 0.0

    mid = n // 2
    early = values[:mid]
    late = values[mid:]

    if not early or not late:
        return 0.0

    return (sum(late) / len(late)) - (sum(early) / len(early))


def streak_count(results: List[str], side: str) -> int:
    """
    計算最近連續同邊次數，和局不列入連續方向。
    """
    count = 0
    for r in reversed(results):
        if r == "T":
            continue
        if r == side:
            count += 1
        else:
            break
    return count


def ai_simulation_layer(
    player_point: int,
    banker_point: int,
    rounds: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    本地 AI 修正層 v3：點數序列趨勢版。

    分工原則：
    - point_db：看「當前這一局點數組合」的統計。
    - pattern_db：看「B/P/T 牌路字串」的規律。
    - AI 層：看「歷史點數序列」的趨勢、動能、過熱反轉。

    注意：AI 層只做小幅微調，避免蓋過 point_db / pattern_db。
    """

    diff = player_point - banker_point
    key = feature_key(player_point, banker_point)

    banker = BASE_BANKER_NO_TIE
    reasons: List[str] = []
    history_adjust = 0.0

    # 1) 當前點數差：保留原本 v2 的基礎判斷
    if diff == 0:
        banker += stable_noise(key + ":ai_tie", 0.006)
        reasons.append("current_tie_point_noise")

    elif abs(diff) <= 2:
        adj = -0.018 if diff > 0 else 0.018
        banker += adj
        reasons.append("current_small_diff_adjust")

    elif abs(diff) <= 5:
        # diff = 閒點 - 莊點；diff > 0 代表閒點較高，banker 應下修。
        adj = -0.022 if diff > 0 else 0.022
        banker += adj
        reasons.append("current_mid_diff_adjust")

    else:
        # 大點差也維持同一方向：閒高偏閒、莊高偏莊。
        adj = -0.012 if diff > 0 else 0.012
        banker += adj
        reasons.append("current_large_diff_adjust")

    # 2) 歷史點數序列：真正讓 AI 層吃 rounds
    point_history = extract_round_points(rounds)
    recent = point_history[-AI_HISTORY_WINDOW:] if point_history else []

    if len(recent) >= 3:
        player_points = [p for p, _ in recent]
        banker_points = [b for _, b in recent]
        diffs = [p - b for p, b in recent]
        results = ["P" if p > b else "B" if b > p else "T" for p, b in recent]

        p_trend = trend_delta(player_points)
        b_trend = trend_delta(banker_points)
        diff_trend = trend_delta(diffs)

        # 閒點數近期升得比莊明顯：偏閒，banker 下修
        if p_trend - b_trend >= 1.5:
            adj = -AI_TREND_STRENGTH
            history_adjust += adj
            reasons.append("player_point_trend_up")

        # 莊點數近期升得比閒明顯：偏莊，banker 上修
        elif b_trend - p_trend >= 1.5:
            adj = AI_TREND_STRENGTH
            history_adjust += adj
            reasons.append("banker_point_trend_up")

        # 點差動能：diff = 閒點 - 莊點；diff 越往正，代表閒動能增強
        if diff_trend >= 1.25:
            adj = -AI_DIFF_MOMENTUM_STRENGTH
            history_adjust += adj
            reasons.append("player_diff_momentum")
        elif diff_trend <= -1.25:
            adj = AI_DIFF_MOMENTUM_STRENGTH
            history_adjust += adj
            reasons.append("banker_diff_momentum")

        # 過熱反轉：某一邊連續高點數，下一手不追太滿，只做小幅反向保護
        recent_player_hot = sum(1 for x in player_points[-3:] if x >= 7)
        recent_banker_hot = sum(1 for x in banker_points[-3:] if x >= 7)

        if recent_player_hot >= 3:
            history_adjust += AI_REVERSAL_STRENGTH
            reasons.append("player_hot_reversal_guard")
        elif recent_banker_hot >= 3:
            history_adjust -= AI_REVERSAL_STRENGTH
            reasons.append("banker_hot_reversal_guard")

        # 連莊 / 連閒保護：不是直接反打，只是避免 AI 層過度追單邊
        b_streak = streak_count(results, "B")
        p_streak = streak_count(results, "P")

        if b_streak >= 4:
            history_adjust -= AI_REVERSAL_STRENGTH * 0.7
            reasons.append("banker_streak_guard")
        elif p_streak >= 4:
            history_adjust += AI_REVERSAL_STRENGTH * 0.7
            reasons.append("player_streak_guard")

    history_adjust = clamp(history_adjust, -AI_HISTORY_MAX_ADJUST, AI_HISTORY_MAX_ADJUST)
    banker += history_adjust

    banker += stable_noise(key + ":ai_v3_history", AI_NOISE_SCALE)
    banker = clamp(banker, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)

    return {
        "banker_prob": banker,
        "player_prob": 1.0 - banker,
        "source": "LOCAL_AI_SIMULATION_POINT_SEQUENCE_V3",
        "history_points_used": len(recent),
        "history_adjust": history_adjust,
        "history_reasons": reasons,
    }


# ============================================================
# Monte Carlo 風控驗證層
# ============================================================

def monte_carlo_verify_from_probs(
    banker_prob: float,
    player_prob: float,
    n_sim: Optional[int] = None,
    seed_key: str = "",
) -> Dict[str, Any]:
    """
    安全版 Monte Carlo：只拿 predict 已經算好的最終機率做穩定度抽樣。

    重要：這裡不能再呼叫 predict()，否則會造成 predict -> MC -> predict 的無限遞迴。
    定位：風控驗證層，不是主預測層。
    """
    if n_sim is None:
        n_sim = MC_SIMULATIONS

    try:
        n_sim = int(n_sim)
    except Exception:
        n_sim = 300

    min_sim = max(20, int(MC_MIN_SIMULATIONS))
    max_sim = max(min_sim, int(MC_MAX_SIMULATIONS))
    n_sim = max(min_sim, min(n_sim, max_sim))

    banker_prob, player_prob = normalize_prob_pair(float(banker_prob), float(player_prob))

    rng = random.Random(f"{MC_SEED}:{seed_key}")

    wins = {
        "banker": 0,
        "player": 0,
        "tie": 0,
    }

    for _ in range(n_sim):
        # 微幅擾動用來測試目前推薦方向是否穩定。
        noise = rng.uniform(-MC_MAX_NOISE, MC_MAX_NOISE)
        b = clamp(banker_prob + noise, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)
        p = 1.0 - b

        # 只抽一次亂數，避免 banker / player 判斷機率失真。
        r = rng.random()

        if r < b:
            wins["banker"] += 1
        elif r < b + p:
            wins["player"] += 1
        else:
            wins["tie"] += 1

    total = wins["banker"] + wins["player"] + wins["tie"]

    if total <= 0:
        banker_rate = BASE_BANKER_NO_TIE
        player_rate = 1.0 - BASE_BANKER_NO_TIE
        tie_rate = 0.0
    else:
        banker_rate = wins["banker"] / total
        player_rate = wins["player"] / total
        tie_rate = wins["tie"] / total

    mc_gap = abs(banker_rate - player_rate)
    mc_recommend = "莊" if banker_rate >= player_rate else "閒"

    return {
        "mc_enabled": True,
        "mc_simulations": n_sim,
        "mc_banker_rate": round(banker_rate * 100, PERCENT_DECIMALS),
        "mc_player_rate": round(player_rate * 100, PERCENT_DECIMALS),
        "mc_tie_rate": round(tie_rate * 100, PERCENT_DECIMALS),
        "mc_banker_rate_raw": banker_rate,
        "mc_player_rate_raw": player_rate,
        "mc_tie_rate_raw": tie_rate,
        "mc_recommend": mc_recommend,
        "mc_gap": round(mc_gap * 100, PERCENT_DECIMALS),
        "mc_gap_raw": mc_gap,
        "mc_source": "MONTE_CARLO_PROB_STABILITY_CHECK_SAFE_V1",
        "mc_note": "MC only verifies final probability stability; it does not call predict().",
    }


def disabled_monte_carlo_result() -> Dict[str, Any]:
    return {
        "mc_enabled": False,
        "mc_simulations": 0,
        "mc_source": "MONTE_CARLO_DISABLED",
    }



# ============================================================
# 點數組成 / 補牌可能性 MC 輔助層
# ============================================================

def composition_mc_layer(player_point: int, banker_point: int, rounds: Optional[List[Any]] = None) -> Dict[str, Any]:
    if not USE_COMPOSITION_MC or not callable(composition_mc_lookup):
        return {
            "available": False,
            "feature_key": "COMPOSITION_MC_DISABLED",
            "banker_prob": BASE_BANKER_NO_TIE,
            "player_prob": 1.0 - BASE_BANKER_NO_TIE,
            "source": "COMPOSITION_MC_DISABLED",
            "sample_size": 0,
            "total_simulated_samples": 0,
            "scenario_debug": [],
        }

    try:
        rec = composition_mc_lookup(
            player_point=player_point,
            banker_point=banker_point,
            n_sim=COMPOSITION_MC_SIMULATIONS,
            max_combos=COMPOSITION_MC_MAX_COMBOS,
            seed_key=f"{player_point}:{banker_point}:{len(rounds or [])}",
        )

        if not isinstance(rec, dict):
            raise ValueError("composition_mc_lookup returned non-dict")

        banker, player = normalize_prob_pair(
            float(rec.get("banker_prob", BASE_BANKER_NO_TIE)),
            float(rec.get("player_prob", 1.0 - BASE_BANKER_NO_TIE)),
        )

        return {
            "available": bool(rec.get("available", False)),
            "feature_key": rec.get("feature_key", f"P{player_point}_B{banker_point}_COMPOSITION_MC"),
            "banker_prob": banker,
            "player_prob": player,
            "source": rec.get("source", "POINT_COMPOSITION_MC"),
            "sample_size": int(rec.get("sample_size", 0) or 0),
            "total_simulated_samples": int(rec.get("total_simulated_samples", rec.get("sample_size", 0)) or 0),
            "scenario_debug": rec.get("scenario_debug", []),
        }

    except Exception as e:
        return {
            "available": False,
            "feature_key": f"P{player_point}_B{banker_point}_COMPOSITION_MC_ERROR",
            "banker_prob": BASE_BANKER_NO_TIE,
            "player_prob": 1.0 - BASE_BANKER_NO_TIE,
            "source": f"COMPOSITION_MC_ERROR:{e}",
            "sample_size": 0,
            "total_simulated_samples": 0,
            "scenario_debug": [],
        }


# ============================================================
# 和局保護
# ============================================================

def apply_tie_point_protection(banker: float, is_tie_point: bool) -> float:
    if not is_tie_point:
        return banker

    return BASE_BANKER_NO_TIE + (banker - BASE_BANKER_NO_TIE) * TIE_SHRINK


# ============================================================
# 進場判斷
# ============================================================

def build_entry_decision(is_tie_point: bool, gap: float, recommend: str) -> Tuple[bool, str, str]:
    if is_tie_point and gap < TIE_MIN_GAP_FOR_ENTRY:
        return (
            False,
            "no_entry",
            "上一局為和局點數，莊閒優勢不足，建議觀察一局",
        )

    if gap < MIN_GAP_FOR_ENTRY:
        return (
            False,
            "no_entry",
            "莊閒機率差距不足，建議觀察一局",
        )

    if gap >= STRONG_GAP_FOR_ENTRY:
        return (
            True,
            "strong",
            "",
        )

    return (
        True,
        "normal",
        "",
    )


# ============================================================
# 主預測函式
# ============================================================

def predict(player_point: int, banker_point: int, rounds: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    player_point = validate_point(player_point)
    banker_point = validate_point(banker_point)

    last_result = get_last_result(player_point, banker_point)
    is_tie_point = player_point == banker_point

    point = point_db_lookup(player_point, banker_point)
    pattern = choose_pattern_layer(player_point, banker_point, rounds=rounds)
    ai = choose_ai_layer(player_point, banker_point, rounds=rounds)
    comp = composition_mc_layer(player_point, banker_point, rounds=rounds)

    p_w = float(POINT_WEIGHT)
    pat_w = float(PATTERN_WEIGHT)
    sim_w = float(SIM_WEIGHT)
    comp_w = float(COMPOSITION_MC_WEIGHT) if comp.get("available") else 0.0

    if not USE_POINT_DB:
        p_w = 0.0

    if not USE_PATTERN_DB:
        pat_w = 0.0

    if is_tie_point:
        sim_w = min(sim_w, TIE_AI_MAX_WEIGHT)
        comp_w = min(comp_w, COMPOSITION_MC_WEIGHT * 0.50)

    total_weight = max(p_w + pat_w + sim_w + comp_w, 0.0001)

    # 核心修正：
    # POINT_WEIGHT 只給 point_db
    # PATTERN_WEIGHT 只給 pattern_db
    # SIM_WEIGHT 只給 AI
    # COMPOSITION_MC_WEIGHT 只給「點數組成 / 補牌可能性 MC」
    banker = (
        point["banker_prob"] * p_w +
        pattern["banker_prob"] * pat_w +
        ai["banker_prob"] * sim_w +
        comp["banker_prob"] * comp_w
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

    result = {
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
        "ai_history_points_used": ai.get("history_points_used", 0),
        "ai_history_adjust": ai.get("history_adjust", 0.0),
        "ai_history_reasons": ai.get("history_reasons", []),
        "predict_mode": PREDICT_MODE,
        "pattern_layer_mode": pattern.get("layer_mode", "unknown"),
        "ai_layer_mode": ai.get("layer_mode", "unknown"),
        "composition_mc_source": comp.get("source", "COMPOSITION_MC_UNKNOWN"),
        "composition_mc_available": comp.get("available", False),
        "composition_mc_sample_size": comp.get("sample_size", 0),

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
            "composition_mc": comp_w,
            "total": total_weight,
        },

        "raw_layers": {
            "point_banker_prob": point["banker_prob"],
            "pattern_banker_prob": pattern["banker_prob"],
            "ai_banker_prob": ai["banker_prob"],
            "composition_mc_banker_prob": comp["banker_prob"],
            "point_player_prob": point["player_prob"],
            "pattern_player_prob": pattern["player_prob"],
            "ai_player_prob": ai["player_prob"],
            "composition_mc_player_prob": comp["player_prob"],
        },

        "history_used": bool(rounds) and (
            pattern.get("source") != "PATTERN_COLD_START" or ai.get("history_points_used", 0) >= 3
        ),
        "rounds_ignored": False,
        "pattern_string": pattern.get("pattern", ""),
        "pattern_window": pattern.get("window", 0),
        "pattern_rounds_enriched": True,

        "mode": "POINT_DB_PLUS_FEATURE_DB_COMPOSITION_MC_HYBRID_V8",
    }

    if USE_MONTE_CARLO:
        mc_result = monte_carlo_verify_from_probs(
            banker_prob=banker,
            player_prob=player,
            seed_key=f"{player_point}:{banker_point}:{pattern.get('feature_key', '')}:{ai.get('history_adjust', 0.0)}:{comp.get('banker_prob', 0.0)}",
        )
        result["monte_carlo"] = mc_result

        mc_gap_raw = float(mc_result.get("mc_gap_raw", 0.0) or 0.0)
        mc_recommend = mc_result.get("mc_recommend", recommend)

        # MC 不改推薦方向，只做風控；若 MC 顯示差距不足，就降級觀察。
        if MC_BLOCK_LOW_GAP and result["entry_allowed"] and mc_gap_raw < MC_MIN_GAP_FOR_ENTRY:
            result["entry_allowed"] = False
            result["entry_level"] = "no_entry"
            result["weak_reason"] = "Monte Carlo 穩定度不足，莊閒差距偏小，建議觀察一局"
            result["no_observe"] = True
            result["mc_entry_blocked"] = True
        else:
            result["mc_entry_blocked"] = False

        # 預設不因 MC 方向不一致直接擋單；若你想更保守，可設 MC_DIRECTION_MISMATCH_BLOCK=1。
        if (
            MC_DIRECTION_MISMATCH_BLOCK
            and result["entry_allowed"]
            and mc_recommend in {"莊", "閒"}
            and mc_recommend != recommend
        ):
            result["entry_allowed"] = False
            result["entry_level"] = "no_entry"
            result["weak_reason"] = "Monte Carlo 模擬方向與主模型不一致，建議觀察一局"
            result["no_observe"] = True
            result["mc_entry_blocked"] = True
    else:
        result["monte_carlo"] = disabled_monte_carlo_result()
        result["mc_entry_blocked"] = False

    return result
