import hashlib
import math
from typing import Dict, Any, List

from config import (
    POINT_WEIGHT,
    PATTERN_WEIGHT,
    SIM_WEIGHT,
    MIN_OUTPUT_PROB,
    MAX_OUTPUT_PROB,
    PERCENT_DECIMALS,
    USE_POINT_DB,
)

from point_db import get_point_record, point_db_meta


BASE_BANKER_NO_TIE = 0.5068

# ============================================================
# 和局點數保護參數
# ============================================================
# TIE_AI_MAX_WEIGHT:
# 上一局為和局點數時，限制 AI simulation 最大權重，避免 noise 拉歪結果。
TIE_AI_MAX_WEIGHT = 0.02

# TIE_SHRINK:
# 上一局為和局點數時，將預測機率往 BASE_BANKER_NO_TIE 拉回。
# 數值越低越保守：
# 0.20 = 很保守
# 0.30 = 建議穩定版
# 0.40 = 比較積極
TIE_SHRINK = 0.30

# TIE_MIN_GAP_FOR_ENTRY:
# 上一局為和局點數時，莊閒差距低於此值就建議觀察。
# 0.08 = 8%
TIE_MIN_GAP_FOR_ENTRY = 0.08


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


def point_db_lookup(player_point: int, banker_point: int) -> Dict[str, Any]:
    """
    不吃使用者歷史紀錄。
    每一局只用當前輸入點數去查 point_db。
    """
    rec = get_point_record(player_point, banker_point)

    banker = float(rec["next_banker_rate"])
    player = float(rec["next_player_rate"])

    total = banker + player
    if total > 0:
        banker = banker / total
        player = player / total
    else:
        banker = BASE_BANKER_NO_TIE
        player = 1.0 - banker

    banker = clamp(banker, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)
    player = 1.0 - banker

    return {
        "available": True,
        "feature_key": f"P{player_point}_B{banker_point}",
        "banker_prob": banker,
        "player_prob": player,
        "source": rec.get("source", "POINT_DB_ONLY"),
        "sample_size": int(rec.get("sample", 0)),
        "total_simulated_samples": int(point_db_meta().get("total_simulated_samples", 0)),
    }


def fallback_point_lookup(player_point: int, banker_point: int) -> Dict[str, Any]:
    """
    point_db 查不到時的備用規則。
    同樣不吃使用者歷史紀錄。
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


def ai_simulation_layer(player_point: int, banker_point: int) -> Dict[str, Any]:
    """
    本地 AI 模擬層。
    只看當前點數，不吃使用者歷史紀錄。
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
        "source": "LOCAL_AI_SIMULATION_POINT_ONLY",
    }


def apply_tie_point_protection(banker: float, is_tie_point: bool) -> float:
    """
    和局點數保護：
    如果上一局為和局點數，例如 P6_B6、P7_B7、P8_B8，
    則不完全相信 point_db 或 AI 拉出的偏移值，
    而是將機率往百家樂基準莊率 BASE_BANKER_NO_TIE 拉回。

    目的：
    - 降低和局後預測過度自信
    - 避免和局樣本偏差導致方向亂喊
    - 讓和局後的訊號更保守
    """
    if not is_tie_point:
        return banker

    return BASE_BANKER_NO_TIE + (banker - BASE_BANKER_NO_TIE) * TIE_SHRINK


def predict(player_point: int, banker_point: int, rounds: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    純資料庫對應版。

    重要：
    - rounds 參數保留只是為了相容 server.py 原本呼叫方式。
    - 本函數完全不使用 rounds。
    - 不吃使用者歷史紀錄。
    - 不使用 pattern_lookup(rounds)。
    - 不使用 Count Gap。
    - 不使用 Route Detector。
    - 不使用 Break Route。

    本版新增：
    - 和局點數保護 tie_point_mode。
    - 上一局如果是和局點數，降低 AI simulation 干擾。
    - 上一局如果是和局點數，機率拉回 BASE_BANKER_NO_TIE。
    - 上一局如果是和局點數且莊閒差距不足，回傳 entry_allowed=False。
    """

    player_point = validate_point(player_point)
    banker_point = validate_point(banker_point)

    last_result = get_last_result(player_point, banker_point)
    is_tie_point = player_point == banker_point

    try:
        point = point_db_lookup(player_point, banker_point) if USE_POINT_DB else fallback_point_lookup(player_point, banker_point)
    except Exception:
        point = fallback_point_lookup(player_point, banker_point)

    ai = ai_simulation_layer(player_point, banker_point)

    # 不使用 pattern_lookup(rounds)，所以 pattern 權重併回 point。
    p_w = POINT_WEIGHT + PATTERN_WEIGHT
    pat_w = 0.0
    sim_w = SIM_WEIGHT

    # 和局點數時，降低 AI 模擬層干擾。
    if is_tie_point:
        sim_w = min(sim_w, TIE_AI_MAX_WEIGHT)

    total_weight = max(p_w + sim_w, 0.0001)

    banker = (
        point["banker_prob"] * p_w +
        ai["banker_prob"] * sim_w
    ) / total_weight

    # 和局點數保護：把機率拉回基準莊率附近。
    banker = apply_tie_point_protection(banker, is_tie_point)

    banker = clamp(banker, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)
    player = 1.0 - banker

    gap = abs(banker - player)

    recommend = "莊" if banker >= player else "閒"

    entry_allowed = True
    weak_reason = ""

    # 和局後如果莊閒差距不足，建議觀察。
    if is_tie_point and gap < TIE_MIN_GAP_FOR_ENTRY:
        entry_allowed = False
        weak_reason = "上一局為和局點數，莊閒優勢不足，建議觀察一局"

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
        "weak_reason": weak_reason,

        "tie_point_mode": is_tie_point,
        "tie_ai_max_weight": TIE_AI_MAX_WEIGHT if is_tie_point else None,
        "tie_shrink": TIE_SHRINK if is_tie_point else None,
        "tie_min_gap_for_entry": TIE_MIN_GAP_FOR_ENTRY if is_tie_point else None,

        "feature_key": point["feature_key"],

        "point_source": point["source"],
        "pattern_source": "DISABLED_HISTORY_NOT_USED",
        "ai_source": ai["source"],

        "point_sample_size": point["sample_size"],
        "pattern_sample_size": 0,

        "point_total_samples": point["total_simulated_samples"],
        "pattern_total_samples": 0,

        "matched_patterns": [],

        "weights": {
            "point": p_w,
            "pattern": pat_w,
            "simulation": sim_w,
        },

        "history_used": False,
        "rounds_ignored": True,

        "mode": "POINT_DB_ONLY_NO_USER_HISTORY_TIE_PROTECTION",
        "no_observe": not entry_allowed,
    }
