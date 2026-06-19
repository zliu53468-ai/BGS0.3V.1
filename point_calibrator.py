# -*- coding: utf-8 -*-
"""
point_calibrator.py - V9.6 point confidence calibrator

用途：
- 不改 point_db 原始資料。
- 不記錄用戶歷史。
- 只在每次 predict() 當下，比對 point_db / combo_db / 補牌MC / road_profile / AI 的方向一致性。
- 當點數層與多數強訊號衝突時，把 point_db 拉回中性並降低 point 權重。
- 當點數層與多數強訊號一致時，微幅提高 point 權重。
"""

import os
from typing import Dict, Any, List, Tuple


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


BASE_BANKER_NO_TIE = 0.5000

USE_POINT_CALIBRATOR = env_bool("USE_POINT_CALIBRATOR", "1")
POINT_CALIBRATION_STRENGTH = env_float("POINT_CALIBRATION_STRENGTH", "0.65")
POINT_CONFLICT_SHRINK = env_float("POINT_CONFLICT_SHRINK", "0.35")
POINT_SOFT_CONFLICT_SHRINK = env_float("POINT_SOFT_CONFLICT_SHRINK", "0.65")
POINT_AGREEMENT_BOOST = env_float("POINT_AGREEMENT_BOOST", "1.08")
POINT_NEUTRAL_GAP = env_float("POINT_NEUTRAL_GAP", "0.006")
POINT_MIN_SIGNAL_GAP = env_float("POINT_MIN_SIGNAL_GAP", "0.008")
POINT_STRONG_SIGNAL_GAP = env_float("POINT_STRONG_SIGNAL_GAP", "0.025")
POINT_CALIBRATOR_MIN_SUPPORTS = env_int("POINT_CALIBRATOR_MIN_SUPPORTS", "2")
POINT_CALIBRATOR_MAX_WEIGHT_MULT = env_float("POINT_CALIBRATOR_MAX_WEIGHT_MULT", "1.12")
POINT_CALIBRATOR_MIN_WEIGHT_MULT = env_float("POINT_CALIBRATOR_MIN_WEIGHT_MULT", "0.35")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _normalize_pair(banker: float, player: float) -> Tuple[float, float]:
    banker = float(banker)
    player = float(player)
    if banker > 1:
        banker /= 100.0
    if player > 1:
        player /= 100.0
    total = banker + player
    if total <= 0:
        return 0.5, 0.5
    return banker / total, player / total


def _direction(banker: float, player: float, neutral_gap: float = 0.0) -> Tuple[str, float]:
    banker, player = _normalize_pair(banker, player)
    gap = abs(banker - player)
    if gap < neutral_gap:
        return "NEUTRAL", gap
    return ("BANKER" if banker > player else "PLAYER"), gap


def _layer_signal(name: str, layer: Dict[str, Any], min_gap: float, base_weight: float) -> Dict[str, Any]:
    if not isinstance(layer, dict):
        return {"name": name, "available": False, "direction": "NEUTRAL", "gap": 0.0, "score_weight": 0.0}

    available = bool(layer.get("available", True))
    sample = int(layer.get("sample_size", layer.get("sample", 0)) or 0)
    banker = layer.get("banker_prob", layer.get("next_banker_rate", 0.5))
    player = layer.get("player_prob", layer.get("next_player_rate", 0.5))
    direction, gap = _direction(banker, player, neutral_gap=min_gap)

    if not available or direction == "NEUTRAL":
        score_weight = 0.0
    else:
        # 樣本越高權重略高，但避免過度支配。
        sample_factor = 1.0
        if sample > 0:
            sample_factor = _clamp(sample / 10000.0, 0.55, 1.60)
        gap_factor = _clamp(gap / max(POINT_STRONG_SIGNAL_GAP, 0.0001), 0.50, 1.50)
        score_weight = base_weight * sample_factor * gap_factor

    return {
        "name": name,
        "available": available,
        "direction": direction,
        "gap": gap,
        "sample": sample,
        "score_weight": score_weight,
        "banker_prob": float(banker) if banker is not None else 0.5,
        "player_prob": float(player) if player is not None else 0.5,
    }


def calibrate_point_layer(
    point: Dict[str, Any],
    combo: Dict[str, Any] = None,
    composition_mc: Dict[str, Any] = None,
    road_profile: Dict[str, Any] = None,
    ai: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    回傳：
    {
      point: 修正後 point dict,
      weight_multiplier: point 權重倍率,
      status: AGREEMENT / SOFT_CONFLICT / HEAVY_CONFLICT / NO_CALIBRATION
      ...
    }
    """
    if not USE_POINT_CALIBRATOR or not isinstance(point, dict):
        return {
            "point": point,
            "weight_multiplier": 1.0,
            "status": "DISABLED",
            "signals": [],
            "support_count": 0,
            "conflict_count": 0,
        }

    pb = float(point.get("banker_prob", 0.5) or 0.5)
    pp = float(point.get("player_prob", 0.5) or 0.5)
    pb, pp = _normalize_pair(pb, pp)

    point_dir, point_gap = _direction(pb, pp, neutral_gap=POINT_NEUTRAL_GAP)

    signals: List[Dict[str, Any]] = [
        _layer_signal("combo", combo or {}, POINT_MIN_SIGNAL_GAP, 1.25),
        _layer_signal("composition_mc", composition_mc or {}, POINT_MIN_SIGNAL_GAP, 1.15),
        _layer_signal("road_profile", road_profile or {}, POINT_MIN_SIGNAL_GAP, 0.75),
        _layer_signal("ai", ai or {}, POINT_MIN_SIGNAL_GAP, 0.55),
    ]

    support_score = 0.0
    conflict_score = 0.0
    support_count = 0
    conflict_count = 0

    for sig in signals:
        d = sig.get("direction")
        w = float(sig.get("score_weight", 0.0) or 0.0)
        if d == "NEUTRAL" or w <= 0:
            continue
        if point_dir != "NEUTRAL" and d == point_dir:
            support_score += w
            support_count += 1
        elif point_dir != "NEUTRAL" and d != point_dir:
            conflict_score += w
            conflict_count += 1

    calibrated_b = pb
    status = "NO_CALIBRATION"
    weight_multiplier = 1.0

    if point_dir == "NEUTRAL":
        status = "POINT_NEUTRAL"
        weight_multiplier = 0.75
    elif conflict_count >= POINT_CALIBRATOR_MIN_SUPPORTS and conflict_score > support_score:
        # 強衝突：把 point_db 往中性拉，並降低 point 權重。
        shrink = _clamp(POINT_CONFLICT_SHRINK, 0.05, 0.95)
        calibrated_b = BASE_BANKER_NO_TIE + (pb - BASE_BANKER_NO_TIE) * shrink
        weight_multiplier = POINT_CALIBRATOR_MIN_WEIGHT_MULT
        status = "HEAVY_CONFLICT_SHRINK_POINT"
    elif conflict_count >= 1 and conflict_score > support_score * 1.20:
        # 軟衝突：保留一部分點數訊號，但不要讓它主導。
        shrink = _clamp(POINT_SOFT_CONFLICT_SHRINK, 0.10, 1.00)
        calibrated_b = BASE_BANKER_NO_TIE + (pb - BASE_BANKER_NO_TIE) * shrink
        weight_multiplier = max(POINT_CALIBRATOR_MIN_WEIGHT_MULT, 0.65)
        status = "SOFT_CONFLICT_SHRINK_POINT"
    elif support_count >= POINT_CALIBRATOR_MIN_SUPPORTS and support_score >= conflict_score:
        # 多層同向：微幅提高 point 權重，不改大方向。
        weight_multiplier = min(POINT_AGREEMENT_BOOST, POINT_CALIBRATOR_MAX_WEIGHT_MULT)
        status = "AGREEMENT_BOOST_POINT"
    else:
        status = "MIXED_OR_WEAK_KEEP_POINT"
        weight_multiplier = 0.90 if conflict_count else 1.0

    # 校準強度：避免一次修太大。
    strength = _clamp(POINT_CALIBRATION_STRENGTH, 0.0, 1.0)
    final_b = pb + (calibrated_b - pb) * strength
    final_b = _clamp(final_b, 0.38, 0.62)
    final_p = 1.0 - final_b

    new_point = dict(point)
    new_point["banker_prob_original"] = pb
    new_point["player_prob_original"] = pp
    new_point["banker_prob"] = final_b
    new_point["player_prob"] = final_p
    new_point["source"] = str(point.get("source", "POINT_DB")) + "+POINT_CALIBRATOR_V9_6"
    new_point["calibrated"] = True

    return {
        "point": new_point,
        "weight_multiplier": _clamp(weight_multiplier, POINT_CALIBRATOR_MIN_WEIGHT_MULT, POINT_CALIBRATOR_MAX_WEIGHT_MULT),
        "status": status,
        "point_direction": point_dir,
        "point_gap": point_gap,
        "support_count": support_count,
        "conflict_count": conflict_count,
        "support_score": support_score,
        "conflict_score": conflict_score,
        "signals": signals,
        "original_banker_prob": pb,
        "calibrated_banker_prob": final_b,
    }
