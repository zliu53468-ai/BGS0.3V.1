"""BGS V10.7 動態牌路專家集成。

處理順序：
1. 正規化圖片辨識或使用者回報的 B/P/T；和局保留統計但不新增大路格位。
2. 先辨識牌路狀態（長龍、單跳、雙跳、轉折、混亂）。
3. 讓短／中／長視窗、路型、Markov、歷史相似型態等模型各自提出方向。
4. 依每個模型的樣本、可靠度與當前分歧動態計算權重，不以路型名稱硬指定答案。
5. 將各牌路子模型機率與共識結果交給全模型主引擎，與有限牌組模型共同參與最終方向。

本模組只分析已發生的 B/P/T 序列，不取得真人桌隱藏牌序，也不保證下一局結果。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import math
import os
import secrets

import numpy as np

from full_road_pattern_model import analyze_full_road_pattern


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(str(os.getenv(name, default)).strip())
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(str(os.getenv(name, default)).strip())
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


ROAD_MIN_SAMPLES = _env_int("ROAD_MIN_SAMPLES", 12, 4, 100)
ROAD_HISTORY_LIMIT = _env_int("ROAD_HISTORY_LIMIT", 500, 36, 2000)
ROAD_SIMULATIONS = _env_int("ROAD_SIMULATIONS", 5000, 500, 100_000)
ROAD_MIN_EDGE = _env_float("ROAD_MIN_EDGE", 0.075, 0.0, 0.30)
ROAD_MAX_UNCERTAINTY = _env_float("ROAD_MAX_UNCERTAINTY", 0.095, 0.01, 0.40)
ROAD_RECENCY_DECAY = _env_float("ROAD_RECENCY_DECAY", 0.90, 0.50, 0.999)
ROAD_MIN_CONSENSUS = _env_float("ROAD_MIN_CONSENSUS", 0.62, 0.50, 1.0)
ROAD_SHORT_WINDOW = _env_int("ROAD_SHORT_WINDOW", 8, 4, 30)
ROAD_MID_WINDOW = _env_int("ROAD_MID_WINDOW", 18, 8, 60)
ROAD_LONG_WINDOW = _env_int("ROAD_LONG_WINDOW", 36, 12, 120)
ROAD_MARKOV1_MIN_SUPPORT = _env_int("ROAD_MARKOV1_MIN_SUPPORT", 5, 1, 100)
ROAD_MARKOV2_MIN_SUPPORT = _env_int("ROAD_MARKOV2_MIN_SUPPORT", 4, 1, 100)
ROAD_MARKOV3_MIN_SUPPORT = _env_int("ROAD_MARKOV3_MIN_SUPPORT", 3, 1, 100)
ROAD_MAX_MODEL_DISAGREEMENT = _env_float("ROAD_MAX_MODEL_DISAGREEMENT", 0.135, 0.03, 0.40)

ROAD_FUSION_ENABLED = os.getenv("ROAD_FUSION_ENABLED", "1").strip() == "1"
ROAD_FUSION_WEIGHT = _env_float("ROAD_FUSION_WEIGHT", 0.08, 0.0, 0.20)
ROAD_FUSION_MIN_SAMPLES = _env_int("ROAD_FUSION_MIN_SAMPLES", 16, 4, 100)
ROAD_FUSION_MIN_CONFIDENCE = _env_float("ROAD_FUSION_MIN_CONFIDENCE", 0.58, 0.0, 1.0)
ROAD_FUSION_MAX_UNCERTAINTY = _env_float("ROAD_FUSION_MAX_UNCERTAINTY", 0.14, 0.01, 0.50)


def normalize_raw_outcomes(values: Iterable[Any]) -> List[str]:
    raw: List[str] = []
    for item in values:
        if isinstance(item, Mapping):
            value = str(item.get("outcome") or item.get("actual") or "").upper().strip()
        else:
            value = str(item or "").upper().strip()
        if value in {"B", "P", "T"}:
            raw.append(value)
    return raw[-max(ROAD_HISTORY_LIMIT * 2, 200):]


def normalize_road_sequence(values: Iterable[Any]) -> List[str]:
    return [v for v in normalize_raw_outcomes(values) if v in {"B", "P"}][-ROAD_HISTORY_LIMIT:]


def _clip_probability(value: float) -> float:
    return max(0.02, min(0.98, float(value)))


def _weighted_probability(sequence: Sequence[str], decay: float = ROAD_RECENCY_DECAY) -> float:
    if not sequence:
        return 0.5
    banker = 3.0
    player = 3.0
    for reverse_index, outcome in enumerate(reversed(sequence)):
        weight = decay ** reverse_index
        if outcome == "B":
            banker += weight
        else:
            player += weight
    return banker / max(1e-12, banker + player)


def _window_model(sequence: Sequence[str], size: int) -> Dict[str, Any]:
    window = list(sequence[-size:])
    probability = _weighted_probability(window)
    effective = len(window)
    reliability = min(1.0, effective / max(1.0, float(size)))
    return {
        "banker_probability": probability,
        "direction": "B" if probability >= 0.5 else "P",
        "edge": abs(probability - 0.5) * 2.0,
        "support": effective,
        "reliability": reliability,
    }


def _markov_model(sequence: Sequence[str], order: int, minimum_support: int) -> Dict[str, Any]:
    if len(sequence) <= order:
        return {"active": False, "support": 0, "banker_probability": 0.5, "direction": "B", "reliability": 0.0}
    context = tuple(sequence[-order:])
    banker = 2.5
    player = 2.5
    support = 0
    for index in range(order, len(sequence)):
        if tuple(sequence[index-order:index]) != context:
            continue
        support += 1
        if sequence[index] == "B":
            banker += 1.0
        else:
            player += 1.0
    probability = banker / (banker + player)
    active = support >= minimum_support
    reliability = min(1.0, support / max(float(minimum_support * 3), 1.0)) if active else 0.0
    return {
        "active": active,
        "support": support,
        "banker_probability": probability,
        "direction": "B" if probability >= 0.5 else "P",
        "edge": abs(probability - 0.5) * 2.0,
        "reliability": reliability,
        "context": "".join(context),
    }


def _run_lengths(sequence: Sequence[str]) -> List[Tuple[str, int]]:
    runs: List[Tuple[str, int]] = []
    for value in sequence:
        if runs and runs[-1][0] == value:
            runs[-1] = (value, runs[-1][1] + 1)
        else:
            runs.append((value, 1))
    return runs


def _detect_regime(sequence: Sequence[str]) -> Dict[str, Any]:
    recent = list(sequence[-18:])
    if len(recent) < 4:
        return {"name": "building", "confidence": 0.0, "alternation_rate": 0.0, "current_run": len(recent)}
    changes = sum(1 for a, b in zip(recent, recent[1:]) if a != b)
    alternation_rate = changes / max(1, len(recent) - 1)
    runs = _run_lengths(recent)
    current_run = runs[-1][1]
    pair_runs = [length for _, length in runs[-8:]]
    pair_score = sum(1 for length in pair_runs if length == 2) / max(1, len(pair_runs))
    if current_run >= 4:
        name = "streak"
        confidence = min(1.0, 0.55 + 0.10 * (current_run - 4))
    elif alternation_rate >= 0.78:
        name = "alternating"
        confidence = min(1.0, 0.55 + (alternation_rate - 0.78) * 1.5)
    elif pair_score >= 0.60:
        name = "double"
        confidence = min(1.0, 0.50 + pair_score * 0.35)
    elif alternation_rate <= 0.35 and max(pair_runs or [0]) >= 3:
        name = "clustered"
        confidence = 0.58
    elif 0.42 <= alternation_rate <= 0.68:
        name = "chaotic"
        confidence = 0.48
    else:
        name = "transition"
        confidence = 0.52
    return {
        "name": name,
        "confidence": confidence,
        "alternation_rate": alternation_rate,
        "current_run": current_run,
        "last_side": recent[-1],
        "pair_score": pair_score,
        "runs": runs[-8:],
    }


def _pattern_state_vector(sequence: Sequence[str]) -> Tuple[float, ...]:
    recent = list(sequence[-36:])
    runs = _run_lengths(recent)
    lengths = [length for _, length in runs[-10:]]
    current_run = lengths[-1] if lengths else 0

    def change_rate(size: int) -> float:
        window = recent[-size:]
        if len(window) < 2:
            return 0.5
        return sum(a != b for a, b in zip(window, window[1:])) / (len(window) - 1)

    mean_run = sum(lengths) / max(1, len(lengths))
    variance = sum((value - mean_run) ** 2 for value in lengths) / max(1, len(lengths))
    pair_rate = sum(value == 2 for value in lengths) / max(1, len(lengths))
    return (
        min(1.0, current_run / 6.0),
        change_rate(8),
        change_rate(18),
        change_rate(36),
        min(1.0, mean_run / 5.0),
        min(1.0, variance / 8.0),
        pair_rate,
    )


def _pattern_model(sequence: Sequence[str], regime: Mapping[str, Any]) -> Dict[str, Any]:
    """使用過去相似狀態，不將長龍／單跳名稱直接寫成預測答案。"""
    if len(sequence) < 8:
        return {
            "active": False,
            "support": 0,
            "banker_probability": 0.5,
            "direction": "B",
            "reliability": 0.0,
            "rule": "historical-state-building",
        }

    target = _pattern_state_vector(sequence)
    candidates: List[Tuple[float, str]] = []
    start = max(8, len(sequence) - 220)
    feature_weights = (1.15, 1.00, 0.85, 0.65, 0.75, 0.55, 0.75)

    for index in range(start, len(sequence)):
        prefix = sequence[:index]
        if len(prefix) < 8:
            continue
        vector = _pattern_state_vector(prefix)
        distance = math.sqrt(
            sum(
                feature_weights[position] * (a - b) ** 2
                for position, (a, b) in enumerate(zip(target, vector))
            ) / sum(feature_weights)
        )
        similarity = math.exp(-4.5 * distance)
        candidates.append((similarity, sequence[index]))

    candidates.sort(key=lambda item: item[0], reverse=True)
    neighbors = candidates[:24]
    banker = 2.5
    player = 2.5
    support = 0
    similarity_total = 0.0
    for similarity, actual in neighbors:
        similarity_total += similarity
        if similarity >= 0.30:
            support += 1
        if actual == "B":
            banker += similarity
        else:
            player += similarity

    probability = banker / (banker + player)
    mean_similarity = similarity_total / max(1, len(neighbors))
    reliability = min(
        1.0,
        min(1.0, support / 12.0)
        * (0.45 + 0.55 * mean_similarity)
        * min(1.0, len(sequence) / 42.0),
    )
    active = support >= 3
    return {
        "active": active,
        "support": support,
        "banker_probability": probability,
        "direction": "B" if probability >= 0.5 else "P",
        "edge": abs(probability - 0.5) * 2.0,
        "reliability": reliability if active else 0.0,
        "rule": "historical-state-neighbors",
        "mean_similarity": mean_similarity,
        "regime_descriptor": str(regime.get("name") or ""),
        "hard_pattern_rule": False,
    }


def _analogue_model(sequence: Sequence[str]) -> Dict[str, Any]:
    best = None
    for order in (5, 4, 3):
        if len(sequence) <= order:
            continue
        suffix = tuple(sequence[-order:])
        b = 2.0
        p = 2.0
        support = 0
        for i in range(order, len(sequence)):
            if tuple(sequence[i-order:i]) == suffix:
                support += 1
                if sequence[i] == "B":
                    b += 1
                else:
                    p += 1
        if support >= 3:
            probability = b / (b + p)
            best = {
                "active": True,
                "order": order,
                "support": support,
                "banker_probability": probability,
                "direction": "B" if probability >= 0.5 else "P",
                "reliability": min(1.0, support / 12.0),
            }
            break
    return best or {"active": False, "order": 0, "support": 0, "banker_probability": 0.5, "direction": "B", "reliability": 0.0}


def _base_weights(regime_name: str = "") -> Dict[str, float]:
    """中性冷啟動先驗；regime 只保留描述用途，不再直接改權重。"""
    return {
        "short": 0.11,
        "mid": 0.11,
        "long": 0.09,
        "pattern": 0.12,
        "full_road": 0.18,
        "markov1": 0.10,
        "markov2": 0.10,
        "markov3": 0.08,
        "analogue": 0.11,
    }


def _dynamic_effective_weights(
    models: Mapping[str, Mapping[str, Any]],
    base_weights: Mapping[str, float],
) -> Dict[str, float]:
    raw: Dict[str, float] = {}
    for name, model in models.items():
        active = bool(model.get("active", True))
        if not active:
            raw[name] = 0.0
            continue
        reliability = max(
            0.0, min(1.0, float(model.get("reliability", 0.0) or 0.0))
        )
        support = max(0, int(model.get("support", 0) or 0))
        support_score = min(1.0, support / 16.0)
        probability = _clip_probability(
            float(model.get("banker_probability", 0.5) or 0.5)
        )
        edge_score = min(1.0, abs(probability - 0.5) / 0.12)
        quality = (
            0.58 * reliability
            + 0.24 * support_score
            + 0.18 * edge_score
        )
        raw[name] = base_weights.get(name, 0.0) * max(0.03, quality)

    total = sum(raw.values())
    if total <= 1e-12:
        return {"short": 1.0}
    return {name: value / total for name, value in raw.items()}


def calculate_road_probabilities(values: Iterable[Any], seed: int | None = None, *, grid_cells: Sequence[Mapping[str, Any]] | None = None, initial_image_count: int = 0, manual_count: int = 0) -> Dict[str, Any]:
    raw_outcomes = normalize_raw_outcomes(values)
    sequence = [v for v in raw_outcomes if v in {"B", "P"}][-ROAD_HISTORY_LIMIT:]
    length = len(sequence)
    tie_count = sum(1 for v in raw_outcomes if v == "T")
    raw_count = len(raw_outcomes)
    tie_rate = tie_count / raw_count if raw_count else 0.0
    run_seed = int(seed if seed is not None else secrets.randbits(32)) & 0xFFFFFFFF
    rng = np.random.default_rng(run_seed)

    regime = _detect_regime(sequence)
    models: Dict[str, Dict[str, Any]] = {
        "short": _window_model(sequence, ROAD_SHORT_WINDOW),
        "mid": _window_model(sequence, ROAD_MID_WINDOW),
        "long": _window_model(sequence, ROAD_LONG_WINDOW),
        "pattern": _pattern_model(sequence, regime),
        "full_road": analyze_full_road_pattern(sequence, grid_cells=grid_cells, initial_image_count=initial_image_count, manual_count=manual_count),
        "markov1": _markov_model(sequence, 1, ROAD_MARKOV1_MIN_SUPPORT),
        "markov2": _markov_model(sequence, 2, ROAD_MARKOV2_MIN_SUPPORT),
        "markov3": _markov_model(sequence, 3, ROAD_MARKOV3_MIN_SUPPORT),
        "analogue": _analogue_model(sequence),
    }
    weights = _base_weights()
    effective = _dynamic_effective_weights(models, weights)

    component_probabilities = {
        name: _clip_probability(float(model.get("banker_probability", 0.5) or 0.5))
        for name, model in models.items()
    }
    center_b = sum(effective.get(name, 0.0) * probability for name, probability in component_probabilities.items())
    model_disagreement = math.sqrt(sum(
        effective.get(name, 0.0) * (probability - center_b) ** 2
        for name, probability in component_probabilities.items()
    ))

    # 不是只用平均機率：另計算方向票數／共識，避免 50.1% 對 49.9% 的勉強選邊。
    banker_vote = sum(effective.get(name, 0.0) for name, model in models.items() if model.get("direction") == "B")
    player_vote = sum(effective.get(name, 0.0) for name, model in models.items() if model.get("direction") == "P")
    vote_total = max(1e-12, banker_vote + player_vote)
    banker_consensus = banker_vote / vote_total
    player_consensus = player_vote / vote_total
    direction = "B" if banker_consensus >= player_consensus else "P"
    consensus = max(banker_consensus, player_consensus)

    # 依各子模型機率與可靠度建立 Beta 後驗抽樣，僅估計穩定度，不取代共識方向。
    samples = np.zeros(ROAD_SIMULATIONS, dtype=np.float64)
    for name, model in models.items():
        weight = effective.get(name, 0.0)
        if weight <= 0:
            continue
        probability = component_probabilities[name]
        reliability = max(0.05, float(model.get("reliability", 0.0) or 0.0))
        concentration = 8.0 + 28.0 * reliability
        alpha = max(0.5, probability * concentration)
        beta = max(0.5, (1.0 - probability) * concentration)
        samples += weight * rng.beta(alpha, beta, ROAD_SIMULATIONS)

    banker = float(np.mean(samples)) if length else 0.5
    player = 1.0 - banker
    uncertainty = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.5
    probability_direction = "B" if banker >= 0.5 else "P"
    edge = abs(banker - player)

    window_directions = [models["short"]["direction"], models["mid"]["direction"], models["long"]["direction"]]
    window_agreement = max(window_directions.count("B"), window_directions.count("P")) / 3.0
    regime_name = str(regime.get("name") or "")
    direction_probability_consistent = direction == probability_direction
    disagreement_ok = model_disagreement <= ROAD_MAX_MODEL_DISAGREEMENT

    signal_allowed = bool(
        length >= ROAD_MIN_SAMPLES
        and consensus >= ROAD_MIN_CONSENSUS
        and edge >= ROAD_MIN_EDGE
        and uncertainty <= ROAD_MAX_UNCERTAINTY
        and disagreement_ok
        and direction_probability_consistent
    )
    action = direction if signal_allowed else "O"

    confidence_score = max(0.0, min(1.0,
        0.28 * min(1.0, length / 36.0)
        + 0.28 * consensus
        + 0.18 * (1.0 - min(1.0, model_disagreement / ROAD_MAX_MODEL_DISAGREEMENT))
        + 0.14 * min(1.0, edge / max(ROAD_MIN_EDGE, 1e-9))
        + 0.12 * (1.0 - min(1.0, uncertainty / ROAD_MAX_UNCERTAINTY))
    ))
    confidence_label = "較高" if confidence_score >= 0.72 else "中等" if confidence_score >= 0.50 else "偏低"

    eligible_for_core = bool(
        ROAD_FUSION_ENABLED
        and length >= ROAD_FUSION_MIN_SAMPLES
        and confidence_score >= ROAD_FUSION_MIN_CONFIDENCE
        and uncertainty <= ROAD_FUSION_MAX_UNCERTAINTY
        and consensus >= ROAD_MIN_CONSENSUS
        and disagreement_ok
        and direction_probability_consistent
    )
    suggested_core_weight = (
        ROAD_FUSION_WEIGHT
        * min(1.0, length / 36.0)
        * min(1.0, confidence_score / 0.72)
        * min(1.0, consensus / max(ROAD_MIN_CONSENSUS, 1e-9))
        if eligible_for_core else 0.0
    )

    if signal_allowed:
        signal_reason = "牌路專家可靠度、機率差距與模型分歧門檻均通過"
        signal_status_text = "牌路多模型方向已建立"
    else:
        reasons: List[str] = []
        if length < ROAD_MIN_SAMPLES: reasons.append("牌路樣本仍在累積")
        if consensus < ROAD_MIN_CONSENSUS: reasons.append("子模型方向共識不足")
        if edge < ROAD_MIN_EDGE: reasons.append("綜合方向差距不足")
        if uncertainty > ROAD_MAX_UNCERTAINTY: reasons.append("模型不確定性偏高")
        if not disagreement_ok: reasons.append("牌路子模型分歧過高")
        if not direction_probability_consistent: reasons.append("共識方向與綜合機率方向不一致")
        signal_reason = "、".join(reasons) or "牌路資料尚未形成明確方向"
        signal_status_text = "牌路多模型評估中"

    model_outputs = {}
    for name, model in models.items():
        model_outputs[name] = {
            **model,
            "banker_probability": round(component_probabilities[name], 6),
            "effective_weight": round(effective.get(name, 0.0), 6),
        }

    return {
        "ok": bool(sequence),
        "engine": "ROAD_DYNAMIC_EXPERT_ENSEMBLE_V10_7",
        "pipeline_stage": "parallel_road_experts_dynamic_reliability",
        "run_seed": run_seed,
        "sequence": sequence,
        "raw_outcomes": raw_outcomes,
        "sample_count": length,
        "raw_sample_count": raw_count,
        "tie_count": tie_count,
        "observed_tie_rate": round(tie_rate, 6),
        "banker_probability": banker,
        "player_probability": player,
        "banker_rate": round(banker * 100.0, 2),
        "player_rate": round(player * 100.0, 2),
        "direction": direction,
        "direction_text": "莊" if direction == "B" else "閒",
        "probability_direction": probability_direction,
        "action": action,
        "action_text": "莊" if action == "B" else "閒" if action == "P" else "觀望",
        "signal_allowed": signal_allowed,
        "signal_status_text": signal_status_text,
        "signal_reason": signal_reason,
        "edge": round(edge, 6),
        "edge_percent": round(edge * 100.0, 4),
        "uncertainty": round(uncertainty, 6),
        "model_disagreement": round(model_disagreement, 6),
        "max_model_disagreement": ROAD_MAX_MODEL_DISAGREEMENT,
        "confidence_score": round(confidence_score, 6),
        "confidence_label": confidence_label,
        "eligible_for_core": eligible_for_core,
        "suggested_core_weight": round(suggested_core_weight, 8),
        "max_core_weight": ROAD_FUSION_WEIGHT,
        "consensus": {
            "banker_vote": round(banker_consensus, 6),
            "player_vote": round(player_consensus, 6),
            "winning_share": round(consensus, 6),
            "minimum_required": ROAD_MIN_CONSENSUS,
            "window_agreement": round(window_agreement, 6),
        },
        "regime": {**regime, "descriptor_only": True, "controls_weight": False},
        "full_road_analysis": model_outputs.get("full_road", {}),
        "models": model_outputs,
        "component_probabilities": {
            name: {
                "B": round(component_probabilities[name], 8),
                "P": round(1.0 - component_probabilities[name], 8),
                "T": 0.0,
            }
            for name in models
        },
        "weights": {name: round(weight, 6) for name, weight in effective.items()},
        "supports": {
            "first_order": int(models["markov1"].get("support", 0)),
            "second_order": int(models["markov2"].get("support", 0)),
            "third_order": int(models["markov3"].get("support", 0)),
            "analogue": int(models["analogue"].get("support", 0)),
        },
        "center_probability_before_simulation": round(center_b, 6),
        "data_scope": "walk_forward_history_with_dynamic_expert_reliability",
        "full_history_used_count": length,
        "initial_image_count": max(0, int(initial_image_count or 0)),
        "manual_count": max(0, int(manual_count or 0)),
        "grid_cell_count": len(list(grid_cells or [])),
    }


def build_road_context(values: Iterable[Any], seed: int | None = None, *, grid_cells: Sequence[Mapping[str, Any]] | None = None, initial_image_count: int = 0, manual_count: int = 0) -> Dict[str, Any]:
    return calculate_road_probabilities(values, seed=seed, grid_cells=grid_cells, initial_image_count=initial_image_count, manual_count=manual_count)


def _probability(value: Any, fallback: float = 0.0) -> float:
    try:
        return max(0.0, float(value))
    except Exception:
        return fallback


def fuse_road_with_main_prediction(
    main_prediction: Mapping[str, Any],
    road_analysis: Mapping[str, Any],
) -> Dict[str, Any]:
    """相容舊版 app。

    V9.5 的正確流程已在主引擎內整合 road context；若結果已含
    ``road_integration.processed_inside_core``，此函式只補相容欄位，不再二次融合。
    舊版 predictor 尚未整合時，才執行保守的後置低權重融合。
    """
    result = dict(main_prediction or {})
    road = dict(road_analysis or {})
    result["road_support"] = road
    internal = dict(result.get("road_integration") or {})
    if internal.get("processed_inside_core"):
        result["road_fusion"] = dict(internal)
        return result

    main_b = _probability(result.get("banker_rate")) / 100.0
    main_p = _probability(result.get("player_rate")) / 100.0
    main_t = _probability(result.get("tie_rate")) / 100.0
    total = main_b + main_p + main_t
    if total <= 0:
        result["road_fusion"] = {"applied": False, "reason": "主模型機率資料不足"}
        return result
    main_b, main_p, main_t = main_b / total, main_p / total, main_t / total

    effective_weight = min(
        ROAD_FUSION_WEIGHT,
        _probability(road.get("suggested_core_weight"), 0.0),
    )
    eligible = bool(road.get("eligible_for_core"))
    if not eligible:
        effective_weight = 0.0
    road_b = min(1.0, _probability(road.get("banker_probability"), 0.5))
    main_bp_total = max(1e-12, main_b + main_p)
    main_b_no_tie = main_b / main_bp_total
    fused_b_no_tie = main_b_no_tie * (1.0 - effective_weight) + road_b * effective_weight
    fused_b = (1.0 - main_t) * fused_b_no_tie
    fused_p = (1.0 - main_t) * (1.0 - fused_b_no_tie)
    direction = "B" if fused_b >= fused_p else "P"

    result.update({
        "banker_rate": round(fused_b * 100.0, 2),
        "player_rate": round(fused_p * 100.0, 2),
        "tie_rate": round(main_t * 100.0, 2),
        "probabilities": {"B": fused_b, "P": fused_p, "T": main_t},
        "recommend": direction,
        "recommend_text": "莊" if direction == "B" else "閒",
        "direction_source": "legacy_post_fusion" if effective_weight > 0 else "main_model",
        "road_fusion": {
            "applied": effective_weight > 0,
            "eligible": eligible,
            "effective_weight": round(effective_weight, 8),
            "processed_inside_core": False,
        },
    })
    return result


__all__ = [
    "build_road_context",
    "calculate_road_probabilities",
    "fuse_road_with_main_prediction",
    "normalize_raw_outcomes",
    "normalize_road_sequence",
]