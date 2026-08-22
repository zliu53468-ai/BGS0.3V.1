"""風險分層式自適應集成。

正常區間只在累積足夠已結算樣本後，依各子模型的歷史 Brier score
做小幅再加權；低優勢或模型衝突時，選可靠度／歷史 Brier 最佳的
單一核心模型，避免平均回 50/50。任一模型回報極端未知區間時，
仍立即執行零信心、零注碼、No Bet，不允許核心模型繞過硬熔斷。
"""
from __future__ import annotations

from typing import Any, Dict, Mapping
import math
import os

from performance_tracker import get_performance_summary


OUTCOMES = ("B", "P", "T")
ADAPTIVE_ENABLED = os.getenv("ADAPTIVE_ENSEMBLE_ENABLED", "1").strip() == "1"
ADAPTIVE_MIN_SAMPLES = max(50, int(os.getenv("ADAPTIVE_MIN_SAMPLES", "300") or "300"))
ADAPTIVE_MAX_SHARE = max(0.0, min(0.35, float(os.getenv("ADAPTIVE_MAX_SHARE", "0.15") or "0.15")))
ADAPTIVE_TEMPERATURE = max(0.01, min(2.0, float(os.getenv("ADAPTIVE_TEMPERATURE", "0.18") or "0.18")))
ADAPTIVE_CHAMPION_ENABLED = os.getenv(
    "ADAPTIVE_CHAMPION_ENABLED", "1"
).strip() == "1"
ADAPTIVE_CHAMPION_MIN_HISTORY = max(
    8,
    min(100, int(os.getenv("ADAPTIVE_CHAMPION_MIN_HISTORY", "12") or "12")),
)
ADAPTIVE_CHAMPION_MIN_EDGE = max(
    0.0,
    min(0.20, float(os.getenv("ADAPTIVE_CHAMPION_MIN_EDGE", "0.015") or "0.015")),
)
ADAPTIVE_CHAMPION_MIN_RELIABILITY = max(
    0.0,
    min(0.90, float(os.getenv("ADAPTIVE_CHAMPION_MIN_RELIABILITY", "0.20") or "0.20")),
)
ADAPTIVE_CHAMPION_PERFORMANCE_SAMPLES = max(
    12,
    min(500, int(os.getenv("ADAPTIVE_CHAMPION_PERFORMANCE_SAMPLES", "30") or "30")),
)
ADAPTIVE_CHAMPION_OVERRIDE_BASE_EDGE = max(
    0.0,
    min(0.20, float(os.getenv("ADAPTIVE_CHAMPION_OVERRIDE_BASE_EDGE", "0.035") or "0.035")),
)
ADAPTIVE_CHAMPION_TEMPERATURE = max(
    0.35,
    min(1.0, float(os.getenv("ADAPTIVE_CHAMPION_TEMPERATURE", "0.85") or "0.85")),
)


def _normalize(values: Mapping[str, Any]) -> Dict[str, float]:
    data = {key: max(1e-12, float(values.get(key, 0.0) or 0.0)) for key in OUTCOMES}
    total = sum(data.values())
    return {key: data[key] / total for key in OUTCOMES}


def _components(prediction: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    mapping = {
        "hypergeometric": prediction.get("hypergeometric_probabilities"),
        "monte_carlo": prediction.get("monte_carlo_probabilities"),
        "particle": prediction.get("particle_probabilities"),
        "sequence": prediction.get("sequence_probabilities"),
        "core_before_road": prediction.get("core_probabilities_before_road"),
    }
    result: Dict[str, Dict[str, float]] = {}
    nested = prediction.get("component_probabilities")
    if isinstance(nested, Mapping):
        for name, values in nested.items():
            if not isinstance(values, Mapping):
                continue
            try:
                result[str(name)] = _normalize(values)
            except Exception:
                pass
    for name, values in mapping.items():
        if isinstance(values, Mapping):
            try:
                result.setdefault(name, _normalize(values))
            except Exception:
                pass
    return result


def _select_component_champion(
    prediction: Mapping[str, Any],
    components: Mapping[str, Mapping[str, float]],
    historical_scores: Mapping[str, Any],
    component_sample_counts: Mapping[str, Any],
) -> Dict[str, Any]:
    """模型互相抵消時，選出一個可稽核的核心模型，不做盲目平均。"""
    road = prediction.get("road_support")
    road_data = dict(road) if isinstance(road, Mapping) else {}
    history_count = int(road_data.get("sample_count", 0) or 0)
    if history_count < ADAPTIVE_CHAMPION_MIN_HISTORY:
        return {}

    raw_models = road_data.get("models")
    model_metadata = dict(raw_models) if isinstance(raw_models, Mapping) else {}
    candidates = []
    for name, values in components.items():
        b = max(0.0, float(values.get("B", 0.0) or 0.0))
        p = max(0.0, float(values.get("P", 0.0) or 0.0))
        bp_total = b + p
        if bp_total <= 1e-12:
            continue
        banker = b / bp_total
        edge = abs(2.0 * banker - 1.0)
        direction = "B" if banker >= 0.5 else "P"

        raw_meta = model_metadata.get(name)
        meta = dict(raw_meta) if isinstance(raw_meta, Mapping) else {}
        model_active = bool(
            meta.get("active", meta.get("ok", True))
        )
        reliability = max(
            0.0,
            min(1.0, float(meta.get("reliability", 0.0) or 0.0)),
        )
        if not model_active:
            continue
        if edge < ADAPTIVE_CHAMPION_MIN_EDGE:
            continue
        if reliability < ADAPTIVE_CHAMPION_MIN_RELIABILITY:
            continue

        performance_samples = int(component_sample_counts.get(name, 0) or 0)
        historical_brier = historical_scores.get(name)
        historical_quality = None
        if performance_samples >= ADAPTIVE_CHAMPION_PERFORMANCE_SAMPLES:
            try:
                score = max(0.0, float(historical_brier))
                historical_quality = math.exp(-score / 0.50)
            except Exception:
                historical_quality = None

        evidence_quality = (
            float(historical_quality)
            if historical_quality is not None
            else reliability
        )
        selection_score = edge * (0.45 + 0.55 * evidence_quality)
        candidates.append({
            "name": str(name),
            "direction": direction,
            "banker_probability": float(banker),
            "edge": float(edge),
            "reliability": float(reliability),
            "support": int(meta.get("support", 0) or 0),
            "performance_samples": performance_samples,
            "historical_brier": (
                float(historical_brier)
                if historical_quality is not None
                else None
            ),
            "historical_performance_used": historical_quality is not None,
            "selection_score": float(selection_score),
        })

    if not candidates:
        return {}
    candidates.sort(
        key=lambda item: (
            float(item["selection_score"]),
            int(item["performance_samples"]),
            float(item["reliability"]),
            float(item["edge"]),
            str(item["name"]),
        ),
        reverse=True,
    )
    winner = dict(candidates[0])
    probability = max(
        1e-6,
        min(1.0 - 1e-6, float(winner["banker_probability"])),
    )
    logit = math.log(probability / (1.0 - probability))
    sharpened = 1.0 / (
        1.0 + math.exp(-logit / ADAPTIVE_CHAMPION_TEMPERATURE)
    )
    winner.update({
        "tempered_banker_probability": float(sharpened),
        "tempered_edge": float(abs(2.0 * sharpened - 1.0)),
        "models_conflict": len({item["direction"] for item in candidates}) > 1,
        "candidate_count": len(candidates),
        "history_count": history_count,
        "candidates": candidates,
    })
    return winner


def _is_extreme_unseen(prediction: Mapping[str, Any]) -> bool:
    """接受新舊欄位，避免部署期間因版本差異再次造成訊號斷層。"""
    def flagged(model: Mapping[str, Any]) -> bool:
        braking = model.get("uncertainty_braking")
        predictor_signal = model.get("predictor_signal")
        return bool(
            model.get("is_extreme_unseen")
            or model.get("extreme_uncertainty_signal")
            or model.get("unknown_region_active")
            or (
                isinstance(braking, Mapping)
                and (
                    braking.get("is_extreme_unseen")
                    or braking.get("active")
                )
            )
            or (
                isinstance(predictor_signal, Mapping)
                and (
                    predictor_signal.get("is_extreme_unseen")
                    or predictor_signal.get("extreme_uncertainty")
                )
            )
        )

    if flagged(prediction):
        return True

    # 預留其他模型以巢狀結果接入的契約：任一成員亮起極端標籤，
    # 都優先於正常加權與樣本成熟度，立即觸發全局硬熔斷。
    for collection_name in (
        "model_predictions",
        "component_predictions",
        "ensemble_members",
        "models",
    ):
        collection = prediction.get(collection_name)
        if isinstance(collection, Mapping):
            members = collection.values()
        elif isinstance(collection, (list, tuple)):
            members = collection
        else:
            continue
        if any(
            isinstance(member, Mapping) and flagged(member)
            for member in members
        ):
            return True
    return False


def _base_confidence(prediction: Mapping[str, Any], base: Mapping[str, float]) -> float:
    """統一為 [0, 1]；優先沿用 cMAB 已有 quality_score。"""
    try:
        quality = float(prediction.get("quality_score", 0.0) or 0.0)
    except Exception:
        quality = 0.0
    if quality > 0.0:
        return max(0.0, min(1.0, quality))
    return max(0.0, min(1.0, abs(float(base["B"]) - float(base["P"]))))


def adapt_prediction(
    prediction: Mapping[str, Any],
    *,
    venue: str = "",
    room: str = "",
) -> Dict[str, Any]:
    result = dict(prediction or {})
    base = _normalize(
        result.get("probabilities")
        if isinstance(result.get("probabilities"), Mapping)
        else {
            "B": float(result.get("banker_rate", 0.0) or 0.0) / 100.0,
            "P": float(result.get("player_rate", 0.0) or 0.0) / 100.0,
            "T": float(result.get("tie_rate", 0.0) or 0.0) / 100.0,
        }
    )
    result.setdefault("raw_probabilities", dict(base))
    components = _components(result)
    summary = get_performance_summary(venue=venue, room=room, limit=5000)
    sample_count = int(summary.get("sample_count", 0) or 0)
    historical_scores = dict(summary.get("component_brier_scores") or {})
    component_sample_counts = dict(
        summary.get("component_sample_counts") or {}
    )
    extreme_unseen = _is_extreme_unseen(result)
    base_confidence = _base_confidence(result, base)

    if extreme_unseen:
        # 硬熔斷優先於成熟樣本與正常融合；保留原始分數僅供稽核。
        learning_arm = str(result.get("selected_arm") or "").upper().strip()
        result["pre_hard_brake_probabilities"] = dict(base)
        result["pre_hard_brake_recommend"] = str(
            result.get("recommend") or learning_arm
        )
        tie_probability = max(0.0, min(0.30, float(base.get("T", 0.0))))
        neutral_bp = (1.0 - tie_probability) * 0.5
        result["probabilities"] = {
            "B": neutral_bp,
            "P": neutral_bp,
            "T": tie_probability,
        }
        result["banker_rate"] = round(neutral_bp * 100.0, 2)
        result["player_rate"] = round(neutral_bp * 100.0, 2)
        result["tie_rate"] = round(tie_probability * 100.0, 2)
        result["recommend"] = "O"
        result["recommend_text"] = "觀望"
        result["action"] = "O"
        result["action_text"] = "觀望／絕對不下注"
        result["internal_recommend"] = "O"
        result["internal_action"] = "O"
        result["next_round_direction"] = "O"
        result["next_round_direction_text"] = "觀望"
        result["signal_allowed"] = False
        result["signal_status_code"] = "HARD_BRAKE_NO_BET"
        result["signal_status_text"] = "統計混沌硬熔斷：絕對不下注"
        result["signal_reason"] = (
            "任一模型回報極端未知區間；集成層不做方向修正，"
            "本局信心與注碼強制歸零。"
        )
        result["internal_signal_reason"] = result["signal_reason"]
        result["direction_source"] = "adaptive_ensemble_hard_brake"
        result["ensemble_confidence"] = 0.0
        result["confidence"] = 0.0
        result["quality_score"] = 0.0
        result["confidence_label"] = "零信心／硬熔斷"
        result["bet_multiplier"] = 0.0
        result["hard_brake_active"] = True
        # selected_arm 不覆寫：即使不下注，實際結果仍可用固定 1 倍
        # 被動更新模型與共享 context 方差。
        if learning_arm in {"B", "P"}:
            result["selected_arm"] = learning_arm
        result["is_extreme_unseen"] = True
        result["adaptive_ensemble"] = {
            "active": True,
            "mode": "statistical_chaos_hard_brake",
            "circuit_breaker_active": True,
            "hard_brake_active": True,
            "sample_count": sample_count,
            "minimum_samples_bypassed_for_safety": True,
            "is_extreme_unseen": True,
            "variance": float(result.get("variance", 0.0) or 0.0),
            "bandit_weight_before": 1.0,
            "bandit_weight_after": 0.0,
            "weight_reduction_ratio": 1.0,
            "weight_reduction_percent": 100.0,
            "alternative_model_weight": 0.0,
            "alternative_components_available": sorted(components),
            "alternative_fusion_attempted": False,
            "fallback_required": True,
            "shadow_backtest_required": True,
            "overall_confidence": 0.0,
            "final_action": "O",
            "bet_multiplier": 0.0,
            "reason": "極端未知區間執行零信心、零注碼硬熔斷",
        }
        return result

    champion = _select_component_champion(
        result,
        components,
        historical_scores,
        component_sample_counts,
    ) if ADAPTIVE_CHAMPION_ENABLED else {}
    base_action = str(result.get("action") or "O").upper().strip()
    try:
        base_edge = max(0.0, float(result.get("direction_edge", 0.0) or 0.0))
    except Exception:
        base_edge = 0.0
    champion_takeover = bool(
        champion
        and (
            base_action not in {"B", "P"}
            or (
                bool(champion.get("models_conflict"))
                and base_edge < ADAPTIVE_CHAMPION_OVERRIDE_BASE_EDGE
            )
        )
    )
    if champion_takeover:
        # 低優勢或模型衝突時不再把所有方向平均回 50/50；改由當前
        # 可靠度最高、或已有足夠歷史 Brier 證據的單一核心模型接管。
        # 真正 extreme_unseen 已在上方返回，因此此分支絕不繞過硬熔斷。
        result["pre_champion_probabilities"] = dict(base)
        tie_probability = max(0.0, min(0.30, float(base.get("T", 0.0))))
        bp_mass = 1.0 - tie_probability
        conditional_banker = float(
            champion["tempered_banker_probability"]
        )
        banker = bp_mass * conditional_banker
        player = bp_mass * (1.0 - conditional_banker)
        direction = str(champion["direction"])
        champion_edge = float(champion["tempered_edge"])
        champion_confidence = max(
            0.50,
            min(
                0.78,
                0.44
                + 0.24 * float(champion["reliability"])
                + 0.22 * min(1.0, champion_edge),
            ),
        )
        result["probabilities"] = {
            "B": float(banker),
            "P": float(player),
            "T": float(tie_probability),
        }
        result["banker_rate"] = round(banker * 100.0, 2)
        result["player_rate"] = round(player * 100.0, 2)
        result["tie_rate"] = round(tie_probability * 100.0, 2)
        result["recommend"] = direction
        result["recommend_text"] = "莊" if direction == "B" else "閒"
        result["action"] = direction
        result["action_text"] = result["recommend_text"]
        result["internal_recommend"] = direction
        result["internal_action"] = direction
        result["next_round_direction"] = direction
        result["next_round_direction_text"] = result["recommend_text"]
        result["signal_allowed"] = True
        result["signal_status_code"] = "DYNAMIC_CHAMPION_DIRECTION"
        result["signal_status_text"] = "核心模型優先：正式方向已啟用"
        result["signal_reason"] = (
            f"多模型低優勢／衝突時停止平均抵消，改由核心模型 "
            f"{champion['name']} 依"
            + (
                f" {champion['performance_samples']} 局歷史 Brier 表現"
                if champion["historical_performance_used"]
                else "當前牌路可靠度與方向差"
            )
            + "接管莊／閒方向。"
        )
        result["internal_signal_reason"] = result["signal_reason"]
        result["direction_source"] = "adaptive_ensemble_dynamic_champion"
        result["direction_edge"] = champion_edge
        result["direction_edge_percent"] = round(champion_edge * 100.0, 4)
        result["ensemble_confidence"] = champion_confidence
        result["confidence"] = champion_confidence
        result["quality_score"] = max(
            float(result.get("quality_score", 0.0) or 0.0),
            champion_confidence,
        )
        result["confidence_label"] = (
            "較高"
            if champion_confidence >= 0.72
            else "中等"
            if champion_confidence >= 0.56
            else "偏低"
        )
        result["bet_multiplier"] = 1.0
        result["hard_brake_active"] = False
        result["is_extreme_unseen"] = False
        result["component_champion"] = dict(champion)
        result["adaptive_ensemble"] = {
            "active": True,
            "mode": "dynamic_component_champion",
            "circuit_breaker_active": False,
            "hard_brake_active": False,
            "is_extreme_unseen": False,
            "sample_count": sample_count,
            "base_action_before": base_action,
            "base_edge_before": base_edge,
            "models_conflict": bool(champion["models_conflict"]),
            "champion_model": str(champion["name"]),
            "champion_direction": direction,
            "champion_edge": champion_edge,
            "champion_reliability": float(champion["reliability"]),
            "historical_performance_used": bool(
                champion["historical_performance_used"]
            ),
            "overall_confidence": champion_confidence,
            "final_action": direction,
            "bet_multiplier": 1.0,
            "fallback_required": False,
            "reason": "低優勢／衝突時由單一核心模型接管，避免平均抵消",
        }
        return result

    eligible = {
        name: probabilities
        for name, probabilities in components.items()
        if name in historical_scores
        and int(component_sample_counts.get(name, 0) or 0) >= ADAPTIVE_MIN_SAMPLES
    }
    active = bool(
        ADAPTIVE_ENABLED
        and sample_count >= ADAPTIVE_MIN_SAMPLES
        and len(eligible) >= 2
        and ADAPTIVE_MAX_SHARE > 0
    )

    if not active:
        result["ensemble_confidence"] = float(base_confidence)
        result["confidence"] = float(base_confidence)
        result["hard_brake_active"] = False
        result["adaptive_ensemble"] = {
            "active": False,
            "circuit_breaker_active": False,
            "hard_brake_active": False,
            "is_extreme_unseen": False,
            "sample_count": sample_count,
            "minimum_samples": ADAPTIVE_MIN_SAMPLES,
            "reason": "已結算樣本不足或可比較子模型不足",
            "effective_share": 0.0,
            "bandit_weight_before": 1.0,
            "bandit_weight_after": 1.0,
            "weight_reduction_ratio": 0.0,
            "fallback_required": False,
            "overall_confidence": float(base_confidence),
        }
        return result

    raw_weights = {
        name: math.exp(-float(historical_scores[name]) / ADAPTIVE_TEMPERATURE)
        for name in eligible
    }
    weight_total = sum(raw_weights.values()) or 1.0
    weights = {name: value / weight_total for name, value in raw_weights.items()}
    adaptive = {
        outcome: sum(weights[name] * eligible[name][outcome] for name in eligible)
        for outcome in OUTCOMES
    }
    maturity = min(1.0, (sample_count - ADAPTIVE_MIN_SAMPLES + 1) / max(1.0, ADAPTIVE_MIN_SAMPLES * 3.0))
    share = ADAPTIVE_MAX_SHARE * maturity
    blended = _normalize({
        outcome: base[outcome] * (1.0 - share) + adaptive[outcome] * share
        for outcome in OUTCOMES
    })
    result["probabilities"] = blended
    result["banker_rate"] = round(blended["B"] * 100.0, 2)
    result["player_rate"] = round(blended["P"] * 100.0, 2)
    result["tie_rate"] = round(blended["T"] * 100.0, 2)
    result["ensemble_confidence"] = float(base_confidence)
    result["confidence"] = float(base_confidence)
    result["hard_brake_active"] = False
    result["adaptive_ensemble"] = {
        "active": True,
        "circuit_breaker_active": False,
        "hard_brake_active": False,
        "is_extreme_unseen": False,
        "sample_count": sample_count,
        "minimum_samples": ADAPTIVE_MIN_SAMPLES,
        "effective_share": round(share, 6),
        "bandit_weight_before": 1.0,
        "bandit_weight_after": round(1.0 - share, 6),
        "weight_reduction_ratio": round(share, 6),
        "fallback_required": False,
        "overall_confidence": float(base_confidence),
        "weights": {name: round(value, 6) for name, value in weights.items()},
        "component_brier_scores": {
            name: round(float(historical_scores[name]), 6) for name in eligible
        },
    }
    return result


__all__ = ["adapt_prediction"]
