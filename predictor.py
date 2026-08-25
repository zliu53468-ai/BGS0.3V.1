"""BGS unified predictor: Full Road Adaptive Ensemble + CUSUM-LinUCB.

This is the previous V35.0 road + CUSUM architecture, with one deliberate change:
formal output always stays B/P. CUSUM reset still reduces confidence and expert
weight during the rebuild period, but it no longer emits Observe (O).
"""
from __future__ import annotations

from hashlib import sha256
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
import math
import secrets

from adaptive_ensemble import adapt_prediction
from contextual_bandit import MODEL_VERSION as CUSUM_MODEL_VERSION, predict_bandit
from road_model import build_road_context

ADAPTIVE_MODEL_VARIANT = "V35.0_ROAD_ADAPTIVE_PLUS_CUSUM_LINUCB_MARKOV_CROSS_RESONANCE"
MAX_CUSUM_ENSEMBLE_WEIGHT = 0.45
MARKOV_RESONANCE_THRESHOLD = 0.53
MARKOV_COUNTER_THRESHOLD = 0.55
MARKOV_FUSION_ALPHA = 0.62
MARKOV_LOCAL_FALLBACK_WINDOW = 36

DB_HOLDOUT: Dict[str, Any] = {
    "status": "removed",
    "replacement": "FULL_ROAD_ADAPTIVE_PLUS_CUSUM_LINUCB",
    "note": "正式方向由 Full Road Adaptive 與 CUSUM-LinUCB 動態融合；不使用舊算牌/粒子層。",
}


def _normalize_outcome_history(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for item in values:
        if isinstance(item, Mapping):
            raw = (
                item.get("outcome")
                or item.get("actual")
                or item.get("actual_outcome")
                or item.get("virtual_outcome")
            )
        else:
            raw = item
        value = str(raw or "").upper().strip()
        if value in {"B", "P", "T"}:
            result.append(value)
    return result[-2000:]


def _bandit_learning_scope(*, user_id: str, venue: str, room: str) -> str:
    raw = "|".join((
        str(user_id or "__anonymous__"),
        str(venue or "").upper().strip(),
        str(room or "").strip(),
    ))
    return "__cusum_scope__:" + sha256(raw.encode("utf-8")).hexdigest()[:32]


def _road_seed_prediction(road: Mapping[str, Any], history: Iterable[Any]) -> Dict[str, Any]:
    try:
        banker = float(road.get("banker_probability", 0.5) or 0.5)
    except (TypeError, ValueError):
        banker = 0.5
    try:
        player = float(road.get("player_probability", 1.0 - banker) or 0.0)
    except (TypeError, ValueError):
        player = 1.0 - banker
    bp_total = banker + player
    if bp_total <= 1e-12:
        banker, player = 0.5, 0.5
    else:
        banker, player = banker / bp_total, player / bp_total

    try:
        observed_tie_rate = float(road.get("observed_tie_rate", 0.0) or 0.0)
    except (TypeError, ValueError):
        observed_tie_rate = 0.0
    tie = max(0.0, min(0.30, observed_tie_rate))
    bp_mass = 1.0 - tie
    banker_probability = bp_mass * banker
    player_probability = bp_mass * player

    direction = str(road.get("direction") or "").upper().strip()
    if direction not in {"B", "P"}:
        direction = "B" if banker >= player else "P"

    normalized_history = _normalize_outcome_history(history)
    history_fingerprint = sha256(
        "|".join(normalized_history).encode("utf-8")
    ).hexdigest()[:24]
    return {
        "model_version": "FULL_ROAD_ADAPTIVE_V35.0",
        "model_variant": ADAPTIVE_MODEL_VARIANT,
        "prediction_fingerprint": history_fingerprint,
        "probabilities": {
            "B": float(banker_probability),
            "P": float(player_probability),
            "T": float(tie),
        },
        "raw_probabilities": {
            "B": float(banker_probability),
            "P": float(player_probability),
            "T": float(tie),
        },
        "banker_rate": round(banker_probability * 100.0, 2),
        "player_rate": round(player_probability * 100.0, 2),
        "tie_rate": round(tie * 100.0, 2),
        "action": direction,
        "recommend": direction,
        "internal_action": direction,
        "internal_recommend": direction,
        "next_round_direction": direction,
    }


def _normalize_probabilities(values: Any) -> Dict[str, float]:
    if not isinstance(values, Mapping):
        return {"B": 0.455, "P": 0.455, "T": 0.09}
    raw = {
        key: max(0.0, float(values.get(key, 0.0) or 0.0))
        for key in ("B", "P", "T")
    }
    total = sum(raw.values())
    if total <= 1e-12:
        return {"B": 0.455, "P": 0.455, "T": 0.09}
    return {key: raw[key] / total for key in raw}


def _conditional_banker(values: Any) -> float:
    probabilities = _normalize_probabilities(values)
    bp_total = probabilities["B"] + probabilities["P"]
    if bp_total <= 1e-12:
        return 0.5
    return float(probabilities["B"] / bp_total)


def _safe_confidence(result: Mapping[str, Any], default: float = 0.5) -> float:
    for key in ("ensemble_confidence", "confidence", "quality_score"):
        try:
            value = float(result.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if value > 0.0:
            return max(0.0, min(1.0, value))
    return max(0.0, min(1.0, default))


class DualChannelMarkovFusion:
    """Independent fusion-side Markov analyzer; does not update LinUCB/CUSUM state.

    Global channel uses the whole B/P shoe. Local channel consumes the regime-aware
    Markov state emitted by contextual_bandit.py (which is reset by CUSUM). Both
    channels calculate first- and full second-order transitions with Laplace smoothing.
    """

    def __init__(self, *, alpha: float = 1.0, support_prior: float = 5.0) -> None:
        self.alpha = max(1e-9, float(alpha))
        self.support_prior = max(1e-9, float(support_prior))

    @staticmethod
    def _bp(values: Iterable[Any]) -> List[str]:
        out: List[str] = []
        for value in values:
            if isinstance(value, bool):
                continue
            if isinstance(value, int) and value in (0, 1):
                out.append("B" if value == 1 else "P")
                continue
            text = str(value or "").upper().strip()
            if text in {"B", "P"}:
                out.append(text)
        return out

    def _pair(self, banker_count: int, player_count: int) -> Dict[str, float]:
        support = int(banker_count + player_count)
        denominator = support + 2.0 * self.alpha
        p_b = (float(banker_count) + self.alpha) / denominator
        p_p = (float(player_count) + self.alpha) / denominator
        reliability = support / (support + self.support_prior)
        return {
            "B": float(p_b),
            "P": float(p_p),
            "support": support,
            "reliability": float(reliability),
        }

    def _conditional(self, sequence: List[str], context: str) -> Dict[str, float]:
        order = len(context)
        banker_count = 0
        player_count = 0
        if order > 0:
            for i in range(order, len(sequence)):
                if "".join(sequence[i-order:i]) != context:
                    continue
                if sequence[i] == "B":
                    banker_count += 1
                else:
                    player_count += 1
        result = self._pair(banker_count, player_count)
        result["context"] = context
        result["order"] = order
        return result

    def channel(self, values: Iterable[Any]) -> Dict[str, Any]:
        sequence = self._bp(values)
        state1 = sequence[-1] if sequence else ""
        state2 = "".join(sequence[-2:]) if len(sequence) >= 2 else ""
        first = self._conditional(sequence, state1) if state1 else self._pair(0, 0)
        if state1 == "":
            first.update({"context": "", "order": 1})
        second = self._conditional(sequence, state2) if state2 else self._pair(0, 0)
        if state2 == "":
            second.update({"context": "", "order": 2})

        # Support-aware backoff. Second order is preferred only when the exact active
        # state has actually occurred enough times; otherwise first order anchors it.
        second_weight = float(second.get("reliability", 0.0))
        active_b = (
            second_weight * float(second["B"])
            + (1.0 - second_weight) * float(first["B"])
        )
        active_b = max(1e-6, min(1.0 - 1e-6, active_b))
        active_reliability = 1.0 - (
            (1.0 - float(first.get("reliability", 0.0)))
            * (1.0 - float(second.get("reliability", 0.0)))
        )
        return {
            "sample_count": len(sequence),
            "state_order1": state1,
            "state_order2": state2,
            "first_order": first,
            "second_order": second,
            "active_probability": {"B": active_b, "P": 1.0 - active_b},
            "active_direction": "B" if active_b >= 0.5 else "P",
            "reliability": float(active_reliability),
        }

    @staticmethod
    def _logit(probability: float) -> float:
        p = max(1e-6, min(1.0 - 1e-6, float(probability)))
        return math.log(p / (1.0 - p))

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0.0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    def analyze(
        self,
        history: Iterable[Any],
        bandit_prediction: Mapping[str, Any],
    ) -> Dict[str, Any]:
        global_channel = self.channel(history)
        markov_state = bandit_prediction.get("markov_state")
        local_values: List[Any] = []
        if isinstance(markov_state, Mapping):
            candidate = markov_state.get("values")
            if isinstance(candidate, list):
                local_values = list(candidate)
        if not local_values:
            local_values = self._bp(history)[-MARKOV_LOCAL_FALLBACK_WINDOW:]
        local_channel = self.channel(local_values)

        global_b = float(global_channel["active_probability"]["B"])
        local_b = float(local_channel["active_probability"]["B"])
        global_rel = float(global_channel.get("reliability", 0.0))
        local_rel = float(local_channel.get("reliability", 0.0))

        # Geometric/Bayesian-style odds pooling. Global remains the stable anchor;
        # local receives more weight as the post-CUSUM regime accumulates evidence.
        global_weight = 0.50 + 0.20 * global_rel
        local_weight = 0.50 + 0.35 * local_rel
        weight_total = max(1e-9, global_weight + local_weight)
        pooled_logit = (
            global_weight * self._logit(global_b)
            + local_weight * self._logit(local_b)
        ) / weight_total
        fused_b = self._sigmoid(pooled_logit)
        return {
            "global": global_channel,
            "local": local_channel,
            "fused_probability": {"B": fused_b, "P": 1.0 - fused_b},
            "direction": "B" if fused_b >= 0.5 else "P",
            "weights": {
                "global": global_weight / weight_total,
                "local": local_weight / weight_total,
            },
        }


_MARKOV_FUSION = DualChannelMarkovFusion()


def _apply_cross_resonance(
    fused_prediction: Mapping[str, Any],
    adaptive_prediction: Mapping[str, Any],
    bandit_prediction: Mapping[str, Any],
    history: Iterable[Any],
) -> Dict[str, Any]:
    """Calibrate the existing Adaptive+CUSUM output with dual-channel Markov evidence."""
    result = dict(fused_prediction or {})
    adaptive = dict(adaptive_prediction or {})
    base_probs = _normalize_probabilities(result.get("probabilities"))
    road_probs = _normalize_probabilities(
        adaptive.get("adaptive_only_probabilities")
        if isinstance(adaptive.get("adaptive_only_probabilities"), Mapping)
        else adaptive.get("probabilities")
    )
    base_b = _conditional_banker(base_probs)
    road_b = _conditional_banker(road_probs)
    road_direction = "B" if road_b >= 0.5 else "P"
    road_side_probability = road_b if road_direction == "B" else 1.0 - road_b

    markov = _MARKOV_FUSION.analyze(history, bandit_prediction)
    global_b = float(markov["global"]["active_probability"]["B"])
    local_b = float(markov["local"]["active_probability"]["B"])
    markov_b = float(markov["fused_probability"]["B"])
    global_side = global_b if road_direction == "B" else 1.0 - global_b
    local_side = local_b if road_direction == "B" else 1.0 - local_b
    markov_side = markov_b if road_direction == "B" else 1.0 - markov_b
    opposite = "P" if road_direction == "B" else "B"
    global_opp = 1.0 - global_side
    local_opp = 1.0 - local_side

    dual_confirm = (
        global_side >= MARKOV_RESONANCE_THRESHOLD
        and local_side >= MARKOV_RESONANCE_THRESHOLD
    )
    strong_counter = (
        global_opp >= MARKOV_COUNTER_THRESHOLD
        and local_opp >= MARKOV_COUNTER_THRESHOLD
        and min(
            float(markov["global"].get("reliability", 0.0)),
            float(markov["local"].get("reliability", 0.0)),
        ) >= 0.20
    )
    near_random = (
        abs(global_b - 0.5) < 0.03
        and abs(local_b - 0.5) < 0.03
    )

    alpha = MARKOV_FUSION_ALPHA
    base_confidence = _safe_confidence(result, 0.5)
    original_bet = max(0.0, min(1.0, float(result.get("bet_multiplier", 1.0) or 0.0)))

    if dual_confirm:
        road_markov_b = alpha * road_b + (1.0 - alpha) * markov_b
        final_b = 0.72 * road_markov_b + 0.28 * base_b
        classification = "TRUE_PATTERN_RESONANCE"
        confidence = min(
            0.90,
            max(base_confidence, 0.54 + 0.80 * abs(final_b - 0.5)),
        )
        bet_multiplier = min(1.0, original_bet * (0.95 + 0.10 * confidence))
        reason = "Road 與全局/局部 Markov 同向且皆通過 53% 共鳴門檻。"
    elif strong_counter:
        # Two independent Markov channels both reject the road pattern. Allow a
        # controlled counter-direction override instead of blindly following graphics.
        final_b = 0.32 * base_b + 0.68 * markov_b
        classification = "FALSE_PATTERN_COUNTER_SIGNAL"
        confidence = min(0.76, 0.52 + 0.75 * abs(final_b - 0.5))
        bet_multiplier = min(original_bet, max(0.30, original_bet * 0.72))
        reason = f"Road 偏 {road_direction}，但全局/局部 Markov 同時強烈偏 {opposite}。"
    elif near_random:
        # Markov carries no useful confirmation; shrink the apparent road edge.
        final_b = 0.5 + 0.52 * (base_b - 0.5)
        classification = "RANDOM_NOISE_LOW_CONFIDENCE"
        confidence = min(0.52, base_confidence * 0.72)
        bet_multiplier = min(original_bet, original_bet * 0.50)
        reason = "Markov 接近 50/50，Road 圖形缺乏統計共鳴，降低信心。"
    else:
        # Mixed evidence: retain the original fusion direction tendency but compress
        # the edge and let Markov contribute only as a weak calibrator.
        final_b = 0.5 + 0.38 * (base_b - 0.5) + 0.18 * (markov_b - 0.5)
        classification = "MIXED_OR_FALSE_PATTERN"
        confidence = min(0.54, base_confidence * 0.78)
        bet_multiplier = min(original_bet, original_bet * 0.58)
        reason = "Road 與雙通道 Markov 未形成雙重確認，視為混合/假規律區。"

    final_b = max(1e-6, min(1.0 - 1e-6, final_b))
    final_direction = "B" if final_b > 0.5 else "P" if final_b < 0.5 else road_direction
    tie_probability = max(0.0, min(0.30, base_probs["T"]))
    bp_mass = 1.0 - tie_probability
    banker_probability = bp_mass * final_b
    player_probability = bp_mass * (1.0 - final_b)
    final_text = "莊" if final_direction == "B" else "閒"

    road_predict = {
        "direction": road_direction,
        "banker_probability": road_b,
        "player_probability": 1.0 - road_b,
        "selected_probability": road_side_probability,
    }
    markov_predict = {
        **markov,
        "road_direction_probability": {
            "global": global_side,
            "local": local_side,
            "fused": markov_side,
        },
        "threshold": MARKOV_RESONANCE_THRESHOLD,
    }
    fusion_decision = {
        "classification": classification,
        "direction": final_direction,
        "banker_probability": final_b,
        "player_probability": 1.0 - final_b,
        "alpha_road": alpha,
        "reason": reason,
    }

    result.update({
        "model_variant": ADAPTIVE_MODEL_VARIANT,
        "decision_pipeline": "full_road_adaptive_cusum_then_dual_markov_cross_resonance",
        "probabilities": {
            "B": float(banker_probability),
            "P": float(player_probability),
            "T": float(tie_probability),
        },
        "banker_rate": round(banker_probability * 100.0, 2),
        "player_rate": round(player_probability * 100.0, 2),
        "tie_rate": round(tie_probability * 100.0, 2),
        "recommend": final_direction,
        "recommend_text": final_text,
        "action": final_direction,
        "action_text": final_text,
        "internal_recommend": final_direction,
        "internal_action": final_direction,
        "next_round_direction": final_direction,
        "next_round_direction_text": final_text,
        "confidence": float(confidence),
        "ensemble_confidence": float(confidence),
        "quality_score": float(confidence),
        "confidence_label": (
            "較高" if confidence >= 0.72 else
            "中等" if confidence >= 0.55 else "偏低"
        ),
        "bet_multiplier": float(bet_multiplier),
        "direction_edge": float(abs(2.0 * final_b - 1.0)),
        "direction_edge_percent": round(abs(2.0 * final_b - 1.0) * 100.0, 4),
        "direction_source": "road_markov_cross_resonance_fusion",
        "signal_status_code": "ROAD_MARKOV_CROSS_RESONANCE",
        "signal_status_text": "Road + Dual Markov Cross-Resonance",
        "signal_reason": reason,
        "internal_signal_reason": classification,
        "road_predict": road_predict,
        "markov_predict": markov_predict,
        "fusion_decision": fusion_decision,
        "cross_resonance": {
            "active": True,
            "classification": classification,
            "dual_confirm": dual_confirm,
            "strong_counter": strong_counter,
            "near_random": near_random,
            "road_predict": road_predict,
            "markov_predict": markov_predict,
            "fusion_decision": fusion_decision,
            "confidence": float(confidence),
            "bet_multiplier": float(bet_multiplier),
        },
    })
    adaptive_ensemble = dict(result.get("adaptive_ensemble") or {})
    adaptive_ensemble.update({
        "mode": "adaptive_road_plus_cusum_plus_dual_markov_cross_resonance",
        "overall_confidence": float(confidence),
        "bet_multiplier": float(bet_multiplier),
        "final_action": final_direction,
        "cross_resonance_classification": classification,
    })
    result["adaptive_ensemble"] = adaptive_ensemble
    return result


def _fuse_adaptive_and_cusum(
    adaptive_prediction: Mapping[str, Any],
    bandit_prediction: Mapping[str, Any],
) -> Dict[str, Any]:
    """Fuse Adaptive road model and CUSUM expert; always return B/P."""
    result = dict(adaptive_prediction or {})
    bandit = dict(bandit_prediction or {})
    risk = dict(bandit.get("risk_control") or {})

    adaptive_probs = _normalize_probabilities(
        result.get("adaptive_only_probabilities")
        if isinstance(result.get("adaptive_only_probabilities"), Mapping)
        else result.get("probabilities")
    )
    bandit_probs = _normalize_probabilities(bandit.get("probabilities"))
    adaptive_banker = _conditional_banker(adaptive_probs)
    bandit_banker = _conditional_banker(bandit_probs)

    try:
        requested_weight = float(
            bandit.get(
                "ensemble_weight_suggestion",
                risk.get("ensemble_weight_suggestion", 0.0),
            ) or 0.0
        )
    except (TypeError, ValueError):
        requested_weight = 0.0
    bandit_weight = max(0.0, min(MAX_CUSUM_ENSEMBLE_WEIGHT, requested_weight))

    vacuum_active = bool(
        bandit.get("post_reset_vacuum_active")
        or risk.get("post_reset_vacuum_active")
    )
    # Explicitly disable formal Observe. Reset only reduces weight/confidence.
    force_observe = False

    road_weight = 1.0 - bandit_weight
    conditional_banker = (
        road_weight * adaptive_banker + bandit_weight * bandit_banker
    )
    conditional_banker = max(1e-6, min(1.0 - 1e-6, conditional_banker))

    tie_probability = max(0.0, min(0.30, adaptive_probs["T"]))
    bp_mass = 1.0 - tie_probability
    banker_probability = bp_mass * conditional_banker
    player_probability = bp_mass * (1.0 - conditional_banker)

    adaptive_direction = str(
        result.get("adaptive_only_direction")
        or result.get("action")
        or result.get("recommend")
        or ""
    ).upper().strip()
    if adaptive_direction not in {"B", "P"}:
        adaptive_direction = "B" if adaptive_banker >= 0.5 else "P"

    bandit_direction = str(
        bandit.get("selected_arm")
        or bandit.get("next_round_direction")
        or bandit.get("recommend")
        or ""
    ).upper().strip()
    if bandit_direction not in {"B", "P"}:
        bandit_direction = "B" if bandit_banker >= 0.5 else "P"

    if abs(conditional_banker - 0.5) <= 1e-12:
        final_direction = adaptive_direction
    else:
        final_direction = "B" if conditional_banker > 0.5 else "P"

    adaptive_confidence = _safe_confidence(result, 0.5)
    try:
        bandit_confidence = float(
            bandit.get("confidence_score", risk.get("confidence_score", 0.0)) or 0.0
        )
    except (TypeError, ValueError):
        bandit_confidence = 0.0
    bandit_confidence = max(0.0, min(1.0, bandit_confidence))
    confidence = road_weight * adaptive_confidence + bandit_weight * bandit_confidence
    if vacuum_active:
        confidence = min(confidence, 0.50)

    try:
        risk_bet_multiplier = float(
            bandit.get("bet_multiplier", risk.get("bet_multiplier", 1.0)) or 0.0
        )
    except (TypeError, ValueError):
        risk_bet_multiplier = 1.0
    risk_bet_multiplier = max(0.0, min(1.0, risk_bet_multiplier))
    bet_multiplier = risk_bet_multiplier

    result.update({
        "model_version": f"FULL-ROAD-ADAPTIVE+CUSUM-LINUCB::{CUSUM_MODEL_VERSION}",
        "model_variant": ADAPTIVE_MODEL_VARIANT,
        "decision_pipeline": "full_road_adaptive_then_cusum_linucb_dynamic_fusion_no_observe",
        "probabilities": {
            "B": float(banker_probability),
            "P": float(player_probability),
            "T": float(tie_probability),
        },
        "banker_rate": round(banker_probability * 100.0, 2),
        "player_rate": round(player_probability * 100.0, 2),
        "tie_rate": round(tie_probability * 100.0, 2),
        "adaptive_only_direction": adaptive_direction,
        "bandit_only_direction": bandit_direction,
        "contextual_bandit_enabled": True,
        "contextual_bandit_update_enabled": True,
        "cusum_linucb_enabled": True,
        "ucb_influenced_final_direction": bool(bandit_weight > 0.0),
        "post_reset_vacuum_active": vacuum_active,
        "force_observe": False,
        "ensemble_confidence": float(confidence),
        "confidence": float(confidence),
        "quality_score": float(confidence),
        "confidence_label": (
            "重置低權重期" if vacuum_active else
            "較高" if confidence >= 0.72 else
            "中等" if confidence >= 0.55 else "偏低"
        ),
        "bet_multiplier": float(bet_multiplier),
        "cusum_bandit": bandit,
        "ensemble_scheduler": {
            "road_weight": float(road_weight),
            "cusum_linucb_weight": float(bandit_weight),
            "requested_cusum_weight": float(requested_weight),
            "max_cusum_weight": float(MAX_CUSUM_ENSEMBLE_WEIGHT),
            "post_reset_vacuum_active": vacuum_active,
            "force_observe": False,
            "observations_since_reset": int(
                risk.get(
                    "observations_since_reset",
                    bandit.get("observations_since_reset", 0),
                ) or 0
            ),
            "reason": (
                "CUSUM 重置後低權重恢復期：仍輸出 B/P"
                if vacuum_active else
                "正常區間：依 CUSUM-LinUCB confidence 動態分配權重"
            ),
        },
    })

    adaptive = dict(result.get("adaptive_ensemble") or {})
    adaptive.update({
        "active": True,
        "mode": "adaptive_road_plus_cusum_linucb_no_observe",
        "contextual_bandit_enabled": True,
        "cusum_linucb_enabled": True,
        "cusum_linucb_weight": float(bandit_weight),
        "road_weight": float(road_weight),
        "post_reset_vacuum_active": vacuum_active,
        "overall_confidence": float(confidence),
        "bet_multiplier": float(bet_multiplier),
        "hard_brake_active": False,
        "circuit_breaker_active": False,
        "final_action": final_direction,
    })
    result["adaptive_ensemble"] = adaptive

    final_text = "莊" if final_direction == "B" else "閒"
    result.update({
        "recommend": final_direction,
        "recommend_text": final_text,
        "action": final_direction,
        "action_text": final_text,
        "internal_recommend": final_direction,
        "internal_action": final_direction,
        "next_round_direction": final_direction,
        "next_round_direction_text": final_text,
        "signal_allowed": True,
        "signal_status_code": "ADAPTIVE_CUSUM_LINUCB_FUSION",
        "signal_status_text": "Full Road Adaptive + CUSUM-LinUCB 動態融合",
        "signal_reason": (
            f"CUSUM 重置後低權重恢復；Adaptive 權重 {road_weight:.3f}；CUSUM-LinUCB 權重 {bandit_weight:.3f}。"
            if vacuum_active else
            f"Adaptive 權重 {road_weight:.3f}；CUSUM-LinUCB 權重 {bandit_weight:.3f}。"
        ),
        "internal_signal_reason": "ADAPTIVE_CUSUM_DYNAMIC_WEIGHT_NO_OBSERVE",
        "direction_source": "adaptive_cusum_linucb_fusion",
        "hard_brake_active": False,
        "is_extreme_unseen": False,
    })

    result["direction_edge"] = float(abs(2.0 * conditional_banker - 1.0))
    result["direction_edge_percent"] = round(result["direction_edge"] * 100.0, 4)
    return result


class ShadowBacktestController:
    """Compatibility wrapper retained for callers from earlier predictor versions."""
    def __init__(self) -> None:
        self.shadow_buffer: List[str] = []

    @staticmethod
    def stream_key(*, user_id: str, venue: str, room: str, shoe_id: str) -> str:
        raw = "|".join((
            str(user_id or "__anonymous__"),
            str(venue or "").upper().strip(),
            str(room or "").strip(),
            str(shoe_id or "").strip(),
        ))
        return sha256(raw.encode("utf-8")).hexdigest()[:24]

    def apply(
        self,
        history: List[str],
        prediction: Mapping[str, Any],
        *,
        stream_key: str = "__default__",
    ) -> Dict[str, Any]:
        del stream_key
        self.shadow_buffer = [value for value in history if value in {"B", "P"}][-3:]
        result = dict(prediction or {})
        result.setdefault("shadow_buffer", list(self.shadow_buffer))
        return result


ShortTermTakeoverController = ShadowBacktestController
_SHADOW_CONTROLLER = ShadowBacktestController()
_SHORT_TERM_CONTROLLER = _SHADOW_CONTROLLER


def predict(
    history: Union[str, Iterable[Any], None] = None,
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
    user_id: str = "",
    run_seed: Optional[int] = None,
    shoe_context: Optional[Mapping[str, Any]] = None,
    road_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Unified API compatible with app.py and screenshot_predictor.py."""
    if history is None:
        history_values: List[Any] = []
    elif isinstance(history, str):
        history_values = [
            part for part in history.replace("|", ",").split(",") if part.strip()
        ]
    else:
        history_values = list(history)
    cleaned = _normalize_outcome_history(history_values)

    road = dict(road_context or {})
    if not isinstance(road.get("models"), Mapping):
        road = build_road_context(cleaned, seed=run_seed)

    road_seed = _road_seed_prediction(road, cleaned)
    road_seed["road_support"] = dict(road)
    road_seed["component_probabilities"] = dict(
        road.get("component_probabilities") or {}
    )
    road_seed["decision_pipeline"] = "full_road_pattern_to_adaptive_ensemble"
    road_seed["model_variant"] = ADAPTIVE_MODEL_VARIANT
    adaptive = adapt_prediction(
        road_seed,
        venue=venue,
        room=room,
        shoe_id=shoe_id,
    )

    bandit_scope = _bandit_learning_scope(
        user_id=user_id,
        venue=venue,
        room=room,
    )
    bandit = predict_bandit(
        cleaned,
        road_context=road,
        venue=venue,
        room=room,
        user_id=bandit_scope,
        run_seed=run_seed,
    )
    result = _fuse_adaptive_and_cusum(adaptive, bandit)
    result = _apply_cross_resonance(result, adaptive, bandit, cleaned)

    model_fingerprint = str(result.get("prediction_fingerprint") or "").strip()
    bandit_fingerprint = str(bandit.get("prediction_fingerprint") or "").strip()
    result["model_prediction_fingerprint"] = model_fingerprint
    result["prediction_fingerprint"] = sha256(
        "|".join((
            model_fingerprint,
            bandit_fingerprint,
            str(venue or "").upper().strip(),
            str(room or "").strip(),
            str(shoe_id or "__unspecified_shoe__").strip(),
        )).encode("utf-8")
    ).hexdigest()[:24]
    result.update({
        "shoe_id": str(shoe_id or ""),
        "bandit_learning_user_id": bandit_scope,
        "bandit_scope_mode": "user_venue_room_hashed",
        "bandit_shoe_isolated": False,
        "shoe_event_isolated": True,
        "composition_quality": "not_applicable_road_cusum",
        "remaining_counts_source": "not_used",
        "shoe_context_ignored": bool(shoe_context),
        "road_quality_ok": bool(
            road.get("quality_ok", road.get("recognition_quality_ok", True))
        ),
        "road_support": dict(road),
        "component_probabilities": dict(
            road.get("component_probabilities") or {}
        ),
        "input_required": False,
        "probability_semantics": "direction_score_not_guaranteed_outcome_probability",
    })
    return result


def run_virtual_round(
    session: Mapping[str, Any],
    run_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Retain virtual-shoe compatibility API used by app.py."""
    from particle_filter_points import counts_from_shoe, deal_ordered_hand

    hidden_shoe = [int(card) for card in list(session.get("virtual_shoe") or [])]
    if len(hidden_shoe) < 6:
        raise ValueError("虛擬牌靴不足，請重新建立牌靴。")
    outcome_history = _normalize_outcome_history(
        list(session.get("round_history") or [])
    )
    seed = int(
        run_seed if run_seed is not None else secrets.randbits(32)
    ) & 0xFFFFFFFF
    prediction = predict(
        history=outcome_history,
        venue=str(session.get("venue") or ""),
        room=str(session.get("room") or ""),
        shoe_id=str(session.get("shoe_id") or ""),
        user_id=str(session.get("user_id") or ""),
        run_seed=seed,
        road_context=None,
    )
    hand, remaining_shoe = deal_ordered_hand(hidden_shoe)
    hand_data = hand.as_dict()
    predicted_side = str(
        prediction.get("action") or prediction.get("recommend") or ""
    ).upper()
    actual = str(hand.outcome or "").upper()
    verdict = (
        "TIE_SKIPPED" if actual == "T" else
        "HIT" if predicted_side == actual else "MISS"
    )
    prediction.update({
        "ok": True,
        "mode": "virtual_shoe_adaptive_cusum_compatibility",
        "virtual_hand": hand_data,
        "virtual_outcome": actual,
        "virtual_outcome_text": hand_data["outcome_text"],
        "verdict": verdict,
        "verdict_text": {
            "HIT": "命中",
            "MISS": "未命中",
            "TIE_SKIPPED": "和局不計",
        }[verdict],
        "cards_consumed": int(hand.cards_used),
        "remaining_cards_after": len(remaining_shoe),
        "remaining_counts_after": counts_from_shoe(remaining_shoe),
        "round_number": int(session.get("hand_number", 0) or 0) + 1,
        "warmup_rounds": int(session.get("warmup_rounds", 0) or 0),
        "bandit_learning_applied": True,
        "disclaimer": "虛擬相容模式方向使用 Full Road Adaptive + CUSUM-LinUCB。",
    })
    return {
        "prediction": prediction,
        "hand": hand_data,
        "remaining_shoe": remaining_shoe,
    }


def parse_point_observation(value: Any) -> None:
    del value
    return None


__all__ = [
    "DB_HOLDOUT",
    "ShadowBacktestController",
    "ShortTermTakeoverController",
    "parse_point_observation",
    "predict",
    "run_virtual_round",
]
