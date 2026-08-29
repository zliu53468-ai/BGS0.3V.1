"""BGS 動態預測決策政策層。

本模組不改寫任何既有預測模型，只負責在模型完成各自分析後，依指定規則
動態調整證據權重與投注決策：

1. Shoe Progress Weight
   - <= 20 局：降低 Shoe 物理證據權重，讓牌路證據相對優先。
   - 21~40 局：維持原始融合權重。
   - > 40 局：提高 Shoe 物理證據權重，降低牌路家族的最終信心。
2. Skip Gate
   - 最終 B/P resolved confidence < 55% 時輸出 SKIP。
3. +EV Gate
   - 只有經濟 posterior 的勝率 × 含本金總回報倍率 > 1 才允許下注。
4. HSMM / Entropy Penalty
   - Regime transition / posterior uncertainty / entropy 偏高時，只降低牌路家族信心。
5. Online Performance Feedback
   - 追蹤同一使用者最近 5 個已結算 B/P 方向；若最近連錯 >= 2 局，
     將牌路家族信心乘以 0.5。

重要：
- 不修改 Markov、HSMM、Derived Road、Hazard、Shoe 的模型輸出。
- 不建立追莊、反莊、追閒、反閒的硬規則。
- direction 仍保留 B/P；是否實際出手由 decision=B/P/SKIP 控制。
"""
from __future__ import annotations

from hashlib import sha256
from threading import local
from typing import Any, Iterable, Mapping, Sequence
import math

from money_management import BANKER_NET_PAYOUT, PLAYER_NET_PAYOUT
from pattern_survival import PHYSICAL_PRIOR
from performance_tracker import get_resolved_records

POLICY_VERSION = "DYNAMIC-SHOE-SKIP-EV-ONLINE-V1"

EARLY_SHOE_MAX_ROUNDS = 20
LATE_SHOE_MIN_ROUNDS = 41
MIN_DIRECTION_CONFIDENCE = 0.55
ONLINE_WINDOW = 5
ONLINE_CONSECUTIVE_LOSS_TRIGGER = 2
ONLINE_CONFIDENCE_DECAY = 0.50

# 使用者指定的局數動態方向：前期降低物理證據，中後期提高物理證據。
EARLY_SHOE_WEIGHT_FACTOR = 0.50
MID_SHOE_WEIGHT_FACTOR = 1.00
LATE_SHOE_WEIGHT_FACTOR = 1.50
EARLY_ROAD_WEIGHT_FACTOR = 1.00
MID_ROAD_WEIGHT_FACTOR = 1.00
LATE_ROAD_WEIGHT_FACTOR = 0.70

# 內建 particle Shoe 原始上限為 0.30；晚期允許動態提高至 0.45。
DYNAMIC_INTERNAL_SHOE_RELIABILITY_CAP = 0.45

_TLS = local()
_INSTALLED = False


def _clip(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(number):
        return lo
    return max(lo, min(hi, number))


def _normalize(values: Mapping[str, Any]) -> dict[str, float]:
    raw = {
        key: max(1e-12, float(values.get(key, 0.0) or 0.0))
        for key in ("B", "P", "T")
    }
    total = sum(raw.values())
    if total <= 1e-12:
        return dict(PHYSICAL_PRIOR)
    return {key: raw[key] / total for key in raw}


def _bp_direction(probabilities: Mapping[str, Any]) -> str:
    b = float(probabilities.get("B", 0.0) or 0.0)
    p = float(probabilities.get("P", 0.0) or 0.0)
    return "B" if b >= p else "P"


def _resolved_probability(probabilities: Mapping[str, Any], side: str) -> float:
    probs = _normalize(probabilities)
    resolved = probs["B"] + probs["P"]
    if resolved <= 1e-12:
        return 0.5
    return float(probs[side] / resolved)


def _neutral_with_same_tie(probabilities: Mapping[str, Any]) -> dict[str, float]:
    probs = _normalize(probabilities)
    bp_mass = probs["B"] + probs["P"]
    half = bp_mass / 2.0
    return {"B": half, "P": half, "T": probs["T"]}


def _blend_probabilities(
    base: Mapping[str, Any],
    target: Mapping[str, Any],
    factor: float,
) -> dict[str, float]:
    """factor=0 回到 base；factor=1 完整保留 target。"""
    w = _clip(factor)
    left = _normalize(base)
    right = _normalize(target)
    return _normalize({
        key: (1.0 - w) * left[key] + w * right[key]
        for key in ("B", "P", "T")
    })


def _history_round_count(history: str | Iterable[Any]) -> int:
    if isinstance(history, str):
        return sum(char.upper() in {"B", "P", "T"} for char in history)
    if isinstance(history, Sequence):
        return sum(
            str(
                item.get("outcome") if isinstance(item, Mapping) else item
            ).upper().strip() in {"B", "P", "T"}
            for item in history
        )
    return 0


def shoe_progress_policy(rounds: int) -> dict[str, Any]:
    value = max(0, int(rounds or 0))
    if value <= 0:
        return {
            "rounds": 0,
            "phase": "UNKNOWN",
            "shoe_weight_factor": 1.0,
            "road_weight_factor": 1.0,
        }
    if value <= EARLY_SHOE_MAX_ROUNDS:
        return {
            "rounds": value,
            "phase": "EARLY",
            "shoe_weight_factor": EARLY_SHOE_WEIGHT_FACTOR,
            "road_weight_factor": EARLY_ROAD_WEIGHT_FACTOR,
        }
    if value < LATE_SHOE_MIN_ROUNDS:
        return {
            "rounds": value,
            "phase": "MID",
            "shoe_weight_factor": MID_SHOE_WEIGHT_FACTOR,
            "road_weight_factor": MID_ROAD_WEIGHT_FACTOR,
        }
    return {
        "rounds": value,
        "phase": "LATE",
        "shoe_weight_factor": LATE_SHOE_WEIGHT_FACTOR,
        "road_weight_factor": LATE_ROAD_WEIGHT_FACTOR,
    }


def recent_user_direction_feedback(
    user_id: str,
    *,
    limit: int = ONLINE_WINDOW,
) -> dict[str, Any]:
    """讀取同一使用者最近已結算方向，只做信心衰減，不反向學習。"""
    raw_user = str(user_id or "")
    if not raw_user:
        return {
            "available": False,
            "sample_count": 0,
            "correct_count": 0,
            "accuracy": 0.0,
            "consecutive_losses": 0,
            "confidence_factor": 1.0,
            "triggered": False,
        }

    uid_key = sha256(raw_user.encode("utf-8")).hexdigest()[:24]
    try:
        records = get_resolved_records(limit=5000)
    except Exception:
        records = []

    recent: list[dict[str, Any]] = []
    for record in reversed(records):
        if str(record.get("uid_key") or "") != uid_key:
            continue
        actual = str(record.get("actual_outcome") or "").upper().strip()
        if actual not in {"B", "P"}:
            continue
        predicted = str(
            record.get("adaptive_only_direction")
            or record.get("action")
            or record.get("recommend")
            or ""
        ).upper().strip()
        if predicted not in {"B", "P"}:
            continue
        recent.append({
            "predicted": predicted,
            "actual": actual,
            "correct": predicted == actual,
        })
        if len(recent) >= max(1, int(limit)):
            break

    correct = sum(int(item["correct"]) for item in recent)
    consecutive_losses = 0
    for item in recent:  # recent 是由最新往舊排列
        if item["correct"]:
            break
        consecutive_losses += 1

    triggered = consecutive_losses >= ONLINE_CONSECUTIVE_LOSS_TRIGGER
    confidence_factor = ONLINE_CONFIDENCE_DECAY if triggered else 1.0
    return {
        "available": bool(recent),
        "sample_count": len(recent),
        "correct_count": int(correct),
        "accuracy": float(correct / max(1, len(recent))),
        "consecutive_losses": int(consecutive_losses),
        "confidence_factor": float(confidence_factor),
        "triggered": bool(triggered),
        "window": int(ONLINE_WINDOW),
        "loss_trigger": int(ONLINE_CONSECUTIVE_LOSS_TRIGGER),
        "semantics": "recent_user_direction_accuracy_confidence_decay_only_no_direction_vote",
    }


def _entropy_regime_penalty(result: Mapping[str, Any]) -> dict[str, Any]:
    """HSMM 轉換/不確定性高且 entropy 高時，單向降低牌路可信度。"""
    markov = dict(result.get("markov") or {})
    pattern = dict(result.get("pattern_survival") or {})
    hidden = dict(pattern.get("hidden_regime") or {})

    try:
        entropy_bits = float(markov.get("entropy_bits", 0.0) or 0.0)
    except (TypeError, ValueError):
        entropy_bits = 0.0
    entropy_norm = _clip(entropy_bits / max(1e-12, math.log2(3.0)))
    transition_probability = _clip(
        float(hidden.get("transition_probability", 0.0) or 0.0)
    )
    concentration = _clip(
        float(hidden.get("posterior_concentration", 0.0) or 0.0)
    )
    uncertainty = max(transition_probability, 1.0 - concentration)

    penalty = _clip(1.0 - 0.50 * entropy_norm * uncertainty, 0.50, 1.0)
    return {
        "factor": float(penalty),
        "entropy_norm": float(entropy_norm),
        "transition_probability": float(transition_probability),
        "posterior_concentration": float(concentration),
        "uncertainty": float(uncertainty),
        "semantics": "entropy_x_hsmm_transition_uncertainty_one_way_confidence_penalty",
    }


def _effective_shoe_reliability(raw: float, progress: Mapping[str, Any]) -> float:
    original = _clip(raw)
    factor = float(progress.get("shoe_weight_factor", 1.0) or 1.0)
    adjusted = original * factor
    # 一般 particle shoe 原始上限 <= 0.30；晚期最多提高至 0.45。
    if original <= 0.3000001:
        return _clip(adjusted, 0.0, DYNAMIC_INTERNAL_SHOE_RELIABILITY_CAP)
    return _clip(adjusted)


def _decision_gate(
    direction_probs: Mapping[str, Any],
    economic_probs: Mapping[str, Any],
    direction: str,
) -> dict[str, Any]:
    side = str(direction or "").upper().strip()
    if side not in {"B", "P"}:
        return {
            "decision": "SKIP",
            "allowed": False,
            "reason": "skip_unresolved_direction",
            "resolved_confidence": 0.5,
            "minimum_confidence": MIN_DIRECTION_CONFIDENCE,
            "ev_product": 0.0,
            "ev_pass": False,
        }

    resolved_confidence = _resolved_probability(direction_probs, side)
    economic_probability = _resolved_probability(economic_probs, side)
    net_payout = BANKER_NET_PAYOUT if side == "B" else PLAYER_NET_PAYOUT
    gross_return_multiplier = 1.0 + net_payout
    ev_product = economic_probability * gross_return_multiplier

    confidence_pass = resolved_confidence >= MIN_DIRECTION_CONFIDENCE
    ev_pass = ev_product > 1.0
    allowed = bool(confidence_pass and ev_pass)
    if not confidence_pass:
        reason = "skip_low_direction_confidence"
    elif not ev_pass:
        reason = "skip_nonpositive_expected_value"
    else:
        reason = "direction_confidence_and_positive_ev_pass"

    return {
        "decision": side if allowed else "SKIP",
        "allowed": allowed,
        "reason": reason,
        "direction": side,
        "resolved_confidence": float(resolved_confidence),
        "minimum_confidence": float(MIN_DIRECTION_CONFIDENCE),
        "confidence_pass": bool(confidence_pass),
        "economic_resolved_probability": float(economic_probability),
        "net_payout": float(net_payout),
        "gross_return_multiplier": float(gross_return_multiplier),
        "ev_product": float(ev_product),
        "ev_pass": bool(ev_pass),
        "rule": "resolved_confidence>=0.55_and_probability_times_gross_return>1",
    }


def _zero_money_for_skip(money: Mapping[str, Any], reason: str) -> dict[str, Any]:
    result = dict(money or {})
    result.update({
        "bet_allowed": False,
        "mandatory_bet": False,
        "bet_percentage": 0.0,
        "bet_amount": 0.0,
        "final_bet_ratio": 0.0,
        "adjusted_ratio": 0.0,
        "pre_tie_adjusted_ratio": 0.0,
        "reason": str(reason or "skip_policy_gate"),
    })
    return result


def _install_engine_wrapper() -> None:
    from baccarat_quant_engine import BaccaratQuantEngine

    current = BaccaratQuantEngine.predict
    if getattr(current, "_dynamic_policy_wrapped", False):
        return
    original_predict = current

    def wrapped_predict(
        self: Any,
        history: str | Iterable[Any],
        *,
        shoe_probs: Mapping[str, Any] | Sequence[float] | None = None,
        shoe_reliability: float = 1.0,
        road_probs: Mapping[str, Any] | Sequence[float] | None = None,
        road_reliability: float = 0.0,
        remaining_card_state: Mapping[str, Any] | None = None,
        bankroll: float = 0.0,
    ) -> dict[str, Any]:
        remaining = dict(remaining_card_state or {})
        rounds = max(
            0,
            int(remaining.get("conditioned_rounds", 0) or 0),
        )
        if rounds <= 0:
            rounds = _history_round_count(history)
        progress = shoe_progress_policy(rounds)
        effective_shoe_reliability = _effective_shoe_reliability(
            float(shoe_reliability or 0.0),
            progress,
        )

        raw_result = original_predict(
            self,
            history,
            shoe_probs=shoe_probs,
            shoe_reliability=effective_shoe_reliability,
            road_probs=road_probs,
            road_reliability=road_reliability,
            remaining_card_state=remaining,
            bankroll=bankroll,
        )
        result = dict(raw_result)

        feedback = dict(getattr(_TLS, "feedback", {}) or {})
        if not feedback:
            feedback = {
                "available": False,
                "confidence_factor": 1.0,
                "triggered": False,
                "sample_count": 0,
                "consecutive_losses": 0,
            }
        online_factor = _clip(
            float(feedback.get("confidence_factor", 1.0) or 1.0),
            ONLINE_CONFIDENCE_DECAY,
            1.0,
        )
        entropy_penalty = _entropy_regime_penalty(result)
        road_progress_factor = _clip(
            float(progress.get("road_weight_factor", 1.0) or 1.0)
        )
        road_confidence_factor = _clip(
            road_progress_factor
            * float(entropy_penalty["factor"])
            * online_factor
        )

        raw_road_family = dict(
            result.get("road_family_probs")
            or dict(result.get("fusion") or {}).get("road_family_posterior")
            or result.get("pattern_calibrated_markov_probs")
            or result.get("markov_probs")
            or PHYSICAL_PRIOR
        )
        neutral_road = _neutral_with_same_tie(raw_road_family)
        adjusted_road_family = _blend_probabilities(
            neutral_road,
            raw_road_family,
            road_confidence_factor,
        )

        direction_probs, policy_direction_fusion = self.bayesian_fuse(
            adjusted_road_family,
            shoe_probs,
            shoe_reliability=effective_shoe_reliability,
        )
        direction = _bp_direction(direction_probs)

        fusion = dict(result.get("fusion") or {})
        economic_road_detail = dict(fusion.get("economic_road_family") or {})
        raw_economic_road = dict(
            economic_road_detail.get("posterior")
            or result.get("economic_probs")
            or PHYSICAL_PRIOR
        )
        adjusted_economic_road = _blend_probabilities(
            PHYSICAL_PRIOR,
            raw_economic_road,
            road_confidence_factor,
        )
        economic_probs, policy_economic_fusion = self.bayesian_fuse(
            adjusted_economic_road,
            shoe_probs,
            shoe_reliability=effective_shoe_reliability,
        )

        base_weight = _clip(
            float(result.get("pattern_calibrated_final_weight", 0.0) or 0.0)
        )
        effective_weight = _clip(
            base_weight
            * float(entropy_penalty["factor"])
            * online_factor
            * road_progress_factor
        )
        money = self.money.allocate(
            direction=direction,
            probabilities=economic_probs,
            final_weight=effective_weight,
            bankroll=float(bankroll or 0.0),
        )
        gate = _decision_gate(direction_probs, economic_probs, direction)
        if not bool(gate["allowed"]):
            money = _zero_money_for_skip(money, str(gate["reason"]))

        pattern_survival = dict(result.get("pattern_survival") or {})
        original_pattern_score = _clip(
            float(pattern_survival.get("score", 0.0) or 0.0)
        )
        effective_pattern_score = _clip(
            original_pattern_score
            * float(entropy_penalty["factor"])
            * online_factor
            * road_progress_factor
        )
        pattern_survival.update({
            "score_before_dynamic_policy": float(original_pattern_score),
            "score": float(effective_pattern_score),
            "dynamic_entropy_penalty": dict(entropy_penalty),
            "online_performance_factor": float(online_factor),
            "online_performance_feedback": feedback,
            "shoe_progress_road_factor": float(road_progress_factor),
            "dynamic_policy_version": POLICY_VERSION,
        })

        diagnostics = {
            "version": POLICY_VERSION,
            "shoe_progress": progress,
            "raw_shoe_reliability": float(_clip(shoe_reliability)),
            "effective_shoe_reliability": float(effective_shoe_reliability),
            "entropy_penalty": entropy_penalty,
            "online_performance": feedback,
            "online_performance_factor": float(online_factor),
            "road_confidence_factor": float(road_confidence_factor),
            "raw_road_family_probs": _normalize(raw_road_family),
            "adjusted_road_family_probs": dict(adjusted_road_family),
            "decision_gate": gate,
            "models_modified": False,
            "policy_semantics": (
                "dynamic_weight_and_skip_policy_only_models_remain_unchanged"
            ),
        }
        money["dynamic_policy"] = diagnostics

        fusion.update({
            "dynamic_policy": diagnostics,
            "policy_direction_fusion": policy_direction_fusion,
            "policy_economic_fusion": policy_economic_fusion,
            "shoe_reliability_before_dynamic_policy": float(_clip(shoe_reliability)),
            "shoe_reliability": float(effective_shoe_reliability),
            "road_family_confidence_factor": float(road_confidence_factor),
        })

        result.update({
            "direction": direction,
            "direction_text": "莊" if direction == "B" else "閒",
            "direction_margin": float(direction_probs["B"] - direction_probs["P"]),
            "direction_selection": {
                "source": "dynamic_policy_final_posterior",
                "margin": float(direction_probs["B"] - direction_probs["P"]),
            },
            "road_family_probs_before_dynamic_policy": _normalize(raw_road_family),
            "road_family_probs": dict(adjusted_road_family),
            "final_probs": dict(direction_probs),
            "direction_probs": dict(direction_probs),
            "economic_probs": dict(economic_probs),
            "final_probability": float(direction_probs[direction]),
            "economic_probability_for_direction": float(economic_probs[direction]),
            "bet_allowed": bool(money.get("bet_allowed", False)),
            "bet_percentage": float(money.get("bet_percentage", 0.0) or 0.0),
            "bet_amount": float(money.get("bet_amount", 0.0) or 0.0),
            "suggested_bet_amount": float(money.get("bet_amount", 0.0) or 0.0),
            "edge": float(money.get("edge", 0.0) or 0.0),
            "edge_percent": float(money.get("edge_percent", 0.0) or 0.0),
            "money_management": money,
            "pattern_survival": pattern_survival,
            "pattern_calibrated_final_weight": float(effective_weight),
            "fusion": fusion,
            "decision": str(gate["decision"]),
            "decision_text": (
                "觀望" if gate["decision"] == "SKIP"
                else "莊" if gate["decision"] == "B" else "閒"
            ),
            "skip": bool(gate["decision"] == "SKIP"),
            "skip_reason": (
                str(gate["reason"]) if gate["decision"] == "SKIP" else ""
            ),
            "decision_gate": gate,
            "dynamic_prediction_policy": diagnostics,
        })
        return result

    wrapped_predict._dynamic_policy_wrapped = True  # type: ignore[attr-defined]
    BaccaratQuantEngine.predict = wrapped_predict


def _install_predictor_wrapper() -> None:
    import predictor as predictor_module

    current = predictor_module.predict
    if getattr(current, "_dynamic_policy_wrapped", False):
        return
    original_predict = current

    def wrapped_predict(
        history: Any = None,
        venue: str = "",
        room: str = "",
        shoe_id: str = "",
        user_id: str = "",
        run_seed: int | None = None,
        shoe_context: Mapping[str, Any] | None = None,
        road_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        feedback = recent_user_direction_feedback(user_id, limit=ONLINE_WINDOW)
        _TLS.feedback = feedback
        try:
            result = dict(original_predict(
                history=history,
                venue=venue,
                room=room,
                shoe_id=shoe_id,
                user_id=user_id,
                run_seed=run_seed,
                shoe_context=shoe_context,
                road_context=road_context,
            ))
        finally:
            _TLS.feedback = {}

        money = dict(result.get("money_management") or {})
        policy = dict(money.get("dynamic_policy") or {})
        gate = dict(policy.get("decision_gate") or {})
        decision = str(gate.get("decision") or result.get("direction") or "SKIP").upper()
        latent_direction = str(result.get("direction") or "").upper().strip()

        result["adaptive_only_direction"] = latent_direction
        result["online_performance_feedback"] = feedback
        result["dynamic_prediction_policy"] = policy
        result["dynamic_policy_version"] = POLICY_VERSION

        if decision == "SKIP":
            reason = str(gate.get("reason") or money.get("reason") or "skip_policy_gate")
            result.update({
                "recommend": "SKIP",
                "recommend_text": "觀望",
                "action": "SKIP",
                "action_text": "觀望",
                "signal_allowed": False,
                "risk_gate_open": False,
                "mandatory_bet": False,
                "force_observe": True,
                "bet_allowed": False,
                "bet_percentage": 0.0,
                "suggested_bet_amount": 0.0,
                "bet_amount": 0.0,
                "final_bet_ratio": 0.0,
                "signal_status_code": (
                    "SKIP_LOW_CONFIDENCE"
                    if reason == "skip_low_direction_confidence"
                    else "SKIP_NONPOSITIVE_EV"
                    if reason == "skip_nonpositive_expected_value"
                    else "SKIP_POLICY_GATE"
                ),
                "signal_status_text": (
                    "觀望：最終 B/P 信心未達 55%"
                    if reason == "skip_low_direction_confidence"
                    else "觀望：目前未通過正期望值 (+EV) 條件"
                    if reason == "skip_nonpositive_expected_value"
                    else "觀望：目前決策條件不足"
                ),
            })
        else:
            decision_text = "莊" if decision == "B" else "閒"
            result.update({
                "recommend": decision,
                "recommend_text": decision_text,
                "action": decision,
                "action_text": decision_text,
                "signal_allowed": bool(result.get("bet_allowed", False)),
                "risk_gate_open": bool(result.get("bet_allowed", False)),
                "mandatory_bet": False,
                "force_observe": False,
            })
        return result

    wrapped_predict._dynamic_policy_wrapped = True  # type: ignore[attr-defined]
    predictor_module.predict = wrapped_predict


def install_dynamic_prediction_policy() -> bool:
    """在 app 載入前安裝一次；不更動既有函式參數名稱。"""
    global _INSTALLED
    if _INSTALLED:
        return True
    _install_engine_wrapper()
    _install_predictor_wrapper()
    _INSTALLED = True
    return True


__all__ = [
    "POLICY_VERSION",
    "MIN_DIRECTION_CONFIDENCE",
    "shoe_progress_policy",
    "recent_user_direction_feedback",
    "install_dynamic_prediction_policy",
]
