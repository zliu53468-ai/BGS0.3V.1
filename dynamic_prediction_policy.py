"""BGS dynamic prediction policy.

This layer keeps the existing Markov / HSMM / Derived Road / Hazard / Shoe
implementations intact and changes only the decision policy around them:

- Hands 1..30: Global Prior Smoothing
    Final_P = (1-alpha) * P_global + alpha * P_local
    alpha = current_hand / 30
- Hands 1..30: no 55% confidence SKIP gate; the selected B/P side only needs
  resolved confidence >= 50%, while the physical minimum expected value is
  fixed at 0.002 (0.2%).
- Hands 1..25: B/P-conditional logit Temperature Scaling with T=0.7 after
  final fusion/global-prior smoothing. Tie mass is preserved.
- Streak diminishing returns: after a B/P run exceeds three consecutive hands,
  each extra continuation hand reduces only the continuation-side probability.
  The decay can shrink confidence to neutral but never manufactures a forced
  opposite-side signal.
- Break-point cooldown: immediately after a 4+ B/P run is broken, the next
  decision is a mandatory observation/SKIP. Decision confidence is halved and
  the active minimum expected-EV gate is raised to 2.0% for that one decision.

The existing shoe-progress weighting, HSMM/entropy confidence calibration and
recent-user feedback remain policy inputs. They do not become direct B/P votes.
"""
from __future__ import annotations

from hashlib import sha256
from threading import local
from typing import Any, Iterable, Mapping, Sequence
import math

from markov_model import (
    GLOBAL_PRIOR_PROBABILITIES,
    GLOBAL_PRIOR_SMOOTH_MAX_ROUNDS,
    blend_with_global_prior,
)
from money_management import BANKER_NET_PAYOUT, PLAYER_NET_PAYOUT
from pattern_survival import PHYSICAL_PRIOR
from performance_tracker import get_resolved_records

POLICY_VERSION = "DYNAMIC-GLOBAL-PRIOR-ANTI-LAG-V3"

EARLY_SHOE_MAX_ROUNDS = 20
LATE_SHOE_MIN_ROUNDS = 41
MIN_DIRECTION_CONFIDENCE = 0.55
EARLY_MIN_DIRECTION_CONFIDENCE = 0.50
PHYSICAL_MIN_EV = 0.002
EARLY_ACTIVE_MAX_ROUNDS = 30
TEMPERATURE_SCALING_MAX_ROUNDS = 25
EARLY_TEMPERATURE = 0.70

STREAK_DECAY_START = 3
STREAK_DECAY_STEP = 0.05
STREAK_DECAY_FLOOR = 0.50
BREAKPOINT_MIN_STREAK = 4
BREAKPOINT_CONFIDENCE_FACTOR = 0.50
BREAKPOINT_MIN_EV = 0.02

ONLINE_WINDOW = 5
ONLINE_CONSECUTIVE_LOSS_TRIGGER = 2
ONLINE_CONFIDENCE_DECAY = 0.50

EARLY_SHOE_WEIGHT_FACTOR = 0.50
MID_SHOE_WEIGHT_FACTOR = 1.00
LATE_SHOE_WEIGHT_FACTOR = 1.50
EARLY_ROAD_WEIGHT_FACTOR = 1.00
MID_ROAD_WEIGHT_FACTOR = 1.00
LATE_ROAD_WEIGHT_FACTOR = 0.70
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
    probs = _normalize(probabilities)
    return "B" if probs["B"] >= probs["P"] else "P"


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
    w = _clip(factor)
    left = _normalize(base)
    right = _normalize(target)
    return _normalize({
        key: (1.0 - w) * left[key] + w * right[key]
        for key in ("B", "P", "T")
    })


def _history_values(history: str | Iterable[Any]) -> list[str]:
    if isinstance(history, str):
        raw_values: Iterable[Any] = list(history)
    elif isinstance(history, Sequence):
        raw_values = history
    else:
        raw_values = list(history)

    result: list[str] = []
    for item in raw_values:
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
    return result


def _history_round_count(history: str | Iterable[Any]) -> int:
    return len(_history_values(history))


def _bp_runs(history: str | Iterable[Any]) -> list[tuple[str, int]]:
    bp = [value for value in _history_values(history) if value in {"B", "P"}]
    runs: list[tuple[str, int]] = []
    for value in bp:
        if runs and runs[-1][0] == value:
            side, count = runs[-1]
            runs[-1] = (side, count + 1)
        else:
            runs.append((value, 1))
    return runs


def _road_run_state(history: str | Iterable[Any]) -> dict[str, Any]:
    raw = _history_values(history)
    runs = _bp_runs(raw)
    current_side = runs[-1][0] if runs else ""
    current_streak = int(runs[-1][1]) if runs else 0
    previous_side = runs[-2][0] if len(runs) >= 2 else ""
    previous_streak = int(runs[-2][1]) if len(runs) >= 2 else 0
    last_raw = raw[-1] if raw else ""

    # Cooldown is exactly the first decision immediately after the break hand.
    # A later Tie does not keep re-triggering the one-hand cooldown.
    breakpoint_active = bool(
        len(runs) >= 2
        and current_streak == 1
        and previous_streak >= BREAKPOINT_MIN_STREAK
        and last_raw in {"B", "P"}
        and last_raw == current_side
        and previous_side != current_side
    )
    return {
        "current_side": current_side,
        "current_streak": int(current_streak),
        "previous_side": previous_side,
        "previous_streak": int(previous_streak),
        "last_raw_outcome": last_raw,
        "breakpoint_cooldown_active": bool(breakpoint_active),
        "breakpoint_min_streak": int(BREAKPOINT_MIN_STREAK),
    }


def _apply_streak_diminishing_returns(
    probabilities: Mapping[str, Any],
    *,
    history: str | Iterable[Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Reduce over-heated continuation confidence without forcing reversal.

    If the active B/P run is longer than three and the current posterior still
    favors continuing that same side, apply:
        decay = 1 - 0.05 * (streak - 3)
    to the resolved continuation probability. The adjusted probability is floored
    at 50%, so this policy can neutralize an over-heated continuation signal but
    cannot create an artificial opposite-side prediction by itself.
    """
    probs = _normalize(probabilities)
    state = _road_run_state(history)
    side = str(state.get("current_side") or "")
    streak = int(state.get("current_streak", 0) or 0)
    dominant = _bp_direction(probs)
    raw_resolved = _resolved_probability(probs, side) if side in {"B", "P"} else 0.5

    applied = bool(
        side in {"B", "P"}
        and streak > STREAK_DECAY_START
        and dominant == side
    )
    decay_factor = 1.0
    adjusted_resolved = raw_resolved
    adjusted = dict(probs)

    if applied:
        decay_factor = max(
            STREAK_DECAY_FLOOR,
            1.0 - STREAK_DECAY_STEP * (streak - STREAK_DECAY_START),
        )
        adjusted_resolved = max(0.5, raw_resolved * decay_factor)
        bp_mass = probs["B"] + probs["P"]
        if side == "B":
            adjusted = _normalize({
                "B": bp_mass * adjusted_resolved,
                "P": bp_mass * (1.0 - adjusted_resolved),
                "T": probs["T"],
            })
        else:
            adjusted = _normalize({
                "B": bp_mass * (1.0 - adjusted_resolved),
                "P": bp_mass * adjusted_resolved,
                "T": probs["T"],
            })

    return adjusted, {
        "applied": bool(applied),
        "current_side": side,
        "streak_count": int(streak),
        "decay_start": int(STREAK_DECAY_START),
        "decay_step": float(STREAK_DECAY_STEP),
        "decay_factor": float(decay_factor),
        "raw_resolved_continuation_probability": float(raw_resolved),
        "adjusted_resolved_continuation_probability": float(adjusted_resolved),
        "never_forces_opposite_side": True,
        "semantics": "diminishing_returns_on_4plus_streak_continuation_confidence",
    }


def _temperature_scale_bp(
    probabilities: Mapping[str, Any],
    temperature: float,
) -> dict[str, float]:
    """Temperature-scale resolved B/P odds while preserving Tie mass."""
    probs = _normalize(probabilities)
    t = max(1e-6, float(temperature))
    bp_mass = probs["B"] + probs["P"]
    if bp_mass <= 1e-12 or abs(t - 1.0) <= 1e-12:
        return probs

    p_b = _clip(probs["B"] / bp_mass, 1e-9, 1.0 - 1e-9)
    p_p = 1.0 - p_b
    exponent = 1.0 / t
    score_b = p_b ** exponent
    score_p = p_p ** exponent
    score_total = score_b + score_p
    if score_total <= 1e-12:
        return probs

    scaled_b = score_b / score_total
    return _normalize({
        "B": bp_mass * scaled_b,
        "P": bp_mass * (1.0 - scaled_b),
        "T": probs["T"],
    })


def _apply_early_probability_policy(
    probabilities: Mapping[str, Any],
    *,
    rounds: int,
    history: str | Iterable[Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Global-prior smoothing followed by T=0.7 scaling in hands 1..25."""
    local = _normalize(probabilities)
    smoothed, global_diag = blend_with_global_prior(
        local,
        rounds,
        history=history,
    )
    temperature_applied = bool(0 < rounds <= TEMPERATURE_SCALING_MAX_ROUNDS)
    final = (
        _temperature_scale_bp(smoothed, EARLY_TEMPERATURE)
        if temperature_applied
        else dict(smoothed)
    )
    return final, {
        "global_prior": global_diag,
        "temperature_applied": temperature_applied,
        "temperature": float(EARLY_TEMPERATURE if temperature_applied else 1.0),
        "temperature_max_rounds": int(TEMPERATURE_SCALING_MAX_ROUNDS),
        "before_temperature": dict(smoothed),
        "after_temperature": dict(final),
    }


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
    for item in recent:
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
    if original <= 0.3000001:
        return _clip(adjusted, 0.0, DYNAMIC_INTERNAL_SHOE_RELIABILITY_CAP)
    return _clip(adjusted)


def _decision_gate(
    direction_probs: Mapping[str, Any],
    economic_probs: Mapping[str, Any],
    direction: str,
    *,
    rounds: int,
    confidence_factor: float = 1.0,
    min_ev_override: float | None = None,
    breakpoint_cooldown: bool = False,
) -> dict[str, Any]:
    side = str(direction or "").upper().strip()
    early_active = bool(0 < rounds <= EARLY_ACTIVE_MAX_ROUNDS)
    minimum_confidence = (
        EARLY_MIN_DIRECTION_CONFIDENCE if early_active else MIN_DIRECTION_CONFIDENCE
    )
    base_min_ev = PHYSICAL_MIN_EV if early_active else 0.0
    min_ev = max(
        base_min_ev,
        max(0.0, float(min_ev_override or 0.0)),
    )
    confidence_factor = _clip(confidence_factor)

    if side not in {"B", "P"}:
        return {
            "decision": "SKIP",
            "allowed": False,
            "reason": "skip_unresolved_direction",
            "raw_resolved_confidence": 0.5,
            "resolved_confidence": 0.5 * confidence_factor,
            "confidence_factor": float(confidence_factor),
            "minimum_confidence": float(minimum_confidence),
            "physical_min_ev": float(min_ev),
            "expected_net_ev": 0.0,
            "ev_pass": False,
            "early_active_policy": early_active,
            "breakpoint_cooldown": bool(breakpoint_cooldown),
        }

    raw_resolved_confidence = _resolved_probability(direction_probs, side)
    resolved_confidence = raw_resolved_confidence * confidence_factor
    economic_probability = _resolved_probability(economic_probs, side)
    net_payout = BANKER_NET_PAYOUT if side == "B" else PLAYER_NET_PAYOUT
    gross_return_multiplier = 1.0 + net_payout
    expected_net_ev = economic_probability * gross_return_multiplier - 1.0
    ev_product = 1.0 + expected_net_ev

    confidence_pass = resolved_confidence >= minimum_confidence
    ev_pass = expected_net_ev >= min_ev
    allowed = bool(confidence_pass and ev_pass and not breakpoint_cooldown)
    if breakpoint_cooldown:
        reason = "skip_breakpoint_cooldown"
    elif not confidence_pass:
        reason = "skip_low_direction_confidence"
    elif not ev_pass:
        reason = "skip_below_physical_min_ev"
    else:
        reason = "early_active_direction_and_ev_pass" if early_active else "direction_confidence_and_positive_ev_pass"

    return {
        "decision": side if allowed else "SKIP",
        "allowed": allowed,
        "reason": reason,
        "direction": side,
        "rounds": int(rounds),
        "early_active_policy": early_active,
        "raw_resolved_confidence": float(raw_resolved_confidence),
        "resolved_confidence": float(resolved_confidence),
        "confidence_factor": float(confidence_factor),
        "minimum_confidence": float(minimum_confidence),
        "confidence_pass": bool(confidence_pass),
        "economic_resolved_probability": float(economic_probability),
        "net_payout": float(net_payout),
        "gross_return_multiplier": float(gross_return_multiplier),
        "ev_product": float(ev_product),
        "expected_net_ev": float(expected_net_ev),
        "physical_min_ev": float(min_ev),
        "ev_pass": bool(ev_pass),
        "breakpoint_cooldown": bool(breakpoint_cooldown),
        "rule": (
            "breakpoint_cooldown_force_skip_confidence_x0.5_min_ev_0.02"
            if breakpoint_cooldown
            else "hands_1_30_confidence>=0.50_and_expected_net_ev>=0.002"
            if early_active
            else "resolved_confidence>=0.55_and_expected_net_ev>=0"
        ),
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
        history_materialized: str | list[Any]
        if isinstance(history, str):
            history_materialized = history
        elif isinstance(history, Sequence):
            history_materialized = list(history)
        else:
            history_materialized = list(history)

        remaining = dict(remaining_card_state or {})
        rounds = max(0, int(remaining.get("conditioned_rounds", 0) or 0))
        if rounds <= 0:
            rounds = _history_round_count(history_materialized)
        progress = shoe_progress_policy(rounds)
        run_state = _road_run_state(history_materialized)
        breakpoint_active = bool(run_state["breakpoint_cooldown_active"])
        breakpoint_confidence_factor = (
            BREAKPOINT_CONFIDENCE_FACTOR if breakpoint_active else 1.0
        )
        active_min_ev = BREAKPOINT_MIN_EV if breakpoint_active else (
            PHYSICAL_MIN_EV if 0 < rounds <= EARLY_ACTIVE_MAX_ROUNDS else 0.0
        )

        effective_shoe_reliability = _effective_shoe_reliability(
            float(shoe_reliability or 0.0), progress
        )

        raw_result = original_predict(
            self,
            history_materialized,
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
            road_progress_factor * float(entropy_penalty["factor"]) * online_factor
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
            neutral_road, raw_road_family, road_confidence_factor
        )

        local_direction_probs, policy_direction_fusion = self.bayesian_fuse(
            adjusted_road_family,
            shoe_probs,
            shoe_reliability=effective_shoe_reliability,
        )
        direction_probs, early_direction_policy = _apply_early_probability_policy(
            local_direction_probs,
            rounds=rounds,
            history=history_materialized,
        )
        direction_probs, streak_direction_policy = _apply_streak_diminishing_returns(
            direction_probs,
            history=history_materialized,
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
            PHYSICAL_PRIOR, raw_economic_road, road_confidence_factor
        )
        local_economic_probs, policy_economic_fusion = self.bayesian_fuse(
            adjusted_economic_road,
            shoe_probs,
            shoe_reliability=effective_shoe_reliability,
        )
        economic_probs, early_economic_policy = _apply_early_probability_policy(
            local_economic_probs,
            rounds=rounds,
            history=history_materialized,
        )
        economic_probs, streak_economic_policy = _apply_streak_diminishing_returns(
            economic_probs,
            history=history_materialized,
        )

        base_weight = _clip(
            float(result.get("pattern_calibrated_final_weight", 0.0) or 0.0)
        )
        effective_weight_before_breakpoint = _clip(
            base_weight
            * float(entropy_penalty["factor"])
            * online_factor
            * road_progress_factor
        )
        effective_weight = _clip(
            effective_weight_before_breakpoint * breakpoint_confidence_factor
        )

        money = self.money.allocate(
            direction=direction,
            probabilities=economic_probs,
            final_weight=effective_weight,
            bankroll=float(bankroll or 0.0),
            minimum_expected_ev=active_min_ev,
        )
        gate = _decision_gate(
            direction_probs,
            economic_probs,
            direction,
            rounds=rounds,
            confidence_factor=breakpoint_confidence_factor,
            min_ev_override=active_min_ev,
            breakpoint_cooldown=breakpoint_active,
        )
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

        breakpoint_policy = {
            **run_state,
            "confidence_factor": float(breakpoint_confidence_factor),
            "active_min_ev": float(active_min_ev),
            "forced_skip": bool(breakpoint_active),
            "effective_weight_before_breakpoint": float(effective_weight_before_breakpoint),
            "effective_weight_after_breakpoint": float(effective_weight),
            "semantics": (
                "one_decision_cooldown_after_4plus_streak_break_"
                "confidence_halved_and_min_ev_raised_to_2pct"
            ),
        }

        diagnostics = {
            "version": POLICY_VERSION,
            "shoe_progress": progress,
            "rounds": int(rounds),
            "raw_shoe_reliability": float(_clip(shoe_reliability)),
            "effective_shoe_reliability": float(effective_shoe_reliability),
            "entropy_penalty": entropy_penalty,
            "online_performance": feedback,
            "online_performance_factor": float(online_factor),
            "road_confidence_factor": float(road_confidence_factor),
            "raw_road_family_probs": _normalize(raw_road_family),
            "adjusted_road_family_probs": dict(adjusted_road_family),
            "local_direction_probs_before_global_prior": dict(local_direction_probs),
            "early_direction_policy": early_direction_policy,
            "streak_direction_policy": streak_direction_policy,
            "local_economic_probs_before_global_prior": dict(local_economic_probs),
            "early_economic_policy": early_economic_policy,
            "streak_economic_policy": streak_economic_policy,
            "breakpoint_policy": breakpoint_policy,
            "global_prior_probabilities": dict(GLOBAL_PRIOR_PROBABILITIES),
            "global_prior_max_rounds": int(GLOBAL_PRIOR_SMOOTH_MAX_ROUNDS),
            "physical_min_ev_early": float(PHYSICAL_MIN_EV),
            "breakpoint_min_ev": float(BREAKPOINT_MIN_EV),
            "decision_gate": gate,
            "models_modified": False,
            "policy_semantics": (
                "global_prior_plus_temperature_then_streak_diminishing_returns_"
                "plus_one_hand_breakpoint_cooldown_models_unchanged"
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
                "source": "dynamic_policy_global_prior_streak_guard_final_posterior",
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
            "streak_policy": streak_direction_policy,
            "breakpoint_policy": breakpoint_policy,
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
            minimum_confidence = float(
                gate.get("minimum_confidence", MIN_DIRECTION_CONFIDENCE)
                or MIN_DIRECTION_CONFIDENCE
            )
            min_ev = float(gate.get("physical_min_ev", 0.0) or 0.0)
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
                    "SKIP_BREAKPOINT_COOLDOWN"
                    if reason == "skip_breakpoint_cooldown"
                    else "SKIP_LOW_CONFIDENCE"
                    if reason == "skip_low_direction_confidence"
                    else "SKIP_BELOW_PHYSICAL_MIN_EV"
                    if reason == "skip_below_physical_min_ev"
                    else "SKIP_POLICY_GATE"
                ),
                "signal_status_text": (
                    "觀望：長龍剛斷，首局冷卻；Confidence×0.5，EV 門檻 2.0%"
                    if reason == "skip_breakpoint_cooldown"
                    else f"觀望：最終 B/P 信心未達 {minimum_confidence * 100:.1f}%"
                    if reason == "skip_low_direction_confidence"
                    else f"觀望：預期淨 EV 未達 {min_ev * 100:.2f}%"
                    if reason == "skip_below_physical_min_ev"
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
    "EARLY_MIN_DIRECTION_CONFIDENCE",
    "PHYSICAL_MIN_EV",
    "EARLY_ACTIVE_MAX_ROUNDS",
    "TEMPERATURE_SCALING_MAX_ROUNDS",
    "EARLY_TEMPERATURE",
    "STREAK_DECAY_START",
    "STREAK_DECAY_STEP",
    "BREAKPOINT_MIN_STREAK",
    "BREAKPOINT_CONFIDENCE_FACTOR",
    "BREAKPOINT_MIN_EV",
    "shoe_progress_policy",
    "recent_user_direction_feedback",
    "install_dynamic_prediction_policy",
]
