"""Remaining-card and soft confidence calibration for the LSTM-primary BGS pipeline.

No function in this module is allowed to choose or reverse B/P direction.
"""
from __future__ import annotations
from typing import Any, Mapping
import math
from hsmm_regime import analyze_hidden_regime

PHYSICAL_PRIOR = {"B": 0.4586, "P": 0.4462, "T": 0.0952}
OUTCOMES = ("B", "P", "T")
SHOE_STAGE_FACTORS = {"OPENING": 0.98, "DEVELOPING": 1.00, "MATURE": 0.98, "LATE": 0.88, "UNKNOWN": 0.95}
LSTM_SHOE_STAGE_FACTORS = {"OPENING": 1.00, "DEVELOPING": 1.00, "MATURE": 0.97, "LATE": 0.88, "UNKNOWN": 0.95}
LSTM_TOTAL_PENALTY_CAP = 0.18
_AGREEMENT_NO_PENALTY_THRESHOLD = 0.75
_AGREEMENT_MAX_CONFLICT_PENALTY = 0.20


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _normalize_threeway(values: Mapping[str, Any]) -> dict[str, float]:
    raw = {o: max(1e-12, float(values.get(o, 0.0) or 0.0)) for o in OUTCOMES}
    total = sum(raw.values())
    return {o: raw[o] / total for o in OUTCOMES} if total > 1e-12 else dict(PHYSICAL_PRIOR)


def _neutral_prior_with_tie(tie_probability: float) -> dict[str, float]:
    tie = _clip(tie_probability, 0.0, 0.999999); half = (1.0 - tie) / 2.0
    return {"B": half, "P": half, "T": tie}


def neutralize_physical_banker_bias(probabilities: Mapping[str, Any]) -> dict[str, float]:
    raw = _normalize_threeway(probabilities); bp_mass = raw["B"] + raw["P"]
    if bp_mass <= 1e-12: return _neutral_prior_with_tie(raw["T"])
    physical_mass = PHYSICAL_PRIOR["B"] + PHYSICAL_PRIOR["P"]
    evidence_b = max(1e-12, (raw["B"] / bp_mass) / (PHYSICAL_PRIOR["B"] / physical_mass))
    evidence_p = max(1e-12, (raw["P"] / bp_mass) / (PHYSICAL_PRIOR["P"] / physical_mass))
    total = evidence_b + evidence_p
    return _normalize_threeway({"B": bp_mass * evidence_b / total, "P": bp_mass * evidence_p / total, "T": raw["T"]})


def _shoe_stage(remaining_ratio: float) -> str:
    ratio = _clip(remaining_ratio)
    if ratio >= 0.84: return "OPENING"
    if ratio >= 0.67: return "DEVELOPING"
    if ratio >= 0.48: return "MATURE"
    return "LATE"


def build_remaining_card_state(shoe_posterior: Mapping[str, Any] | None, *, decks: int = 8) -> dict[str, Any]:
    posterior = dict(shoe_posterior or {}); decks = max(1, min(16, int(decks or 8))); start_cards = 52 * decks
    rounds = max(0, int(posterior.get("conditioned_rounds", 0) or 0))
    try: mean_remaining = float(posterior.get("expected_remaining_cards", start_cards) or start_cards)
    except (TypeError, ValueError): mean_remaining = float(start_cards)
    mean_remaining = max(0.0, min(float(start_cards), mean_remaining))
    depth = dict(posterior.get("depth_constraint") or {}); depth_applied = bool(depth.get("applied", False)); margin = 12
    if rounds > 0:
        physical_min = max(0.0, float(start_cards - 6 * rounds - margin)); physical_max = min(float(start_cards), float(start_cards - 4 * rounds + 4))
    else: physical_min = physical_max = float(start_cards)
    if depth_applied:
        try: std_cards = max(0.5, float(depth.get("post_constraint_std_remaining", 0.0) or 0.0))
        except (TypeError, ValueError): std_cards = 0.0
        if std_cards <= 0.5: std_cards = max(1.0, (physical_max - physical_min) / math.sqrt(12.0))
        interval_low = max(physical_min, mean_remaining - 1.645 * std_cards); interval_high = min(physical_max, mean_remaining + 1.645 * std_cards)
        interval_source = "depth_conditioned_particle_total_approx90"
    else:
        std_cards = max(1.0, (physical_max - physical_min) / math.sqrt(12.0)); interval_low, interval_high = physical_min, physical_max
        interval_source = "physical_4_to_6_cards_per_round_envelope"
    remaining_ratio = _clip(mean_remaining / max(1.0, float(start_cards))); stage = _shoe_stage(remaining_ratio)
    ess = _clip(float(posterior.get("mean_ess_ratio", 0.0) or 0.0)); history_factor = min(1.0, rounds / 24.0)
    concentration = _clip(1.0 - max(0.0, interval_high - interval_low) / max(1.0, 0.20 * start_cards)); depth_factor = 1.0 if depth_applied else 0.78
    reliability = _clip(history_factor * (0.55 + 0.45 * ess) * (0.60 + 0.40 * concentration) * depth_factor)
    return {"available": bool(rounds > 0), "start_cards": int(start_cards), "conditioned_rounds": int(rounds), "mean_remaining_cards": float(mean_remaining),
            "mean_used_cards": float(start_cards - mean_remaining), "approx_std_cards": float(std_cards), "plausible_interval_low": float(interval_low),
            "plausible_interval_high": float(interval_high), "physical_min_remaining": float(physical_min), "physical_max_remaining": float(physical_max),
            "remaining_ratio": float(remaining_ratio), "penetration": float(1.0 - remaining_ratio), "shoe_stage": stage,
            "shoe_stage_factor": float(SHOE_STAGE_FACTORS[stage]), "lstm_shoe_stage_factor": float(LSTM_SHOE_STAGE_FACTORS[stage]),
            "reliability": float(reliability), "depth_constraint_applied": depth_applied, "interval_source": interval_source,
            "direction_authority": False, "semantics": "remaining_card_state_for_confidence_and_sizing_only_not_direction"}


def _agreement_conflict_factor(agreement: float) -> float:
    value = _clip(agreement, 0.5, 1.0)
    if value >= _AGREEMENT_NO_PENALTY_THRESHOLD: return 1.0
    conflict = (_AGREEMENT_NO_PENALTY_THRESHOLD - value) / (_AGREEMENT_NO_PENALTY_THRESHOLD - 0.5)
    return _clip(1.0 - _AGREEMENT_MAX_CONFLICT_PENALTY * _clip(conflict))


def calculate_pattern_survival(markov: Mapping[str, Any], road_analysis: Mapping[str, Any] | None, remaining_card_state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Legacy structural score, softened and calibration-only."""
    road = dict(road_analysis or {}); remaining = dict(remaining_card_state or {}); profile = dict(markov.get("regime_profile") or {})
    regime = str(profile.get("regime") or markov.get("regime") or "MIXED"); base_regime = str(profile.get("base_regime") or regime)
    support = _clip(float(markov.get("support_strength", 0.0) or 0.0)); agreement = _clip(float(markov.get("multi_order_agreement", 0.5) or 0.5))
    agreement_factor = _agreement_conflict_factor(agreement); stability = _clip(float(profile.get("stability", 0.45) or 0.0))
    entropy_stability = _clip(1.0 - max(0.0, float(profile.get("entropy_delta", 0.0) or 0.0)) / 0.50)
    current_run = max(0, int(profile.get("current_run_length", 0) or 0)); alternation = _clip(float(profile.get("alternation_ratio", 0.0) or 0.0))
    recent_runs = [max(0, int(v)) for v in list(profile.get("recent_run_lengths") or [])]
    if base_regime == "DRAGON": recent_pattern = _clip(current_run / 6.0)
    elif base_regime == "CHOP": recent_pattern = alternation
    elif base_regime == "DOUBLE_CHOP":
        window = recent_runs[-4:]; recent_pattern = sum(1 for length in window if length in {1, 2}) / len(window) if window else 0.35
    elif regime == "TRANSITION": recent_pattern = 0.35
    else: recent_pattern = 0.40
    derived = dict(road.get("derived_road_markov") or {}); road_reliability = max(0.0, float(road.get("derived_markov_reliability", derived.get("reliability", 0.0)) or 0.0))
    road_support = _clip(road_reliability / max(1e-9, float(derived.get("max_reliability", 0.18) or 0.18)))
    derived_component = _clip(0.70 * road_support + 0.30 * _clip(abs(float(road.get("derived_road_consensus", 0.0) or 0.0))))
    remaining_reliability = _clip(float(remaining.get("reliability", 0.0) or 0.0)); stage = str(remaining.get("shoe_stage") or "UNKNOWN").upper()
    stage_factor = _clip(float(remaining.get("shoe_stage_factor", SHOE_STAGE_FACTORS.get(stage, 0.95)) or SHOE_STAGE_FACTORS.get(stage, 0.95)))
    change_point = bool(profile.get("change_point", False)); pattern_break = bool(profile.get("pattern_break", False)); change_factor = 0.80 if (change_point or pattern_break or regime == "TRANSITION") else 1.0
    base_score = _clip(0.25 * support + 0.22 * stability + 0.18 * recent_pattern + 0.12 * entropy_stability + 0.13 * derived_component + 0.10 * remaining_reliability)
    raw_score = _clip(base_score * agreement_factor); hidden_regime = analyze_hidden_regime(markov); hidden_factor = _clip(float(hidden_regime.get("pattern_factor", 1.0) or 1.0))
    score = _clip(raw_score * stage_factor * change_factor * hidden_factor)
    return {"score": float(score), "raw_score": float(raw_score), "base_structural_score": float(base_score), "pattern": regime, "base_pattern": base_regime,
            "shoe_stage": stage, "shoe_stage_factor": float(stage_factor), "change_point": change_point, "pattern_break": pattern_break,
            "change_point_factor": float(change_factor), "hidden_regime": hidden_regime, "hidden_regime_factor": float(hidden_factor),
            "multi_order_agreement": float(agreement), "multi_order_conflict_factor": float(agreement_factor), "direction_override": False,
            "semantics": "soft_pattern_survival_calibration_only_not_next_hand_win_probability"}


def calibrate_lstm_confidence(*, direction: str, raw_confidence: float, remaining_card_state: Mapping[str, Any] | None = None,
                              hazard_calibration: Mapping[str, Any] | None = None, transition_calibration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Shrink directional margin toward 50%; chosen side is never changed."""
    side = "B" if str(direction or "").upper().strip() == "B" else "P"; raw = _clip(raw_confidence, 0.5, 0.999999)
    remaining = dict(remaining_card_state or {}); stage = str(remaining.get("shoe_stage") or "UNKNOWN").upper(); reliability = _clip(float(remaining.get("reliability", 0.0) or 0.0))
    anchor = _clip(float(remaining.get("lstm_shoe_stage_factor", LSTM_SHOE_STAGE_FACTORS.get(stage, 0.95)) or LSTM_SHOE_STAGE_FACTORS.get(stage, 0.95)))
    stage_factor = _clip(1.0 - reliability * (1.0 - anchor), 0.88, 1.0)
    hazard_factor = _clip(float(dict(hazard_calibration or {}).get("confidence_factor", 1.0) or 1.0), 0.92, 1.0)
    transition_factor = _clip(float(dict(transition_calibration or {}).get("confidence_factor", 1.0) or 1.0), 0.92, 1.0)
    total_penalty = min(LSTM_TOTAL_PENALTY_CAP, (1.0 - stage_factor) + (1.0 - hazard_factor) + (1.0 - transition_factor)); combined_factor = 1.0 - total_penalty
    final_confidence = _clip(0.5 + (raw - 0.5) * combined_factor, 0.5, 0.999999)
    probabilities = {"B": final_confidence if side == "B" else 1.0 - final_confidence, "P": final_confidence if side == "P" else 1.0 - final_confidence, "T": 0.0}
    return {"direction": side, "raw_confidence": float(raw), "confidence": float(final_confidence), "probabilities": {k: float(v) for k, v in probabilities.items()},
            "combined_factor": float(combined_factor), "total_penalty": float(total_penalty), "penalty_cap": float(LSTM_TOTAL_PENALTY_CAP), "shoe_stage": stage,
            "shoe_stage_factor": float(stage_factor), "shoe_stage_anchor": float(anchor), "shoe_state_reliability": float(reliability), "hazard_factor": float(hazard_factor),
            "transition_factor": float(transition_factor), "direction_override": False, "bet_scale_factor": float(combined_factor),
            "semantics": "shrink_directional_margin_toward_50_percent_only_never_flip_direction"}


def calibrate_markov_probabilities(markov_probs: Mapping[str, Any], survival_score: float) -> dict[str, float]:
    s = _clip(survival_score); directional = neutralize_physical_banker_bias(markov_probs); neutral = _neutral_prior_with_tie(directional["T"])
    return _normalize_threeway({o: (1.0 - s) * neutral[o] + s * directional[o] for o in OUTCOMES})


__all__ = ["PHYSICAL_PRIOR", "SHOE_STAGE_FACTORS", "LSTM_SHOE_STAGE_FACTORS", "LSTM_TOTAL_PENALTY_CAP", "build_remaining_card_state", "calculate_pattern_survival", "calibrate_lstm_confidence", "calibrate_markov_probabilities", "neutralize_physical_banker_bias"]
