"""Compatibility confidence helpers for the LSTM + short-shoe pipeline.

Formal B/P direction is never created or changed here. The active confidence
calibration is shoe/cut-card only. Legacy structural helpers remain import-safe
for older callers but do not participate in the formal decision path.
"""
from __future__ import annotations

from typing import Any, Mapping

PHYSICAL_PRIOR = {"B": 0.4586, "P": 0.4462, "T": 0.0952}
OUTCOMES = ("B", "P", "T")
SHOE_STAGE_FACTORS = {"OPENING": 1.00, "DEVELOPING": 1.00, "MATURE": 0.96, "LATE": 0.88, "UNKNOWN": 0.95}
LSTM_SHOE_STAGE_FACTORS = dict(SHOE_STAGE_FACTORS)
LSTM_TOTAL_PENALTY_CAP = 0.12


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _normalize_threeway(values: Mapping[str, Any]) -> dict[str, float]:
    raw = {outcome: max(1e-12, float(values.get(outcome, 0.0) or 0.0)) for outcome in OUTCOMES}
    total = sum(raw.values())
    if total <= 1e-12:
        return dict(PHYSICAL_PRIOR)
    return {outcome: raw[outcome] / total for outcome in OUTCOMES}


def _neutral_prior_with_tie(tie_probability: float) -> dict[str, float]:
    tie = _clip(tie_probability, 0.0, 0.999999)
    half = (1.0 - tie) / 2.0
    return {"B": half, "P": half, "T": tie}


def neutralize_physical_banker_bias(probabilities: Mapping[str, Any]) -> dict[str, float]:
    raw = _normalize_threeway(probabilities)
    bp_mass = raw["B"] + raw["P"]
    if bp_mass <= 1e-12:
        return _neutral_prior_with_tie(raw["T"])
    physical_mass = PHYSICAL_PRIOR["B"] + PHYSICAL_PRIOR["P"]
    evidence_b = max(1e-12, (raw["B"] / bp_mass) / (PHYSICAL_PRIOR["B"] / physical_mass))
    evidence_p = max(1e-12, (raw["P"] / bp_mass) / (PHYSICAL_PRIOR["P"] / physical_mass))
    total = evidence_b + evidence_p
    return _normalize_threeway({"B": bp_mass * evidence_b / total, "P": bp_mass * evidence_p / total, "T": raw["T"]})


def build_remaining_card_state(shoe_posterior: Mapping[str, Any] | None, *, decks: int = 8) -> dict[str, Any]:
    posterior = dict(shoe_posterior or {})
    start_cards = max(52, 52 * max(1, min(16, int(decks or 8))))
    rounds = max(0, int(posterior.get("conditioned_rounds", 0) or 0))
    try:
        remaining = float(posterior.get("expected_remaining_cards", start_cards) or start_cards)
    except (TypeError, ValueError):
        remaining = float(start_cards)
    remaining = max(0.0, min(float(start_cards), remaining))
    ratio = _clip(remaining / max(1.0, float(start_cards)))
    progress = _clip(rounds / 60.0)
    if progress < 0.25:
        stage = "OPENING"
    elif progress < 0.55:
        stage = "DEVELOPING"
    elif progress < 0.80:
        stage = "MATURE"
    else:
        stage = "LATE"
    reliability = _clip(float(posterior.get("reliability", min(0.85, rounds / 24.0)) or 0.0))
    return {"available": bool(rounds > 0), "start_cards": int(start_cards), "conditioned_rounds": int(rounds), "mean_remaining_cards": float(remaining), "remaining_ratio": float(ratio), "penetration": float(1.0 - ratio), "shoe_stage": stage, "shoe_stage_factor": float(SHOE_STAGE_FACTORS[stage]), "lstm_shoe_stage_factor": float(LSTM_SHOE_STAGE_FACTORS[stage]), "reliability": float(reliability), "direction_authority": False, "semantics": "short_shoe_state_for_confidence_only_not_direction"}


def calculate_pattern_survival(markov: Mapping[str, Any], road_analysis: Mapping[str, Any] | None, remaining_card_state: Mapping[str, Any] | None) -> dict[str, Any]:
    del markov, road_analysis
    remaining = dict(remaining_card_state or {})
    stage = str(remaining.get("shoe_stage") or "UNKNOWN").upper()
    factor = _clip(float(remaining.get("shoe_stage_factor", SHOE_STAGE_FACTORS.get(stage, 0.95)) or 0.95))
    return {"score": float(factor), "raw_score": 1.0, "base_structural_score": 1.0, "pattern": "LSTM_SHOE_ONLY", "base_pattern": "LSTM_SHOE_ONLY", "shoe_stage": stage, "shoe_stage_factor": float(factor), "change_point": False, "pattern_break": False, "change_point_factor": 1.0, "hidden_regime": {}, "hidden_regime_factor": 1.0, "multi_order_agreement": 0.5, "multi_order_conflict_factor": 1.0, "direction_override": False, "diagnostic_only": True, "semantics": "legacy_pattern_survival_disabled_formal_core_is_lstm_plus_shoe"}


def calibrate_lstm_confidence(*, direction: str, raw_confidence: float, remaining_card_state: Mapping[str, Any] | None = None, hazard_calibration: Mapping[str, Any] | None = None, transition_calibration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    del hazard_calibration, transition_calibration
    side = "B" if str(direction or "").upper().strip() == "B" else "P"
    raw = _clip(raw_confidence, 0.5, 0.999999)
    remaining = dict(remaining_card_state or {})
    stage = str(remaining.get("shoe_stage") or "UNKNOWN").upper()
    reliability = _clip(float(remaining.get("reliability", 0.0) or 0.0))
    anchor = _clip(float(remaining.get("lstm_shoe_stage_factor", remaining.get("shoe_confidence_factor", LSTM_SHOE_STAGE_FACTORS.get(stage, 0.95))) or LSTM_SHOE_STAGE_FACTORS.get(stage, 0.95)))
    stage_factor = _clip(1.0 - reliability * (1.0 - anchor), 0.88, 1.0)
    total_penalty = min(LSTM_TOTAL_PENALTY_CAP, 1.0 - stage_factor)
    combined_factor = 1.0 - total_penalty
    final_confidence = _clip(0.5 + (raw - 0.5) * combined_factor, 0.5, 0.999999)
    probabilities = {"B": final_confidence if side == "B" else 1.0 - final_confidence, "P": final_confidence if side == "P" else 1.0 - final_confidence, "T": 0.0}
    return {"direction": side, "raw_confidence": float(raw), "confidence": float(final_confidence), "probabilities": {key: float(value) for key, value in probabilities.items()}, "combined_factor": float(combined_factor), "total_penalty": float(total_penalty), "penalty_cap": float(LSTM_TOTAL_PENALTY_CAP), "shoe_stage": stage, "shoe_stage_factor": float(stage_factor), "shoe_stage_anchor": float(anchor), "shoe_state_reliability": float(reliability), "hazard_factor": 1.0, "transition_factor": 1.0, "direction_override": False, "bet_scale_factor": float(combined_factor), "formal_calibrators": ["shoe_depth", "cut_card"], "semantics": "shoe_and_cut_card_shrink_margin_only_never_flip_lstm_direction"}


def calibrate_markov_probabilities(markov_probs: Mapping[str, Any], survival_score: float) -> dict[str, float]:
    s = _clip(survival_score)
    directional = neutralize_physical_banker_bias(markov_probs)
    neutral = _neutral_prior_with_tie(directional["T"])
    return _normalize_threeway({outcome: (1.0 - s) * neutral[outcome] + s * directional[outcome] for outcome in OUTCOMES})


__all__ = ["PHYSICAL_PRIOR", "SHOE_STAGE_FACTORS", "LSTM_SHOE_STAGE_FACTORS", "LSTM_TOTAL_PENALTY_CAP", "build_remaining_card_state", "calculate_pattern_survival", "calibrate_lstm_confidence", "calibrate_markov_probabilities", "neutralize_physical_banker_bias"]
