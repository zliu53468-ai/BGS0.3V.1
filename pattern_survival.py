"""Remaining-card state and pattern-survival calibration for BGS.

This module does not predict hidden card identities. It converts the existing
probabilistic-shoe posterior into a coarse shoe-stage estimate and uses that
maturity information only to calibrate how strongly the historical road/Markov
pattern is allowed to influence the next-hand posterior.

Pattern Survival is NOT a baccarat win probability.

Direction calibration deliberately separates two concepts:
- PHYSICAL_PRIOR: baccarat's natural B/P/T base rates, retained for shoe/EV math.
- DIRECTION_NEUTRAL_PRIOR: equal B/P mass with the same Tie mass, used when a
  historical direction signal is weak so Banker does not receive a free vote.
"""
from __future__ import annotations

from typing import Any, Mapping
import math

PHYSICAL_PRIOR = {"B": 0.4586, "P": 0.4462, "T": 0.0952}
OUTCOMES = ("B", "P", "T")

SHOE_STAGE_FACTORS = {
    "OPENING": 0.45,
    "DEVELOPING": 0.75,
    "MATURE": 1.00,
    "LATE": 0.80,
    "UNKNOWN": 0.70,
}


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _normalize_threeway(values: Mapping[str, Any]) -> dict[str, float]:
    raw = {
        outcome: max(1e-12, float(values.get(outcome, 0.0) or 0.0))
        for outcome in OUTCOMES
    }
    total = sum(raw.values())
    if total <= 1e-12:
        return dict(PHYSICAL_PRIOR)
    return {outcome: raw[outcome] / total for outcome in OUTCOMES}


def _neutral_prior_with_tie(tie_probability: float) -> dict[str, float]:
    """Return an equal B/P direction baseline while preserving Tie mass."""
    tie = _clip(float(tie_probability), 0.0, 0.999999)
    resolved_mass = max(1e-12, 1.0 - tie)
    half = resolved_mass / 2.0
    return {"B": half, "P": half, "T": tie}


def neutralize_physical_banker_bias(
    probabilities: Mapping[str, Any],
) -> dict[str, float]:
    """Convert a physical-prior-smoothed posterior into pure B/P direction evidence.

    The Markov model is intentionally smoothed with baccarat's physical prior.
    That is useful statistically, but if its output is used directly for a B/P
    recommendation the natural Banker base-rate advantage becomes a permanent
    hidden vote for B.

    We remove only that baseline in conditional B/P odds:

        E_B = P_model(B | resolved) / P_phys(B | resolved)
        E_P = P_model(P | resolved) / P_phys(P | resolved)

        P_dir(B | resolved) = E_B / (E_B + E_P)

    Tie mass is preserved. Therefore an input equal to PHYSICAL_PRIOR becomes an
    exactly neutral B/P direction signal, while genuine deviations from the
    physical baseline remain directional evidence.
    """
    raw = _normalize_threeway(probabilities)
    bp_mass = raw["B"] + raw["P"]
    if bp_mass <= 1e-12:
        return _neutral_prior_with_tie(raw["T"])

    physical_bp_mass = PHYSICAL_PRIOR["B"] + PHYSICAL_PRIOR["P"]
    physical_b = PHYSICAL_PRIOR["B"] / physical_bp_mass
    physical_p = PHYSICAL_PRIOR["P"] / physical_bp_mass
    model_b = raw["B"] / bp_mass
    model_p = raw["P"] / bp_mass

    evidence_b = max(1e-12, model_b / max(1e-12, physical_b))
    evidence_p = max(1e-12, model_p / max(1e-12, physical_p))
    evidence_total = evidence_b + evidence_p
    direction_b = evidence_b / evidence_total
    direction_p = evidence_p / evidence_total

    return _normalize_threeway({
        "B": bp_mass * direction_b,
        "P": bp_mass * direction_p,
        "T": raw["T"],
    })


def _shoe_stage(remaining_ratio: float) -> str:
    ratio = _clip(remaining_ratio)
    if ratio >= 0.84:
        return "OPENING"
    if ratio >= 0.67:
        return "DEVELOPING"
    if ratio >= 0.48:
        return "MATURE"
    return "LATE"


def build_remaining_card_state(
    shoe_posterior: Mapping[str, Any] | None,
    *,
    decks: int = 8,
) -> dict[str, Any]:
    posterior = dict(shoe_posterior or {})
    decks = max(1, min(16, int(decks or 8)))
    start_cards = 52 * decks
    rounds = max(0, int(posterior.get("conditioned_rounds", 0) or 0))

    try:
        mean_remaining = float(
            posterior.get("expected_remaining_cards", start_cards) or start_cards
        )
    except (TypeError, ValueError):
        mean_remaining = float(start_cards)
    mean_remaining = max(0.0, min(float(start_cards), mean_remaining))

    depth = dict(posterior.get("depth_constraint") or {})
    depth_applied = bool(depth.get("applied", False))
    margin = 12
    if rounds > 0:
        physical_min = max(0.0, float(start_cards - 6 * rounds - margin))
        physical_max = min(
            float(start_cards),
            float(start_cards - 4 * rounds + max(4, margin // 3)),
        )
    else:
        physical_min = physical_max = float(start_cards)

    if depth_applied:
        try:
            std_cards = max(
                0.5,
                float(depth.get("post_constraint_std_remaining", 0.0) or 0.0),
            )
        except (TypeError, ValueError):
            std_cards = 0.0
        if std_cards <= 0.5:
            std_cards = max(1.0, (physical_max - physical_min) / math.sqrt(12.0))
        interval_low = max(physical_min, mean_remaining - 1.645 * std_cards)
        interval_high = min(physical_max, mean_remaining + 1.645 * std_cards)
        interval_source = "depth_conditioned_particle_total_approx90"
    else:
        std_cards = max(1.0, (physical_max - physical_min) / math.sqrt(12.0))
        interval_low = physical_min
        interval_high = physical_max
        interval_source = "physical_4_to_6_cards_per_round_envelope"

    remaining_ratio = _clip(mean_remaining / max(1.0, float(start_cards)))
    stage = _shoe_stage(remaining_ratio)

    mean_ess_ratio = _clip(float(posterior.get("mean_ess_ratio", 0.0) or 0.0))
    history_factor = min(1.0, rounds / 24.0)
    interval_width = max(0.0, interval_high - interval_low)
    concentration = _clip(1.0 - interval_width / max(1.0, 0.20 * start_cards))
    depth_factor = 1.0 if depth_applied else 0.78
    reliability = _clip(
        history_factor
        * (0.55 + 0.45 * mean_ess_ratio)
        * (0.60 + 0.40 * concentration)
        * depth_factor
    )

    return {
        "available": bool(rounds > 0),
        "start_cards": int(start_cards),
        "conditioned_rounds": int(rounds),
        "mean_remaining_cards": float(mean_remaining),
        "mean_used_cards": float(start_cards - mean_remaining),
        "approx_std_cards": float(std_cards),
        "plausible_interval_low": float(interval_low),
        "plausible_interval_high": float(interval_high),
        "physical_min_remaining": float(physical_min),
        "physical_max_remaining": float(physical_max),
        "remaining_ratio": float(remaining_ratio),
        "shoe_stage": stage,
        "shoe_stage_factor": float(SHOE_STAGE_FACTORS[stage]),
        "reliability": float(reliability),
        "depth_constraint_applied": depth_applied,
        "interval_source": interval_source,
        "semantics": (
            "remaining_card_state_from_particle_mean_and_physical_depth_envelope_"
            "not_exact_hidden_cards"
        ),
    }


def calculate_pattern_survival(
    markov: Mapping[str, Any],
    road_analysis: Mapping[str, Any] | None,
    remaining_card_state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    road = dict(road_analysis or {})
    remaining = dict(remaining_card_state or {})
    profile = dict(markov.get("regime_profile") or {})

    regime = str(profile.get("regime") or markov.get("regime") or "MIXED")
    base_regime = str(profile.get("base_regime") or regime)
    support = _clip(float(markov.get("support_strength", 0.0) or 0.0))
    agreement = _clip(float(markov.get("multi_order_agreement", 0.5) or 0.5))
    regime_stability = _clip(float(profile.get("stability", 0.45) or 0.0))

    entropy_delta = float(profile.get("entropy_delta", 0.0) or 0.0)
    entropy_stability = _clip(1.0 - max(0.0, entropy_delta) / 0.50)

    current_run = max(0, int(profile.get("current_run_length", 0) or 0))
    alternation = _clip(float(profile.get("alternation_ratio", 0.0) or 0.0))
    recent_runs = [
        max(0, int(value))
        for value in list(profile.get("recent_run_lengths") or [])
    ]
    if base_regime == "DRAGON":
        recent_pattern = _clip(current_run / 6.0)
    elif base_regime == "CHOP":
        recent_pattern = alternation
    elif base_regime == "DOUBLE_CHOP":
        window = recent_runs[-4:]
        recent_pattern = (
            sum(1 for length in window if length in {1, 2}) / len(window)
            if window else 0.35
        )
    elif regime == "TRANSITION":
        recent_pattern = 0.15
    else:
        recent_pattern = 0.40

    derived = dict(road.get("derived_road_markov") or {})
    road_reliability = max(
        0.0,
        float(
            road.get(
                "derived_markov_reliability",
                derived.get("reliability", 0.0),
            )
            or 0.0
        ),
    )
    max_road_reliability = max(
        1e-9,
        float(derived.get("max_reliability", 0.18) or 0.18),
    )
    road_support = _clip(road_reliability / max_road_reliability)
    derived_consensus = _clip(
        abs(float(road.get("derived_road_consensus", 0.0) or 0.0))
    )
    derived_component = _clip(0.70 * road_support + 0.30 * derived_consensus)

    remaining_reliability = _clip(
        float(remaining.get("reliability", 0.0) or 0.0)
    )
    stage = str(remaining.get("shoe_stage") or "UNKNOWN").upper()
    stage_factor = float(
        remaining.get("shoe_stage_factor", SHOE_STAGE_FACTORS.get(stage, 0.70))
        or SHOE_STAGE_FACTORS.get(stage, 0.70)
    )
    stage_factor = _clip(stage_factor)

    change_point = bool(profile.get("change_point", False))
    pattern_break = bool(profile.get("pattern_break", False))
    change_factor = 0.25 if (
        change_point or pattern_break or regime == "TRANSITION"
    ) else 1.0

    raw_score = _clip(
        0.20 * support
        + 0.20 * agreement
        + 0.18 * regime_stability
        + 0.15 * recent_pattern
        + 0.10 * entropy_stability
        + 0.10 * derived_component
        + 0.07 * remaining_reliability
    )
    score = _clip(raw_score * stage_factor * change_factor)

    return {
        "score": float(score),
        "raw_score": float(raw_score),
        "pattern": regime,
        "base_pattern": base_regime,
        "shoe_stage": stage,
        "shoe_stage_factor": float(stage_factor),
        "change_point": change_point,
        "pattern_break": pattern_break,
        "change_point_factor": float(change_factor),
        "components": {
            "support": float(support),
            "multi_order_agreement": float(agreement),
            "regime_stability": float(regime_stability),
            "recent_pattern": float(recent_pattern),
            "entropy_stability": float(entropy_stability),
            "derived_road_support": float(derived_component),
            "remaining_card_reliability": float(remaining_reliability),
        },
        "semantics": (
            "pattern_survival_calibration_score_not_next_hand_win_probability"
        ),
    }


def calibrate_markov_probabilities(
    markov_probs: Mapping[str, Any],
    survival_score: float,
) -> dict[str, float]:
    """Return a Banker-neutral direction posterior for Markov evidence.

    P_dir_cal(y) = (1-S) * P_neutral(y) + S * P_dir_signal(y)

    Low Pattern Survival now returns B/P toward equality, not toward the natural
    Banker base-rate advantage. Tie mass is preserved.
    """
    s = _clip(survival_score)
    directional = neutralize_physical_banker_bias(markov_probs)
    neutral = _neutral_prior_with_tie(directional["T"])
    calibrated = {
        outcome: (1.0 - s) * neutral[outcome] + s * directional[outcome]
        for outcome in OUTCOMES
    }
    return _normalize_threeway(calibrated)


__all__ = [
    "PHYSICAL_PRIOR",
    "SHOE_STAGE_FACTORS",
    "build_remaining_card_state",
    "calculate_pattern_survival",
    "calibrate_markov_probabilities",
    "neutralize_physical_banker_bias",
]
