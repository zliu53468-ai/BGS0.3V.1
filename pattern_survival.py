"""Remaining-card state and pattern-survival calibration for BGS.

This module does not predict hidden card identities. It converts the existing
probabilistic-shoe posterior into a coarse shoe-stage estimate and calibrates how
strongly historical road/Markov structure may influence the next-hand posterior.

Pattern Survival is NOT a baccarat win probability.

Direction calibration separates:
- PHYSICAL_PRIOR: natural baccarat B/P/T base rates for shoe/EV math.
- an equal-B/P directional baseline for weak historical signals.

The optional hidden-regime layer is HSMM-inspired and duration-aware, but it is
not claimed to be an offline-trained HSMM because the repository has no in-repo
multi-shoe road-sequence training set. It may only reduce pattern confidence.

Multi-order Markov agreement is treated conservatively: agreement does not add
an extra confidence bonus because order-1..4 contexts are nested and correlated;
only meaningful disagreement may reduce Pattern Survival confidence.
"""
from __future__ import annotations

from typing import Any, Mapping
import math

from hsmm_regime import analyze_hidden_regime

PHYSICAL_PRIOR = {"B": 0.4586, "P": 0.4462, "T": 0.0952}
OUTCOMES = ("B", "P", "T")

SHOE_STAGE_FACTORS = {
    "OPENING": 0.45,
    "DEVELOPING": 0.75,
    "MATURE": 1.00,
    "LATE": 0.80,
    "UNKNOWN": 0.70,
}

# Markov multi-order agreement lies in [0.5, 1.0]. Agreement at or above this
# threshold receives no bonus and no penalty. Lower values only reduce trust.
_AGREEMENT_NO_PENALTY_THRESHOLD = 0.75
_AGREEMENT_MAX_CONFLICT_PENALTY = 0.25


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
    tie = _clip(float(tie_probability), 0.0, 0.999999)
    resolved_mass = max(1e-12, 1.0 - tie)
    half = resolved_mass / 2.0
    return {"B": half, "P": half, "T": tie}


def neutralize_physical_banker_bias(
    probabilities: Mapping[str, Any],
) -> dict[str, float]:
    """Convert physical-prior-smoothed probabilities into pure B/P evidence."""
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
            std_cards = max(
                1.0,
                (physical_max - physical_min) / math.sqrt(12.0),
            )
        interval_low = max(physical_min, mean_remaining - 1.645 * std_cards)
        interval_high = min(physical_max, mean_remaining + 1.645 * std_cards)
        interval_source = "depth_conditioned_particle_total_approx90"
    else:
        std_cards = max(
            1.0,
            (physical_max - physical_min) / math.sqrt(12.0),
        )
        interval_low = physical_min
        interval_high = physical_max
        interval_source = "physical_4_to_6_cards_per_round_envelope"

    remaining_ratio = _clip(mean_remaining / max(1.0, float(start_cards)))
    stage = _shoe_stage(remaining_ratio)

    mean_ess_ratio = _clip(
        float(posterior.get("mean_ess_ratio", 0.0) or 0.0)
    )
    history_factor = min(1.0, rounds / 24.0)
    interval_width = max(0.0, interval_high - interval_low)
    concentration = _clip(
        1.0 - interval_width / max(1.0, 0.20 * start_cards)
    )
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


def _agreement_conflict_factor(agreement: float) -> float:
    """Return 1.0 for adequate agreement; only disagreement creates a penalty."""
    value = _clip(float(agreement), 0.5, 1.0)
    if value >= _AGREEMENT_NO_PENALTY_THRESHOLD:
        return 1.0
    conflict = (
        (_AGREEMENT_NO_PENALTY_THRESHOLD - value)
        / max(1e-9, _AGREEMENT_NO_PENALTY_THRESHOLD - 0.5)
    )
    penalty = _AGREEMENT_MAX_CONFLICT_PENALTY * _clip(conflict)
    return _clip(1.0 - penalty)


def calculate_pattern_survival(
    markov: Mapping[str, Any],
    road_analysis: Mapping[str, Any] | None,
    remaining_card_state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a 0..1 confidence gate for currently observed road structure.

    The hidden-regime factor is one-way: it may only reduce the pre-existing
    Pattern Survival score. It never creates a B/P direction.
    """
    road = dict(road_analysis or {})
    remaining = dict(remaining_card_state or {})
    profile = dict(markov.get("regime_profile") or {})

    regime = str(profile.get("regime") or markov.get("regime") or "MIXED")
    base_regime = str(profile.get("base_regime") or regime)
    support = _clip(float(markov.get("support_strength", 0.0) or 0.0))
    agreement = _clip(
        float(markov.get("multi_order_agreement", 0.5) or 0.5)
    )
    agreement_factor = _agreement_conflict_factor(agreement)
    regime_stability = _clip(
        float(profile.get("stability", 0.45) or 0.0)
    )

    entropy_delta = float(profile.get("entropy_delta", 0.0) or 0.0)
    entropy_stability = _clip(
        1.0 - max(0.0, entropy_delta) / 0.50
    )

    current_run = max(
        0,
        int(profile.get("current_run_length", 0) or 0),
    )
    alternation = _clip(
        float(profile.get("alternation_ratio", 0.0) or 0.0)
    )
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
    derived_component = _clip(
        0.70 * road_support + 0.30 * derived_consensus
    )

    remaining_reliability = _clip(
        float(remaining.get("reliability", 0.0) or 0.0)
    )
    stage = str(
        remaining.get("shoe_stage") or "UNKNOWN"
    ).upper()
    stage_factor = float(
        remaining.get(
            "shoe_stage_factor",
            SHOE_STAGE_FACTORS.get(stage, 0.70),
        )
        or SHOE_STAGE_FACTORS.get(stage, 0.70)
    )
    stage_factor = _clip(stage_factor)

    change_point = bool(profile.get("change_point", False))
    pattern_break = bool(profile.get("pattern_break", False))
    change_factor = 0.25 if (
        change_point
        or pattern_break
        or regime == "TRANSITION"
    ) else 1.0

    # Base structural score intentionally excludes a positive agreement term.
    # The weights sum to 1.0. Agreement only applies a one-way conflict penalty.
    base_score = _clip(
        0.25 * support
        + 0.22 * regime_stability
        + 0.18 * recent_pattern
        + 0.12 * entropy_stability
        + 0.13 * derived_component
        + 0.10 * remaining_reliability
    )
    raw_score = _clip(base_score * agreement_factor)

    hidden_regime = analyze_hidden_regime(markov)
    hidden_factor = _clip(
        float(hidden_regime.get("pattern_factor", 1.0) or 1.0)
    )

    pre_hidden_score = _clip(
        raw_score * stage_factor * change_factor
    )
    score = _clip(pre_hidden_score * hidden_factor)

    return {
        "score": float(score),
        "raw_score": float(raw_score),
        "base_structural_score": float(base_score),
        "pre_hidden_regime_score": float(pre_hidden_score),
        "pattern": regime,
        "base_pattern": base_regime,
        "shoe_stage": stage,
        "shoe_stage_factor": float(stage_factor),
        "change_point": change_point,
        "pattern_break": pattern_break,
        "change_point_factor": float(change_factor),
        "hidden_regime": hidden_regime,
        "hidden_regime_factor": float(hidden_factor),
        "multi_order_agreement": float(agreement),
        "multi_order_conflict_factor": float(agreement_factor),
        "components": {
            "support": float(support),
            "multi_order_agreement": float(agreement),
            "multi_order_conflict_factor": float(agreement_factor),
            "regime_stability": float(regime_stability),
            "recent_pattern": float(recent_pattern),
            "entropy_stability": float(entropy_stability),
            "derived_road_support": float(derived_component),
            "remaining_card_reliability": float(remaining_reliability),
            "hidden_regime_factor": float(hidden_factor),
        },
        "semantics": (
            "pattern_survival_with_agreement_conflict_only_and_duration_aware_"
            "hidden_regime_downweight_not_next_hand_win_probability"
        ),
    }


def calibrate_markov_probabilities(
    markov_probs: Mapping[str, Any],
    survival_score: float,
) -> dict[str, float]:
    """Return a Banker-neutral direction posterior for Markov evidence.

    Low Pattern Survival returns B/P toward equality, not toward Banker's natural
    base-rate advantage. Tie mass is preserved.
    """
    s = _clip(survival_score)
    directional = neutralize_physical_banker_bias(markov_probs)
    neutral = _neutral_prior_with_tie(directional["T"])
    calibrated = {
        outcome: (
            (1.0 - s) * neutral[outcome]
            + s * directional[outcome]
        )
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
