"""Conservative calibration of Particle Shoe with the 5M shoe-state database.

The 5M database contains remaining-shoe composition/depth statistics only.  It
must not become a direct road-pattern or B/P voting model.  This module therefore
uses the database only to calibrate the existing Particle Shoe posterior.

The calibrated shoe posterior still passes through BaccaratQuantEngine's physical
baseline correction before it can affect the B/P direction recommendation.  That
prevents baccarat's natural Banker base rate from becoming a free directional vote.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

MAX_DATABASE_WEIGHT = 0.15
DEPTH_ONLY_MAX_RELIABILITY = 0.25
DEPTH_SAMPLE_SCALE = 250_000.0
EXACT_RELIABILITY_FLOOR = 0.20

OUTCOMES = ("B", "P", "T")


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _normalize(probabilities: Mapping[str, Any]) -> dict[str, float]:
    raw = {
        outcome: max(1e-12, float(probabilities.get(outcome, 0.0) or 0.0))
        for outcome in OUTCOMES
    }
    total = sum(raw.values())
    if total <= 1e-12:
        return {"B": 1.0 / 3.0, "P": 1.0 / 3.0, "T": 1.0 / 3.0}
    return {outcome: raw[outcome] / total for outcome in OUTCOMES}


def _state_reliability(estimate: Any) -> float:
    level = str(getattr(estimate, "level", "") or "")
    if level == "exact_shoe_state":
        exact = _clip(float(getattr(estimate, "reliability", 0.0) or 0.0))
        return _clip(EXACT_RELIABILITY_FLOOR + (1.0 - EXACT_RELIABILITY_FLOOR) * exact)
    if level == "depth":
        depth_samples = max(0.0, float(getattr(estimate, "depth_samples", 0) or 0.0))
        sample_strength = depth_samples / (depth_samples + DEPTH_SAMPLE_SCALE)
        return _clip(DEPTH_ONLY_MAX_RELIABILITY * sample_strength)
    return 0.0


def calibrate_particle_shoe_with_database(
    *,
    particle_probabilities: Mapping[str, Any],
    particles: Sequence[Sequence[int]],
    decks: int,
    conditioned_rounds: int,
    mean_ess_ratio: float,
    depth_constraint: Mapping[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Blend the Particle Shoe posterior with matching historical shoe states.

    Each posterior particle is mapped to the 5M database's coarse state key.  We
    query each unique key once, average the database predictive probabilities by
    particle multiplicity, then apply a strictly capped calibration weight.

    This function is a safe no-op if the SQLite database is unavailable.
    """
    base_probabilities = _normalize(particle_probabilities)
    particle_list = [list(particle) for particle in particles if particle]
    base_detail: dict[str, Any] = {
        "available": False,
        "applied": False,
        "max_weight": float(MAX_DATABASE_WEIGHT),
        "weight": 0.0,
        "particle_probabilities": dict(base_probabilities),
        "database_probabilities": None,
        "calibrated_probabilities": dict(base_probabilities),
        "particle_count": len(particle_list),
        "unique_states": 0,
        "exact_state_particles": 0,
        "depth_state_particles": 0,
        "baseline_particles": 0,
        "database_hands": 0,
        "reason": "database_not_checked",
        "semantics": (
            "historical_remaining_shoe_state_calibration_only_not_road_pattern_vote"
        ),
    }
    if not particle_list:
        base_detail["reason"] = "no_particles"
        return base_probabilities, base_detail

    try:
        from shoe_state_db import get_shoe_state_database, state_key_from_counts

        database = get_shoe_state_database()
        info = dict(database.database_info())
        base_detail["available"] = bool(info.get("available", False))
        base_detail["database_hands"] = int(info.get("hands", 0) or 0)
        if not base_detail["available"]:
            base_detail["reason"] = "database_unavailable"
            return base_probabilities, base_detail

        state_counts: Counter[tuple[int, int, int, int, int]] = Counter()
        representatives: dict[tuple[int, int, int, int, int], list[int]] = {}
        for particle in particle_list:
            key = state_key_from_counts(particle, decks)
            state_counts[key] += 1
            representatives.setdefault(key, particle)

        base_detail["unique_states"] = len(state_counts)
        total_particles = float(len(particle_list))
        db_accumulator = {outcome: 0.0 for outcome in OUTCOMES}
        reliability_accumulator = 0.0
        exact_particles = 0
        depth_particles = 0
        baseline_particles = 0
        weighted_exact_samples = 0.0
        weighted_depth_samples = 0.0

        for key, multiplicity in state_counts.items():
            estimate = database.estimate(representatives[key], decks=decks)
            fraction = float(multiplicity) / total_particles
            estimate_probs = _normalize(estimate.probabilities)
            for outcome in OUTCOMES:
                db_accumulator[outcome] += fraction * estimate_probs[outcome]

            state_reliability = _state_reliability(estimate)
            reliability_accumulator += fraction * state_reliability
            weighted_exact_samples += fraction * max(0, int(estimate.samples))
            weighted_depth_samples += fraction * max(0, int(estimate.depth_samples))

            if estimate.level == "exact_shoe_state":
                exact_particles += multiplicity
            elif estimate.level == "depth":
                depth_particles += multiplicity
            else:
                baseline_particles += multiplicity

        db_probabilities = _normalize(db_accumulator)
        history_factor = _clip(float(conditioned_rounds) / 24.0)
        particle_quality = 0.50 + 0.50 * _clip(mean_ess_ratio)
        depth = dict(depth_constraint or {})
        if depth.get("applied"):
            depth_consistency = _clip(float(depth.get("pre_constraint_consistency", 1.0) or 0.0))
            depth_factor = 0.80 + 0.20 * depth_consistency
        else:
            depth_factor = 0.85

        database_weight = _clip(
            MAX_DATABASE_WEIGHT
            * reliability_accumulator
            * history_factor
            * particle_quality
            * depth_factor,
            0.0,
            MAX_DATABASE_WEIGHT,
        )

        calibrated = _normalize({
            outcome: (
                (1.0 - database_weight) * base_probabilities[outcome]
                + database_weight * db_probabilities[outcome]
            )
            for outcome in OUTCOMES
        })

        base_detail.update({
            "applied": bool(database_weight > 1e-9),
            "weight": float(database_weight),
            "database_probabilities": dict(db_probabilities),
            "calibrated_probabilities": dict(calibrated),
            "mean_state_reliability": float(reliability_accumulator),
            "history_factor": float(history_factor),
            "particle_quality_factor": float(particle_quality),
            "depth_factor": float(depth_factor),
            "exact_state_particles": int(exact_particles),
            "depth_state_particles": int(depth_particles),
            "baseline_particles": int(baseline_particles),
            "mean_exact_samples": float(weighted_exact_samples),
            "mean_depth_samples": float(weighted_depth_samples),
            "reason": (
                "bounded_particle_weighted_5m_shoe_state_calibration"
                if database_weight > 1e-9
                else "database_evidence_too_weak"
            ),
            "direction_handling": (
                "database_calibrates_shoe_only_downstream_quant_removes_physical_"
                "banker_baseline_before_direction_fusion"
            ),
        })
        return calibrated, base_detail
    except Exception as exc:
        base_detail["reason"] = "database_calibration_error"
        base_detail["error"] = str(exc)[:240]
        return base_probabilities, base_detail


__all__ = [
    "MAX_DATABASE_WEIGHT",
    "calibrate_particle_shoe_with_database",
]
