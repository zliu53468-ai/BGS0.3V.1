"""HSMM-inspired hidden-regime gate for the Road-Primary baccarat predictor.

This module is deliberately not a standalone Banker/Player predictor. It combines
run-length hazard evidence with a duration-aware hidden-state posterior to decide
how much a small continuation/turn residual correction may influence the existing
Road V1 + derived-road probability.

States:
- S0_PERSISTENT: current structure is persisting;
- S1_ALTERNATING: switching / single-jump structure is persisting;
- S2_TRANSITION: road statistics are changing;
- S3_NOISE: no stable regime is evident.

The directional sign comes from the empirical run-length hazard model. HSMM-like
state probabilities only gate reliability. Derived-road sequence/geometry breaks
are used as cross-confirmation, never as an independent extra vote.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import math

from run_length_hazard import MAX_HAZARD_RELIABILITY, analyze_run_length_hazard, build_runs

MODEL_VERSION = "ROAD-HSMM-HAZARD-REGIME-GATE-V1"
STATE_NAMES = ("S0_PERSISTENT", "S1_ALTERNATING", "S2_TRANSITION", "S3_NOISE")
MAX_REGIME_DIRECTION_WEIGHT = 0.12
MIN_REGIME_HISTORY = 8

_STATE_PROFILES = {
    "S0_PERSISTENT": {
        "mean": (0.24, 0.68, 0.24, 0.34, 0.28, 0.28),
        "std": (0.20, 0.25, 0.22, 0.22, 0.24, 0.24),
        "duration_mean": 5.5,
    },
    "S1_ALTERNATING": {
        "mean": (0.82, 0.18, 0.28, 0.68, 0.26, 0.26),
        "std": (0.18, 0.18, 0.22, 0.22, 0.24, 0.24),
        "duration_mean": 5.0,
    },
    "S2_TRANSITION": {
        "mean": (0.52, 0.30, 0.68, 0.62, 0.68, 0.68),
        "std": (0.26, 0.24, 0.20, 0.22, 0.22, 0.22),
        "duration_mean": 2.5,
    },
    "S3_NOISE": {
        "mean": (0.52, 0.25, 0.82, 0.50, 0.54, 0.54),
        "std": (0.28, 0.22, 0.16, 0.25, 0.25, 0.25),
        "duration_mean": 3.0,
    },
}
_PRIOR = {"S0_PERSISTENT": 0.27, "S1_ALTERNATING": 0.23, "S2_TRANSITION": 0.22, "S3_NOISE": 0.28}


def _clip(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(number):
        return lo
    return max(lo, min(hi, number))


def _normalize(values: Mapping[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(values.get(name, 0.0))) for name in STATE_NAMES)
    if total <= 1e-18:
        return {name: 1.0 / len(STATE_NAMES) for name in STATE_NAMES}
    return {name: max(0.0, float(values.get(name, 0.0))) / total for name in STATE_NAMES}


def _gaussian_logpdf(value: float, mean: float, std: float) -> float:
    sigma = max(0.07, float(std))
    z = (float(value) - float(mean)) / sigma
    return -0.5 * z * z - math.log(sigma)


def _duration_log_likelihood(observed: float, expected: float) -> float:
    sigma = 0.78
    obs = math.log1p(max(0.0, float(observed)))
    mean = math.log1p(max(1.0, float(expected)))
    z = (obs - mean) / sigma
    return -0.5 * z * z - math.log(sigma)


def _run_height_volatility(run_lengths: Sequence[int]) -> float:
    values = [max(1, int(value)) for value in run_lengths[-7:]]
    if len(values) < 2:
        return 0.25
    diffs = [abs(right - left) for left, right in zip(values, values[1:])]
    return _clip((sum(diffs) / len(diffs)) / 3.0)


def _recent_switch_rate(sequence: Sequence[str], size: int = 10) -> float:
    values = list(sequence[-max(2, int(size)):])
    if len(values) < 2:
        return 0.5
    return sum(left != right for left, right in zip(values, values[1:])) / float(len(values) - 1)


def _trailing_alternation_duration(sequence: Sequence[str]) -> int:
    values = list(sequence)
    if not values:
        return 1
    duration = 1
    for left, right in reversed(list(zip(values, values[1:]))):
        if left != right:
            duration += 1
        else:
            break
    return duration


def _mean_model_value(layer: Mapping[str, Any], field: str, default: float = 0.5) -> float:
    models = dict(layer.get("models") or {})
    values: list[float] = []
    for model in models.values():
        if not isinstance(model, Mapping):
            continue
        try:
            values.append(_clip(model.get(field, default)))
        except Exception:
            continue
    return sum(values) / len(values) if values else default


def _derived_break_features(derived: Mapping[str, Any]) -> tuple[float, float, float, float]:
    sequence_layer = dict(derived.get("sequence_layer") or derived)
    geometry_layer = dict(derived.get("geometry_layer") or {})
    sequence_break = _mean_model_value(sequence_layer, "pattern_break_probability", 0.5)
    geometry_break = _mean_model_value(geometry_layer, "shape_break_probability", 0.5)
    likelihood = dict(derived.get("likelihood") or {})
    p_b = _clip(likelihood.get("B", 0.5))
    ask_separation = abs(p_b - 0.5) * 2.0
    agreement = _clip(derived.get("cross_road_agreement", 0.0))
    if "layer_agreement" in derived and not bool(derived.get("layer_agreement")):
        agreement *= 0.75
    return sequence_break, geometry_break, ask_separation, agreement


def _state_duration_proxy(
    state: str,
    *,
    sequence: Sequence[str],
    current_run: int,
    run_lengths: Sequence[int],
    transition_pressure: float,
) -> float:
    if state == "S0_PERSISTENT":
        return float(max(1, current_run))
    if state == "S1_ALTERNATING":
        return float(max(1, _trailing_alternation_duration(sequence)))
    if state == "S2_TRANSITION":
        return 1.5 if transition_pressure >= 0.55 else 2.5
    return float(max(1, min(4, len(run_lengths))))


def _state_posterior(
    *,
    sequence: Sequence[str],
    hazard: Mapping[str, Any],
    derived: Mapping[str, Any],
) -> dict[str, Any]:
    runs = build_runs(sequence)
    run_lengths = [int(length) for _, length in runs]
    current_run = run_lengths[-1] if run_lengths else 1
    switch_rate = _recent_switch_rate(sequence, 10)
    current_run_norm = _clip(current_run / 6.0)
    volatility = _run_height_volatility(run_lengths)
    hazard_turn = _clip(hazard.get("turn_probability", 0.5))
    sequence_break, geometry_break, ask_separation, cross_agreement = _derived_break_features(derived)
    transition_pressure = _clip(
        0.30 * hazard_turn
        + 0.26 * sequence_break
        + 0.24 * geometry_break
        + 0.12 * volatility
        + 0.08 * (1.0 - cross_agreement)
    )
    observation = (
        switch_rate,
        current_run_norm,
        volatility,
        hazard_turn,
        sequence_break,
        geometry_break,
    )

    raw: dict[str, float] = {}
    for state in STATE_NAMES:
        profile = _STATE_PROFILES[state]
        score = math.log(max(1e-12, _PRIOR[state]))
        for value, mean, std in zip(observation, profile["mean"], profile["std"]):
            score += _gaussian_logpdf(value, mean, std)
        duration = _state_duration_proxy(
            state,
            sequence=sequence,
            current_run=current_run,
            run_lengths=run_lengths,
            transition_pressure=transition_pressure,
        )
        score += 0.28 * _duration_log_likelihood(duration, float(profile["duration_mean"]))
        if state == "S2_TRANSITION":
            score += math.log(0.85 + 0.55 * transition_pressure)
        elif state == "S3_NOISE":
            score += math.log(0.90 + 0.30 * volatility)
        raw[state] = math.exp(max(-40.0, min(30.0, score)))

    posterior = _normalize(raw)
    entropy = -sum(p * math.log(p) for p in posterior.values() if p > 1e-15)
    concentration = _clip(1.0 - entropy / math.log(len(STATE_NAMES)))
    history_factor = _clip(len(sequence) / 24.0)
    reliability = _clip(history_factor * (0.30 + 0.70 * concentration))
    dominant_state = max(posterior, key=posterior.get)
    transition_probability = _clip(posterior["S2_TRANSITION"] + 0.40 * posterior["S3_NOISE"])
    stable_probability = _clip(posterior["S0_PERSISTENT"] + posterior["S1_ALTERNATING"])

    return {
        "state_posterior": {name: float(posterior[name]) for name in STATE_NAMES},
        "dominant_state": dominant_state,
        "posterior_concentration": float(concentration),
        "reliability": float(reliability),
        "transition_probability": float(transition_probability),
        "stable_probability": float(stable_probability),
        "transition_pressure": float(transition_pressure),
        "observation": {
            "switch_rate_10": float(switch_rate),
            "current_run_length": int(current_run),
            "current_run_norm": float(current_run_norm),
            "run_height_volatility": float(volatility),
            "hazard_turn_probability": float(hazard_turn),
            "derived_sequence_break": float(sequence_break),
            "derived_geometry_break": float(geometry_break),
            "derived_ask_separation": float(ask_separation),
            "derived_cross_road_agreement": float(cross_agreement),
        },
    }


def analyze_road_regime_gate(
    sequence: Sequence[str],
    *,
    derived_analysis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a capped hazard-direction signal gated by hidden-regime evidence."""
    values = [str(value).upper().strip() for value in sequence if str(value).upper().strip() in {"B", "P"}]
    derived = dict(derived_analysis or {})
    hazard = dict(analyze_run_length_hazard(values))
    regime = _state_posterior(sequence=values, hazard=hazard, derived=derived)

    hazard_likelihood = dict(hazard.get("likelihood") or {"B": 0.5, "P": 0.5})
    hazard_p_b = _clip(hazard_likelihood.get("B", 0.5), 0.20, 0.80)
    hazard_separation = abs(hazard_p_b - 0.5) * 2.0
    hazard_rel = _clip(hazard.get("reliability", 0.0), 0.0, MAX_HAZARD_RELIABILITY)
    hazard_support_ratio = _clip(hazard_rel / max(1e-9, MAX_HAZARD_RELIABILITY))
    regime_rel = _clip(regime.get("reliability", 0.0))
    posterior = dict(regime.get("state_posterior") or {})
    persistent = _clip(posterior.get("S0_PERSISTENT", 0.0))
    alternating = _clip(posterior.get("S1_ALTERNATING", 0.0))
    transition = _clip(posterior.get("S2_TRANSITION", 0.0))
    noise = _clip(posterior.get("S3_NOISE", 0.0))

    current_side = str(hazard.get("current_side") or (values[-1] if values else "B"))
    hazard_prefers_continue = (
        (hazard_p_b >= 0.5 and current_side == "B")
        or (hazard_p_b < 0.5 and current_side == "P")
    )
    state_alignment = (
        persistent
        if hazard_prefers_continue
        else _clip(alternating + 0.72 * transition)
    )
    state_alignment = _clip(0.35 + 0.65 * state_alignment)

    derived_likelihood = dict(derived.get("likelihood") or {})
    derived_p_b = _clip(derived_likelihood.get("B", 0.5))
    derived_rel = _clip(derived.get("reliability", 0.0), 0.0, 0.18)
    derived_active = int(derived.get("active_road_count", 0) or 0)
    derived_agrees = ((derived_p_b >= 0.5) == (hazard_p_b >= 0.5))
    if derived_active >= 2 and derived_rel > 0.0 and abs(derived_p_b - 0.5) >= 0.03:
        cross_factor = 1.10 if derived_agrees else 0.72
    else:
        cross_factor = 0.92

    noise_factor = _clip(1.0 - 0.55 * noise, 0.55, 1.0)
    history_factor = _clip(len(values) / 20.0)
    if len(values) < MIN_REGIME_HISTORY or not bool(hazard.get("available")):
        reliability = 0.0
    else:
        reliability = (
            MAX_REGIME_DIRECTION_WEIGHT
            * history_factor
            * hazard_support_ratio
            * regime_rel
            * (0.45 + 0.55 * hazard_separation)
            * state_alignment
            * cross_factor
            * noise_factor
        )
        reliability = _clip(reliability, 0.0, MAX_REGIME_DIRECTION_WEIGHT)

    return {
        "model_version": MODEL_VERSION,
        "available": bool(reliability > 0.0),
        "likelihood": {"B": float(hazard_p_b), "P": float(1.0 - hazard_p_b)},
        "reliability": float(reliability),
        "max_reliability": float(MAX_REGIME_DIRECTION_WEIGHT),
        "hazard": hazard,
        "hazard_reliability": float(hazard_rel),
        "hazard_support_ratio": float(hazard_support_ratio),
        "hazard_separation": float(hazard_separation),
        "current_side": current_side,
        "hazard_prefers_continue": bool(hazard_prefers_continue),
        "derived_cross_confirmation": {
            "active_road_count": int(derived_active),
            "reliability": float(derived_rel),
            "p_b": float(derived_p_b),
            "agrees_with_hazard": bool(derived_agrees),
            "cross_factor": float(cross_factor),
        },
        **regime,
        "parameter_source": "conservative_duration_aware_profiles_plus_empirical_run_hazard_no_offline_multishoe_fit",
        "direction_authority": "capped_residual_only",
        "semantics": "hazard_supplies_continue_turn_direction_hsmm_hidden_regime_only_gates_reliability",
    }


__all__ = [
    "MODEL_VERSION",
    "STATE_NAMES",
    "MAX_REGIME_DIRECTION_WEIGHT",
    "MIN_REGIME_HISTORY",
    "analyze_road_regime_gate",
]
