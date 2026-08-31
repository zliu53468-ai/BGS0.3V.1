"""Duration-aware hidden regime calibration for BGS.

This module is intentionally NOT a direct next-hand B/P predictor.

The repository currently has no in-repo multi-shoe road-sequence training set.
The existing 5M SQLite shoe database is composition/depth only, so this module
must not pretend to be an offline-trained HMM/HSMM. Instead it provides a
conservative HSMM-inspired hidden-regime posterior from the Markov engine's
current structural diagnostics and recent run durations.

Its production contract is one-way: it may reduce pattern confidence when the
road appears transitional/noisy; it never creates a B/P vote or increases model
confidence above the pre-existing value.
"""
from __future__ import annotations

from typing import Any, Mapping
import math

MODEL_VERSION = "HSMM-REGIME-CALIBRATION-V2-SMOOTH-TRANSITION"
STATE_NAMES = (
    "S0_PERSISTENT",
    "S1_ALTERNATING",
    "S2_TRANSITION",
    "S3_NOISE",
)

_STATE_PROFILES = {
    "S0_PERSISTENT": {
        "mean": (0.25, 0.70, 0.62, 0.26),
        "std": (0.24, 0.28, 0.24, 0.24),
        "duration_mean": 6.0,
        "pattern_factor": 1.00,
        "markov_factor": 1.00,
        "road_factor": 0.95,
        "hazard_factor": 0.92,
    },
    "S1_ALTERNATING": {
        "mean": (0.84, 0.18, 0.70, 0.30),
        "std": (0.18, 0.20, 0.23, 0.24),
        "duration_mean": 5.0,
        "pattern_factor": 0.95,
        "markov_factor": 0.95,
        "road_factor": 0.95,
        "hazard_factor": 0.90,
    },
    "S2_TRANSITION": {
        "mean": (0.52, 0.34, 0.82, 0.72),
        "std": (0.28, 0.26, 0.18, 0.23),
        "duration_mean": 3.4,
        "pattern_factor": 0.72,
        "markov_factor": 0.70,
        "road_factor": 0.70,
        "hazard_factor": 0.84,
    },
    "S3_NOISE": {
        "mean": (0.55, 0.27, 0.94, 0.55),
        "std": (0.30, 0.24, 0.11, 0.28),
        "duration_mean": 3.0,
        "pattern_factor": 0.60,
        "markov_factor": 0.62,
        "road_factor": 0.55,
        "hazard_factor": 0.62,
    },
}

_PRIOR = {
    "S0_PERSISTENT": 0.25,
    "S1_ALTERNATING": 0.25,
    "S2_TRANSITION": 0.20,
    "S3_NOISE": 0.30,
}

_TRANSITION_WINDOW = 5


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _normalize(values: Mapping[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(value)) for value in values.values())
    if total <= 1e-18:
        return {state: 1.0 / len(STATE_NAMES) for state in STATE_NAMES}
    return {
        state: max(0.0, float(values.get(state, 0.0))) / total
        for state in STATE_NAMES
    }


def _gaussian_logpdf(value: float, mean: float, std: float) -> float:
    sigma = max(0.06, float(std))
    z = (float(value) - float(mean)) / sigma
    return -0.5 * z * z - math.log(sigma)


def _run_height_volatility(recent_runs: list[int]) -> float:
    heights = [max(0, int(value)) for value in recent_runs[-6:] if int(value) > 0]
    if len(heights) < 2:
        return 0.25
    diffs = [abs(right - left) for left, right in zip(heights, heights[1:])]
    return _clip((sum(diffs) / len(diffs)) / 3.0)


def _transition_stability(recent_runs: list[int]) -> float:
    """Return 0..1 convergence of the last 3-5 completed run heights."""
    heights = [
        max(1, int(value))
        for value in recent_runs[-_TRANSITION_WINDOW:]
        if int(value) > 0
    ]
    if len(heights) < 3:
        return 0.50

    diffs = [
        abs(right - left)
        for left, right in zip(heights, heights[1:])
    ]
    if len(diffs) < 2:
        return 0.50

    recent_count = min(2, len(diffs))
    recent = diffs[-recent_count:]
    earlier = diffs[:-recent_count] or diffs[:1]
    recent_mean = sum(recent) / len(recent)
    earlier_mean = sum(earlier) / len(earlier)

    convergence = _clip(
        0.50
        + (earlier_mean - recent_mean)
        / max(1.0, earlier_mean + recent_mean)
    )
    recent_spread = max(recent) - min(recent)
    dispersion_stability = 1.0 - _clip(
        recent_spread / max(1.0, max(recent) if recent else 1.0)
    )
    low_motion = 1.0 - _clip(recent_mean / 4.0)
    return _clip(
        0.45 * convergence
        + 0.35 * dispersion_stability
        + 0.20 * low_motion
    )


def _trailing_alternation_duration(recent5: list[str]) -> int:
    bp = [str(value).upper() for value in recent5 if str(value).upper() in {"B", "P"}]
    if not bp:
        return 1
    duration = 1
    for left, right in reversed(list(zip(bp, bp[1:]))):
        if left != right:
            duration += 1
        else:
            break
    return max(1, duration)


def _state_duration_proxy(
    state: str,
    *,
    current_run: int,
    recent5: list[str],
    recent_runs: list[int],
    change_point: bool,
    transition_stability: float,
) -> float:
    if state == "S0_PERSISTENT":
        return float(max(1, current_run))
    if state == "S1_ALTERNATING":
        return float(_trailing_alternation_duration(recent5))
    if state == "S2_TRANSITION":
        # A single change-point hand should not immediately look like a fully
        # established transition regime. Converging run heights gradually raise
        # the duration proxy toward the transition state's expected duration.
        if change_point:
            return float(1.35 + 1.65 * _clip(transition_stability))
        return float(2.20 + 0.90 * _clip(transition_stability))
    return float(max(1, min(4, len(recent_runs))))


def _duration_log_likelihood(observed: float, expected: float) -> float:
    obs = math.log1p(max(0.0, observed))
    mean = math.log1p(max(1.0, expected))
    sigma = 0.68
    z = (obs - mean) / sigma
    return -0.5 * z * z - math.log(sigma)


def analyze_hidden_regime(markov: Mapping[str, Any]) -> dict[str, Any]:
    """Return an HSMM-inspired hidden regime posterior from current diagnostics.

    The result is calibration metadata. No B/P probability is produced.
    """
    profile = dict(markov.get("regime_profile") or {})
    state = dict(markov.get("state") or {})

    alternation = _clip(float(profile.get("alternation_ratio", 0.5) or 0.5))
    current_run = max(0, int(profile.get("current_run_length", 0) or 0))
    current_run_norm = _clip(current_run / 6.0)

    try:
        entropy_bits = float(markov.get("entropy_bits", 0.0) or 0.0)
    except (TypeError, ValueError):
        entropy_bits = 0.0
    entropy_norm = _clip(entropy_bits / math.log2(3.0))

    recent_runs = [
        max(0, int(value))
        for value in list(profile.get("recent_run_lengths") or [])
    ]
    volatility = _run_height_volatility(recent_runs)
    transition_stability = _transition_stability(recent_runs)
    recent5 = [
        str(value).upper()
        for value in list(state.get("recent5") or [])
        if str(value).upper() in {"B", "P", "T"}
    ]
    change_point = bool(profile.get("change_point", False))
    pattern_break = bool(profile.get("pattern_break", False))

    event_strength = _clip(
        0.50 * (1.0 if change_point else 0.0)
        + 0.30 * (1.0 if pattern_break else 0.0)
    )
    transition_evidence = _clip(
        0.60 * event_strength
        + 0.25 * volatility
        + 0.15 * (1.0 - transition_stability)
    )
    if not (change_point or pattern_break):
        transition_evidence *= 0.55

    observation = (alternation, current_run_norm, entropy_norm, volatility)
    raw_scores: dict[str, float] = {}

    for hidden_state in STATE_NAMES:
        config = _STATE_PROFILES[hidden_state]
        log_score = math.log(max(1e-12, float(_PRIOR[hidden_state])))
        for value, mean, std in zip(observation, config["mean"], config["std"]):
            log_score += _gaussian_logpdf(value, mean, std)

        duration_proxy = _state_duration_proxy(
            hidden_state,
            current_run=current_run,
            recent5=recent5,
            recent_runs=recent_runs,
            change_point=change_point,
            transition_stability=transition_stability,
        )
        duration_weight = 0.50 if hidden_state == "S2_TRANSITION" else 0.35
        log_score += duration_weight * _duration_log_likelihood(
            duration_proxy,
            float(config["duration_mean"]),
        )

        if hidden_state == "S2_TRANSITION":
            # Continuous evidence replaces the old hard 2.8x jump.
            log_score += math.log(1.0 + 0.80 * transition_evidence)
        elif hidden_state in {"S0_PERSISTENT", "S1_ALTERNATING"}:
            # Stable regimes fade gently, so one or two noisy hands cannot force
            # an abrupt transition posterior.
            log_score += math.log(max(0.75, 1.0 - 0.22 * transition_evidence))

        raw_scores[hidden_state] = math.exp(max(-40.0, min(30.0, log_score)))

    posterior = _normalize(raw_scores)
    dominant_state = max(posterior, key=posterior.get)

    posterior_entropy = 0.0
    for probability in posterior.values():
        if probability > 1e-15:
            posterior_entropy -= probability * math.log(probability)
    concentration = _clip(
        1.0 - posterior_entropy / math.log(len(STATE_NAMES))
    )

    sample_count = max(0, int(markov.get("sample_count", 0) or 0))
    history_factor = _clip(sample_count / 24.0)
    reliability = _clip(history_factor * (0.35 + 0.65 * concentration))

    def factor(name: str) -> float:
        weighted = sum(
            posterior[hidden_state] * float(_STATE_PROFILES[hidden_state][name])
            for hidden_state in STATE_NAMES
        )
        return _clip((1.0 - reliability) + reliability * weighted)

    transition_probability = _clip(
        posterior["S2_TRANSITION"] + 0.55 * posterior["S3_NOISE"]
    )
    stable_probability = _clip(
        posterior["S0_PERSISTENT"] + posterior["S1_ALTERNATING"]
    )

    duration_proxy = {
        hidden_state: float(
            _state_duration_proxy(
                hidden_state,
                current_run=current_run,
                recent5=recent5,
                recent_runs=recent_runs,
                change_point=change_point,
                transition_stability=transition_stability,
            )
        )
        for hidden_state in STATE_NAMES
    }

    return {
        "model_version": MODEL_VERSION,
        "trained_from_sequence_dataset": False,
        "parameter_source": (
            "conservative_default_profiles_no_inrepo_multishoe_road_dataset"
        ),
        "available": bool(sample_count >= 4),
        "dominant_state": dominant_state,
        "state_posterior": {
            hidden_state: float(posterior[hidden_state])
            for hidden_state in STATE_NAMES
        },
        "duration_proxy": duration_proxy,
        "posterior_concentration": float(concentration),
        "reliability": float(reliability),
        "transition_probability": float(transition_probability),
        "stable_probability": float(stable_probability),
        "transition_stability": float(transition_stability),
        "transition_evidence": float(transition_evidence),
        "pattern_factor": float(factor("pattern_factor")),
        "markov_factor": float(factor("markov_factor")),
        "road_factor": float(factor("road_factor")),
        "hazard_factor": float(factor("hazard_factor")),
        "observation": {
            "alternation_ratio": float(alternation),
            "current_run_norm": float(current_run_norm),
            "entropy_norm": float(entropy_norm),
            "run_height_volatility": float(volatility),
            "transition_stability": float(transition_stability),
            "transition_evidence": float(transition_evidence),
            "change_point": change_point,
            "pattern_break": pattern_break,
        },
        "semantics": (
            "hsmm_inspired_duration_aware_hidden_regime_calibration_only_"
            "not_next_hand_probability_not_offline_trained"
        ),
    }


__all__ = ["MODEL_VERSION", "STATE_NAMES", "analyze_hidden_regime"]
