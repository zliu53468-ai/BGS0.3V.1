"""Duration-aware hidden-regime calibration for BGS.

This is a calibration layer, not a direct B/P predictor. For the LSTM-primary
pipeline it only softens confidence when the road looks transitional/noisy; it
never creates or overrides the LSTM direction.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping
import math

from run_length_hazard import build_runs

MODEL_VERSION = "HSMM-REGIME-CALIBRATION-V2-LSTM-SOFT"
STATE_NAMES = ("S0_PERSISTENT", "S1_ALTERNATING", "S2_TRANSITION", "S3_NOISE")
_STATE_PROFILES = {
    "S0_PERSISTENT": {"mean": (0.25, 0.70, 0.62, 0.26), "std": (0.24, 0.28, 0.24, 0.24), "duration_mean": 6.0, "pattern_factor": 1.00, "markov_factor": 1.00, "road_factor": 0.98, "hazard_factor": 0.96},
    "S1_ALTERNATING": {"mean": (0.84, 0.18, 0.70, 0.30), "std": (0.18, 0.20, 0.23, 0.24), "duration_mean": 5.0, "pattern_factor": 0.98, "markov_factor": 0.98, "road_factor": 0.97, "hazard_factor": 0.95},
    "S2_TRANSITION": {"mean": (0.52, 0.34, 0.82, 0.72), "std": (0.28, 0.26, 0.18, 0.23), "duration_mean": 3.0, "pattern_factor": 0.84, "markov_factor": 0.85, "road_factor": 0.84, "hazard_factor": 0.90},
    "S3_NOISE": {"mean": (0.55, 0.27, 0.94, 0.55), "std": (0.30, 0.24, 0.11, 0.28), "duration_mean": 3.0, "pattern_factor": 0.88, "markov_factor": 0.88, "road_factor": 0.86, "hazard_factor": 0.88},
}
_PRIOR = {"S0_PERSISTENT": 0.25, "S1_ALTERNATING": 0.25, "S2_TRANSITION": 0.20, "S3_NOISE": 0.30}
LSTM_TRANSITION_MIN_FACTOR = 0.92


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _normalize(values: Mapping[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(v)) for v in values.values())
    if total <= 1e-18:
        return {state: 1.0 / len(STATE_NAMES) for state in STATE_NAMES}
    return {state: max(0.0, float(values.get(state, 0.0))) / total for state in STATE_NAMES}


def _gaussian_logpdf(value: float, mean: float, std: float) -> float:
    sigma = max(0.06, float(std)); z = (float(value) - float(mean)) / sigma
    return -0.5 * z * z - math.log(sigma)


def _run_height_volatility(recent_runs: list[int]) -> float:
    heights = [max(0, int(v)) for v in recent_runs[-6:] if int(v) > 0]
    if len(heights) < 2:
        return 0.25
    diffs = [abs(r - l) for l, r in zip(heights, heights[1:])]
    return _clip((sum(diffs) / len(diffs)) / 3.0)


def _trailing_alternation_duration(recent5: list[str]) -> int:
    bp = [str(v).upper() for v in recent5 if str(v).upper() in {"B", "P"}]
    if not bp:
        return 1
    duration = 1
    for left, right in reversed(list(zip(bp, bp[1:]))):
        if left != right: duration += 1
        else: break
    return max(1, duration)


def _state_duration_proxy(state: str, *, current_run: int, recent5: list[str], recent_runs: list[int], change_point: bool) -> float:
    if state == "S0_PERSISTENT": return float(max(1, current_run))
    if state == "S1_ALTERNATING": return float(_trailing_alternation_duration(recent5))
    if state == "S2_TRANSITION": return 1.5 if change_point else 2.5
    return float(max(1, min(4, len(recent_runs))))


def _duration_log_likelihood(observed: float, expected: float) -> float:
    obs, mean, sigma = math.log1p(max(0.0, observed)), math.log1p(max(1.0, expected)), 0.75
    z = (obs - mean) / sigma
    return -0.5 * z * z - math.log(sigma)


def analyze_hidden_regime(markov: Mapping[str, Any]) -> dict[str, Any]:
    profile = dict(markov.get("regime_profile") or {}); state = dict(markov.get("state") or {})
    alternation = _clip(float(profile.get("alternation_ratio", 0.5) or 0.5))
    current_run = max(0, int(profile.get("current_run_length", 0) or 0)); current_run_norm = _clip(current_run / 6.0)
    try: entropy_bits = float(markov.get("entropy_bits", 0.0) or 0.0)
    except (TypeError, ValueError): entropy_bits = 0.0
    entropy_norm = _clip(entropy_bits / math.log2(3.0))
    recent_runs = [max(0, int(v)) for v in list(profile.get("recent_run_lengths") or [])]
    volatility = _run_height_volatility(recent_runs)
    recent5 = [str(v).upper() for v in list(state.get("recent5") or []) if str(v).upper() in {"B", "P", "T"}]
    change_point = bool(profile.get("change_point", False)); pattern_break = bool(profile.get("pattern_break", False))
    observation = (alternation, current_run_norm, entropy_norm, volatility)
    event_strength = _clip(0.55 * float(change_point) + 0.35 * float(pattern_break) + 0.10 * volatility)
    raw_scores: dict[str, float] = {}
    for hidden_state in STATE_NAMES:
        config = _STATE_PROFILES[hidden_state]; log_score = math.log(max(1e-12, float(_PRIOR[hidden_state])))
        for value, mean, std in zip(observation, config["mean"], config["std"]): log_score += _gaussian_logpdf(value, mean, std)
        duration_proxy = _state_duration_proxy(hidden_state, current_run=current_run, recent5=recent5, recent_runs=recent_runs, change_point=change_point)
        log_score += 0.30 * _duration_log_likelihood(duration_proxy, float(config["duration_mean"]))
        if hidden_state == "S2_TRANSITION" and (change_point or pattern_break): log_score += math.log(1.0 + 0.60 * event_strength)
        elif hidden_state in {"S0_PERSISTENT", "S1_ALTERNATING"} and change_point: log_score += math.log(max(0.80, 1.0 - 0.18 * event_strength))
        raw_scores[hidden_state] = math.exp(max(-40.0, min(30.0, log_score)))
    posterior = _normalize(raw_scores); dominant_state = max(posterior, key=posterior.get)
    posterior_entropy = -sum(p * math.log(p) for p in posterior.values() if p > 1e-15)
    concentration = _clip(1.0 - posterior_entropy / math.log(len(STATE_NAMES)))
    sample_count = max(0, int(markov.get("sample_count", 0) or 0)); history_factor = _clip(sample_count / 24.0)
    reliability = _clip(history_factor * (0.35 + 0.65 * concentration))
    def factor(name: str) -> float:
        weighted = sum(posterior[s] * float(_STATE_PROFILES[s][name]) for s in STATE_NAMES)
        return _clip((1.0 - reliability) + reliability * weighted)
    transition_probability = _clip(posterior["S2_TRANSITION"] + 0.45 * posterior["S3_NOISE"])
    return {
        "model_version": MODEL_VERSION, "trained_from_sequence_dataset": False,
        "parameter_source": "conservative_default_profiles_no_inrepo_multishoe_road_dataset",
        "available": bool(sample_count >= 4), "dominant_state": dominant_state,
        "state_posterior": {s: float(posterior[s]) for s in STATE_NAMES},
        "posterior_concentration": float(concentration), "reliability": float(reliability),
        "transition_probability": float(transition_probability),
        "stable_probability": float(_clip(posterior["S0_PERSISTENT"] + posterior["S1_ALTERNATING"])),
        "pattern_factor": float(factor("pattern_factor")), "markov_factor": float(factor("markov_factor")),
        "road_factor": float(factor("road_factor")), "hazard_factor": float(factor("hazard_factor")),
        "observation": {"alternation_ratio": float(alternation), "current_run_norm": float(current_run_norm), "entropy_norm": float(entropy_norm), "run_height_volatility": float(volatility), "change_point": change_point, "pattern_break": pattern_break},
        "semantics": "hsmm_inspired_duration_aware_hidden_regime_calibration_only_not_next_hand_probability_not_offline_trained",
    }


def _clean_bp(history: Iterable[Any] | None) -> list[str]:
    result: list[str] = []
    for item in history or []:
        raw = item.get("outcome") or item.get("actual") or item.get("actual_outcome") or item.get("virtual_outcome") if isinstance(item, Mapping) else item
        value = str(raw or "").upper().strip()
        if value in {"B", "P"}: result.append(value)
    return result[-2000:]


def analyze_lstm_transition(history: Iterable[Any] | None, *, hazard: Mapping[str, Any] | None = None) -> dict[str, Any]:
    sequence = _clean_bp(history)
    if len(sequence) < 4:
        return {"available": False, "transition_probability": 0.0, "confidence_factor": 1.0, "direction_override": False, "reason": "insufficient_history"}
    recent = sequence[-12:]
    alternation = sum(l != r for l, r in zip(recent, recent[1:])) / max(1, len(recent) - 1)
    run_lengths = [length for _, length in build_runs(sequence)]; volatility = _run_height_volatility(run_lengths)
    recent_runs = run_lengths[-5:]
    if len(recent_runs) >= 3:
        diffs = [abs(r - l) for l, r in zip(recent_runs, recent_runs[1:])]
        convergence = _clip(1.0 - diffs[-1] / max(1.0, float(max(diffs))))
    else: convergence = 0.5
    turn_probability = _clip(float(dict(hazard or {}).get("turn_probability", 0.5) or 0.5))
    hazard_pressure = _clip((turn_probability - 0.55) / 0.25)
    alternation_transition = _clip(1.0 - abs(alternation - 0.5) / 0.5)
    transition_probability = _clip(0.35 * alternation_transition + 0.30 * volatility + 0.20 * hazard_pressure + 0.15 * (1.0 - convergence))
    factor = _clip(1.0 - (1.0 - LSTM_TRANSITION_MIN_FACTOR) * transition_probability, LSTM_TRANSITION_MIN_FACTOR, 1.0)
    return {
        "available": True, "transition_probability": float(transition_probability), "confidence_factor": float(factor), "penalty": float(1.0 - factor),
        "alternation_ratio": float(alternation), "run_height_volatility": float(volatility), "run_height_convergence": float(convergence),
        "hazard_turn_probability": float(turn_probability), "hazard_pressure": float(hazard_pressure), "direction_override": False,
        "semantics": "soft_transition_confidence_calibration_only_no_BP_vote",
    }


__all__ = ["MODEL_VERSION", "STATE_NAMES", "LSTM_TRANSITION_MIN_FACTOR", "analyze_hidden_regime", "analyze_lstm_transition"]
