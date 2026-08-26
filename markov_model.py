"""Variable-order road-state three-way Markov predictor for BGS.

B, P and T remain fully modeled.  The predictor combines:
- variable-order raw B/P/T contexts (orders 1..4),
- B/P road run-length state with nested coarse/full fallback,
- regime detection (DRAGON / CHOP / DOUBLE_CHOP / MIXED / TRANSITION),
- regime-adaptive exponential decay,
- Bayesian smoothing and support-aware hierarchical blending.

The model is still history-based and does not imply deterministic baccarat patterns.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping
import math

from shoe_depth_estimator import ShoeDepthEstimator

MODEL_VERSION = "THREEWAY-VARIABLE-ORDER-ROAD-STATE-V3.1-NESTED-ROAD-CANDIDATE"
OUTCOMES = ("B", "P", "T")
PHYSICAL_PRIOR = {"B": 0.4586, "P": 0.4462, "T": 0.0952}

DECAY = 0.95
BAYES_PRIOR_STRENGTH = 6.0
MAX_ENTROPY = math.log2(3.0)
MAX_ORDER = 4

_ORDER_SUPPORT_SCALE = {1: 3.0, 2: 4.5, 3: 6.5, 4: 9.0}
_ORDER_ALPHA_CAP = {1: 0.72, 2: 0.64, 3: 0.54, 4: 0.44}
_ROAD_SUPPORT_SCALE = {"road_coarse": 5.0, "road_full": 8.0}
_ROAD_ALPHA_CAP = {"road_coarse": 0.34, "road_full": 0.26}
_ROAD_FULL_MIN_RELIABILITY = 0.50


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _clean_threeway(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for item in values:
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
        if value in OUTCOMES:
            result.append(value)
    return result[-2000:]


def _clean_bp(sequence: Iterable[str]) -> list[str]:
    return [value for value in sequence if value in {"B", "P"}]


def _bp_runs(sequence: Iterable[str]) -> list[tuple[str, int]]:
    bp = _clean_bp(sequence)
    if not bp:
        return []
    runs: list[tuple[str, int]] = []
    side = bp[0]
    length = 1
    for value in bp[1:]:
        if value == side:
            length += 1
        else:
            runs.append((side, length))
            side = value
            length = 1
    runs.append((side, length))
    return runs


def _run_bucket(length: int) -> str:
    value = max(0, int(length))
    if value <= 1:
        return "1"
    if value == 2:
        return "2"
    if value == 3:
        return "3"
    return "4+"


def _alternation_ratio(sequence: Iterable[str], window: int = 10) -> float:
    bp = _clean_bp(sequence)[-max(2, int(window)):]
    if len(bp) < 2:
        return 0.0
    changes = sum(1 for left, right in zip(bp, bp[1:]) if left != right)
    return changes / max(1, len(bp) - 1)


def _base_regime(sequence: Iterable[str]) -> str:
    bp = _clean_bp(sequence)
    runs = _bp_runs(bp)
    if len(bp) < 4 or not runs:
        return "MIXED"

    current_length = runs[-1][1]
    recent_runs = [length for _, length in runs[-5:]]
    recent_completed = [length for _, length in runs[-5:-1]]

    if current_length >= 4:
        return "DRAGON"

    alt_ratio = _alternation_ratio(bp, window=10)
    one_ratio = (
        sum(1 for length in recent_runs if length == 1) / len(recent_runs)
        if recent_runs else 0.0
    )
    if alt_ratio >= 0.72 and one_ratio >= 0.60:
        return "CHOP"

    double_evidence = recent_completed[-3:]
    if len(double_evidence) >= 2:
        near_two = sum(1 for length in double_evidence if length == 2)
        building = current_length in {1, 2}
        if near_two >= 2 and building:
            return "DOUBLE_CHOP"

    return "MIXED"


def _detect_regime(sequence: Iterable[str]) -> Dict[str, Any]:
    values = list(sequence)
    bp = _clean_bp(values)
    runs = _bp_runs(bp)
    current_base = _base_regime(values)
    previous_base = _base_regime(values[:-3]) if len(values) >= 7 else "MIXED"

    break_from_dragon = (
        len(runs) >= 2
        and runs[-2][1] >= 4
        and runs[-1][1] == 1
    )
    break_from_chop = (
        previous_base == "CHOP"
        and runs
        and runs[-1][1] >= 2
    )
    current_run_length = runs[-1][1] if runs else 0
    regime_changed = (
        len(values) >= 7
        and previous_base != current_base
        and (
            (current_base == "DRAGON" and current_run_length == 4)
            or (current_base == "MIXED" and previous_base != "MIXED")
            or (current_base == "DOUBLE_CHOP" and current_run_length == 1)
        )
    )

    transition = bool(break_from_dragon or break_from_chop or regime_changed)
    regime = "TRANSITION" if transition else current_base

    recent_runs = [length for _, length in runs[-5:]]
    if regime == "TRANSITION":
        stability = 0.25
    elif regime == "MIXED":
        stability = 0.45
    elif len(recent_runs) >= 3:
        stability = 0.80
    else:
        stability = 0.60

    return {
        "regime": regime,
        "base_regime": current_base,
        "previous_regime": previous_base,
        "transition": transition,
        "stability": float(stability),
        "alternation_ratio": float(_alternation_ratio(values, window=10)),
        "recent_run_lengths": recent_runs,
        "current_run_length": int(runs[-1][1]) if runs else 0,
        "previous_run_length": int(runs[-2][1]) if len(runs) >= 2 else 0,
    }


def _adaptive_decay(sequence: Iterable[str], base_decay: float) -> float:
    profile = _detect_regime(sequence)
    base = _clip(base_decay, 0.80, 0.995)
    regime = profile["regime"]

    if regime == "TRANSITION":
        return _clip(base - 0.08, 0.82, 0.94)
    if regime == "DRAGON":
        return _clip(base + 0.03, 0.90, 0.985)
    if regime in {"CHOP", "DOUBLE_CHOP"}:
        return _clip(base + 0.02, 0.90, 0.98)
    return _clip(base, 0.86, 0.97)


def _density_state(sequence: list[str]) -> Dict[str, Any]:
    recent5 = sequence[-5:]
    banker_count = recent5.count("B")
    player_count = recent5.count("P")
    density_delta = banker_count - player_count
    if density_delta >= 2:
        density = "High"
    elif density_delta <= -2:
        density = "Low"
    else:
        density = "Medium"
    return {
        "recent5": recent5,
        "banker_count_recent5": banker_count,
        "player_count_recent5": player_count,
        "density_delta": density_delta,
        "density": density,
    }


def _road_state(sequence: list[str]) -> Dict[str, Any]:
    runs = _bp_runs(sequence)
    regime = _detect_regime(sequence)
    density = _density_state(sequence)

    current_side = runs[-1][0] if runs else ""
    current_length = runs[-1][1] if runs else 0
    previous_length = runs[-2][1] if len(runs) >= 2 else 0
    previous2_length = runs[-3][1] if len(runs) >= 3 else 0
    previous_side = runs[-2][0] if len(runs) >= 2 else ""

    current_bucket = _run_bucket(current_length) if current_length else "0"
    previous_bucket = _run_bucket(previous_length) if previous_length else "0"
    previous2_bucket = _run_bucket(previous2_length) if previous2_length else "0"

    tie_trigger = "HasTie" if "T" in sequence[-3:] else "NoTie"
    coarse_key = (
        f"RC|side={current_side or 'NA'}|cur={current_bucket}|"
        f"prev={previous_bucket}|reg={regime['regime']}|tie={tie_trigger}"
    )
    full_key = (
        f"RF|side={current_side or 'NA'}|cur={current_bucket}|"
        f"prev={previous_bucket}|prev2={previous2_bucket}|"
        f"reg={regime['regime']}|density={density['density']}|tie={tie_trigger}"
    )

    return {
        "current_side": current_side,
        "current_run_length": int(current_length),
        "current_run_bucket": current_bucket,
        "previous_side": previous_side,
        "previous_run_length": int(previous_length),
        "previous_run_bucket": previous_bucket,
        "previous2_run_length": int(previous2_length),
        "previous2_run_bucket": previous2_bucket,
        "coarse_key": coarse_key,
        "full_key": full_key,
        "regime": regime,
        **density,
    }


def encode_threeway_state(history: Iterable[Any]) -> Dict[str, Any]:
    sequence = _clean_threeway(history)
    density = _density_state(sequence)
    road = _road_state(sequence)
    direction_context = "".join(sequence[-2:]) if len(sequence) >= 2 else "".join(sequence)
    recent3 = sequence[-3:]
    tie_trigger = "HasTie" if "T" in recent3 else "NoTie"

    key = road["full_key"] if sequence else ""
    return {
        "ready": len(sequence) >= 2,
        "direction_context": direction_context,
        "density": density["density"],
        "tie_trigger": tie_trigger,
        "key": key,
        "recent5": density["recent5"],
        "recent3": recent3,
        "banker_count_recent5": density["banker_count_recent5"],
        "player_count_recent5": density["player_count_recent5"],
        "density_delta": density["density_delta"],
        "current_side": road["current_side"],
        "current_run_length": road["current_run_length"],
        "current_run_bucket": road["current_run_bucket"],
        "previous_run_length": road["previous_run_length"],
        "previous_run_bucket": road["previous_run_bucket"],
        "previous2_run_length": road["previous2_run_length"],
        "previous2_run_bucket": road["previous2_run_bucket"],
        "road_coarse_key": road["coarse_key"],
        "road_full_key": road["full_key"],
        "regime": road["regime"]["regime"],
        "base_regime": road["regime"]["base_regime"],
        "previous_regime": road["regime"]["previous_regime"],
        "regime_transition": road["regime"]["transition"],
        "regime_stability": road["regime"]["stability"],
        "alternation_ratio": road["regime"]["alternation_ratio"],
        "recent_run_lengths": road["regime"]["recent_run_lengths"],
    }


def _context_keys(sequence: list[str]) -> Dict[str, str]:
    keys: Dict[str, str] = {}
    for order in range(1, MAX_ORDER + 1):
        if len(sequence) >= order:
            keys[f"order_{order}"] = f"O{order}|{''.join(sequence[-order:])}"
    if sequence:
        road = _road_state(sequence)
        keys["road_coarse"] = road["coarse_key"]
        keys["road_full"] = road["full_key"]
    return keys


def _decay_all(
    transitions: Dict[str, Dict[str, float]],
    decay: float,
) -> None:
    for counts in transitions.values():
        for outcome in OUTCOMES:
            counts[outcome] *= decay


def _build_decayed_transition_table(
    sequence: list[str],
    *,
    base_decay: float,
) -> Dict[str, Dict[str, float]]:
    transitions: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"B": 0.0, "P": 0.0, "T": 0.0}
    )

    # Leakage-safe: every context for outcome i is encoded only from outcomes < i.
    for index in range(1, len(sequence)):
        prefix = sequence[:index]
        step_decay = _adaptive_decay(prefix, base_decay)
        _decay_all(transitions, step_decay)
        for key in _context_keys(prefix).values():
            transitions[key][sequence[index]] += 1.0

    return {key: dict(value) for key, value in transitions.items()}


def _posterior_probabilities(
    counts: Mapping[str, float],
    *,
    prior_strength: float,
) -> Dict[str, float]:
    denominator = (
        sum(float(counts.get(outcome, 0.0) or 0.0) for outcome in OUTCOMES)
        + prior_strength
    )
    if denominator <= 0.0:
        return dict(PHYSICAL_PRIOR)
    return {
        outcome: (
            float(counts.get(outcome, 0.0) or 0.0)
            + prior_strength * PHYSICAL_PRIOR[outcome]
        ) / denominator
        for outcome in OUTCOMES
    }


def _blend(
    base: Mapping[str, float],
    overlay: Mapping[str, float],
    alpha: float,
) -> Dict[str, float]:
    weight = _clip(alpha)
    mixed = {
        outcome: (1.0 - weight) * float(base[outcome])
        + weight * float(overlay[outcome])
        for outcome in OUTCOMES
    }
    total = sum(mixed.values())
    if total <= 1e-12:
        return dict(PHYSICAL_PRIOR)
    return {outcome: mixed[outcome] / total for outcome in OUTCOMES}


def _support(counts: Mapping[str, float]) -> float:
    return sum(float(counts.get(outcome, 0.0) or 0.0) for outcome in OUTCOMES)


def _reliability(support: float, scale: float) -> float:
    value = max(0.0, float(support))
    denominator = value + max(1e-9, float(scale))
    return value / denominator


def _direction_vote(probabilities: Mapping[str, float]) -> str:
    return "B" if float(probabilities["B"]) >= float(probabilities["P"]) else "P"


def _hierarchical_probabilities(
    sequence: list[str],
    transition_table: Mapping[str, Mapping[str, float]],
    *,
    prior_strength: float,
) -> tuple[Dict[str, float], Dict[str, Any]]:
    contexts = _context_keys(sequence)
    blended = dict(PHYSICAL_PRIOR)
    details: Dict[str, Any] = {}
    weighted_votes: list[tuple[str, float]] = []
    dominant_name = "physical_prior"
    dominant_score = 0.0
    dominant_counts = {"B": 0.0, "P": 0.0, "T": 0.0}

    for order in range(1, MAX_ORDER + 1):
        name = f"order_{order}"
        key = contexts.get(name)
        if not key:
            continue
        counts = dict(transition_table.get(key, {"B": 0.0, "P": 0.0, "T": 0.0}))
        support = _support(counts)
        posterior = _posterior_probabilities(counts, prior_strength=prior_strength)
        reliability = _reliability(support, _ORDER_SUPPORT_SCALE[order])
        alpha = _ORDER_ALPHA_CAP[order] * reliability
        blended = _blend(blended, posterior, alpha)

        score = alpha * max(0.25, support)
        if score > dominant_score:
            dominant_name = name
            dominant_score = score
            dominant_counts = dict(counts)
        if alpha > 0.0:
            weighted_votes.append((_direction_vote(posterior), alpha))

        details[name] = {
            "key": key,
            "support": float(support),
            "reliability": float(reliability),
            "alpha": float(alpha),
            "probabilities": dict(posterior),
            "counts": dict(counts),
        }

    # road_full is a refinement of road_coarse, so they must not both influence
    # the same decision.  Use full only once it has enough support; otherwise
    # back off to coarse.  Both candidates remain visible in diagnostics.
    road_candidates: Dict[str, Dict[str, Any]] = {}
    for name in ("road_coarse", "road_full"):
        key = contexts.get(name)
        if not key:
            continue
        counts = dict(transition_table.get(key, {"B": 0.0, "P": 0.0, "T": 0.0}))
        support = _support(counts)
        posterior = _posterior_probabilities(counts, prior_strength=prior_strength)
        reliability = _reliability(support, _ROAD_SUPPORT_SCALE[name])
        raw_alpha = _ROAD_ALPHA_CAP[name] * reliability
        road_candidates[name] = {
            "key": key,
            "support": float(support),
            "reliability": float(reliability),
            "raw_alpha": float(raw_alpha),
            "probabilities": dict(posterior),
            "counts": dict(counts),
        }

    full_candidate = road_candidates.get("road_full")
    coarse_candidate = road_candidates.get("road_coarse")
    full_ready = bool(
        full_candidate
        and float(full_candidate["reliability"]) >= _ROAD_FULL_MIN_RELIABILITY
    )
    if full_ready:
        selected_road_name = "road_full"
    elif coarse_candidate:
        selected_road_name = "road_coarse"
    elif full_candidate:
        selected_road_name = "road_full"
    else:
        selected_road_name = ""

    for name, candidate in road_candidates.items():
        applied = bool(name == selected_road_name)
        alpha = float(candidate["raw_alpha"]) if applied else 0.0
        details[name] = {
            "key": str(candidate["key"]),
            "support": float(candidate["support"]),
            "reliability": float(candidate["reliability"]),
            "raw_alpha": float(candidate["raw_alpha"]),
            "alpha": float(alpha),
            "applied": applied,
            "probabilities": dict(candidate["probabilities"]),
            "counts": dict(candidate["counts"]),
        }

    if selected_road_name:
        selected = road_candidates[selected_road_name]
        selected_alpha = float(selected["raw_alpha"])
        blended = _blend(blended, selected["probabilities"], selected_alpha)
        selected_support = float(selected["support"])
        score = selected_alpha * max(0.25, selected_support)
        if score > dominant_score:
            dominant_name = selected_road_name
            dominant_score = score
            dominant_counts = dict(selected["counts"])
        if selected_alpha > 0.0:
            weighted_votes.append(
                (_direction_vote(selected["probabilities"]), selected_alpha)
            )

    vote_total = sum(weight for _, weight in weighted_votes)
    if vote_total <= 1e-12:
        agreement = 0.5
    else:
        banker_weight = sum(weight for vote, weight in weighted_votes if vote == "B")
        player_weight = sum(weight for vote, weight in weighted_votes if vote == "P")
        agreement = max(banker_weight, player_weight) / vote_total

    support_values = [
        float(item["support"])
        for name, item in details.items()
        if isinstance(item, Mapping)
        and (
            not str(name).startswith("road_")
            or bool(item.get("applied", False))
        )
    ]
    max_support = max(support_values, default=0.0)
    support_strength = _reliability(max_support, 8.0)

    diagnostics = {
        "contexts": details,
        "dominant_context": dominant_name,
        "dominant_counts": dominant_counts,
        "multi_order_agreement": float(agreement),
        "support_strength": float(support_strength),
        "max_context_support": float(max_support),
        "road_selection": {
            "mode": "nested_full_else_coarse",
            "selected_context": selected_road_name,
            "full_min_reliability": float(_ROAD_FULL_MIN_RELIABILITY),
            "full_reliability": (
                float(full_candidate["reliability"])
                if full_candidate else 0.0
            ),
            "coarse_reliability": (
                float(coarse_candidate["reliability"])
                if coarse_candidate else 0.0
            ),
            "double_count_prevented": True,
        },
    }
    return blended, diagnostics


def _entropy(probabilities: Mapping[str, float]) -> float:
    total = 0.0
    for outcome in OUTCOMES:
        p = max(1e-15, min(1.0, float(probabilities[outcome])))
        total -= p * math.log2(p)
    return total


def update_and_predict_engine(
    history: Iterable[Any],
    *,
    decay: float = DECAY,
    prior_strength: float = BAYES_PRIOR_STRENGTH,
) -> Dict[str, Any]:
    sequence = _clean_threeway(history)
    base_decay = _clip(float(decay), 0.80, 0.995)
    prior_strength = max(1e-9, float(prior_strength))
    shoe = ShoeDepthEstimator().estimate(sequence)
    current_state = encode_threeway_state(sequence)
    regime_profile = _detect_regime(sequence)
    current_decay = _adaptive_decay(sequence, base_decay)
    transition_table = _build_decayed_transition_table(
        sequence,
        base_decay=base_decay,
    )

    probabilities, hierarchy = _hierarchical_probabilities(
        sequence,
        transition_table,
        prior_strength=prior_strength,
    )

    entropy = _entropy(probabilities)
    entropy_weight = _clip(1.0 - entropy / MAX_ENTROPY)
    agreement = float(hierarchy["multi_order_agreement"])
    support_strength = float(hierarchy["support_strength"])
    regime_stability = float(regime_profile["stability"])

    # Entropy remains the primary confidence source. Support, multi-order
    # agreement and regime stability prevent sparse high-order states from
    # looking artificially certain.
    evidence_quality = _clip(
        0.55
        + 0.20 * agreement
        + 0.15 * support_strength
        + 0.10 * regime_stability
    )
    base_weight = _clip(entropy_weight * evidence_quality)
    final_weight = _clip(base_weight * float(shoe.shoe_progress))

    direction = "B" if probabilities["B"] >= probabilities["P"] else "P"
    dominant_counts = dict(hierarchy["dominant_counts"])
    effective_support = _support(dominant_counts)
    tie_risk_active = probabilities["T"] > 0.15

    return {
        "model_version": MODEL_VERSION,
        "engine": "THREEWAY_VARIABLE_ORDER_ROAD_STATE_MARKOV",
        "history": sequence,
        "sample_count": len(sequence),
        "state": current_state,
        "state_key": current_state["key"],
        "decay": float(current_decay),
        "base_decay": float(base_decay),
        "adaptive_decay": float(current_decay),
        "regime": str(regime_profile["regime"]),
        "regime_profile": dict(regime_profile),
        "prior": dict(PHYSICAL_PRIOR),
        "prior_strength": prior_strength,
        "max_order": MAX_ORDER,
        "transition_counts": {
            outcome: float(dominant_counts.get(outcome, 0.0) or 0.0)
            for outcome in OUTCOMES
        },
        "effective_support": float(effective_support),
        "state_count": len(transition_table),
        "probabilities": {
            outcome: float(probabilities[outcome]) for outcome in OUTCOMES
        },
        "direction": direction,
        "entropy_bits": float(entropy),
        "max_entropy_bits": float(MAX_ENTROPY),
        "entropy_weight": float(entropy_weight),
        "base_weight": float(base_weight),
        "shoe_progress": float(shoe.shoe_progress),
        "final_weight": float(final_weight),
        "shoe_depth": shoe.as_dict(),
        "tie_risk_active": bool(tie_risk_active),
        "tie_risk_threshold": 0.15,
        "hierarchical_backoff": dict(hierarchy),
        "multi_order_agreement": float(agreement),
        "support_strength": float(support_strength),
        "dominant_context": str(hierarchy["dominant_context"]),
        "confidence_semantics": (
            "entropy_support_agreement_regime_weight_not_guaranteed_win_probability"
        ),
    }


def predict_markov(
    history: Iterable[Any],
    *,
    road_context: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    del road_context
    return update_and_predict_engine(history)


__all__ = [
    "MODEL_VERSION",
    "OUTCOMES",
    "PHYSICAL_PRIOR",
    "DECAY",
    "BAYES_PRIOR_STRENGTH",
    "MAX_ORDER",
    "encode_threeway_state",
    "update_and_predict_engine",
    "predict_markov",
]
