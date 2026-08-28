"""Support-aware variable-order three-way Markov predictor for BGS.

V3.3 keeps the stable V3 road/regime design and adds:
- explicit order-1..4 transition counters,
- support-threshold backoff (K=4, alpha=0.75),
- 12-hand entropy change-point detection,
- regime-aware adaptive forgetting with soft recent refocus,
- direction-neutral recent-density state buckets,
- Bayesian/Dirichlet smoothing,
- nested road coarse/full auxiliary state without double counting.

This is a stochastic history model. It does not imply deterministic baccarat
patterns or guaranteed future outcomes.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping
import math

from shoe_depth_estimator import ShoeDepthEstimator

MODEL_VERSION = "THREEWAY-VARIABLE-ORDER-SUPPORT-BACKOFF-V3.3-MOMENTUM-DEBIAS"
OUTCOMES = ("B", "P", "T")
PHYSICAL_PRIOR = {"B": 0.4586, "P": 0.4462, "T": 0.0952}

MAX_ORDER = 4
SUPPORT_THRESHOLD = 4
BACKOFF_ALPHA = 0.75
BAYES_PRIOR_STRENGTH = 6.0
DECAY = 0.95  # legacy name: this is retention lambda, not decay intensity.
MAX_ENTROPY = math.log2(3.0)
ENTROPY_WINDOW = 12
ENTROPY_SPIKE_THRESHOLD = 0.22
RECENT_FOCUS_WINDOW = 6

_ROAD_SUPPORT_SCALE = {"road_coarse": 5.0, "road_full": 8.0}
_ROAD_ALPHA_CAP = {"road_coarse": 0.34, "road_full": 0.26}
_ROAD_FULL_MIN_RELIABILITY = 0.50


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _normalize(values: Mapping[str, float]) -> Dict[str, float]:
    raw = {outcome: max(1e-12, float(values.get(outcome, 0.0) or 0.0)) for outcome in OUTCOMES}
    total = sum(raw.values())
    if total <= 1e-12:
        return dict(PHYSICAL_PRIOR)
    return {outcome: raw[outcome] / total for outcome in OUTCOMES}


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
    side, length = bp[0], 1
    for value in bp[1:]:
        if value == side:
            length += 1
        else:
            runs.append((side, length))
            side, length = value, 1
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


def _window_entropy(sequence: Iterable[str], window: int = ENTROPY_WINDOW) -> float:
    values = list(sequence)[-max(1, int(window)):]
    if not values:
        return MAX_ENTROPY
    counts = {outcome: values.count(outcome) for outcome in OUTCOMES}
    total = float(len(values))
    entropy = 0.0
    for outcome in OUTCOMES:
        p = counts[outcome] / total
        if p > 0.0:
            entropy -= p * math.log2(p)
    return float(entropy)


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
        if near_two >= 2 and current_length in {1, 2}:
            return "DOUBLE_CHOP"

    return "MIXED"


def _detect_regime(sequence: Iterable[str]) -> Dict[str, Any]:
    values = list(sequence)
    bp = _clean_bp(values)
    runs = _bp_runs(bp)

    current_base = _base_regime(values)
    previous_base = _base_regime(values[:-3]) if len(values) >= 7 else "MIXED"

    entropy_current = _window_entropy(values, ENTROPY_WINDOW)
    if len(values) >= ENTROPY_WINDOW * 2:
        previous_window = values[-ENTROPY_WINDOW * 2:-ENTROPY_WINDOW]
        entropy_previous = _window_entropy(previous_window, ENTROPY_WINDOW)
    else:
        entropy_previous = entropy_current
    entropy_delta = entropy_current - entropy_previous
    entropy_spike = bool(
        len(values) >= ENTROPY_WINDOW * 2
        and entropy_delta >= ENTROPY_SPIKE_THRESHOLD
    )

    break_from_dragon = bool(
        len(runs) >= 2 and runs[-2][1] >= 4 and runs[-1][1] == 1
    )

    previous_chop = _base_regime(values[:-1]) == "CHOP" if len(values) >= 5 else False
    break_from_chop = bool(
        previous_chop and runs and runs[-1][1] >= 2
    )

    current_run_length = runs[-1][1] if runs else 0
    regime_changed = bool(
        len(values) >= 7
        and previous_base != current_base
        and (
            (current_base == "DRAGON" and current_run_length == 4)
            or (current_base == "MIXED" and previous_base != "MIXED")
            or (current_base == "DOUBLE_CHOP" and current_run_length == 1)
        )
    )

    pattern_break = bool(break_from_dragon or break_from_chop or regime_changed)
    change_point = bool(entropy_spike or pattern_break)
    transition = change_point
    regime = "TRANSITION" if transition else current_base

    recent_runs = [length for _, length in runs[-5:]]
    if regime == "TRANSITION":
        stability = 0.20
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
        "change_point": change_point,
        "pattern_break": pattern_break,
        "entropy_window": ENTROPY_WINDOW,
        "entropy_current": float(entropy_current),
        "entropy_previous": float(entropy_previous),
        "entropy_delta": float(entropy_delta),
        "entropy_spike": entropy_spike,
        "entropy_spike_threshold": float(ENTROPY_SPIKE_THRESHOLD),
        "stability": float(stability),
        "alternation_ratio": float(_alternation_ratio(values, window=10)),
        "recent_run_lengths": recent_runs,
        "current_run_length": int(current_run_length),
        "previous_run_length": int(runs[-2][1]) if len(runs) >= 2 else 0,
        "break_from_dragon": break_from_dragon,
        "break_from_chop": break_from_chop,
        "regime_changed": regime_changed,
        "focus_window": int(RECENT_FOCUS_WINDOW if change_point else 0),
    }


def _adaptive_retention_lambda(
    sequence: Iterable[str],
    base_lambda: float = DECAY,
) -> float:
    """Return retention lambda used by w(age)=lambda**age.

    The user-facing "decay intensity" is delta = 1-lambda.
    Therefore a change point *raises decay intensity* by lowering lambda.
    """
    profile = _detect_regime(sequence)
    base = _clip(base_lambda, 0.70, 0.995)

    if profile["change_point"]:
        return 0.72
    if profile["regime"] == "DRAGON":
        return _clip(base + 0.025, 0.90, 0.98)
    if profile["regime"] in {"CHOP", "DOUBLE_CHOP"}:
        return _clip(base + 0.015, 0.89, 0.975)
    if profile["regime"] == "MIXED":
        return _clip(base - 0.015, 0.86, 0.96)
    return _clip(base - 0.05, 0.78, 0.94)


def _density_state(sequence: list[str]) -> Dict[str, Any]:
    recent5 = sequence[-5:]
    banker_count = recent5.count("B")
    player_count = recent5.count("P")
    delta = banker_count - player_count
    imbalance = abs(delta)
    if imbalance >= 3:
        density = "STRONG_IMBALANCE"
    elif imbalance >= 2:
        density = "MODERATE_IMBALANCE"
    else:
        density = "BALANCED"
    dominant_side = "B" if delta > 0 else "P" if delta < 0 else ""
    return {
        "recent5": recent5,
        "banker_count_recent5": banker_count,
        "player_count_recent5": player_count,
        "density_delta": delta,
        "density_abs_delta": int(imbalance),
        "dominant_side_recent5": dominant_side,
        "density": density,
        "density_semantics": "direction_neutral_recent5_imbalance_bucket",
    }


def _road_state(sequence: list[str]) -> Dict[str, Any]:
    runs = _bp_runs(sequence)
    regime = _detect_regime(sequence)
    density = _density_state(sequence)

    current_side = runs[-1][0] if runs else ""
    current_length = runs[-1][1] if runs else 0
    previous_length = runs[-2][1] if len(runs) >= 2 else 0
    previous2_length = runs[-3][1] if len(runs) >= 3 else 0

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
    recent3 = sequence[-3:]
    return {
        "ready": len(sequence) >= 2,
        "direction_context": "".join(sequence[-2:]) if sequence else "",
        "density": density["density"],
        "density_semantics": density["density_semantics"],
        "tie_trigger": "HasTie" if "T" in recent3 else "NoTie",
        "key": road["full_key"] if sequence else "",
        "recent5": density["recent5"],
        "recent3": recent3,
        "banker_count_recent5": density["banker_count_recent5"],
        "player_count_recent5": density["player_count_recent5"],
        "density_delta": density["density_delta"],
        "density_abs_delta": density["density_abs_delta"],
        "dominant_side_recent5": density["dominant_side_recent5"],
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
        "change_point": road["regime"]["change_point"],
        "entropy_current": road["regime"]["entropy_current"],
        "entropy_delta": road["regime"]["entropy_delta"],
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


def _new_counts() -> Dict[str, float]:
    return {"B": 0.0, "P": 0.0, "T": 0.0}


def _build_transition_tables(
    sequence: list[str],
    *,
    retention_lambda: float,
) -> tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Build decayed and raw transition counters for all 1..4 orders.

    For a transition with age a:
        effective_count = lambda ** a
    Raw support remains the literal number of observations and is used for the
    K=4 backoff gate.
    """
    weighted: Dict[str, Dict[str, float]] = defaultdict(_new_counts)
    raw: Dict[str, Dict[str, float]] = defaultdict(_new_counts)
    n = len(sequence)

    for index in range(1, n):
        prefix = sequence[:index]
        age = max(0, n - 1 - index)
        weight = float(retention_lambda ** age)
        outcome = sequence[index]
        for key in _context_keys(prefix).values():
            weighted[key][outcome] += weight
            raw[key][outcome] += 1.0

    return (
        {key: dict(value) for key, value in weighted.items()},
        {key: dict(value) for key, value in raw.items()},
    )


def _support(counts: Mapping[str, float]) -> float:
    return sum(float(counts.get(outcome, 0.0) or 0.0) for outcome in OUTCOMES)


def _reliability(support: float, scale: float) -> float:
    value = max(0.0, float(support))
    return value / (value + max(1e-9, float(scale)))


def _posterior_probabilities(
    counts: Mapping[str, float],
    *,
    prior_strength: float,
    parent_prior: Mapping[str, float] | None = None,
) -> Dict[str, float]:
    """Dirichlet/Bayesian smoothing.

    P(y|c) = (N(c,y) + beta * pi_y) / (N(c) + beta)
    where pi is either the physical baccarat prior or a supplied parent prior.
    """
    prior = _normalize(parent_prior or PHYSICAL_PRIOR)
    denominator = _support(counts) + prior_strength
    if denominator <= 1e-12:
        return dict(prior)
    return {
        outcome: (
            float(counts.get(outcome, 0.0) or 0.0)
            + prior_strength * prior[outcome]
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
    return _normalize(mixed)


def _direction_vote(probabilities: Mapping[str, float]) -> str:
    return "B" if float(probabilities["B"]) >= float(probabilities["P"]) else "P"


def _support_aware_markov(
    sequence: list[str],
    weighted_table: Mapping[str, Mapping[str, float]],
    raw_table: Mapping[str, Mapping[str, float]],
    *,
    prior_strength: float,
) -> tuple[Dict[str, float], Dict[str, Any]]:
    """Strict variable-order backoff.

    Start from the highest available order N. If raw support < K, transfer the
    decision to N-1 and multiply confidence by alpha=0.75 for every backoff step.

        backoff_penalty = alpha ** (# failed higher-order gates)

    The selected lower-order posterior is then shrunk toward the physical prior
    by that penalty, preventing sparse contexts from looking overconfident.
    """
    contexts = _context_keys(sequence)
    highest = min(MAX_ORDER, len(sequence))
    details: Dict[str, Any] = {}
    selected_order = 0
    selected_name = "physical_prior"
    selected_counts = _new_counts()
    selected_posterior = dict(PHYSICAL_PRIOR)
    penalty = 1.0
    backoff_steps = 0

    for order in range(highest, 0, -1):
        name = f"order_{order}"
        key = contexts.get(name)
        if not key:
            continue

        weighted_counts = dict(weighted_table.get(key, _new_counts()))
        raw_counts = dict(raw_table.get(key, _new_counts()))
        raw_support = _support(raw_counts)
        effective_support = _support(weighted_counts)
        posterior = _posterior_probabilities(
            weighted_counts,
            prior_strength=prior_strength,
        )
        qualifies = bool(raw_support >= SUPPORT_THRESHOLD)

        details[name] = {
            "key": key,
            "raw_support": float(raw_support),
            "effective_support": float(effective_support),
            "support_threshold": int(SUPPORT_THRESHOLD),
            "qualifies": qualifies,
            "posterior": dict(posterior),
            "counts": weighted_counts,
        }

        if qualifies:
            selected_order = order
            selected_name = name
            selected_counts = weighted_counts
            selected_posterior = posterior
            break

        # Only an actual N -> N-1 transfer consumes one backoff penalty.
        # Order-1 has no lower order, so a sparse O1 is handled as the final
        # fallback without inventing a fourth backoff step.
        if order > 1:
            penalty *= BACKOFF_ALPHA
            backoff_steps += 1

    if selected_order == 0:
        # No order reached K. Use order-1 if it exists; the accumulated penalty
        # records the uncertainty created by all failed gates.
        key = contexts.get("order_1")
        if key:
            selected_order = 1
            selected_name = "order_1"
            selected_counts = dict(weighted_table.get(key, _new_counts()))
            selected_posterior = _posterior_probabilities(
                selected_counts,
                prior_strength=prior_strength,
            )
        else:
            penalty = 0.0

    probability = _blend(PHYSICAL_PRIOR, selected_posterior, penalty)

    # Diagnostics for all lower orders not visited above.
    for order in range(1, highest + 1):
        name = f"order_{order}"
        if name in details:
            continue
        key = contexts.get(name)
        if not key:
            continue
        weighted_counts = dict(weighted_table.get(key, _new_counts()))
        raw_counts = dict(raw_table.get(key, _new_counts()))
        raw_support = _support(raw_counts)
        details[name] = {
            "key": key,
            "raw_support": float(raw_support),
            "effective_support": float(_support(weighted_counts)),
            "support_threshold": int(SUPPORT_THRESHOLD),
            "qualifies": bool(raw_support >= SUPPORT_THRESHOLD),
            "posterior": _posterior_probabilities(
                weighted_counts,
                prior_strength=prior_strength,
            ),
            "counts": weighted_counts,
        }

    votes: list[tuple[str, float]] = []
    for order in range(1, highest + 1):
        item = details.get(f"order_{order}")
        if not item:
            continue
        raw_support = float(item["raw_support"])
        reliability = _reliability(raw_support, SUPPORT_THRESHOLD)
        votes.append((_direction_vote(item["posterior"]), reliability))

    vote_total = sum(weight for _, weight in votes)
    if vote_total <= 1e-12:
        agreement = 0.5
    else:
        b_weight = sum(weight for vote, weight in votes if vote == "B")
        p_weight = sum(weight for vote, weight in votes if vote == "P")
        agreement = max(b_weight, p_weight) / vote_total

    return probability, {
        "mode": "strict_support_aware_variable_order_backoff",
        "support_threshold": int(SUPPORT_THRESHOLD),
        "backoff_alpha": float(BACKOFF_ALPHA),
        "highest_available_order": int(highest),
        "selected_order": int(selected_order),
        "selected_context": selected_name,
        "backoff_steps": int(backoff_steps),
        "backoff_penalty": float(penalty),
        "contexts": details,
        "selected_counts": dict(selected_counts),
        "selected_posterior": dict(selected_posterior),
        "multi_order_agreement": float(agreement),
    }


def _apply_nested_road(
    sequence: list[str],
    base_probability: Mapping[str, float],
    weighted_table: Mapping[str, Mapping[str, float]],
    raw_table: Mapping[str, Mapping[str, float]],
    *,
    prior_strength: float,
) -> tuple[Dict[str, float], Dict[str, Any]]:
    contexts = _context_keys(sequence)
    candidates: Dict[str, Dict[str, Any]] = {}

    for name in ("road_coarse", "road_full"):
        key = contexts.get(name)
        if not key:
            continue
        weighted_counts = dict(weighted_table.get(key, _new_counts()))
        raw_counts = dict(raw_table.get(key, _new_counts()))
        raw_support = _support(raw_counts)
        effective_support = _support(weighted_counts)
        posterior = _posterior_probabilities(
            weighted_counts,
            prior_strength=prior_strength,
            parent_prior=base_probability,
        )
        reliability = _reliability(effective_support, _ROAD_SUPPORT_SCALE[name])
        candidates[name] = {
            "key": key,
            "raw_support": float(raw_support),
            "effective_support": float(effective_support),
            "posterior": posterior,
            "reliability": float(reliability),
            "raw_alpha": float(_ROAD_ALPHA_CAP[name] * reliability),
            "counts": weighted_counts,
        }

    full = candidates.get("road_full")
    coarse = candidates.get("road_coarse")
    full_ready = bool(
        full
        and float(full["raw_support"]) >= SUPPORT_THRESHOLD
        and float(full["reliability"]) >= _ROAD_FULL_MIN_RELIABILITY
    )
    selected_name = (
        "road_full" if full_ready
        else "road_coarse" if coarse
        else "road_full" if full
        else ""
    )

    probability = dict(base_probability)
    selected_alpha = 0.0
    if selected_name:
        selected = candidates[selected_name]
        selected_alpha = float(selected["raw_alpha"])
        probability = _blend(probability, selected["posterior"], selected_alpha)

    details: Dict[str, Any] = {}
    for name, item in candidates.items():
        details[name] = {
            **item,
            "applied": bool(name == selected_name),
            "alpha": float(item["raw_alpha"]) if name == selected_name else 0.0,
        }

    return probability, {
        "mode": "nested_full_else_coarse",
        "selected_context": selected_name,
        "selected_alpha": float(selected_alpha),
        "full_min_reliability": float(_ROAD_FULL_MIN_RELIABILITY),
        "double_count_prevented": True,
        "contexts": details,
    }


def _entropy(probabilities: Mapping[str, float]) -> float:
    value = 0.0
    for outcome in OUTCOMES:
        p = max(1e-15, min(1.0, float(probabilities[outcome])))
        value -= p * math.log2(p)
    return float(value)


def update_and_predict_engine(
    history: Iterable[Any],
    *,
    decay: float = DECAY,
    prior_strength: float = BAYES_PRIOR_STRENGTH,
) -> Dict[str, Any]:
    sequence = _clean_threeway(history)
    prior_strength = max(1e-9, float(prior_strength))
    shoe = ShoeDepthEstimator().estimate(sequence)
    current_state = encode_threeway_state(sequence)
    regime_profile = _detect_regime(sequence)

    retention_lambda = _adaptive_retention_lambda(sequence, float(decay))
    decay_intensity = 1.0 - retention_lambda

    # A change point no longer discards the old shoe and trains on only six hands.
    # The existing lambda=0.72 change-point decay already gives the most recent
    # observations much more weight while keeping older transitions as weak context.
    training_sequence = sequence
    focus_applied = bool(regime_profile["change_point"])
    focus_mode = (
        "soft_recent_refocus_full_history"
        if focus_applied else "normal_full_history"
    )

    weighted_table, raw_table = _build_transition_tables(
        training_sequence,
        retention_lambda=retention_lambda,
    )

    markov_probability, backoff = _support_aware_markov(
        sequence,
        weighted_table,
        raw_table,
        prior_strength=prior_strength,
    )
    probabilities, road_selection = _apply_nested_road(
        sequence,
        markov_probability,
        weighted_table,
        raw_table,
        prior_strength=prior_strength,
    )

    entropy_bits = _entropy(probabilities)
    entropy_weight = _clip(1.0 - entropy_bits / MAX_ENTROPY)
    agreement = float(backoff["multi_order_agreement"])
    selected_order = int(backoff["selected_order"])
    selected_raw_support = 0.0
    selected_item = dict(backoff["contexts"].get(f"order_{selected_order}") or {})
    if selected_item:
        selected_raw_support = float(selected_item.get("raw_support", 0.0) or 0.0)
    support_strength = _reliability(selected_raw_support, SUPPORT_THRESHOLD)
    regime_stability = float(regime_profile["stability"])
    backoff_penalty = float(backoff["backoff_penalty"])

    evidence_quality = _clip(
        0.35
        + 0.20 * agreement
        + 0.20 * support_strength
        + 0.15 * regime_stability
        + 0.10 * backoff_penalty
    )
    base_weight = _clip(entropy_weight * evidence_quality)

    # Maturity only calibrates confidence; it never creates deterministic certainty.
    maturity = max(0.15, float(shoe.shoe_progress))
    final_weight = _clip(base_weight * maturity)

    direction = "B" if probabilities["B"] >= probabilities["P"] else "P"
    selected_counts = dict(backoff["selected_counts"])
    effective_support = _support(selected_counts)

    hierarchy = {
        "mode": "support_aware_backoff_plus_nested_road",
        "markov_backoff": backoff,
        "road_selection": road_selection,
        "dominant_context": str(backoff["selected_context"]),
        "dominant_counts": selected_counts,
        "multi_order_agreement": float(agreement),
        "support_strength": float(support_strength),
        "max_context_support": float(
            max(
                [
                    float(item.get("raw_support", 0.0) or 0.0)
                    for item in backoff["contexts"].values()
                ]
                or [0.0]
            )
        ),
    }

    return {
        "model_version": MODEL_VERSION,
        "engine": "THREEWAY_VARIABLE_ORDER_SUPPORT_BACKOFF_MARKOV",
        "history": sequence,
        "sample_count": len(sequence),
        "state": current_state,
        "state_key": current_state["key"],
        "decay": float(retention_lambda),
        "base_decay": float(decay),
        "adaptive_decay": float(retention_lambda),
        "retention_lambda": float(retention_lambda),
        "decay_intensity": float(decay_intensity),
        "focus_applied": bool(focus_applied),
        "focus_mode": focus_mode,
        "focus_window": int(RECENT_FOCUS_WINDOW if focus_applied else 0),
        "focus_history_preserved": True,
        "training_sample_count": int(len(training_sequence)),
        "regime": str(regime_profile["regime"]),
        "regime_profile": dict(regime_profile),
        "prior": dict(PHYSICAL_PRIOR),
        "prior_strength": float(prior_strength),
        "max_order": int(MAX_ORDER),
        "support_threshold": int(SUPPORT_THRESHOLD),
        "backoff_alpha": float(BACKOFF_ALPHA),
        "selected_order": int(selected_order),
        "backoff_steps": int(backoff["backoff_steps"]),
        "backoff_penalty": float(backoff_penalty),
        "transition_counts": {
            outcome: float(selected_counts.get(outcome, 0.0) or 0.0)
            for outcome in OUTCOMES
        },
        "effective_support": float(effective_support),
        "state_count": int(len(weighted_table)),
        "probabilities": {
            outcome: float(probabilities[outcome]) for outcome in OUTCOMES
        },
        "markov_probabilities_before_road": {
            outcome: float(markov_probability[outcome]) for outcome in OUTCOMES
        },
        "direction": direction,
        "entropy_bits": float(entropy_bits),
        "max_entropy_bits": float(MAX_ENTROPY),
        "entropy_weight": float(entropy_weight),
        "base_weight": float(base_weight),
        "shoe_progress": float(shoe.shoe_progress),
        "final_weight": float(final_weight),
        "shoe_depth": shoe.as_dict(),
        "tie_risk_active": bool(probabilities["T"] > 0.15),
        "tie_risk_threshold": 0.15,
        "hierarchical_backoff": hierarchy,
        "multi_order_agreement": float(agreement),
        "support_strength": float(support_strength),
        "dominant_context": str(backoff["selected_context"]),
        "confidence_semantics": (
            "entropy_support_backoff_regime_maturity_weight_not_win_probability"
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
    "SUPPORT_THRESHOLD",
    "BACKOFF_ALPHA",
    "ENTROPY_WINDOW",
    "encode_threeway_state",
    "update_and_predict_engine",
    "predict_markov",
]