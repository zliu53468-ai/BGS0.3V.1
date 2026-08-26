"""Adaptive-context three-way Markov predictor for BGS.

V4 keeps B/P/T fully modeled and upgrades V3 with:
- Bayesian context-tree backoff over orders 1..6,
- support maturity gates before higher-order contexts can participate,
- short/long dual-memory models,
- probabilistic road-regime scoring,
- short-vs-long change-point detection,
- regime/change-aware decay,
- aleatoric/epistemic uncertainty diagnostics.

This remains a history-based stochastic model and does not imply deterministic
baccarat patterns.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping
import math

from shoe_depth_estimator import ShoeDepthEstimator

MODEL_VERSION = "THREEWAY-ADAPTIVE-CONTEXT-MARKOV-V4.1-MATURITY-GATE-CANDIDATE"
OUTCOMES = ("B", "P", "T")
PHYSICAL_PRIOR = {"B": 0.4586, "P": 0.4462, "T": 0.0952}

DECAY = 0.95
BAYES_PRIOR_STRENGTH = 6.0
MAX_ENTROPY = math.log2(3.0)
MAX_ORDER = 6
SHORT_MEMORY_WINDOW = 14
LONG_MEMORY_WINDOW = 48
CHANGE_SHORT_WINDOW = 10
CHANGE_LONG_WINDOW = 36

_ORDER_SUPPORT_SCALE = {1: 2.5, 2: 3.5, 3: 5.0, 4: 7.0, 5: 10.0, 6: 14.0}
_ORDER_ALPHA_CAP = {1: 0.78, 2: 0.70, 3: 0.62, 4: 0.54, 5: 0.44, 6: 0.34}
_ORDER_MIN_SUPPORT = {1: 2.0, 2: 3.0, 3: 4.0, 4: 6.0, 5: 9.0, 6: 12.0}
_ROAD_SUPPORT_SCALE = {"road_coarse": 5.0, "road_full": 8.0}
_ROAD_ALPHA_CAP = {"road_coarse": 0.34, "road_full": 0.26}


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _normalize(values: Mapping[str, float]) -> Dict[str, float]:
    probs = {k: max(1e-12, float(values.get(k, 0.0) or 0.0)) for k in OUTCOMES}
    total = sum(probs.values())
    if total <= 1e-12:
        return dict(PHYSICAL_PRIOR)
    return {k: probs[k] / total for k in OUTCOMES}


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
    return [x for x in sequence if x in {"B", "P"}]


def _bp_runs(sequence: Iterable[str]) -> list[tuple[str, int]]:
    bp = _clean_bp(sequence)
    if not bp:
        return []
    result: list[tuple[str, int]] = []
    side, length = bp[0], 1
    for value in bp[1:]:
        if value == side:
            length += 1
        else:
            result.append((side, length))
            side, length = value, 1
    result.append((side, length))
    return result


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
    changes = sum(1 for a, b in zip(bp, bp[1:]) if a != b)
    return changes / max(1, len(bp) - 1)


def _density_state(sequence: list[str]) -> Dict[str, Any]:
    recent5 = sequence[-5:]
    banker = recent5.count("B")
    player = recent5.count("P")
    delta = banker - player
    density = "High" if delta >= 2 else "Low" if delta <= -2 else "Medium"
    return {
        "recent5": recent5,
        "banker_count_recent5": banker,
        "player_count_recent5": player,
        "density_delta": delta,
        "density": density,
    }


def _regime_scores(sequence: Iterable[str]) -> Dict[str, Any]:
    values = list(sequence)
    runs = _bp_runs(values)
    recent = [n for _, n in runs[-6:]]
    current = runs[-1][1] if runs else 0
    previous = runs[-2][1] if len(runs) >= 2 else 0
    previous2 = runs[-3][1] if len(runs) >= 3 else 0
    alt = _alternation_ratio(values, 10)
    one_ratio = sum(n == 1 for n in recent) / len(recent) if recent else 0.0
    two_ratio = sum(n == 2 for n in recent) / len(recent) if recent else 0.0
    long_ratio = sum(n >= 3 for n in recent) / len(recent) if recent else 0.0

    dragon = _clip(0.10 + 0.18 * max(0, current - 1) + 0.20 * long_ratio + (0.12 if current >= 4 else 0.0))
    chop = _clip(0.10 + 0.52 * alt + 0.30 * one_ratio - 0.20 * long_ratio)
    double = _clip(0.08 + 0.52 * two_ratio + 0.12 * (current in {1, 2}) + 0.08 * (previous == 2))

    break_dragon = len(runs) >= 2 and previous >= 4 and current == 1
    break_chop = (
        alt < 0.72
        and len(recent) >= 4
        and sum(n == 1 for n in recent[-4:-1]) >= 2
        and current >= 2
    )
    rhythm_break = (
        len(runs) >= 3
        and previous == previous2
        and previous in {1, 2}
        and current > previous
    )
    transition = _clip(
        0.06
        + (0.62 if break_dragon else 0.0)
        + (0.42 if break_chop else 0.0)
        + (0.24 if rhythm_break else 0.0)
    )
    structural = max(dragon, chop, double)
    mixed = _clip(0.34 + 0.44 * (1.0 - structural) + 0.16 * (1.0 - abs(alt - 0.5) * 2.0))

    raw = {
        "DRAGON": max(1e-6, dragon),
        "CHOP": max(1e-6, chop),
        "DOUBLE_CHOP": max(1e-6, double),
        "MIXED": max(1e-6, mixed),
        "TRANSITION": max(1e-6, transition),
    }
    total = sum(raw.values())
    probs = {k: v / total for k, v in raw.items()}
    regime = max(probs, key=probs.get)
    stability = _clip(probs[regime] + 0.20 * (1.0 - probs["TRANSITION"]))
    return {
        "regime": regime,
        "base_regime": max(("DRAGON", "CHOP", "DOUBLE_CHOP", "MIXED"), key=lambda k: probs[k]),
        "previous_regime": "",
        "transition": bool(probs["TRANSITION"] >= 0.34 or break_dragon or break_chop),
        "stability": float(stability),
        "probabilities": probs,
        "alternation_ratio": float(alt),
        "recent_run_lengths": recent,
        "current_run_length": int(current),
        "previous_run_length": int(previous),
        "previous2_run_length": int(previous2),
        "break_from_dragon": bool(break_dragon),
        "break_from_chop": bool(break_chop),
        "rhythm_break": bool(rhythm_break),
    }


def _detect_regime(sequence: Iterable[str]) -> Dict[str, Any]:
    values = list(sequence)
    profile = _regime_scores(values)
    if len(values) >= 7:
        previous = _regime_scores(values[:-3])
        profile["previous_regime"] = str(previous["regime"])
        if previous["regime"] != profile["regime"]:
            profile["transition"] = True
            probs = dict(profile["probabilities"])
            probs["TRANSITION"] += 0.12
            total = sum(probs.values())
            probs = {k: v / total for k, v in probs.items()}
            profile["probabilities"] = probs
            profile["regime"] = max(probs, key=probs.get)
    else:
        profile["previous_regime"] = "MIXED"
    return profile


def _adaptive_decay(sequence: Iterable[str], base_decay: float, *, change_score: float = 0.0) -> float:
    profile = _detect_regime(sequence)
    probs = profile["probabilities"]
    base = _clip(base_decay, 0.80, 0.995)
    bonus = 0.030 * probs["DRAGON"] + 0.022 * probs["CHOP"] + 0.020 * probs["DOUBLE_CHOP"]
    penalty = (
        0.12 * probs["TRANSITION"]
        + 0.07 * _clip(change_score)
        + (0.02 if profile["transition"] else 0.0)
        + 0.012 * probs["MIXED"]
    )
    return _clip(base + bonus - penalty, 0.82, 0.988)


def _road_state(sequence: list[str]) -> Dict[str, Any]:
    runs = _bp_runs(sequence)
    regime = _detect_regime(sequence)
    density = _density_state(sequence)
    current_side = runs[-1][0] if runs else ""
    current = runs[-1][1] if runs else 0
    previous = runs[-2][1] if len(runs) >= 2 else 0
    previous2 = runs[-3][1] if len(runs) >= 3 else 0
    current_b = _run_bucket(current) if current else "0"
    previous_b = _run_bucket(previous) if previous else "0"
    previous2_b = _run_bucket(previous2) if previous2 else "0"
    tie = "HasTie" if "T" in sequence[-3:] else "NoTie"
    coarse = f"RC|side={current_side or 'NA'}|cur={current_b}|prev={previous_b}|reg={regime['regime']}|tie={tie}"
    full = f"RF|side={current_side or 'NA'}|cur={current_b}|prev={previous_b}|prev2={previous2_b}|reg={regime['regime']}|density={density['density']}|tie={tie}"
    return {
        "current_side": current_side,
        "current_run_length": int(current),
        "current_run_bucket": current_b,
        "previous_run_length": int(previous),
        "previous_run_bucket": previous_b,
        "previous2_run_length": int(previous2),
        "previous2_run_bucket": previous2_b,
        "coarse_key": coarse,
        "full_key": full,
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
        "tie_trigger": "HasTie" if "T" in recent3 else "NoTie",
        "key": road["full_key"] if sequence else "",
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
        "regime_probabilities": dict(road["regime"]["probabilities"]),
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


def _decay_all(table: Dict[str, Dict[str, float]], decay: float) -> None:
    for counts in table.values():
        for outcome in OUTCOMES:
            counts[outcome] *= decay


def _build_table(sequence: list[str], *, base_decay: float) -> Dict[str, Dict[str, float]]:
    table: Dict[str, Dict[str, float]] = defaultdict(lambda: {"B": 0.0, "P": 0.0, "T": 0.0})
    for index in range(1, len(sequence)):
        prefix = sequence[:index]
        _decay_all(table, _adaptive_decay(prefix, base_decay))
        for key in _context_keys(prefix).values():
            table[key][sequence[index]] += 1.0
    return {k: dict(v) for k, v in table.items()}


def _support(counts: Mapping[str, float]) -> float:
    return sum(float(counts.get(x, 0.0) or 0.0) for x in OUTCOMES)


def _reliability(support: float, scale: float) -> float:
    value = max(0.0, float(support))
    return value / (value + max(1e-9, float(scale)))


def _posterior_with_parent(counts: Mapping[str, float], parent: Mapping[str, float], prior_strength: float) -> Dict[str, float]:
    strength = max(1e-9, float(prior_strength))
    support = _support(counts)
    return _normalize({
        outcome: (
            float(counts.get(outcome, 0.0) or 0.0)
            + strength * float(parent.get(outcome, PHYSICAL_PRIOR[outcome]))
        ) / (support + strength)
        for outcome in OUTCOMES
    })


def _blend(base: Mapping[str, float], overlay: Mapping[str, float], alpha: float) -> Dict[str, float]:
    weight = _clip(alpha)
    return _normalize({x: (1.0 - weight) * float(base[x]) + weight * float(overlay[x]) for x in OUTCOMES})


def _log_pool(long_probs: Mapping[str, float], short_probs: Mapping[str, float], short_weight: float) -> Dict[str, float]:
    weight = _clip(short_weight)
    scores = {
        x: math.exp((1.0 - weight) * math.log(max(1e-12, float(long_probs[x]))) + weight * math.log(max(1e-12, float(short_probs[x]))))
        for x in OUTCOMES
    }
    return _normalize(scores)


def _direction_vote(probabilities: Mapping[str, float]) -> str:
    return "B" if float(probabilities["B"]) >= float(probabilities["P"]) else "P"


def _context_tree(sequence: list[str], table: Mapping[str, Mapping[str, float]], *, prior_strength: float) -> tuple[Dict[str, float], Dict[str, Any]]:
    contexts = _context_keys(sequence)
    probability = dict(PHYSICAL_PRIOR)
    details: Dict[str, Any] = {}
    votes: list[tuple[str, float]] = []
    dominant_name = "physical_prior"
    dominant_score = 0.0
    dominant_counts = {"B": 0.0, "P": 0.0, "T": 0.0}

    for order in range(1, MAX_ORDER + 1):
        name = f"order_{order}"
        key = contexts.get(name)
        if not key:
            continue
        counts = dict(table.get(key, {"B": 0.0, "P": 0.0, "T": 0.0}))
        support = _support(counts)
        minimum_support = float(_ORDER_MIN_SUPPORT[order])
        maturity_gate_open = bool(support >= minimum_support)
        posterior = _posterior_with_parent(
            counts,
            probability,
            prior_strength + 0.8 * (order - 1),
        )
        if maturity_gate_open:
            reliability = _reliability(support, _ORDER_SUPPORT_SCALE[order])
            alpha = _ORDER_ALPHA_CAP[order] * reliability
            probability = _blend(probability, posterior, alpha)
        else:
            reliability = 0.0
            alpha = 0.0
        score = alpha * max(0.25, support)
        if score > dominant_score:
            dominant_name, dominant_score, dominant_counts = name, score, dict(counts)
        if alpha > 0.0:
            votes.append((_direction_vote(posterior), alpha))
        details[name] = {
            "key": key,
            "support": support,
            "minimum_support": minimum_support,
            "support_deficit": max(0.0, minimum_support - support),
            "maturity_gate_open": maturity_gate_open,
            "reliability": reliability,
            "alpha": alpha,
            "probabilities": posterior,
            "counts": counts,
        }

    for name in ("road_coarse", "road_full"):
        key = contexts.get(name)
        if not key:
            continue
        counts = dict(table.get(key, {"B": 0.0, "P": 0.0, "T": 0.0}))
        support = _support(counts)
        reliability = _reliability(support, _ROAD_SUPPORT_SCALE[name])
        posterior = _posterior_with_parent(counts, probability, prior_strength + 2.0)
        alpha = _ROAD_ALPHA_CAP[name] * reliability
        probability = _blend(probability, posterior, alpha)
        score = alpha * max(0.25, support)
        if score > dominant_score:
            dominant_name, dominant_score, dominant_counts = name, score, dict(counts)
        if alpha > 0.0:
            votes.append((_direction_vote(posterior), alpha))
        details[name] = {"key": key, "support": support, "reliability": reliability, "alpha": alpha, "probabilities": posterior, "counts": counts}

    vote_total = sum(weight for _, weight in votes)
    if vote_total <= 1e-12:
        agreement = 0.5
    else:
        b_weight = sum(weight for vote, weight in votes if vote == "B")
        p_weight = sum(weight for vote, weight in votes if vote == "P")
        agreement = max(b_weight, p_weight) / vote_total
    max_support = max((float(x["support"]) for x in details.values()), default=0.0)
    active_orders = [
        order
        for order in range(1, MAX_ORDER + 1)
        if bool(dict(details.get(f"order_{order}") or {}).get("maturity_gate_open"))
    ]
    blocked_orders = [
        order
        for order in range(1, MAX_ORDER + 1)
        if f"order_{order}" in details and order not in active_orders
    ]
    return probability, {
        "contexts": details,
        "dominant_context": dominant_name,
        "dominant_counts": dominant_counts,
        "multi_order_agreement": float(agreement),
        "support_strength": float(_reliability(max_support, 8.0)),
        "max_context_support": float(max_support),
        "maturity_gate_thresholds": dict(_ORDER_MIN_SUPPORT),
        "active_orders": active_orders,
        "blocked_orders": blocked_orders,
    }


def _empirical(sequence: list[str], window: int) -> Dict[str, float]:
    sample = sequence[-max(1, int(window)):]
    if not sample:
        return dict(PHYSICAL_PRIOR)
    strength = 1.5
    return _normalize({x: (sample.count(x) + strength * PHYSICAL_PRIOR[x]) / (len(sample) + strength) for x in OUTCOMES})


def _kl(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return max(0.0, sum(max(1e-12, float(left[x])) * math.log2(max(1e-12, float(left[x])) / max(1e-12, float(right[x]))) for x in OUTCOMES))


def _js(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    midpoint = {x: 0.5 * (float(left[x]) + float(right[x])) for x in OUTCOMES}
    return _clip(0.5 * _kl(left, midpoint) + 0.5 * _kl(right, midpoint))


def _change_point(sequence: list[str], short_probs: Mapping[str, float], long_probs: Mapping[str, float]) -> Dict[str, Any]:
    predictive_js = _js(short_probs, long_probs)
    empirical_js = _js(_empirical(sequence, CHANGE_SHORT_WINDOW), _empirical(sequence, CHANGE_LONG_WINDOW))
    regime = _detect_regime(sequence)
    transition_probability = float(regime["probabilities"].get("TRANSITION", 0.0))
    run_break = bool(regime.get("break_from_dragon") or regime.get("break_from_chop") or regime.get("rhythm_break"))
    score = _clip(0.48 * predictive_js + 0.30 * empirical_js + 0.16 * transition_probability + 0.06 * float(run_break))
    return {
        "score": float(score),
        "active": bool(score >= 0.22),
        "predictive_js_bits": float(predictive_js),
        "empirical_js_bits": float(empirical_js),
        "transition_probability": transition_probability,
        "run_break": run_break,
    }


def _entropy(probabilities: Mapping[str, float]) -> float:
    return -sum(max(1e-15, float(probabilities[x])) * math.log2(max(1e-15, float(probabilities[x]))) for x in OUTCOMES)


def update_and_predict_engine(history: Iterable[Any], *, decay: float = DECAY, prior_strength: float = BAYES_PRIOR_STRENGTH) -> Dict[str, Any]:
    sequence = _clean_threeway(history)
    base_decay = _clip(float(decay), 0.80, 0.995)
    prior_strength = max(1e-9, float(prior_strength))
    shoe = ShoeDepthEstimator().estimate(sequence)
    state = encode_threeway_state(sequence)
    regime = _detect_regime(sequence)

    short_sequence = sequence[-SHORT_MEMORY_WINDOW:]
    long_sequence = sequence[-LONG_MEMORY_WINDOW:]
    short_base = _clip(base_decay - 0.055, 0.82, 0.94)
    long_base = _clip(base_decay + 0.020, 0.90, 0.985)
    short_table = _build_table(short_sequence, base_decay=short_base)
    long_table = _build_table(long_sequence, base_decay=long_base)
    short_probs, short_tree = _context_tree(short_sequence, short_table, prior_strength=prior_strength)
    long_probs, long_tree = _context_tree(long_sequence, long_table, prior_strength=prior_strength)

    change = _change_point(sequence, short_probs, long_probs)
    change_score = float(change["score"])
    short_weight = _clip(
        0.32
        + 0.90 * change_score
        + 0.38 * float(regime["probabilities"].get("TRANSITION", 0.0)),
        0.28,
        0.82,
    )
    long_weight = 1.0 - short_weight
    probabilities = _log_pool(long_probs, short_probs, short_weight)
    current_decay = _adaptive_decay(sequence, base_decay, change_score=change_score)

    memory_distance = _clip(0.5 * sum(abs(float(short_probs[x]) - float(long_probs[x])) for x in OUTCOMES))
    memory_agreement = 1.0 - memory_distance
    agreement = short_weight * float(short_tree["multi_order_agreement"]) + long_weight * float(long_tree["multi_order_agreement"])
    support_strength = short_weight * float(short_tree["support_strength"]) + long_weight * float(long_tree["support_strength"])

    entropy = _entropy(probabilities)
    aleatoric = _clip(entropy / MAX_ENTROPY)
    epistemic = _clip(1.0 - (0.42 * support_strength + 0.28 * agreement + 0.20 * memory_agreement + 0.10 * float(regime["stability"])))
    entropy_weight = 1.0 - aleatoric
    evidence_quality = _clip(1.0 - 0.52 * epistemic - 0.28 * change_score)
    base_weight = _clip(entropy_weight * evidence_quality)
    final_weight = _clip(base_weight * float(shoe.shoe_progress))
    direction = "B" if probabilities["B"] >= probabilities["P"] else "P"

    dominant_tree = short_tree if short_weight >= long_weight else long_tree
    dominant_counts = dict(dominant_tree["dominant_counts"])
    effective_support = _support(dominant_counts)
    hierarchy = {
        "mode": "dual_memory_bayesian_context_tree_with_maturity_gate",
        "short": short_tree,
        "long": long_tree,
        "short_memory_weight": float(short_weight),
        "long_memory_weight": float(long_weight),
        "multi_order_agreement": float(agreement),
        "support_strength": float(support_strength),
        "memory_agreement": float(memory_agreement),
        "dominant_context": str(dominant_tree["dominant_context"]),
        "dominant_counts": dominant_counts,
        "maturity_gate_thresholds": dict(_ORDER_MIN_SUPPORT),
    }

    return {
        "model_version": MODEL_VERSION,
        "engine": "THREEWAY_ADAPTIVE_CONTEXT_MARKOV",
        "history": sequence,
        "sample_count": len(sequence),
        "state": state,
        "state_key": state["key"],
        "decay": float(current_decay),
        "base_decay": float(base_decay),
        "adaptive_decay": float(current_decay),
        "short_memory_decay": float(_adaptive_decay(short_sequence, short_base, change_score=change_score)),
        "long_memory_decay": float(_adaptive_decay(long_sequence, long_base, change_score=change_score * 0.5)),
        "regime": str(regime["regime"]),
        "regime_profile": dict(regime),
        "regime_probabilities": dict(regime["probabilities"]),
        "change_point": change,
        "change_point_score": change_score,
        "prior": dict(PHYSICAL_PRIOR),
        "prior_strength": prior_strength,
        "max_order": MAX_ORDER,
        "order_min_support": dict(_ORDER_MIN_SUPPORT),
        "short_memory_window": SHORT_MEMORY_WINDOW,
        "long_memory_window": LONG_MEMORY_WINDOW,
        "short_memory_weight": float(short_weight),
        "long_memory_weight": float(long_weight),
        "short_memory_probabilities": dict(short_probs),
        "long_memory_probabilities": dict(long_probs),
        "transition_counts": {x: float(dominant_counts.get(x, 0.0) or 0.0) for x in OUTCOMES},
        "effective_support": float(effective_support),
        "state_count": len(set(short_table) | set(long_table)),
        "probabilities": {x: float(probabilities[x]) for x in OUTCOMES},
        "direction": direction,
        "entropy_bits": float(entropy),
        "max_entropy_bits": float(MAX_ENTROPY),
        "entropy_weight": float(entropy_weight),
        "aleatoric_uncertainty": float(aleatoric),
        "epistemic_uncertainty": float(epistemic),
        "memory_agreement": float(memory_agreement),
        "base_weight": float(base_weight),
        "shoe_progress": float(shoe.shoe_progress),
        "final_weight": float(final_weight),
        "shoe_depth": shoe.as_dict(),
        "tie_risk_active": bool(probabilities["T"] > 0.15),
        "tie_risk_threshold": 0.15,
        "hierarchical_backoff": hierarchy,
        "context_tree": hierarchy,
        "multi_order_agreement": float(agreement),
        "support_strength": float(support_strength),
        "dominant_context": str(dominant_tree["dominant_context"]),
        "confidence_semantics": "entropy_epistemic_change_memory_maturity_gate_weight_not_guaranteed_win_probability",
    }


def predict_markov(history: Iterable[Any], *, road_context: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    del road_context
    return update_and_predict_engine(history)


__all__ = [
    "MODEL_VERSION",
    "OUTCOMES",
    "PHYSICAL_PRIOR",
    "DECAY",
    "BAYES_PRIOR_STRENGTH",
    "MAX_ORDER",
    "SHORT_MEMORY_WINDOW",
    "LONG_MEMORY_WINDOW",
    "encode_threeway_state",
    "update_and_predict_engine",
    "predict_markov",
]
