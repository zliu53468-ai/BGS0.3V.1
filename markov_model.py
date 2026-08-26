"""Three-way high-dimensional Markov predictor for BGS.

B, P and T are all retained in the state and transition model. The active state is:
- Direction_Context: last two outcomes (9 possible B/P/T pairs)
- Density: B-vs-P density over the latest five outcomes
- Tie_Trigger: whether T occurred in the latest three outcomes

Historical state/outcome counts are rebuilt prequentially with exponential decay.
The output is Bayesian-smoothed with the standard baccarat B/P/T baseline prior.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping
import math

from shoe_depth_estimator import ShoeDepthEstimator

MODEL_VERSION = "THREEWAY-HDMARKOV-SHOE-DEPTH-V2"
OUTCOMES = ("B", "P", "T")
PHYSICAL_PRIOR = {"B": 0.4586, "P": 0.4462, "T": 0.0952}
DECAY = 0.95
BAYES_PRIOR_STRENGTH = 6.0
MAX_ENTROPY = math.log2(3.0)


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


def encode_threeway_state(history: Iterable[Any]) -> Dict[str, Any]:
    sequence = _clean_threeway(history)
    if len(sequence) < 2:
        return {
            "ready": False,
            "direction_context": "",
            "density": "Medium",
            "tie_trigger": "HasTie" if "T" in sequence[-3:] else "NoTie",
            "key": "",
            "recent5": sequence[-5:],
            "recent3": sequence[-3:],
        }

    direction_context = "".join(sequence[-2:])
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

    recent3 = sequence[-3:]
    tie_trigger = "HasTie" if "T" in recent3 else "NoTie"
    key = f"{direction_context}|{density}|{tie_trigger}"
    return {
        "ready": True,
        "direction_context": direction_context,
        "density": density,
        "tie_trigger": tie_trigger,
        "key": key,
        "recent5": recent5,
        "recent3": recent3,
        "banker_count_recent5": banker_count,
        "player_count_recent5": player_count,
        "density_delta": density_delta,
    }


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
    decay: float,
) -> Dict[str, Dict[str, float]]:
    transitions: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"B": 0.0, "P": 0.0, "T": 0.0}
    )
    # Leakage-safe: the state for outcome i is encoded only from outcomes < i.
    for index in range(2, len(sequence)):
        _decay_all(transitions, decay)
        state = encode_threeway_state(sequence[:index])
        if not state["ready"]:
            continue
        transitions[state["key"]][sequence[index]] += 1.0
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
    decay = max(0.0, min(1.0, float(decay)))
    prior_strength = max(1e-9, float(prior_strength))
    shoe = ShoeDepthEstimator().estimate(sequence)
    current_state = encode_threeway_state(sequence)
    transition_table = _build_decayed_transition_table(sequence, decay=decay)

    if current_state["ready"]:
        counts = transition_table.get(
            current_state["key"], {"B": 0.0, "P": 0.0, "T": 0.0}
        )
    else:
        counts = {"B": 0.0, "P": 0.0, "T": 0.0}

    probabilities = _posterior_probabilities(
        counts,
        prior_strength=prior_strength,
    )
    entropy = _entropy(probabilities)
    base_weight = max(0.0, min(1.0, 1.0 - entropy / MAX_ENTROPY))
    final_weight = max(
        0.0,
        min(1.0, base_weight * float(shoe.shoe_progress)),
    )

    # T remains a fully modeled outcome, but formal wager direction is B or P only.
    direction = "B" if probabilities["B"] >= probabilities["P"] else "P"
    effective_support = sum(float(counts.get(x, 0.0) or 0.0) for x in OUTCOMES)
    tie_risk_active = probabilities["T"] > 0.15

    return {
        "model_version": MODEL_VERSION,
        "engine": "THREEWAY_HIGH_DIMENSIONAL_MARKOV",
        "history": sequence,
        "sample_count": len(sequence),
        "state": current_state,
        "state_key": current_state["key"],
        "decay": decay,
        "prior": dict(PHYSICAL_PRIOR),
        "prior_strength": prior_strength,
        "transition_counts": {
            outcome: float(counts.get(outcome, 0.0) or 0.0)
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
        "base_weight": float(base_weight),
        "shoe_progress": float(shoe.shoe_progress),
        "final_weight": float(final_weight),
        "shoe_depth": shoe.as_dict(),
        "tie_risk_active": bool(tie_risk_active),
        "tie_risk_threshold": 0.15,
        "confidence_semantics": "entropy_based_signal_weight_not_guaranteed_win_probability",
    }


# Backward-compatible public alias.
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
    "encode_threeway_state",
    "update_and_predict_engine",
    "predict_markov",
]
