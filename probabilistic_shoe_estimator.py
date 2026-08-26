"""Outcome-conditioned probabilistic shoe estimator V2.

This module does not claim to reconstruct the exact hidden shoe from B/P/T alone.
It maintains a bounded particle posterior over plausible remaining point-count
compositions and estimates the next-hand distribution by forward simulation.

V2 upgrades the next-hand "shoe tendency" judgement with:
- more particles / likelihood draws,
- per-particle next-hand B/P estimates,
- direction consensus and posterior stability,
- reliability shrinkage driven by ESS + consensus + stability,
- explicit BANKER_LEAN / PLAYER_LEAN / BALANCED diagnostics.

The tendency is a posterior-composition signal, not evidence that road patterns
cause future baccarat outcomes.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence
import hashlib
import math
import random
import statistics

MODEL_VERSION = "PROBABILISTIC-SHOE-PARTICLE-V2-TENDENCY"
OUTCOMES = ("B", "P", "T")
PHYSICAL_PRIOR = {"B": 0.4586, "P": 0.4462, "T": 0.0952}
DECKS = 8

PARTICLE_COUNT = 64
LIKELIHOOD_DRAWS = 6
NEXT_HAND_DRAWS_PER_PARTICLE = 12
MAX_REJECTION_DRAWS = 64
MAX_CONDITIONING_ROUNDS = 96
MAX_FUSION_WEIGHT = 0.30
LIKELIHOOD_PRIOR_STRENGTH = 2.5
NEXT_PRIOR_STRENGTH = 12.0
TENDENCY_MARGIN_THRESHOLD = 0.018


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
    return result


def _fresh_counts(decks: int = DECKS) -> list[int]:
    decks = max(1, min(16, int(decks)))
    # Point-value composition: 0 contains 10/J/Q/K => 16 cards/deck.
    return [16 * decks] + [4 * decks] * 9


def _draw_point(counts: list[int], rng: random.Random) -> int:
    total = sum(counts)
    if total <= 0:
        raise ValueError("shoe has no cards left")
    target = rng.randrange(total)
    cumulative = 0
    for point, count in enumerate(counts):
        cumulative += count
        if target < cumulative:
            counts[point] -= 1
            return point
    raise RuntimeError("weighted draw failed")


def _banker_should_draw(banker_total: int, player_third_card: int | None) -> bool:
    total = int(banker_total)
    if player_third_card is None:
        return total <= 5
    third = int(player_third_card)
    if total <= 2:
        return True
    if total == 3:
        return third != 8
    if total == 4:
        return 2 <= third <= 7
    if total == 5:
        return 4 <= third <= 7
    if total == 6:
        return 6 <= third <= 7
    return False


def _simulate_hand(
    counts: Sequence[int],
    rng: random.Random,
) -> tuple[str, list[int], int] | None:
    if sum(int(x) for x in counts) < 6:
        return None

    remaining = [max(0, int(x)) for x in counts]
    player = [_draw_point(remaining, rng)]
    banker = [_draw_point(remaining, rng)]
    player.append(_draw_point(remaining, rng))
    banker.append(_draw_point(remaining, rng))
    player_total = sum(player) % 10
    banker_total = sum(banker) % 10

    if player_total not in {8, 9} and banker_total not in {8, 9}:
        player_third: int | None = None
        if player_total <= 5:
            player_third = _draw_point(remaining, rng)
            player.append(player_third)
            player_total = sum(player) % 10
        if _banker_should_draw(banker_total, player_third):
            banker.append(_draw_point(remaining, rng))
            banker_total = sum(banker) % 10

    outcome = (
        "P" if player_total > banker_total
        else "B" if banker_total > player_total
        else "T"
    )
    return outcome, remaining, len(player) + len(banker)


def _systematic_resample(
    particles: list[list[int]],
    weights: list[float],
    rng: random.Random,
) -> list[list[int]]:
    if not particles:
        return []
    total = sum(max(0.0, float(w)) for w in weights)
    if total <= 0.0:
        return [list(particles[rng.randrange(len(particles))]) for _ in particles]

    normalized = [max(0.0, float(w)) / total for w in weights]
    cumulative: list[float] = []
    running = 0.0
    for weight in normalized:
        running += weight
        cumulative.append(running)

    count = len(particles)
    start = rng.random() / count
    output: list[list[int]] = []
    index = 0
    for offset in range(count):
        point = start + offset / count
        while index < count - 1 and point > cumulative[index]:
            index += 1
        output.append(list(particles[index]))
    return output


def _entropy(probabilities: Mapping[str, float]) -> float:
    result = 0.0
    for outcome in OUTCOMES:
        p = max(1e-15, min(1.0, float(probabilities[outcome])))
        result -= p * math.log2(p)
    return result


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def estimate_probabilistic_shoe(
    history: Iterable[Any],
    *,
    decks: int = DECKS,
    particle_count: int = PARTICLE_COUNT,
) -> Dict[str, Any]:
    full_sequence = _clean_threeway(history)
    sequence = full_sequence[-MAX_CONDITIONING_ROUNDS:]
    particle_count = max(24, min(160, int(particle_count)))
    decks = max(1, min(16, int(decks)))

    seed_material = (
        f"{MODEL_VERSION}|{decks}|{particle_count}|{''.join(sequence)}"
    ).encode("utf-8")
    seed = int(hashlib.sha256(seed_material).hexdigest()[:16], 16)
    rng = random.Random(seed)

    particles = [_fresh_counts(decks) for _ in range(particle_count)]
    ess_ratios: list[float] = []
    conditioned_rounds = 0
    rejection_fallbacks = 0

    for observed in sequence:
        advanced: list[list[int]] = []
        weights: list[float] = []

        for counts in particles:
            candidate_matches: list[list[int]] = []
            valid_draws = 0

            for _ in range(LIKELIHOOD_DRAWS):
                candidate = _simulate_hand(counts, rng)
                if candidate is None:
                    continue
                valid_draws += 1
                outcome, after_counts, _ = candidate
                if outcome == observed:
                    candidate_matches.append(after_counts)

            if valid_draws <= 0:
                continue

            # Smoothed likelihood:
            # L(y|particle) = (matches + beta*pi_y)/(draws + beta)
            prior_mass = LIKELIHOOD_PRIOR_STRENGTH * PHYSICAL_PRIOR[observed]
            likelihood = (
                len(candidate_matches) + prior_mass
            ) / (valid_draws + LIKELIHOOD_PRIOR_STRENGTH)

            if candidate_matches:
                next_counts = list(
                    candidate_matches[rng.randrange(len(candidate_matches))]
                )
            else:
                next_counts = None
                for _ in range(MAX_REJECTION_DRAWS):
                    candidate = _simulate_hand(counts, rng)
                    if candidate is None:
                        break
                    outcome, after_counts, _ = candidate
                    if outcome == observed:
                        next_counts = list(after_counts)
                        break
                if next_counts is None:
                    rejection_fallbacks += 1
                    candidate = _simulate_hand(counts, rng)
                    if candidate is None:
                        continue
                    _, next_counts, _ = candidate
                    likelihood *= 0.05

            advanced.append(next_counts)
            weights.append(max(1e-12, float(likelihood)))

        if not advanced:
            break

        weight_total = sum(weights)
        normalized = [weight / weight_total for weight in weights]
        ess = 1.0 / sum(weight * weight for weight in normalized)
        ess_ratios.append(min(1.0, ess / len(advanced)))
        particles = _systematic_resample(advanced, weights, rng)
        conditioned_rounds += 1

    aggregate_next = {outcome: 0 for outcome in OUTCOMES}
    total_next_draws = 0
    particle_bp_probs: list[float] = []
    particle_directions: list[str] = []

    for counts in particles:
        local = {outcome: 0 for outcome in OUTCOMES}
        for _ in range(NEXT_HAND_DRAWS_PER_PARTICLE):
            candidate = _simulate_hand(counts, rng)
            if candidate is None:
                continue
            outcome, _, _ = candidate
            aggregate_next[outcome] += 1
            local[outcome] += 1
            total_next_draws += 1

        local_bp = local["B"] + local["P"]
        if local_bp > 0:
            p_b_resolved = local["B"] / local_bp
            particle_bp_probs.append(float(p_b_resolved))
            particle_directions.append("B" if p_b_resolved >= 0.5 else "P")

    if total_next_draws <= 0:
        probabilities = dict(PHYSICAL_PRIOR)
    else:
        # Dirichlet shrinkage around the physical baccarat prior.
        probabilities = {
            outcome: (
                aggregate_next[outcome]
                + NEXT_PRIOR_STRENGTH * PHYSICAL_PRIOR[outcome]
            ) / (total_next_draws + NEXT_PRIOR_STRENGTH)
            for outcome in OUTCOMES
        }
        total = sum(probabilities.values())
        probabilities = {
            outcome: float(probabilities[outcome] / total)
            for outcome in OUTCOMES
        }

    if particles:
        expected_counts = [
            sum(particle[point] for particle in particles) / len(particles)
            for point in range(10)
        ]
        std_counts = [
            statistics.pstdev([particle[point] for particle in particles])
            for point in range(10)
        ]
    else:
        expected_counts = [float(x) for x in _fresh_counts(decks)]
        std_counts = [0.0] * 10

    expected_remaining_cards = float(sum(expected_counts))
    start_cards = float(52 * decks)
    expected_cards_used = max(0.0, start_cards - expected_remaining_cards)
    mean_cards_per_round = (
        expected_cards_used / conditioned_rounds
        if conditioned_rounds > 0 else 0.0
    )
    mean_ess_ratio = (
        sum(ess_ratios) / len(ess_ratios) if ess_ratios else 1.0
    )

    bp_mass = probabilities["B"] + probabilities["P"]
    if bp_mass > 1e-12:
        p_b_resolved = probabilities["B"] / bp_mass
        p_p_resolved = probabilities["P"] / bp_mass
    else:
        p_b_resolved = p_p_resolved = 0.5
    bp_margin = float(p_b_resolved - p_p_resolved)

    if particle_bp_probs:
        bp_std = float(statistics.pstdev(particle_bp_probs))
        stability = _clip(1.0 - bp_std / 0.25)
    else:
        bp_std = 0.25
        stability = 0.0

    if particle_directions:
        b_votes = particle_directions.count("B")
        p_votes = particle_directions.count("P")
        consensus = max(b_votes, p_votes) / len(particle_directions)
    else:
        consensus = 0.5

    if abs(bp_margin) < TENDENCY_MARGIN_THRESHOLD:
        tendency = "BALANCED"
    else:
        tendency = "BANKER_LEAN" if bp_margin > 0 else "PLAYER_LEAN"

    tendency_strength = _clip(
        (abs(bp_margin) / 0.12) * (0.5 + 0.5 * consensus) * stability
    )

    history_factor = min(1.0, conditioned_rounds / 50.0)
    ess_factor = 0.60 + 0.40 * mean_ess_ratio
    consensus_factor = 0.70 + 0.30 * consensus
    stability_factor = 0.60 + 0.40 * stability
    reliability = min(
        MAX_FUSION_WEIGHT,
        MAX_FUSION_WEIGHT
        * history_factor
        * ess_factor
        * consensus_factor
        * stability_factor,
    )
    fusion_weight = float(reliability)

    entropy_bits = _entropy(probabilities)
    direction = "B" if probabilities["B"] >= probabilities["P"] else "P"

    return {
        "model_version": MODEL_VERSION,
        "available": bool(conditioned_rounds > 0 and particles),
        "direction": direction,
        "probabilities": probabilities,
        "bp_conditional_probabilities": {
            "B": float(p_b_resolved),
            "P": float(p_p_resolved),
        },
        "history_count": len(full_sequence),
        "conditioned_rounds": int(conditioned_rounds),
        "history_truncated": len(full_sequence) > len(sequence),
        "particle_count": int(len(particles)),
        "likelihood_draws_per_particle": int(LIKELIHOOD_DRAWS),
        "next_hand_draws_per_particle": int(NEXT_HAND_DRAWS_PER_PARTICLE),
        "expected_remaining_counts": [float(x) for x in expected_counts],
        "remaining_count_std": [float(x) for x in std_counts],
        "expected_remaining_cards": expected_remaining_cards,
        "expected_remaining_decks": expected_remaining_cards / 52.0,
        "expected_cards_used": expected_cards_used,
        "mean_cards_per_conditioned_round": float(mean_cards_per_round),
        "mean_ess_ratio": float(mean_ess_ratio),
        "rejection_fallbacks": int(rejection_fallbacks),
        "entropy_bits": float(entropy_bits),
        "reliability": float(reliability),
        "fusion_weight": fusion_weight,
        "max_fusion_weight": float(MAX_FUSION_WEIGHT),
        "shoe_tendency": {
            "label": tendency,
            "bp_margin": float(bp_margin),
            "margin_threshold": float(TENDENCY_MARGIN_THRESHOLD),
            "direction_consensus": float(consensus),
            "particle_bp_std": float(bp_std),
            "posterior_stability": float(stability),
            "strength": float(tendency_strength),
            "semantics": (
                "posterior_composition_tendency_not_road_pattern_causality"
            ),
        },
        "inference_semantics": (
            "outcome_conditioned_particle_posterior_not_exact_remaining_cards"
        ),
    }


__all__ = [
    "MODEL_VERSION",
    "MAX_FUSION_WEIGHT",
    "estimate_probabilistic_shoe",
]
