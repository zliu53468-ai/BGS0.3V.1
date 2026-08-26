"""Outcome-conditioned probabilistic shoe estimator.

This module does NOT claim to reconstruct the real shoe from B/P/T alone. It keeps
a bounded particle population of plausible remaining point-count compositions,
conditions those particles on the observed B/P/T sequence, and returns a weak
secondary next-hand probability distribution for fusion with the primary Markov
model.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence
import hashlib
import math
import random
import statistics

MODEL_VERSION = "PROBABILISTIC-SHOE-PARTICLE-V1"
OUTCOMES = ("B", "P", "T")
PHYSICAL_PRIOR = {"B": 0.4586, "P": 0.4462, "T": 0.0952}
DECKS = 8

PARTICLE_COUNT = 48
LIKELIHOOD_DRAWS = 4
NEXT_HAND_DRAWS_PER_PARTICLE = 8
MAX_REJECTION_DRAWS = 48
MAX_CONDITIONING_ROUNDS = 80
MAX_FUSION_WEIGHT = 0.25
LIKELIHOOD_PRIOR_STRENGTH = 2.0


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
        probability = max(1e-15, min(1.0, float(probabilities[outcome])))
        result -= probability * math.log2(probability)
    return result


def estimate_probabilistic_shoe(
    history: Iterable[Any],
    *,
    decks: int = DECKS,
    particle_count: int = PARTICLE_COUNT,
) -> Dict[str, Any]:
    full_sequence = _clean_threeway(history)
    sequence = full_sequence[:MAX_CONDITIONING_ROUNDS]
    particle_count = max(16, min(128, int(particle_count)))
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

    next_counts = {outcome: 0 for outcome in OUTCOMES}
    total_next_draws = 0
    for counts in particles:
        for _ in range(NEXT_HAND_DRAWS_PER_PARTICLE):
            candidate = _simulate_hand(counts, rng)
            if candidate is None:
                continue
            outcome, _, _ = candidate
            next_counts[outcome] += 1
            total_next_draws += 1

    if total_next_draws <= 0:
        probabilities = dict(PHYSICAL_PRIOR)
    else:
        smoothing = 3.0
        probabilities = {
            outcome: (
                next_counts[outcome] + smoothing * PHYSICAL_PRIOR[outcome]
            ) / (total_next_draws + smoothing)
            for outcome in OUTCOMES
        }

    probability_total = sum(probabilities.values())
    probabilities = {
        outcome: float(probabilities[outcome] / probability_total)
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

    # B/P/T-only history is weak information about exact card composition.
    # Reliability is deliberately capped so the Markov model remains primary.
    history_factor = min(1.0, conditioned_rounds / 50.0)
    stability_factor = 0.65 + 0.35 * mean_ess_ratio
    reliability = min(
        MAX_FUSION_WEIGHT,
        MAX_FUSION_WEIGHT * history_factor * stability_factor,
    )
    fusion_weight = float(reliability)

    entropy_bits = _entropy(probabilities)
    direction = "B" if probabilities["B"] >= probabilities["P"] else "P"

    return {
        "model_version": MODEL_VERSION,
        "available": bool(conditioned_rounds > 0 and particles),
        "direction": direction,
        "probabilities": probabilities,
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
        "inference_semantics": (
            "outcome_conditioned_particle_posterior_not_exact_remaining_cards"
        ),
    }


__all__ = [
    "MODEL_VERSION",
    "MAX_FUSION_WEIGHT",
    "estimate_probabilistic_shoe",
]
