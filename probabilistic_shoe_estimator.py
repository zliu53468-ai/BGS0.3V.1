"""Outcome- and depth-conditioned probabilistic baccarat shoe estimator V3.

The model does NOT reconstruct the real hidden shoe from a Big-Road screenshot.
It maintains particles over plausible remaining point-value compositions, conditions
them on the observed B/P/T history, and can additionally apply a *soft* total-card
(depth) likelihood when a screenshot/session supplies a plausible remaining-card
estimate.

The total remaining-card count constrains how many 4/5/6-card hands the simulated
history could have consumed. It does not reveal the identities of the remaining cards.

The same next-hand simulations now retain their 4/5/6-card consumption and remaining-
card totals as diagnostics. These diagnostics describe uncertainty in shoe depth only;
they do not add a separate directional vote and do not alter the existing B/P/T fusion.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence
import hashlib
import math
import random
import statistics

# Keep MODEL_VERSION stable because it is part of the deterministic RNG seed.
MODEL_VERSION = "PROBABILISTIC-SHOE-PARTICLE-V3-DEPTH-CONDITIONED"
DRAW_DIAGNOSTICS_VERSION = "SHOE-DRAW-DIAGNOSTICS-V1"
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

# A baccarat hand consumes 4, 5 or 6 cards. Screenshot/session depth is treated
# as a soft observation because OCR, burn-card handling and manual 5-card decrements
# can make the displayed total approximate rather than exact.
DEPTH_DEFAULT_RELIABILITY = 0.65
DEPTH_MIN_SIGMA_CARDS = 2.5
DEPTH_MAX_SIGMA_CARDS = 10.0
DEPTH_PLAUSIBILITY_MARGIN_CARDS = 12


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
        p = max(1e-15, min(1.0, float(probabilities[outcome])))
        result -= p * math.log2(p)
    return result


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _particle_total_stats(particles: Sequence[Sequence[int]]) -> tuple[float, float]:
    totals = [float(sum(int(x) for x in particle)) for particle in particles]
    if not totals:
        return 0.0, 0.0
    return float(sum(totals) / len(totals)), float(
        statistics.pstdev(totals) if len(totals) > 1 else 0.0
    )


def _quantile(values: Sequence[float], quantile: float) -> float:
    """Linear-interpolated quantile without adding a numpy dependency."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    q = _clip(quantile)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return (
        ordered[lower_index] * (1.0 - fraction)
        + ordered[upper_index] * fraction
    )


def _card_interval(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0, "width_p90_p10": 0.0}
    p10 = _quantile(values, 0.10)
    p50 = _quantile(values, 0.50)
    p90 = _quantile(values, 0.90)
    return {
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
        "width_p90_p10": float(max(0.0, p90 - p10)),
    }


def _shoe_phase(start_cards: int, expected_remaining_cards: float) -> Dict[str, Any]:
    if start_cards <= 0:
        progress = 0.0
    else:
        progress = _clip(
            (float(start_cards) - float(expected_remaining_cards)) / float(start_cards)
        )

    if progress < 0.25:
        label = "EARLY"
    elif progress < 0.55:
        label = "MID"
    elif progress < 0.80:
        label = "LATE"
    else:
        label = "VERY_LATE"

    return {
        "label": label,
        "nominal_progress": float(progress),
        "semantics": "estimated_shoe_depth_phase_not_directional_signal",
    }


def _apply_soft_depth_constraint(
    particles: list[list[int]],
    *,
    target_remaining_cards: int | None,
    conditioned_rounds: int,
    start_cards: int,
    reliability: float,
    rng: random.Random,
) -> tuple[list[list[int]], Dict[str, Any]]:
    """Softly condition particles on a remaining-card total.

    L_depth(particle) = exp(-0.5 * ((R_particle - R_screen) / sigma)^2)

    We reject obviously impossible/default totals before weighting. Each completed
    baccarat hand consumes 4..6 cards; a small margin covers burn-card/display
    approximation without pretending the screenshot reveals exact card identities.
    """
    requested = target_remaining_cards is not None
    base: Dict[str, Any] = {
        "requested": bool(requested),
        "applied": False,
        "target_remaining_cards": (
            int(target_remaining_cards) if target_remaining_cards is not None else None
        ),
        "reliability": float(_clip(reliability)),
        "reason": "not_requested",
    }
    if not particles or target_remaining_cards is None:
        return particles, base

    try:
        target = int(target_remaining_cards)
    except (TypeError, ValueError):
        base["reason"] = "invalid_target"
        return particles, base

    if target <= 0 or target > int(start_cards):
        base["reason"] = "target_out_of_shoe_range"
        return particles, base

    rounds = max(0, int(conditioned_rounds))
    rel = _clip(reliability)
    if rel <= 0.0:
        base["reason"] = "zero_reliability"
        return particles, base

    if rounds <= 0:
        if target != int(start_cards):
            base["reason"] = "no_rounds_but_depth_not_full_shoe"
            return particles, base
        base.update({
            "reason": "full_shoe_no_conditioning_needed",
            "plausible_min_remaining": int(start_cards),
            "plausible_max_remaining": int(start_cards),
        })
        return particles, base

    margin = int(DEPTH_PLAUSIBILITY_MARGIN_CARDS)
    plausible_min = max(0, int(start_cards) - 6 * rounds - margin)
    plausible_max = min(
        int(start_cards),
        int(start_cards) - 4 * rounds + max(4, margin // 3),
    )
    base.update({
        "plausible_min_remaining": int(plausible_min),
        "plausible_max_remaining": int(plausible_max),
    })
    if target < plausible_min or target > plausible_max:
        base["reason"] = "inconsistent_with_observed_round_count"
        return particles, base

    before_mean, before_std = _particle_total_stats(particles)
    sigma = DEPTH_MAX_SIGMA_CARDS - (
        DEPTH_MAX_SIGMA_CARDS - DEPTH_MIN_SIGMA_CARDS
    ) * rel
    sigma = max(DEPTH_MIN_SIGMA_CARDS, float(sigma))

    weights: list[float] = []
    for particle in particles:
        remaining = float(sum(particle))
        z = (remaining - target) / sigma
        gaussian = math.exp(-0.5 * z * z)
        weight = max(1e-12, gaussian ** max(0.05, rel))
        weights.append(weight)

    weight_total = sum(weights)
    if weight_total <= 1e-12:
        base["reason"] = "depth_weights_underflow"
        return particles, base

    normalized = [weight / weight_total for weight in weights]
    ess = 1.0 / sum(weight * weight for weight in normalized)
    ess_ratio = min(1.0, ess / len(particles))
    constrained = _systematic_resample(particles, weights, rng)
    after_mean, after_std = _particle_total_stats(constrained)

    consistency = math.exp(
        -0.5 * ((before_mean - float(target)) / max(sigma, 1e-9)) ** 2
    )
    base.update({
        "applied": True,
        "reason": "soft_gaussian_depth_likelihood",
        "sigma_cards": float(sigma),
        "pre_constraint_mean_remaining": float(before_mean),
        "pre_constraint_std_remaining": float(before_std),
        "post_constraint_mean_remaining": float(after_mean),
        "post_constraint_std_remaining": float(after_std),
        "post_constraint_error_cards": float(after_mean - target),
        "depth_ess_ratio": float(ess_ratio),
        "pre_constraint_consistency": float(_clip(consistency)),
        "semantics": "soft_total_card_depth_condition_not_exact_remaining_composition",
    })
    return constrained, base


def estimate_probabilistic_shoe(
    history: Iterable[Any],
    *,
    decks: int = DECKS,
    particle_count: int = PARTICLE_COUNT,
    target_remaining_cards: int | None = None,
    depth_reliability: float = DEPTH_DEFAULT_RELIABILITY,
) -> Dict[str, Any]:
    full_sequence = _clean_threeway(history)
    sequence = full_sequence[-MAX_CONDITIONING_ROUNDS:]
    particle_count = max(24, min(160, int(particle_count)))
    decks = max(1, min(16, int(decks)))
    start_cards = 52 * decks

    # Keep the base outcome-conditioned particle population deterministic for a
    # given history. A rejected depth estimate must not change the RNG path.
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

    particles, depth_constraint = _apply_soft_depth_constraint(
        particles,
        target_remaining_cards=target_remaining_cards,
        conditioned_rounds=conditioned_rounds,
        start_cards=start_cards,
        reliability=depth_reliability,
        rng=rng,
    )

    aggregate_next = {outcome: 0 for outcome in OUTCOMES}
    total_next_draws = 0
    particle_bp_probs: list[float] = []
    particle_directions: list[str] = []
    next_hand_card_counts = {4: 0, 5: 0, 6: 0}
    next_remaining_totals: list[float] = []

    for counts in particles:
        local = {outcome: 0 for outcome in OUTCOMES}
        for _ in range(NEXT_HAND_DRAWS_PER_PARTICLE):
            candidate = _simulate_hand(counts, rng)
            if candidate is None:
                continue
            outcome, after_counts, cards_used = candidate
            aggregate_next[outcome] += 1
            local[outcome] += 1
            total_next_draws += 1
            if cards_used in next_hand_card_counts:
                next_hand_card_counts[cards_used] += 1
            next_remaining_totals.append(float(sum(after_counts)))

        local_bp = local["B"] + local["P"]
        if local_bp > 0:
            p_b_local = local["B"] / local_bp
            particle_bp_probs.append(float(p_b_local))
            particle_directions.append("B" if p_b_local >= 0.5 else "P")

    if total_next_draws <= 0:
        probabilities = dict(PHYSICAL_PRIOR)
    else:
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
        remaining_totals = [
            float(sum(int(x) for x in particle)) for particle in particles
        ]
    else:
        expected_counts = [float(x) for x in _fresh_counts(decks)]
        std_counts = [0.0] * 10
        remaining_totals = [float(sum(expected_counts))]

    expected_remaining_cards = float(sum(expected_counts))
    expected_cards_used = max(0.0, float(start_cards) - expected_remaining_cards)
    mean_cards_per_round = (
        expected_cards_used / conditioned_rounds
        if conditioned_rounds > 0 else 0.0
    )
    mean_ess_ratio = (
        sum(ess_ratios) / len(ess_ratios) if ess_ratios else 1.0
    )

    remaining_cards_interval = _card_interval(remaining_totals)
    remaining_cards_interval["semantics"] = (
        "posterior_particle_interval_not_exact_remaining_card_count"
    )

    total_card_count_samples = sum(next_hand_card_counts.values())
    if total_card_count_samples > 0:
        next_hand_card_probabilities = {
            str(cards): float(count / total_card_count_samples)
            for cards, count in next_hand_card_counts.items()
        }
        expected_next_hand_cards = sum(
            cards * next_hand_card_probabilities[str(cards)]
            for cards in (4, 5, 6)
        )
        most_likely_next_hand_cards = max(
            (4, 5, 6),
            key=lambda cards: (
                next_hand_card_probabilities[str(cards)],
                -cards,
            ),
        )
        card_count_concentration = max(next_hand_card_probabilities.values())
    else:
        next_hand_card_probabilities = {"4": 0.0, "5": 0.0, "6": 0.0}
        expected_next_hand_cards = 0.0
        most_likely_next_hand_cards = None
        card_count_concentration = 0.0

    next_remaining_cards_interval = _card_interval(next_remaining_totals)
    next_remaining_cards_interval["semantics"] = (
        "simulated_after_next_hand_remaining_card_interval_not_exact_count"
    )

    next_hand_draw_profile = {
        "probabilities": next_hand_card_probabilities,
        "expected_cards": float(expected_next_hand_cards),
        "most_likely_cards": most_likely_next_hand_cards,
        "concentration": float(card_count_concentration),
        "samples": int(total_card_count_samples),
        "semantics": (
            "baccarat_rule_based_4_5_6_card_consumption_distribution_not_directional_vote"
        ),
    }
    phase = _shoe_phase(start_cards, expected_remaining_cards)

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

    if depth_constraint.get("applied"):
        depth_consistency = float(
            depth_constraint.get("pre_constraint_consistency", 1.0) or 0.0
        )
        depth_factor = 0.75 + 0.25 * _clip(depth_consistency)
    else:
        depth_factor = 1.0

    reliability = min(
        MAX_FUSION_WEIGHT,
        MAX_FUSION_WEIGHT
        * history_factor
        * ess_factor
        * consensus_factor
        * stability_factor
        * depth_factor,
    )
    fusion_weight = float(reliability)

    entropy_bits = _entropy(probabilities)
    direction = "B" if probabilities["B"] >= probabilities["P"] else "P"
    inference_semantics = (
        "outcome_and_soft_depth_conditioned_particle_posterior_not_exact_remaining_cards"
        if depth_constraint.get("applied")
        else "outcome_conditioned_particle_posterior_not_exact_remaining_cards"
    )

    return {
        "model_version": MODEL_VERSION,
        "draw_diagnostics_version": DRAW_DIAGNOSTICS_VERSION,
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
        "remaining_cards_interval": remaining_cards_interval,
        "expected_remaining_decks": expected_remaining_cards / 52.0,
        "expected_cards_used": expected_cards_used,
        "mean_cards_per_conditioned_round": float(mean_cards_per_round),
        "next_hand_draw_profile": next_hand_draw_profile,
        "next_remaining_cards_interval": next_remaining_cards_interval,
        "shoe_phase": phase,
        "mean_ess_ratio": float(mean_ess_ratio),
        "rejection_fallbacks": int(rejection_fallbacks),
        "entropy_bits": float(entropy_bits),
        "reliability": float(reliability),
        "fusion_weight": fusion_weight,
        "max_fusion_weight": float(MAX_FUSION_WEIGHT),
        "target_remaining_cards": (
            int(target_remaining_cards) if target_remaining_cards is not None else None
        ),
        "depth_constraint_applied": bool(depth_constraint.get("applied", False)),
        "depth_constraint": depth_constraint,
        "shoe_tendency": {
            "label": tendency,
            "bp_margin": float(bp_margin),
            "margin_threshold": float(TENDENCY_MARGIN_THRESHOLD),
            "direction_consensus": float(consensus),
            "particle_bp_std": float(bp_std),
            "posterior_stability": float(stability),
            "strength": float(tendency_strength),
            "semantics": "posterior_composition_tendency_not_road_pattern_causality",
        },
        "inference_semantics": inference_semantics,
    }


__all__ = [
    "MODEL_VERSION",
    "DRAW_DIAGNOSTICS_VERSION",
    "MAX_FUSION_WEIGHT",
    "DEPTH_DEFAULT_RELIABILITY",
    "estimate_probabilistic_shoe",
]
