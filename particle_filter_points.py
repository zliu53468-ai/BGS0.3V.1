"""Finite-population baccarat engine with an exact hypergeometric core.

The dealer owns a hidden shuffled virtual shoe.  The predictor receives only:
- the remaining card-value counts (0-9), and
- past virtual outcomes / draw paths.

The main probability layer exactly enumerates next-hand outcomes under sampling
without replacement.  That is the multivariate-hypergeometric structure of a
finite baccarat shoe.  Monte Carlo replicas and a low-weight hidden-order
particle ensemble are retained as validation / uncertainty layers.

This is a simulation engine.  It does not read or predict an external live table.
V10 combines the road group, finite-shoe group, and sequence group into one weighted ensemble;
all active models participate in the displayed next-round direction.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import math
import os
import random
import secrets

import numpy as np


DEFAULT_BASELINE = np.asarray([0.458597, 0.446247, 0.095156], dtype=np.float64)
OUTCOME_NAMES = ("B", "P", "T")
PATH_NAMES = ("none", "player_only", "banker_only", "both")
PATH_SUFFIXES = ("N", "P", "B", "D")


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)).strip())
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)).strip())
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


DECKS = _env_int("PF_DECKS", 8, 1, 16)
PARTICLE_COUNT = _env_int("PF_PARTICLES", 500, 64, 4000)
REPLICA_COUNT = _env_int("PF_REPLICAS", 5, 3, 11)
SIMULATIONS_PER_REPLICA = _env_int(
    "PF_PREDICT_SIMULATIONS_PER_REPLICA", 1200, 200, 20_000
)
PARTICLE_DRAWS_PER_PARTICLE = _env_int(
    "PF_DRAWS_PER_PARTICLE", 2, 1, 12
)
HISTORY_WINDOW = _env_int("PF_HISTORY_WINDOW", 60, 8, 300)

# New V7 fusion settings.  New HG_* names intentionally avoid stale V5/V6
# environment values from silently overpowering the exact finite-population core.
HYPERGEOMETRIC_WEIGHT = _env_float("HG_CORE_WEIGHT", 0.82, 0.55, 1.0)
MONTE_CARLO_WEIGHT = _env_float("HG_MC_WEIGHT", 0.10, 0.0, 0.30)
PARTICLE_WEIGHT = _env_float("HG_PARTICLE_WEIGHT", 0.05, 0.0, 0.20)
HISTORY_MAX_WEIGHT = _env_float("HG_HISTORY_MAX_WEIGHT", 0.03, 0.0, 0.08)
BASELINE_SHRINK = _env_float("HG_BASELINE_SHRINK", 0.02, 0.0, 0.15)
MIN_DIRECTION_EDGE = _env_float("HG_MIN_DIRECTION_EDGE", 0.016, 0.0, 0.10)
MAX_SIGNAL_UNCERTAINTY = _env_float(
    "HG_MAX_UNCERTAINTY", 0.012, 0.001, 0.10
)
MAX_VALIDATION_GAP = _env_float(
    "HG_MAX_VALIDATION_GAP", 0.025, 0.001, 0.15
)

# 牌路先行 context 由 road_model.py 建立，再交給本引擎統一判斷。
ROAD_CONTEXT_MAX_WEIGHT = _env_float("ROAD_FUSION_WEIGHT", 0.08, 0.0, 0.20)
ROAD_CONTEXT_MIN_SAMPLES = _env_int("ROAD_FUSION_MIN_SAMPLES", 10, 4, 100)
ROAD_CONTEXT_MIN_CONFIDENCE = _env_float(
    "ROAD_FUSION_MIN_CONFIDENCE", 0.45, 0.0, 1.0
)
ROAD_CONTEXT_MAX_UNCERTAINTY = _env_float(
    "ROAD_FUSION_MAX_UNCERTAINTY", 0.16, 0.01, 0.50
)


# V10 全模型群組集成。有限牌組三模型彼此高度相關，先在群組內融合，
# 再與牌路群組、額外序列群組共同參與最終方向，避免重複計票。
ENSEMBLE_ROAD_GROUP_WEIGHT = _env_float("ENSEMBLE_ROAD_GROUP_WEIGHT", 0.50, 0.0, 0.80)
ENSEMBLE_FINITE_GROUP_WEIGHT = _env_float("ENSEMBLE_FINITE_GROUP_WEIGHT", 0.40, 0.10, 0.90)
ENSEMBLE_SEQUENCE_GROUP_WEIGHT = _env_float("ENSEMBLE_SEQUENCE_GROUP_WEIGHT", 0.10, 0.0, 0.30)
ENSEMBLE_MIN_MODEL_AGREEMENT = _env_float("ENSEMBLE_MIN_MODEL_AGREEMENT", 0.55, 0.50, 1.0)

BANKER_COMMISSION = _env_float("PF_BANKER_COMMISSION", 0.05, 0.0, 0.20)


DB_HOLDOUT: Dict[str, Any] = {
    "passed": False,
    "point_map_rate": 0.4999,
    "baseline_rate": 0.5068,
    "samples": 500_000,
    "note": (
        "V7 uses an exact finite-population hypergeometric core for the internal "
        "virtual shoe; no external-table signal is available."
    ),
}


@dataclass(frozen=True)
class HandResult:
    player_cards: Tuple[int, ...]
    banker_cards: Tuple[int, ...]
    player_total: int
    banker_total: int
    outcome: str
    draw_path: str
    cards_used: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "player_cards": list(self.player_cards),
            "banker_cards": list(self.banker_cards),
            "player_total": self.player_total,
            "banker_total": self.banker_total,
            "outcome": self.outcome,
            "outcome_text": {"B": "莊", "P": "閒", "T": "和"}[self.outcome],
            "draw_path": self.draw_path,
            "draw_path_text": {
                "N": "雙方不補牌",
                "P": "僅閒家補牌",
                "B": "僅莊家補牌",
                "D": "雙方補牌",
            }[self.draw_path],
            "cards_used": self.cards_used,
        }


@dataclass(frozen=True)
class EngineSettings:
    decks: int = DECKS
    particles: int = PARTICLE_COUNT
    replicas: int = REPLICA_COUNT
    simulations_per_replica: int = SIMULATIONS_PER_REPLICA
    particle_draws_per_particle: int = PARTICLE_DRAWS_PER_PARTICLE


def fresh_counts(decks: int = DECKS) -> List[int]:
    """Return baccarat point-value counts for a fresh shoe."""
    decks = max(1, min(16, int(decks)))
    return [16 * decks] + [4 * decks] * 9


def create_virtual_shoe(
    decks: int = DECKS,
    seed: Optional[int] = None,
) -> List[int]:
    """Create a shuffled virtual shoe; 0 represents 10/J/Q/K."""
    counts = fresh_counts(decks)
    shoe: List[int] = []
    for value, count in enumerate(counts):
        shoe.extend([value] * count)
    rng = random.Random(seed if seed is not None else secrets.randbits(64))
    rng.shuffle(shoe)
    return shoe


def counts_from_shoe(shoe: Sequence[int]) -> List[int]:
    counts = [0] * 10
    for card in shoe:
        value = int(card)
        if value < 0 or value > 9:
            raise ValueError("virtual shoe contains an invalid card value")
        counts[value] += 1
    return counts


def baccarat_total(cards: Iterable[int]) -> int:
    return sum(int(card) for card in cards) % 10


def banker_should_draw(
    banker_total: int,
    player_third_card: Optional[int],
) -> bool:
    """Official punto banco banker third-card rule."""
    banker_total %= 10
    if player_third_card is None:
        return banker_total <= 5
    third = int(player_third_card) % 10
    if banker_total <= 2:
        return True
    if banker_total == 3:
        return third != 8
    if banker_total == 4:
        return 2 <= third <= 7
    if banker_total == 5:
        return 4 <= third <= 7
    if banker_total == 6:
        return third in {6, 7}
    return False


def _play_with_draw(draw_card: Any) -> HandResult:
    player = [int(draw_card()), int(draw_card())]
    banker = [int(draw_card()), int(draw_card())]

    player_total = baccarat_total(player)
    banker_total = baccarat_total(banker)
    player_drew = False
    banker_drew = False

    if player_total not in {8, 9} and banker_total not in {8, 9}:
        player_third: Optional[int] = None
        if player_total <= 5:
            player_third = int(draw_card())
            player.append(player_third)
            player_drew = True
        if banker_should_draw(banker_total, player_third):
            banker.append(int(draw_card()))
            banker_drew = True

    player_total = baccarat_total(player)
    banker_total = baccarat_total(banker)
    outcome = (
        "B"
        if banker_total > player_total
        else "P"
        if player_total > banker_total
        else "T"
    )
    draw_path = (
        "D"
        if player_drew and banker_drew
        else "P"
        if player_drew
        else "B"
        if banker_drew
        else "N"
    )
    return HandResult(
        player_cards=tuple(player),
        banker_cards=tuple(banker),
        player_total=player_total,
        banker_total=banker_total,
        outcome=outcome,
        draw_path=draw_path,
        cards_used=len(player) + len(banker),
    )


def deal_ordered_hand(shoe: Sequence[int]) -> Tuple[HandResult, List[int]]:
    """Deal one hand from the hidden ordered shoe and return the remainder."""
    cards = [int(card) for card in shoe]
    if len(cards) < 6:
        raise ValueError("not enough cards in virtual shoe")
    index = 0

    def draw() -> int:
        nonlocal index
        if index >= len(cards):
            raise ValueError("virtual shoe exhausted during deal")
        card = cards[index]
        index += 1
        return card

    result = _play_with_draw(draw)
    return result, cards[index:]


def _draw_value_from_counts(counts: List[int], rng: np.random.Generator) -> int:
    """Draw one value exactly according to finite-population counts."""
    total = int(sum(counts))
    if total <= 0:
        raise ValueError("empty card counts")
    target = int(rng.integers(0, total))
    cumulative = 0
    for value, count in enumerate(counts):
        cumulative += int(count)
        if target < cumulative:
            counts[value] -= 1
            return value
    raise RuntimeError("count draw failed")


def simulate_one_from_counts(
    source_counts: Sequence[int],
    rng: np.random.Generator,
) -> HandResult:
    """Sample one hand without replacement from the finite population."""
    local = [max(0, int(value)) for value in source_counts]
    if len(local) != 10 or sum(local) < 6:
        raise ValueError("remaining_counts must contain ten values and at least six cards")

    def draw() -> int:
        return _draw_value_from_counts(local, rng)

    return _play_with_draw(draw)


def _normalize_probability(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array = np.maximum(array, 1e-15)
    total = float(array.sum())
    return array / total if total > 0 else DEFAULT_BASELINE.copy()


@lru_cache(maxsize=512)
def _exact_hypergeometric_cached(
    counts_key: Tuple[int, ...],
) -> Tuple[Tuple[float, ...], Tuple[float, ...], int]:
    """Exact next-hand distribution under sampling without replacement.

    This enumerates all feasible point-value paths.  The product of conditional
    finite-population draw probabilities is algebraically equivalent to the
    multivariate hypergeometric model, while preserving card order so baccarat
    third-card rules can be applied exactly.
    """
    counts = list(counts_key)
    if len(counts) != 10 or any(value < 0 for value in counts):
        raise ValueError("remaining_counts must be ten non-negative integers")
    population = int(sum(counts))
    if population < 6:
        raise ValueError("virtual shoe does not contain enough cards")

    outcomes = [0.0, 0.0, 0.0]
    paths = [0.0, 0.0, 0.0, 0.0]
    terminal_branches = 0

    def add_result(probability: float, player_total: int, banker_total: int, path: int) -> None:
        nonlocal terminal_branches
        index = 0 if banker_total > player_total else 1 if player_total > banker_total else 2
        outcomes[index] += probability
        paths[path] += probability
        terminal_branches += 1

    # Two player cards followed by two banker cards is distributionally
    # equivalent to alternating the initial deal because draws are exchangeable.
    for p1 in range(10):
        if counts[p1] <= 0:
            continue
        q1 = counts[p1] / population
        counts[p1] -= 1
        for p2 in range(10):
            if counts[p2] <= 0:
                continue
            q2 = q1 * counts[p2] / (population - 1)
            counts[p2] -= 1
            player_total = (p1 + p2) % 10
            for b1 in range(10):
                if counts[b1] <= 0:
                    continue
                q3 = q2 * counts[b1] / (population - 2)
                counts[b1] -= 1
                for b2 in range(10):
                    if counts[b2] <= 0:
                        continue
                    q4 = q3 * counts[b2] / (population - 3)
                    counts[b2] -= 1
                    banker_total = (b1 + b2) % 10

                    if player_total in {8, 9} or banker_total in {8, 9}:
                        add_result(q4, player_total, banker_total, 0)
                    elif player_total <= 5:
                        remaining_after_four = population - 4
                        for p3 in range(10):
                            if counts[p3] <= 0:
                                continue
                            q5 = q4 * counts[p3] / remaining_after_four
                            counts[p3] -= 1
                            final_player = (player_total + p3) % 10
                            if banker_should_draw(banker_total, p3):
                                remaining_after_five = population - 5
                                for b3 in range(10):
                                    if counts[b3] <= 0:
                                        continue
                                    q6 = q5 * counts[b3] / remaining_after_five
                                    final_banker = (banker_total + b3) % 10
                                    add_result(q6, final_player, final_banker, 3)
                            else:
                                add_result(q5, final_player, banker_total, 1)
                            counts[p3] += 1
                    elif banker_should_draw(banker_total, None):
                        remaining_after_four = population - 4
                        for b3 in range(10):
                            if counts[b3] <= 0:
                                continue
                            q5 = q4 * counts[b3] / remaining_after_four
                            final_banker = (banker_total + b3) % 10
                            add_result(q5, player_total, final_banker, 2)
                    else:
                        add_result(q4, player_total, banker_total, 0)

                    counts[b2] += 1
                counts[b1] += 1
            counts[p2] += 1
        counts[p1] += 1

    outcome_array = _normalize_probability(outcomes)
    path_array = _normalize_probability(paths)
    return (
        tuple(float(value) for value in outcome_array),
        tuple(float(value) for value in path_array),
        terminal_branches,
    )


def hypergeometric_probabilities(
    remaining_counts: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return exact B/P/T and draw-path probabilities."""
    counts_key = tuple(int(value) for value in remaining_counts)
    outcomes, paths, branches = _exact_hypergeometric_cached(counts_key)
    return (
        np.asarray(outcomes, dtype=np.float64),
        np.asarray(paths, dtype=np.float64),
        {
            "exact": True,
            "population_size": int(sum(counts_key)),
            "terminal_branches": int(branches),
            "cache": _exact_hypergeometric_cached.cache_info()._asdict(),
        },
    )


def _monte_carlo_replica(
    counts: Sequence[int],
    simulations: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    outcomes = np.zeros(3, dtype=np.float64)
    paths = np.zeros(4, dtype=np.float64)
    for _ in range(max(1, int(simulations))):
        result = simulate_one_from_counts(counts, rng)
        outcomes[OUTCOME_NAMES.index(result.outcome)] += 1.0
        paths[PATH_SUFFIXES.index(result.draw_path)] += 1.0
    return outcomes / outcomes.sum(), paths / paths.sum()


def _particle_ensemble(
    counts: Sequence[int],
    particles: int,
    draws_per_particle: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Sample possible hidden orders consistent with the exact remaining counts.

    With no external observations, particles are correctly equal-weighted.
    Therefore this layer measures hidden-order sampling uncertainty; it does not
    claim information beyond the exact hypergeometric population state.
    """
    rng = np.random.default_rng(seed)
    particle_count = max(1, int(particles))
    weights = np.full(particle_count, 1.0 / particle_count, dtype=np.float64)
    outcome_probability = np.zeros(3, dtype=np.float64)
    path_probability = np.zeros(4, dtype=np.float64)

    for particle_index in range(particle_count):
        local_outcomes = np.zeros(3, dtype=np.float64)
        local_paths = np.zeros(4, dtype=np.float64)
        for _ in range(max(1, int(draws_per_particle))):
            result = simulate_one_from_counts(counts, rng)
            local_outcomes[OUTCOME_NAMES.index(result.outcome)] += 1.0
            local_paths[PATH_SUFFIXES.index(result.draw_path)] += 1.0
        local_outcomes /= local_outcomes.sum()
        local_paths /= local_paths.sum()
        outcome_probability += weights[particle_index] * local_outcomes
        path_probability += weights[particle_index] * local_paths

    ess = 1.0 / float(np.square(weights).sum())
    return (
        _normalize_probability(outcome_probability),
        _normalize_probability(path_probability),
        ess,
    )


def _sequence_probabilities(history: Sequence[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Small, strongly shrunk descriptive history component."""
    cleaned = [
        str(item).upper()
        for item in history
        if str(item).upper() in OUTCOME_NAMES
    ][-HISTORY_WINDOW:]
    baseline = DEFAULT_BASELINE.copy()
    if len(cleaned) < 4:
        return baseline, {"order": 0, "support": len(cleaned), "weight": 0.0}

    candidates: List[Tuple[int, np.ndarray, int]] = []
    for order in (3, 2, 1):
        if len(cleaned) <= order:
            continue
        suffix = tuple(cleaned[-order:])
        observed = np.zeros(3, dtype=np.float64)
        support = 0
        for index in range(order, len(cleaned)):
            if tuple(cleaned[index - order:index]) == suffix:
                observed[OUTCOME_NAMES.index(cleaned[index])] += 1.0
                support += 1
        if support > 0:
            prior_strength = 24.0 + 10.0 * order
            posterior = (observed + baseline * prior_strength) / (
                support + prior_strength
            )
            candidates.append((order, posterior, support))

    ew_counts = baseline * 30.0
    decay = 0.94
    for reverse_index, outcome in enumerate(reversed(cleaned)):
        ew_counts[OUTCOME_NAMES.index(outcome)] += decay ** reverse_index
    ew_probability = _normalize_probability(ew_counts)

    if not candidates:
        return ew_probability, {
            "order": 0,
            "support": len(cleaned),
            "weight": 0.01,
        }

    order, posterior, support = max(candidates, key=lambda item: (item[0], item[2]))
    support_weight = min(0.35, support / (support + 24.0))
    probability = _normalize_probability(
        posterior * support_weight + ew_probability * (1.0 - support_weight)
    )
    return probability, {
        "order": order,
        "support": support,
        "weight": round(support_weight, 6),
    }


def _confidence_interval(
    replica_matrix: np.ndarray,
    z_score: float = 1.96,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = replica_matrix.mean(axis=0)
    if len(replica_matrix) <= 1:
        standard_error = np.zeros_like(center)
    else:
        standard_error = replica_matrix.std(axis=0, ddof=1) / math.sqrt(
            len(replica_matrix)
        )
    lower = np.maximum(0.0, center - z_score * standard_error)
    upper = np.minimum(1.0, center + z_score * standard_error)
    return center, lower, upper, standard_error


class VirtualShoeParticleEngine:
    """Exact hypergeometric core with MC, particle and road-context validation."""

    def __init__(self, settings: Optional[EngineSettings] = None) -> None:
        self.settings = settings or EngineSettings()

    @staticmethod
    def _road_context_state(
        road_context: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        road = dict(road_context or {})
        try:
            sample_count = max(0, int(road.get("sample_count", 0) or 0))
        except Exception:
            sample_count = 0
        try:
            confidence = max(0.0, min(1.0, float(road.get("confidence_score", 0.0) or 0.0)))
        except Exception:
            confidence = 0.0
        try:
            uncertainty = max(0.0, float(road.get("uncertainty", 1.0) or 1.0))
        except Exception:
            uncertainty = 1.0
        try:
            banker_probability = max(
                0.0,
                min(1.0, float(road.get("banker_probability", 0.5) or 0.5)),
            )
        except Exception:
            banker_probability = 0.5
        direction = str(
            road.get("direction") or ("B" if banker_probability >= 0.5 else "P")
        ).upper()
        if direction not in {"B", "P"}:
            direction = "B" if banker_probability >= 0.5 else "P"
        try:
            suggested = max(0.0, float(road.get("suggested_core_weight", 0.0) or 0.0))
        except Exception:
            suggested = 0.0

        eligible = bool(
            road.get("ok")
            and bool(road.get("eligible_for_core", True))
            and sample_count >= ROAD_CONTEXT_MIN_SAMPLES
            and confidence >= ROAD_CONTEXT_MIN_CONFIDENCE
            and uncertainty <= ROAD_CONTEXT_MAX_UNCERTAINTY
        )
        effective_weight = min(ROAD_CONTEXT_MAX_WEIGHT, suggested) if eligible else 0.0
        return {
            "present": bool(road),
            "ok": bool(road.get("ok")),
            "eligible": eligible,
            "sample_count": sample_count,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "banker_probability": banker_probability,
            "player_probability": 1.0 - banker_probability,
            "direction": direction,
            "requested_weight": suggested,
            "effective_weight": effective_weight,
            "engine": str(road.get("engine") or ""),
            "signal_allowed": bool(road.get("signal_allowed")),
            "signal_reason": str(road.get("signal_reason") or ""),
        }

    def analyze(
        self,
        remaining_counts: Sequence[int],
        history: Optional[Sequence[str]] = None,
        draw_path_history: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
        road_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        counts = [int(value) for value in remaining_counts]
        if len(counts) != 10 or any(value < 0 for value in counts):
            raise ValueError("remaining_counts must be ten non-negative integers")
        if sum(counts) < 6:
            raise ValueError("virtual shoe does not contain enough cards")

        run_seed = int(seed if seed is not None else secrets.randbits(32)) & 0xFFFFFFFF
        seed_sequence = np.random.SeedSequence(run_seed)
        child_seeds = seed_sequence.spawn(self.settings.replicas + 1)

        hyper_probability, hyper_paths, hyper_meta = hypergeometric_probabilities(counts)

        replica_probabilities: List[np.ndarray] = []
        replica_paths: List[np.ndarray] = []
        for index in range(self.settings.replicas):
            replica_seed = int(child_seeds[index].generate_state(1, dtype=np.uint32)[0])
            probability, paths = _monte_carlo_replica(
                counts,
                self.settings.simulations_per_replica,
                replica_seed,
            )
            replica_probabilities.append(probability)
            replica_paths.append(paths)

        replica_matrix = np.vstack(replica_probabilities)
        path_matrix = np.vstack(replica_paths)
        mc_probability, ci_lower, ci_upper, replica_se = _confidence_interval(replica_matrix)
        mc_path_probability = path_matrix.mean(axis=0)

        particle_seed = int(child_seeds[-1].generate_state(1, dtype=np.uint32)[0])
        particle_probability, particle_paths, particle_ess = _particle_ensemble(
            counts,
            self.settings.particles,
            self.settings.particle_draws_per_particle,
            particle_seed,
        )

        road_state = self._road_context_state(road_context)
        sequence_probability, sequence_meta = _sequence_probabilities(history or [])
        history_length = len([
            item for item in (history or []) if str(item).upper() in OUTCOME_NAMES
        ])

        # V10：所有模型共同參與方向。有限牌組三模型先在群組內融合，
        # 避免超幾何、蒙地卡羅、粒子因使用同一估計牌組而被重複算成三票。
        finite_requested = np.asarray(
            [HYPERGEOMETRIC_WEIGHT, MONTE_CARLO_WEIGHT, PARTICLE_WEIGHT],
            dtype=np.float64,
        )
        finite_requested = finite_requested / finite_requested.sum()
        hyper_weight, mc_weight, particle_weight = (
            float(value) for value in finite_requested
        )
        finite_group = _normalize_probability(
            hyper_probability * hyper_weight
            + mc_probability * mc_weight
            + particle_probability * particle_weight
        )

        uncertainty = float(max(replica_se))
        validation_gap = float(max(
            np.max(np.abs(mc_probability - hyper_probability)),
            np.max(np.abs(particle_probability - hyper_probability)),
        ))
        adaptive_shrink = min(0.15, BASELINE_SHRINK + uncertainty * 2.0)
        finite_group = _normalize_probability(
            finite_group * (1.0 - adaptive_shrink)
            + DEFAULT_BASELINE * adaptive_shrink
        )

        core_banker, core_player, core_tie = (float(value) for value in finite_group)
        core_direction = "B" if core_banker > core_player else "P"

        road_direction = str(road_state.get("direction") or "").upper()
        road_available = bool(
            road_state.get("ok")
            and road_direction in {"B", "P"}
            and history_length >= 4
        )
        sequence_available = history_length >= 4

        # 牌路模型只提供 B/P，因此沿用有限牌組群組估計的和局比例，
        # 再把剩餘 B/P 比例依牌路機率分配。
        road_three = np.asarray(
            [
                (1.0 - core_tie) * float(road_state["banker_probability"]),
                (1.0 - core_tie) * float(road_state["player_probability"]),
                core_tie,
            ],
            dtype=np.float64,
        )
        road_three = _normalize_probability(road_three)

        group_weights = {
            "road": ENSEMBLE_ROAD_GROUP_WEIGHT if road_available else 0.0,
            "finite": ENSEMBLE_FINITE_GROUP_WEIGHT,
            "sequence": ENSEMBLE_SEQUENCE_GROUP_WEIGHT if sequence_available else 0.0,
        }
        group_total = sum(group_weights.values()) or 1.0
        group_weights = {name: value / group_total for name, value in group_weights.items()}

        ensemble_probability = _normalize_probability(
            road_three * group_weights["road"]
            + finite_group * group_weights["finite"]
            + sequence_probability * group_weights["sequence"]
        )
        banker, player, tie = (float(value) for value in ensemble_probability)
        bp_total = max(1e-12, banker + player)
        banker_no_tie = banker / bp_total
        player_no_tie = player / bp_total
        direction = "B" if banker > player else "P"
        direction_edge = abs(banker_no_tie - player_no_tie)

        model_directions = {
            "road": road_direction if road_available else "",
            "hypergeometric": "B" if hyper_probability[0] > hyper_probability[1] else "P",
            "monte_carlo": "B" if mc_probability[0] > mc_probability[1] else "P",
            "particle": "B" if particle_probability[0] > particle_probability[1] else "P",
            "sequence": "B" if sequence_probability[0] > sequence_probability[1] else "P",
        }
        active_directions = [value for value in model_directions.values() if value in {"B", "P"}]
        agreement = (
            active_directions.count(direction) / max(1, len(active_directions))
        )
        road_aligned_with_core = bool(
            not road_available or road_direction == core_direction
        )
        road_aligned_with_final = bool(
            not road_available or road_direction == direction
        )

        banker_ev = banker * (1.0 - BANKER_COMMISSION) - player
        player_ev = player - banker
        tie_ev = tie * 8.0 - (banker + player)
        expected_values = {"B": banker_ev, "P": player_ev, "T": tie_ev}
        best_ev_side = max(expected_values, key=expected_values.get)
        best_ev = float(expected_values[best_ev_side])

        edge_ok = direction_edge >= MIN_DIRECTION_EDGE
        uncertainty_ok = uncertainty <= MAX_SIGNAL_UNCERTAINTY
        validation_ok = validation_gap <= MAX_VALIDATION_GAP
        agreement_ok = agreement >= ENSEMBLE_MIN_MODEL_AGREEMENT
        signal_allowed = bool(
            edge_ok and uncertainty_ok and validation_ok and agreement_ok
        )
        action = direction if signal_allowed else "O"
        road_weight = group_weights["road"]
        sequence_weight = group_weights["sequence"]
        fused_direction = direction
        road_ready = road_available

        uncertainty_score = 1.0 - min(
            1.0, uncertainty / max(MAX_SIGNAL_UNCERTAINTY, 1e-9)
        )
        edge_score = min(1.0, direction_edge / max(MIN_DIRECTION_EDGE, 1e-9))
        validation_score = 1.0 - min(
            1.0, validation_gap / max(MAX_VALIDATION_GAP, 1e-9)
        )
        particle_score = min(1.0, particle_ess / max(1.0, self.settings.particles))
        base_quality = max(
            0.0,
            min(
                1.0,
                0.30 * uncertainty_score
                + 0.25 * edge_score
                + 0.20 * validation_score
                + 0.15 * hyper_weight
                + 0.10 * particle_score,
            ),
        )
        quality_score = base_quality
        if road_weight > 0.0:
            quality_score = (
                base_quality * (1.0 - road_weight)
                + float(road_state["confidence"]) * road_weight
            )
            if not road_aligned_with_core:
                quality_score *= 0.92
        quality_score = max(0.0, min(1.0, quality_score))

        confidence_label = (
            "較高" if quality_score >= 0.72
            else "中等" if quality_score >= 0.50
            else "偏低"
        )
        alignment_score = 1.0 if road_aligned_with_core else 0.0
        model_consistency = validation_score
        if road_weight > 0.0:
            model_consistency = (
                validation_score * (1.0 - road_weight) + alignment_score * road_weight
            )
        model_consistency = max(0.0, min(1.0, model_consistency))

        if signal_allowed:
            signal_reason = "牌路、有限牌組、蒙地卡羅、粒子與序列模型已完成全模型集成，方向與品質門檻通過"
            signal_status_text = "全模型方向訊號已開放"
        else:
            reasons: List[str] = []
            if not edge_ok:
                reasons.append("全模型集成方向差距尚未達門檻")
            if not uncertainty_ok:
                reasons.append("模擬不確定性仍偏高")
            if not validation_ok:
                reasons.append("有限牌組與驗證層的一致度不足")
            if not agreement_ok:
                reasons.append("各模型方向共識不足")
            if not road_available:
                reasons.append("牌路模型樣本不足，已由其餘模型重新分配權重")
            signal_reason = "、".join(reasons) or "目前資料尚未形成正式方向訊號"
            signal_status_text = "等待更明確的全模型共識"

        road_integration = {
            "processed_inside_core": True,
            "present": road_state["present"],
            "applied": road_weight > 0.0,
            "eligible": road_state["eligible"],
            "sample_count": road_state["sample_count"],
            "road_direction": road_state["direction"],
            "core_direction_before_road": core_direction,
            "final_direction": direction,
            "fused_direction_before_override": fused_direction,
            "road_direction_primary": False,
            "all_models_participate": True,
            "model_directions": model_directions,
            "model_agreement": round(agreement, 6),
            "aligned_with_core": road_aligned_with_core,
            "aligned_with_final": road_aligned_with_final,
            "requested_weight": round(float(road_state["requested_weight"]), 8),
            "effective_weight": round(road_weight, 8),
            "road_confidence": round(float(road_state["confidence"]), 6),
            "road_uncertainty": round(float(road_state["uncertainty"]), 6),
            "road_engine": road_state["engine"],
        }

        return {
            "ok": True,
            "engine": "V10_ALL_MODEL_GROUP_ENSEMBLE",
            "model_core": "road_finite_sequence_group_ensemble",
            "pipeline_order": [
                "road_multi_model_group",
                "hypergeometric_monte_carlo_particle_group",
                "sequence_group",
                "all_model_weighted_ensemble",
                "quality_and_observe_decision",
            ],
            "run_seed": run_seed,
            "virtual_only": True,
            "remaining_cards": int(sum(counts)),
            "remaining_counts": counts,
            "probabilities": {"B": banker, "P": player, "T": tie},
            "raw_probabilities": {"B": banker, "P": player, "T": tie},
            "tie_signal_allowed": False,
            "banker_rate": round(banker * 100.0, 2),
            "player_rate": round(player * 100.0, 2),
            "tie_rate": round(tie * 100.0, 2),
            "no_tie_probabilities": {"B": banker_no_tie, "P": player_no_tie},
            "core_probabilities_before_road": {
                "B": core_banker,
                "P": core_player,
                "T": core_tie,
            },
            "group_probabilities": {
                "road": {"B": float(road_three[0]), "P": float(road_three[1]), "T": float(road_three[2])},
                "finite": {"B": float(finite_group[0]), "P": float(finite_group[1]), "T": float(finite_group[2])},
                "sequence": {"B": float(sequence_probability[0]), "P": float(sequence_probability[1]), "T": float(sequence_probability[2])},
            },
            "hypergeometric_probabilities": {
                key: float(hyper_probability[index])
                for index, key in enumerate(OUTCOME_NAMES)
            },
            "monte_carlo_probabilities": {
                key: float(mc_probability[index])
                for index, key in enumerate(OUTCOME_NAMES)
            },
            "particle_probabilities": {
                key: float(particle_probability[index])
                for index, key in enumerate(OUTCOME_NAMES)
            },
            "sequence_probabilities": {
                key: float(sequence_probability[index])
                for index, key in enumerate(OUTCOME_NAMES)
            },
            "draw_path_probabilities": {
                PATH_NAMES[index]: float(hyper_paths[index]) for index in range(4)
            },
            "monte_carlo_draw_path_probabilities": {
                PATH_NAMES[index]: float(mc_path_probability[index]) for index in range(4)
            },
            "particle_draw_path_probabilities": {
                PATH_NAMES[index]: float(particle_paths[index]) for index in range(4)
            },
            "confidence_interval_95": {
                key: {"low": float(ci_lower[index]), "high": float(ci_upper[index])}
                for index, key in enumerate(OUTCOME_NAMES)
            },
            "recommend": direction,
            "recommend_text": "莊" if direction == "B" else "閒" if direction == "P" else "觀望",
            "action": action,
            "action_text": "莊" if action == "B" else "閒" if action == "P" else "觀望",
            "direction_edge": float(direction_edge),
            "direction_edge_percent": round(direction_edge * 100.0, 4),
            "uncertainty": uncertainty,
            "validation_gap": validation_gap,
            "max_validation_gap": MAX_VALIDATION_GAP,
            "max_signal_uncertainty": MAX_SIGNAL_UNCERTAINTY,
            "quality_score": round(quality_score, 6),
            "confidence_label": confidence_label,
            "model_consistency": round(model_consistency, 6),
            "signal_allowed": signal_allowed,
            "signal_status_text": signal_status_text,
            "signal_reason": signal_reason,
            "direction_source": "all_model_group_ensemble",
            "road_direction_primary": False,
            "all_models_participate": True,
            "model_directions": model_directions,
            "model_agreement": round(agreement, 6),
            "fused_direction_before_road_override": fused_direction,
            "road_integration": road_integration,
            "expected_values": expected_values,
            "best_ev_side": best_ev_side,
            "best_ev": best_ev,
            "positive_ev": best_ev > 0.0,
            "weights": {
                "hypergeometric": round(hyper_weight, 6),
                "monte_carlo": round(mc_weight, 6),
                "particle": round(particle_weight, 6),
                "sequence": round(sequence_weight, 6),
                "road_context": round(road_weight, 6),
                "finite_group": round(group_weights["finite"], 6),
                "road_group": round(group_weights["road"], 6),
                "sequence_group": round(group_weights["sequence"], 6),
                "baseline_shrink": round(adaptive_shrink, 6),
            },
            "hypergeometric_meta": hyper_meta,
            "sequence_meta": sequence_meta,
            "diagnostics": {
                "particles": self.settings.particles,
                "particle_ess": round(float(particle_ess), 3),
                "replicas": self.settings.replicas,
                "simulations_per_replica": self.settings.simulations_per_replica,
                "total_mc_simulations": self.settings.replicas * self.settings.simulations_per_replica,
                "replica_standard_error": {
                    key: float(replica_se[index]) for index, key in enumerate(OUTCOME_NAMES)
                },
                "history_length": history_length,
                "draw_path_history_length": len(draw_path_history or []),
            },
        }


class V5IndependentBaccaratEngine(VirtualShoeParticleEngine):
    """Compatibility alias for older imports."""


__all__ = [
    "BANKER_COMMISSION",
    "MAX_VALIDATION_GAP",
    "ROAD_CONTEXT_MAX_WEIGHT",
    "DB_HOLDOUT",
    "DEFAULT_BASELINE",
    "EngineSettings",
    "HandResult",
    "V5IndependentBaccaratEngine",
    "VirtualShoeParticleEngine",
    "baccarat_total",
    "banker_should_draw",
    "counts_from_shoe",
    "create_virtual_shoe",
    "deal_ordered_hand",
    "fresh_counts",
    "hypergeometric_probabilities",
    "simulate_one_from_counts",
]
