"""Finite-population baccarat engine with an exact hypergeometric core.

The dealer owns a hidden shuffled virtual shoe.  The predictor receives only:
- the remaining card-value counts (0-9), and
- past virtual outcomes / draw paths.

The main probability layer exactly enumerates next-hand outcomes under sampling
without replacement.  That is the multivariate-hypergeometric structure of a
finite baccarat shoe.  Monte Carlo replicas and a low-weight hidden-order
particle ensemble are retained as validation / uncertainty layers.

This is a simulation engine.  It does not read or predict an external live table.
V10.8 fits full-history probabilities, plans the complete road, bounds recent-trend influence, and then performs constrained stacking with posterior simulation.
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

from global_probability_model import analyze_global_probability
from stacking_model import constrained_stacking


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
PARTICLE_COUNT = _env_int("PF_PARTICLES", 250, 64, 4000)
REPLICA_COUNT = _env_int("PF_REPLICAS", 3, 3, 11)
SIMULATIONS_PER_REPLICA = _env_int(
    "PF_PREDICT_SIMULATIONS_PER_REPLICA", 600, 200, 20_000
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
            "model_disagreement": float(road.get("model_disagreement", 0.20) or 0.20),
            "models": dict(road.get("models") or {}),
            "regime": dict(road.get("regime") or {}),
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

        clean_history = [
            str(item).upper().strip()
            for item in (history or [])
            if str(item).upper().strip() in OUTCOME_NAMES
        ]
        history_length = len(clean_history)
        run_seed = int(seed if seed is not None else secrets.randbits(32)) & 0xFFFFFFFF
        seed_sequence = np.random.SeedSequence(run_seed)
        child_seeds = seed_sequence.spawn(self.settings.replicas + 1)

        # 有限牌組群組：精確超幾何為中心，蒙地卡羅與粒子只做驗證／不確定性。
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

        finite_requested = np.asarray(
            [HYPERGEOMETRIC_WEIGHT, MONTE_CARLO_WEIGHT, PARTICLE_WEIGHT],
            dtype=np.float64,
        )
        finite_requested /= finite_requested.sum()
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

        # 全歷史機率擬合獨立運算，不從近期牌路結果推導。
        global_result = analyze_global_probability(clean_history)
        global_probability = _normalize_probability([
            float(dict(global_result.get("probabilities") or {}).get("B", DEFAULT_BASELINE[0])),
            float(dict(global_result.get("probabilities") or {}).get("P", DEFAULT_BASELINE[1])),
            float(dict(global_result.get("probabilities") or {}).get("T", DEFAULT_BASELINE[2])),
        ])

        # 畫面模式由 screenshot_predictor 先建好 road_context；虛擬模式則在這裡補建。
        resolved_road_context = dict(road_context or {})
        if not resolved_road_context and clean_history:
            try:
                from road_model import build_road_context
                resolved_road_context = build_road_context(
                    clean_history,
                    seed=(run_seed ^ 0x9E3779B9) & 0xFFFFFFFF,
                )
            except Exception:
                resolved_road_context = {}

        road_state = self._road_context_state(resolved_road_context)
        core_tie = float(finite_group[2])
        planning_b = max(0.0, min(1.0, float(
            resolved_road_context.get(
                "planning_probability",
                resolved_road_context.get("banker_probability", 0.5),
            ) or 0.5
        )))
        recent_b = max(0.0, min(1.0, float(
            resolved_road_context.get(
                "recent_probability",
                resolved_road_context.get("banker_probability", 0.5),
            ) or 0.5
        )))
        planning_three = _normalize_probability([
            (1.0 - core_tie) * planning_b,
            (1.0 - core_tie) * (1.0 - planning_b),
            core_tie,
        ])
        recent_three = _normalize_probability([
            (1.0 - core_tie) * recent_b,
            (1.0 - core_tie) * (1.0 - recent_b),
            core_tie,
        ])

        sequence_probability, sequence_meta = _sequence_probabilities(clean_history)

        particle_ess_ratio = min(
            1.0,
            particle_ess / max(1.0, float(self.settings.particles)),
        )
        validation_quality = 1.0 - min(1.0, validation_gap / 0.06)
        uncertainty_quality = 1.0 - min(1.0, uncertainty / 0.025)
        finite_quality = max(
            0.0,
            min(1.0, 0.46 * validation_quality + 0.34 * uncertainty_quality + 0.20 * particle_ess_ratio),
        )
        sequence_support = max(0, int(sequence_meta.get("support", 0) or 0))
        sequence_order = max(0, int(sequence_meta.get("order", 0) or 0))
        sequence_quality = max(0.0, min(1.0,
            0.38 * min(1.0, history_length / 42.0)
            + 0.42 * min(1.0, sequence_support / 18.0)
            + 0.20 * min(1.0, sequence_order / 3.0)
        ))

        planning_reliability = max(0.0, min(1.0, float(
            resolved_road_context.get("planning_reliability", 0.0) or 0.0
        )))
        recent_reliability = max(0.0, min(1.0, float(
            resolved_road_context.get("recent_reliability", 0.0) or 0.0
        )))
        global_reliability = max(0.0, min(1.0, float(
            global_result.get("reliability", 0.0) or 0.0
        )))

        composition_quality = "estimated" if road_context is not None else "observed"
        stacking = constrained_stacking(
            probabilities={
                "global_history": global_probability,
                "road_planning": planning_three,
                "recent_road": recent_three,
                "finite": finite_group,
                "sequence": sequence_probability,
            },
            qualities={
                "global_history": global_reliability,
                "road_planning": planning_reliability,
                "recent_road": recent_reliability,
                "finite": finite_quality,
                "sequence": sequence_quality,
            },
            availability={
                "global_history": bool(global_result.get("ok")),
                "road_planning": bool(resolved_road_context.get("planning_available")) or history_length >= 10,
                "recent_road": history_length >= 4,
                "finite": True,
                "sequence": history_length >= 4,
            },
            composition_quality=composition_quality,
            seed=(run_seed ^ 0xA5A5A5A5) & 0xFFFFFFFF,
        )

        stacked_probabilities = dict(stacking["probabilities"])
        ensemble_probability = _normalize_probability([
            stacked_probabilities["B"],
            stacked_probabilities["P"],
            stacked_probabilities["T"],
        ])
        banker, player, tie = (float(value) for value in ensemble_probability)
        bp_total = max(1e-12, banker + player)
        banker_no_tie = banker / bp_total
        player_no_tie = player / bp_total
        direction = "B" if banker >= player else "P"
        direction_edge = abs(banker_no_tie - player_no_tie)

        weights = dict(stacking["weights"])
        posterior = dict(stacking.get("posterior") or {})
        direction_stability = float(posterior.get("direction_stability", 0.0) or 0.0)
        minimum_stability = float(posterior.get("minimum_direction_stability", 0.58) or 0.58)
        ensemble_uncertainty = float(posterior.get("bp_difference_std", 1.0) or 1.0)
        weighted_agreement = float(stacking.get("weighted_agreement", 0.0) or 0.0)

        group_directions = {
            name: str(dict(stacking.get("contributions") or {}).get(name, {}).get("direction") or "")
            for name in ("global_history", "road_planning", "recent_road", "finite", "sequence")
        }
        model_directions = {
            **group_directions,
            "hypergeometric": "B" if hyper_probability[0] >= hyper_probability[1] else "P",
            "monte_carlo": "B" if mc_probability[0] >= mc_probability[1] else "P",
            "particle": "B" if particle_probability[0] >= particle_probability[1] else "P",
        }

        edge_ok = direction_edge >= MIN_DIRECTION_EDGE
        uncertainty_ok = uncertainty <= MAX_SIGNAL_UNCERTAINTY
        validation_ok = validation_gap <= MAX_VALIDATION_GAP
        agreement_ok = weighted_agreement >= ENSEMBLE_MIN_MODEL_AGREEMENT
        stability_ok = direction_stability >= minimum_stability
        signal_allowed = bool(
            edge_ok and uncertainty_ok and validation_ok and agreement_ok and stability_ok
        )
        action = direction if signal_allowed else "O"

        banker_ev = banker * (1.0 - BANKER_COMMISSION) - player
        player_ev = player - banker
        tie_ev = tie * 8.0 - (banker + player)
        expected_values = {"B": banker_ev, "P": player_ev, "T": tie_ev}
        best_ev_side = max(expected_values, key=expected_values.get)
        best_ev = float(expected_values[best_ev_side])

        quality_score = max(0.0, min(1.0,
            0.22 * global_reliability
            + 0.22 * planning_reliability
            + 0.12 * recent_reliability
            + 0.16 * finite_quality
            + 0.08 * sequence_quality
            + 0.12 * direction_stability
            + 0.08 * weighted_agreement
        ))
        confidence_label = "較高" if quality_score >= 0.72 else "中等" if quality_score >= 0.50 else "偏低"

        if signal_allowed:
            signal_reason = "全歷史機率、完整牌路規劃、受限近期專家、有限牌組與序列 Stacking 門檻均通過"
            signal_status_text = "V10.8 全歷史統合方向已開放"
        else:
            reasons: List[str] = []
            if not edge_ok:
                reasons.append("Stacking 莊閒差距不足")
            if not uncertainty_ok:
                reasons.append("蒙地卡羅不確定性偏高")
            if not validation_ok:
                reasons.append("有限牌組驗證層差距偏高")
            if not agreement_ok:
                reasons.append("五群組加權共識不足")
            if not stability_ok:
                reasons.append("後驗方向穩定度不足")
            signal_reason = "、".join(reasons) or "目前資料尚未形成正式方向訊號"
            signal_status_text = "等待更明確的全歷史統合訊號"

        road_weight = weights.get("road_planning", 0.0) + weights.get("recent_road", 0.0)
        road_integration = {
            "processed_inside_core": True,
            "present": bool(resolved_road_context),
            "applied": road_weight > 0.0,
            "eligible": bool(resolved_road_context.get("ok")),
            "sample_count": history_length,
            "road_direction": "B" if planning_b >= 0.5 else "P",
            "final_direction": direction,
            "road_direction_primary": False,
            "all_models_participate": True,
            "effective_weight": round(road_weight, 8),
            "planning_weight": round(weights.get("road_planning", 0.0), 8),
            "recent_weight": round(weights.get("recent_road", 0.0), 8),
            "global_history_weight": round(weights.get("global_history", 0.0), 8),
            "road_engine": str(resolved_road_context.get("engine") or ""),
        }

        return {
            "ok": True,
            "engine": "V10_8_FULL_HISTORY_ROAD_PLANNING_STACKING",
            "model_core": "global_probability_road_planning_bounded_recent_finite_sequence",
            "pipeline_order": [
                "full_history_probability_fit",
                "complete_road_planning",
                "bounded_recent_road_experts",
                "finite_hypergeometric_mc_particle",
                "sequence_probability",
                "constrained_five_group_stacking",
                "posterior_stability_and_signal_gate",
            ],
            "run_seed": run_seed,
            "virtual_only": True,
            "remaining_cards": int(sum(counts)),
            "remaining_counts": counts,
            "probabilities": {"B": banker, "P": player, "T": tie},
            "raw_probabilities": {"B": banker, "P": player, "T": tie},
            "stacked_probabilities_before_adaptation": {"B": banker, "P": player, "T": tie},
            "tie_signal_allowed": False,
            "banker_rate": round(banker * 100.0, 2),
            "player_rate": round(player * 100.0, 2),
            "tie_rate": round(tie * 100.0, 2),
            "no_tie_probabilities": {"B": banker_no_tie, "P": player_no_tie},
            "core_probabilities_before_road": {
                "B": float(finite_group[0]),
                "P": float(finite_group[1]),
                "T": float(finite_group[2]),
            },
            "group_probabilities": {
                "global_history": {key: float(global_probability[index]) for index, key in enumerate(OUTCOME_NAMES)},
                "road_planning": {key: float(planning_three[index]) for index, key in enumerate(OUTCOME_NAMES)},
                "recent_road": {key: float(recent_three[index]) for index, key in enumerate(OUTCOME_NAMES)},
                "finite": {key: float(finite_group[index]) for index, key in enumerate(OUTCOME_NAMES)},
                "sequence": {key: float(sequence_probability[index]) for index, key in enumerate(OUTCOME_NAMES)},
                "road": {
                    "B": float((planning_three[0] * weights.get("road_planning", 0.0) + recent_three[0] * weights.get("recent_road", 0.0)) / max(1e-12, road_weight)),
                    "P": float((planning_three[1] * weights.get("road_planning", 0.0) + recent_three[1] * weights.get("recent_road", 0.0)) / max(1e-12, road_weight)),
                    "T": core_tie,
                } if road_weight > 0 else {"B": 0.5 * (1.0 - core_tie), "P": 0.5 * (1.0 - core_tie), "T": core_tie},
            },
            "global_probability_model": global_result,
            "road_planning_model": dict(resolved_road_context.get("full_road_analysis") or {}),
            "recent_road_model": {
                "banker_probability": recent_b,
                "reliability": recent_reliability,
                "uncertainty": resolved_road_context.get("recent_uncertainty"),
                "model_disagreement": resolved_road_context.get("recent_model_disagreement"),
                "models": dict(resolved_road_context.get("models") or {}),
            },
            "hypergeometric_probabilities": {key: float(hyper_probability[index]) for index, key in enumerate(OUTCOME_NAMES)},
            "monte_carlo_probabilities": {key: float(mc_probability[index]) for index, key in enumerate(OUTCOME_NAMES)},
            "particle_probabilities": {key: float(particle_probability[index]) for index, key in enumerate(OUTCOME_NAMES)},
            "sequence_probabilities": {key: float(sequence_probability[index]) for index, key in enumerate(OUTCOME_NAMES)},
            "draw_path_probabilities": {PATH_NAMES[index]: float(hyper_paths[index]) for index in range(4)},
            "monte_carlo_draw_path_probabilities": {PATH_NAMES[index]: float(mc_path_probability[index]) for index in range(4)},
            "particle_draw_path_probabilities": {PATH_NAMES[index]: float(particle_paths[index]) for index in range(4)},
            "confidence_interval_95": {
                key: {"low": float(ci_lower[index]), "high": float(ci_upper[index])}
                for index, key in enumerate(OUTCOME_NAMES)
            },
            "recommend": direction,
            "recommend_text": "莊" if direction == "B" else "閒",
            "action": action,
            "action_text": "莊" if action == "B" else "閒" if action == "P" else "觀望",
            "direction_edge": float(direction_edge),
            "direction_edge_percent": round(direction_edge * 100.0, 4),
            "uncertainty": uncertainty,
            "ensemble_uncertainty": ensemble_uncertainty,
            "posterior_direction_stability": direction_stability,
            "posterior_interval_crosses_zero": bool(posterior.get("bp_difference_interval_crosses_zero", True)),
            "validation_gap": validation_gap,
            "max_validation_gap": MAX_VALIDATION_GAP,
            "max_signal_uncertainty": MAX_SIGNAL_UNCERTAINTY,
            "quality_score": round(quality_score, 6),
            "confidence_label": confidence_label,
            "model_consistency": round(weighted_agreement, 6),
            "signal_allowed": signal_allowed,
            "signal_status_text": signal_status_text,
            "signal_reason": signal_reason,
            "direction_source": "full_history_constrained_stacking",
            "road_direction_primary": False,
            "all_models_participate": True,
            "model_directions": model_directions,
            "model_agreement": round(weighted_agreement, 6),
            "fused_direction_before_road_override": direction,
            "road_integration": road_integration,
            "stacking": stacking,
            "dynamic_ensemble": stacking,
            "posterior_simulation": posterior,
            "composition_quality": composition_quality,
            "expected_values": expected_values,
            "best_ev_side": best_ev_side,
            "best_ev": best_ev,
            "positive_ev": best_ev > 0.0,
            "weights": {
                "hypergeometric": round(hyper_weight, 6),
                "monte_carlo": round(mc_weight, 6),
                "particle": round(particle_weight, 6),
                "global_history_group": round(weights.get("global_history", 0.0), 6),
                "road_planning_group": round(weights.get("road_planning", 0.0), 6),
                "recent_road_group": round(weights.get("recent_road", 0.0), 6),
                "finite_group": round(weights.get("finite", 0.0), 6),
                "sequence_group": round(weights.get("sequence", 0.0), 6),
                "road_context": round(road_weight, 6),
                "road_group": round(road_weight, 6),
                "sequence": round(weights.get("sequence", 0.0), 6),
                "baseline_shrink": round(adaptive_shrink, 6),
                "dynamic_weighting": True,
                "bounded_stacking": True,
            },
            "hypergeometric_meta": hyper_meta,
            "sequence_meta": sequence_meta,
            "diagnostics": {
                "particles": self.settings.particles,
                "particle_ess": round(float(particle_ess), 3),
                "replicas": self.settings.replicas,
                "simulations_per_replica": self.settings.simulations_per_replica,
                "total_mc_simulations": self.settings.replicas * self.settings.simulations_per_replica,
                "replica_standard_error": {key: float(replica_se[index]) for index, key in enumerate(OUTCOME_NAMES)},
                "history_length": history_length,
                "full_history_used_count": history_length,
                "draw_path_history_length": len(draw_path_history or []),
                "posterior_simulations": int(posterior.get("simulations", 0) or 0),
                "composition_quality": composition_quality,
                "recent_weight_capped": weights.get("recent_road", 0.0) <= 0.25 + 1e-9,
                "finite_weight_not_double_penalized": weights.get("finite", 0.0) >= 0.12 - 1e-9,
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