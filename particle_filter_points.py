"""V5 stateless baccarat particle engine for the official LINE predictor.

Design guarantees
-----------------
* Every prediction creates fresh particles and discards them afterward.
* Only the newest Player/Banker final-point observation is used.
* No road, streak, Markov, point-sequence, previous prediction, or UID state.
* Unknown shoe depth uses the calibrated 10/30/40/20 stratified prior.
* Final points are conditioned with exact legal third-card completion and
  importance weights. Duplicate paths are intentionally preserved.
* Every conditioned particle remains paired with its true ancestor shoe.
* Forecast uses common random numbers and antithetic variates for paired
  conditioned-vs-control comparisons.
* The 5M shoe database is diagnostic by default because its point mapping did
  not beat the baseline in a separate holdout test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
import hashlib
import math
import os

import numpy as np

from shoe_state_db import DEFAULT_BASELINE, DEFAULT_DRAW, get_shoe_state_database


# ---------------------------------------------------------------------------
# Environment helpers and V5 defaults
# ---------------------------------------------------------------------------
def _env_int(name: str, default: int, minimum: int = 0, maximum: Optional[int] = None) -> int:
    try:
        value = int(os.getenv(name, str(default)).strip())
    except Exception:
        value = default
    value = max(minimum, value)
    return min(maximum, value) if maximum is not None else value


def _env_float(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(os.getenv(name, str(default)).strip())
    except Exception:
        value = default
    return max(low, min(high, value))


def _env_choice(name: str, default: str, allowed: Sequence[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    return value if value in allowed else default


DECKS = _env_int("PF_DECKS", 8, 1, 16)
PARTICLE_COUNT = _env_int("PF_PARTICLES", 192, 64, 768)
REPLICA_COUNT = _env_int("PF_REPLICAS", 7, 3, 15)
TARGET_MATCHES = _env_int("PF_TARGET_MATCHES", 192, 32, 2048)
TARGET_ESS = _env_float("PF_TARGET_ESS", 120.0, 8.0, 2048.0)
MIN_MATCHES = _env_int("PF_MIN_MATCHES", 24, 1, 2048)
MAX_UPDATE_PROPOSALS = _env_int("PF_MAX_UPDATE_PROPOSALS", 25_000, 500, 500_000)
PREDICT_SIMULATIONS_PER_REPLICA = _env_int(
    "PF_PREDICT_SIMULATIONS_PER_REPLICA", 1_500, 300, 100_000
)
DATABASE_WEIGHT = _env_float("PF_DATABASE_WEIGHT", 0.20, 0.0, 0.75)
DATABASE_MAX_ADJUSTMENT = _env_float("PF_DATABASE_MAX_ADJUSTMENT", 0.005, 0.0, 0.05)
DATABASE_VALIDATION_MODE = _env_choice(
    "PF_DATABASE_VALIDATION_MODE",
    "validated_only",
    ("validated_only", "diagnostic", "force"),
)
UNCERTAINTY_PENALTY = _env_float("PF_UNCERTAINTY_PENALTY", 1.28, 0.0, 5.0)
MIN_VALIDATED_EDGE = _env_float("PF_MIN_VALIDATED_EDGE", 0.0012, 0.0, 0.05)
MIN_REPLICA_AGREEMENT = _env_float("PF_MIN_REPLICA_AGREEMENT", 0.71, 0.50, 1.0)
BANKER_COMMISSION = _env_float("PF_BANKER_COMMISSION", 0.05, 0.0, 0.20)
DECISION_MODE = _env_choice("PF_DECISION_MODE", "validated", ("validated", "centered", "raw", "ev"))

BASELINE = np.asarray(DEFAULT_BASELINE, dtype=float)
DRAW_BASELINE = np.asarray(DEFAULT_DRAW, dtype=float)
DRAW_NAMES = ("none", "player_only", "banker_only", "both")

# Holdout numbers embedded in V5. The database remains diagnostic unless forced.
DB_HOLDOUT: Dict[str, Any] = {
    "passed": False,
    "point_map_rate": 0.4999,
    "baseline_rate": 0.5068,
    "samples": 500_000,
    "note": "樣本外點數映射未優於固定莊家基準，預設抑制資料庫方向校正",
}

# Calibrated unknown-shoe depth profile used on every independent prediction.
CALIBRATED_DEPTH_PROFILE: Tuple[Tuple[int, int, float], ...] = (
    (0, 10, 0.10),
    (11, 25, 0.30),
    (26, 40, 0.40),
    (41, 55, 0.20),
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class HandResult:
    player_total: int
    banker_total: int
    outcome: str
    draw_path: int
    counts_after: np.ndarray
    cards_used: int


@dataclass
class ConditionalCandidate:
    counts: np.ndarray
    depth: int
    draw_path: int
    weight: float
    control_counts: np.ndarray
    control_depth: int


@dataclass
class ConditionedPopulation:
    particles: List[np.ndarray]
    depths: List[int]
    control_particles: List[np.ndarray]
    control_depths: List[int]
    updated: bool
    low_sample: bool
    matches: int
    attempts: int
    ess: float
    acceptance: float
    draw_paths: np.ndarray
    unique_particles: int
    accepted_unique: int
    ancestry_paired: bool


@dataclass
class ReplicaResult:
    pf: np.ndarray
    control: np.ndarray
    database: np.ndarray
    fused: np.ndarray
    paired_center: float
    effective_database_weight: float
    database_reliability: float
    database_samples: float
    composition: Dict[str, Any]
    digest: str
    seed: int
    mean_depth: float
    min_depth: int
    max_depth: int
    matches: int
    attempts: int
    ess: float
    acceptance: float
    draw_paths: np.ndarray
    unique_particles: int
    accepted_unique: int
    diversity: float
    updated: bool
    low_sample: bool
    ancestry_paired: bool


# ---------------------------------------------------------------------------
# Random helpers
# ---------------------------------------------------------------------------
def mix_seed(seed: int, index: int) -> int:
    """32-bit avalanche mixer mirroring the V5 browser seed separation."""
    x = (int(seed) + ((index + 1) * 0x9E3779B9)) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    x ^= x >> 16
    return x & 0xFFFFFFFF


class UniformStream:
    """Deterministic stream optionally transformed into antithetic uniforms."""

    def __init__(self, seed: int, antithetic: bool = False) -> None:
        self.rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        self.antithetic = bool(antithetic)

    def random(self) -> float:
        value = float(self.rng.random())
        if self.antithetic:
            value = 1.0 - value
            # Keep the half-open [0, 1) invariant.
            if value >= 1.0:
                value = np.nextafter(1.0, 0.0)
        return value


# ---------------------------------------------------------------------------
# Baccarat mechanics
# ---------------------------------------------------------------------------
def fresh_shoe_counts(decks: int = DECKS) -> np.ndarray:
    # 0 contains 10/J/Q/K: 16 cards per deck. Values 1..9: 4 per deck.
    return np.asarray([16 * decks] + [4 * decks] * 9, dtype=np.int16)


def _draw_uniform(counts: np.ndarray, random_fn: Callable[[], float]) -> int:
    total = int(counts.sum())
    if total <= 0:
        raise RuntimeError("shoe is empty")
    ticket = min(total - 1, int(random_fn() * total))
    running = 0
    for value in range(10):
        running += int(counts[value])
        if ticket < running:
            counts[value] -= 1
            return value
    counts[0] -= 1
    return 0


def _draw_np(counts: np.ndarray, rng: np.random.Generator) -> int:
    total = int(counts.sum())
    if total <= 0:
        raise RuntimeError("shoe is empty")
    ticket = int(rng.integers(0, total))
    running = 0
    for value in range(10):
        running += int(counts[value])
        if ticket < running:
            counts[value] -= 1
            return value
    counts[0] -= 1
    return 0


def banker_draws(banker_total: int, player_third: Optional[int]) -> bool:
    if player_third is None:
        return banker_total <= 5
    if banker_total <= 2:
        return True
    if banker_total == 3:
        return player_third != 8
    if banker_total == 4:
        return 2 <= player_third <= 7
    if banker_total == 5:
        return 4 <= player_third <= 7
    if banker_total == 6:
        return 6 <= player_third <= 7
    return False


def simulate_hand_uniform(source: np.ndarray, random_fn: Callable[[], float]) -> HandResult:
    counts = np.asarray(source, dtype=np.int16).copy()
    p1 = _draw_uniform(counts, random_fn)
    b1 = _draw_uniform(counts, random_fn)
    p2 = _draw_uniform(counts, random_fn)
    b2 = _draw_uniform(counts, random_fn)
    player_total = (p1 + p2) % 10
    banker_total = (b1 + b2) % 10
    player_third: Optional[int] = None
    banker_third: Optional[int] = None
    if player_total < 8 and banker_total < 8:
        if player_total <= 5:
            player_third = _draw_uniform(counts, random_fn)
            player_total = (player_total + player_third) % 10
        if banker_draws(banker_total, player_third):
            banker_third = _draw_uniform(counts, random_fn)
            banker_total = (banker_total + banker_third) % 10
    outcome = "B" if banker_total > player_total else "P" if player_total > banker_total else "T"
    draw_path = (1 if player_third is not None else 0) + (2 if banker_third is not None else 0)
    cards_used = 4 + (1 if player_third is not None else 0) + (1 if banker_third is not None else 0)
    return HandResult(player_total, banker_total, outcome, draw_path, counts, cards_used)


def simulate_hand_np(source: np.ndarray, rng: np.random.Generator) -> HandResult:
    counts = np.asarray(source, dtype=np.int16).copy()
    p1 = _draw_np(counts, rng)
    b1 = _draw_np(counts, rng)
    p2 = _draw_np(counts, rng)
    b2 = _draw_np(counts, rng)
    player_total = (p1 + p2) % 10
    banker_total = (b1 + b2) % 10
    player_third: Optional[int] = None
    banker_third: Optional[int] = None
    if player_total < 8 and banker_total < 8:
        if player_total <= 5:
            player_third = _draw_np(counts, rng)
            player_total = (player_total + player_third) % 10
        if banker_draws(banker_total, player_third):
            banker_third = _draw_np(counts, rng)
            banker_total = (banker_total + banker_third) % 10
    outcome = "B" if banker_total > player_total else "P" if player_total > banker_total else "T"
    draw_path = (1 if player_third is not None else 0) + (2 if banker_third is not None else 0)
    cards_used = 4 + (1 if player_third is not None else 0) + (1 if banker_third is not None else 0)
    return HandResult(player_total, banker_total, outcome, draw_path, counts, cards_used)


def _required_card_probability(counts: np.ndarray, value: int) -> float:
    total = int(counts.sum())
    if total <= 0 or int(counts[value]) <= 0:
        return 0.0
    return float(counts[value]) / float(total)


def exact_conditional_complete(
    source: np.ndarray,
    rng: np.random.Generator,
    observed_player: int,
    observed_banker: int,
) -> Optional[Tuple[np.ndarray, int, float, int]]:
    """Sample the first four cards, then exactly complete legal third cards.

    The first four cards are drawn from the proposal distribution. Required
    third cards are forced from the requested final totals and their remaining-
    shoe probabilities are retained as importance weights.
    """
    counts = np.asarray(source, dtype=np.int16).copy()
    try:
        p1 = _draw_np(counts, rng)
        b1 = _draw_np(counts, rng)
        p2 = _draw_np(counts, rng)
        b2 = _draw_np(counts, rng)
    except RuntimeError:
        return None

    player_total = (p1 + p2) % 10
    banker_total = (b1 + b2) % 10
    player_third: Optional[int] = None
    banker_third: Optional[int] = None
    weight = 1.0

    # Natural: neither side draws, and the exact final totals must already fit.
    if player_total >= 8 or banker_total >= 8:
        if player_total != observed_player or banker_total != observed_banker:
            return None
    else:
        if player_total <= 5:
            player_third = (observed_player - player_total) % 10
            probability = _required_card_probability(counts, player_third)
            if probability <= 0.0:
                return None
            weight *= probability
            counts[player_third] -= 1
            player_total = observed_player
        elif player_total != observed_player:
            return None

        if banker_draws(banker_total, player_third):
            banker_third = (observed_banker - banker_total) % 10
            probability = _required_card_probability(counts, banker_third)
            if probability <= 0.0:
                return None
            weight *= probability
            counts[banker_third] -= 1
            banker_total = observed_banker
        elif banker_total != observed_banker:
            return None

    if player_total != observed_player or banker_total != observed_banker:
        return None

    draw_path = (1 if player_third is not None else 0) + (2 if banker_third is not None else 0)
    cards_used = 4 + (1 if player_third is not None else 0) + (1 if banker_third is not None else 0)
    return counts, draw_path, max(1e-12, float(weight)), cards_used


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def normalize_array(values: Sequence[float], fallback: Sequence[float] = BASELINE) -> np.ndarray:
    arr = np.maximum(0.0, np.asarray(values, dtype=float))
    total = float(arr.sum())
    if total <= 0.0:
        arr = np.asarray(fallback, dtype=float).copy()
        total = float(arr.sum())
    return arr / total


def weighted_ess(items: Sequence[ConditionalCandidate]) -> float:
    weights = np.asarray([max(0.0, float(item.weight)) for item in items], dtype=float)
    total = float(weights.sum())
    denom = float(np.square(weights).sum())
    return (total * total / denom) if denom > 0.0 else 0.0


def systematic_weighted_resample(
    items: Sequence[ConditionalCandidate], n: int, rng: np.random.Generator
) -> List[ConditionalCandidate]:
    weights = np.asarray([max(0.0, float(item.weight)) for item in items], dtype=float)
    total = float(weights.sum())
    if total <= 0.0 or not items:
        return []
    weights /= total
    cumulative = np.cumsum(weights)
    positions = (float(rng.random()) + np.arange(n, dtype=float)) / float(n)
    indexes = np.searchsorted(cumulative, positions, side="left")
    indexes = np.clip(indexes, 0, len(items) - 1)
    return [items[int(index)] for index in indexes]


def median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def _outcome_index(outcome: str) -> int:
    return 0 if outcome == "B" else 1 if outcome == "P" else 2


def _composition(particles: Sequence[np.ndarray], decks: int) -> Dict[str, Any]:
    matrix = np.stack([np.asarray(p, dtype=float) for p in particles], axis=0)
    mean = matrix.mean(axis=0)
    total = float(mean.sum())
    groups = np.asarray(
        [mean[0], mean[1:4].sum(), mean[4:7].sum(), mean[7:10].sum()], dtype=float
    )
    base = np.asarray([16 / 52, 12 / 52, 12 / 52, 12 / 52], dtype=float)
    ratios = groups / max(1e-12, total)
    removed = 52 * decks - total
    return {
        "mean_value_counts": [round(float(v), 6) for v in mean.tolist()],
        "cards_remaining": total,
        "cards_removed": removed,
        "shoe_depth": removed / float(52 * decks),
        "group_counts": {
            "zero": float(groups[0]),
            "low_1_3": float(groups[1]),
            "mid_4_6": float(groups[2]),
            "high_7_9": float(groups[3]),
        },
        "group_ratios": {
            "zero": float(ratios[0]),
            "low_1_3": float(ratios[1]),
            "mid_4_6": float(ratios[2]),
            "high_7_9": float(ratios[3]),
        },
        "relative_composition": {
            "zero": float(ratios[0] / base[0] - 1.0),
            "low_1_3": float(ratios[1] / base[1] - 1.0),
            "mid_4_6": float(ratios[2] / base[2] - 1.0),
            "high_7_9": float(ratios[3] / base[3] - 1.0),
        },
    }


def _digest(particles: Sequence[np.ndarray], seed: int) -> str:
    h = hashlib.sha1()
    h.update(str(int(seed) & 0xFFFFFFFF).encode("ascii"))
    for particle in particles[: min(32, len(particles))]:
        h.update(np.asarray(particle, dtype=np.int16).tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# V5 replica engine
# ---------------------------------------------------------------------------
class V5ReplicaEngine:
    def __init__(self, seed: int, particle_count: int = PARTICLE_COUNT, decks: int = DECKS) -> None:
        self.seed = int(seed) & 0xFFFFFFFF
        self.particle_count = max(64, int(particle_count))
        self.decks = max(1, int(decks))
        self.rng = np.random.default_rng(self.seed)

    def build_stratified_prior(self) -> Tuple[List[np.ndarray], List[int]]:
        allocations = [int(math.floor(self.particle_count * band[2])) for band in CALIBRATED_DEPTH_PROFILE]
        assigned = sum(allocations)
        index = 0
        while assigned < self.particle_count:
            allocations[index % len(allocations)] += 1
            assigned += 1
            index += 1

        particles: List[np.ndarray] = []
        depths: List[int] = []
        for (low, high, _weight), count in zip(CALIBRATED_DEPTH_PROFILE, allocations):
            span = high - low + 1
            for item_index in range(count):
                counts = fresh_shoe_counts(self.decks)
                quantile = (item_index + 0.5) / max(1, count)
                base_depth = low + min(span - 1, int(math.floor(quantile * span)))
                jitter = int(self.rng.integers(-1, 2))
                requested = max(low, min(high, base_depth + jitter))
                completed = 0
                for _ in range(requested):
                    if int(counts.sum()) < 12:
                        break
                    try:
                        hand = simulate_hand_np(counts, self.rng)
                    except RuntimeError:
                        break
                    counts = hand.counts_after
                    completed += 1
                particles.append(counts)
                depths.append(completed)
        return particles, depths

    def condition(
        self,
        prior_particles: Sequence[np.ndarray],
        prior_depths: Sequence[int],
        player_total: int,
        banker_total: int,
        target_matches: int = TARGET_MATCHES,
        target_ess: float = TARGET_ESS,
        min_matches: int = MIN_MATCHES,
        max_proposals: int = MAX_UPDATE_PROPOSALS,
    ) -> ConditionedPopulation:
        accepted: List[ConditionalCandidate] = []
        path_weights = np.zeros(4, dtype=float)
        attempts = 0
        ess = 0.0

        while attempts < max_proposals and (len(accepted) < target_matches or ess < target_ess):
            parent_index = int(self.rng.integers(0, len(prior_particles)))
            source = prior_particles[parent_index]
            completed = exact_conditional_complete(
                source,
                self.rng,
                int(player_total) % 10,
                int(banker_total) % 10,
            )
            attempts += 1
            if completed is None:
                continue
            counts, draw_path, weight, _cards_used = completed
            candidate = ConditionalCandidate(
                counts=counts,
                depth=int(prior_depths[parent_index]) + 1,
                draw_path=draw_path,
                weight=weight,
                control_counts=np.asarray(source, dtype=np.int16).copy(),
                control_depth=int(prior_depths[parent_index]),
            )
            accepted.append(candidate)
            path_weights[draw_path] += weight
            if len(accepted) % 16 == 0 or len(accepted) >= target_matches:
                ess = weighted_ess(accepted)

        if accepted:
            ess = weighted_ess(accepted)

        if not accepted:
            particles = [np.asarray(item, dtype=np.int16).copy() for item in prior_particles]
            unique = len({item.tobytes() for item in particles})
            return ConditionedPopulation(
                particles=particles,
                depths=[int(v) for v in prior_depths],
                control_particles=[item.copy() for item in particles],
                control_depths=[int(v) for v in prior_depths],
                updated=False,
                low_sample=True,
                matches=0,
                attempts=attempts,
                ess=0.0,
                acceptance=0.0,
                draw_paths=DRAW_BASELINE.copy(),
                unique_particles=unique,
                accepted_unique=0,
                ancestry_paired=False,
            )

        sampled = systematic_weighted_resample(accepted, self.particle_count, self.rng)
        particles = [item.counts.copy() for item in sampled]
        depths = [item.depth for item in sampled]
        controls = [item.control_counts.copy() for item in sampled]
        control_depths = [item.control_depth for item in sampled]
        unique = len({item.tobytes() for item in particles})
        accepted_unique = len({item.counts.tobytes() + int(item.depth).to_bytes(2, "little") for item in accepted})
        draw_paths = normalize_array(path_weights, DRAW_BASELINE)
        return ConditionedPopulation(
            particles=particles,
            depths=depths,
            control_particles=controls,
            control_depths=control_depths,
            updated=True,
            low_sample=len(accepted) < int(min_matches),
            matches=len(accepted),
            attempts=attempts,
            ess=ess,
            acceptance=len(accepted) / max(1, attempts),
            draw_paths=draw_paths,
            unique_particles=unique,
            accepted_unique=accepted_unique,
            ancestry_paired=True,
        )

    def forecast(
        self,
        population: ConditionedPopulation,
        simulations: int = PREDICT_SIMULATIONS_PER_REPLICA,
        database_weight: float = DATABASE_WEIGHT,
        database_max_adjustment: float = DATABASE_MAX_ADJUSTMENT,
        database_validation_mode: str = DATABASE_VALIDATION_MODE,
        target_ess: float = TARGET_ESS,
    ) -> ReplicaResult:
        simulations = max(300, int(simulations))
        direct = np.zeros(3, dtype=np.int64)
        control = np.zeros(3, dtype=np.int64)
        pair_count = int(math.ceil(simulations / 2.0))

        for pair_index in range(pair_count):
            pair_seed = mix_seed(self.seed, 1000 + pair_index)
            chooser = np.random.default_rng(mix_seed(self.seed, 50_000 + pair_index))
            particle_index = int(chooser.integers(0, len(population.particles)))
            for flip in (False, True):
                simulation_index = pair_index * 2 + (1 if flip else 0)
                if simulation_index >= simulations:
                    continue
                conditioned_stream = UniformStream(pair_seed, antithetic=flip)
                control_stream = UniformStream(pair_seed, antithetic=flip)
                conditioned_hand = simulate_hand_uniform(
                    population.particles[particle_index], conditioned_stream.random
                )
                control_hand = simulate_hand_uniform(
                    population.control_particles[particle_index], control_stream.random
                )
                direct[_outcome_index(conditioned_hand.outcome)] += 1
                control[_outcome_index(control_hand.outcome)] += 1

        pf = normalize_array(direct, BASELINE)
        control_probs = normalize_array(control, BASELINE)
        paired_center = float((pf[0] - control_probs[0]) - (pf[1] - control_probs[1]))

        database = get_shoe_state_database()
        db_total = np.zeros(3, dtype=float)
        reliability = 0.0
        sample_average = 0.0
        weight = 1.0 / len(population.particles)
        for particle in population.particles:
            estimate = database.estimate(particle, self.decks)
            db_total += weight * np.asarray(
                [
                    estimate.probabilities["B"],
                    estimate.probabilities["P"],
                    estimate.probabilities["T"],
                ],
                dtype=float,
            )
            reliability += weight * estimate.reliability
            sample_average += weight * estimate.samples
        db_probs = normalize_array(db_total, BASELINE)

        sample_scale = min(1.0, population.ess / max(1.0, float(target_ess)))
        database_passed = bool(DB_HOLDOUT["passed"])
        allow_database = database_validation_mode == "force" or (
            database_validation_mode == "validated_only" and database_passed
        )
        if database_validation_mode == "diagnostic":
            allow_database = False
        effective = (
            max(0.0, min(0.75, float(database_weight)))
            * reliability
            * (0.65 + 0.35 * sample_scale)
            if allow_database
            else 0.0
        )
        delta = np.clip(db_probs - BASELINE, -database_max_adjustment, database_max_adjustment)
        fused = normalize_array(pf + effective * delta, BASELINE)
        composition = _composition(population.particles, self.decks)

        return ReplicaResult(
            pf=pf,
            control=control_probs,
            database=db_probs,
            fused=fused,
            paired_center=paired_center,
            effective_database_weight=float(effective),
            database_reliability=float(reliability),
            database_samples=float(sample_average),
            composition=composition,
            digest=_digest(population.particles, self.seed),
            seed=self.seed,
            mean_depth=float(np.mean(population.depths)),
            min_depth=int(min(population.depths)),
            max_depth=int(max(population.depths)),
            matches=population.matches,
            attempts=population.attempts,
            ess=population.ess,
            acceptance=population.acceptance,
            draw_paths=population.draw_paths.copy(),
            unique_particles=population.unique_particles,
            accepted_unique=population.accepted_unique,
            diversity=population.unique_particles / float(self.particle_count),
            updated=population.updated,
            low_sample=population.low_sample,
            ancestry_paired=population.ancestry_paired,
        )


# ---------------------------------------------------------------------------
# Ensemble decision and public engine
# ---------------------------------------------------------------------------
def _mean_probabilities(rows: Sequence[ReplicaResult], attribute: str) -> np.ndarray:
    matrix = np.stack([np.asarray(getattr(row, attribute), dtype=float) for row in rows], axis=0)
    return normalize_array(matrix.mean(axis=0), BASELINE)


def _basic_decision(fused: np.ndarray, mode: str, commission: float) -> Dict[str, Any]:
    banker_delta = float(fused[0] - BASELINE[0])
    player_delta = float(fused[1] - BASELINE[1])
    centered = banker_delta - player_delta
    banker_ev = float(fused[0] * (1.0 - commission) - fused[1])
    player_ev = float(fused[1] - fused[0])
    if mode == "raw":
        side = "B" if fused[0] >= fused[1] else "P"
        reason = "原始最大機率"
    elif mode == "ev":
        side = "B" if banker_ev >= player_ev else "P"
        reason = "抽水後EV"
    else:
        side = "B" if centered >= 0.0 else "P"
        reason = "相對500萬局基準偏移"
    edge = abs(centered)
    signal = "HIGH" if edge >= 0.010 else "MEDIUM" if edge >= 0.004 else "LOW"
    return {
        "recommend": side,
        "reason": reason,
        "signal_level": signal,
        "edge": edge,
        "center": centered,
        "banker_ev": banker_ev,
        "player_ev": player_ev,
    }


def decide_ensemble(
    fused: np.ndarray,
    replicas: Sequence[ReplicaResult],
    agreement: float,
    average_ess: float,
    average_diversity: float,
    settings: Mapping[str, Any],
) -> Dict[str, Any]:
    mode = str(settings.get("decision_mode", DECISION_MODE)).lower()
    commission = float(settings.get("banker_commission", BANKER_COMMISSION))
    if mode != "validated":
        basic = _basic_decision(fused, mode, commission)
        return {
            **basic,
            "decision_source": "UNVALIDATED_COMPARISON",
            "validated_signal": False,
            "lower_bound": 0.0,
            "model_side": None,
            "raw_center": 0.0,
            "median_center": 0.0,
            "center_std": 0.0,
            "center_se": 0.0,
            "fallback_score": 0.0,
            "fused_center": basic["center"],
            "quality_pass": False,
        }

    centers = np.asarray([row.paired_center for row in replicas], dtype=float)
    mean_center = float(centers.mean())
    median_center = float(np.median(centers))
    center_std = float(centers.std(ddof=1)) if len(centers) > 1 else 0.0
    center_se = center_std / math.sqrt(max(1, len(centers)))
    robust = 0.4 * mean_center + 0.6 * median_center
    penalty = float(settings.get("uncertainty_penalty", UNCERTAINTY_PENALTY))
    lower_bound = max(0.0, abs(robust) - penalty * center_se)
    model_side = "B" if robust >= 0.0 else "P"
    target_ess = float(settings.get("target_ess", TARGET_ESS))
    min_agreement = float(settings.get("min_replica_agreement", MIN_REPLICA_AGREEMENT))
    min_edge = float(settings.get("min_validated_edge", MIN_VALIDATED_EDGE))
    quality_pass = (
        agreement >= min_agreement
        and average_ess >= target_ess * 0.8
        and average_diversity >= 0.45
        and all(row.updated for row in replicas)
        and all(row.ancestry_paired for row in replicas)
    )
    validated = quality_pass and lower_bound >= min_edge

    fused_center = float((fused[0] - BASELINE[0]) - (fused[1] - BASELINE[1]))
    fallback_score = 0.75 * robust + 0.25 * fused_center
    tie_side = "B" if (int(replicas[0].seed) & 1) else "P"
    if fallback_score > 1e-12:
        fallback_side = "B"
    elif fallback_score < -1e-12:
        fallback_side = "P"
    else:
        fallback_side = tie_side
    recommend = model_side if validated else fallback_side
    decision_source = "VALIDATED_MODEL" if validated else "LOW_CONFIDENCE_BALANCED"
    signal = (
        "HIGH"
        if validated and lower_bound >= 0.005
        else "MEDIUM"
        if validated and lower_bound >= 0.002
        else "LOW"
    )
    banker_ev = float(fused[0] * (1.0 - commission) - fused[1])
    player_ev = float(fused[1] - fused[0])
    reason = (
        "祖先配對共同亂數差通過信賴下界與品質閘門"
        if validated
        else "模型訊號未通過驗證，仍以對稱後驗方向輸出；不固定回退莊家"
    )
    return {
        "recommend": recommend,
        "reason": reason,
        "signal_level": signal,
        "edge": lower_bound,
        "center": robust,
        "raw_center": mean_center,
        "median_center": median_center,
        "center_std": center_std,
        "center_se": center_se,
        "lower_bound": lower_bound,
        "model_side": model_side,
        "validated_signal": validated,
        "quality_pass": quality_pass,
        "decision_source": decision_source,
        "banker_ev": banker_ev,
        "player_ev": player_ev,
        "fallback_score": fallback_score,
        "fused_center": fused_center,
    }


class V5IndependentBaccaratEngine:
    """Run the full V5 seven-replica stateless prediction."""

    def __init__(self, settings: Optional[Mapping[str, Any]] = None) -> None:
        supplied = dict(settings or {})
        self.settings: Dict[str, Any] = {
            "decks": int(supplied.get("decks", DECKS)),
            "particles": int(supplied.get("particles", PARTICLE_COUNT)),
            "replicas": int(supplied.get("replicas", REPLICA_COUNT)),
            "target_matches": int(supplied.get("target_matches", TARGET_MATCHES)),
            "target_ess": float(supplied.get("target_ess", TARGET_ESS)),
            "min_matches": int(supplied.get("min_matches", MIN_MATCHES)),
            "max_update_proposals": int(
                supplied.get("max_update_proposals", MAX_UPDATE_PROPOSALS)
            ),
            "predict_simulations_per_replica": int(
                supplied.get(
                    "predict_simulations_per_replica", PREDICT_SIMULATIONS_PER_REPLICA
                )
            ),
            "database_weight": float(supplied.get("database_weight", DATABASE_WEIGHT)),
            "database_max_adjustment": float(
                supplied.get("database_max_adjustment", DATABASE_MAX_ADJUSTMENT)
            ),
            "database_validation_mode": str(
                supplied.get("database_validation_mode", DATABASE_VALIDATION_MODE)
            ).lower(),
            "decision_mode": str(supplied.get("decision_mode", DECISION_MODE)).lower(),
            "uncertainty_penalty": float(
                supplied.get("uncertainty_penalty", UNCERTAINTY_PENALTY)
            ),
            "min_validated_edge": float(
                supplied.get("min_validated_edge", MIN_VALIDATED_EDGE)
            ),
            "min_replica_agreement": float(
                supplied.get("min_replica_agreement", MIN_REPLICA_AGREEMENT)
            ),
            "banker_commission": float(
                supplied.get("banker_commission", BANKER_COMMISSION)
            ),
        }

    def analyze(self, player_total: int, banker_total: int, seed: int) -> Dict[str, Any]:
        replica_results: List[ReplicaResult] = []
        replica_count = max(3, int(self.settings["replicas"]))
        particle_count = max(64, int(self.settings["particles"]))
        for replica_index in range(replica_count):
            replica_seed = mix_seed(seed, replica_index)
            engine = V5ReplicaEngine(
                replica_seed,
                particle_count=particle_count,
                decks=int(self.settings["decks"]),
            )
            prior_particles, prior_depths = engine.build_stratified_prior()
            conditioned = engine.condition(
                prior_particles,
                prior_depths,
                player_total,
                banker_total,
                target_matches=int(self.settings["target_matches"]),
                target_ess=float(self.settings["target_ess"]),
                min_matches=int(self.settings["min_matches"]),
                max_proposals=int(self.settings["max_update_proposals"]),
            )
            result = engine.forecast(
                conditioned,
                simulations=int(self.settings["predict_simulations_per_replica"]),
                database_weight=float(self.settings["database_weight"]),
                database_max_adjustment=float(self.settings["database_max_adjustment"]),
                database_validation_mode=str(self.settings["database_validation_mode"]),
                target_ess=float(self.settings["target_ess"]),
            )
            replica_results.append(result)

        pf = _mean_probabilities(replica_results, "pf")
        control = _mean_probabilities(replica_results, "control")
        database = _mean_probabilities(replica_results, "database")
        fused = _mean_probabilities(replica_results, "fused")
        draw = normalize_array(
            np.stack([row.draw_paths for row in replica_results], axis=0).mean(axis=0),
            DRAW_BASELINE,
        )

        votes = {"B": 0, "P": 0}
        directions: List[str] = []
        for row in replica_results:
            side = "B" if row.paired_center >= 0.0 else "P"
            votes[side] += 1
            directions.append(side)
        agreement = max(votes.values()) / float(len(replica_results))
        average_matches = float(np.mean([row.matches for row in replica_results]))
        average_ess = float(np.mean([row.ess for row in replica_results]))
        average_acceptance = float(np.mean([row.acceptance for row in replica_results]))
        average_attempts = float(np.mean([row.attempts for row in replica_results]))
        average_diversity = float(np.mean([row.diversity for row in replica_results]))
        average_effective = float(
            np.mean([row.effective_database_weight for row in replica_results])
        )
        average_samples = float(np.mean([row.database_samples for row in replica_results]))
        mean_depth = float(np.mean([row.mean_depth for row in replica_results]))
        min_depth = int(min(row.min_depth for row in replica_results))
        max_depth = int(max(row.max_depth for row in replica_results))
        average_cards = float(
            np.mean([row.composition["cards_remaining"] for row in replica_results])
        )
        average_shoe_depth = float(
            np.mean([row.composition["shoe_depth"] for row in replica_results])
        )

        decision = decide_ensemble(
            fused,
            replica_results,
            agreement,
            average_ess,
            average_diversity,
            self.settings,
        )

        if (
            decision["validated_signal"]
            and agreement >= 0.71
            and average_ess >= float(self.settings["target_ess"]) * 0.8
            and average_diversity >= 0.45
        ):
            stability = "STABLE"
        elif (
            agreement >= 0.57
            and average_ess >= float(self.settings["target_ess"]) * 0.45
            and average_diversity >= 0.30
        ):
            stability = "WATCH"
        else:
            stability = "UNSTABLE"

        weakness: List[str] = []
        if not decision["validated_signal"]:
            weakness.append(
                "模型訊號未通過信賴下界或品質閘門，已使用低信心對稱後驗方向，不固定回退莊家"
            )
        if agreement < float(self.settings["min_replica_agreement"]):
            weakness.append("副本配對方向共識低於設定門檻")
        if average_ess < float(self.settings["target_ess"]) * 0.8:
            weakness.append("條件候選有效樣本ESS未達目標80%")
        if average_diversity < 0.45:
            weakness.append("重採樣後粒子多樣性不足45%")
        if float(decision["center_se"]) > max(
            0.0015, abs(float(decision["raw_center"])) * 0.8
        ):
            weakness.append("副本標準誤相對配對訊號偏高")
        if float(decision["lower_bound"]) < float(self.settings["min_validated_edge"]):
            weakness.append("信賴下界未達最低驗證偏移")
        if not DB_HOLDOUT["passed"] and self.settings["database_validation_mode"] != "force":
            weakness.append("500萬資料庫樣本外驗證未優於基準，方向校正已抑制")
        if any(row.low_sample for row in replica_results):
            weakness.append("至少一個副本條件候選數偏少")
        if not all(row.updated for row in replica_results):
            weakness.append("至少一個副本無法建立條件後驗")
        if not weakness:
            weakness.append("配對訊號、信賴下界、ESS、共識與多樣性均通過")

        combined_digest = hashlib.sha1()
        for row in replica_results:
            combined_digest.update(row.digest.encode("ascii"))

        return {
            "pf": pf,
            "control": control,
            "database": database,
            "fused": fused,
            "draw_paths": draw,
            **decision,
            "seed": int(seed) & 0xFFFFFFFF,
            "replicas": len(replica_results),
            "replica_directions": directions,
            "replica_agreement": agreement,
            "votes": votes,
            "stability": stability,
            "weakness_reason": "；".join(weakness),
            "average_matches": average_matches,
            "average_ess": average_ess,
            "average_acceptance": average_acceptance,
            "average_attempts": average_attempts,
            "average_diversity": average_diversity,
            "database_effective_weight": average_effective,
            "database_samples": average_samples,
            "mean_depth": mean_depth,
            "min_depth": min_depth,
            "max_depth": max_depth,
            "cards_remaining": average_cards,
            "shoe_depth": average_shoe_depth,
            "state_digest": combined_digest.hexdigest()[:16],
            "total_forecast_simulations": len(replica_results)
            * int(self.settings["predict_simulations_per_replica"]),
            "total_condition_attempts": int(sum(row.attempts for row in replica_results)),
            "replica_rows": [
                {
                    "index": index + 1,
                    "seed": row.seed,
                    "pf": row.pf.tolist(),
                    "control": row.control.tolist(),
                    "database": row.database.tolist(),
                    "fused": row.fused.tolist(),
                    "paired_center": row.paired_center,
                    "matches": row.matches,
                    "attempts": row.attempts,
                    "ess": row.ess,
                    "acceptance": row.acceptance,
                    "diversity": row.diversity,
                    "updated": row.updated,
                    "low_sample": row.low_sample,
                    "ancestry_paired": row.ancestry_paired,
                    "digest": row.digest,
                }
                for index, row in enumerate(replica_results)
            ],
            "settings": dict(self.settings),
            "database_holdout": dict(DB_HOLDOUT),
            "independent": True,
            "history_used": 0,
            "persistent_state": False,
            "conditional_generator": "EXACT_COMPLETION_IMPORTANCE_WEIGHTED",
            "variance_reduction": "COMMON_RANDOM_NUMBERS_ANTITHETIC",
            "depth_profile": "CALIBRATED_10_30_40_20",
            "deduplicated": False,
            "all_replicas_updated": all(row.updated for row in replica_results),
            "all_ancestry_paired": all(row.ancestry_paired for row in replica_results),
            "fallback_to_unconditioned": not all(row.updated for row in replica_results),
        }
