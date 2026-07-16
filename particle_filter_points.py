"""V5.4 draw-path-fusion point-conditioned baccarat particle engine.

The official LINE predictor uses only the newest final-point observation.
Every request creates fresh conditional candidates, replicas and forecast samples.
A calibrated stratified prior may be reused as an immutable speed cache; no road,
streak, Markov state, previous recommendation or per-UID particle state is carried
into the next request.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
import hashlib
import math
import os
import threading

import numpy as np

from shoe_state_db import DEFAULT_BASELINE, DEFAULT_DRAW, get_shoe_state_database


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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_choice(name: str, default: str, allowed: Sequence[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    return value if value in allowed else default


DECKS = _env_int("PF_DECKS", 8, 1, 16)
PARTICLE_COUNT = _env_int("PF_PARTICLES", 500, 64, 1000)
REPLICA_COUNT = _env_int("PF_REPLICAS", 5, 3, 11)
TARGET_MATCHES = _env_int("PF_TARGET_MATCHES", 320, 32, 4000)
TARGET_ESS = _env_float("PF_TARGET_ESS", 210.0, 8.0, 4000.0)
MIN_MATCHES = _env_int("PF_MIN_MATCHES", 40, 1, 4000)
MAX_UPDATE_PROPOSALS = _env_int("PF_MAX_UPDATE_PROPOSALS", 45_000, 500, 500_000)
PATH_TARGET_MATCHES = _env_int("PF_DRAW_PATH_TARGET_MATCHES", 16, 4, 256)
PATH_MIN_MATCHES = _env_int("PF_DRAW_PATH_MIN_MATCHES", 5, 1, 128)
PATH_MIN_ESS = _env_float("PF_DRAW_PATH_MIN_ESS", 3.0, 1.0, 128.0)
MIN_PATH_COVERAGE = _env_float("PF_MIN_DRAW_PATH_COVERAGE", 0.80, 0.0, 1.0)
PATH_UNCERTAINTY = _env_float("PF_DRAW_PATH_UNCERTAINTY", 0.40, 0.0, 2.0)
PREDICT_SIMS = _env_int("PF_PREDICT_SIMULATIONS_PER_REPLICA", 600, 100, 100_000)
POINT_JOINT_SIMS = _env_int("PF_POINT_JOINT_SIMULATIONS_PER_REPLICA", 600, 100, 100_000)
SPLIT_UNCERTAINTY = _env_float("PF_SPLIT_UNCERTAINTY", 0.50, 0.0, 2.0)
MIN_EFFECTIVE_REPLICAS = _env_float("PF_MIN_EFFECTIVE_REPLICAS", 3.5, 1.0, 11.0)
ADAPTIVE_REPLICA_WEIGHT = _env_bool("PF_ADAPTIVE_REPLICA_WEIGHT", True)
DATABASE_WEIGHT = _env_float("PF_DATABASE_WEIGHT", 0.0, 0.0, 0.75)
DATABASE_MAX_ADJUSTMENT = _env_float("PF_DATABASE_MAX_ADJUSTMENT", 0.005, 0.0, 0.05)
DATABASE_VALIDATION_MODE = _env_choice(
    "PF_DATABASE_VALIDATION_MODE", "diagnostic", ("validated_only", "diagnostic", "force")
)
UNCERTAINTY_PENALTY = _env_float("PF_UNCERTAINTY_PENALTY", 1.28, 0.0, 5.0)
MIN_VALIDATED_EDGE = _env_float("PF_MIN_VALIDATED_EDGE", 0.0012, 0.0, 0.05)
MIN_REPLICA_AGREEMENT = _env_float("PF_MIN_REPLICA_AGREEMENT", 0.71, 0.50, 1.0)
BANKER_COMMISSION = _env_float("PF_BANKER_COMMISSION", 0.05, 0.0, 0.20)
DECISION_MODE = _env_choice("PF_DECISION_MODE", "validated", ("validated", "centered", "raw", "ev"))

# Runtime acceleration. The default fast path preserves the particle/filter
# mathematics while avoiding work that cannot affect the recommendation.
FAST_MODE = _env_bool("PF_FAST_MODE", True)
CACHE_STRATIFIED_PRIORS = _env_bool("PF_CACHE_STRATIFIED_PRIORS", True)
SKIP_UNUSED_DB_DIAGNOSTICS = _env_bool("PF_SKIP_UNUSED_DB_DIAGNOSTICS", True)
FORECAST_SAMPLE_CAP = _env_int("PF_FORECAST_SAMPLE_CAP", 1000, 200, 200_000)
FAST_PARTICLE_CAP = _env_int("PF_FAST_PARTICLE_CAP", 500, 64, 1000)
FAST_TARGET_MATCHES_CAP = _env_int("PF_FAST_TARGET_MATCHES_CAP", 180, 32, 4000)
FAST_TARGET_ESS_CAP = _env_float("PF_FAST_TARGET_ESS_CAP", 120.0, 8.0, 4000.0)
FAST_MAX_UPDATE_PROPOSALS = _env_int(
    "PF_FAST_MAX_UPDATE_PROPOSALS", 10_000, 500, 500_000
)
FAST_PATH_TARGET_MATCHES_CAP = _env_int(
    "PF_FAST_PATH_TARGET_MATCHES_CAP", 10, 4, 256
)

# Three-level decision gate. V5.3 deliberately restores meaningful general
# thresholds: a GENERAL signal must be internally stable, uncertainty-adjusted,
# path-consistent and materially separated from zero. This lowers entry
# frequency instead of relabelling nearly-random output as a usable signal.
GENERAL_REPLICA_AGREEMENT = _env_float("PF_GENERAL_REPLICA_AGREEMENT", 0.70, 0.50, 1.0)
GENERAL_ESS_RATIO = _env_float("PF_GENERAL_ESS_RATIO", 0.65, 0.05, 1.0)
GENERAL_DIVERSITY = _env_float("PF_GENERAL_DIVERSITY", 0.35, 0.05, 1.0)
GENERAL_SPLIT_AGREEMENT = _env_float("PF_GENERAL_SPLIT_AGREEMENT", 0.60, 0.0, 1.0)
GENERAL_PATH_COVERAGE = _env_float("PF_GENERAL_PATH_COVERAGE", 0.70, 0.0, 1.0)
GENERAL_EFFECTIVE_REPLICA_RATIO = _env_float(
    "PF_GENERAL_EFFECTIVE_REPLICA_RATIO", 0.85, 0.10, 1.0
)
GENERAL_UPDATED_RATIO = _env_float("PF_GENERAL_UPDATED_RATIO", 0.80, 0.0, 1.0)
GENERAL_MIN_RAW_EDGE = _env_float("PF_GENERAL_MIN_RAW_EDGE", 0.0030, 0.0, 0.05)
GENERAL_MIN_LOWER_EDGE = _env_float("PF_GENERAL_MIN_LOWER_EDGE", 0.0, 0.0, 0.05)
GENERAL_MIN_CURRENT_PATH_AGREEMENT = _env_float(
    "PF_GENERAL_MIN_CURRENT_PATH_AGREEMENT", 0.55, 0.0, 1.0
)
GENERAL_MIN_NEXT_DRAW_AGREEMENT = _env_float(
    "PF_GENERAL_MIN_NEXT_DRAW_AGREEMENT", 0.55, 0.0, 1.0
)
GENERAL_REQUIRE_DIRECTION_CONSISTENCY = _env_bool(
    "PF_GENERAL_REQUIRE_DIRECTION_CONSISTENCY", True
)

# V5.4 directly fuses the observed-hand draw-path posterior and the simulated
# next-hand draw-path-specific outcome effect into the final direction. These
# are predictive weights, not observe gates.
CURRENT_PATH_SIGNAL_WEIGHT = _env_float("PF_CURRENT_PATH_SIGNAL_WEIGHT", 0.28, 0.0, 0.75)
NEXT_PATH_SIGNAL_WEIGHT = _env_float("PF_NEXT_PATH_SIGNAL_WEIGHT", 0.34, 0.0, 0.75)
BASE_SIGNAL_WEIGHT = _env_float("PF_BASE_SIGNAL_WEIGHT", 0.38, 0.0, 1.0)
PATH_SIGNAL_SHRINK = _env_float("PF_PATH_SIGNAL_SHRINK", 0.72, 0.0, 1.0)
PATH_MIN_SAMPLES_FOR_SIGNAL = _env_int("PF_PATH_MIN_SAMPLES_FOR_SIGNAL", 12, 4, 5000)

_PRIOR_CACHE: Dict[Tuple[int, int, int], Tuple[Tuple[np.ndarray, ...], Tuple[int, ...]]] = {}
_PRIOR_CACHE_LOCK = threading.RLock()

BASELINE = np.asarray(DEFAULT_BASELINE, dtype=float)
DRAW_BASELINE = np.asarray(DEFAULT_DRAW, dtype=float)
DRAW_NAMES = ("none", "player_only", "banker_only", "both")
DB_HOLDOUT: Dict[str, Any] = {
    "passed": False,
    "point_map_rate": 0.4999,
    "baseline_rate": 0.5068,
    "samples": 500_000,
    "note": "樣本外點數映射未優於固定莊家基準，預設抑制資料庫方向校正",
}
CALIBRATED_DEPTH_PROFILE: Tuple[Tuple[int, int, float], ...] = (
    (0, 10, 0.10),
    (11, 25, 0.30),
    (26, 40, 0.40),
    (41, 55, 0.20),
)


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
    current_paths: List[int]
    updated: bool
    low_sample: bool
    matches: int
    attempts: int
    ess: float
    acceptance: float
    draw_paths: np.ndarray
    path_candidate_counts: np.ndarray
    path_ess: np.ndarray
    path_allocated: np.ndarray
    path_coverage: float
    legacy_path_coverage: float
    path_ess_quality: float
    feasible_paths: np.ndarray
    known_path: Optional[int]
    unique_particles: int
    accepted_unique: int
    ancestry_paired: bool


@dataclass
class ReplicaResult:
    pf: np.ndarray
    control: np.ndarray
    split_pf: Tuple[np.ndarray, np.ndarray]
    split_control: Tuple[np.ndarray, np.ndarray]
    database: np.ndarray
    fused: np.ndarray
    next_draw_paths: np.ndarray
    next_path_centers: np.ndarray
    point_matrix: np.ndarray
    top_points: List[Dict[str, Any]]
    paired_center: float
    split_centers: Tuple[float, float]
    split_agreement: float
    split_disagreement: float
    internal_se: float
    current_path_centers: np.ndarray
    current_path_directions: List[str]
    current_path_samples: np.ndarray
    current_path_agreement: float
    current_path_dispersion: float
    current_path_internal_se: float
    draw_agreement: float
    point_concentration: float
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
    path_candidate_counts: np.ndarray
    path_ess: np.ndarray
    path_allocated: np.ndarray
    path_coverage: float
    legacy_path_coverage: float
    path_ess_quality: float
    unique_particles: int
    accepted_unique: int
    diversity: float
    updated: bool
    low_sample: bool
    ancestry_paired: bool
    base_weight: float = 1.0
    robust_factor: float = 1.0
    final_weight: float = 1.0


def mix_seed(seed: int, index: int) -> int:
    x = (int(seed) + ((index + 1) * 0x9E3779B9)) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    x ^= x >> 16
    return x & 0xFFFFFFFF


class UniformStream:
    def __init__(self, seed: int, antithetic: bool = False) -> None:
        self.rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        self.antithetic = antithetic

    def random(self) -> float:
        value = float(self.rng.random())
        if self.antithetic:
            value = 1.0 - value
            if value >= 1.0:
                value = float(np.nextafter(1.0, 0.0))
        return value


def fresh_shoe_counts(decks: int = DECKS) -> np.ndarray:
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
    p1, b1, p2, b2 = (_draw_uniform(counts, random_fn) for _ in range(4))
    player_total, banker_total = (p1 + p2) % 10, (b1 + b2) % 10
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
    path = (1 if player_third is not None else 0) + (2 if banker_third is not None else 0)
    cards = 4 + (1 if player_third is not None else 0) + (1 if banker_third is not None else 0)
    return HandResult(player_total, banker_total, outcome, path, counts, cards)


def simulate_hand_np(source: np.ndarray, rng: np.random.Generator) -> HandResult:
    return simulate_hand_uniform(source, rng.random)


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
    counts = np.asarray(source, dtype=np.int16).copy()
    try:
        p1, b1, p2, b2 = (_draw_np(counts, rng) for _ in range(4))
    except RuntimeError:
        return None
    player_total, banker_total = (p1 + p2) % 10, (b1 + b2) % 10
    player_third: Optional[int] = None
    banker_third: Optional[int] = None
    weight = 1.0
    if player_total >= 8 or banker_total >= 8:
        if player_total != observed_player or banker_total != observed_banker:
            return None
    else:
        if player_total <= 5:
            player_third = (observed_player - player_total) % 10
            probability = _required_card_probability(counts, player_third)
            if probability <= 0:
                return None
            weight *= probability
            counts[player_third] -= 1
            player_total = observed_player
        elif player_total != observed_player:
            return None
        if banker_draws(banker_total, player_third):
            banker_third = (observed_banker - banker_total) % 10
            probability = _required_card_probability(counts, banker_third)
            if probability <= 0:
                return None
            weight *= probability
            counts[banker_third] -= 1
            banker_total = observed_banker
        elif banker_total != observed_banker:
            return None
    if player_total != observed_player or banker_total != observed_banker:
        return None
    path = (1 if player_third is not None else 0) + (2 if banker_third is not None else 0)
    cards = 4 + (1 if player_third is not None else 0) + (1 if banker_third is not None else 0)
    return counts, path, max(1e-12, float(weight)), cards


def feasible_current_paths(player_total: int, banker_total: int, known_path: Optional[int]) -> np.ndarray:
    possible = np.zeros(4, dtype=bool)
    for p0 in range(10):
        for b0 in range(10):
            if p0 >= 8 or b0 >= 8:
                if p0 == player_total and b0 == banker_total:
                    possible[0] = True
                continue
            if p0 <= 5:
                for pc in range(10):
                    if (p0 + pc) % 10 != player_total:
                        continue
                    if banker_draws(b0, pc):
                        for bc in range(10):
                            if (b0 + bc) % 10 == banker_total:
                                possible[3] = True
                    elif b0 == banker_total:
                        possible[1] = True
            elif banker_draws(b0, None):
                if p0 == player_total:
                    for bc in range(10):
                        if (b0 + bc) % 10 == banker_total:
                            possible[2] = True
            elif p0 == player_total and b0 == banker_total:
                possible[0] = True
    if known_path is not None:
        possible = np.asarray([bool(v and i == known_path) for i, v in enumerate(possible)], dtype=bool)
    return possible


def normalize_array(values: Sequence[float], fallback: Sequence[float]) -> np.ndarray:
    arr = np.maximum(0.0, np.asarray(values, dtype=float))
    total = float(arr.sum())
    if total <= 0:
        arr = np.asarray(fallback, dtype=float).copy()
        total = float(arr.sum())
    return arr / total


def weighted_ess(items: Sequence[ConditionalCandidate]) -> float:
    if not items:
        return 0.0
    w = np.asarray([max(0.0, float(x.weight)) for x in items], dtype=float)
    sw = float(w.sum())
    sw2 = float(np.square(w).sum())
    return sw * sw / sw2 if sw2 > 0 else 0.0


def systematic_weighted_resample(
    items: Sequence[ConditionalCandidate], n: int, rng: np.random.Generator
) -> List[ConditionalCandidate]:
    if not items or n <= 0:
        return []
    w = np.asarray([max(0.0, float(x.weight)) for x in items], dtype=float)
    total = float(w.sum())
    if total <= 0:
        return []
    w /= total
    cumulative = np.cumsum(w)
    positions = (float(rng.random()) + np.arange(n, dtype=float)) / float(n)
    idx = np.searchsorted(cumulative, positions, side="left")
    idx = np.clip(idx, 0, len(items) - 1)
    return [items[int(i)] for i in idx]


def allocate_path_particles(masses: np.ndarray, total: int, available: np.ndarray) -> np.ndarray:
    masses = np.asarray(masses, dtype=float)
    available = np.asarray(available, dtype=bool)
    alloc = np.zeros(4, dtype=int)
    raw = np.maximum(0.0, masses) * total
    for i in range(4):
        if available[i]:
            alloc[i] = int(math.floor(raw[i]))
            if masses[i] > 0 and alloc[i] == 0:
                alloc[i] = 1
    while int(alloc.sum()) > total:
        candidates = [i for i in range(4) if alloc[i] > 1]
        if not candidates:
            break
        alloc[max(candidates, key=lambda i: alloc[i])] -= 1
    fractions = sorted(range(4), key=lambda i: raw[i] - math.floor(raw[i]), reverse=True)
    cursor = 0
    while int(alloc.sum()) < total:
        i = fractions[cursor % 4]
        cursor += 1
        if available[i]:
            alloc[i] += 1
    return alloc


def _outcome_index(outcome: str) -> int:
    return 0 if outcome == "B" else 1 if outcome == "P" else 2


def _center(conditioned: Sequence[float], control: Sequence[float]) -> float:
    cp = normalize_array(conditioned, BASELINE)
    up = normalize_array(control, BASELINE)
    return float((cp[0] - up[0]) - (cp[1] - up[1]))


def _composition(particles: Sequence[np.ndarray], decks: int) -> Dict[str, Any]:
    matrix = np.stack([np.asarray(p, dtype=float) for p in particles])
    mean = matrix.mean(axis=0)
    total = float(mean.sum())
    removed = 52 * decks - total
    return {"cards_remaining": total, "shoe_depth": removed / float(52 * decks)}


def _digest(particles: Sequence[np.ndarray], seed: int) -> str:
    h = hashlib.sha1(str(int(seed) & 0xFFFFFFFF).encode("ascii"))
    for particle in particles[:32]:
        h.update(np.asarray(particle, dtype=np.int16).tobytes())
    return h.hexdigest()[:16]


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    total = sum(max(0.0, w) for _, w in pairs)
    if total <= 0:
        return float(np.median(np.asarray(values, dtype=float)))
    acc = 0.0
    for value, weight in pairs:
        acc += max(0.0, weight)
        if acc >= total / 2:
            return float(value)
    return float(pairs[-1][0])


class V5ReplicaEngine:
    def __init__(self, seed: int, particle_count: int, decks: int) -> None:
        self.seed = int(seed) & 0xFFFFFFFF
        self.particle_count = max(64, int(particle_count))
        self.decks = max(1, int(decks))
        self.rng = np.random.default_rng(self.seed)

    def build_stratified_prior(self) -> Tuple[List[np.ndarray], List[int]]:
        allocations = [int(math.floor(self.particle_count * w)) for _, _, w in CALIBRATED_DEPTH_PROFILE]
        index = 0
        while sum(allocations) < self.particle_count:
            allocations[index % len(allocations)] += 1
            index += 1
        particles: List[np.ndarray] = []
        depths: List[int] = []
        for (low, high, _), count in zip(CALIBRATED_DEPTH_PROFILE, allocations):
            span = high - low + 1
            for item_index in range(count):
                counts = fresh_shoe_counts(self.decks)
                q = (item_index + 0.5) / max(1, count)
                base_depth = low + min(span - 1, int(math.floor(q * span)))
                requested = max(low, min(high, base_depth + int(self.rng.integers(-1, 2))))
                completed = 0
                for _ in range(requested):
                    if int(counts.sum()) < 12:
                        break
                    try:
                        counts = simulate_hand_np(counts, self.rng).counts_after
                        completed += 1
                    except RuntimeError:
                        break
                particles.append(counts)
                depths.append(completed)
        return particles, depths

    def condition(
        self,
        prior_particles: Sequence[np.ndarray],
        prior_depths: Sequence[int],
        player_total: int,
        banker_total: int,
        known_path: Optional[int],
        settings: Mapping[str, Any],
    ) -> ConditionedPopulation:
        feasible = feasible_current_paths(player_total, banker_total, known_path)
        by_path: List[List[ConditionalCandidate]] = [[], [], [], []]
        accepted: List[ConditionalCandidate] = []
        attempts = 0
        ess = 0.0
        accepted_weight_sum = 0.0
        accepted_weight_sq_sum = 0.0
        max_proposals = int(settings["max_update_proposals"])
        while attempts < max_proposals:
            parent_index = int(self.rng.integers(0, len(prior_particles)))
            source = prior_particles[parent_index]
            completed = exact_conditional_complete(
                source, self.rng, int(player_total) % 10, int(banker_total) % 10
            )
            attempts += 1
            if completed is not None:
                counts, path, weight, _ = completed
                if feasible[path]:
                    item = ConditionalCandidate(
                        counts=counts,
                        depth=int(prior_depths[parent_index]) + 1,
                        draw_path=path,
                        weight=weight,
                        control_counts=np.asarray(source, dtype=np.int16).copy(),
                        control_depth=int(prior_depths[parent_index]),
                    )
                    accepted.append(item)
                    by_path[path].append(item)
                    accepted_weight_sum += max(0.0, float(weight))
                    accepted_weight_sq_sum += max(0.0, float(weight)) ** 2
            if attempts % 64 == 0:
                ess = (
                    accepted_weight_sum * accepted_weight_sum / accepted_weight_sq_sum
                    if accepted_weight_sq_sum > 0.0
                    else 0.0
                )
                total_ready = (
                    len(accepted) >= int(settings["target_matches"])
                    and ess >= float(settings["target_ess"])
                )
                strata_ready = all(
                    (not feasible[p]) or len(by_path[p]) >= int(settings["path_target_matches"])
                    for p in range(4)
                )
                if total_ready and strata_ready:
                    break
        ess = (
            accepted_weight_sum * accepted_weight_sum / accepted_weight_sq_sum
            if accepted_weight_sq_sum > 0.0
            else 0.0
        )
        path_counts = np.asarray([len(rows) for rows in by_path], dtype=float)
        path_ess = np.asarray([weighted_ess(rows) for rows in by_path], dtype=float)
        path_weight_sums = np.asarray(
            [sum(max(0.0, float(x.weight)) for x in rows) for rows in by_path], dtype=float
        )
        fallback_draw = np.where(feasible, DRAW_BASELINE, 0.0)
        if float(fallback_draw.sum()) <= 0:
            fallback_draw = feasible.astype(float)
        path_posterior = normalize_array(path_weight_sums, fallback_draw)
        if not accepted:
            particles = [np.asarray(x, dtype=np.int16).copy() for x in prior_particles]
            unique = len({x.tobytes() for x in particles})
            return ConditionedPopulation(
                particles=particles,
                depths=[int(v) for v in prior_depths],
                control_particles=[x.copy() for x in particles],
                control_depths=[int(v) for v in prior_depths],
                current_paths=[-1] * len(particles),
                updated=False,
                low_sample=True,
                matches=0,
                attempts=attempts,
                ess=0.0,
                acceptance=0.0,
                draw_paths=normalize_array(fallback_draw, DRAW_BASELINE),
                path_candidate_counts=path_counts,
                path_ess=path_ess,
                path_allocated=np.zeros(4, dtype=int),
                path_coverage=0.0,
                legacy_path_coverage=0.0,
                path_ess_quality=0.0,
                feasible_paths=feasible,
                known_path=known_path,
                unique_particles=unique,
                accepted_unique=0,
                ancestry_paired=False,
            )
        available = path_counts > 0
        allocation = allocate_path_particles(path_posterior, self.particle_count, available)
        particles: List[np.ndarray] = []
        depths: List[int] = []
        controls: List[np.ndarray] = []
        control_depths: List[int] = []
        current_paths: List[int] = []
        for path in range(4):
            rows = systematic_weighted_resample(by_path[path], int(allocation[path]), self.rng)
            for item in rows:
                particles.append(item.counts.copy())
                depths.append(item.depth)
                controls.append(item.control_counts.copy())
                control_depths.append(item.control_depth)
                current_paths.append(path)
        while len(particles) < self.particle_count:
            item = accepted[int(self.rng.integers(0, len(accepted)))]
            particles.append(item.counts.copy())
            depths.append(item.depth)
            controls.append(item.control_counts.copy())
            control_depths.append(item.control_depth)
            current_paths.append(item.draw_path)
        particles = particles[: self.particle_count]
        depths = depths[: self.particle_count]
        controls = controls[: self.particle_count]
        control_depths = control_depths[: self.particle_count]
        current_paths = current_paths[: self.particle_count]
        path_min_matches = int(settings["path_min_matches"])
        path_min_ess = float(settings["path_min_ess"])
        path_target_ess = max(path_min_ess, float(settings["path_target_matches"]) * 0.55)
        path_coverage = float(
            sum(
                path_posterior[p]
                for p in range(4)
                if path_counts[p] >= path_min_matches and path_ess[p] >= path_min_ess
            )
        )
        legacy_path_coverage = float(
            sum(
                path_posterior[p]
                for p in range(4)
                if path_counts[p] >= path_min_matches and path_ess[p] >= 2.0
            )
        )
        path_ess_quality = float(
            sum(path_posterior[p] * min(1.0, path_ess[p] / path_target_ess) for p in range(4))
        )
        significant = path_posterior >= 0.03
        low_path_sample = any(
            significant[p] and (path_counts[p] < path_min_matches or path_ess[p] < path_min_ess)
            for p in range(4)
        )
        unique = len({x.tobytes() for x in particles})
        accepted_unique = len(
            {
                item.counts.tobytes()
                + int(item.depth).to_bytes(2, "little", signed=False)
                + int(item.draw_path).to_bytes(1, "little", signed=False)
                for item in accepted
            }
        )
        return ConditionedPopulation(
            particles=particles,
            depths=depths,
            control_particles=controls,
            control_depths=control_depths,
            current_paths=current_paths,
            updated=True,
            low_sample=len(accepted) < int(settings["min_matches"]) or low_path_sample,
            matches=len(accepted),
            attempts=attempts,
            ess=ess,
            acceptance=len(accepted) / max(1, attempts),
            draw_paths=path_posterior,
            path_candidate_counts=path_counts,
            path_ess=path_ess,
            path_allocated=allocation,
            path_coverage=path_coverage,
            legacy_path_coverage=legacy_path_coverage,
            path_ess_quality=path_ess_quality,
            feasible_paths=feasible,
            known_path=known_path,
            unique_particles=unique,
            accepted_unique=accepted_unique,
            ancestry_paired=True,
        )

    def forecast(self, population: ConditionedPopulation, settings: Mapping[str, Any]) -> ReplicaResult:
        requested_samples = max(
            200,
            int(settings["predict_simulations_per_replica"])
            + int(settings["point_joint_simulations_per_replica"]),
        )
        samples = (
            min(requested_samples, int(settings["forecast_sample_cap"]))
            if bool(settings["fast_mode"])
            else requested_samples
        )
        pairs = int(math.ceil(samples / 2.0))
        conditioned = np.zeros(3, dtype=float)
        control = np.zeros(3, dtype=float)
        conditioned_draw = np.zeros(4, dtype=float)
        control_draw = np.zeros(4, dtype=float)
        path_outcome_c = np.zeros((4, 3), dtype=float)
        path_outcome_u = np.zeros((4, 3), dtype=float)
        current_path_c = np.zeros((4, 3), dtype=float)
        current_path_u = np.zeros((4, 3), dtype=float)
        current_path_samples = np.zeros(4, dtype=float)
        point_matrix = np.zeros(100, dtype=float)
        split_c = np.zeros((2, 3), dtype=float)
        split_u = np.zeros((2, 3), dtype=float)
        particle_count = len(population.particles)
        cycle = 0
        order = np.random.default_rng(mix_seed(self.seed, 880001)).permutation(particle_count)
        for n in range(pairs):
            if n > 0 and n % particle_count == 0:
                cycle += 1
                order = np.random.default_rng(mix_seed(self.seed, 880001 + cycle)).permutation(
                    particle_count
                )
            idx = int(order[n % particle_count])
            pair_seed = mix_seed(self.seed, 700000 + n)
            half = n & 1
            current_path = population.current_paths[idx] if idx < len(population.current_paths) else -1
            for flip in (False, True):
                k = n * 2 + (1 if flip else 0)
                if k >= samples:
                    continue
                c_stream = UniformStream(pair_seed, flip)
                u_stream = UniformStream(pair_seed, flip)
                hc = simulate_hand_uniform(population.particles[idx], c_stream.random)
                hu = simulate_hand_uniform(population.control_particles[idx], u_stream.random)
                ci, ui = _outcome_index(hc.outcome), _outcome_index(hu.outcome)
                conditioned[ci] += 1
                control[ui] += 1
                conditioned_draw[hc.draw_path] += 1
                control_draw[hu.draw_path] += 1
                path_outcome_c[hc.draw_path, ci] += 1
                path_outcome_u[hu.draw_path, ui] += 1
                point_matrix[hc.player_total * 10 + hc.banker_total] += 1
                split_c[half, ci] += 1
                split_u[half, ui] += 1
                if 0 <= current_path < 4:
                    current_path_c[current_path, ci] += 1
                    current_path_u[current_path, ui] += 1
                    current_path_samples[current_path] += 1
        pf = normalize_array(conditioned, BASELINE)
        control_probs = normalize_array(control, BASELINE)
        split_pf = (
            normalize_array(split_c[0], BASELINE),
            normalize_array(split_c[1], BASELINE),
        )
        split_control = (
            normalize_array(split_u[0], BASELINE),
            normalize_array(split_u[1], BASELINE),
        )
        split_centers = (
            _center(split_c[0], split_u[0]),
            _center(split_c[1], split_u[1]),
        )
        split_agreement = 1.0 if (split_centers[0] >= 0) == (split_centers[1] >= 0) else 0.0
        split_disagreement = abs(split_centers[0] - split_centers[1])
        internal_se = 0.5 * split_disagreement
        next_draw = normalize_array(conditioned_draw, DRAW_BASELINE)
        next_path_centers = np.zeros(4, dtype=float)
        for path in range(4):
            path_n = float(path_outcome_c[path].sum())
            if path_n >= float(settings["path_min_samples_for_signal"]):
                next_path_centers[path] = _center(
                    path_outcome_c[path],
                    path_outcome_u[path],
                )
        matrix = point_matrix / max(1.0, float(point_matrix.sum()))
        top_idx = np.argsort(matrix)[::-1][:10]
        top_points = []
        for idx in top_idx:
            p, b = int(idx // 10), int(idx % 10)
            top_points.append(
                {
                    "point": f"{p}{b}",
                    "probability": float(matrix[idx]),
                    "outcome": "B" if b > p else "P" if p > b else "T",
                }
            )
        positive_mass = negative_mass = decisive_mass = 0.0
        for path in range(4):
            c_mass = float(path_outcome_c[path].sum()) / max(1.0, samples)
            u_mass = float(path_outcome_u[path].sum()) / max(1.0, samples)
            mass = 0.5 * (c_mass + u_mass)
            delta = float(
                (path_outcome_c[path, 0] - path_outcome_c[path, 1])
                - (path_outcome_u[path, 0] - path_outcome_u[path, 1])
            ) / max(1.0, samples)
            if abs(delta) <= 1e-12 or mass <= 0:
                continue
            decisive_mass += mass
            if delta > 0:
                positive_mass += mass
            else:
                negative_mass += mass
        draw_agreement = max(positive_mass, negative_mass) / decisive_mass if decisive_mass else 0.5
        paired_center = _center(conditioned, control)
        current_centers = np.zeros(4, dtype=float)
        current_directions = ["—"] * 4
        cp_positive = cp_negative = cp_decisive = 0.0
        disp_num = disp_mass = 0.0
        for path in range(4):
            n = float(current_path_samples[path])
            mass = n / max(1.0, samples)
            if n < 8:
                continue
            value = _center(current_path_c[path], current_path_u[path])
            current_centers[path] = value
            current_directions[path] = "B" if value >= 0 else "P"
            cp_decisive += mass
            if value >= 0:
                cp_positive += mass
            else:
                cp_negative += mass
            disp_num += mass * (value - paired_center) ** 2
            disp_mass += mass
        current_agreement = max(cp_positive, cp_negative) / cp_decisive if cp_decisive else 0.5
        current_dispersion = math.sqrt(disp_num / disp_mass) if disp_mass else 0.0
        effective_paths = max(1, int(np.count_nonzero(current_path_samples >= 8)))
        current_internal_se = current_dispersion / math.sqrt(effective_paths)
        entropy = -float(sum(x * math.log(x) for x in matrix if x > 0))
        entropy_norm = min(1.0, entropy / math.log(100))
        top10_mass = float(matrix[top_idx].sum())
        point_concentration = max(
            0.0,
            min(1.0, 0.55 * (1.0 - entropy_norm) + 0.45 * max(0.0, min(1.0, (top10_mass - 0.10) / 0.22))),
        )
        database = get_shoe_state_database()
        db_allowed = settings["database_validation_mode"] == "force" or (
            settings["database_validation_mode"] == "validated_only" and DB_HOLDOUT["passed"]
        )
        if settings["database_validation_mode"] == "diagnostic":
            db_allowed = False

        # With zero/disabled database weight, the per-particle SQLite lookup
        # cannot change fused probabilities. Skipping it removes hundreds or
        # thousands of unnecessary state-bucket queries per request.
        skip_db_scan = bool(settings["fast_mode"]) and bool(
            settings["skip_unused_db_diagnostics"]
        ) and (not db_allowed or float(settings["database_weight"]) <= 0.0)
        if skip_db_scan:
            db_probs = BASELINE.copy()
            db_rel = 0.0
            db_samples = 0.0
        else:
            db_total = np.zeros(3, dtype=float)
            db_rel = db_samples = 0.0
            inv = 1.0 / max(1, len(population.particles))
            for particle in population.particles:
                estimate = database.estimate(particle, self.decks)
                db_total += inv * np.asarray(
                    [
                        estimate.probabilities["B"],
                        estimate.probabilities["P"],
                        estimate.probabilities["T"],
                    ],
                    dtype=float,
                )
                db_rel += inv * estimate.reliability
                db_samples += inv * estimate.samples
            db_probs = normalize_array(db_total, BASELINE)
        sample_scale = min(1.0, population.ess / max(1.0, float(settings["target_ess"])))
        effective_db = (
            float(settings["database_weight"]) * db_rel * (0.65 + 0.35 * sample_scale)
            if db_allowed
            else 0.0
        )
        delta = np.clip(
            db_probs - BASELINE,
            -float(settings["database_max_adjustment"]),
            float(settings["database_max_adjustment"]),
        )
        fused = normalize_array(pf + effective_db * delta, BASELINE)
        path_quality = 0.55 * population.path_coverage + 0.45 * population.path_ess_quality
        base_weight = (
            max(
                0.05,
                0.12
                + 0.36 * min(1.0, population.ess / max(1.0, float(settings["target_ess"])))
                + 0.20 * min(1.0, population.unique_particles / max(1, self.particle_count))
                + 0.17 * path_quality
                + 0.15 * (1.0 if population.updated and population.ancestry_paired else 0.0),
            )
            if bool(settings["adaptive_replica_weight"])
            else 1.0
        )
        return ReplicaResult(
            pf=pf,
            control=control_probs,
            split_pf=split_pf,
            split_control=split_control,
            database=db_probs,
            fused=fused,
            next_draw_paths=next_draw,
            next_path_centers=next_path_centers,
            point_matrix=matrix,
            top_points=top_points,
            paired_center=paired_center,
            split_centers=split_centers,
            split_agreement=split_agreement,
            split_disagreement=split_disagreement,
            internal_se=internal_se,
            current_path_centers=current_centers,
            current_path_directions=current_directions,
            current_path_samples=current_path_samples,
            current_path_agreement=current_agreement,
            current_path_dispersion=current_dispersion,
            current_path_internal_se=current_internal_se,
            draw_agreement=draw_agreement,
            point_concentration=point_concentration,
            effective_database_weight=float(effective_db),
            database_reliability=float(db_rel),
            database_samples=float(db_samples),
            composition=_composition(population.particles, self.decks),
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
            path_candidate_counts=population.path_candidate_counts.copy(),
            path_ess=population.path_ess.copy(),
            path_allocated=population.path_allocated.copy(),
            path_coverage=population.path_coverage,
            legacy_path_coverage=population.legacy_path_coverage,
            path_ess_quality=population.path_ess_quality,
            unique_particles=population.unique_particles,
            accepted_unique=population.accepted_unique,
            diversity=population.unique_particles / max(1.0, self.particle_count),
            updated=population.updated,
            low_sample=population.low_sample,
            ancestry_paired=population.ancestry_paired,
            base_weight=base_weight,
            final_weight=base_weight,
        )


def _get_stratified_prior(
    particle_count: int,
    decks: int,
    replica_index: int,
    request_seed: int,
    use_cache: bool,
) -> Tuple[Sequence[np.ndarray], Sequence[int]]:
    """Return an immutable calibrated prior, optionally shared across requests.

    Conditioning always copies/removes cards from candidate arrays, so cached
    prior arrays are read-only inputs and are safe to reuse. Request-specific
    randomness remains in conditional completion and forecast sampling.
    """
    if not use_cache:
        builder = V5ReplicaEngine(request_seed, particle_count, decks)
        return builder.build_stratified_prior()

    key = (int(particle_count), int(decks), int(replica_index))
    with _PRIOR_CACHE_LOCK:
        cached = _PRIOR_CACHE.get(key)
        if cached is None:
            prior_seed = mix_seed(0x51A7E5D3 ^ (particle_count << 8) ^ decks, replica_index)
            builder = V5ReplicaEngine(prior_seed, particle_count, decks)
            particles, depths = builder.build_stratified_prior()
            cached = (
                tuple(np.asarray(item, dtype=np.int16) for item in particles),
                tuple(int(value) for value in depths),
            )
            _PRIOR_CACHE[key] = cached
        return cached


def clear_runtime_caches() -> int:
    """Clear reusable calibrated priors and return the number removed."""
    with _PRIOR_CACHE_LOCK:
        removed = len(_PRIOR_CACHE)
        _PRIOR_CACHE.clear()
    return removed


def _weighted_average(rows: Sequence[ReplicaResult], attr: str, size: int) -> np.ndarray:
    weights = np.asarray([max(1e-6, row.final_weight) for row in rows], dtype=float)
    weights /= float(weights.sum())
    out = np.zeros(size, dtype=float)
    for row, w in zip(rows, weights):
        out += w * np.asarray(getattr(row, attr), dtype=float)
    fallback = BASELINE if size == 3 else DRAW_BASELINE if size == 4 else np.full(size, 1.0 / size)
    return normalize_array(out, fallback)


def _apply_robust_weights(rows: Sequence[ReplicaResult], enabled: bool) -> Tuple[float, int]:
    centers = np.asarray([x.paired_center for x in rows], dtype=float)
    med = float(np.median(centers))
    mad = float(np.median(np.abs(centers - med)))
    scale = max(1e-6, 1.4826 * mad)
    outliers = 0
    for row in rows:
        if enabled and mad >= 1e-10:
            z = abs(row.paired_center - med) / (2.5 * scale)
            factor = max(0.12, min(1.0, 1.0 / (1.0 + z * z)))
        else:
            factor = 1.0
        row.robust_factor = factor
        row.final_weight = max(0.01, row.base_weight * factor) if enabled else 1.0
        if factor < 0.5:
            outliers += 1
    return mad, outliers


def _basic_decision(fused: np.ndarray, mode: str, commission: float) -> Dict[str, Any]:
    centered = float((fused[0] - BASELINE[0]) - (fused[1] - BASELINE[1]))
    banker_ev = float(fused[0] * (1.0 - commission) - fused[1])
    player_ev = float(fused[1] - fused[0])
    if mode == "raw":
        side, reason = ("B" if fused[0] >= fused[1] else "P"), "原始最大機率"
    elif mode == "ev":
        side, reason = ("B" if banker_ev >= player_ev else "P"), "抽水後EV"
    else:
        side, reason = ("B" if centered >= 0 else "P"), "相對基準偏移"
    edge = abs(centered)
    return {
        "recommend": side,
        "reason": reason,
        "signal_level": "HIGH" if edge >= 0.010 else "MEDIUM" if edge >= 0.004 else "LOW",
        "edge": edge,
        "center": centered,
        "raw_center": centered,
        "median_center": centered,
        "center_std": 0.0,
        "center_se": 0.0,
        "base_se": 0.0,
        "split_se": 0.0,
        "path_se": 0.0,
        "lower_bound": 0.0,
        "model_side": side,
        "validated_signal": False,
        "quality_pass": False,
        "general_quality_pass": True,
        "decision_tier": "GENERAL",
        "is_observe": False,
        "decision_source": "UNVALIDATED_COMPARISON",
        "banker_ev": banker_ev,
        "player_ev": player_ev,
        "fallback_score": centered,
        "effective_replicas": 1.0,
        "direction_consistency": True,
        "updated_ratio": 1.0,
    }


def decide_ensemble(
    fused: np.ndarray,
    control: np.ndarray,
    rows: Sequence[ReplicaResult],
    quality: Mapping[str, float],
    settings: Mapping[str, Any],
) -> Dict[str, Any]:
    mode = str(settings["decision_mode"])
    commission = float(settings["banker_commission"])
    if mode != "validated":
        return _basic_decision(fused, mode, commission)

    # Fuse three complementary signals per replica:
    # 1) overall conditioned-vs-control effect,
    # 2) posterior-weighted effect of how the observed hand was completed,
    # 3) probability-weighted effect of the next hand's four draw paths.
    # Sparse path effects are shrunk rather than used as a hard observe gate.
    base_w = float(settings["base_signal_weight"])
    current_w = float(settings["current_path_signal_weight"])
    next_w = float(settings["next_path_signal_weight"])
    total_w = max(1e-12, base_w + current_w + next_w)
    shrink = float(settings["path_signal_shrink"])
    centers = []
    for row in rows:
        current_signal = float(np.dot(row.draw_paths, row.current_path_centers)) * shrink
        next_signal = float(np.dot(row.next_draw_paths, row.next_path_centers)) * shrink
        fused_center = (
            base_w * row.paired_center
            + current_w * current_signal
            + next_w * next_signal
        ) / total_w
        centers.append(fused_center)
    weights = [max(1e-6, row.final_weight) for row in rows]
    sw = sum(weights)
    sw2 = sum(w * w for w in weights)
    effective_replicas = max(1.0, sw * sw / max(1e-12, sw2))
    mean = sum(c * w for c, w in zip(centers, weights)) / sw
    med = _weighted_median(centers, weights)
    numerator = sum(w * (c - mean) ** 2 for c, w in zip(centers, weights))
    denominator = max(1e-12, sw - sw2 / sw)
    variance = numerator / denominator if len(rows) > 1 else 0.0
    std = math.sqrt(max(0.0, variance))
    base_se = std / math.sqrt(effective_replicas)
    internal_rms = math.sqrt(sum(w * row.internal_se**2 for row, w in zip(rows, weights)) / sw)
    split_se = float(settings["split_uncertainty"]) * internal_rms / math.sqrt(effective_replicas)
    path_rms = math.sqrt(
        sum(w * row.current_path_internal_se**2 for row, w in zip(rows, weights)) / sw
    )
    path_se = float(settings["path_uncertainty"]) * path_rms / math.sqrt(effective_replicas)
    se = math.sqrt(base_se**2 + split_se**2 + path_se**2)
    robust = 0.50 * mean + 0.50 * med
    lower = max(0.0, abs(robust) - float(settings["uncertainty_penalty"]) * se)
    model_side = "B" if robust >= 0 else "P"
    global_center = _center(fused, control)
    direction_consistency = abs(global_center) < 1e-12 or (global_center >= 0) == (robust >= 0)
    updated_ratio = sum(
        1 for row in rows if row.updated and row.ancestry_paired
    ) / max(1, len(rows))

    # Strict validation keeps the original gate and therefore remains the only
    # tier labelled VALIDATED_MODEL.
    strict_quality_pass = (
        quality["agreement"] >= float(settings["min_replica_agreement"])
        and quality["average_ess"] >= float(settings["target_ess"]) * 0.8
        and quality["average_diversity"] >= 0.45
        and quality["split_agreement"] >= 0.60
        and quality["path_coverage"] >= float(settings["min_path_coverage"])
        and effective_replicas >= float(settings["min_effective_replicas"])
        and direction_consistency
        and updated_ratio >= 0.999
    )
    validated = strict_quality_pass and lower >= float(settings["min_validated_edge"])

    # General validation accepts ordinary-quality runs without pretending they
    # passed the strict statistical gate. Thresholds are configurable and still
    # reject failed conditioning, extremely low diversity and severe replica
    # disagreement.
    general_effective_min = max(
        1.0,
        float(settings["min_effective_replicas"])
        * float(settings["general_effective_replica_ratio"]),
    )
    general_quality_pass = (
        quality["agreement"] >= float(settings["general_replica_agreement"])
        and quality["average_ess"]
        >= float(settings["target_ess"]) * float(settings["general_ess_ratio"])
        and quality["average_diversity"] >= float(settings["general_diversity"])
        and quality["split_agreement"] >= float(settings["general_split_agreement"])
        and quality["path_coverage"] >= float(settings["general_path_coverage"])
        and effective_replicas >= general_effective_min
        and updated_ratio >= float(settings["general_updated_ratio"])
        and abs(robust) >= float(settings["general_min_raw_edge"])
        and lower >= float(settings["general_min_lower_edge"])
        and quality.get("current_path_agreement", 0.0)
        >= float(settings["general_min_current_path_agreement"])
        and quality.get("next_draw_agreement", 0.0)
        >= float(settings["general_min_next_draw_agreement"])
        and (
            direction_consistency
            or not bool(settings["general_require_direction_consistency"])
        )
    )

    if validated:
        recommend = model_side
        decision_source = "VALIDATED_MODEL"
        decision_tier = "STRICT"
        signal = "HIGH" if lower >= 0.005 else "MEDIUM"
        edge = lower
        reason = "補牌路徑分層、統一配對樣本池與穩健誤差修正後通過正式品質閘門"
    elif general_quality_pass:
        recommend = model_side
        decision_source = "LOW_CONFIDENCE_BALANCED"
        decision_tier = "GENERAL"
        signal = "LOW"
        edge = abs(robust)
        reason = "通過一般品質閘門；保留方向但未標示為正式驗證訊號"
    else:
        # V5.4 always returns the path-fused direction. Quality remains visible
        # as FALLBACK, but path disagreement no longer suppresses the result.
        recommend = model_side
        decision_source = "DRAW_PATH_FUSION"
        decision_tier = "FALLBACK"
        signal = "LOW"
        edge = abs(robust)
        reason = "未通過正式品質閘門；改用當局與下一局四種補牌路徑融合方向"

    return {
        "recommend": recommend,
        "reason": reason,
        "signal_level": signal,
        "edge": edge,
        "center": robust,
        "raw_center": mean,
        "median_center": med,
        "center_std": std,
        "center_se": se,
        "base_se": base_se,
        "split_se": split_se,
        "path_se": path_se,
        "lower_bound": lower,
        "model_side": model_side,
        "validated_signal": validated,
        "quality_pass": strict_quality_pass,
        "general_quality_pass": general_quality_pass,
        "decision_tier": decision_tier,
        "is_observe": False,
        "decision_source": decision_source,
        "banker_ev": float(fused[0] * (1.0 - commission) - fused[1]),
        "player_ev": float(fused[1] - fused[0]),
        "fallback_score": robust,
        "effective_replicas": effective_replicas,
        "direction_consistency": direction_consistency,
        "updated_ratio": updated_ratio,
    }


class V5IndependentBaccaratEngine:
    def __init__(self, settings: Optional[Mapping[str, Any]] = None) -> None:
        supplied = dict(settings or {})
        self.settings: Dict[str, Any] = {
            "decks": int(supplied.get("decks", DECKS)),
            "particles": int(supplied.get("particles", PARTICLE_COUNT)),
            "replicas": int(supplied.get("replicas", REPLICA_COUNT)),
            "target_matches": int(supplied.get("target_matches", TARGET_MATCHES)),
            "target_ess": float(supplied.get("target_ess", TARGET_ESS)),
            "min_matches": int(supplied.get("min_matches", MIN_MATCHES)),
            "max_update_proposals": int(supplied.get("max_update_proposals", MAX_UPDATE_PROPOSALS)),
            "path_target_matches": int(supplied.get("path_target_matches", PATH_TARGET_MATCHES)),
            "path_min_matches": int(supplied.get("path_min_matches", PATH_MIN_MATCHES)),
            "path_min_ess": float(supplied.get("path_min_ess", PATH_MIN_ESS)),
            "min_path_coverage": float(supplied.get("min_path_coverage", MIN_PATH_COVERAGE)),
            "path_uncertainty": float(supplied.get("path_uncertainty", PATH_UNCERTAINTY)),
            "predict_simulations_per_replica": int(
                supplied.get("predict_simulations_per_replica", PREDICT_SIMS)
            ),
            "point_joint_simulations_per_replica": int(
                supplied.get("point_joint_simulations_per_replica", POINT_JOINT_SIMS)
            ),
            "split_uncertainty": float(supplied.get("split_uncertainty", SPLIT_UNCERTAINTY)),
            "min_effective_replicas": float(
                supplied.get("min_effective_replicas", MIN_EFFECTIVE_REPLICAS)
            ),
            "adaptive_replica_weight": bool(
                supplied.get("adaptive_replica_weight", ADAPTIVE_REPLICA_WEIGHT)
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
            "banker_commission": float(supplied.get("banker_commission", BANKER_COMMISSION)),
            "fast_mode": bool(supplied.get("fast_mode", FAST_MODE)),
            "cache_stratified_priors": bool(
                supplied.get("cache_stratified_priors", CACHE_STRATIFIED_PRIORS)
            ),
            "skip_unused_db_diagnostics": bool(
                supplied.get("skip_unused_db_diagnostics", SKIP_UNUSED_DB_DIAGNOSTICS)
            ),
            "forecast_sample_cap": int(
                supplied.get("forecast_sample_cap", FORECAST_SAMPLE_CAP)
            ),
            "fast_particle_cap": int(
                supplied.get("fast_particle_cap", FAST_PARTICLE_CAP)
            ),
            "fast_target_matches_cap": int(
                supplied.get("fast_target_matches_cap", FAST_TARGET_MATCHES_CAP)
            ),
            "fast_target_ess_cap": float(
                supplied.get("fast_target_ess_cap", FAST_TARGET_ESS_CAP)
            ),
            "fast_max_update_proposals": int(
                supplied.get("fast_max_update_proposals", FAST_MAX_UPDATE_PROPOSALS)
            ),
            "fast_path_target_matches_cap": int(
                supplied.get(
                    "fast_path_target_matches_cap",
                    FAST_PATH_TARGET_MATCHES_CAP,
                )
            ),
            "general_replica_agreement": float(
                supplied.get("general_replica_agreement", GENERAL_REPLICA_AGREEMENT)
            ),
            "general_ess_ratio": float(supplied.get("general_ess_ratio", GENERAL_ESS_RATIO)),
            "general_diversity": float(
                supplied.get("general_diversity", GENERAL_DIVERSITY)
            ),
            "general_split_agreement": float(
                supplied.get("general_split_agreement", GENERAL_SPLIT_AGREEMENT)
            ),
            "general_path_coverage": float(
                supplied.get("general_path_coverage", GENERAL_PATH_COVERAGE)
            ),
            "general_effective_replica_ratio": float(
                supplied.get(
                    "general_effective_replica_ratio",
                    GENERAL_EFFECTIVE_REPLICA_RATIO,
                )
            ),
            "general_updated_ratio": float(
                supplied.get("general_updated_ratio", GENERAL_UPDATED_RATIO)
            ),
            "general_min_raw_edge": float(
                supplied.get("general_min_raw_edge", GENERAL_MIN_RAW_EDGE)
            ),
            "general_min_lower_edge": float(
                supplied.get("general_min_lower_edge", GENERAL_MIN_LOWER_EDGE)
            ),
            "general_min_current_path_agreement": float(
                supplied.get(
                    "general_min_current_path_agreement",
                    GENERAL_MIN_CURRENT_PATH_AGREEMENT,
                )
            ),
            "general_min_next_draw_agreement": float(
                supplied.get(
                    "general_min_next_draw_agreement",
                    GENERAL_MIN_NEXT_DRAW_AGREEMENT,
                )
            ),
            "general_require_direction_consistency": bool(
                supplied.get(
                    "general_require_direction_consistency",
                    GENERAL_REQUIRE_DIRECTION_CONSISTENCY,
                )
            ),
            "current_path_signal_weight": float(
                supplied.get("current_path_signal_weight", CURRENT_PATH_SIGNAL_WEIGHT)
            ),
            "next_path_signal_weight": float(
                supplied.get("next_path_signal_weight", NEXT_PATH_SIGNAL_WEIGHT)
            ),
            "base_signal_weight": float(
                supplied.get("base_signal_weight", BASE_SIGNAL_WEIGHT)
            ),
            "path_signal_shrink": float(
                supplied.get("path_signal_shrink", PATH_SIGNAL_SHRINK)
            ),
            "path_min_samples_for_signal": int(
                supplied.get("path_min_samples_for_signal", PATH_MIN_SAMPLES_FOR_SIGNAL)
            ),
        }

        if bool(self.settings["fast_mode"]):
            self.settings["particles"] = min(
                int(self.settings["particles"]),
                int(self.settings["fast_particle_cap"]),
            )
            self.settings["target_matches"] = min(
                int(self.settings["target_matches"]),
                int(self.settings["fast_target_matches_cap"]),
            )
            self.settings["target_ess"] = min(
                float(self.settings["target_ess"]),
                float(self.settings["fast_target_ess_cap"]),
            )
            self.settings["max_update_proposals"] = min(
                int(self.settings["max_update_proposals"]),
                int(self.settings["fast_max_update_proposals"]),
            )
            self.settings["path_target_matches"] = min(
                int(self.settings["path_target_matches"]),
                int(self.settings["fast_path_target_matches_cap"]),
            )

    def analyze(
        self,
        player_total: int,
        banker_total: int,
        seed: int,
        known_path: Optional[int] = None,
    ) -> Dict[str, Any]:
        rows: List[ReplicaResult] = []
        for replica_index in range(max(3, int(self.settings["replicas"]))):
            replica_seed = mix_seed(seed, replica_index)
            engine = V5ReplicaEngine(
                replica_seed,
                particle_count=max(64, int(self.settings["particles"])),
                decks=int(self.settings["decks"]),
            )
            prior_particles, prior_depths = _get_stratified_prior(
                particle_count=max(64, int(self.settings["particles"])),
                decks=int(self.settings["decks"]),
                replica_index=replica_index,
                request_seed=replica_seed,
                use_cache=bool(self.settings["cache_stratified_priors"]),
            )
            population = engine.condition(
                prior_particles,
                prior_depths,
                int(player_total) % 10,
                int(banker_total) % 10,
                known_path,
                self.settings,
            )
            rows.append(engine.forecast(population, self.settings))
        robust_mad, outlier_count = _apply_robust_weights(
            rows, bool(self.settings["adaptive_replica_weight"])
        )
        pf = _weighted_average(rows, "pf", 3)
        control = _weighted_average(rows, "control", 3)
        database = _weighted_average(rows, "database", 3)
        fused = _weighted_average(rows, "fused", 3)
        draw = _weighted_average(rows, "draw_paths", 4)
        next_draw = _weighted_average(rows, "next_draw_paths", 4)
        point_matrix = _weighted_average(rows, "point_matrix", 100)
        top_idx = np.argsort(point_matrix)[::-1][:10]
        top_points = [
            {
                "point": f"{int(i // 10)}{int(i % 10)}",
                "probability": float(point_matrix[i]),
                "outcome": "B" if int(i % 10) > int(i // 10) else "P" if int(i // 10) > int(i % 10) else "T",
            }
            for i in top_idx
        ]
        weight_sum = sum(max(1e-6, row.final_weight) for row in rows)
        weighted_votes = {"B": 0.0, "P": 0.0}
        votes = {"B": 0, "P": 0}
        directions: List[str] = []
        for row in rows:
            side = "B" if row.paired_center >= 0 else "P"
            weighted_votes[side] += max(1e-6, row.final_weight)
            votes[side] += 1
            directions.append(side)
        agreement = max(weighted_votes.values()) / max(1e-12, weight_sum)
        split_agreement = sum(
            max(1e-6, row.final_weight) * row.split_agreement for row in rows
        ) / max(1e-12, weight_sum)
        average_ess = float(np.mean([row.ess for row in rows]))
        average_diversity = float(np.mean([row.diversity for row in rows]))
        average_path_coverage = float(np.mean([row.path_coverage for row in rows]))
        average_current_path_agreement = float(
            np.mean([row.current_path_agreement for row in rows])
        )
        average_draw_agreement = float(np.mean([row.draw_agreement for row in rows]))
        quality = {
            "agreement": agreement,
            "average_ess": average_ess,
            "average_diversity": average_diversity,
            "split_agreement": split_agreement,
            "path_coverage": average_path_coverage,
            "current_path_agreement": average_current_path_agreement,
            "next_draw_agreement": average_draw_agreement,
        }
        decision = decide_ensemble(fused, control, rows, quality, self.settings)
        if decision["decision_tier"] == "STRICT":
            stability = "STABLE"
        elif decision["decision_tier"] == "GENERAL":
            stability = "WATCH"
        else:
            stability = "UNSTABLE"

        weakness: List[str] = []
        if decision["decision_tier"] == "GENERAL":
            weakness.append("已通過一般品質閘門，但未通過正式信賴下界")
        elif decision["decision_tier"] == "OBSERVE":
            weakness.append("模型品質不足，本局觀望且不計入方向戰績")
        if agreement < float(self.settings["min_replica_agreement"]):
            weakness.append("副本穩健方向共識未達正式門檻")
        if split_agreement < 0.60:
            weakness.append("統一樣本池分半方向一致率未達正式60%門檻")
        if average_ess < float(self.settings["target_ess"]) * 0.8:
            weakness.append("條件候選ESS未達正式目標80%")
        if average_diversity < 0.45:
            weakness.append("粒子多樣性未達正式45%門檻")
        if average_path_coverage < float(self.settings["min_path_coverage"]):
            weakness.append("補牌路徑有效覆蓋率未達正式門檻")
        if outlier_count:
            weakness.append(f"已抑制{outlier_count}個偏離中位數副本")
        if any(row.low_sample for row in rows):
            weakness.append("至少一個副本的總候選或補牌路徑候選偏少")
        if not DB_HOLDOUT["passed"] and self.settings["database_validation_mode"] != "force":
            weakness.append("500萬資料庫樣本外驗證未優於基準，方向校正已抑制")
        if not weakness:
            weakness.append("500粒子、補牌路徑ESS、統一樣本池與穩健誤差均通過")
        combined = hashlib.sha1()
        for row in rows:
            combined.update(row.digest.encode("ascii"))
        return {
            "pf": pf,
            "control": control,
            "database": database,
            "fused": fused,
            "draw_paths": draw,
            "next_draw_paths": next_draw,
            "point_matrix": point_matrix,
            "top_points": top_points,
            **decision,
            "seed": int(seed) & 0xFFFFFFFF,
            "replicas": len(rows),
            "replica_directions": directions,
            "replica_agreement": agreement,
            "votes": votes,
            "weighted_votes": weighted_votes,
            "split_agreement": split_agreement,
            "stability": stability,
            "weakness_reason": "；".join(weakness),
            "average_matches": float(np.mean([row.matches for row in rows])),
            "average_ess": average_ess,
            "average_acceptance": float(np.mean([row.acceptance for row in rows])),
            "average_attempts": float(np.mean([row.attempts for row in rows])),
            "average_diversity": average_diversity,
            "average_path_coverage": average_path_coverage,
            "average_legacy_path_coverage": float(
                np.mean([row.legacy_path_coverage for row in rows])
            ),
            "average_path_ess_quality": float(np.mean([row.path_ess_quality for row in rows])),
            "average_current_path_agreement": average_current_path_agreement,
            "average_draw_agreement": average_draw_agreement,
            "average_point_concentration": float(
                np.mean([row.point_concentration for row in rows])
            ),
            "average_path_candidates": np.mean(
                np.stack([row.path_candidate_counts for row in rows]), axis=0, dtype=float
            ),
            "average_path_ess": np.mean(np.stack([row.path_ess for row in rows]), axis=0),
            "average_path_allocated": np.mean(
                np.stack([row.path_allocated for row in rows]), axis=0
            ),
            "average_current_path_centers": np.mean(
                np.stack([row.current_path_centers for row in rows]), axis=0
            ),
            "average_database_weight": float(
                np.mean([row.effective_database_weight for row in rows])
            ),
            "database_samples": float(np.mean([row.database_samples for row in rows])),
            "mean_depth": float(np.mean([row.mean_depth for row in rows])),
            "min_depth": int(min(row.min_depth for row in rows)),
            "max_depth": int(max(row.max_depth for row in rows)),
            "cards_remaining": float(
                np.mean([row.composition["cards_remaining"] for row in rows])
            ),
            "shoe_depth": float(np.mean([row.composition["shoe_depth"] for row in rows])),
            "state_digest": combined.hexdigest()[:16],
            "robust_mad": robust_mad,
            "outlier_count": outlier_count,
            "total_forecast_simulations": int(
                len(rows)
                * (
                    min(
                        int(self.settings["predict_simulations_per_replica"])
                        + int(self.settings["point_joint_simulations_per_replica"]),
                        int(self.settings["forecast_sample_cap"]),
                    )
                    if bool(self.settings["fast_mode"])
                    else int(self.settings["predict_simulations_per_replica"])
                    + int(self.settings["point_joint_simulations_per_replica"])
                )
            ),
            "total_condition_attempts": int(sum(row.attempts for row in rows)),
            "all_ancestry_paired": all(row.ancestry_paired for row in rows),
            "all_replicas_updated": all(row.updated for row in rows),
            "fallback_to_unconditioned": any(not row.updated for row in rows),
            "conditional_generator": "DRAW_PATH_STRATIFIED_EXACT_COMPLETION_IMPORTANCE_WEIGHTED",
            "variance_reduction": "UNIFIED_BALANCED_PARTICLE_COMMON_RANDOM_ANTITHETIC",
            "depth_profile": "CALIBRATED_10_30_40_20",
            "settings": dict(self.settings),
            "replica_rows": rows,
        }
