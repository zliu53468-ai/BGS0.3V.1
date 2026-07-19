"""V5.4.1 aggressive-quality-gated draw-path HYBRID baccarat particle engine.

Every request creates fresh particles, fresh replicas and fresh forecast samples.
Only the latest factual hand is used: current points, optional N/P/B/D draw path,
hand number and exact cards explicitly supplied by the caller.

No road, streak, learned Markov state, previous recommendation, win/loss result,
or per-UID particle direction is carried into the next request.  A small fixed
single-step draw-path prior may use the current hand path only; it is quality
gated and adds no simulation loop. High-quality path agreement may lower the
validated-edge threshold and raise the dynamic particle fusion cap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
import hashlib
import math
import os

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


MIN_PARTICLE_COUNT = 64
MAX_PARTICLE_COUNT = 2000
DECKS = _env_int("PF_DECKS", 8, 1, 16)

# 384 x 5 is the default real-time profile. Operators can still raise either
# value through environment variables after measuring their Render CPU.
PARTICLE_COUNT = _env_int("PF_PARTICLES", 384, MIN_PARTICLE_COUNT, MAX_PARTICLE_COUNT)
REPLICA_COUNT = _env_int("PF_REPLICAS", 5, 3, 11)
TARGET_MATCHES = _env_int("PF_TARGET_MATCHES", 230, 32, 8000)
TARGET_ESS = _env_float("PF_TARGET_ESS", 146.0, 8.0, 8000.0)
MIN_MATCHES = _env_int("PF_MIN_MATCHES", 31, 1, 8000)
MAX_UPDATE_PROPOSALS = _env_int("PF_MAX_UPDATE_PROPOSALS", 56_000, 500, 500_000)
PATH_TARGET_MATCHES = _env_int("PF_DRAW_PATH_TARGET_MATCHES", 10, 4, 512)
PATH_MIN_MATCHES = _env_int("PF_DRAW_PATH_MIN_MATCHES", 4, 1, 256)
PATH_MIN_ESS = _env_float("PF_DRAW_PATH_MIN_ESS", 2.5, 1.0, 256.0)
MIN_PATH_COVERAGE = _env_float("PF_MIN_DRAW_PATH_COVERAGE", 0.70, 0.0, 1.0)
PATH_UNCERTAINTY = _env_float("PF_DRAW_PATH_UNCERTAINTY", 0.35, 0.0, 2.0)
PATH_PRIOR_STRENGTH = _env_float("PF_DRAW_PATH_PRIOR_STRENGTH", 1.5, 0.0, 50.0)
MIN_PATH_PARTICLES = _env_int("PF_MIN_PARTICLES_PER_DRAW_PATH", 12, 1, 512)
PREDICT_SIMS = _env_int("PF_PREDICT_SIMULATIONS_PER_REPLICA", 200, 100, 100_000)
POINT_JOINT_SIMS = _env_int("PF_POINT_JOINT_SIMULATIONS_PER_REPLICA", 200, 100, 100_000)
SPLIT_UNCERTAINTY = _env_float("PF_SPLIT_UNCERTAINTY", 0.45, 0.0, 2.0)
MIN_EFFECTIVE_REPLICAS = _env_float("PF_MIN_EFFECTIVE_REPLICAS", 3.3, 1.0, 11.0)
ADAPTIVE_REPLICA_WEIGHT = _env_bool("PF_ADAPTIVE_REPLICA_WEIGHT", True)
DATABASE_WEIGHT = _env_float("PF_DATABASE_WEIGHT", 0.0, 0.0, 0.75)
DATABASE_MAX_ADJUSTMENT = _env_float("PF_DATABASE_MAX_ADJUSTMENT", 0.005, 0.0, 0.05)
DATABASE_VALIDATION_MODE = _env_choice(
    "PF_DATABASE_VALIDATION_MODE", "diagnostic", ("validated_only", "diagnostic", "force")
)
UNCERTAINTY_PENALTY = _env_float("PF_UNCERTAINTY_PENALTY", 1.20, 0.0, 5.0)
MIN_VALIDATED_EDGE = _env_float("PF_MIN_VALIDATED_EDGE", 0.0012, 0.0, 0.05)
MIN_REPLICA_AGREEMENT = _env_float("PF_MIN_REPLICA_AGREEMENT", 0.68, 0.50, 1.0)
BANKER_COMMISSION = _env_float("PF_BANKER_COMMISSION", 0.05, 0.0, 0.20)
DECISION_MODE = _env_choice("PF_DECISION_MODE", "validated", ("validated", "centered", "raw", "ev"))

# HYBRID combines only independent factual sources. No road or streak features
# are accepted by this engine.
HYBRID_MODE = _env_choice("PF_HYBRID_MODE", "hybrid", ("hybrid", "particle"))
HYBRID_PARTICLE_MAX_WEIGHT = _env_float("PF_HYBRID_PARTICLE_MAX_WEIGHT", 0.82, 0.0, 1.0)
HYBRID_EXACT_STATE_MAX_WEIGHT = _env_float(
    "PF_HYBRID_EXACT_STATE_MAX_WEIGHT", 0.24, 0.0, 0.60
)
HYBRID_BASELINE_MIN_WEIGHT = _env_float(
    "PF_HYBRID_BASELINE_MIN_WEIGHT", 0.12, 0.0, 0.90
)
HYBRID_MAX_COMPONENT_ADJUSTMENT = _env_float(
    "PF_HYBRID_MAX_COMPONENT_ADJUSTMENT", 0.020, 0.0, 0.10
)
HAND_NUMBER_UNCERTAINTY = _env_int("PF_HAND_NUMBER_UNCERTAINTY", 0, 0, 5)
STATE_SIMULATIONS = _env_int("PF_STATE_SIMULATIONS", 1200, 200, 100_000)

# Draw-path quality controls.  These defaults are intentionally conservative and
# do not change particle count, proposal count, replica count or forecast loops.
THIRD_CARD_LIKELIHOOD_POWER = _env_float(
    "PF_THIRD_CARD_LIKELIHOOD_POWER", 0.92, 0.70, 1.0
)
NO_DRAW_ALLOCATION_PROTECTION = _env_float(
    "PF_NO_DRAW_ALLOCATION_PROTECTION", 1.20, 1.0, 2.0
)
PATH_TRANSITION_PRIOR_WEIGHT = _env_float(
    "PF_PATH_TRANSITION_PRIOR_WEIGHT", 0.16, 0.0, 0.40
)
PATH_TRANSITION_SELF_BIAS = _env_float(
    "PF_PATH_TRANSITION_SELF_BIAS", 0.10, 0.0, 0.30
)
HYBRID_HIGH_PATH_THRESHOLD = _env_float(
    "PF_HYBRID_HIGH_PATH_THRESHOLD", 0.75, 0.55, 0.95
)
HYBRID_HIGH_PATH_WEIGHT_BOOST = _env_float(
    "PF_HYBRID_HIGH_PATH_WEIGHT_BOOST", 0.07, 0.0, 0.18
)
HIGH_QUALITY_EDGE_RELIEF = _env_float(
    "PF_HIGH_QUALITY_EDGE_RELIEF", 0.40, 0.0, 0.60
)
ADVANTAGE_CONTINUITY_BONUS = _env_float(
    "PF_ADVANTAGE_CONTINUITY_BONUS", 0.22, 0.0, 0.50
)

# Independent draw-path head. It reuses the existing forecast samples and adds
# no extra particle, replica or Monte Carlo loops. The head is residualized
# against the ordinary particle forecast before it can affect final probabilities.
INDEPENDENT_PATH_MODEL_ENABLED = _env_bool(
    "PF_INDEPENDENT_PATH_MODEL_ENABLED", True
)
INDEPENDENT_PATH_MODEL_MAX_WEIGHT = _env_float(
    "PF_INDEPENDENT_PATH_MODEL_MAX_WEIGHT", 0.18, 0.0, 0.40
)
INDEPENDENT_PATH_MODEL_MIN_RELIABILITY = _env_float(
    "PF_INDEPENDENT_PATH_MODEL_MIN_RELIABILITY", 0.55, 0.0, 1.0
)
INDEPENDENT_PATH_MODEL_MAX_ADJUSTMENT = _env_float(
    "PF_INDEPENDENT_PATH_MODEL_MAX_ADJUSTMENT", 0.012, 0.0, 0.05
)
INDEPENDENT_PATH_MODEL_PRIOR_STRENGTH = _env_float(
    "PF_INDEPENDENT_PATH_MODEL_PRIOR_STRENGTH", 6.0, 0.0, 100.0
)
INDEPENDENT_PATH_MODEL_PRIOR_MIN = _env_float(
    "PF_INDEPENDENT_PATH_MODEL_PRIOR_MIN", 2.0, 0.0, 100.0
)
INDEPENDENT_PATH_MODEL_PRIOR_MAX = _env_float(
    "PF_INDEPENDENT_PATH_MODEL_PRIOR_MAX", 18.0, 0.0, 100.0
)
INDEPENDENT_PATH_MODEL_PRIOR_DEPTH_RELIEF = _env_float(
    "PF_INDEPENDENT_PATH_MODEL_PRIOR_DEPTH_RELIEF", 0.35, 0.0, 0.90
)
INDEPENDENT_PATH_MODEL_PRIOR_LOW_SUPPORT_GAIN = _env_float(
    "PF_INDEPENDENT_PATH_MODEL_PRIOR_LOW_SUPPORT_GAIN", 1.10, 0.0, 5.0
)
INDEPENDENT_PATH_MODEL_PRIOR_IMBALANCE_GAIN = _env_float(
    "PF_INDEPENDENT_PATH_MODEL_PRIOR_IMBALANCE_GAIN", 0.65, 0.0, 5.0
)
INDEPENDENT_PATH_MODEL_SUPPORT_TARGET_FRACTION = _env_float(
    "PF_INDEPENDENT_PATH_MODEL_SUPPORT_TARGET_FRACTION", 0.35, 0.05, 2.0
)
INDEPENDENT_PATH_MODEL_SUPPORT_MIN_COUNT = _env_float(
    "PF_INDEPENDENT_PATH_MODEL_SUPPORT_MIN_COUNT", 6.0, 1.0, 200.0
)
INDEPENDENT_PATH_MODEL_RESIDUAL_SHRINK_POWER = _env_float(
    "PF_INDEPENDENT_PATH_MODEL_RESIDUAL_SHRINK_POWER", 0.85, 0.10, 3.0
)
INDEPENDENT_PATH_MODEL_RESIDUAL_Z_SCORE = _env_float(
    "PF_INDEPENDENT_PATH_MODEL_RESIDUAL_Z_SCORE", 1.0, 0.0, 5.0
)
INDEPENDENT_PATH_MODEL_DEPTH_START = _env_float(
    "PF_INDEPENDENT_PATH_MODEL_DEPTH_START", 0.28, 0.0, 0.95
)
INDEPENDENT_PATH_MODEL_DEPTH_FULL = _env_float(
    "PF_INDEPENDENT_PATH_MODEL_DEPTH_FULL", 0.68, 0.05, 1.0
)
INDEPENDENT_PATH_MODEL_LATE_MAX_WEIGHT = _env_float(
    "PF_INDEPENDENT_PATH_MODEL_LATE_MAX_WEIGHT", 0.24, 0.0, 0.40
)

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


def _scaled_runtime_settings(settings: Mapping[str, Any], particle_count: int) -> Dict[str, Any]:
    """Return effective settings after scaling sample-quality targets by particles."""
    out = dict(settings)
    particles = max(MIN_PARTICLE_COUNT, min(MAX_PARTICLE_COUNT, int(particle_count)))
    out["particles"] = particles
    out["target_matches"] = max(int(out["target_matches"]), int(math.ceil(particles * 0.60)))
    out["target_ess"] = max(float(out["target_ess"]), particles * 0.38)
    out["min_matches"] = max(int(out["min_matches"]), int(math.ceil(particles * 0.08)))
    out["path_target_matches"] = max(
        int(out["path_target_matches"]), int(math.ceil(particles * 0.024))
    )
    out["max_update_proposals"] = min(
        500_000,
        max(int(out["max_update_proposals"]), int(out["target_matches"]) * 240),
    )
    return out


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
    next_draw_paths_raw: np.ndarray
    path_transition_prior: np.ndarray
    path_transition_weight: float
    path_transition_quality: float
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
    path_quality: float
    path_fusion_gain: float
    independent_path_probabilities: np.ndarray
    independent_path_control_probabilities: np.ndarray
    independent_path_next_draw: np.ndarray
    independent_path_outcome_matrix: np.ndarray
    independent_path_control_outcome_matrix: np.ndarray
    independent_path_support: np.ndarray
    independent_path_reliability: float
    independent_path_effective_weight: float
    independent_path_direction_agreement: float
    independent_path_weighted_support: float
    independent_path_residual_shrinkage: float
    independent_path_depth_factor: float
    independent_path_depth_consistency: float
    independent_path_dynamic_max_weight: float
    independent_path_dynamic_prior_strength: np.ndarray
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


def _resolve_observed_completion(
    player_initial: int,
    banker_initial: int,
    observed_player: int,
    observed_banker: int,
) -> Optional[Tuple[int, Optional[int], Optional[int], int]]:
    """Resolve the unique legal completion from two-card totals to final totals.

    This helper is shared by feasibility checks and conditional completion so the
    path rules cannot drift apart.  Path 0 is accepted only for a natural or for
    the explicit 6/7 versus 6/7 stand branch.
    """
    p0 = int(player_initial) % 10
    b0 = int(banker_initial) % 10
    final_p = int(observed_player) % 10
    final_b = int(observed_banker) % 10

    natural = p0 >= 8 or b0 >= 8
    if natural:
        if p0 == final_p and b0 == final_b:
            return 0, None, None, 4
        return None

    player_third: Optional[int] = None
    banker_third: Optional[int] = None
    if p0 <= 5:
        player_third = (final_p - p0) % 10
    elif p0 != final_p:
        return None

    if banker_draws(b0, player_third):
        banker_third = (final_b - b0) % 10
    elif b0 != final_b:
        return None

    path = (1 if player_third is not None else 0) + (
        2 if banker_third is not None else 0
    )
    if path == 0 and not (p0 in {6, 7} and b0 in {6, 7}):
        return None
    cards = 4 + int(player_third is not None) + int(banker_third is not None)
    return path, player_third, banker_third, cards


def exact_conditional_complete(
    source: np.ndarray,
    rng: np.random.Generator,
    observed_player: int,
    observed_banker: int,
    known_path: Optional[int] = None,
    likelihood_power: Optional[float] = None,
) -> Optional[Tuple[np.ndarray, int, float, int]]:
    """Complete one proposal under exact baccarat draw rules.

    Initial four cards are sampled physically.  Any required third card is then
    solved from the observed final total and scored by its sequential shoe
    probability.  A mild likelihood tempering raises the effective contribution
    of valid one-card/two-card completions without creating extra proposals.
    """
    counts = np.asarray(source, dtype=np.int16).copy()
    try:
        p1, b1, p2, b2 = (_draw_np(counts, rng) for _ in range(4))
    except RuntimeError:
        return None

    completion = _resolve_observed_completion(
        (p1 + p2) % 10,
        (b1 + b2) % 10,
        int(observed_player) % 10,
        int(observed_banker) % 10,
    )
    if completion is None:
        return None
    path, player_third, banker_third, cards = completion
    if known_path is not None and path != int(known_path):
        return None

    raw_likelihood = 1.0
    for required_card in (player_third, banker_third):
        if required_card is None:
            continue
        probability = _required_card_probability(counts, int(required_card))
        if probability <= 0.0:
            return None
        raw_likelihood *= probability
        counts[int(required_card)] -= 1

    draw_count = int(player_third is not None) + int(banker_third is not None)
    power = max(
        0.70,
        min(
            1.0,
            float(
                THIRD_CARD_LIKELIHOOD_POWER
                if likelihood_power is None
                else likelihood_power
            ),
        ),
    )
    weight = raw_likelihood if draw_count == 0 else raw_likelihood ** power
    return counts, path, max(1e-12, float(weight)), cards



def normalize_known_cards(value: Any) -> Optional[Dict[str, List[int]]]:
    """Normalize an exact current hand entered as Player/Banker card values."""
    if not isinstance(value, Mapping):
        return None
    out: Dict[str, List[int]] = {}
    aliases = {
        "player": ("player", "P", "閒", "闲"),
        "banker": ("banker", "B", "莊", "庄"),
    }
    for side, names in aliases.items():
        raw = None
        for name in names:
            if name in value:
                raw = value.get(name)
                break
        if raw is None:
            return None
        try:
            cards = [int(card) % 10 for card in list(raw)]
        except Exception:
            return None
        if len(cards) not in {2, 3}:
            return None
        out[side] = cards
    return out


def validate_known_hand(
    observed_player: int,
    observed_banker: int,
    known_path: Optional[int],
    known_cards: Mapping[str, Sequence[int]],
) -> Tuple[bool, int, str]:
    """Validate exact cards against baccarat third-card rules and final totals."""
    normalized = normalize_known_cards(known_cards)
    if normalized is None:
        return False, -1, "missing_or_invalid_cards"

    player = normalized["player"]
    banker = normalized["banker"]
    p0 = (player[0] + player[1]) % 10
    b0 = (banker[0] + banker[1]) % 10

    natural = p0 >= 8 or b0 >= 8
    if natural:
        player_draw = False
        banker_draw = False
    else:
        player_draw = p0 <= 5
        player_third = player[2] if player_draw and len(player) == 3 else None
        banker_draw = banker_draws(b0, player_third)

    expected_player_len = 3 if player_draw else 2
    expected_banker_len = 3 if banker_draw else 2
    if len(player) != expected_player_len or len(banker) != expected_banker_len:
        return False, -1, "cards_conflict_with_draw_rules"

    final_player = sum(player) % 10
    final_banker = sum(banker) % 10
    if final_player != int(observed_player) % 10 or final_banker != int(observed_banker) % 10:
        return False, -1, "cards_conflict_with_final_points"

    path = (1 if player_draw else 0) + (2 if banker_draw else 0)
    if known_path is not None and int(known_path) != path:
        return False, path, "cards_conflict_with_path_suffix"
    return True, path, "ok"


def exact_known_hand_complete(
    source: np.ndarray,
    observed_player: int,
    observed_banker: int,
    known_path: Optional[int],
    known_cards: Mapping[str, Sequence[int]],
) -> Optional[Tuple[np.ndarray, int, float, int]]:
    """Condition a prior particle on an exact entered hand.

    The likelihood is computed in physical dealing order, then the known cards
    are removed from the selected prior shoe. This supplies genuinely new card
    composition information without using road or result history.
    """
    normalized = normalize_known_cards(known_cards)
    if normalized is None:
        return None
    valid, path, _ = validate_known_hand(
        observed_player,
        observed_banker,
        known_path,
        normalized,
    )
    if not valid:
        return None

    player = normalized["player"]
    banker = normalized["banker"]
    sequence: List[int] = [player[0], banker[0], player[1], banker[1]]
    if len(player) == 3:
        sequence.append(player[2])
    if len(banker) == 3:
        sequence.append(banker[2])

    counts = np.asarray(source, dtype=np.int16).copy()
    weight = 1.0
    for value in sequence:
        probability = _required_card_probability(counts, int(value) % 10)
        if probability <= 0:
            return None
        weight *= probability
        counts[int(value) % 10] -= 1

    return counts, path, max(1e-12, float(weight)), len(sequence)


def _valid_remaining_counts(value: Any, decks: int) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(list(value), dtype=np.int16)
    except Exception:
        return None
    if arr.shape != (10,) or np.any(arr < 0):
        return None
    maximum = fresh_shoe_counts(decks)
    if np.any(arr > maximum) or int(arr.sum()) < 6:
        return None
    return arr.copy()


def estimate_exact_state_probabilities(
    remaining_counts: Sequence[int],
    decks: int,
    seed: int,
    samples: int,
) -> np.ndarray:
    """Monte Carlo next-hand probabilities from an exactly tracked shoe."""
    counts = _valid_remaining_counts(remaining_counts, decks)
    if counts is None:
        return BASELINE.copy()
    total = np.zeros(3, dtype=float)
    pair_count = int(math.ceil(max(200, int(samples)) / 2.0))
    for pair_index in range(pair_count):
        pair_seed = mix_seed(seed, 910000 + pair_index)
        for antithetic in (False, True):
            stream = UniformStream(pair_seed, antithetic)
            hand = simulate_hand_uniform(counts, stream.random)
            total[_outcome_index(hand.outcome)] += 1.0
    return normalize_array(total, BASELINE)

def feasible_current_paths(
    player_total: int,
    banker_total: int,
    known_path: Optional[int],
) -> np.ndarray:
    """Return paths that can exactly produce the observed final totals.

    The same completion resolver used by ``exact_conditional_complete`` checks
    every legal two-card total pair.  This makes natural/stand path 0 explicit
    and prevents feasibility and importance sampling from disagreeing.
    """
    possible = np.zeros(4, dtype=bool)
    try:
        final_player = int(player_total)
        final_banker = int(banker_total)
    except (TypeError, ValueError):
        return possible
    if not (0 <= final_player <= 9 and 0 <= final_banker <= 9):
        return possible

    path_filter: Optional[int] = None
    if known_path is not None:
        try:
            path_filter = int(known_path)
        except (TypeError, ValueError):
            return possible
        if path_filter not in {0, 1, 2, 3}:
            return possible

    for player_initial in range(10):
        for banker_initial in range(10):
            completion = _resolve_observed_completion(
                player_initial,
                banker_initial,
                final_player,
                final_banker,
            )
            if completion is not None:
                possible[int(completion[0])] = True

    if path_filter is not None:
        possible &= np.arange(4) == path_filter
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


def allocate_path_particles(
    masses: np.ndarray,
    total: int,
    available: np.ndarray,
    minimum_per_available: int = 1,
    support_quality: Optional[np.ndarray] = None,
    no_draw_protection: Optional[float] = None,
) -> np.ndarray:
    """Allocate particles by posterior mass, legality and path protection.

    Path 0 receives an explicit floor/priority guard because exact no-draw
    completions can be structurally narrow.  One-side and both-draw paths retain
    smaller rarity guards so valid low-mass paths are not erased by resampling.
    The total particle budget is unchanged.
    """
    masses = np.maximum(0.0, np.asarray(masses, dtype=float))
    available = np.asarray(available, dtype=bool)
    alloc = np.zeros(4, dtype=int)
    eligible = [index for index in range(4) if available[index]]
    total = max(0, int(total))
    if total <= 0 or not eligible:
        return alloc

    masked = np.where(available, masses, 0.0)
    if float(masked.sum()) <= 0:
        masked = available.astype(float)
    posterior = masked / float(masked.sum())

    if support_quality is None:
        support = available.astype(float)
    else:
        support = np.clip(np.asarray(support_quality, dtype=float), 0.0, 1.0)
        support = np.where(available, support, 0.0)

    full_floor = min(
        max(1, int(minimum_per_available)),
        max(1, total // len(eligible)),
    )
    rare_floor = min(full_floor, max(1, full_floor // 3))
    no_draw_floor = min(
        full_floor,
        max(rare_floor, int(math.ceil(full_floor * 0.75))),
    )
    one_side_floor = min(
        full_floor,
        max(rare_floor, int(math.ceil(full_floor * 0.50))),
    )
    both_floor = min(
        full_floor,
        max(rare_floor, int(math.ceil(full_floor * 0.60))),
    )
    protected_floors = np.asarray(
        [no_draw_floor, one_side_floor, one_side_floor, both_floor],
        dtype=int,
    )

    significant_mass = max(0.015, 0.5 / max(1.0, float(total)))
    for index in eligible:
        if posterior[index] >= significant_mass or support[index] >= 0.50:
            alloc[index] = full_floor
        else:
            alloc[index] = int(protected_floors[index])

    configured_no_draw_protection = max(
        1.0,
        min(
            2.0,
            float(
                NO_DRAW_ALLOCATION_PROTECTION
                if no_draw_protection is None
                else no_draw_protection
            ),
        ),
    )
    protection = np.asarray(
        [configured_no_draw_protection, 1.08, 1.08, 1.14],
        dtype=float,
    )
    priority = posterior * (0.62 + 0.38 * support) * protection
    while int(alloc.sum()) > total:
        reducible = [index for index in eligible if alloc[index] > 1]
        if not reducible:
            break
        index = min(reducible, key=lambda item: (priority[item], alloc[item]))
        alloc[index] -= 1

    remaining = total - int(alloc.sum())
    if remaining <= 0:
        return alloc

    tempered = normalize_array(np.sqrt(posterior), available.astype(float))
    supported = normalize_array(
        posterior * (0.45 + 0.55 * support),
        posterior,
    )
    protected = normalize_array(
        posterior * protection * (0.55 + 0.45 * support),
        posterior,
    )
    target = normalize_array(
        0.58 * posterior
        + 0.17 * tempered
        + 0.10 * supported
        + 0.15 * protected,
        posterior,
    )

    raw = target * remaining
    base = np.floor(raw).astype(int)
    alloc += base
    left = total - int(alloc.sum())
    order = sorted(
        eligible,
        key=lambda index: (
            raw[index] - base[index],
            target[index],
            priority[index],
        ),
        reverse=True,
    )
    for cursor in range(left):
        alloc[order[cursor % len(order)]] += 1
    return alloc


def apply_current_path_transition_prior(
    raw_next_draw: Sequence[float],
    current_draw_paths: Sequence[float],
    known_path: Optional[int],
    path_coverage: float,
    path_ess_quality: float,
    settings: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Apply a stronger but still bounded current-hand-only path carry prior.

    This is not a learned transition model and does not read road/history. It
    blends one fixed one-step prior using only the current hand path plus the
    existing path coverage/ESS diagnostics, so the extra influence adds no
    particle, proposal, replica or forecast loop.
    """
    raw = normalize_array(raw_next_draw, DRAW_BASELINE)
    if known_path is not None and int(known_path) in {0, 1, 2, 3}:
        current = np.zeros(4, dtype=float)
        current[int(known_path)] = 1.0
        current_certainty = 1.0
    else:
        current = normalize_array(current_draw_paths, DRAW_BASELINE)
        current_certainty = max(0.0, min(1.0, float(np.max(current))))

    self_bias = max(
        0.0,
        min(0.30, float(settings.get("path_transition_self_bias", 0.10))),
    )
    transition = np.zeros((4, 4), dtype=float)
    for path in range(4):
        row = DRAW_BASELINE.astype(float).copy()
        row[path] += self_bias * 1.20
        if path == 0:
            row[0] += self_bias * 0.70
        elif path == 1:
            row[2] += self_bias * 0.24
            row[3] += self_bias * 0.08
        elif path == 2:
            row[1] += self_bias * 0.24
            row[3] += self_bias * 0.08
        else:
            row[3] += self_bias * 0.50
            row[1] += self_bias * 0.10
            row[2] += self_bias * 0.10
        transition[path] = normalize_array(row, DRAW_BASELINE)

    prior = normalize_array(current @ transition, DRAW_BASELINE)
    path_quality = max(
        0.0,
        min(
            1.0,
            0.45 * float(path_coverage)
            + 0.35 * float(path_ess_quality)
            + 0.20 * current_certainty,
        ),
    )
    transition_quality = max(
        0.0,
        min(1.0, path_quality * (0.75 + 0.25 * current_certainty)),
    )
    configured_weight = max(
        0.0,
        min(0.40, float(settings.get("path_transition_prior_weight", 0.16))),
    )
    certainty_multiplier = 0.92 + 0.08 * current_certainty
    effective_weight = min(
        configured_weight,
        configured_weight * transition_quality * certainty_multiplier,
    )
    adjusted = normalize_array(
        (1.0 - effective_weight) * raw + effective_weight * prior,
        DRAW_BASELINE,
    )
    return adjusted, prior, float(effective_weight), float(transition_quality)



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


def build_independent_draw_path_model(
    conditioned_draw: Sequence[float],
    control_draw: Sequence[float],
    path_outcome_conditioned: np.ndarray,
    path_outcome_control: np.ndarray,
    path_coverage: float,
    path_ess_quality: float,
    current_path_agreement: float,
    draw_agreement: float,
    settings: Mapping[str, Any],
    *,
    particle_probabilities: Optional[Sequence[float]] = None,
    path_candidate_counts: Optional[Sequence[float]] = None,
    path_ess: Optional[Sequence[float]] = None,
    path_allocated: Optional[Sequence[float]] = None,
    population_ess: float = 0.0,
    target_ess: float = 1.0,
    mean_depth: float = 0.0,
    min_depth: int = 0,
    max_depth: int = 0,
    shoe_depth: Optional[float] = None,
    decks: int = DECKS,
) -> Dict[str, Any]:
    """Build a depth-adaptive four-path conditional outcome head.

    The head still reuses only samples already produced by ``forecast``.  It
    adds no particle, replica or Monte Carlo pass.  Compared with the original
    implementation, path priors, support, residual shrinkage and maximum weight
    are adapted to paired sample quality and factual shoe depth.
    """

    def clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def scale_quality(
        value: float,
        low: float = 0.50,
        high: float = 0.85,
    ) -> float:
        if high <= low:
            return 0.0
        return clamp01((float(value) - low) / (high - low))

    def path_vector(value: Optional[Sequence[float]]) -> np.ndarray:
        if value is None:
            return np.zeros(4, dtype=float)
        try:
            arr = np.asarray(value, dtype=float)
        except Exception:
            return np.zeros(4, dtype=float)
        if arr.shape != (4,):
            return np.zeros(4, dtype=float)
        return np.maximum(0.0, arr)

    conditioned_rows = np.asarray(path_outcome_conditioned, dtype=float)
    control_rows = np.asarray(path_outcome_control, dtype=float)
    if conditioned_rows.shape != (4, 3):
        conditioned_rows = np.zeros((4, 3), dtype=float)
    if control_rows.shape != (4, 3):
        control_rows = np.zeros((4, 3), dtype=float)
    conditioned_rows = np.maximum(0.0, conditioned_rows)
    control_rows = np.maximum(0.0, control_rows)

    path_quality = clamp01(
        0.55 * float(path_coverage)
        + 0.45 * float(path_ess_quality)
    )
    population_ess_quality = clamp01(
        float(population_ess) / max(1.0, float(target_ess))
    )

    deck_count = max(1, int(decks))
    maximum_hand_depth = 90.0
    shoe_depth_ratio = clamp01(
        float(shoe_depth)
        if shoe_depth is not None
        else float(mean_depth) / maximum_hand_depth
    )
    depth_start = clamp01(
        float(settings.get("independent_path_model_depth_start", 0.28))
    )
    depth_full = clamp01(
        float(settings.get("independent_path_model_depth_full", 0.68))
    )
    if depth_full <= depth_start:
        depth_full = min(1.0, depth_start + 0.05)
    depth_position = clamp01(
        (shoe_depth_ratio - depth_start) / max(1e-12, depth_full - depth_start)
    )
    depth_factor = depth_position * depth_position * (3.0 - 2.0 * depth_position)
    depth_span = max(0.0, float(max_depth) - float(min_depth))
    depth_consistency = clamp01(
        1.0 - depth_span / max(1.0, maximum_hand_depth * 0.35)
    )
    depth_effect = depth_factor * (0.55 + 0.45 * depth_consistency)

    conditioned_path = normalize_array(conditioned_draw, DRAW_BASELINE)
    control_path = normalize_array(control_draw, DRAW_BASELINE)
    conditioned_blend = min(
        0.84,
        0.70 + 0.12 * depth_effect * path_quality,
    )
    control_blend = max(
        0.10,
        0.20 - 0.06 * depth_effect * path_quality,
    )
    baseline_blend = max(
        0.05,
        1.0 - conditioned_blend - control_blend,
    )
    blend = normalize_array(
        [conditioned_blend, control_blend, baseline_blend],
        [0.70, 0.20, 0.10],
    )
    predicted_path = normalize_array(
        blend[0] * conditioned_path
        + blend[1] * control_path
        + blend[2] * DRAW_BASELINE,
        DRAW_BASELINE,
    )

    candidate_counts = path_vector(path_candidate_counts)
    path_ess_values = path_vector(path_ess)
    allocated = path_vector(path_allocated)
    metadata_available = bool(
        np.any(candidate_counts > 0)
        or np.any(path_ess_values > 0)
        or np.any(allocated > 0)
    )

    conditioned_matrix = np.zeros((4, 3), dtype=float)
    control_matrix = np.zeros((4, 3), dtype=float)
    support = np.zeros(4, dtype=float)
    support_targets = np.zeros(4, dtype=float)
    balance_quality = np.zeros(4, dtype=float)
    signal_quality = np.zeros(4, dtype=float)
    path_centers = np.zeros(4, dtype=float)
    dynamic_prior = np.zeros(4, dtype=float)

    conditioned_total = float(conditioned_rows.sum())
    control_total = float(control_rows.sum())
    average_total = 0.5 * (conditioned_total + control_total)
    support_target_fraction = max(
        0.01,
        float(
            settings.get(
                "independent_path_model_support_target_fraction",
                0.35,
            )
        ),
    )
    support_min_count = max(
        1.0,
        float(
            settings.get(
                "independent_path_model_support_min_count",
                6.0,
            )
        ),
    )
    path_target_matches = max(
        1.0,
        float(settings.get("path_target_matches", 10.0)),
    )
    path_min_ess = max(
        1.0,
        float(settings.get("path_min_ess", 2.5)),
    )
    min_path_particles = max(
        1.0,
        float(settings.get("min_path_particles", 12.0)),
    )

    base_prior = max(
        0.0,
        float(settings.get("independent_path_model_prior_strength", 6.0)),
    )
    prior_min = max(
        0.0,
        float(settings.get("independent_path_model_prior_min", 2.0)),
    )
    prior_max = max(
        prior_min,
        float(settings.get("independent_path_model_prior_max", 18.0)),
    )
    prior_depth_relief = clamp01(
        float(
            settings.get(
                "independent_path_model_prior_depth_relief",
                0.35,
            )
        )
    )
    prior_low_support_gain = max(
        0.0,
        float(
            settings.get(
                "independent_path_model_prior_low_support_gain",
                1.10,
            )
        ),
    )
    prior_imbalance_gain = max(
        0.0,
        float(
            settings.get(
                "independent_path_model_prior_imbalance_gain",
                0.65,
            )
        ),
    )

    for path in range(4):
        c_count = float(conditioned_rows[path].sum())
        u_count = float(control_rows[path].sum())
        paired_total = c_count + u_count
        paired_effective = (
            2.0 * c_count * u_count / paired_total
            if paired_total > 1e-12
            else 0.0
        )
        balance = (
            2.0 * min(c_count, u_count) / paired_total
            if paired_total > 1e-12
            else 0.0
        )
        balance_quality[path] = clamp01(balance)

        expected_count = average_total * max(
            0.02,
            float(predicted_path[path]),
        )
        support_target = max(
            support_min_count,
            support_target_fraction * expected_count,
        )
        support_targets[path] = support_target
        count_quality = 1.0 - math.exp(
            -paired_effective / max(1e-12, support_target)
        )

        if metadata_available:
            candidate_quality = 1.0 - math.exp(
                -candidate_counts[path] / path_target_matches
            )
            ess_quality = clamp01(
                path_ess_values[path] / path_min_ess
            )
            allocation_quality = clamp01(
                allocated[path] / min_path_particles
            )
            structural_quality = clamp01(
                0.35 * candidate_quality
                + 0.40 * ess_quality
                + 0.25 * allocation_quality
            )
        else:
            structural_quality = clamp01(path_ess_quality)

        support[path] = clamp01(
            count_quality
            * (
                0.50
                + 0.30 * balance_quality[path]
                + 0.20 * structural_quality
            )
            * (
                0.72
                + 0.18 * clamp01(path_coverage)
                + 0.10 * population_ess_quality
            )
        )

        depth_prior_multiplier = max(
            0.10,
            1.0 - prior_depth_relief * depth_effect,
        )
        prior_value = base_prior * depth_prior_multiplier
        prior_value *= (
            1.0
            + prior_low_support_gain * (1.0 - support[path])
            + prior_imbalance_gain * (1.0 - balance_quality[path])
        )
        dynamic_prior[path] = max(
            prior_min,
            min(prior_max, prior_value),
        )

        conditioned_matrix[path] = normalize_array(
            conditioned_rows[path] + dynamic_prior[path] * BASELINE,
            BASELINE,
        )
        control_matrix[path] = normalize_array(
            control_rows[path] + dynamic_prior[path] * BASELINE,
            BASELINE,
        )

        path_centers[path] = _center(
            conditioned_matrix[path],
            control_matrix[path],
        )
        conditioned_mean = float(
            conditioned_matrix[path, 0] - conditioned_matrix[path, 1]
        )
        control_mean = float(
            control_matrix[path, 0] - control_matrix[path, 1]
        )
        conditioned_variance = max(
            0.0,
            float(
                conditioned_matrix[path, 0]
                + conditioned_matrix[path, 1]
                - conditioned_mean * conditioned_mean
            ),
        )
        control_variance = max(
            0.0,
            float(
                control_matrix[path, 0]
                + control_matrix[path, 1]
                - control_mean * control_mean
            ),
        )
        standard_error = math.sqrt(
            conditioned_variance
            / max(1.0, c_count + dynamic_prior[path])
            + control_variance
            / max(1.0, u_count + dynamic_prior[path])
        )
        residual_z_score = max(
            0.0,
            float(
                settings.get(
                    "independent_path_model_residual_z_score",
                    1.0,
                )
            ),
        )
        absolute_center = abs(path_centers[path])
        signal_quality[path] = clamp01(
            absolute_center
            / (
                absolute_center
                + residual_z_score * standard_error
                + 1e-12
            )
        )

    positive_mass = 0.0
    negative_mass = 0.0
    for path in range(4):
        direction_mass = (
            predicted_path[path]
            * support[path]
            * (0.35 + 0.65 * signal_quality[path])
        )
        if path_centers[path] > 1e-12:
            positive_mass += direction_mass
        elif path_centers[path] < -1e-12:
            negative_mass += direction_mass

    weighted_support = float(np.dot(predicted_path, support))
    supported_mass = float(np.dot(predicted_path, support))
    weighted_signal_quality = (
        float(np.dot(predicted_path * support, signal_quality))
        / max(1e-12, supported_mass)
        if supported_mass > 1e-12
        else 0.0
    )
    decisive = positive_mass + negative_mass
    direction_agreement = (
        max(positive_mass, negative_mass) / decisive
        if decisive > 1e-12
        else 0.5
    )
    direction_q = scale_quality(direction_agreement)

    conditioned_mix = normalize_array(
        np.sum(predicted_path[:, None] * conditioned_matrix, axis=0),
        BASELINE,
    )
    control_mix = normalize_array(
        np.sum(predicted_path[:, None] * control_matrix, axis=0),
        BASELINE,
    )
    control_probs = normalize_array(
        np.sum(control_rows, axis=0),
        BASELINE,
    )
    mechanism_probabilities = normalize_array(
        control_probs + (conditioned_mix - control_mix),
        BASELINE,
    )
    particle_probs = normalize_array(
        particle_probabilities
        if particle_probabilities is not None
        else np.sum(conditioned_rows, axis=0),
        BASELINE,
    )

    residual_shrink_power = max(
        0.10,
        float(
            settings.get(
                "independent_path_model_residual_shrink_power",
                0.85,
            )
        ),
    )
    residual_shrinkage = clamp01(
        (weighted_support ** residual_shrink_power)
        * (0.50 + 0.50 * weighted_signal_quality)
        * (0.78 + 0.22 * population_ess_quality)
        * (0.72 + 0.28 * depth_effect)
        * (0.82 + 0.18 * direction_q)
    )
    raw_residual = mechanism_probabilities - particle_probs
    probabilities = normalize_array(
        particle_probs + residual_shrinkage * raw_residual,
        BASELINE,
    )

    current_q = scale_quality(current_path_agreement)
    draw_q = scale_quality(draw_agreement)
    reliability = (
        0.31 * path_quality
        + 0.22 * weighted_support
        + 0.18 * current_q
        + 0.09 * draw_q
        + 0.08 * direction_q
        + 0.07 * weighted_signal_quality
        + 0.05 * population_ess_quality
    )
    reliability *= 0.82 + 0.18 * depth_consistency
    reliability *= 0.90 + 0.10 * depth_effect
    reliability = clamp01(reliability)

    enabled = bool(settings.get("independent_path_model_enabled", True))
    minimum_reliability = clamp01(
        float(settings.get("independent_path_model_min_reliability", 0.55))
    )
    maximum_weight = max(
        0.0,
        min(
            0.40,
            float(settings.get("independent_path_model_max_weight", 0.18)),
        ),
    )
    late_maximum_weight = max(
        maximum_weight,
        min(
            0.40,
            float(
                settings.get(
                    "independent_path_model_late_max_weight",
                    0.24,
                )
            ),
        ),
    )
    dynamic_maximum_weight = (
        maximum_weight
        + (late_maximum_weight - maximum_weight)
        * depth_effect
        * path_quality
    )
    effective_weight = 0.0
    if enabled and reliability >= minimum_reliability:
        effective_weight = dynamic_maximum_weight * reliability
        effective_weight *= 0.55 + 0.45 * path_quality
        effective_weight *= 0.68 + 0.32 * direction_q
        effective_weight *= 0.72 + 0.28 * residual_shrinkage
        effective_weight = max(
            0.0,
            min(dynamic_maximum_weight, effective_weight),
        )

    return {
        "enabled": enabled,
        "probabilities": probabilities,
        "raw_probabilities": mechanism_probabilities,
        "particle_probabilities": particle_probs,
        "control_probabilities": control_probs,
        "next_draw_paths": predicted_path,
        "conditioned_path_outcomes": conditioned_matrix,
        "control_path_outcomes": control_matrix,
        "support": support,
        "support_targets": support_targets,
        "balance_quality": balance_quality,
        "signal_quality": signal_quality,
        "path_centers": path_centers,
        "dynamic_prior_strength": dynamic_prior,
        "weighted_support": weighted_support,
        "weighted_signal_quality": weighted_signal_quality,
        "residual_shrinkage": residual_shrinkage,
        "reliability": reliability,
        "minimum_reliability": minimum_reliability,
        "maximum_weight": maximum_weight,
        "late_maximum_weight": late_maximum_weight,
        "dynamic_maximum_weight": dynamic_maximum_weight,
        "effective_weight": effective_weight,
        "direction_agreement": direction_agreement,
        "path_quality": path_quality,
        "population_ess_quality": population_ess_quality,
        "shoe_depth": shoe_depth_ratio,
        "depth_factor": depth_factor,
        "depth_consistency": depth_consistency,
        "depth_effect": depth_effect,
        "uses_additional_simulations": False,
    }


class V5ReplicaEngine:
    def __init__(self, seed: int, particle_count: int, decks: int) -> None:
        self.seed = int(seed) & 0xFFFFFFFF
        self.particle_count = max(64, int(particle_count))
        self.decks = max(1, int(decks))
        self.rng = np.random.default_rng(self.seed)

    def build_stratified_prior(
        self,
        hand_number: Optional[int] = None,
        hand_uncertainty: int = 0,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Build the shoe immediately before the observed current hand.

        When a factual hand number is known, every particle is generated near
        that physical depth instead of mixing early-, mid- and late-shoe states.
        If the hand number is absent, the calibrated broad prior is retained for
        backward compatibility.
        """
        particles: List[np.ndarray] = []
        depths: List[int] = []

        try:
            current_hand = int(hand_number or 0)
        except Exception:
            current_hand = 0

        if current_hand > 0:
            base_prior_depth = max(0, min(90, current_hand - 1))
            uncertainty = max(0, min(5, int(hand_uncertainty)))
            offsets = list(range(-uncertainty, uncertainty + 1)) or [0]
            for item_index in range(self.particle_count):
                requested = max(
                    0,
                    min(
                        90,
                        base_prior_depth + offsets[item_index % len(offsets)],
                    ),
                )
                counts = fresh_shoe_counts(self.decks)
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

        allocations = [
            int(math.floor(self.particle_count * w))
            for _, _, w in CALIBRATED_DEPTH_PROFILE
        ]
        index = 0
        while sum(allocations) < self.particle_count:
            allocations[index % len(allocations)] += 1
            index += 1
        for (low, high, _), count in zip(
            CALIBRATED_DEPTH_PROFILE,
            allocations,
        ):
            span = high - low + 1
            for item_index in range(count):
                counts = fresh_shoe_counts(self.decks)
                q = (item_index + 0.5) / max(1, count)
                base_depth = low + min(
                    span - 1,
                    int(math.floor(q * span)),
                )
                requested = max(
                    low,
                    min(
                        high,
                        base_depth + int(self.rng.integers(-1, 2)),
                    ),
                )
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
        known_cards: Optional[Mapping[str, Sequence[int]]] = None,
    ) -> ConditionedPopulation:
        effective_known_path = known_path
        if known_cards:
            valid_cards, inferred_path, _ = validate_known_hand(
                player_total,
                banker_total,
                known_path,
                known_cards,
            )
            if valid_cards:
                effective_known_path = inferred_path
        feasible = feasible_current_paths(
            player_total,
            banker_total,
            effective_known_path,
        )
        if not bool(np.any(feasible)):
            # An explicitly supplied path that cannot produce these final totals
            # should not spend the full proposal budget searching for impossible
            # candidates.  Preserve the prior as a low-quality fallback instead.
            particles = [
                np.asarray(item, dtype=np.int16).copy()
                for item in prior_particles
            ]
            unique = len({item.tobytes() for item in particles})
            return ConditionedPopulation(
                particles=particles,
                depths=[int(value) for value in prior_depths],
                control_particles=[item.copy() for item in particles],
                control_depths=[int(value) for value in prior_depths],
                current_paths=[-1] * len(particles),
                updated=False,
                low_sample=True,
                matches=0,
                attempts=0,
                ess=0.0,
                acceptance=0.0,
                draw_paths=DRAW_BASELINE.copy(),
                path_candidate_counts=np.zeros(4, dtype=float),
                path_ess=np.zeros(4, dtype=float),
                path_allocated=np.zeros(4, dtype=int),
                path_coverage=0.0,
                legacy_path_coverage=0.0,
                path_ess_quality=0.0,
                feasible_paths=feasible,
                known_path=effective_known_path,
                unique_particles=unique,
                accepted_unique=0,
                ancestry_paired=False,
            )
        by_path: List[List[ConditionalCandidate]] = [[], [], [], []]
        accepted: List[ConditionalCandidate] = []
        attempts = 0
        ess = 0.0
        # Scale the accepted conditional population with the particle population.
        # Otherwise increasing PF_PARTICLES mostly duplicates the same old candidates.
        effective = _scaled_runtime_settings(settings, self.particle_count)
        target_matches = int(effective["target_matches"])
        target_ess = float(effective["target_ess"])
        min_matches = int(effective["min_matches"])
        path_target_matches = int(effective["path_target_matches"])
        max_proposals = int(effective["max_update_proposals"])
        while attempts < max_proposals:
            parent_index = int(self.rng.integers(0, len(prior_particles)))
            source = prior_particles[parent_index]
            if known_cards:
                completed = exact_known_hand_complete(
                    source,
                    int(player_total) % 10,
                    int(banker_total) % 10,
                    effective_known_path,
                    known_cards,
                )
            else:
                completed = exact_conditional_complete(
                    source,
                    self.rng,
                    int(player_total) % 10,
                    int(banker_total) % 10,
                    effective_known_path,
                    likelihood_power=float(
                        settings["third_card_likelihood_power"]
                    ),
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
            if attempts % 32 == 0:
                ess = weighted_ess(accepted)
                total_ready = (
                    len(accepted) >= target_matches
                    and ess >= target_ess
                )
                strata_ready = all(
                    (not feasible[p]) or len(by_path[p]) >= path_target_matches
                    for p in range(4)
                )
                if total_ready and strata_ready:
                    break
        ess = weighted_ess(accepted)
        path_counts = np.asarray([len(rows) for rows in by_path], dtype=float)
        path_ess = np.asarray([weighted_ess(rows) for rows in by_path], dtype=float)
        path_weight_sums = np.asarray(
            [sum(max(0.0, float(x.weight)) for x in rows) for rows in by_path], dtype=float
        )
        fallback_draw = np.where(feasible, DRAW_BASELINE, 0.0)
        if float(fallback_draw.sum()) <= 0:
            fallback_draw = feasible.astype(float)
        # Weak Dirichlet smoothing stabilizes rare legal draw paths without
        # overriding the importance-weighted conditional evidence.
        path_prior = normalize_array(fallback_draw, DRAW_BASELINE)
        path_posterior = normalize_array(
            path_weight_sums + float(settings["path_prior_strength"]) * path_prior,
            fallback_draw,
        )
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
                known_path=effective_known_path,
                unique_particles=unique,
                accepted_unique=0,
                ancestry_paired=False,
            )
        path_min_matches = int(settings["path_min_matches"])
        path_min_ess = float(settings["path_min_ess"])
        path_target_ess = max(
            path_min_ess,
            float(path_target_matches) * 0.55,
        )
        count_quality = np.minimum(
            1.0,
            path_counts / max(1.0, float(path_target_matches)),
        )
        ess_quality_by_path = np.minimum(
            1.0,
            path_ess / max(1.0, path_target_ess),
        )
        support_quality = np.sqrt(count_quality * ess_quality_by_path)
        available = feasible & (path_counts > 0)
        allocation = allocate_path_particles(
            path_posterior,
            self.particle_count,
            available,
            int(settings["min_path_particles"]),
            support_quality,
            no_draw_protection=float(
                settings["no_draw_allocation_protection"]
            ),
        )
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
            sum(
                path_posterior[path] * ess_quality_by_path[path]
                for path in range(4)
            )
        )
        significant = feasible & (
            path_posterior
            >= max(0.02, 0.5 / max(1.0, float(self.particle_count)))
        )
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
            low_sample=len(accepted) < min_matches or ess < target_ess * 0.60 or low_path_sample,
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
            known_path=effective_known_path,
            unique_particles=unique,
            accepted_unique=accepted_unique,
            ancestry_paired=True,
        )

    def forecast(self, population: ConditionedPopulation, settings: Mapping[str, Any]) -> ReplicaResult:
        particle_count = len(population.particles)
        samples = max(
            200,
            int(settings["predict_simulations_per_replica"])
            + int(settings["point_joint_simulations_per_replica"]),
            particle_count * 2,
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
        next_draw_raw = normalize_array(conditioned_draw, DRAW_BASELINE)
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
        (
            next_draw,
            path_transition_prior,
            path_transition_weight,
            path_transition_quality,
        ) = apply_current_path_transition_prior(
            next_draw_raw,
            population.draw_paths,
            population.known_path,
            population.path_coverage,
            population.path_ess_quality,
            settings,
        )
        entropy = -float(sum(x * math.log(x) for x in matrix if x > 0))
        entropy_norm = min(1.0, entropy / math.log(100))
        top10_mass = float(matrix[top_idx].sum())
        point_concentration = max(
            0.0,
            min(1.0, 0.55 * (1.0 - entropy_norm) + 0.45 * max(0.0, min(1.0, (top10_mass - 0.10) / 0.22))),
        )
        database = get_shoe_state_database()
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
        db_allowed = settings["database_validation_mode"] == "force" or (
            settings["database_validation_mode"] == "validated_only" and DB_HOLDOUT["passed"]
        )
        if settings["database_validation_mode"] == "diagnostic":
            db_allowed = False
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
        current_agreement_q = _bounded_quality(
            current_agreement,
            0.50,
            0.85,
        )
        path_quality = (
            0.50 * population.path_coverage
            + 0.30 * population.path_ess_quality
            + 0.20 * current_agreement_q
        )
        draw_agreement_q = _bounded_quality(
            draw_agreement,
            0.50,
            0.85,
        )
        path_direction_quality = (
            0.70 * current_agreement_q
            + 0.30 * draw_agreement_q
        )
        path_fusion_gain = (
            0.24 * path_quality * path_direction_quality
        )
        path_adjustment_limit = min(
            0.020,
            max(
                0.004,
                float(settings["hybrid_max_component_adjustment"]),
            ),
        )
        path_delta = np.clip(
            pf - control_probs,
            -path_adjustment_limit,
            path_adjustment_limit,
        )
        population_mean_depth = float(np.mean(population.depths))
        population_min_depth = int(min(population.depths))
        population_max_depth = int(max(population.depths))
        population_composition = _composition(
            population.particles,
            self.decks,
        )
        independent_path = build_independent_draw_path_model(
            conditioned_draw,
            control_draw,
            path_outcome_c,
            path_outcome_u,
            population.path_coverage,
            population.path_ess_quality,
            current_agreement,
            draw_agreement,
            settings,
            particle_probabilities=pf,
            path_candidate_counts=population.path_candidate_counts,
            path_ess=population.path_ess,
            path_allocated=population.path_allocated,
            population_ess=population.ess,
            target_ess=float(settings["target_ess"]),
            mean_depth=population_mean_depth,
            min_depth=population_min_depth,
            max_depth=population_max_depth,
            shoe_depth=float(population_composition["shoe_depth"]),
            decks=self.decks,
        )
        independent_residual = (
            np.asarray(
                independent_path["probabilities"],
                dtype=float,
            )
            - pf
        )
        independent_adjustment = np.clip(
            float(independent_path["effective_weight"])
            * independent_residual,
            -float(settings["independent_path_model_max_adjustment"]),
            float(settings["independent_path_model_max_adjustment"]),
        )
        fused = normalize_array(
            pf
            + path_fusion_gain * path_delta
            + independent_adjustment
            + effective_db * delta,
            BASELINE,
        )
        base_weight = (
            max(
                0.05,
                0.12
                + 0.33 * min(1.0, population.ess / max(1.0, float(settings["target_ess"])))
                + 0.16 * min(1.0, population.unique_particles / max(1, self.particle_count))
                + 0.24 * path_quality
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
            next_draw_paths_raw=next_draw_raw,
            path_transition_prior=path_transition_prior,
            path_transition_weight=path_transition_weight,
            path_transition_quality=path_transition_quality,
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
            path_quality=path_quality,
            path_fusion_gain=path_fusion_gain,
            independent_path_probabilities=np.asarray(
                independent_path["probabilities"],
                dtype=float,
            ),
            independent_path_control_probabilities=np.asarray(
                independent_path["control_probabilities"],
                dtype=float,
            ),
            independent_path_next_draw=np.asarray(
                independent_path["next_draw_paths"],
                dtype=float,
            ),
            independent_path_outcome_matrix=np.asarray(
                independent_path["conditioned_path_outcomes"],
                dtype=float,
            ),
            independent_path_control_outcome_matrix=np.asarray(
                independent_path["control_path_outcomes"],
                dtype=float,
            ),
            independent_path_support=np.asarray(
                independent_path["support"],
                dtype=float,
            ),
            independent_path_reliability=float(
                independent_path["reliability"]
            ),
            independent_path_effective_weight=float(
                independent_path["effective_weight"]
            ),
            independent_path_direction_agreement=float(
                independent_path["direction_agreement"]
            ),
            independent_path_weighted_support=float(
                independent_path["weighted_support"]
            ),
            independent_path_residual_shrinkage=float(
                independent_path["residual_shrinkage"]
            ),
            independent_path_depth_factor=float(
                independent_path["depth_factor"]
            ),
            independent_path_depth_consistency=float(
                independent_path["depth_consistency"]
            ),
            independent_path_dynamic_max_weight=float(
                independent_path["dynamic_maximum_weight"]
            ),
            independent_path_dynamic_prior_strength=np.asarray(
                independent_path["dynamic_prior_strength"],
                dtype=float,
            ),
            point_concentration=point_concentration,
            effective_database_weight=float(effective_db),
            database_reliability=float(db_rel),
            database_samples=float(db_samples),
            composition=population_composition,
            digest=_digest(population.particles, self.seed),
            seed=self.seed,
            mean_depth=population_mean_depth,
            min_depth=population_min_depth,
            max_depth=population_max_depth,
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



def _bounded_quality(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (float(value) - low) / (high - low)))


def build_hybrid_probabilities(
    particle_probs: Sequence[float],
    database_probs: Sequence[float],
    exact_state_probs: Sequence[float],
    quality: Mapping[str, float],
    settings: Mapping[str, Any],
    exact_state_reliability: float,
    effective_database_weight: float,
    independent_path_probs: Optional[Sequence[float]] = None,
    independent_path_reliability: float = 0.0,
    independent_path_effective_weight: float = 0.0,
) -> Dict[str, Any]:
    """Quality-gated fusion of independent factual components.

    The baseline share is deliberately retained. Adding more components must not
    magnify a tiny Monte Carlo deviation into a strong recommendation.
    """
    pf = normalize_array(particle_probs, BASELINE)
    db = normalize_array(database_probs, BASELINE)
    state = normalize_array(exact_state_probs, BASELINE)
    independent_path = normalize_array(
        independent_path_probs if independent_path_probs is not None else pf,
        BASELINE,
    )
    independent_enabled = bool(
        settings.get("independent_path_model_enabled", True)
    )
    independent_min_reliability = max(
        0.0,
        min(
            1.0,
            float(settings.get("independent_path_model_min_reliability", 0.55)),
        ),
    )
    independent_max_weight = max(
        0.0,
        min(
            0.40,
            max(
                float(settings.get("independent_path_model_max_weight", 0.18)),
                float(settings.get("independent_path_model_late_max_weight", 0.24)),
            ),
        ),
    )
    independent_reliability = max(
        0.0,
        min(1.0, float(independent_path_reliability)),
    )
    independent_weight = 0.0
    if independent_enabled and independent_reliability >= independent_min_reliability:
        independent_weight = min(
            independent_max_weight,
            max(0.0, float(independent_path_effective_weight)),
        )

    if str(settings.get("hybrid_mode", "hybrid")) != "hybrid":
        independent_residual = independent_path - pf
        independent_adjustment = np.clip(
            independent_weight * independent_residual,
            -float(settings["independent_path_model_max_adjustment"]),
            float(settings["independent_path_model_max_adjustment"]),
        )
        particle_fused = normalize_array(
            pf + independent_adjustment,
            BASELINE,
        )
        return {
            "probabilities": particle_fused,
            "weights": {
                "particle": 1.0,
                "exact_shoe_state": 0.0,
                "database": 0.0,
                "baseline": 0.0,
            },
            "gate": 1.0,
            "state_reliability": 0.0,
            "independent_path_enabled": independent_enabled,
            "independent_path_reliability": independent_reliability,
            "independent_path_effective_weight": independent_weight,
            "independent_path_residual_adjustment": independent_adjustment,
            "mode": "particle",
        }

    agreement_q = _bounded_quality(
        float(quality.get("agreement", 0.5)),
        0.50,
        0.85,
    )
    ess_q = min(
        1.0,
        float(quality.get("average_ess", 0.0))
        / max(1.0, float(settings["target_ess"])),
    )
    diversity_q = _bounded_quality(
        float(quality.get("average_diversity", 0.0)),
        0.25,
        0.75,
    )
    split_q = _bounded_quality(
        float(quality.get("split_agreement", 0.0)),
        0.50,
        1.00,
    )
    path_coverage_q = max(
        0.0,
        min(1.0, float(quality.get("path_coverage", 0.0))),
    )
    path_ess_q = max(
        0.0,
        min(1.0, float(quality.get("path_ess_quality", 0.0))),
    )
    current_path_q = _bounded_quality(
        float(quality.get("current_path_agreement", 0.5)),
        0.50,
        0.85,
    )
    path_q = (
        0.42 * path_coverage_q
        + 0.28 * path_ess_q
        + 0.30 * current_path_q
    )
    context_q = (
        0.45 * max(0.0, min(1.0, float(quality.get("hand_number_known", 0.0))))
        + 0.25 * max(0.0, min(1.0, float(quality.get("known_path", 0.0))))
        + 0.30 * max(0.0, min(1.0, float(exact_state_reliability)))
    )
    gate = (
        0.15 * agreement_q
        + 0.14 * ess_q
        + 0.09 * diversity_q
        + 0.10 * split_q
        + 0.42 * path_q
        + 0.10 * context_q
    )
    gate = max(0.12, min(1.0, gate))

    baseline_floor = float(settings["hybrid_baseline_min_weight"])
    usable = max(0.0, 1.0 - baseline_floor)

    configured_particle_max = max(
        0.0,
        min(1.0, float(settings["hybrid_particle_max_weight"])),
    )
    high_path_threshold = max(
        0.55,
        min(
            0.95,
            float(settings.get("hybrid_high_path_threshold", 0.75)),
        ),
    )
    high_path_boost_cap = max(
        0.0,
        min(
            0.18,
            float(settings.get("hybrid_high_path_weight_boost", 0.07)),
        ),
    )
    high_path_excess = _bounded_quality(path_q, high_path_threshold, 1.0)
    high_path_boost = (
        high_path_boost_cap
        * high_path_excess
        * (0.70 + 0.30 * current_path_q)
    )
    dynamic_particle_max = min(
        1.0,
        configured_particle_max + high_path_boost,
    )
    particle_weight = min(
        dynamic_particle_max,
        usable * (0.35 + 0.65 * gate),
    )
    state_weight = min(
        float(settings["hybrid_exact_state_max_weight"]),
        usable
        * max(0.0, min(1.0, float(exact_state_reliability)))
        * (0.45 + 0.55 * gate),
    )
    database_weight = min(
        max(0.0, float(effective_database_weight)),
        max(0.0, usable - particle_weight - state_weight),
    )

    model_total = particle_weight + state_weight + database_weight
    if model_total > usable and model_total > 0:
        scale = usable / model_total
        particle_weight *= scale
        state_weight *= scale
        database_weight *= scale
    baseline_weight = max(
        baseline_floor,
        1.0 - particle_weight - state_weight - database_weight,
    )

    combined = (
        baseline_weight * BASELINE
        + particle_weight * pf
        + state_weight * state
        + database_weight * db
    )
    max_adjustment = float(settings["hybrid_max_component_adjustment"])
    delta = np.clip(
        combined - BASELINE,
        -max_adjustment,
        max_adjustment,
    )
    # Only the path head residual relative to PF is added. If the two models say
    # the same thing, this contribution is exactly zero, preventing double count.
    independent_residual = independent_path - pf
    independent_adjustment = np.clip(
        independent_weight * independent_residual,
        -float(settings["independent_path_model_max_adjustment"]),
        float(settings["independent_path_model_max_adjustment"]),
    )
    fused = normalize_array(
        BASELINE + delta + independent_adjustment,
        BASELINE,
    )
    return {
        "probabilities": fused,
        "weights": {
            "particle": float(particle_weight),
            "exact_shoe_state": float(state_weight),
            "database": float(database_weight),
            "baseline": float(baseline_weight),
        },
        "gate": float(gate),
        "path_gate": float(path_q),
        "configured_particle_max_weight": float(configured_particle_max),
        "dynamic_particle_max_weight": float(dynamic_particle_max),
        "high_path_threshold": float(high_path_threshold),
        "high_path_particle_boost": float(high_path_boost),
        "path_coverage_quality": float(path_coverage_q),
        "path_ess_quality": float(path_ess_q),
        "current_path_agreement_quality": float(current_path_q),
        "state_reliability": float(exact_state_reliability),
        "independent_path_enabled": independent_enabled,
        "independent_path_reliability": independent_reliability,
        "independent_path_effective_weight": independent_weight,
        "independent_path_residual_adjustment": independent_adjustment,
        "mode": "hybrid",
    }

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
        "validated_edge_after_bonus": edge,
        "base_min_validated_edge": 0.0,
        "effective_min_validated_edge": 0.0,
        "validated_edge_relief": 0.0,
        "advantage_continuity_score": 0.0,
        "advantage_continuity_bonus": 0.0,
        "replica_advantage_continuity": 0.0,
        "split_advantage_continuity": 0.0,
        "path_advantage_continuity": 0.0,
        "decision_confidence": max(fused[0], fused[1]) / max(1e-12, fused[0] + fused[1]),
        "decision_confidence_quality": 0.0,
        "decision_aggression_score": 0.0,
        "model_side": None,
        "validated_signal": False,
        "quality_pass": False,
        "decision_source": "UNVALIDATED_COMPARISON",
        "banker_ev": banker_ev,
        "player_ev": player_ev,
        "fallback_score": centered,
        "effective_replicas": float(len(fused)),
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

    centers = [row.paired_center for row in rows]
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
    internal_rms = math.sqrt(
        sum(w * row.internal_se**2 for row, w in zip(rows, weights)) / sw
    )
    split_se = (
        float(settings["split_uncertainty"])
        * internal_rms
        / math.sqrt(effective_replicas)
    )
    path_rms = math.sqrt(
        sum(
            w * row.current_path_internal_se**2
            for row, w in zip(rows, weights)
        )
        / sw
    )

    current_path_agreement = max(
        0.0,
        min(1.0, float(quality.get("current_path_agreement", 0.5))),
    )
    current_path_q = _bounded_quality(current_path_agreement, 0.50, 0.85)
    path_disagreement_multiplier = 1.0 + 0.30 * (1.0 - current_path_q)
    path_se = (
        float(settings["path_uncertainty"])
        * path_rms
        * path_disagreement_multiplier
        / math.sqrt(effective_replicas)
    )
    se = math.sqrt(base_se**2 + split_se**2 + path_se**2)
    robust = 0.50 * mean + 0.50 * med
    raw_lower = max(
        0.0,
        abs(robust) - float(settings["uncertainty_penalty"]) * se,
    )
    model_side = "B" if robust >= 0 else "P"
    target_sign = 1.0 if robust >= 0 else -1.0

    global_center = _center(fused, control)
    direction_consistency = (
        abs(global_center) < 1e-12
        or (global_center >= 0) == (robust >= 0)
    )

    path_coverage = max(
        0.0,
        min(1.0, float(quality.get("path_coverage", 0.0))),
    )
    path_ess_quality = max(
        0.0,
        min(1.0, float(quality.get("path_ess_quality", 0.0))),
    )
    path_quality_score = (
        0.42 * path_coverage
        + 0.28 * path_ess_quality
        + 0.30 * current_path_q
    )
    path_quality_threshold = max(
        0.60,
        min(0.95, float(settings["min_path_coverage"]) * 0.92),
    )
    path_ess_floor = 0.42
    current_path_agreement_floor = 0.56
    path_quality_pass = (
        path_coverage >= float(settings["min_path_coverage"])
        and path_ess_quality >= path_ess_floor
        and current_path_agreement >= current_path_agreement_floor
        and path_quality_score >= path_quality_threshold
    )

    # Current-request-only continuity: repeated same-side evidence across
    # replicas, split estimates and draw-path branches. No road history,
    # recommendation history or win/loss state is consumed.
    replica_advantage_continuity = (
        sum(
            w
            for center, w in zip(centers, weights)
            if center * target_sign > 0.0
        )
        / max(1e-12, sw)
    )

    split_same = 0.0
    split_total = 0.0
    for row, weight in zip(rows, weights):
        for split_center in row.split_centers:
            split_total += weight
            if float(split_center) * target_sign > 0.0:
                split_same += weight
    split_advantage_continuity = (
        split_same / split_total
        if split_total > 0.0
        else float(quality.get("split_agreement", 0.5))
    )

    path_same = 0.0
    path_total = 0.0
    for row, weight in zip(rows, weights):
        centers_row = np.asarray(row.current_path_centers, dtype=float)
        samples_row = np.maximum(
            0.0,
            np.asarray(row.current_path_samples, dtype=float),
        )
        if centers_row.shape != (4,) or samples_row.shape != (4,):
            continue
        for path_center, sample_count in zip(centers_row, samples_row):
            evidence_weight = weight * float(sample_count)
            if evidence_weight <= 0.0:
                continue
            path_total += evidence_weight
            if float(path_center) * target_sign > 0.0:
                path_same += evidence_weight
    path_advantage_continuity = (
        path_same / path_total
        if path_total > 0.0
        else current_path_agreement
    )

    advantage_continuity_score = max(
        0.0,
        min(
            1.0,
            0.45 * replica_advantage_continuity
            + 0.30 * split_advantage_continuity
            + 0.25 * path_advantage_continuity,
        ),
    )
    advantage_continuity_q = _bounded_quality(
        advantage_continuity_score,
        0.62,
        0.92,
    )

    non_tie_total = max(1e-12, float(fused[0] + fused[1]))
    confidence = max(float(fused[0]), float(fused[1])) / non_tie_total
    confidence_q = _bounded_quality(confidence, 0.515, 0.555)
    high_path_q = _bounded_quality(path_quality_score, 0.75, 0.90)
    high_current_path_q = _bounded_quality(
        current_path_agreement,
        0.68,
        0.88,
    )

    quality_pass = (
        quality["agreement"] >= float(settings["min_replica_agreement"])
        and quality["average_ess"] >= float(settings["target_ess"]) * 0.8
        and quality["average_diversity"] >= 0.45
        and quality["split_agreement"] >= 0.60
        and path_quality_pass
        and effective_replicas >= float(settings["min_effective_replicas"])
        and direction_consistency
        and all(row.updated and row.ancestry_paired for row in rows)
    )

    base_min_validated_edge = max(
        0.0,
        float(settings["min_validated_edge"]),
    )
    high_quality_triplet = min(
        high_path_q,
        high_current_path_q,
        confidence_q,
    )
    max_edge_relief = max(
        0.0,
        min(
            0.60,
            float(settings.get("high_quality_edge_relief", 0.40)),
        ),
    )
    validated_edge_relief = (
        max_edge_relief
        * high_quality_triplet
        * (0.80 + 0.20 * advantage_continuity_q)
    )
    effective_min_validated_edge = max(
        base_min_validated_edge * 0.50,
        base_min_validated_edge * (1.0 - validated_edge_relief),
    )

    edge_strength_q = _bounded_quality(
        abs(robust),
        max(1e-6, base_min_validated_edge * 0.75),
        max(0.008, base_min_validated_edge * 4.0),
    )
    continuity_bonus_factor = max(
        0.0,
        min(
            0.50,
            float(settings.get("advantage_continuity_bonus", 0.22)),
        ),
    )
    advantage_continuity_bonus = (
        base_min_validated_edge
        * continuity_bonus_factor
        * advantage_continuity_q
        * edge_strength_q
        * (0.55 + 0.45 * min(high_path_q, high_current_path_q))
    )
    validated_edge_after_bonus = raw_lower + advantage_continuity_bonus

    decision_aggression_score = max(
        0.0,
        min(
            1.0,
            0.34 * high_path_q
            + 0.28 * high_current_path_q
            + 0.24 * confidence_q
            + 0.14 * advantage_continuity_q,
        ),
    )

    validated = (
        quality_pass
        and validated_edge_after_bonus >= effective_min_validated_edge
    )

    fallback_score = robust
    if fallback_score > 1e-12:
        fallback_side = "B"
    elif fallback_score < -1e-12:
        fallback_side = "P"
    else:
        fallback_side = "B" if int(rows[0].seed) & 1 else "P"

    recommend = model_side if validated else fallback_side
    decision_source = (
        "VALIDATED_AGGRESSIVE_PATH_MODEL"
        if validated and decision_aggression_score >= 0.65
        else "VALIDATED_MODEL"
        if validated
        else "LOW_CONFIDENCE_BALANCED"
    )
    signal = (
        "HIGH"
        if (
            validated
            and decision_aggression_score >= 0.72
            and validated_edge_after_bonus
            >= max(0.0035, effective_min_validated_edge * 2.0)
        )
        else "MEDIUM"
        if validated
        else "LOW"
    )

    return {
        "recommend": recommend,
        "reason": (
            f"{int(settings['particles'])}粒子補牌路徑品質、方向連續性與高信心動態門檻均通過"
            if validated and decision_aggression_score >= 0.65
            else f"{int(settings['particles'])}粒子補牌路徑分層、統一配對樣本池與穩健誤差修正後通過品質閘門"
            if validated
            else "訊號未通過完整驗證，仍沿用低信心對稱後驗方向，不固定回退莊家"
        ),
        "signal_level": signal,
        "edge": validated_edge_after_bonus,
        "center": robust,
        "raw_center": mean,
        "median_center": med,
        "center_std": std,
        "center_se": se,
        "base_se": base_se,
        "split_se": split_se,
        "path_se": path_se,
        "lower_bound": raw_lower,
        "validated_edge_after_bonus": validated_edge_after_bonus,
        "base_min_validated_edge": base_min_validated_edge,
        "effective_min_validated_edge": effective_min_validated_edge,
        "validated_edge_relief": validated_edge_relief,
        "advantage_continuity_score": advantage_continuity_score,
        "advantage_continuity_bonus": advantage_continuity_bonus,
        "replica_advantage_continuity": replica_advantage_continuity,
        "split_advantage_continuity": split_advantage_continuity,
        "path_advantage_continuity": path_advantage_continuity,
        "decision_confidence": confidence,
        "decision_confidence_quality": confidence_q,
        "decision_aggression_score": decision_aggression_score,
        "model_side": model_side,
        "validated_signal": validated,
        "quality_pass": quality_pass,
        "path_quality_pass": path_quality_pass,
        "path_quality_score": path_quality_score,
        "path_quality_threshold": path_quality_threshold,
        "path_ess_floor": path_ess_floor,
        "current_path_agreement": current_path_agreement,
        "current_path_agreement_quality": current_path_q,
        "current_path_agreement_floor": current_path_agreement_floor,
        "decision_source": decision_source,
        "banker_ev": float(fused[0] * (1.0 - commission) - fused[1]),
        "player_ev": float(fused[1] - fused[0]),
        "fallback_score": fallback_score,
        "effective_replicas": effective_replicas,
        "direction_consistency": direction_consistency,
    }


class V5IndependentBaccaratEngine:
    def __init__(self, settings: Optional[Mapping[str, Any]] = None) -> None:
        supplied = dict(settings or {})
        requested_particles = int(supplied.get("particles", PARTICLE_COUNT))
        safe_particles = max(MIN_PARTICLE_COUNT, min(MAX_PARTICLE_COUNT, requested_particles))
        self.settings: Dict[str, Any] = {
            "decks": max(1, min(16, int(supplied.get("decks", DECKS)))),
            "particles": safe_particles,
            "replicas": max(3, min(11, int(supplied.get("replicas", REPLICA_COUNT)))),
            "target_matches": int(supplied.get("target_matches", TARGET_MATCHES)),
            "target_ess": float(supplied.get("target_ess", TARGET_ESS)),
            "min_matches": int(supplied.get("min_matches", MIN_MATCHES)),
            "max_update_proposals": int(supplied.get("max_update_proposals", MAX_UPDATE_PROPOSALS)),
            "path_target_matches": int(supplied.get("path_target_matches", PATH_TARGET_MATCHES)),
            "path_min_matches": int(supplied.get("path_min_matches", PATH_MIN_MATCHES)),
            "path_min_ess": float(supplied.get("path_min_ess", PATH_MIN_ESS)),
            "min_path_coverage": float(supplied.get("min_path_coverage", MIN_PATH_COVERAGE)),
            "path_uncertainty": float(supplied.get("path_uncertainty", PATH_UNCERTAINTY)),
            "path_prior_strength": max(
                0.0, min(50.0, float(supplied.get("path_prior_strength", PATH_PRIOR_STRENGTH)))
            ),
            "min_path_particles": max(
                1, min(512, int(supplied.get("min_path_particles", MIN_PATH_PARTICLES)))
            ),
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
            "hybrid_mode": str(
                supplied.get("hybrid_mode", HYBRID_MODE)
            ).lower(),
            "hybrid_particle_max_weight": float(
                supplied.get(
                    "hybrid_particle_max_weight",
                    HYBRID_PARTICLE_MAX_WEIGHT,
                )
            ),
            "hybrid_exact_state_max_weight": float(
                supplied.get(
                    "hybrid_exact_state_max_weight",
                    HYBRID_EXACT_STATE_MAX_WEIGHT,
                )
            ),
            "hybrid_baseline_min_weight": float(
                supplied.get(
                    "hybrid_baseline_min_weight",
                    HYBRID_BASELINE_MIN_WEIGHT,
                )
            ),
            "hybrid_max_component_adjustment": float(
                supplied.get(
                    "hybrid_max_component_adjustment",
                    HYBRID_MAX_COMPONENT_ADJUSTMENT,
                )
            ),
            "hybrid_high_path_threshold": max(
                0.55,
                min(
                    0.95,
                    float(
                        supplied.get(
                            "hybrid_high_path_threshold",
                            HYBRID_HIGH_PATH_THRESHOLD,
                        )
                    ),
                ),
            ),
            "hybrid_high_path_weight_boost": max(
                0.0,
                min(
                    0.18,
                    float(
                        supplied.get(
                            "hybrid_high_path_weight_boost",
                            HYBRID_HIGH_PATH_WEIGHT_BOOST,
                        )
                    ),
                ),
            ),
            "high_quality_edge_relief": max(
                0.0,
                min(
                    0.60,
                    float(
                        supplied.get(
                            "high_quality_edge_relief",
                            HIGH_QUALITY_EDGE_RELIEF,
                        )
                    ),
                ),
            ),
            "advantage_continuity_bonus": max(
                0.0,
                min(
                    0.50,
                    float(
                        supplied.get(
                            "advantage_continuity_bonus",
                            ADVANTAGE_CONTINUITY_BONUS,
                        )
                    ),
                ),
            ),
            "independent_path_model_enabled": bool(
                supplied.get(
                    "independent_path_model_enabled",
                    INDEPENDENT_PATH_MODEL_ENABLED,
                )
            ),
            "independent_path_model_max_weight": max(
                0.0,
                min(
                    0.40,
                    float(
                        supplied.get(
                            "independent_path_model_max_weight",
                            INDEPENDENT_PATH_MODEL_MAX_WEIGHT,
                        )
                    ),
                ),
            ),
            "independent_path_model_min_reliability": max(
                0.0,
                min(
                    1.0,
                    float(
                        supplied.get(
                            "independent_path_model_min_reliability",
                            INDEPENDENT_PATH_MODEL_MIN_RELIABILITY,
                        )
                    ),
                ),
            ),
            "independent_path_model_max_adjustment": max(
                0.0,
                min(
                    0.05,
                    float(
                        supplied.get(
                            "independent_path_model_max_adjustment",
                            INDEPENDENT_PATH_MODEL_MAX_ADJUSTMENT,
                        )
                    ),
                ),
            ),
            "independent_path_model_prior_strength": max(
                0.0,
                min(
                    100.0,
                    float(
                        supplied.get(
                            "independent_path_model_prior_strength",
                            INDEPENDENT_PATH_MODEL_PRIOR_STRENGTH,
                        )
                    ),
                ),
            ),
            "independent_path_model_prior_min": max(
                0.0,
                min(
                    100.0,
                    float(
                        supplied.get(
                            "independent_path_model_prior_min",
                            INDEPENDENT_PATH_MODEL_PRIOR_MIN,
                        )
                    ),
                ),
            ),
            "independent_path_model_prior_max": max(
                0.0,
                min(
                    100.0,
                    float(
                        supplied.get(
                            "independent_path_model_prior_max",
                            INDEPENDENT_PATH_MODEL_PRIOR_MAX,
                        )
                    ),
                ),
            ),
            "independent_path_model_prior_depth_relief": max(
                0.0,
                min(
                    0.90,
                    float(
                        supplied.get(
                            "independent_path_model_prior_depth_relief",
                            INDEPENDENT_PATH_MODEL_PRIOR_DEPTH_RELIEF,
                        )
                    ),
                ),
            ),
            "independent_path_model_prior_low_support_gain": max(
                0.0,
                min(
                    5.0,
                    float(
                        supplied.get(
                            "independent_path_model_prior_low_support_gain",
                            INDEPENDENT_PATH_MODEL_PRIOR_LOW_SUPPORT_GAIN,
                        )
                    ),
                ),
            ),
            "independent_path_model_prior_imbalance_gain": max(
                0.0,
                min(
                    5.0,
                    float(
                        supplied.get(
                            "independent_path_model_prior_imbalance_gain",
                            INDEPENDENT_PATH_MODEL_PRIOR_IMBALANCE_GAIN,
                        )
                    ),
                ),
            ),
            "independent_path_model_support_target_fraction": max(
                0.05,
                min(
                    2.0,
                    float(
                        supplied.get(
                            "independent_path_model_support_target_fraction",
                            INDEPENDENT_PATH_MODEL_SUPPORT_TARGET_FRACTION,
                        )
                    ),
                ),
            ),
            "independent_path_model_support_min_count": max(
                1.0,
                min(
                    200.0,
                    float(
                        supplied.get(
                            "independent_path_model_support_min_count",
                            INDEPENDENT_PATH_MODEL_SUPPORT_MIN_COUNT,
                        )
                    ),
                ),
            ),
            "independent_path_model_residual_shrink_power": max(
                0.10,
                min(
                    3.0,
                    float(
                        supplied.get(
                            "independent_path_model_residual_shrink_power",
                            INDEPENDENT_PATH_MODEL_RESIDUAL_SHRINK_POWER,
                        )
                    ),
                ),
            ),
            "independent_path_model_residual_z_score": max(
                0.0,
                min(
                    5.0,
                    float(
                        supplied.get(
                            "independent_path_model_residual_z_score",
                            INDEPENDENT_PATH_MODEL_RESIDUAL_Z_SCORE,
                        )
                    ),
                ),
            ),
            "independent_path_model_depth_start": max(
                0.0,
                min(
                    0.95,
                    float(
                        supplied.get(
                            "independent_path_model_depth_start",
                            INDEPENDENT_PATH_MODEL_DEPTH_START,
                        )
                    ),
                ),
            ),
            "independent_path_model_depth_full": max(
                0.05,
                min(
                    1.0,
                    float(
                        supplied.get(
                            "independent_path_model_depth_full",
                            INDEPENDENT_PATH_MODEL_DEPTH_FULL,
                        )
                    ),
                ),
            ),
            "independent_path_model_late_max_weight": max(
                0.0,
                min(
                    0.40,
                    float(
                        supplied.get(
                            "independent_path_model_late_max_weight",
                            INDEPENDENT_PATH_MODEL_LATE_MAX_WEIGHT,
                        )
                    ),
                ),
            ),
            "hand_number_uncertainty": int(
                supplied.get(
                    "hand_number_uncertainty",
                    HAND_NUMBER_UNCERTAINTY,
                )
            ),
            "state_simulations": int(
                supplied.get("state_simulations", STATE_SIMULATIONS)
            ),
            "third_card_likelihood_power": max(
                0.70,
                min(
                    1.0,
                    float(
                        supplied.get(
                            "third_card_likelihood_power",
                            THIRD_CARD_LIKELIHOOD_POWER,
                        )
                    ),
                ),
            ),
            "no_draw_allocation_protection": max(
                1.0,
                min(
                    2.0,
                    float(
                        supplied.get(
                            "no_draw_allocation_protection",
                            NO_DRAW_ALLOCATION_PROTECTION,
                        )
                    ),
                ),
            ),
            "path_transition_prior_weight": max(
                0.0,
                min(
                    0.40,
                    float(
                        supplied.get(
                            "path_transition_prior_weight",
                            PATH_TRANSITION_PRIOR_WEIGHT,
                        )
                    ),
                ),
            ),
            "path_transition_self_bias": max(
                0.0,
                min(
                    0.30,
                    float(
                        supplied.get(
                            "path_transition_self_bias",
                            PATH_TRANSITION_SELF_BIAS,
                        )
                    ),
                ),
            ),
        }

    def analyze(
        self,
        player_total: int,
        banker_total: int,
        seed: int,
        known_path: Optional[int] = None,
        hand_number: Optional[int] = None,
        known_cards: Optional[Mapping[str, Sequence[int]]] = None,
        remaining_counts: Optional[Sequence[int]] = None,
        state_complete: bool = False,
    ) -> Dict[str, Any]:
        runtime_settings = _scaled_runtime_settings(
            self.settings,
            int(self.settings["particles"]),
        )

        normalized_cards = normalize_known_cards(known_cards)
        card_validation = "not_supplied"
        effective_known_path = known_path
        if normalized_cards is not None:
            valid_cards, inferred_path, card_validation = validate_known_hand(
                int(player_total) % 10,
                int(banker_total) % 10,
                known_path,
                normalized_cards,
            )
            if not valid_cards:
                raise ValueError(f"invalid_known_cards:{card_validation}")
            effective_known_path = inferred_path

        try:
            physical_hand_number = max(0, min(120, int(hand_number or 0)))
        except Exception:
            physical_hand_number = 0

        exact_remaining = (
            _valid_remaining_counts(
                remaining_counts,
                int(runtime_settings["decks"]),
            )
            if state_complete
            else None
        )
        exact_state_reliability = 1.0 if exact_remaining is not None else 0.0

        rows: List[ReplicaResult] = []
        for replica_index in range(
            max(3, int(runtime_settings["replicas"]))
        ):
            replica_seed = mix_seed(seed, replica_index)
            engine = V5ReplicaEngine(
                replica_seed,
                particle_count=int(runtime_settings["particles"]),
                decks=int(runtime_settings["decks"]),
            )
            prior_particles, prior_depths = engine.build_stratified_prior(
                hand_number=physical_hand_number or None,
                hand_uncertainty=int(
                    runtime_settings["hand_number_uncertainty"]
                ),
            )
            population = engine.condition(
                prior_particles,
                prior_depths,
                int(player_total) % 10,
                int(banker_total) % 10,
                effective_known_path,
                runtime_settings,
                known_cards=normalized_cards,
            )
            rows.append(engine.forecast(population, runtime_settings))

        robust_mad, outlier_count = _apply_robust_weights(
            rows,
            bool(runtime_settings["adaptive_replica_weight"]),
        )
        pf = _weighted_average(rows, "pf", 3)
        control = _weighted_average(rows, "control", 3)
        database = _weighted_average(rows, "database", 3)
        particle_database_fused = _weighted_average(rows, "fused", 3)
        independent_path_probabilities = _weighted_average(
            rows,
            "independent_path_probabilities",
            3,
        )
        independent_path_control_probabilities = _weighted_average(
            rows,
            "independent_path_control_probabilities",
            3,
        )
        independent_path_next_draw = _weighted_average(
            rows,
            "independent_path_next_draw",
            4,
        )
        draw = _weighted_average(rows, "draw_paths", 4)
        next_draw = _weighted_average(rows, "next_draw_paths", 4)
        next_draw_raw = _weighted_average(rows, "next_draw_paths_raw", 4)
        path_transition_prior = _weighted_average(
            rows,
            "path_transition_prior",
            4,
        )
        point_matrix = _weighted_average(rows, "point_matrix", 100)

        top_idx = np.argsort(point_matrix)[::-1][:10]
        top_points = [
            {
                "point": f"{int(i // 10)}{int(i % 10)}",
                "probability": float(point_matrix[i]),
                "outcome": (
                    "B"
                    if int(i % 10) > int(i // 10)
                    else "P"
                    if int(i // 10) > int(i % 10)
                    else "T"
                ),
            }
            for i in top_idx
        ]

        weight_sum = sum(
            max(1e-6, row.final_weight)
            for row in rows
        )
        normalized_row_weights = np.asarray(
            [max(1e-6, row.final_weight) for row in rows],
            dtype=float,
        )
        normalized_row_weights /= max(
            1e-12,
            float(normalized_row_weights.sum()),
        )
        independent_path_outcome_matrix = np.zeros((4, 3), dtype=float)
        independent_path_control_outcome_matrix = np.zeros((4, 3), dtype=float)
        independent_path_support = np.zeros(4, dtype=float)
        for row, row_weight in zip(rows, normalized_row_weights):
            independent_path_outcome_matrix += (
                row_weight * row.independent_path_outcome_matrix
            )
            independent_path_control_outcome_matrix += (
                row_weight * row.independent_path_control_outcome_matrix
            )
            independent_path_support += (
                row_weight * row.independent_path_support
            )
        independent_path_reliability = float(
            sum(
                row_weight * row.independent_path_reliability
                for row, row_weight in zip(rows, normalized_row_weights)
            )
        )
        independent_path_effective_weight = float(
            sum(
                row_weight * row.independent_path_effective_weight
                for row, row_weight in zip(rows, normalized_row_weights)
            )
        )
        independent_path_direction_agreement = float(
            sum(
                row_weight * row.independent_path_direction_agreement
                for row, row_weight in zip(rows, normalized_row_weights)
            )
        )
        independent_path_weighted_support = float(
            sum(
                row_weight * row.independent_path_weighted_support
                for row, row_weight in zip(rows, normalized_row_weights)
            )
        )
        independent_path_residual_shrinkage = float(
            sum(
                row_weight * row.independent_path_residual_shrinkage
                for row, row_weight in zip(rows, normalized_row_weights)
            )
        )
        independent_path_depth_factor = float(
            sum(
                row_weight * row.independent_path_depth_factor
                for row, row_weight in zip(rows, normalized_row_weights)
            )
        )
        independent_path_depth_consistency = float(
            sum(
                row_weight * row.independent_path_depth_consistency
                for row, row_weight in zip(rows, normalized_row_weights)
            )
        )
        independent_path_dynamic_max_weight = float(
            sum(
                row_weight * row.independent_path_dynamic_max_weight
                for row, row_weight in zip(rows, normalized_row_weights)
            )
        )
        independent_path_dynamic_prior_strength = np.zeros(4, dtype=float)
        for row, row_weight in zip(rows, normalized_row_weights):
            independent_path_dynamic_prior_strength += (
                row_weight * row.independent_path_dynamic_prior_strength
            )

        weighted_votes = {"B": 0.0, "P": 0.0}
        votes = {"B": 0, "P": 0}
        directions: List[str] = []
        for row in rows:
            side = "B" if row.paired_center >= 0 else "P"
            weighted_votes[side] += max(1e-6, row.final_weight)
            votes[side] += 1
            directions.append(side)

        agreement = max(weighted_votes.values()) / max(
            1e-12,
            weight_sum,
        )
        split_agreement = sum(
            max(1e-6, row.final_weight) * row.split_agreement
            for row in rows
        ) / max(1e-12, weight_sum)
        average_ess = float(np.mean([row.ess for row in rows]))
        average_diversity = float(
            np.mean([row.diversity for row in rows])
        )
        average_path_coverage = sum(
            max(1e-6, row.final_weight) * row.path_coverage
            for row in rows
        ) / max(1e-12, weight_sum)
        average_path_ess_quality = sum(
            max(1e-6, row.final_weight) * row.path_ess_quality
            for row in rows
        ) / max(1e-12, weight_sum)
        average_current_path_agreement = sum(
            max(1e-6, row.final_weight) * row.current_path_agreement
            for row in rows
        ) / max(1e-12, weight_sum)
        average_draw_agreement = sum(
            max(1e-6, row.final_weight) * row.draw_agreement
            for row in rows
        ) / max(1e-12, weight_sum)
        average_path_quality = sum(
            max(1e-6, row.final_weight) * row.path_quality
            for row in rows
        ) / max(1e-12, weight_sum)
        average_path_fusion_gain = sum(
            max(1e-6, row.final_weight) * row.path_fusion_gain
            for row in rows
        ) / max(1e-12, weight_sum)
        average_path_transition_weight = sum(
            max(1e-6, row.final_weight) * row.path_transition_weight
            for row in rows
        ) / max(1e-12, weight_sum)
        average_path_transition_quality = sum(
            max(1e-6, row.final_weight) * row.path_transition_quality
            for row in rows
        ) / max(1e-12, weight_sum)
        average_database_weight = float(
            np.mean([row.effective_database_weight for row in rows])
        )

        quality = {
            "agreement": agreement,
            "average_ess": average_ess,
            "average_diversity": average_diversity,
            "split_agreement": split_agreement,
            "path_coverage": average_path_coverage,
            "path_ess_quality": average_path_ess_quality,
            "current_path_agreement": average_current_path_agreement,
            "draw_agreement": average_draw_agreement,
            "independent_path_reliability": independent_path_reliability,
            "independent_path_effective_weight": independent_path_effective_weight,
            "independent_path_weighted_support": independent_path_weighted_support,
            "independent_path_residual_shrinkage": independent_path_residual_shrinkage,
            "independent_path_depth_factor": independent_path_depth_factor,
            "hand_number_known": 1.0 if physical_hand_number > 0 else 0.0,
            "known_path": 1.0 if effective_known_path is not None else 0.0,
        }

        state_probs = (
            estimate_exact_state_probabilities(
                exact_remaining,
                int(runtime_settings["decks"]),
                mix_seed(seed, 991001),
                int(runtime_settings["state_simulations"]),
            )
            if exact_remaining is not None
            else BASELINE.copy()
        )
        hybrid = build_hybrid_probabilities(
            pf,
            database,
            state_probs,
            quality,
            runtime_settings,
            exact_state_reliability,
            average_database_weight,
            independent_path_probabilities,
            independent_path_reliability,
            independent_path_effective_weight,
        )
        fused = normalize_array(
            hybrid["probabilities"],
            BASELINE,
        )

        decision = decide_ensemble(
            fused,
            control,
            rows,
            quality,
            runtime_settings,
        )

        stability = "UNSTABLE"
        if (
            decision["validated_signal"]
            and agreement
            >= float(runtime_settings["min_replica_agreement"])
            and split_agreement >= 0.80
            and average_ess
            >= float(runtime_settings["target_ess"]) * 0.8
            and average_diversity >= 0.45
            and average_path_coverage
            >= float(runtime_settings["min_path_coverage"])
            and decision.get("path_quality_pass", False)
            and average_current_path_agreement >= 0.60
            and decision["effective_replicas"]
            >= float(runtime_settings["min_effective_replicas"])
        ):
            stability = "STABLE"
        elif (
            agreement >= 0.57
            and split_agreement >= 0.60
            and average_ess
            >= float(runtime_settings["target_ess"]) * 0.45
            and average_diversity >= 0.30
            and average_path_coverage >= 0.55
            and average_path_ess_quality >= 0.35
            and average_current_path_agreement >= 0.54
        ):
            stability = "WATCH"

        weakness: List[str] = []
        if not decision["validated_signal"]:
            weakness.append(
                "HYBRID訊號未通過補牌路徑信賴下界或品質閘門"
            )
        if agreement < float(
            runtime_settings["min_replica_agreement"]
        ):
            weakness.append("副本穩健方向共識低於設定門檻")
        if split_agreement < 0.60:
            weakness.append("統一樣本池分半方向一致率低於60%")
        if average_ess < float(
            runtime_settings["target_ess"]
        ) * 0.8:
            weakness.append("條件候選ESS未達目標80%")
        if average_diversity < 0.45:
            weakness.append("粒子多樣性不足45%")
        if average_path_coverage < float(
            runtime_settings["min_path_coverage"]
        ):
            weakness.append("補牌路徑有效覆蓋率不足")
        elif not decision.get("path_quality_pass", False):
            weakness.append("補牌路徑覆蓋率與路徑ESS綜合品質不足")
        if (
            runtime_settings["independent_path_model_enabled"]
            and independent_path_reliability
            < float(runtime_settings["independent_path_model_min_reliability"])
        ):
            weakness.append("獨立補牌模型可靠度不足，已自動停用其機率修正")
        if physical_hand_number <= 0:
            weakness.append(
                "未提供牌靴局數，仍使用早中晚牌靴混合先驗"
            )
        if normalized_cards is None:
            weakness.append(
                "未提供本局實際牌值，僅依最終點數反推可能牌組"
            )
        if not state_complete:
            weakness.append(
                "牌靴卡牌追蹤不完整，精確剩餘牌組分量未啟用"
            )
        if outlier_count:
            weakness.append(
                f"已抑制{outlier_count}個偏離中位數副本"
            )
        if any(row.low_sample for row in rows):
            weakness.append(
                "至少一個副本的總候選或補牌路徑候選偏少"
            )
        if (
            not DB_HOLDOUT["passed"]
            and runtime_settings["database_validation_mode"] != "force"
        ):
            weakness.append(
                "資料庫樣本外驗證未優於基準，方向校正已抑制"
            )
        if not weakness:
            weakness.append(
                f"{int(runtime_settings['particles'])}粒子、"
                "牌靴深度、補牌路徑及精確牌組品質均通過"
            )

        combined = hashlib.sha1()
        for row in rows:
            combined.update(row.digest.encode("ascii"))

        base_forecast_samples = max(
            int(runtime_settings["predict_simulations_per_replica"])
            + int(
                runtime_settings[
                    "point_joint_simulations_per_replica"
                ]
            ),
            int(runtime_settings["particles"]) * 2,
        )
        state_samples = (
            int(runtime_settings["state_simulations"])
            if exact_remaining is not None
            else 0
        )

        return {
            "pf": pf,
            "control": control,
            "database": database,
            "particle_database_fused": particle_database_fused,
            "shoe_state": state_probs,
            "fused": fused,
            "hybrid": {
                **hybrid,
                "probabilities": fused,
                "card_validation": card_validation,
                "hand_number_used": physical_hand_number,
                "known_path_used": effective_known_path,
                "exact_state_enabled": exact_remaining is not None,
            },
            "draw_paths": draw,
            "next_draw_paths": next_draw,
            "next_draw_paths_raw": next_draw_raw,
            "path_transition_prior": path_transition_prior,
            "average_path_transition_weight": average_path_transition_weight,
            "average_path_transition_quality": average_path_transition_quality,
            "independent_draw_path_model": {
                "enabled": bool(runtime_settings["independent_path_model_enabled"]),
                "probabilities": independent_path_probabilities,
                "control_probabilities": independent_path_control_probabilities,
                "next_draw_paths": independent_path_next_draw,
                "path_outcome_probabilities": independent_path_outcome_matrix,
                "control_path_outcome_probabilities": (
                    independent_path_control_outcome_matrix
                ),
                "path_support": independent_path_support,
                "reliability": independent_path_reliability,
                "minimum_reliability": float(
                    runtime_settings["independent_path_model_min_reliability"]
                ),
                "configured_max_weight": float(
                    runtime_settings["independent_path_model_max_weight"]
                ),
                "configured_late_max_weight": float(
                    runtime_settings["independent_path_model_late_max_weight"]
                ),
                "effective_weight": float(
                    hybrid.get(
                        "independent_path_effective_weight",
                        independent_path_effective_weight,
                    )
                ),
                "direction_agreement": independent_path_direction_agreement,
                "weighted_support": independent_path_weighted_support,
                "residual_shrinkage": independent_path_residual_shrinkage,
                "depth_factor": independent_path_depth_factor,
                "depth_consistency": independent_path_depth_consistency,
                "dynamic_max_weight": independent_path_dynamic_max_weight,
                "dynamic_prior_strength": independent_path_dynamic_prior_strength,
                "shoe_depth": float(
                    np.mean([row.composition["shoe_depth"] for row in rows])
                ),
                "residual_adjustment": np.asarray(
                    hybrid.get(
                        "independent_path_residual_adjustment",
                        np.zeros(3, dtype=float),
                    ),
                    dtype=float,
                ),
                "max_adjustment": float(
                    runtime_settings["independent_path_model_max_adjustment"]
                ),
                "uses_additional_simulations": False,
            },
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
            "average_matches": float(
                np.mean([row.matches for row in rows])
            ),
            "average_ess": average_ess,
            "average_acceptance": float(
                np.mean([row.acceptance for row in rows])
            ),
            "average_attempts": float(
                np.mean([row.attempts for row in rows])
            ),
            "average_diversity": average_diversity,
            "average_path_coverage": average_path_coverage,
            "average_legacy_path_coverage": float(
                np.mean(
                    [row.legacy_path_coverage for row in rows]
                )
            ),
            "average_path_ess_quality": average_path_ess_quality,
            "average_path_quality": average_path_quality,
            "average_path_fusion_gain": average_path_fusion_gain,
            "average_current_path_agreement": average_current_path_agreement,
            "average_draw_agreement": average_draw_agreement,
            "average_point_concentration": float(
                np.mean(
                    [row.point_concentration for row in rows]
                )
            ),
            "average_path_candidates": np.mean(
                np.stack(
                    [row.path_candidate_counts for row in rows]
                ),
                axis=0,
                dtype=float,
            ),
            "average_path_ess": np.mean(
                np.stack([row.path_ess for row in rows]),
                axis=0,
            ),
            "average_path_allocated": np.mean(
                np.stack([row.path_allocated for row in rows]),
                axis=0,
            ),
            "average_current_path_centers": np.mean(
                np.stack(
                    [row.current_path_centers for row in rows]
                ),
                axis=0,
            ),
            "average_database_weight": average_database_weight,
            "database_samples": float(
                np.mean([row.database_samples for row in rows])
            ),
            "mean_depth": float(
                np.mean([row.mean_depth for row in rows])
            ),
            "min_depth": int(
                min(row.min_depth for row in rows)
            ),
            "max_depth": int(
                max(row.max_depth for row in rows)
            ),
            "cards_remaining": float(
                np.mean(
                    [
                        row.composition["cards_remaining"]
                        for row in rows
                    ]
                )
            ),
            "shoe_depth": float(
                np.mean(
                    [row.composition["shoe_depth"] for row in rows]
                )
            ),
            "physical_hand_number": physical_hand_number,
            "exact_state_reliability": exact_state_reliability,
            "state_digest": combined.hexdigest()[:16],
            "robust_mad": robust_mad,
            "outlier_count": outlier_count,
            "total_forecast_simulations": int(
                len(rows) * base_forecast_samples + state_samples
            ),
            "total_condition_attempts": int(
                sum(row.attempts for row in rows)
            ),
            "all_ancestry_paired": all(
                row.ancestry_paired for row in rows
            ),
            "all_replicas_updated": all(
                row.updated for row in rows
            ),
            "fallback_to_unconditioned": any(
                not row.updated for row in rows
            ),
            "conditional_generator": (
                "EXACT_CURRENT_CARDS_IMPORTANCE_WEIGHTED"
                if normalized_cards is not None
                else "DRAW_PATH_STRATIFIED_EXACT_COMPLETION_IMPORTANCE_WEIGHTED_V5_PATH0_PROTECTED"
            ),
            "variance_reduction": (
                "FULL_PARTICLE_TWO_PASS_COMMON_RANDOM_ANTITHETIC"
            ),
            "depth_profile": (
                f"PHYSICAL_HAND_{physical_hand_number}"
                if physical_hand_number > 0
                else "CALIBRATED_10_30_40_20"
            ),
            "settings": dict(runtime_settings),
            "configured_settings": dict(self.settings),
            "replica_rows": rows,
        }
