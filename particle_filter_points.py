"""Card-depletion particle filter for baccarat from observed final points.

Each particle represents one possible remaining 8-deck shoe. Because final
Player/Banker totals do not reveal the exact cards, the filter keeps many
possible shoe states and updates them by simulation/importance weighting.

This improves internal consistency and uncertainty estimation. It does not
make an independent random game reliably predictable and cannot guarantee a
68% hit rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import hashlib
import math
import os

import numpy as np


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except Exception:
        return default


def _env_float(name: str, default: float, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(os.getenv(name, str(default)))))
    except Exception:
        return default


DECKS = _env_int('PF_DECKS', 8, 1)
PARTICLE_COUNT = _env_int('PF_PARTICLES', 384, 64)
PROPOSALS_PER_PARTICLE = _env_int('PF_PROPOSALS_PER_PARTICLE', 4, 1)
RESAMPLE_RATIO = _env_float('PF_RESAMPLE_RATIO', 0.50, 0.10, 0.95)
POINT_SIGMA = _env_float('PF_POINT_SIGMA', 0.80, 0.10, 3.0)
OUTCOME_BONUS = _env_float('PF_OUTCOME_BONUS', 2.5, 1.0, 10.0)
PREDICT_SIMS = _env_int('PF_PREDICT_SIMULATIONS', 6000, 500)
RANDOM_SEED = _env_int('PF_RANDOM_SEED', 20260713, 0)


@dataclass
class Particle:
    counts: np.ndarray
    weight: float


@dataclass
class HandResult:
    player_total: int
    banker_total: int
    outcome: str
    counts_after: np.ndarray


def fresh_shoe_counts(decks: int = DECKS) -> np.ndarray:
    # Baccarat card values: 10/J/Q/K are all zero-value cards.
    return np.asarray([16 * decks] + [4 * decks] * 9, dtype=np.int16)


def _draw(counts: np.ndarray, rng: np.random.Generator) -> int:
    total = int(counts.sum())
    if total <= 0:
        raise RuntimeError('shoe is empty')
    index = int(rng.choice(10, p=counts / total))
    counts[index] -= 1
    return index


def _banker_draws(banker_total: int, player_third: Optional[int]) -> bool:
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


def simulate_hand(counts: np.ndarray, rng: np.random.Generator) -> HandResult:
    remaining = counts.copy()
    p_cards = [_draw(remaining, rng)]
    b_cards = [_draw(remaining, rng)]
    p_cards.append(_draw(remaining, rng))
    b_cards.append(_draw(remaining, rng))

    p_total = sum(p_cards) % 10
    b_total = sum(b_cards) % 10
    player_third: Optional[int] = None

    if p_total < 8 and b_total < 8:
        if p_total <= 5:
            player_third = _draw(remaining, rng)
            p_cards.append(player_third)
            p_total = sum(p_cards) % 10
        if _banker_draws(b_total, player_third):
            b_cards.append(_draw(remaining, rng))
            b_total = sum(b_cards) % 10

    outcome = 'B' if b_total > p_total else 'P' if p_total > b_total else 'T'
    return HandResult(p_total, b_total, outcome, remaining)


def _circular_point_distance(a: int, b: int) -> int:
    d = abs(int(a) - int(b)) % 10
    return min(d, 10 - d)


def observation_likelihood(
    simulated: HandResult,
    player_total: int,
    banker_total: int,
    outcome: str,
) -> float:
    dp = _circular_point_distance(simulated.player_total, player_total)
    db = _circular_point_distance(simulated.banker_total, banker_total)
    sigma2 = max(1e-6, POINT_SIGMA * POINT_SIGMA)
    likelihood = math.exp(-0.5 * (dp * dp + db * db) / sigma2)
    if simulated.outcome == outcome:
        likelihood *= OUTCOME_BONUS
    return max(1e-12, likelihood)


class PointParticleFilter:
    def __init__(self, key: str, particle_count: int = PARTICLE_COUNT) -> None:
        digest = int(hashlib.sha1(key.encode('utf-8')).hexdigest()[:16], 16)
        self.rng = np.random.default_rng((RANDOM_SEED + digest) % (2**32 - 1))
        self.particles: List[Particle] = [
            Particle(fresh_shoe_counts(), 1.0 / particle_count)
            for _ in range(particle_count)
        ]
        self.observation_count = 0
        self.last_update: Dict[str, float] = {}

    def _normalize(self) -> None:
        total = sum(max(0.0, p.weight) for p in self.particles)
        if total <= 0:
            uniform = 1.0 / len(self.particles)
            for p in self.particles:
                p.weight = uniform
            return
        for p in self.particles:
            p.weight = max(0.0, p.weight) / total

    def effective_sample_size(self) -> float:
        weights = np.asarray([p.weight for p in self.particles], dtype=float)
        denom = float(np.square(weights).sum())
        return 1.0 / denom if denom > 0 else 0.0

    def _systematic_resample(self) -> None:
        weights = np.asarray([p.weight for p in self.particles], dtype=float)
        cumulative = np.cumsum(weights)
        n = len(self.particles)
        positions = (self.rng.random() + np.arange(n)) / n
        indexes = np.searchsorted(cumulative, positions, side='left')
        indexes = np.clip(indexes, 0, n - 1)
        self.particles = [
            Particle(self.particles[int(i)].counts.copy(), 1.0 / n)
            for i in indexes
        ]

    def update(self, player_total: int, banker_total: int) -> Dict[str, float]:
        player_total = int(player_total) % 10
        banker_total = int(banker_total) % 10
        outcome = 'B' if banker_total > player_total else 'P' if player_total > banker_total else 'T'
        mean_likelihood = 0.0

        for particle in self.particles:
            proposals: List[HandResult] = []
            likelihoods: List[float] = []
            for _ in range(PROPOSALS_PER_PARTICLE):
                try:
                    hand = simulate_hand(particle.counts, self.rng)
                except RuntimeError:
                    hand = HandResult(player_total, banker_total, outcome, fresh_shoe_counts())
                proposals.append(hand)
                likelihoods.append(
                    observation_likelihood(hand, player_total, banker_total, outcome)
                )
            likelihood_array = np.asarray(likelihoods, dtype=float)
            proposal_prob = likelihood_array / max(1e-12, float(likelihood_array.sum()))
            chosen = int(self.rng.choice(len(proposals), p=proposal_prob))
            particle.counts = proposals[chosen].counts_after
            average_likelihood = float(likelihood_array.mean())
            particle.weight *= average_likelihood
            mean_likelihood += average_likelihood

        self._normalize()
        ess_before = self.effective_sample_size()
        resampled = ess_before < len(self.particles) * RESAMPLE_RATIO
        if resampled:
            self._systematic_resample()
        self.observation_count += 1
        self.last_update = {
            'player_total': player_total,
            'banker_total': banker_total,
            'outcome': outcome,
            'effective_sample_size': ess_before,
            'resampled': float(resampled),
            'mean_likelihood': mean_likelihood / len(self.particles),
        }
        return dict(self.last_update)

    def predict(self, simulations: int = PREDICT_SIMS) -> Dict[str, object]:
        simulations = max(500, int(simulations))
        weights = np.asarray([p.weight for p in self.particles], dtype=float)
        weights /= max(1e-12, float(weights.sum()))
        particle_indexes = self.rng.choice(
            len(self.particles), size=simulations, p=weights
        )
        counts = {'B': 0, 'P': 0, 'T': 0}
        p_totals = np.zeros(10, dtype=int)
        b_totals = np.zeros(10, dtype=int)
        for index in particle_indexes:
            hand = simulate_hand(self.particles[int(index)].counts, self.rng)
            counts[hand.outcome] += 1
            p_totals[hand.player_total] += 1
            b_totals[hand.banker_total] += 1
        probs = {key: counts[key] / simulations for key in ('B', 'P', 'T')}
        return {
            'probabilities': probs,
            'counts': counts,
            'simulations': simulations,
            'effective_sample_size': self.effective_sample_size(),
            'observations': self.observation_count,
            'player_total_distribution': (p_totals / simulations).round(6).tolist(),
            'banker_total_distribution': (b_totals / simulations).round(6).tolist(),
            'last_update': dict(self.last_update),
        }
