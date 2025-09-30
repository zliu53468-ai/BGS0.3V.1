# bgs/pfilter.py — 修正版粒子濾波器
"""修正版粒子濾波器，確保環境變數能正確生效，優化趨勢偏見"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

import numpy as np

@dataclass
class Particle:
    p: np.ndarray
    w: float

def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 1e-12, None)
    s = float(v.sum())
    return (v / s) if s > 0 else np.array([1/3, 1/3, 1/3], dtype=v.dtype)

class OutcomePF:
    """
    與 server.py 相容的粒子濾波器
    """
    def __init__(
        self,
        decks: int = 8,
        seed: int = 42,
        n_particles: int = 200,
        sims_lik: int = 80,
        resample_thr: float = 0.5,
        backend: str = "mc",
        dirichlet_eps: float = 0.01,
        **kwargs,
    ) -> None:
        self.decks = int(decks)
        self.backend = str(backend).lower()
        self.rng = np.random.default_rng(int(seed))
        random.seed(int(seed))

        self.n = int(n_particles)
        self.n_particles = self.n
        self.sims_lik = int(sims_lik)
        self.resample_thr = float(resample_thr)
        self.alpha0 = max(1e-6, float(dirichlet_eps))

        self.counts = np.zeros(3, dtype=np.float64)
        self.outcome_history = []
        self.max_history = 50
        self.streak_count = 0
        self.last_outcome = None

        self.particles: List[Particle] = []
        base_prior = [self.alpha0] * 3
        for _ in range(self.n):
            p = self.rng.dirichlet(base_prior).astype(np.float64)
            self.particles.append(Particle(p=p, w=1.0 / self.n))

    def update_outcome(self, outcome: int) -> None:
        if outcome not in (0, 1, 2):
            return
            
        # 更新連勝計數
        if outcome == self.last_outcome and outcome in (0, 1):
            self.streak_count += 1
        else:
            self.streak_count = 1 if outcome in (0, 1) else 0
        self.last_outcome = outcome
            
        self.outcome_history.append(outcome)
        if len(self.outcome_history) > self.max_history:
            self.outcome_history.pop(0)
            
        self.counts[outcome] += 1.0

        w = np.fromiter((max(1e-12, pt.p[outcome]) for pt in self.particles), 
                        dtype=np.float64, count=self.n)
        w_sum = float(w.sum())
        w = (w / w_sum) if w_sum > 0 else np.full(self.n, 1.0 / self.n, dtype=np.float64)

        for i, weight in enumerate(w):
            self.particles[i].w = float(weight)

        ess = 1.0 / float((w ** 2).sum())
        if ess / self.n < self.resample_thr:
            self._resample()

        alpha_base = self.alpha0 + self.counts
        for particle in self.particles:
            particle.p = self.rng.dirichlet(alpha_base).astype(np.float64)

    def _resample(self) -> None:
        weights = np.array([particle.w for particle in self.particles], dtype=np.float64)
        total = float(weights.sum())
        if total <= 0:
            weights.fill(1.0 / self.n)
        else:
            weights /= total
        
        indices = self._systematic_resample(weights)
        new_particles = []
        for i in indices:
            new_particles.append(Particle(p=self.particles[i].p.copy(), w=1.0 / self.n))
        self.particles = new_particles

    def _systematic_resample(self, weights: np.ndarray) -> np.ndarray:
        n = len(weights)
        positions = (np.arange(n) + self.rng.random()) / n
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0
        return np.searchsorted(cumulative_sum, positions)

    def predict(self, sims_per_particle: int = 0) -> np.ndarray:
        ps = np.stack([particle.p for particle in self.particles], axis=0)
        ws = np.array([particle.w for particle in self.particles], dtype=np.float64)
        ws = ws / max(1e-12, float(ws.sum()))
        mix = (ps * ws[:, None]).sum(axis=0)

        theoretical_base = np.array([0.4586, 0.4462, 0.0952], dtype=np.float64)
        
        total_obs = float(self.counts.sum())
        if total_obs < 20 or self.streak_count >= 5:
            mix_weight = 0.2
        else:
            mix_weight = 0.5
            
        out = mix_weight * mix + (1 - mix_weight) * theoretical_base
        out = _normalize(out)
            
        return out.astype(np.float32)

    @property
    def sims_lik(self) -> int:
        return self._sims_lik

    @sims_lik.setter
    def sims_lik(self, value: int):
        self._sims_lik = value
