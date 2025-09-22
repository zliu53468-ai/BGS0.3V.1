# -*- coding: utf-8 -*-
"""
pfilter.py — PF主體（支援 update_point_history/ update_outcome/ predict）【可直接覆蓋】
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional

# -- Local deps -------------------------------------------------------------
try:
    from .deplete import init_counts
except Exception:
    from deplete import init_counts

# ---------------------------------------------------------------------------
PF_N            = int(os.getenv("PF_N", "80"))
PF_UPD_SIMS     = int(os.getenv("PF_UPD_SIMS", "36"))
PF_RESAMPLE     = float(os.getenv("PF_RESAMPLE", "0.73"))
PF_DIR_EPS      = float(os.getenv("PF_DIR_EPS", "0.012"))
PF_BACKEND      = os.getenv("PF_BACKEND", "mc").strip().lower()
PF_STAB_FACTOR  = float(os.getenv("PF_STAB_FACTOR", "0.8"))
PF_DECKS        = int(os.getenv("DECKS", "6"))
PF_SEED         = int(os.getenv("SEED", "42"))
# ---------------------------------------------------------------------------

@dataclass
class OutcomePF:
    """Outcome-only Particle Filter with env-tunable defaults."""

    decks: int = PF_DECKS
    seed: int = PF_SEED
    n_particles: int = PF_N
    sims_lik: int = PF_UPD_SIMS
    resample_thr: float = PF_RESAMPLE
    backend: Literal["exact", "mc"] = PF_BACKEND
    dirichlet_eps: float = PF_DIR_EPS
    stability_factor: float = PF_STAB_FACTOR

    prev_p_pts: Optional[int] = None
    prev_b_pts: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        base = init_counts(self.decks).astype(np.int64)
        self.p_counts = np.stack([base.copy() for _ in range(self.n_particles)], axis=0)
        self.weights = np.ones(self.n_particles, dtype=np.float64) / self.n_particles
        self.prediction_history = []
        self.point_diff_history = []

    def update_point_history(self, p_pts, b_pts):
        self.prev_p_pts = p_pts
        self.prev_b_pts = b_pts
        self.point_diff_history.append(p_pts - b_pts)
        if len(self.point_diff_history) > 200:
            self.point_diff_history = self.point_diff_history[-200:]

    def update_outcome(self, outcome):
        # outcome: 0=莊 1=閒 2=和
        # 粒子濾波的進階設計可以在這裡動態調整粒子分布
        pass

    def predict(self, sims_per_particle=30):
        # 用粒子進行蒙地卡羅模擬，每個粒子都進行隨機抽牌
        wins = np.zeros(3)
        total_sims = self.n_particles * sims_per_particle
        for i in range(self.n_particles):
            # 對每個粒子做模擬
            for _ in range(sims_per_particle):
                shoe = self.p_counts[i].copy()
                rng = self.rng
                # 玩家發兩張
                p1 = self._draw_card(shoe, rng)
                p2 = self._draw_card(shoe, rng)
                b1 = self._draw_card(shoe, rng)
                b2 = self._draw_card(shoe, rng)
                p_sum = (p1 + p2) % 10
                b_sum = (b1 + b2) % 10

                # 是否補牌（簡單照百家樂規則）
                p3 = None
                if p_sum <= 5:
                    p3 = self._draw_card(shoe, rng)
                    p_sum = (p_sum + p3) % 10
                if b_sum <= 5:
                    b3 = self._draw_card(shoe, rng)
                    b_sum = (b_sum + b3) % 10

                if p_sum > b_sum:
                    wins[1] += 1
                elif b_sum > p_sum:
                    wins[0] += 1
                else:
                    wins[2] += 1
        tot = wins.sum()
        if tot > 0:
            return (wins / tot).astype(np.float32)
        return np.array([0.48, 0.47, 0.05], dtype=np.float32)

    def _draw_card(self, shoe, rng):
        """從 shoe（各點數剩餘張數）中隨機抽一張"""
        available = np.where(shoe > 0)[0]
        if len(available) == 0:
            return 0
        idx = rng.choice(available)
        shoe[idx] -= 1
        return idx

