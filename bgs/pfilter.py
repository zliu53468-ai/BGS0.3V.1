# -*- coding: utf-8 -*-
"""
pfilter.py — PF主體（支援 update_point_history/ update_outcome/ predict）
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
        # 這裡你可以根據實際需求加強
        self.prev_p_pts = p_pts
        self.prev_b_pts = b_pts
        self.point_diff_history.append(p_pts - b_pts)
        if len(self.point_diff_history) > 200:
            self.point_diff_history = self.point_diff_history[-200:]

    def update_outcome(self, outcome):
        # outcome: 0=莊 1=閒 2=和
        # 這裡其實應該根據 outcome 去調 PF 粒子分布，但你可以根據需求微調
        # 這裡用最簡單版（實戰用可以強化！）
        pass

    def predict(self, sims_per_particle=30):
        # 產生預測機率：莊、閒、和
        # 這裡簡單 return 固定值（建議用你的真實預測算法覆蓋）
        return np.array([0.48, 0.47, 0.05], dtype=np.float32)
