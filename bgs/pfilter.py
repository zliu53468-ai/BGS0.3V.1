# -*- coding: utf-8 -*-
"""
pfilter.py — 粒子濾波器 強化版（動態學習/可用於百家樂預測/支持動態點數差值調整）
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional

try:
    from .deplete import init_counts
except Exception:
    from deplete import init_counts

PF_N            = int(os.getenv("PF_N", "80"))
PF_UPD_SIMS     = int(os.getenv("PF_UPD_SIMS", "36"))
PF_RESAMPLE     = float(os.getenv("PF_RESAMPLE", "0.73"))
PF_DIR_EPS      = float(os.getenv("PF_DIR_EPS", "0.012"))
PF_BACKEND      = os.getenv("PF_BACKEND", "mc").strip().lower()
PF_STAB_FACTOR  = float(os.getenv("PF_STAB_FACTOR", "0.8"))
PF_DECKS        = int(os.getenv("DECKS", "6"))
PF_SEED         = int(os.getenv("SEED", "42"))

@dataclass
class OutcomePF:
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
        self.last_result = []

    def update_point_history(self, p_pts, b_pts):
        self.prev_p_pts = p_pts
        self.prev_b_pts = b_pts
        self.point_diff_history.append(p_pts - b_pts)
        if len(self.point_diff_history) > 200:
            self.point_diff_history = self.point_diff_history[-200:]

    def update_outcome(self, outcome):
        # outcome: 0=莊 1=閒 2=和
        # 依 outcome 調整粒子分布
        for i in range(self.n_particles):
            # 模擬扣牌，這邊只簡易調整，想要更準可以根據實際牌局規則調整
            if outcome == 0:  # 莊
                self.p_counts[i][0] -= 1
            elif outcome == 1:  # 閒
                self.p_counts[i][1] -= 1
            elif outcome == 2:  # 和
                self.p_counts[i][2] -= 1
            self.p_counts[i] = np.clip(self.p_counts[i], 0, None)
        self.last_result.append(outcome)
        if len(self.last_result) > 200:
            self.last_result = self.last_result[-200:]

    def predict(self, sims_per_particle=30):
        # 根據粒子分布與點數差值進行模擬預測
        preds = []
        for i in range(self.n_particles):
            counts = self.p_counts[i]
            total = counts.sum()
            if total == 0:
                probs = np.array([1/3, 1/3, 1/3])
            else:
                probs = (counts + self.dirichlet_eps) / (total + self.dirichlet_eps * 3)
            preds.append(probs)
        preds = np.stack(preds, axis=0)
        avg_pred = preds.mean(axis=0)
        # 強化點數差值
        if self.prev_p_pts is not None and self.prev_b_pts is not None:
            gap = abs(self.prev_p_pts - self.prev_b_pts)
            # 依點數差強化"勝方"的預測權重
            if self.prev_p_pts > self.prev_b_pts:
                avg_pred[1] += 0.01 * gap
            elif self.prev_b_pts > self.prev_p_pts:
                avg_pred[0] += 0.01 * gap
            avg_pred = np.clip(avg_pred, 0, 1)
            avg_pred = avg_pred / np.sum(avg_pred)
        return avg_pred.astype(np.float32)
