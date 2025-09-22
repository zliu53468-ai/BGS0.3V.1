# -*- coding: utf-8 -*-
"""
pfilter.py — 完整粒子濾波/MC學習 百家樂 OutcomePF（可直接覆蓋，支援 update_outcome + MC動態預測）
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

    def update_point_history(self, p_pts, b_pts):
        self.prev_p_pts = p_pts
        self.prev_b_pts = b_pts
        self.point_diff_history.append(p_pts - b_pts)
        if len(self.point_diff_history) > 200:
            self.point_diff_history = self.point_diff_history[-200:]

    def update_outcome(self, outcome):
        # outcome: 0=莊 1=閒 2=和
        # 依據 outcome 對每個粒子進行 "重要性權重" 更新與重採樣
        new_weights = np.zeros_like(self.weights)
        for i in range(self.n_particles):
            # 對每個粒子模擬一次這副牌 outcome 機率
            p_win, b_win, tie = self._simulate_next_outcome(self.p_counts[i])
            if outcome == 0:   # 莊
                prob = b_win
            elif outcome == 1: # 閒
                prob = p_win
            else:              # 和
                prob = tie
            new_weights[i] = self.weights[i] * (prob + self.dirichlet_eps)
        new_weights_sum = np.sum(new_weights)
        if new_weights_sum > 0:
            self.weights = new_weights / new_weights_sum
        else:
            self.weights[:] = 1.0 / self.n_particles

        # 檢查重採樣
        neff = 1.0 / np.sum(self.weights ** 2)
        if neff < self.resample_thr * self.n_particles:
            idxs = self.rng.choice(self.n_particles, self.n_particles, replace=True, p=self.weights)
            self.p_counts = self.p_counts[idxs]
            self.weights = np.ones(self.n_particles, dtype=np.float64) / self.n_particles

    def predict(self, sims_per_particle=30):
        # 對每個粒子進行 MC 抽樣並權重平均
        pred = np.zeros(3, dtype=np.float64)
        for i in range(self.n_particles):
            p = np.zeros(3)
            for _ in range(sims_per_particle):
                # 每次都用粒子的當前牌堆來抽一次
                result = self._simulate_single_game(self.p_counts[i])
                p[result] += 1
            if np.sum(p) > 0:
                p = p / np.sum(p)
            pred += self.weights[i] * p
        # 輸出莊、閒、和 機率
        total = np.sum(pred)
        if total > 0:
            return pred / total
        return np.array([0.48, 0.47, 0.05], dtype=np.float32)

    def _simulate_single_game(self, shoe):
        # 百家樂規則隨機發牌，回傳：0=莊 1=閒 2=和
        s = shoe.copy()
        rng = self.rng
        def draw():
            av = np.where(s > 0)[0]
            if len(av) == 0:
                return 0
            idx = rng.choice(av)
            s[idx] -= 1
            return idx
        p1, p2 = draw(), draw()
        b1, b2 = draw(), draw()
        p_sum = (p1 + p2) % 10
        b_sum = (b1 + b2) % 10
        # 玩家補牌
        p3 = None
        if p_sum <= 5:
            p3 = draw()
            p_sum = (p_sum + p3) % 10
        # 莊家補牌
        if b_sum <= 5:
            b3 = draw()
            b_sum = (b_sum + b3) % 10
        if p_sum > b_sum:
            return 1
        elif b_sum > p_sum:
            return 0
        else:
            return 2

    def _simulate_next_outcome(self, shoe):
        # 對單一粒子模擬一堆遊戲，回傳：莊/閒/和 機率
        p = np.zeros(3)
        N = max(10, self.sims_lik)
        for _ in range(N):
            res = self._simulate_single_game(shoe)
            p[res] += 1
        s = np.sum(p)
        if s > 0:
            return p[1]/s, p[0]/s, p[2]/s  # (p_win, b_win, tie)
        return 0.47, 0.48, 0.05

