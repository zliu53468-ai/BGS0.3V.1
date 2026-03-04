# -*- coding: utf-8 -*-
"""
優化版 pfilter.py — NumPy 陣列 + 高效抽樣 + 階段參數
"""
import os
import numpy as np

def _stage_bounds():
    early_end = int(os.getenv("EARLY_HANDS", "20"))
    mid_end = int(os.getenv("MID_HANDS", os.getenv("LATE_HANDS", "56")))
    return early_end, mid_end

def _stage_prefix(rounds_seen: int | None) -> str:
    if rounds_seen is None: return ""
    e_end, m_end = _stage_bounds()
    if rounds_seen < e_end: return "EARLY_"
    elif rounds_seen < m_end: return "MID_"
    else: return "LATE_"

class OutcomePF:
    def __init__(self, decks: int, seed: int = None,
                 n_particles: int = 2000,
                 sims_lik: int = 300,
                 resample_thr: float = 0.4,
                 backend: str = 'numpy',          # 新增，接收 backend 參數
                 dirichlet_eps: float = 1e-5):
        self.decks = decks
        self.n_particles = n_particles
        self.sims_lik = sims_lik
        self.resample_thr = resample_thr
        self.dirichlet_eps = dirichlet_eps
        self.history_mode = int(os.getenv('HISTORY_MODE', '0'))
        self.lik_denom = sims_lik + 3 * dirichlet_eps
        if seed is not None:
            np.random.seed(seed)
        # 使用固定形狀 NumPy 陣列
        self.base_counts = np.zeros(10, dtype=np.int32)
        self.base_counts[0] = 16 * decks
        self.base_counts[1:10] = 4 * decks
        self.particles = np.tile(self.base_counts, (n_particles, 1)) # (n_particles, 10)
        self.weights = np.ones(n_particles, dtype=np.float64) / n_particles

    def _simulate_round(self, counts: np.ndarray) -> int:
        # 快速加權抽牌
        probs = counts.astype(np.float64)
        probs /= probs.sum() + 1e-12
        draws = np.random.choice(10, size=4, replace=False, p=probs)
        temp_counts = counts.copy()
        for d in draws:
            temp_counts[d] -= 1
        player_total = (draws[0] + draws[2]) % 10
        banker_total = (draws[1] + draws[3]) % 10
        if player_total in (8, 9) or banker_total in (8, 9):
            return 1 if player_total > banker_total else (0 if player_total < banker_total else 2)
        player_third = None
        if player_total <= 5:
            probs_third = temp_counts.astype(np.float64)
            probs_third /= probs_third.sum() + 1e-12
            player_third = np.random.choice(10, p=probs_third)
            temp_counts[player_third] -= 1
            player_total = (player_total + player_third) % 10
        banker_draw = False
        if banker_total <= 2:
            banker_draw = True
        elif banker_total == 3 and (player_third is None or player_third % 10 != 8):
            banker_draw = True
        elif banker_total == 4 and 2 <= (player_third % 10) <= 7:
            banker_draw = True
        elif banker_total == 5 and 4 <= (player_third % 10) <= 7:
            banker_draw = True
        elif banker_total == 6 and (player_third is not None and player_third % 10 in (6, 7)):
            banker_draw = True
        if banker_draw:
            probs_third = temp_counts.astype(np.float64)
            probs_third /= probs_third.sum() + 1e-12
            banker_third = np.random.choice(10, p=probs_third)
            banker_total = (banker_total + banker_third) % 10
        return 1 if player_total > banker_total else (0 if player_total < banker_total else 2)

    def predict(self, sims_per_particle: int = None, rounds_seen: int | None = None) -> np.ndarray:
        if sims_per_particle is None:
            sims_per_particle = int(os.getenv("PF_PRED_SIMS", "5"))
            prefix = _stage_prefix(rounds_seen)
            if prefix:
                sp = os.getenv(prefix + "PF_PRED_SIMS")
                if sp: sims_per_particle = int(float(sp))
        sims = int(sims_per_particle)
        total_prob = np.zeros(3, dtype=np.float64)
        for i in range(self.n_particles):
            outcomes = np.array([self._simulate_round(self.particles[i]) for _ in range(sims)])
            counts = np.bincount(outcomes, minlength=3)
            total_prob += self.weights[i] * (counts / sims)
        total = total_prob.sum()
        if total > 0:
            total_prob /= total
        return total_prob.astype(np.float32)

    def update_outcome(self, outcome: int):
        if self.history_mode == 0:
            return
        if outcome == 2 and int(os.getenv('SKIP_TIE_UPD', '0')):
            return
        N = self.n_particles
        new_weights = np.zeros(N, dtype=np.float64)
        for i in range(N):
            outcomes = np.array([self._simulate_round(self.particles[i]) for _ in range(self.sims_lik)])
            match_count = np.sum(outcomes == outcome)
            likelihood = (match_count + self.dirichlet_eps) / self.lik_denom
            new_weights[i] = self.weights[i] * likelihood
        s = new_weights.sum()
        if s == 0:
            new_weights.fill(1.0 / N)
        else:
            new_weights /= s
        self.weights = new_weights
        ess = 1.0 / np.sum(self.weights ** 2)
        if ess < self.resample_thr * N:
            idx = np.random.choice(N, size=N, p=self.weights)
            self.particles = self.particles[idx].copy()
            self.weights.fill(1.0 / N)
