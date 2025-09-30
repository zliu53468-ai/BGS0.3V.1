# -*- coding: utf-8 -*-
"""
輕量 OutcomePF — Render 免費版友善
- 介面與 server.py 期望一致：OutcomePF(...), update_outcome(outcome), predict(...)
- 參數：
    decks, seed, n_particles, sims_lik(=PF_UPD_SIMS), resample_thr(=PF_RESAMPLE),
    backend, dirichlet_eps(=PF_DIR_EPS)
- 作法：
    以理論機率做 Dirichlet 先驗，逐局用 outcome 更新 alpha，predict 回傳平滑後機率。
    這版不是重型模擬器，但能穩定輸出、相容你現有投注/觀望邏輯。
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

# 百家樂理論機率（近似值）
THEO = np.array([0.4586, 0.4462, 0.0952], dtype=np.float64)  # [B, P, T]

@dataclass
class _Cfg:
    decks: int = 8
    seed: int = 42
    n_particles: int = 50
    sims_lik: int = 30
    resample_thr: float = 0.5
    backend: str = "mc"
    dirichlet_eps: float = 0.05

class OutcomePF:
    def __init__(self,
                 decks: int = 8,
                 seed: int = 42,
                 n_particles: int = 50,
                 sims_lik: int = 30,
                 resample_thr: float = 0.5,
                 backend: str = "mc",
                 dirichlet_eps: float = 0.05):
        self.cfg = _Cfg(
            decks=decks, seed=seed, n_particles=n_particles, sims_lik=sims_lik,
            resample_thr=resample_thr, backend=str(backend), dirichlet_eps=float(dirichlet_eps)
        )
        self.rng = np.random.default_rng(seed)
        # 以理論機率當作先驗強度（等效手數=100 * dirichlet_eps 的概念）
        prior_strength = max(1.0, 100.0 * float(dirichlet_eps))
        self.alpha = THEO * prior_strength  # Dirichlet 參數
        self.total = float(self.alpha.sum())

        # 供 server.py 顯示/記錄
        self.n_particles = n_particles
        self.decks = decks
        self._backend = f"dirichlet-lite({self.cfg.backend})"

    @property
    def backend(self) -> str:
        return self._backend

    def update_outcome(self, outcome: int):
        """
        outcome: 0=Banker, 1=Player, 2=Tie
        Tie 在你的 server 流程中不參與 EV（視作 0EV），但仍納入分佈以供顯示。
        """
        if outcome not in (0, 1, 2):
            return
        self.alpha[outcome] += 1.0
        self.total += 1.0

        # 輕度衰減，避免過度黏著（等效 PF_DECAY 效果）
        decay = 0.995
        self.alpha = np.maximum(self.alpha * decay, 1e-6)
        self.total = float(self.alpha.sum())

    def predict(self, sims_per_particle: int = 5) -> np.ndarray:
        """
        回傳 ndarray([pB, pP, pT])；自帶輕度溫度縮放，降低極端。
        """
        probs = self.alpha / self.alpha.sum()

        # 夾緊和局機率到合理區間（避免極端）
        t_min, t_max = 0.01, 0.25
        pB, pP, pT = float(probs[0]), float(probs[1]), float(probs[2])
        pT = float(np.clip(pT, t_min, t_max))
        scale = (1.0 - pT) / (pB + pP + 1e-12)
        pB *= scale
        pP *= scale

        out = np.array([pB, pP, pT], dtype=np.float32)
        out /= out.sum()
        return out
