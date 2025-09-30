# bgs/pfilter.py — 安全簡化版粒子濾波（介面相容 server.py）
"""Render 版粒子濾波備援實作。

此版本刻意保持計算量極低：僅以莊/閒/和累計勝負作為訊息來源，
並在預測階段以 0.6/0.4 的比例混合粒子加權平均與保守基準，與
`server.py` 內文描述的限制完全對應。這樣的設計無法提供超過 53%
的穩定優勢，但能在 Render 免費機器上確保服務不中斷。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

import numpy as np

@dataclass
class Particle:
    p: np.ndarray   # shape=(3,), sum to 1
    w: float        # weight

def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 1e-12, None)
    s = float(v.sum())
    return (v / s) if s > 0 else np.array([1/3, 1/3, 1/3], dtype=v.dtype)

class OutcomePF:
    """
    與 server.py 相容的介面：
      OutcomePF(decks, seed, n_particles, sims_lik, resample_thr,
                backend='mc', dirichlet_eps=0.003, **kwargs)
      update_outcome(outcome)     # outcome ∈ {0(B),1(P),2(T)}
      predict(sims_per_particle)  # 回傳 np.array([pB,pP,pT], dtype=float32)
      backend 屬性（供 log 顯示）
    """
    def __init__(
        self,
        decks: int = 8,
        seed: int = 42,
        n_particles: int = 200,
        sims_lik: int = 80,
        resample_thr: float = 0.5,
        backend: str = "mc",
        dirichlet_eps: float = 0.003,
        **kwargs,
    ) -> None:
        # 基本參數
        self.decks = int(decks)                    # 目前未使用，保留相容
        self.backend = str(backend).lower()        # 僅供紀錄/除錯
        self.rng = np.random.default_rng(int(seed))
        random.seed(int(seed))

        self.n = int(n_particles)
        self.n = max(1, int(n_particles))
        self.n_particles = self.n  # 與 server.py 的紀錄欄位對齊
        self.sims_lik = int(sims_lik)
        self.resample_thr = float(resample_thr)

        # 把 server 傳入的 dirichlet_eps 直接當成 prior 濃度用（小=更發散）
        self.alpha0 = max(1e-6, float(dirichlet_eps))

        # 後驗計數（觀測到的 B/P/T 次數）
        self.counts = np.zeros(3, dtype=np.float64)

        # 初始化粒子：Dirichlet(alpha0) 取樣
        self.particles: List[Particle] = []
        base_prior = [self.alpha0] * 3
        for _ in range(self.n):
            p = self.rng.dirichlet(base_prior).astype(np.float64)
            self.particles.append(Particle(p=p, w=1.0 / self.n))

    # 新一局結果（0=莊,1=閒,2=和）
    def update_outcome(self, outcome: int) -> None:
        if outcome not in (0, 1, 2):
            return
        self.counts[outcome] += 1.0

        # 重要性重加權（likelihood ~ 粒子對該 outcome 的機率）
        w = np.fromiter((max(1e-12, pt.p[outcome]) for pt in self.particles), dtype=np.float64, count=self.n)
        w_sum = float(w.sum())
        w = (w / w_sum) if w_sum > 0 else np.full(self.n, 1.0 / self.n, dtype=np.float64)

        # 更新粒子權重
        for i, weight in enumerate(w):
            self.particles[i].w = float(weight)

        # 有效樣本數檢查與重採樣
        ess = 1.0 / float((w ** 2).sum())
        if ess / self.n < self.resample_thr:
            self._resample()

        # 用觀測結果更新粒子狀態（Dirichlet 後驗取樣）
        alpha_post = self.alpha0 + self.counts
        for particle in self.particles:
            particle.p = self.rng.dirichlet(alpha_post).astype(np.float64)

    def _resample(self) -> None:
        weights = np.array([particle.w for particle in self.particles], dtype=np.float64)
        total = float(weights.sum())
        if total <= 0:
            weights.fill(1.0 / self.n)
        else:
            weights /= total
        
        # 系統性重採樣，減少變異
        indices = self._systematic_resample(weights)
        new_particles = []
        for i in indices:
            new_particles.append(Particle(p=self.particles[i].p.copy(), w=1.0 / self.n))
        self.particles = new_particles

    def _systematic_resample(self, weights: np.ndarray) -> np.ndarray:
        """系統性重採樣，提供更好的採樣品質"""
        n = len(weights)
        positions = (np.arange(n) + self.rng.random()) / n
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0  # 避免數值誤差
        return np.searchsorted(cumulative_sum, positions)

    def predict(self, sims_per_particle: int = 0) -> np.ndarray:
        # 粒子加權平均
        ps = np.stack([particle.p for particle in self.particles], axis=0)
        ws = np.array([particle.w for particle in self.particles], dtype=np.float64)
        ws = ws / max(1e-12, float(ws.sum()))
        mix = (ps * ws[:, None]).sum(axis=0)

        # 保守基準：使用理論百家樂機率作為穩定錨點
        theoretical_base = np.array([0.4586, 0.4462, 0.0952], dtype=np.float64)
        
        # 根據觀測數據量調整混合比例
        total_obs = float(self.counts.sum())
        if total_obs < 10:
            # 初期更多依賴理論值
            mix_weight = min(0.3, total_obs * 0.03)
        else:
            mix_weight = 0.6  # 穩定後主要依賴粒子濾波
        
        out = mix_weight * mix + (1 - mix_weight) * theoretical_base
        out = _normalize(out)
        
        # 平衡檢查：防止單邊預測
        bp_ratio = out[0] / out[1] if out[1] > 0 else 1.0
        if bp_ratio > 1.2 or bp_ratio < 0.83:
            # 過度偏斜，向理論值靠攏
            correction_strength = min(0.5, abs(bp_ratio - 1.0) * 0.8)
            out = (1 - correction_strength) * out + correction_strength * theoretical_base
            out = _normalize(out)
            
        return out.astype(np.float32)

    @property
    def sims_lik(self) -> int:
        return self._sims_lik

    @sims_lik.setter
    def sims_lik(self, value: int):
        self._sims_lik = max(1, value)
