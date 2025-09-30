# bgs/pfilter.py — 安全簡化版粒子濾波（介面相容 server.py）
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
        self.sims_lik = int(sims_lik)
        self.resample_thr = float(resample_thr)

        # 把 server 傳入的 dirichlet_eps 直接當成 prior 濃度用（小=更發散）
        self.alpha0 = max(1e-6, float(dirichlet_eps))

        # 後驗計數（觀測到的 B/P/T 次數）
        self.counts = np.zeros(3, dtype=np.float64)

        # 初始化粒子：Dirichlet(alpha0) 取樣
        self.particles: List[Particle] = []
        for _ in range(self.n):
            p = self.rng.dirichlet([self.alpha0, self.alpha0, self.alpha0]).astype(np.float64)
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
        for i, pt in enumerate(self.particles):
            pt.w = float(w[i])

        # 有效粒子數 (ESS) 降到門檻以下就重採樣
        ess = 1.0 / float((w ** 2).sum())
        if ess / self.n < self.resample_thr:
            self._resample()

        # 以 Dirichlet(alpha0 + counts) 重新抽樣每個粒子的機率向量（保守移動）
        alpha_post = self.alpha0 + self.counts
        for pt in self.particles:
            pt.p = self.rng.dirichlet(alpha_post).astype(np.float64)

    def _resample(self) -> None:
        w = np.array([pt.w for pt in self.particles], dtype=np.float64)
        w_sum = float(w.sum())
        if w_sum <= 0:
            w = np.full(self.n, 1.0 / self.n, dtype=np.float64)
        else:
            w = w / w_sum
        idx = self.rng.choice(self.n, size=self.n, replace=True, p=w)
        self.particles = [Particle(p=self.particles[i].p.copy(), w=1.0 / self.n) for i in idx]

    def predict(self, sims_per_particle: int = 0) -> np.ndarray:
        # 後驗 baseline
        alpha_post = self.alpha0 + self.counts
        base = alpha_post / alpha_post.sum()

        # 粒子加權平均
        ps = np.stack([pt.p for pt in self.particles], axis=0)           # (n,3)
        ws = np.array([pt.w for pt in self.particles], dtype=np.float64) # (n,)
        ws = ws / max(1e-12, float(ws.sum()))
        mix = (ps * ws[:, None]).sum(axis=0)

        # 混合 baseline，避免過度自信
        out = _normalize(0.6 * mix + 0.4 * base).astype(np.float32)
        return out
