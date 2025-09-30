# bgs/pfilter.py  — 安全簡化版粒子濾波（介面相容）
# Author: 親愛的 x GPT-5 Thinking

from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

# 狀態：只追蹤 B/P/T 機率向量，讓它能被簡單更新與抽樣
@dataclass
class Particle:
    p: np.ndarray  # shape=(3,), sums to 1.0
    w: float       # 權重

def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 1e-12, None)
    return v / v.sum()

class OutcomePF:
    """
    符合 server.py 的使用介面：
      PF = OutcomePF(decks, seed, n_particles, sims_lik, resample_thr, dirichlet_alpha, use_exact)
      PF.update_outcome(o)   # o in {0(B),1(P),2(T)}
      PF.predict(sims_per_particle) -> np.array([pB,pP,pT], dtype=float32)
    """
    def __init__(
        self,
        decks: int = 8,
        seed: int = 42,
        n_particles: int = 200,
        sims_lik: int = 80,
        resample_thr: float = 0.5,
        dirichlet_alpha: float = 0.8,
        use_exact: bool = False,
    ) -> None:
        self.decks = int(decks)
        self.rng = np.random.default_rng(int(seed))
        random.seed(int(seed))

        self.n = int(n_particles)
        self.sims_lik = int(sims_lik)
        self.resample_thr = float(resample_thr)
        self.alpha0 = float(dirichlet_alpha)
        self.use_exact = bool(use_exact)

        # 後驗計數（觀測到的 B/P/T 次數）
        self.counts = np.zeros(3, dtype=np.float64)

        # 初始化粒子：由 Dirichlet(alpha0) 取樣
        self.particles: List[Particle] = []
        for _ in range(self.n):
            p = self.rng.dirichlet([self.alpha0, self.alpha0, self.alpha0]).astype(np.float64)
            self.particles.append(Particle(p=p, w=1.0 / self.n))

    # 有新一局結果（0=莊,1=閒,2=和）就更新後驗與粒子
    def update_outcome(self, outcome: int) -> None:
        if outcome not in (0, 1, 2):
            return
        self.counts[outcome] += 1.0

        # 重要性重加權：越接近觀測結果的粒子權重越大
        for pt in self.particles:
            like = float(np.clip(pt.p[outcome], 1e-12, None))
            pt.w *= like

        # 正規化權重
        w = np.array([pt.w for pt in self.particles], dtype=np.float64)
        w_sum = float(w.sum())
        if w_sum <= 0:
            w[:] = 1.0 / self.n
        else:
            w /= w_sum
        for i, pt in enumerate(self.particles):
            pt.w = float(w[i])

        # 有效粒子數 (ESS) 低於門檻則重抽樣
        ess = 1.0 / float((w ** 2).sum())
        if ess / self.n < self.resample_thr:
            self._resample()

        # 針對每個粒子做輕微後驗漂移：Dirichlet(α + counts)
        alpha_post = self.alpha0 + self.counts
        for pt in self.particles:
            pt.p = self.rng.dirichlet(alpha_post).astype(np.float64)

    def _resample(self) -> None:
        w = np.array([pt.w for pt in self.particles], dtype=np.float64)
        if w.sum() <= 0:
            w[:] = 1.0 / self.n
        else:
            w = w / w.sum()
        idx = self.rng.choice(len(self.particles), size=self.n, replace=True, p=w)
        new_particles = [Particle(p=self.particles[i].p.copy(), w=1.0 / self.n) for i in idx]
        self.particles = new_particles

    def predict(self, sims_per_particle: int = 200) -> np.ndarray:
        """
        回傳下一局的機率向量。用加權平均 + 少量 MC 擾動。
        """
        # 先用 Dirichlet(α+counts) 的後驗期望當 baseline
        alpha_post = self.alpha0 + self.counts
        base = alpha_post / alpha_post.sum()

        # 粒子加權平均
        ps = np.stack([pt.p for pt in self.particles], axis=0)  # (n,3)
        ws = np.array([pt.w for pt in self.particles], dtype=np.float64).reshape(-1, 1)  # (n,1)
        mix = (ps * ws).sum(axis=0)

        # 小幅混合，避免過度自信
        out = _normalize(0.6 * mix + 0.4 * base).astype(np.float32)
        return out
