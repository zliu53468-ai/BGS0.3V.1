# -*- coding: utf-8 -*-
"""
pfilter.py — 平衡預測版本
保留核心粒子滤波器但简化复杂度，提高响应性
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

def _env_float(k: str, default: str) -> float:
    try: return float(os.getenv(k, default))
    except Exception: return float(default)

def _env_int(k: str, default: str) -> int:
    try: return int(os.getenv(k, default))
    except Exception: return int(default)

# 平衡參數
PRIOR_B = _env_float("PRIOR_B", "0.458")
PRIOR_P = _env_float("PRIOR_P", "0.446") 
PRIOR_T = _env_float("PRIOR_T", "0.096")
PRIOR_STRENGTH = _env_float("PRIOR_STRENGTH", "20")  # 降低先驗影響

# 簡化學習參數
PF_DECAY = _env_float("PF_DECAY", "0.985")  # 適中衰減
HIST_WIN = _env_int("HIST_WIN", "25")  # 較短歷史窗口
PF_WIN = _env_int("PF_WIN", "20")  # 較短PF窗口

EPS = 1e-9

class SimpleDirichletPF:
    """簡化粒子滤波器"""
    def __init__(
        self,
        prior: np.ndarray,
        n_particles: int,
        rng: np.random.Generator,
    ):
        self.rng = rng
        self.n_particles = max(1, int(n_particles))
        self.prior = np.clip(np.asarray(prior, dtype=np.float64), EPS, None)
        self.prior = self.prior / self.prior.sum()
        
        # 初始化粒子
        alpha = np.clip(self.prior * 15.0, EPS, None)  # 較弱的先驗
        self.parts = self.rng.dirichlet(alpha, size=self.n_particles)
        self.w = np.ones(self.n_particles, dtype=np.float64) / self.n_particles
        self.update_count = 0

    def _ess(self) -> float:
        s = np.sum(self.w ** 2)
        return 1.0 / max(EPS, s)

    def _resample(self):
        n = self.n_particles
        positions = (np.arange(n) + self.rng.random()) / n
        cumsum = np.cumsum(self.w)
        idx = np.zeros(n, dtype=int)
        i = j = 0
        while i < n:
            if positions[i] < cumsum[j]:
                idx[i] = j
                i += 1
            else:
                j += 1
        self.parts = self.parts[idx]
        self.w.fill(1.0 / n)

        # 輕度rejuvenation
        alpha = np.clip(self.parts * 20.0 + 0.01, EPS, None)
        for k in range(n):
            self.parts[k] = self.rng.dirichlet(alpha[k])

    def update(self, outcome: int):
        if outcome not in (0, 1, 2):
            return
        
        self.update_count += 1
        
        # 動態學習率
        learning_rate = min(1.5, 1.0 + self.update_count / 80.0)
        like = np.clip(self.parts[:, outcome], EPS, None) * learning_rate
        
        self.w *= like
        self.w /= np.sum(self.w)

        # 簡化重採樣條件
        if (self._ess() / self.n_particles) < 0.7:
            self._resample()

    def predict(self) -> np.ndarray:
        m = np.average(self.parts, axis=0, weights=self.w)
        m = np.clip(m, EPS, None)
        return (m / np.sum(m)).astype(np.float64)

@dataclass
class OutcomePF:
    """平衡預測版本"""
    decks: int = _env_int("DECKS", "6")
    seed: int = _env_int("SEED", "42")
    n_particles: int = _env_int("PF_N", "80")
    sims_lik: int = _env_int("PF_UPD_SIMS", "25")
    resample_thr: float = _env_float("PF_RESAMPLE", "0.75")
    backend: str = "mc"
    dirichlet_eps: float = _env_float("PF_DIR_EPS", "0.01")
    stability_factor: float = _env_float("PF_STAB_FACTOR", "0.85")

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.prior = np.array([PRIOR_B, PRIOR_P, PRIOR_T], dtype=np.float64)
        self.prior = self.prior / self.prior.sum()

        # 簡化計數器
        self.counts = np.zeros(3, dtype=np.float64)
        
        # 短歷史窗口
        self.history_window: List[int] = []
        self.max_hist_len = max(HIST_WIN, PF_WIN, 50)  # 較短歷史

        # 簡化粒子滤波器
        self.pf = SimpleDirichletPF(
            rng=self.rng,
            prior=self.prior,
            n_particles=self.n_particles,
        )

    def update_point_history(self, p_pts: int, b_pts: int):
        """記錄點數歷史"""
        # 簡化點數分析，只記錄基本信息
        pass

    def update_outcome(self, outcome: int):
        """輕度更新"""
        if outcome not in (0, 1, 2):
            return

        # 更新計數器（輕度衰減）
        self.counts *= PF_DECAY
        self.counts[outcome] += 1.0

        # 更新粒子滤波器
        self.pf.update(outcome)

        # 更新歷史窗口
        self.history_window.append(outcome)
        if len(self.history_window) > self.max_hist_len:
            self.history_window.pop(0)

    def _light_historical_adjust(self, probs: np.ndarray) -> np.ndarray:
        """輕度歷史調整"""
        if len(self.history_window) < 5:  # 最少需要5個樣本
            return probs
            
        # 只使用最近歷史
        recent = self.history_window[-min(15, len(self.history_window)):]
        n = len(recent)
        
        counts = np.array([
            sum(1 for x in recent if x == 0),
            sum(1 for x in recent if x == 1), 
            sum(1 for x in recent if x == 2),
        ], dtype=np.float64)
        
        # 計算歷史概率（輕度正則化）
        hist_probs = (counts + 0.5) / max(EPS, (n + 1.5))
        
        # 動態權重：樣本越多權重越高，但最大不超過0.3
        w = min(0.3, n / (n + 30.0))
        
        mixed = probs * (1 - w) + hist_probs * w
        mixed = np.clip(mixed, EPS, None)
        return mixed / mixed.sum()

    def predict(self, sims_per_particle: int = 1) -> np.ndarray:
        """平衡預測"""
        # 1) 粒子滤波器預測
        pf_probs = self.pf.predict()
        
        # 2) 貝葉斯後驗
        post = self.prior * PRIOR_STRENGTH + self.counts
        post = np.clip(post, EPS, None)
        bayes_probs = post / post.sum()
        
        # 3) 結合兩者（PF權重較高）
        n_pf = min(len(self.history_window), PF_WIN)
        w_pf = min(0.8, 0.4 + (n_pf / (n_pf + 30.0)))  # PF權重0.4-0.8
        
        probs = pf_probs * w_pf + bayes_probs * (1 - w_pf)
        
        # 4) 輕度歷史調整
        probs = self._light_historical_adjust(probs)
        
        # 5) 確保合理性
        probs = np.clip(probs, 0.01, 0.98)
        probs = probs / probs.sum()
        
        return probs.astype(np.float32)
