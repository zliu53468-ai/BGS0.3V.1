# -*- coding: utf-8 -*-
"""
pfilter.py — 粒子濾波/貝氏簡化器（支援「獨立模式」與「學習模式」）
- 獨立模式（預設，MODEL_MODE=indep）：每手獨立預測，不吃歷史；update_outcome 不影響下手。
- 學習模式（MODEL_MODE=learn）：以Dirichlet後驗做溫和學習（含遺忘），仍保持穩定不追單邊。

你可用環境變數調整：
  MODEL_MODE        indep | learn（預設 indep）
  PRIOR_B/P/T       先驗機率（預設 0.458/0.446/0.096）
  PRIOR_STRENGTH    先驗權重，學習模式才用（預設 40）
  PF_DECAY          遺忘係數（學習模式）預設 0.985
  TIE_MIN/TIE_MAX   和局夾制（預設 0.03/0.18）
  PROB_JITTER       獨立模式下針對單手的極小抖動（避免完全固定）預設 0.0（關閉）
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import Optional

# --- 模式與先驗 ---
MODEL_MODE       = os.getenv("MODEL_MODE", "indep").strip().lower()  # indep | learn
PRIOR_B          = float(os.getenv("PRIOR_B", "0.458"))
PRIOR_P          = float(os.getenv("PRIOR_P", "0.446"))
PRIOR_T          = float(os.getenv("PRIOR_T", "0.096"))
PRIOR_STRENGTH   = float(os.getenv("PRIOR_STRENGTH", "40"))
PF_DECAY         = float(os.getenv("PF_DECAY", "0.985"))
TIE_MIN          = float(os.getenv("TIE_MIN", "0.03"))
TIE_MAX          = float(os.getenv("TIE_MAX", "0.18"))
PROB_JITTER      = float(os.getenv("PROB_JITTER", "0.0"))  # 0~0.01 建議；0 代表關

EPS              = 1e-9

@dataclass
class OutcomePF:
    # 與你server.py保留相容接口（不破壞參數）
    decks: int = int(os.getenv("DECKS", "6"))
    seed: int = int(os.getenv("SEED", "42"))
    n_particles: int = int(os.getenv("PF_N", "80"))
    sims_lik: int = int(os.getenv("PF_UPD_SIMS", "36"))
    resample_thr: float = float(os.getenv("PF_RESAMPLE", "0.73"))
    backend: str = os.getenv("PF_BACKEND", "mc").strip().lower()
    dirichlet_eps: float = float(os.getenv("PF_DIR_EPS", "0.012"))
    stability_factor: float = float(os.getenv("PF_STAB_FACTOR", "0.8"))

    prev_p_pts: Optional[int] = None
    prev_b_pts: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.prior = np.array([PRIOR_B, PRIOR_P, PRIOR_T], dtype=np.float64)
        self.prior = self.prior / self.prior.sum()
        # 學習模式才會用到的後驗累計
        self.counts = np.zeros(3, dtype=np.float64)

    # 只保留介面，不在獨立模式中引入跨手依賴
    def update_point_history(self, p_pts: int, b_pts: int):
        self.prev_p_pts = int(p_pts)
        self.prev_b_pts = int(b_pts)

    def update_outcome(self, outcome: int):
        if MODEL_MODE != "learn":
            return
        # 學習模式：做「遺忘+增量」
        self.counts *= PF_DECAY
        if outcome in (0, 1, 2):
            self.counts[outcome] += 1.0

    def _posterior_mean(self) -> np.ndarray:
        # 學習模式的後驗；獨立模式其實不會用到counts（保持0）
        post = self.prior * PRIOR_STRENGTH + self.counts
        post = np.clip(post, EPS, None)
        return post / post.sum()

    def predict(self, sims_per_particle: int = 30) -> np.ndarray:
        if MODEL_MODE == "indep":
            probs = self.prior.copy()
            # 可選：對單手加極小抖動，避免完全固定（不會導致追單邊）
            if PROB_JITTER > 0:
                jitter = self.rng.normal(0.0, PROB_JITTER, size=3)
                probs = probs + jitter
                probs = np.clip(probs, EPS, None)
                probs = probs / probs.sum()
        else:
            probs = self._posterior_mean()

        # 和局夾制
        probs[2] = np.clip(probs[2], TIE_MIN, TIE_MAX)
        probs = np.clip(probs, EPS, None)
        probs = probs / probs.sum()
        return probs.astype(np.float32)
