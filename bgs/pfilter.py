# -*- coding: utf-8 -*-
"""
pfilter.py — 粒子濾波/貝氏簡化器（支援「獨立模式」與「學習模式」）
- 獨立模式（預設，MODEL_MODE=indep）：
  每手獨立預測，不吃歷史；會依「上一手點數差」做單手微調 + 輕微抖動，避免百分比卡死。
- 學習模式（MODEL_MODE=learn）：
  Dirichlet 後驗做溫和學習（含遺忘），長打時可慢慢貼近實況。

環境變數（需要再調時才設；不設有合理預設）：
  MODEL_MODE       indep | learn（預設 indep）
  PRIOR_B/P/T      先驗機率（預設 0.458/0.446/0.096）
  PRIOR_STRENGTH   先驗權重（學習模式用，預設 40）
  PF_DECAY         遺忘係數（學習模式用，預設 0.985）
  TIE_MIN/TIE_MAX  和局上下限（預設 0.03/0.18）
  PROB_JITTER      單手抖動幅度（預設 0.003，建議 0~0.01）
  GAP_BOOST        點差增強係數（預設 0.010）
  GAP_MAX          點差作用上限（預設 6）
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
PROB_JITTER      = float(os.getenv("PROB_JITTER", "0.003"))  # 避免機率固定
GAP_BOOST        = float(os.getenv("GAP_BOOST", "0.010"))    # 點差加權
GAP_MAX          = float(os.getenv("GAP_MAX", "6"))          # 點差作用上限

EPS              = 1e-9

@dataclass
class OutcomePF:
    # 與 server.py 保持接口相容
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
        # 學習模式才會累積
        self.counts = np.zeros(3, dtype=np.float64)

    # 接口：記錄上一手點
    def update_point_history(self, p_pts: int, b_pts: int):
        self.prev_p_pts = int(p_pts)
        self.prev_b_pts = int(b_pts)

    # 接口：學習模式才更新；獨立模式直接忽略
    def update_outcome(self, outcome: int):
        if MODEL_MODE != "learn":
            return
        self.counts *= PF_DECAY
        if outcome in (0, 1, 2):
            self.counts[outcome] += 1.0

    def _posterior_mean(self) -> np.ndarray:
        post = self.prior * PRIOR_STRENGTH + self.counts
        post = np.clip(post, EPS, None)
        return post / post.sum()

    def _apply_gap_adjust(self, probs: np.ndarray) -> np.ndarray:
        """依「上一手點差」做單手微調（不跨局累加、不追單邊）。修正：反轉 boost 方向，實現反追（見莊打閒、見閒打莊）。"""
        if self.prev_p_pts is None or self.prev_b_pts is None:
            return probs
        gap = abs(self.prev_p_pts - self.prev_b_pts)
        if gap <= 0:
            return probs
        gap_eff = min(gap, GAP_MAX)
        boost = GAP_BOOST * gap_eff
        if self.prev_p_pts > self.prev_b_pts:
            probs[0] += boost  # 閒贏 → 下手莊微強化（反轉原邏輯）
        elif self.prev_b_pts > self.prev_p_pts:
            probs[1] += boost  # 莊贏 → 下手閒微強化（反轉原邏輯）
        return probs

    def predict(self, sims_per_particle: int = 30) -> np.ndarray:
        if MODEL_MODE == "indep":
            probs = self.prior.copy()
            # 單手點差微調（本手即用，不累積） - 修正：移除以實現獨立預測，提升準確度並減少龍風險
            # probs = self._apply_gap_adjust(probs)
            # 輕微抖動，避免機率長時間固定
            if PROB_JITTER > 0:
                jitter = self.rng.normal(0.0, PROB_JITTER, size=3)
                probs = probs + jitter
        else:
            probs = self._posterior_mean()
            # probs = self._apply_gap_adjust(probs)  # 修正：移除

        # 夾制和局區間
        probs[2] = np.clip(probs[2], TIE_MIN, TIE_MAX)
        probs = np.clip(probs, EPS, None)
        probs = probs / probs.sum()
        return probs.astype(np.float32)
