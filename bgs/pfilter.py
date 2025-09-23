# -*- coding: utf-8 -*-
"""
pfilter.py — 粒子濾波/貝氏簡化器（路線A：純PF｜獨立手｜對稱先驗｜關閉點差）
- 獨立模式（預設，MODEL_MODE=indep）：
  每手獨立預測，不吃歷史；僅用先驗 + 輕微抖動避免機率長時間黏一邊。
- 學習模式（MODEL_MODE=learn）：
  保留相容介面，但預設不啟用；若要啟用可自行設 MODEL_MODE=learn。

環境變數（不設則用預設值）：
  MODEL_MODE       indep | learn（預設 indep）
  PRIOR_B/P/T      先驗機率（預設 0.452/0.452/0.096，對稱以免偏莊）
  PRIOR_STRENGTH   先驗權重（learn 模式用，預設 40）
  PF_DECAY         遺忘係數（learn 模式用，預設 0.985）
  TIE_MIN/TIE_MAX  和局上下限（預設 0.03/0.18）
  PROB_JITTER      單手抖動幅度（預設 0.006，建議 0~0.01）

注意：
- 已關閉「上一手點差微調」，避免造成方向來回翻（鬼打牆）。
- 保留 update_outcome / counts 以與 server.py 相容；但在 indep 模式下不會生效。
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import Optional

# === 模式與先驗（預設對稱，避免起手偏莊） ===
MODEL_MODE       = os.getenv("MODEL_MODE", "indep").strip().lower()  # indep | learn
PRIOR_B          = float(os.getenv("PRIOR_B", "0.452"))
PRIOR_P          = float(os.getenv("PRIOR_P", "0.452"))
PRIOR_T          = float(os.getenv("PRIOR_T", "0.096"))
PRIOR_STRENGTH   = float(os.getenv("PRIOR_STRENGTH", "40"))
PF_DECAY         = float(os.getenv("PF_DECAY", "0.985"))
TIE_MIN          = float(os.getenv("TIE_MIN", "0.03"))
TIE_MAX          = float(os.getenv("TIE_MAX", "0.18"))
PROB_JITTER      = float(os.getenv("PROB_JITTER", "0.006"))  # 稍提高，避免長黏

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
        # learn 模式才會累積
        self.counts = np.zeros(3, dtype=np.float64)

    # 接口：記錄上一手點（路線A不使用它來調整機率，但保留API）
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

    def predict(self, sims_per_particle: int = 30) -> np.ndarray:
        # 路線A：獨立手，不做任何點差微調與跨手平滑
        if MODEL_MODE == "indep":
            probs = self.prior.copy()
            if PROB_JITTER > 0:
                jitter = self.rng.normal(0.0, PROB_JITTER, size=3)
                probs = probs + jitter
        else:
            probs = self._posterior_mean()

        # 夾制和局區間並正規化
        probs[2] = np.clip(probs[2], TIE_MIN, TIE_MAX)
        probs = np.clip(probs, EPS, None)
        probs = probs / probs.sum()
        return probs.astype(np.float32)
