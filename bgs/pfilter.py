# -*- coding: utf-8 -*-
"""
pfilter.py — 粒子濾波/貝式強化版（穩定學習 + 點差校正）
說明：
- 使用 Dirichlet–Multinomial 作為「莊/閒/和」先驗 + 後驗學習
- 每局 outcome 做「遺忘因子 + 增量更新」
- 以上一手點數差 gap 做小幅校正（強化勝方）
- 不再誤用「牌點桶」當成 outcome 維度，避免學習方向錯位
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import Optional

# ---- 可調參數（環境變數） -----------------------------------------
PF_SEED         = int(os.getenv("SEED", "42"))

# 先驗機率（莊/閒/和）
PRIOR_B         = float(os.getenv("PRIOR_B", "0.458"))
PRIOR_P         = float(os.getenv("PRIOR_P", "0.446"))
PRIOR_T         = float(os.getenv("PRIOR_T", "0.096"))
PRIOR_STRENGTH  = float(os.getenv("PRIOR_STRENGTH", "40"))   # 先驗權重

# 遺忘（0.0~1.0，越接近1越慢忘，預設略慢忘）
PF_DECAY        = float(os.getenv("PF_DECAY", "0.985"))

# 點數差校正強度（每 1 點差增加多少機率點）
GAP_BOOST       = float(os.getenv("GAP_BOOST", "0.010"))

# 和局上下限
TIE_MIN         = float(os.getenv("TIE_MIN", "0.03"))
TIE_MAX         = float(os.getenv("TIE_MAX", "0.18"))

# 安全夾制
EPS             = 1e-9

@dataclass
class OutcomePF:
    # 下面幾個參數維持與舊版相容，但內部不再用到「牌點桶」
    decks: int = int(os.getenv("DECKS", "6"))
    seed: int = PF_SEED
    n_particles: int = int(os.getenv("PF_N", "80"))
    sims_lik: int = int(os.getenv("PF_UPD_SIMS", "36"))
    resample_thr: float = float(os.getenv("PF_RESAMPLE", "0.73"))
    backend: str = os.getenv("PF_BACKEND", "mc").strip().lower()
    dirichlet_eps: float = float(os.getenv("PF_DIR_EPS", "0.012"))
    stability_factor: float = float(os.getenv("PF_STAB_FACTOR", "0.8"))

    # 狀態
    prev_p_pts: Optional[int] = None
    prev_b_pts: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

        # Dirichlet 先驗 / 後驗
        self.prior = np.array([PRIOR_B, PRIOR_P, PRIOR_T], dtype=np.float64)
        self.prior = self.prior / self.prior.sum()
        self.counts = np.zeros(3, dtype=np.float64)  # 後驗增量（可遺忘）

        # 紀錄
        self.point_diff_history = []
        self.last_result = []

    # ========== 資料/學習 ==========
    def update_point_history(self, p_pts: int, b_pts: int):
        self.prev_p_pts = int(p_pts)
        self.prev_b_pts = int(b_pts)
        self.point_diff_history.append(self.prev_p_pts - self.prev_b_pts)
        if len(self.point_diff_history) > 200:
            self.point_diff_history = self.point_diff_history[-200:]

    def update_outcome(self, outcome: int):
        """
        outcome: 0=莊 1=閒 2=和
        做「遺忘 + 增量」：counts = decay*counts；counts[outcome]+=1
        """
        self.counts *= PF_DECAY
        if outcome in (0, 1, 2):
            self.counts[outcome] += 1.0
            self.last_result.append(outcome)
            if len(self.last_result) > 200:
                self.last_result = self.last_result[-200:]

    # ========== 預測 ==========
    def _posterior_mean(self) -> np.ndarray:
        post = self.prior * PRIOR_STRENGTH + self.counts
        post = np.clip(post, EPS, None)
        return post / post.sum()

    def predict(self, sims_per_particle: int = 30) -> np.ndarray:
        """
        回傳 [B, P, T] 機率；之後會在 server.py 做溫度縮放/EMA 平滑
        """
        probs = self._posterior_mean()

        # 依上一手點差做小幅校正（強化上一手勝方）
        if self.prev_p_pts is not None and self.prev_b_pts is not None:
            gap = abs(self.prev_p_pts - self.prev_b_pts)
            if gap > 0:
                if self.prev_p_pts > self.prev_b_pts:
                    probs[1] += GAP_BOOST * gap  # 閒
                elif self.prev_b_pts > self.prev_p_pts:
                    probs[0] += GAP_BOOST * gap  # 莊

        # 夾制和局合理區間
        probs[2] = np.clip(probs[2], TIE_MIN, TIE_MAX)

        # 正規化
        probs = np.clip(probs, EPS, None)
        probs = probs / probs.sum()
        return probs.astype(np.float32)
