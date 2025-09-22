# -*- coding: utf-8 -*-
"""
pfilter.py — Env‑driven Particle Filter (Deepseak‑sync 2025‑09‑22)
=================================================================
• **目的**：讓核心 PF 參數完全由環境變數控制，與 server.py 相同邏輯同步。
• **影響檔**：僅改動常數定義 + dataclass `OutcomePF` 預設值，其餘演算法原封不動。
• **相容性**：現有 server.py 不需改；如果外部程式手動傳參，仍可覆蓋這些預設。

✔ 新增 `os` 匯入與下列 env 映射
   PF_N / PF_UPD_SIMS / PF_RESAMPLE / PF_DIR_EPS / PF_BACKEND / PF_STAB_FACTOR / PF_DECKS / PF_SEED

✔ `OutcomePF` now defaults to these env‑derived values.

"""

import os  # <— 新增，需在所有 numpy 前 import 亦可
import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional, Dict

# -- Local deps -------------------------------------------------------------
try:
    from .deplete import init_counts  # type: ignore
except Exception:
    from deplete import init_counts   # type: ignore

# ---------------------------------------------------------------------------
#  🔧  環境變數 → 全域預設
# ---------------------------------------------------------------------------
PF_N            = int(os.getenv("PF_N", "120"))            # 粒子數量
PF_UPD_SIMS     = int(os.getenv("PF_UPD_SIMS", "40"))      # 每粒子 MC 次數 (lik)
PF_RESAMPLE     = float(os.getenv("PF_RESAMPLE", "0.85"))  # 重採樣門檻 (Neff/N)
PF_DIR_EPS      = float(os.getenv("PF_DIR_EPS", "0.025"))  # Dirichlet eps
PF_BACKEND      = os.getenv("PF_BACKEND", "mc").strip().lower()  # exact / mc
PF_STAB_FACTOR  = float(os.getenv("PF_STAB_FACTOR", "0.8"))
PF_DECKS        = int(os.getenv("DECKS", "8"))
PF_SEED         = int(os.getenv("PF_SEED", "42"))
# ---------------------------------------------------------------------------

# ---------- 百家樂規則 (unchanged) -----------------------------------------

def points_add(a, b):
    return (a + b) % 10


def third_player(p_sum):
    return p_sum <= 5


def third_banker(b_sum, p3):
    if b_sum <= 2:
        return True
    if b_sum == 3:
        return p3 != 8
    if b_sum == 4:
        return p3 in (2, 3, 4, 5, 6, 7)
    if b_sum == 5:
        return p3 in (4, 5, 6, 7)
    if b_sum == 6:
        return p3 in (6, 7)
    return False

# ---------------- <以下演算法區段全部未變動> ------------------------------
# (因篇幅，同前版內容保持一致，只展示改動區域) --------------------------

# ... 省略 _prob_draw_seq_4 / calibration / _rb_exact_prob / _mc_prob ...

# ---------- 粒子濾波主體 ----------------------------------------------------
@dataclass
class OutcomePF:
    """Outcome‑only Particle Filter with env‑tunable defaults."""

    # 將原本硬寫常數改為環境變數預設
    decks: int = PF_DECKS
    seed: int = PF_SEED
    n_particles: int = PF_N
    sims_lik: int = PF_UPD_SIMS
    resample_thr: float = PF_RESAMPLE
    backend: Literal["exact", "mc"] = PF_BACKEND  # noqa: E501
    dirichlet_eps: float = PF_DIR_EPS
    stability_factor: float = PF_STAB_FACTOR

    prev_p_pts: Optional[int] = None
    prev_b_pts: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        base = init_counts(self.decks).astype(np.int64)
        self.p_counts = np.stack([base.copy() for _ in range(self.n_particles)], axis=0)
        self.weights = np.ones(self.n_particles, dtype=np.float64) / self.n_particles
        self.prediction_history = []
        self.point_diff_history = []

    # 其餘方法 **完全不變**
    # _forward_prob • update_point_history • update_outcome • predict
    # get_reversal_probability • get_accuracy_metrics

# ---------------------------------------------------------------------------
# END OF FILE — 與 server.py 同步後僅需修改環境變數即可調參。
