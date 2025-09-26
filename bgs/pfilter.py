# -*- coding: utf-8 -*-
"""
pfilter.py — 粒子濾波/貝氏簡化器（強化版）
修補點：
1) 動態和局範圍：加入 Beta 先驗 + EMA 平滑；樣本不足時保持原範圍
2) PF/歷史視窗：使用較長視窗 + Dirichlet/Laplace 平滑，避免 0 機率
3) 學習權重 pf_weight：依有效樣本量自動調整，與視窗一致
4) 擾動策略：採用 Dirichlet 抖動（機率單純上），避免高斯剪裁偏差
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

# ---------- Helpers / Env ----------
def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1","true","yes","on")

def _env_float(name: str, default: str) -> float:
    try: return float(os.getenv(name, default))
    except: return float(default)

def _env_int(name: str, default: str) -> int:
    try: return int(os.getenv(name, default))
    except: return int(default)

# === Modes & priors ===
MODEL_MODE = os.getenv("MODEL_MODE", "indep").strip().lower()  # indep | learn
PRIOR_B = _env_float("PRIOR_B", "0.452")
PRIOR_P = _env_float("PRIOR_P", "0.452")
PRIOR_T = _env_float("PRIOR_T", "0.096")
PRIOR_STRENGTH = _env_float("PRIOR_STRENGTH", "40")

# PF decay (used by learn mode)
PF_DECAY = _env_float("PF_DECAY", "0.985")

# Tie clamp baseline (global guard rails)
TIE_MIN = _env_float("TIE_MIN", "0.03")
TIE_MAX = _env_float("TIE_MAX", "0.18")
TIE_MAX_CAP = _env_float("TIE_MAX_CAP", "0.25")  # 絕對上限保護
TIE_MIN_FLOOR = _env_float("TIE_MIN_FLOOR", "0.01")

# Dynamic tie settings (smoothed)
DYNAMIC_TIE_RANGE = _env_flag("DYNAMIC_TIE_RANGE", "1")
TIE_BETA_A = _env_float("TIE_BETA_A", "9.6")     # Beta 先驗：以 9.6% * 100 手的概念
TIE_BETA_B = _env_float("TIE_BETA_B", "90.4")
TIE_EMA_ALPHA = _env_float("TIE_EMA_ALPHA", "0.2")
TIE_MIN_SAMPLES = _env_int("TIE_MIN_SAMPLES", "40")  # 样本不足時不動態調整
TIE_DELTA = _env_float("TIE_DELTA", "0.35")          # ±35% 漂移範圍（較保守）

# Indeps / history smoothing
PROB_JITTER = _env_float("PROB_JITTER", "0.006")  # 抖動強度（僅控制 Dirichlet 強度）
HISTORICAL_WEIGHT = _env_float("HISTORICAL_WEIGHT", "0.2")
HIST_WIN = _env_int("HIST_WIN", "60")             # 歷史視窗（比 20 大，降噪）
HIST_PSEUDO = _env_float("HIST_PSEUDO", "1.0")    # Laplace 偽計數
HIST_WEIGHT_MAX = _env_float("HIST_WEIGHT_MAX", "0.35")  # indep混合上限

# Particle filter approx
PF_WIN = _env_int("PF_WIN", "50")                 # PF 近似視窗長度（比 10 大）
PF_ALPHA = _env_float("PF_ALPHA", "0.5")          # Dirichlet/Laplace 平滑 α
PF_WEIGHT_MAX = _env_float("PF_WEIGHT_MAX", "0.7")
PF_WEIGHT_K = _env_float("PF_WEIGHT_K", "80.0")   # 權重飽和平滑常數

EPS = 1e-9

@dataclass
class OutcomePF:
    decks: int = _env_int("DECKS", "6")
    seed: int = _env_int("SEED", "42")
    n_particles: int = _env_int("PF_N", "80")
    sims_lik: int = _env_int("PF_UPD_SIMS", "36")
    resample_thr: float = _env_float("PF_RESAMPLE", "0.73")
    backend: str = os.getenv("PF_BACKEND", "mc").strip().lower()
    dirichlet_eps: float = _env_float("PF_DIR_EPS", "0.012")
    stability_factor: float = _env_float("PF_STAB_FACTOR", "0.8")

    prev_p_pts: Optional[int] = None
    prev_b_pts: Optional[int] = None

    # internal states
    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

        # base prior
        self.prior = np.array([PRIOR_B, PRIOR_P, PRIOR_T], dtype=np.float64)
        self.prior = self.prior / self.prior.sum()

        # long-term counts for bayes posterior (learn)
        self.counts = np.zeros(3, dtype=np.float64)

        # rolling window outcomes (0:B, 1:P, 2:T)
        self.history_window: List[int] = []
        self.max_hist_len = max(HIST_WIN, PF_WIN, 100)  # 至少保 100，避免過敏

        # dynamic tie trackers
        self.t_ema = None           # tie ratio EMA
        self.tie_samples = 0        # 近端樣本量
        self.adaptive_tie_min = TIE_MIN
        self.adaptive_tie_max = TIE_MAX

    # -------- external updates --------
    def update_point_history(self, p_pts: int, b_pts: int):
        self.prev_p_pts = int(p_pts)
        self.prev_b_pts = int(b_pts)

    def update_outcome(self, outcome: int):
        if outcome not in (0, 1, 2):
            return

        # learn: long-term decayed counts
        if MODEL_MODE == "learn":
            self.counts *= PF_DECAY
            self.counts[outcome] += 1.0

        # rolling window (cap)
        self.history_window.append(outcome)
        if len(self.history_window) > self.max_hist_len:
            self.history_window.pop(0)

        # update tie trackers
        if DYNAMIC_TIE_RANGE:
            self._update_tie_trackers()

    # -------- dynamic tie (smoothed) --------
    def _update_tie_trackers(self):
        # 使用較長視窗估計近期 tie 比例，避免 20 手零和→立刻崩到很低
        window = self.history_window[-HIST_WIN:] if len(self.history_window) >= HIST_WIN else self.history_window[:]
        n = len(window)
        if n == 0:
            return

        tie_cnt = sum(1 for x in window if x == 2)
        # 以 Beta(A,B) 與視窗計數形成 posterior 的期望
        beta_a = TIE_BETA_A + tie_cnt
        beta_b = TIE_BETA_B + (n - tie_cnt)
        mu_post = beta_a / max(EPS, (beta_a + beta_b))  # 平滑 tie 比率

        # 再以 EMA 平滑降低短期震盪
        if self.t_ema is None:
            self.t_ema = mu_post
        else:
            a = np.clip(TIE_EMA_ALPHA, 0.0, 1.0)
            self.t_ema = a * mu_post + (1 - a) * self.t_ema

        self.tie_samples = n

        # 樣本不足則維持原始保護欄位，不做動態夾取
        if self.tie_samples < TIE_MIN_SAMPLES:
            self.adaptive_tie_min = TIE_MIN
            self.adaptive_tie_max = TIE_MAX
            return

        # 以 t_ema 為中心做保守 ±delta 漂移，再加上全域 guard rails
        center = float(self.t_ema)
        lo = max(TIE_MIN_FLOOR, TIE_MIN, center * (1 - TIE_DELTA))
        hi = min(TIE_MAX_CAP, max(TIE_MAX, center * (1 + TIE_DELTA)))  # 允許略高於 TIE_MAX，但不超 CAP
        # 保序
        hi = max(hi, lo + 1e-4)
        self.adaptive_tie_min = lo
        self.adaptive_tie_max = hi

    # -------- smoothing utilities --------
    def _dirichlet_jitter(self, probs: np.ndarray, strength: float = 120.0) -> np.ndarray:
        """
        在機率單純上做 Dirichlet 擾動，避免高斯 + clip 的結構性偏差。
        strength 越大 → 抖動越小（α = probs * strength + 1）
        """
        alpha = np.clip(probs, EPS, 1.0) * strength + 1.0
        return self.rng.dirichlet(alpha)

    def _light_historical_update(self, probs: np.ndarray) -> np.ndarray:
        """
        獨立模式：用較長視窗 + 偽計數平滑的歷史比例做溫和混合。
        避免 20 手極端頻率把機率壓到 0 或 1。
        """
        window = self.history_window[-HIST_WIN:]
        n = len(window)
        if n == 0:
            return probs

        counts = np.array([
            sum(1 for x in window if x == 0),
            sum(1 for x in window if x == 1),
            sum(1 for x in window if x == 2)
        ], dtype=np.float64)

        # Laplace/Dirichlet 平滑
        hist_probs = (counts + HIST_PSEUDO) / max(EPS, (n + 3.0 * HIST_PSEUDO))

        # 動態權重：隨樣本量上升，但有上限，並與使用者 HISTORICAL_WEIGHT 取 min
        w_dyn = n / (n + 80.0)  # 平滑上升；n≈80 時 ~0.5
        w = min(HISTORICAL_WEIGHT, HIST_WEIGHT_MAX, w_dyn)
        mixed = probs * (1 - w) + hist_probs * w
        return mixed / mixed.sum()

    def _particle_filter_predict(self) -> np.ndarray:
        """
        PF 近似：使用較長視窗 + Dirichlet 平滑，避免 0 機率。
        """
        window = self.history_window[-PF_WIN:]
        n = len(window)
        if n == 0:
            return self.prior.copy()

        counts = np.array([
            sum(1 for x in window if x == 0),
            sum(1 for x in window if x == 1),
            sum(1 for x in window if x == 2)
        ], dtype=np.float64)

        # Dirichlet/Laplace 平滑後作為 likelihood
        like = (counts + PF_ALPHA) / max(EPS, (n + 3.0 * PF_ALPHA))

        # 後驗 ~ prior * likelihood
        post = self.prior * like
        post = np.clip(post, EPS, None)
        post = post / post.sum()
        return post

    def _posterior_mean(self) -> np.ndarray:
        post = self.prior * PRIOR_STRENGTH + self.counts
        post = np.clip(post, EPS, None)
        return post / post.sum()

    # -------- main predict --------
    def predict(self, sims_per_particle: int = 30) -> np.ndarray:
        # 基礎先驗
        probs = self.prior.copy()

        if MODEL_MODE == "indep":
            # 用平滑的歷史視窗做溫和混合
            probs = self._light_historical_update(probs)

            # 採 Dirichlet 擾動（在機率單純上）
            if PROB_JITTER > 0:
                # PROB_JITTER 作為 1/strength 的近似控制，值越小 → 擾動越小
                strength = max(50.0, min(400.0, 1.0 / max(1e-6, PROB_JITTER)))
                probs = self._dirichlet_jitter(probs, strength)

        elif MODEL_MODE == "learn":
            # PF 近似（短期） vs 後驗均值（長期）
            pf_probs = self._particle_filter_predict()
            bayes_probs = self._posterior_mean()

            # 依 PF 視窗樣本量動態給權重
            n_pf = min(len(self.history_window), PF_WIN)
            # 權重隨樣本量上升，且不超 PF_WEIGHT_MAX
            w_pf = min(PF_WEIGHT_MAX, n_pf / (n_pf + PF_WEIGHT_K))
            probs = pf_probs * w_pf + bayes_probs * (1 - w_pf)

        else:
            # fallback
            probs = self.prior.copy()

        # 動態和局夾取（有樣本才動；無樣本維持全域）
        if DYNAMIC_TIE_RANGE and self.tie_samples >= TIE_MIN_SAMPLES:
            tie_min = self.adaptive_tie_min
            tie_max = self.adaptive_tie_max
        else:
            tie_min, tie_max = TIE_MIN, TIE_MAX

        # 先正規化，再夾 T，最後再正規化一次，避免結構性偏差
        probs = np.clip(probs, EPS, None)
        probs = probs / probs.sum()
        probs[2] = np.clip(probs[2], tie_min, tie_max)
        probs = np.clip(probs, EPS, None)
        probs = probs / probs.sum()

        return probs.astype(np.float32)
