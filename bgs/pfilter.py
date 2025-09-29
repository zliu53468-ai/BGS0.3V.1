# -*- coding: utf-8 -*-
"""
pfilter.py — 粒子濾波/貝氏簡化器（最終版）
- indep：長視窗 + Dirichlet 抖動
- learn：SimpleDirichletPF（Dirichlet 狀態轉移 + ESS 重採樣 + 再活化）
- 點數差偏置：僅在 B/P 子機率內重分配，T 不動（且在 Tie 夾取之後施加）
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

# ---------- Helpers / Env ----------
def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1","true","yes","on")

def _env_float(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return float(default)

def _env_int(name: str, default: str) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return int(default)

# === Modes & priors ===
MODEL_MODE = os.getenv("MODEL_MODE", "indep").strip().lower()  # indep | learn
PRIOR_B = _env_float("PRIOR_B", "0.452")
PRIOR_P = _env_float("PRIOR_P", "0.452")
PRIOR_T = _env_float("PRIOR_T", "0.096")
PRIOR_STRENGTH = _env_float("PRIOR_STRENGTH", "40")

# PF decay (Bayes 長期計數；learn 混合用)
PF_DECAY = _env_float("PF_DECAY", "0.985")

# Tie clamp baseline (global guard rails)
TIE_MIN = _env_float("TIE_MIN", "0.03")
TIE_MAX = _env_float("TIE_MAX", "0.18")
TIE_MAX_CAP = _env_float("TIE_MAX_CAP", "0.25")
TIE_MIN_FLOOR = _env_float("TIE_MIN_FLOOR", "0.01")

# Dynamic tie settings (smoothed)
DYNAMIC_TIE_RANGE = _env_flag("DYNAMIC_TIE_RANGE", "1")
TIE_BETA_A = _env_float("TIE_BETA_A", "9.6")       # ~9.6% * 100 手先驗
TIE_BETA_B = _env_float("TIE_BETA_B", "90.4")
TIE_EMA_ALPHA = _env_float("TIE_EMA_ALPHA", "0.2")
TIE_MIN_SAMPLES = _env_int("TIE_MIN_SAMPLES", "40")
TIE_DELTA = _env_float("TIE_DELTA", "0.35")        # ±35% 漂移幅度

# Indeps / history smoothing
PROB_JITTER = _env_float("PROB_JITTER", "0.006")   # 抖動底強度（越小越穩）
PROB_JITTER_SCALE = _env_float("PROB_JITTER_SCALE", "16.0")
PROB_JITTER_STRENGTH_MAX = _env_float("PROB_JITTER_STRENGTH_MAX", "400.0")
HISTORICAL_WEIGHT = _env_float("HISTORICAL_WEIGHT", "0.2")
HIST_WIN = _env_int("HIST_WIN", "60")
HIST_PSEUDO = _env_float("HIST_PSEUDO", "1.0")
HIST_WEIGHT_MAX = _env_float("HIST_WEIGHT_MAX", "0.35")

# Particle filter（learn 分支混合用權重）
PF_WIN = _env_int("PF_WIN", "50")
PF_ALPHA = _env_float("PF_ALPHA", "0.5")
PF_WEIGHT_MAX = _env_float("PF_WEIGHT_MAX", "0.7")
PF_WEIGHT_K = _env_float("PF_WEIGHT_K", "80.0")

# ===== 真 PF 參數 =====
PF_N = _env_int("PF_N", "120")
PF_RESAMPLE = _env_float("PF_RESAMPLE", "0.73")      # ESS/N 門檻
PF_KAPPA = _env_float("PF_KAPPA", "220.0")           # 狀態轉移濃度（越大越穩）
PF_REJUV = _env_float("PF_REJUV", "220.0")           # 重採樣後再活化濃度
DIRICHLET_EPS = _env_float("PF_DIR_EPS", "0.012")
STABILITY_FACTOR = _env_float("PF_STAB_FACTOR", "0.8")

# ===== 點差偏置（只在 B/P 子空間）=====
POINT_BIAS_ON          = _env_flag("POINT_BIAS_ON", "1")
POINT_BIAS_K           = _env_float("POINT_BIAS_K", "0.35")     # tanh 斜率
POINT_BIAS_MAX_SHIFT   = _env_float("POINT_BIAS_MAX_SHIFT", "0.06")  # P 相對位移上限
POINT_BIAS_MIN_GAP     = _env_int("POINT_BIAS_MIN_GAP", "1")
POINT_BIAS_DAMP_N      = _env_int("POINT_BIAS_DAMP_N", "20")
POINT_BIAS_TIE_DAMP_AT = _env_float("POINT_BIAS_TIE_DAMP_AT", "0.14")
POINT_BIAS_TIE_DAMP    = _env_float("POINT_BIAS_TIE_DAMP", "0.5")

EPS = 1e-9

# ============ SimpleDirichletPF ============

class SimpleDirichletPF:
    """
    在單純形上的 PF：
      - 狀態轉移：theta' ~ Dirichlet(kappa * theta)
      - 權重更新：w *= theta'[obs]（用 log-weights）
      - ESS 退化 → 系統化重採樣 + 再活化（Dirichlet(lam * theta)）
    """
    def __init__(
        self,
        rng: np.random.Generator,
        prior: np.ndarray,
        prior_strength: float,
        n_particles: int,
        resample_thr: float,
        kappa: float,
        rejuvenate: float,
        dirichlet_eps: float,
        stability_factor: float,
    ):
        self.rng = rng
        self.n_particles = int(max(2, n_particles))
        self.resample_thr = float(np.clip(resample_thr, 0.05, 0.99))
        self.kappa = float(max(1.0, kappa))
        self.rejuvenate_alpha = float(max(50.0, rejuvenate))
        self.dirichlet_eps = float(max(EPS, dirichlet_eps))
        self.stability_factor = float(np.clip(stability_factor, 0.05, 2.0))

        p0 = np.clip(prior, EPS, 1.0)
        p0 /= p0.sum()
        alpha0 = p0 * float(max(3.0, prior_strength))

        self.particles = self.rng.dirichlet(alpha0, size=self.n_particles).astype(np.float32)
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=np.float32)
        self.obs_count = 0

    # ---- helpers ----
    def _effective_sample_size(self) -> float:
        w = self.weights.astype(np.float64)
        return 1.0 / max(EPS, np.sum(w * w))

    def _systematic_resample(self):
        N = self.n_particles
        positions = (self.rng.random() + np.arange(N)) / N
        cdf = np.cumsum(self.weights, dtype=np.float64)
        idx = np.searchsorted(cdf, positions, side="right")
        idx = np.clip(idx, 0, N - 1)
        self.particles = self.particles[idx]
        self.weights.fill(1.0 / N)

    def _jitter_strength(self) -> float:
        # 隨觀測量逐步放大 Dirichlet α，避免早期過度分散
        growth = 1.0 + min(5.0, self.obs_count / 60.0)
        return max(1.0, self.kappa * self.stability_factor * growth)

    def _rejuvenate(self, lam: float = None):
        lam = float(self._jitter_strength() if lam is None else lam)
        lam = max(50.0, lam)
        # 逐顆小幅 Dirichlet 抖動
        for i in range(self.n_particles):
            a = np.clip(self.particles[i], EPS, 1.0) * lam + self.dirichlet_eps
            self.particles[i] = self.rng.dirichlet(a).astype(np.float32)

    # ---- PF steps ----
    def propagate(self):
        strength = self._jitter_strength()
        new_particles = np.empty_like(self.particles)
        for i, theta in enumerate(self.particles):
            a = np.clip(theta, EPS, 1.0) * strength + self.dirichlet_eps
            new_particles[i] = self.rng.dirichlet(a)
        self.particles = new_particles.astype(np.float32)

    def update(self, outcome: int):
        if outcome not in (0, 1, 2):
            return
        # 先做狀態轉移（PF 的 predict step）
        self.propagate()

        self.obs_count += 1

        like = np.clip(self.particles[:, outcome].astype(np.float64), EPS, 1.0)
        # log-weights，避免長局下溢
        logw = np.log(self.weights.astype(np.float64) + EPS) + np.log(like)
        logw -= logw.max()
        w = np.exp(logw)
        w_sum = max(EPS, w.sum())
        self.weights = (w / w_sum).astype(np.float32)

        # ESS 退化 → 重採樣 + 再活化
        if self._effective_sample_size() < self.resample_thr * self.n_particles:
            self._systematic_resample()
            self._rejuvenate()

    def predict(self) -> np.ndarray:
        # 不在這裡 propagate；單純回加權平均狀態
        mean = (self.particles.astype(np.float64) * self.weights[:, None].astype(np.float64)).sum(axis=0)
        mean = np.clip(mean, EPS, None)
        mean = (mean / mean.sum()).astype(np.float32)
        return mean

# ============ OutcomePF（外部介面） ============

@dataclass
class OutcomePF:
    decks: int = _env_int("DECKS", "6")
    seed: int = _env_int("SEED", "42")
    n_particles: int = PF_N
    sims_lik: int = _env_int("PF_UPD_SIMS", "36")        # 兼容舊參數（未用）
    resample_thr: float = PF_RESAMPLE
    backend: str = os.getenv("PF_BACKEND", "mc").strip().lower()
    dirichlet_eps: float = DIRICHLET_EPS
    stability_factor: float = STABILITY_FACTOR

    prev_p_pts: Optional[int] = None
    prev_b_pts: Optional[int] = None

    # internal states
    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

        # base prior
        self.prior = np.array([PRIOR_B, PRIOR_P, PRIOR_T], dtype=np.float64)
        self.prior = self.prior / self.prior.sum()

        # long-term counts for bayes posterior (learn 混合用)
        self.counts = np.zeros(3, dtype=np.float64)

        # rolling window outcomes (0:B, 1:P, 2:T)
        self.history_window: List[int] = []
        self.max_hist_len = max(HIST_WIN, PF_WIN, 100)

        # dynamic tie trackers
        self.t_ema = None
        self.tie_samples = 0
        self.adaptive_tie_min = TIE_MIN
        self.adaptive_tie_max = TIE_MAX

        # 真 PF（僅 learn 使用）
        self.pf: Optional[SimpleDirichletPF] = None
        if MODEL_MODE == "learn":
            self.pf = SimpleDirichletPF(
                rng=self.rng,
                prior=self.prior,
                prior_strength=max(30.0, PRIOR_STRENGTH),
                n_particles=self.n_particles,
                resample_thr=self.resample_thr,
                kappa=PF_KAPPA,
                rejuvenate=PF_REJUV,
                dirichlet_eps=self.dirichlet_eps,
                stability_factor=self.stability_factor,
            )

    # -------- external updates --------
    def update_point_history(self, p_pts: int, b_pts: int):
        self.prev_p_pts = int(p_pts)
        self.prev_b_pts = int(b_pts)

    def update_outcome(self, outcome: int):
        if outcome not in (0, 1, 2):
            return

        if MODEL_MODE == "learn":
            self.counts *= PF_DECAY
            self.counts[outcome] += 1.0
            if self.pf is not None:
                self.pf.update(outcome)

        # 滾動視窗
        self.history_window.append(outcome)
        if len(self.history_window) > self.max_hist_len:
            self.history_window.pop(0)

        # update tie trackers
        if DYNAMIC_TIE_RANGE:
            self._update_tie_trackers()

    # -------- dynamic tie (smoothed) --------
    def _update_tie_trackers(self):
        window = self.history_window[-HIST_WIN:] if len(self.history_window) >= HIST_WIN else self.history_window[:]
        n = len(window)
        if n == 0:
            return
        tie_cnt = sum(1 for x in window if x == 2)
        beta_a = TIE_BETA_A + tie_cnt
        beta_b = TIE_BETA_B + (n - tie_cnt)
        mu_post = beta_a / max(EPS, (beta_a + beta_b))
        if self.t_ema is None:
            self.t_ema = mu_post
        else:
            a = np.clip(TIE_EMA_ALPHA, 0.0, 1.0)
            self.t_ema = a * mu_post + (1 - a) * self.t_ema
        self.tie_samples = n
        if self.tie_samples < TIE_MIN_SAMPLES:
            self.adaptive_tie_min = TIE_MIN
            self.adaptive_tie_max = TIE_MAX
            return
        center = float(self.t_ema)
        lo = max(TIE_MIN_FLOOR, TIE_MIN, center * (1 - TIE_DELTA))
        hi = min(TIE_MAX_CAP, max(TIE_MAX, center * (1 + TIE_DELTA)))
        hi = max(hi, lo + 1e-4)
        self.adaptive_tie_min = lo
        self.adaptive_tie_max = hi

    # -------- smoothing utilities (indep 用) --------
    def _dirichlet_jitter(self, probs: np.ndarray, strength: float = 120.0) -> np.ndarray:
        s = float(max(EPS, strength))
        alpha = np.clip(probs, EPS, 1.0) * s
        return self.rng.dirichlet(alpha)

    def _light_historical_update(self, probs: np.ndarray) -> np.ndarray:
        window = self.history_window[-HIST_WIN:]
        n = len(window)
        if n == 0:
            return probs
        counts = np.array([
            sum(1 for x in window if x == 0),
            sum(1 for x in window if x == 1),
            sum(1 for x in window if x == 2)
        ], dtype=np.float64)
        hist_probs = (counts + HIST_PSEUDO) / max(EPS, (n + 3.0 * HIST_PSEUDO))
        w_dyn = n / (n + 80.0)  # 緩升
        w = min(HISTORICAL_WEIGHT, HIST_WEIGHT_MAX, w_dyn)
        mixed = probs * (1 - w) + hist_probs * w
        mixed = np.clip(mixed, EPS, None)
        return mixed / mixed.sum()

    # -------- 粒子近似（fallback 用；保留與舊版兼容） --------
    def _particle_filter_predict(self) -> np.ndarray:
        window = self.history_window[-PF_WIN:]
        n = len(window)
        if n == 0:
            return self.prior.copy()
        counts = np.array([
            sum(1 for x in window if x == 0),
            sum(1 for x in window if x == 1),
            sum(1 for x in window if x == 2)
        ], dtype=np.float64)
        like = (counts + PF_ALPHA) / max(EPS, (n + 3.0 * PF_ALPHA))
        post = self.prior * like
        post = np.clip(post, EPS, None)
        return post / post.sum()

    def _posterior_mean(self) -> np.ndarray:
        post = self.prior * PRIOR_STRENGTH + self.counts
        post = np.clip(post, EPS, None)
        return post / post.sum()

    # -------- 點數差偏置（只動 B/P 子空間；T 不變） --------
    def _apply_point_bias(self, probs: np.ndarray) -> np.ndarray:
        if not POINT_BIAS_ON:
            return probs
        if self.prev_p_pts is None or self.prev_b_pts is None:
            return probs

        gap = int(self.prev_p_pts) - int(self.prev_b_pts)  # >0 偏向 P；<0 偏向 B
        if abs(gap) < POINT_BIAS_MIN_GAP:
            return probs

        # 平滑偏置（最大位移上限）
        bias = POINT_BIAS_MAX_SHIFT * np.tanh(POINT_BIAS_K * abs(gap))
        if gap < 0:
            bias = -bias

        # 小樣本/高和風險衰減
        n = len(self.history_window)
        damp = min(1.0, n / max(1.0, float(POINT_BIAS_DAMP_N)))
        if float(probs[2]) >= float(POINT_BIAS_TIE_DAMP_AT):
            damp *= float(np.clip(POINT_BIAS_TIE_DAMP, 0.0, 1.0))
        bias *= damp
        if abs(bias) < 1e-6:
            return probs

        t = float(probs[2])
        bp = max(EPS, 1.0 - t)
        p_rel = float(probs[1] / bp)
        p_rel = float(np.clip(p_rel + bias, 0.0, 1.0))
        b_rel = 1.0 - p_rel

        probs[0] = b_rel * bp
        probs[1] = p_rel * bp
        # probs[2] 保持 t
        probs = np.clip(probs, EPS, None)
        return probs / probs.sum()

    # -------- main predict --------
    def predict(self, sims_per_particle: int = 30) -> np.ndarray:
        # 1) 產生 base 機率
        if MODEL_MODE == "indep":
            probs = self._light_historical_update(self.prior.copy())
            strength = self._compute_jitter_strength()
            if strength is not None:
                probs = self._dirichlet_jitter(probs, strength)
        elif MODEL_MODE == "learn":
            if self.pf is not None:
                pf_probs = self.pf.predict()
            else:
                pf_probs = self._particle_filter_predict()
            bayes_probs = self._posterior_mean()
            n_pf = min(len(self.history_window), PF_WIN)
            w_pf = min(PF_WEIGHT_MAX, n_pf / (n_pf + PF_WEIGHT_K))
            probs = pf_probs * w_pf + bayes_probs * (1 - w_pf)
        else:
            probs = self.prior.copy()

        # 2) Tie 夾取（guard rails）
        if DYNAMIC_TIE_RANGE and self.tie_samples >= TIE_MIN_SAMPLES:
            tie_min, tie_max = self.adaptive_tie_min, self.adaptive_tie_max
        else:
            tie_min, tie_max = TIE_MIN, TIE_MAX

        probs = np.clip(probs, EPS, None); probs /= probs.sum()
        probs[2] = np.clip(probs[2], tie_min, tie_max)
        probs = np.clip(probs, EPS, None); probs /= probs.sum()

        # 3) 點數差偏置（只動 B/P；在夾 T 之後施加）
        probs = self._apply_point_bias(probs)

        return probs.astype(np.float32)

    # -------- helpers --------
    def _compute_jitter_strength(self) -> Optional[float]:
        if PROB_JITTER <= 0:
            return None
        base = max(50.0, min(PROB_JITTER_STRENGTH_MAX, 1.0 / max(1e-6, PROB_JITTER)))
        n = len(self.history_window)
        if n <= 0 or PROB_JITTER_SCALE <= 0:
            return float(base)
        growth = 1.0 + min(4.0, n / PROB_JITTER_SCALE)
        return float(np.clip(base * growth, 50.0, PROB_JITTER_STRENGTH_MAX))
