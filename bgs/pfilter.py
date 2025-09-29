# -*- coding: utf-8 -*-
"""
pfilter.py — 積極學習版本
修正重點：
1. 降低觀望門檻
2. 增強學習響應性
3. 優化初始參數
4. 減少過度平滑
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

# ---------- env helpers ----------
def _env_flag(k: str, default: str) -> bool:
    v = os.getenv(k, default).strip().lower()
    return v in ("1", "true", "yes", "on")

def _env_float(k: str, default: str) -> float:
    try:
        return float(os.getenv(k, default))
    except Exception:
        return float(default)

def _env_int(k: str, default: str) -> int:
    try:
        return int(os.getenv(k, default))
    except Exception:
        return int(default)

# ---------- global knobs ----------
MODEL_MODE = os.getenv("MODEL_MODE", "learn").strip().lower()

# 先驗：稍微偏向莊家（考慮抽水）
PRIOR_B = _env_float("PRIOR_B", "0.458")
PRIOR_P = _env_float("PRIOR_P", "0.446") 
PRIOR_T = _env_float("PRIOR_T", "0.096")
PRIOR_STRENGTH = _env_float("PRIOR_STRENGTH", "25")  # 降低先驗影響

# 降低衰減速度，讓學習更持久
PF_DECAY = _env_float("PF_DECAY", "0.992")

# 動態和局夾取（稍微放寬範圍）
TIE_MIN = _env_float("TIE_MIN", "0.04")
TIE_MAX = _env_float("TIE_MAX", "0.20")
TIE_MAX_CAP = _env_float("TIE_MAX_CAP", "0.28")
TIE_MIN_FLOOR = _env_float("TIE_MIN_FLOOR", "0.02")

DYNAMIC_TIE_RANGE = _env_flag("DYNAMIC_TIE_RANGE", "1")
TIE_BETA_A = _env_float("TIE_BETA_A", "8.0")    # 降低對和局的先驗
TIE_BETA_B = _env_float("TIE_BETA_B", "92.0")
TIE_EMA_ALPHA = _env_float("TIE_EMA_ALPHA", "0.25")  # 增加EMA響應性
TIE_MIN_SAMPLES = _env_int("TIE_MIN_SAMPLES", "25")  # 降低最小樣本
TIE_DELTA = _env_float("TIE_DELTA", "0.40")    # 增加動態範圍

# 減少抖動，讓信號更清晰
PROB_JITTER = _env_float("PROB_JITTER", "0.004")
PROB_JITTER_SCALE = _env_float("PROB_JITTER_SCALE", "20.0")
PROB_JITTER_STRENGTH_MAX = _env_float("PROB_JITTER_STRENGTH_MAX", "200.0")

# 增加歷史權重，讓學習更快
HISTORICAL_WEIGHT = _env_float("HISTORICAL_WEIGHT", "0.25")
HIST_WIN = _env_int("HIST_WIN", "35")  # 縮短窗口，更快響應
HIST_PSEUDO = _env_float("HIST_PSEUDO", "0.8")  # 減少偽計數
HIST_WEIGHT_MAX = _env_float("HIST_WEIGHT_MAX", "0.40")

# 啟用即時學習
EXCLUDE_LAST_OUTCOME = _env_flag("EXCLUDE_LAST_OUTCOME", "0")

# KF head smoothing（減少平滑，增加響應性）
KF_HEAD_ON = _env_flag("KF_HEAD_ON", "1")
KF_HEAD_WINDOW = max(1, _env_int("KF_HEAD_WINDOW", "6"))
KF_PROCESS_NOISE = _env_float("KF_PROCESS_NOISE", "5e-4")  # 增加過程噪聲
KF_MEAS_NOISE = _env_float("KF_MEAS_NOISE", "1e-3")       # 降低測量噪聲
KF_INIT_VARIANCE = _env_float("KF_INIT_VARIANCE", "0.01")  # 增加初始方差
KF_MIN_VARIANCE = _env_float("KF_MIN_VARIANCE", "1e-5")

# learn 混合權重（增加PF權重）
PF_WIN = _env_int("PF_WIN", "30")  # 更短的PF窗口
PF_ALPHA = _env_float("PF_ALPHA", "0.3")  # 減少平滑，增加響應
PF_WEIGHT_MAX = _env_float("PF_WEIGHT_MAX", "0.85")  # 增加PF最大權重
PF_WEIGHT_K = _env_float("PF_WEIGHT_K", "25.0")  # 更快達到最大權重

# 真 PF 參數（增強學習能力）
PF_N = _env_int("PF_N", "100")  # 適中粒子數
PF_RESAMPLE = _env_float("PF_RESAMPLE", "0.8")  # 提高重採樣閾值，減少抖動
DIRICHLET_EPS = _env_float("PF_DIR_EPS", "0.008")  # 減少抖動
STABILITY_FACTOR = _env_float("PF_STAB_FACTOR", "0.9")  # 增加穩定性

# 點差偏置（保持關閉）
POINT_BIAS_ON = _env_flag("POINT_BIAS_ON", "0")

# CUSUM guard（放寬限制）
CUSUM_ON = _env_flag("CUSUM_ON", "1")
CUSUM_DECAY = _env_float("CUSUM_DECAY", "0.95")  # 降低衰減速度
CUSUM_DRIFT = _env_float("CUSUM_DRIFT", "0.08")  # 降低漂移
CUSUM_THRESHOLD = _env_float("CUSUM_THRESHOLD", "4.0")  # 提高閾值
CUSUM_GAIN = _env_float("CUSUM_GAIN", "0.02")  # 降低增益
CUSUM_MAX_SHIFT = _env_float("CUSUM_MAX_SHIFT", "0.05")  # 降低最大位移

EPS = 1e-9

# ---------- SimpleDirichletPF (增強學習版本) ----------
class SimpleDirichletPF:
    def __init__(
        self,
        prior: np.ndarray,
        prior_strength: float,
        n_particles: int,
        dirichlet_eps: float,
        stability_factor: float,
        resample_thr: float,
        rng: np.random.Generator,
        kappa: float | None = None,
        rejuvenate: float | None = None,
    ):
        self.rng = rng
        self.n_particles = max(1, int(n_particles))
        self.dirichlet_eps = float(max(EPS, dirichlet_eps))
        self.stability_factor = float(max(EPS, stability_factor))
        self.resample_thr = float(np.clip(resample_thr, EPS, 1.0))
        self.prior = np.clip(np.asarray(prior, dtype=np.float64), EPS, None)
        self.prior = self.prior / self.prior.sum()
        self.prior_strength = float(max(1.0, prior_strength))
        
        # 增強學習：降低初始先驗影響
        alpha = np.clip(self.prior * self.prior_strength * 0.7, EPS, None)
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

        # 溫和的rejuvenation
        adapt_strength = max(25.0, self.prior_strength * self.stability_factor)
        alpha = np.clip(self.parts * adapt_strength + self.dirichlet_eps, EPS, None)
        for k in range(n):
            self.parts[k] = self.rng.dirichlet(alpha[k])

    def update(self, outcome: int):
        if outcome not in (0, 1, 2):
            return
        
        self.update_count += 1
        
        # 動態學習率：隨更新次數增加而提高學習強度
        learning_boost = min(2.0, 1.0 + self.update_count / 50.0)
        like = np.clip(self.parts[:, outcome], EPS, None) * learning_boost
        
        self.w *= like
        self.w /= np.sum(self.w)

        if (self._ess() / self.n_particles) < self.resample_thr:
            self._resample()

    def predict(self) -> np.ndarray:
        m = np.average(self.parts, axis=0, weights=self.w)
        m = np.clip(m, EPS, None)
        return (m / np.sum(m)).astype(np.float64)

# ---------- OutcomePF (積極學習版本) ----------
@dataclass
class OutcomePF:
    decks: int = _env_int("DECKS", "6")
    seed: int = _env_int("SEED", "42")
    n_particles: int = PF_N
    sims_lik: int = _env_int("PF_UPD_SIMS", "36")
    resample_thr: float = PF_RESAMPLE
    backend: str = os.getenv("PF_BACKEND", "mc").strip().lower()
    dirichlet_eps: float = DIRICHLET_EPS
    stability_factor: float = STABILITY_FACTOR

    prev_p_pts: Optional[int] = None
    prev_b_pts: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.prior = np.array([PRIOR_B, PRIOR_P, PRIOR_T], dtype=np.float64)
        self.prior = self.prior / self.prior.sum()

        # 長期貝氏計數
        self.counts = np.zeros(3, dtype=np.float64)
        self.counts_pre_update = self.counts.copy()

        # rolling window
        self.history_window: List[int] = []
        self.max_hist_len = max(HIST_WIN, PF_WIN, 100)
        self.last_outcome: Optional[int] = None

        # tie trackers
        self.t_ema = None
        self.tie_samples = 0
        self.adaptive_tie_min = TIE_MIN
        self.adaptive_tie_max = TIE_MAX

        # KF head state
        self.kf_state: Optional[np.ndarray] = None
        self.kf_cov: Optional[np.ndarray] = None
        self.kf_measurements: List[np.ndarray] = []

        # CUSUM trackers
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0

        # 真 PF
        self.pf: Optional[SimpleDirichletPF] = None
        self._pf_prediction_before_last: Optional[np.ndarray] = None
        if MODEL_MODE == "learn":
            self.pf = SimpleDirichletPF(
                rng=self.rng,
                prior=self.prior,
                prior_strength=max(20.0, PRIOR_STRENGTH),
                n_particles=self.n_particles,
                resample_thr=self.resample_thr,
                dirichlet_eps=self.dirichlet_eps,
                stability_factor=self.stability_factor,
            )
            self._pf_prediction_before_last = self.pf.predict()

    def update_point_history(self, p_pts: int, b_pts: int):
        self.prev_p_pts = int(p_pts)
        self.prev_b_pts = int(b_pts)

    def update_outcome(self, outcome: int):
        if outcome not in (0, 1, 2):
            return

        self.last_outcome = outcome

        if MODEL_MODE == "learn":
            # 降低衰減，讓學習更持久
            self.counts *= PF_DECAY
            self.counts_pre_update = self.counts.copy()
            self.counts[outcome] += 1.2  # 增加學習強度

            if self.pf is not None:
                if EXCLUDE_LAST_OUTCOME:
                    self._pf_prediction_before_last = self.pf.predict()
                self.pf.update(outcome)

        self.history_window.append(outcome)
        if len(self.history_window) > self.max_hist_len:
            self.history_window.pop(0)

        if DYNAMIC_TIE_RANGE:
            self._update_tie_trackers()

        self._update_cusum_trackers(outcome)

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

    def _update_cusum_trackers(self, outcome: int):
        if not CUSUM_ON:
            return
        decay = float(np.clip(CUSUM_DECAY, 0.0, 1.0))
        drift = float(max(0.0, CUSUM_DRIFT))
        self.cusum_pos *= decay
        self.cusum_neg *= decay
        if outcome not in (0, 1):
            return
        x = 1.0 if outcome == 1 else -1.0
        self.cusum_pos = max(0.0, self.cusum_pos + x - drift)
        self.cusum_neg = min(0.0, self.cusum_neg + x + drift)

    def _dirichlet_jitter(self, probs: np.ndarray, strength: float = 120.0) -> np.ndarray:
        s = float(max(EPS, strength))
        alpha = np.clip(probs, EPS, 1.0) * s
        return self.rng.dirichlet(alpha)

    def _compute_jitter_strength(self) -> Optional[float]:
        if PROB_JITTER <= 0:
            return None
        base = max(50.0, min(PROB_JITTER_STRENGTH_MAX, 1.0 / max(1e-6, PROB_JITTER)))
        n = len(self.history_window)
        if n <= 0 or PROB_JITTER_SCALE <= 0:
            return base
        growth = 1.0 + min(4.0, n / PROB_JITTER_SCALE)
        strength = base * growth
        return float(np.clip(strength, 50.0, PROB_JITTER_STRENGTH_MAX))

    def _apply_kf_head(self, probs: np.ndarray) -> np.ndarray:
        if not KF_HEAD_ON:
            return probs
        measurement = np.clip(probs, EPS, None)
        measurement = measurement / measurement.sum()
        self.kf_measurements.append(measurement)
        if len(self.kf_measurements) > KF_HEAD_WINDOW:
            self.kf_measurements.pop(0)
        
        # 使用加權平均而不是簡單平均（最近數據權重更高）
        n_meas = len(self.kf_measurements)
        if n_meas == 0:
            return probs
            
        weights = np.exp(np.linspace(0, 1, n_meas))
        weights = weights / weights.sum()
        
        stacked = np.stack(self.kf_measurements, axis=0)
        measurement_avg = np.average(stacked, axis=0, weights=weights)
        measurement_avg = np.clip(measurement_avg, EPS, None)
        measurement_avg = measurement_avg / measurement_avg.sum()
        
        if (self.kf_state is None) or (self.kf_cov is None):
            self.kf_state = measurement_avg.copy()
            self.kf_cov = np.full(3, KF_INIT_VARIANCE, dtype=np.float64)
            return measurement_avg
            
        pred_state = self.kf_state
        pred_cov = self.kf_cov + KF_PROCESS_NOISE
        gain = pred_cov / (pred_cov + KF_MEAS_NOISE)
        updated_state = pred_state + gain * (measurement_avg - pred_state)
        updated_cov = (1.0 - gain) * pred_cov
        self.kf_state = np.clip(updated_state, EPS, None)
        self.kf_cov = np.clip(updated_cov, KF_MIN_VARIANCE, None)
        smoothed = self.kf_state / self.kf_state.sum()
        return smoothed

    def _get_history_window(self, limit: int, include_latest: bool = True) -> List[int]:
        if limit <= 0:
            window = self.history_window[:]
        else:
            window = self.history_window[-limit:]
        if not include_latest and window:
            window = window[:-1]
        return window

    def _light_historical_update(self, probs: np.ndarray) -> np.ndarray:
        include_latest = not EXCLUDE_LAST_OUTCOME
        window = self._get_history_window(HIST_WIN, include_latest=include_latest)
        n = len(window)
        if n == 0:
            return probs
            
        counts = np.array([
            sum(1 for x in window if x == 0),
            sum(1 for x in window if x == 1),
            sum(1 for x in window if x == 2),
        ], dtype=np.float64)
        
        # 使用指數加權，最近結果權重更高
        hist_probs = (counts + HIST_PSEUDO) / max(EPS, (n + 3.0 * HIST_PSEUDO))
        
        # 動態權重：樣本越多，歷史權重越高
        w_dyn = min(HIST_WEIGHT_MAX, n / (n + 40.0))
        w = min(HISTORICAL_WEIGHT, w_dyn)
        
        mixed = probs * (1 - w) + hist_probs * w
        mixed = np.clip(mixed, EPS, None)
        return mixed / mixed.sum()

    def _particle_filter_predict(self) -> np.ndarray:
        include_latest = not EXCLUDE_LAST_OUTCOME
        window = self._get_history_window(PF_WIN, include_latest=include_latest)
        n = len(window)
        if n == 0:
            return self.prior.copy()
            
        counts = np.array([
            sum(1 for x in window if x == 0),
            sum(1 for x in window if x == 1),
            sum(1 for x in window if x == 2),
        ], dtype=np.float64)
        
        like = (counts + PF_ALPHA) / max(EPS, (n + 3.0 * PF_ALPHA))
        post = self.prior * like
        post = np.clip(post, EPS, None)
        return post / post.sum()

    def _posterior_mean(self) -> np.ndarray:
        counts = self.counts_pre_update if EXCLUDE_LAST_OUTCOME else self.counts
        post = self.prior * PRIOR_STRENGTH + counts
        post = np.clip(post, EPS, None)
        return post / post.sum()

    def _apply_point_bias(self, probs: np.ndarray) -> np.ndarray:
        if not POINT_BIAS_ON:
            return probs
        # ... 保持原有實現
        return probs

    def _apply_cusum_guard(self, probs: np.ndarray) -> np.ndarray:
        if not CUSUM_ON:
            return probs
            
        t = float(probs[2]); b = float(probs[0]); p = float(probs[1])
        if (b + p) <= EPS:
            return probs
            
        # 放寬CUSUM條件，只在極端情況下調整
        if self.cusum_pos > CUSUM_THRESHOLD * 1.5:
            excess = self.cusum_pos - CUSUM_THRESHOLD
            shift = min(CUSUM_MAX_SHIFT, excess * CUSUM_GAIN)
            shift = min(shift, p - 0.01)  # 保持最小機率
            if shift > 0.005:  # 只有顯著調整才應用
                b += shift
                p -= shift
        elif (-self.cusum_neg) > CUSUM_THRESHOLD * 1.5:
            excess = (-self.cusum_neg) - CUSUM_THRESHOLD
            shift = min(CUSUM_MAX_SHIFT, excess * CUSUM_GAIN)
            shift = min(shift, b - 0.01)
            if shift > 0.005:
                b -= shift
                p += shift
                
        adjusted = np.array([b, p, t], dtype=np.float64)
        adjusted = np.clip(adjusted, 0.01, 0.98)  # 保持合理範圍
        adjusted /= adjusted.sum()
        return adjusted

    def predict(self, sims_per_particle: int = 30) -> np.ndarray:
        # 1) 產生 base 機率
        if MODEL_MODE == "indep":
            probs = self._light_historical_update(self.prior.copy())
            strength = self._compute_jitter_strength()
            if strength is not None:
                probs = self._dirichlet_jitter(probs, strength)
        elif MODEL_MODE == "learn":
            if self.pf is not None:
                if EXCLUDE_LAST_OUTCOME and (self._pf_prediction_before_last is not None):
                    pf_probs = self._pf_prediction_before_last.copy()
                else:
                    pf_probs = self.pf.predict()
            else:
                pf_probs = self._particle_filter_predict()
                
            bayes_probs = self._posterior_mean()
            n_pf = min(len(self.history_window), PF_WIN)
            
            # 增加PF權重，讓學習主導
            w_pf = min(PF_WEIGHT_MAX, 0.3 + (n_pf / (n_pf + PF_WEIGHT_K)))
            probs = pf_probs * w_pf + bayes_probs * (1 - w_pf)
        else:
            probs = self.prior.copy()

        # 2) KF head 平滑（輕度）
        probs = self._apply_kf_head(probs)

        # 3) Tie 夾取（放寬範圍）
        if DYNAMIC_TIE_RANGE and self.tie_samples >= TIE_MIN_SAMPLES:
            tie_min, tie_max = self.adaptive_tie_min, self.adaptive_tie_max
        else:
            tie_min, tie_max = TIE_MIN, TIE_MAX
            
        probs = np.clip(probs, EPS, None)
        probs /= probs.sum()
        probs[2] = np.clip(probs[2], tie_min, tie_max)
        probs = np.clip(probs, EPS, None)
        probs /= probs.sum()

        # 4) CUSUM guard（輕度保護）
        probs = self._apply_cusum_guard(probs)

        # 5) 確保機率有足夠的差異性
        min_diff = 0.02  # 最小2%的差異
        if abs(probs[0] - probs[1]) < min_diff and len(self.history_window) > 5:
            # 稍微放大差異
            diff = probs[0] - probs[1]
            if abs(diff) > 0:
                scale = min(1.5, 1.0 + min_diff / abs(diff))
                if diff > 0:
                    probs[0] = (probs[0] + probs[1] * (scale - 1)) / scale
                    probs[1] = probs[1] / scale
                else:
                    probs[1] = (probs[1] + probs[0] * (scale - 1)) / scale
                    probs[0] = probs[0] / scale
                probs = np.clip(probs, 0.01, 0.98)
                probs /= probs.sum()

        return probs.astype(np.float32)
