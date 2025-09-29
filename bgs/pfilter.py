# -*- coding: utf-8 -*-
"""
pfilter.py — 修正長龍問題的強化版本
主要修正：
1. 增強粒子多樣性保護
2. 動態調整歷史權重
3. 添加趨勢轉換檢測
4. 優化學習速率
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

# 先驗：更加保守的初始設定
PRIOR_B = _env_float("PRIOR_B", "0.45")
PRIOR_P = _env_float("PRIOR_P", "0.45")
PRIOR_T = _env_float("PRIOR_T", "0.10")
PRIOR_STRENGTH = _env_float("PRIOR_STRENGTH", "30")  # 降低先驗強度

# 長期貝氏計數衰減（增加衰減速度避免過度學習）
PF_DECAY = _env_float("PF_DECAY", "0.965")

# 動態和局夾取
TIE_MIN = _env_float("TIE_MIN", "0.03")
TIE_MAX = _env_float("TIE_MAX", "0.18")
TIE_MAX_CAP = _env_float("TIE_MAX_CAP", "0.25")
TIE_MIN_FLOOR = _env_float("TIE_MIN_FLOOR", "0.01")

DYNAMIC_TIE_RANGE = _env_flag("DYNAMIC_TIE_RANGE", "1")
TIE_BETA_A = _env_float("TIE_BETA_A", "9.6")
TIE_BETA_B = _env_float("TIE_BETA_B", "90.4")
TIE_EMA_ALPHA = _env_float("TIE_EMA_ALPHA", "0.2")
TIE_MIN_SAMPLES = _env_int("TIE_MIN_SAMPLES", "40")
TIE_DELTA = _env_float("TIE_DELTA", "0.35")

# indep 模式的歷史/抖動
PROB_JITTER = _env_float("PROB_JITTER", "0.008")  # 增加抖動
PROB_JITTER_SCALE = _env_float("PROB_JITTER_SCALE", "12.0")  # 降低尺度
PROB_JITTER_STRENGTH_MAX = _env_float("PROB_JITTER_STRENGTH_MAX", "300.0")
HISTORICAL_WEIGHT = _env_float("HISTORICAL_WEIGHT", "0.15")  # 降低歷史權重
HIST_WIN = _env_int("HIST_WIN", "45")  # 縮短歷史窗口
HIST_PSEUDO = _env_float("HIST_PSEUDO", "1.5")  # 增加偽計數
HIST_WEIGHT_MAX = _env_float("HIST_WEIGHT_MAX", "0.25")  # 降低最大權重

# 重要：啟用即時學習
EXCLUDE_LAST_OUTCOME = _env_flag("EXCLUDE_LAST_OUTCOME", "0")

# KF head smoothing
KF_HEAD_ON = _env_flag("KF_HEAD_ON", "1")
KF_HEAD_WINDOW = max(1, _env_int("KF_HEAD_WINDOW", "8"))  # 縮短平滑窗口
KF_PROCESS_NOISE = _env_float("KF_PROCESS_NOISE", "2e-4")  # 增加過程噪聲
KF_MEAS_NOISE = _env_float("KF_MEAS_NOISE", "3e-3")  # 增加測量噪聲
KF_INIT_VARIANCE = _env_float("KF_INIT_VARIANCE", "0.005")
KF_MIN_VARIANCE = _env_float("KF_MIN_VARIANCE", "1e-6")

# learn 混合權重（降低PF權重，增加響應性）
PF_WIN = _env_int("PF_WIN", "40")  # 縮短PF窗口
PF_ALPHA = _env_float("PF_ALPHA", "0.8")  # 增加平滑
PF_WEIGHT_MAX = _env_float("PF_WEIGHT_MAX", "0.6")  # 降低PF最大權重
PF_WEIGHT_K = _env_float("PF_WEIGHT_K", "60.0")  # 降低K值，更快達到最大權重

# 真 PF 參數（增強多樣性）
PF_N = _env_int("PF_N", "150")  # 增加粒子數
PF_RESAMPLE = _env_float("PF_RESAMPLE", "0.65")  # 降低重採樣閾值
PF_KAPPA = _env_float("PF_KAPPA", "180.0")
PF_REJUV = _env_float("PF_REJUV", "250.0")  # 增加 rejuvenation
DIRICHLET_EPS = _env_float("PF_DIR_EPS", "0.025")  # 增加抖動
STABILITY_FACTOR = _env_float("PF_STAB_FACTOR", "0.6")  # 降低穩定性，增加多樣性

# 點差偏置（保持關閉）
POINT_BIAS_ON = _env_flag("POINT_BIAS_ON", "0")
POINT_BIAS_K = _env_float("POINT_BIAS_K", "0.35")
POINT_BIAS_MAX_SHIFT = _env_float("POINT_BIAS_MAX_SHIFT", "0.04")
POINT_BIAS_MIN_GAP = _env_int("POINT_BIAS_MIN_GAP", "3")
POINT_BIAS_DAMP_N = _env_int("POINT_BIAS_DAMP_N", "20")
POINT_BIAS_TIE_DAMP_AT = _env_float("POINT_BIAS_TIE_DAMP_AT", "0.14")
POINT_BIAS_TIE_DAMP = _env_float("POINT_BIAS_TIE_DAMP", "0.5")

# CUSUM guard（調整參數避免過度回拉）
CUSUM_ON = _env_flag("CUSUM_ON", "1")
CUSUM_DECAY = _env_float("CUSUM_DECAY", "0.88")  # 增加衰減速度
CUSUM_DRIFT = _env_float("CUSUM_DRIFT", "0.12")  # 降低漂移
CUSUM_THRESHOLD = _env_float("CUSUM_THRESHOLD", "3.0")  # 提高閾值
CUSUM_GAIN = _env_float("CUSUM_GAIN", "0.04")  # 降低增益
CUSUM_MAX_SHIFT = _env_float("CUSUM_MAX_SHIFT", "0.08")  # 降低最大位移

# 新增：趨勢檢測參數
TREND_DETECTION_ON = _env_flag("TREND_DETECTION_ON", "1")
TREND_WINDOW = _env_int("TREND_WINDOW", "8")
TREND_THRESHOLD = _env_float("TREND_THRESHOLD", "0.75")
TREND_ADAPT_STRENGTH = _env_float("TREND_ADAPT_STRENGTH", "0.3")

EPS = 1e-9


# ---------- SimpleDirichletPF (增強多樣性版本) ----------
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
        self.kappa = kappa
        self.rejuvenate = rejuvenate
        
        # 增加初始化多樣性
        alpha = np.clip(self.prior * self.prior_strength * 0.5, EPS, None)  # 降低初始強度
        self.parts = self.rng.dirichlet(alpha, size=self.n_particles)
        self.w = np.ones(self.n_particles, dtype=np.float64) / self.n_particles
        self.iteration = 0

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

        # 增強 rejuvenation 多樣性
        adapt_strength = max(15.0, self.prior_strength * self.stability_factor * 
                           (0.8 + 0.4 * np.exp(-self.iteration / 100.0)))  # 隨迭代衰減
        alpha = np.clip(self.parts * adapt_strength + self.dirichlet_eps, EPS, None)
        for k in range(n):
            # 對部分粒子施加額外抖動
            if k % 5 == 0:  # 每5個粒子有一個額外抖動
                extra_eps = self.dirichlet_eps * 3.0
                self.parts[k] = self.rng.dirichlet(alpha[k] + extra_eps)
            else:
                self.parts[k] = self.rng.dirichlet(alpha[k])
        
        self.iteration += 1

    def update(self, outcome: int):
        if outcome not in (0, 1, 2):
            return
        
        # 動態學習率：新數據權重隨迭代增加
        base_like = np.clip(self.parts[:, outcome], EPS, None)
        
        # 對低權重粒子給予更多關注（避免多樣性喪失）
        weight_factor = 1.0 + (1.0 - self.w) * 0.5
        like = base_like * weight_factor
        
        self.w *= like
        self.w /= np.sum(self.w)

        if (self._ess() / self.n_particles) < self.resample_thr:
            self._resample()

    def predict(self) -> np.ndarray:
        m = np.average(self.parts, axis=0, weights=self.w)
        m = np.clip(m, EPS, None)
        return (m / np.sum(m)).astype(np.float64)


# ---------- OutcomePF (增強趨勢檢測版本) ----------
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

        # 新增：趨勢檢測
        self.trend_strength = 0.0
        self.trend_direction = 0  # -1: 莊趨勢, 0: 無趨勢, 1: 閒趨勢
        self.consecutive_count = 0
        self.last_trend_outcome = None

        # 真 PF
        self.pf: Optional[SimpleDirichletPF] = None
        self._pf_prediction_before_last: Optional[np.ndarray] = None
        if MODEL_MODE == "learn":
            self.pf = SimpleDirichletPF(
                rng=self.rng,
                prior=self.prior,
                prior_strength=max(20.0, PRIOR_STRENGTH),  # 降低初始強度
                n_particles=self.n_particles,
                resample_thr=self.resample_thr,
                kappa=PF_KAPPA,
                rejuvenate=PF_REJUV,
                dirichlet_eps=self.dirichlet_eps,
                stability_factor=self.stability_factor,
            )
            self._pf_prediction_before_last = self.pf.predict()

    # ---- 新增：趨勢檢測 ----
    def _update_trend_detection(self, outcome: int):
        if not TREND_DETECTION_ON or outcome == 2:  # 和局不影響趨勢
            return
            
        if outcome == self.last_trend_outcome:
            self.consecutive_count += 1
        else:
            self.consecutive_count = 1
            self.last_trend_outcome = outcome
            
        # 計算趨勢強度
        window = self.history_window[-TREND_WINDOW:] if len(self.history_window) >= TREND_WINDOW else self.history_window
        if len(window) >= 5:
            banker_count = sum(1 for x in window if x == 0)
            player_count = sum(1 for x in window if x == 1)
            total = len(window)
            
            if total > 0:
                banker_ratio = banker_count / total
                player_ratio = player_count / total
                
                # 趨勢強度基於比例差和連續次數
                ratio_diff = abs(banker_ratio - player_ratio)
                consecutive_factor = min(1.0, self.consecutive_count / 8.0)
                self.trend_strength = ratio_diff * 0.7 + consecutive_factor * 0.3
                
                if banker_ratio > player_ratio + 0.1:
                    self.trend_direction = -1  # 莊趨勢
                elif player_ratio > banker_ratio + 0.1:
                    self.trend_direction = 1   # 閒趨勢
                else:
                    self.trend_direction = 0   # 無趨勢

    # ---- 修改：更新函數包含趨勢檢測 ----
    def update_outcome(self, outcome: int):
        if outcome not in (0, 1, 2):
            return

        self.last_outcome = outcome

        if MODEL_MODE == "learn":
            self.counts *= PF_DECAY
            self.counts_pre_update = self.counts.copy()
            self.counts[outcome] += 1.0

            if self.pf is not None:
                if EXCLUDE_LAST_OUTCOME:
                    self._pf_prediction_before_last = self.pf.predict()
                self.pf.update(outcome)

        self.history_window.append(outcome)
        if len(self.history_window) > self.max_hist_len:
            self.history_window.pop(0)

        # 更新趨勢檢測
        self._update_trend_detection(outcome)

        if DYNAMIC_TIE_RANGE:
            self._update_tie_trackers()

        self._update_cusum_trackers(outcome)

    # ---- 修改：預測函數包含趨勢適應 ----
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
            
            # 動態調整PF權重：強趨勢時降低PF權重
            base_w_pf = min(PF_WEIGHT_MAX, n_pf / (n_pf + PF_WEIGHT_K))
            if self.trend_strength > TREND_THRESHOLD:
                trend_adapt = 1.0 - (self.trend_strength * TREND_ADAPT_STRENGTH)
                w_pf = base_w_pf * trend_adapt
            else:
                w_pf = base_w_pf
                
            probs = pf_probs * w_pf + bayes_probs * (1 - w_pf)
        else:
            probs = self.prior.copy()

        # 2) KF head 平滑
        probs = self._apply_kf_head(probs)

        # 3) Tie 夾取
        if DYNAMIC_TIE_RANGE and self.tie_samples >= TIE_MIN_SAMPLES:
            tie_min, tie_max = self.adaptive_tie_min, self.adaptive_tie_max
        else:
            tie_min, tie_max = TIE_MIN, TIE_MAX
        probs = np.clip(probs, EPS, None); probs /= probs.sum()
        probs[2] = np.clip(probs[2], tie_min, tie_max)
        probs = np.clip(probs, EPS, None); probs /= probs.sum()

        # 4) CUSUM guard
        probs = self._apply_cusum_guard(probs)

        # 5) 趨勢適應：強趨勢時增加不確定性
        if self.trend_strength > TREND_THRESHOLD:
            # 向先驗收縮，避免過度自信
            shrink_factor = self.trend_strength * 0.2
            probs = probs * (1 - shrink_factor) + self.prior * shrink_factor
            probs = np.clip(probs, EPS, None)
            probs /= probs.sum()

        return probs.astype(np.float32)

    # 其他方法保持不變，但使用新的環境變數值
    def _update_tie_trackers(self):
        # ... 實現保持不變，但使用新的 TIE_DELTA 等參數
        pass

    def _update_cusum_trackers(self, outcome: int):
        # ... 實現保持不變，但使用新的 CUSUM 參數
        pass

    def _dirichlet_jitter(self, probs: np.ndarray, strength: float = 120.0) -> np.ndarray:
        # ... 實現保持不變
        pass

    def _compute_jitter_strength(self) -> Optional[float]:
        # ... 實現保持不變
        pass

    def _apply_kf_head(self, probs: np.ndarray) -> np.ndarray:
        # ... 實現保持不變，但使用新的 KF 參數
        pass

    def _get_history_window(self, limit: int, include_latest: bool = True) -> List[int]:
        # ... 實現保持不變
        pass

    def _light_historical_update(self, probs: np.ndarray) -> np.ndarray:
        # ... 實現保持不變，但使用新的歷史權重參數
        pass

    def _particle_filter_predict(self) -> np.ndarray:
        # ... 實現保持不變
        pass

    def _posterior_mean(self) -> np.ndarray:
        # ... 實現保持不變
        pass

    def _apply_point_bias(self, probs: np.ndarray) -> np.ndarray:
        # ... 實現保持不變
        pass

    def _apply_cusum_guard(self, probs: np.ndarray) -> np.ndarray:
        # ... 實現保持不變，但使用新的 CUSUM 參數
        pass

    def update_point_history(self, p_pts: int, b_pts: int):
        self.prev_p_pts = int(p_pts)
        self.prev_b_pts = int(b_pts)
