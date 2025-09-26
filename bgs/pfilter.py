# -*- coding: utf-8 -*-
"""
pfilter.py — 粒子濾波/貝氏簡化器（強化版）
- 嚴格環境變數解析（錯誤會警示，不再靜默回退）
- MODEL_MODE 以「每實例」決定（constructor/env），不再 import 時凍結
- predict() 會實際使用 sims_per_particle 作為平滑/抖動強度的尺度
- 動態和局範圍：Beta 先驗 + EMA + 樣本門檻（避免過度收縮/放大）
- 歷史/粒子視窗：加入 Dirichlet/Laplace pseudo-counts，避免 zeroing
- 輕量連莊偏壓（streak bias）可開關與調強度
"""

import os
import warnings
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

EPS = 1e-9

# ===== 嚴格環境變數讀取 =====
def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1","true","yes","on")

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, None)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (ValueError, TypeError):
        warnings.warn(f"[pfilter] ENV {name}='{raw}' 無法解析為 float，使用預設 {default}")
        return float(default)

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, None)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (ValueError, TypeError):
        warnings.warn(f"[pfilter] ENV {name}='{raw}' 無法解析為 int，使用預設 {default}")
        return int(default)

# ===== 先驗與全域預設（僅作為 default；真正使用在 __post_init__ 綁死到實例） =====
DEF_PRIOR_B = _env_float("PRIOR_B", 0.452)
DEF_PRIOR_P = _env_float("PRIOR_P", 0.452)
DEF_PRIOR_T = _env_float("PRIOR_T", 0.096)
DEF_PRIOR_STRENGTH = _env_float("PRIOR_STRENGTH", 40.0)

DEF_PF_DECAY = _env_float("PF_DECAY", 0.985)

# 動態和局估計（Beta + EMA + 範圍）
DEF_TIE_MIN = _env_float("TIE_MIN", 0.03)
DEF_TIE_MAX = _env_float("TIE_MAX", 0.18)
DEF_DYNAMIC_TIE_RANGE = _env_flag("DYNAMIC_TIE_RANGE", "1")
DEF_TIE_BETA_A = _env_float("TIE_BETA_A", 9.6)     # 約 ~9.6%
DEF_TIE_BETA_B = _env_float("TIE_BETA_B", 90.4)
DEF_TIE_EMA_ALPHA = _env_float("TIE_EMA_ALPHA", 0.20)
DEF_TIE_MIN_SAMPLES = _env_int("TIE_MIN_SAMPLES", 40)
DEF_TIE_DELTA = _env_float("TIE_DELTA", 0.35)      # 上下各 ±35%
DEF_TIE_MAX_CAP = _env_float("TIE_MAX_CAP", 0.25)
DEF_TIE_MIN_FLOOR = _env_float("TIE_MIN_FLOOR", 0.01)

# 抖動與模式
DEF_PROB_JITTER = _env_float("PROB_JITTER", 0.006)

# 歷史/粒子視窗與 pseudo-counts
DEF_HIST_WIN = _env_int("HIST_WIN", 60)
DEF_HIST_PSEUDO = _env_float("HIST_PSEUDO", 1.0)
DEF_HIST_WEIGHT_MAX = _env_float("HIST_WEIGHT_MAX", 0.35)

DEF_PF_WIN = _env_int("PF_WIN", 50)
DEF_PF_ALPHA = _env_float("PF_ALPHA", 0.5)           # Dirichlet alpha（對應 pseudo-count 強度）
DEF_PF_WEIGHT_MAX = _env_float("PF_WEIGHT_MAX", 0.7)
DEF_PF_WEIGHT_K = _env_float("PF_WEIGHT_K", 80)      # 視窗→權重映射尺度

# 連莊偏壓（在 simplex 內柔性偏壓，不改 transition）
DEF_STREAK_ENABLE = _env_flag("PF_STREAK_BIAS_ENABLE", "1")
DEF_STREAK_MIN = _env_int("PF_STREAK_BIAS_MIN", 3)     # 至少 N 連非和
DEF_STREAK_GAIN = _env_float("PF_STREAK_BIAS_GAIN", 0.08)  # 每超 1 手的增益（log 型）
DEF_STREAK_MAXW = _env_float("PF_STREAK_BIAS_MAX", 0.18)   # 偏壓上限（混合權重）
DEF_STREAK_RELIEF = _env_float("PF_STREAK_RELIEF", 0.55)   # 保留對側比例，避免坍塌

# indep 模式輕量歷史學習（EMA 可放在 server；此處只用視窗法）
DEF_MIN_HISTORY_FOR_LEARNING = _env_int("MIN_HISTORY", 10)
DEF_HISTORICAL_WEIGHT = _env_float("HISTORICAL_WEIGHT", 0.2)


def _safe_normalize(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, None)
    s = float(np.sum(p))
    if s <= 0:
        return np.array([1/3, 1/3, 1/3], dtype=np.float64)
    return (p / s).astype(np.float64)


@dataclass
class OutcomePF:
    # 與 server 相容的參數（大多只作為接口保留）
    decks: int = _env_int("DECKS", 6)
    seed: int = _env_int("SEED", 42)
    n_particles: int = _env_int("PF_N", 80)
    sims_lik: int = _env_int("PF_UPD_SIMS", 36)
    resample_thr: float = _env_float("PF_RESAMPLE", 0.73)
    backend: str = os.getenv("PF_BACKEND", "mc").strip().lower()
    dirichlet_eps: float = _env_float("PF_DIR_EPS", 0.012)

    # 新：模式與先驗可 per-instance 設定
    model_mode: Optional[str] = None   # "indep" / "learn"；None 表示用 ENV
    prior_b: float = DEF_PRIOR_B
    prior_p: float = DEF_PRIOR_P
    prior_t: float = DEF_PRIOR_T
    prior_strength: float = DEF_PRIOR_STRENGTH

    # 內部狀態
    prev_p_pts: Optional[int] = None
    prev_b_pts: Optional[int] = None

    # 動態 tie 估計狀態
    _tie_mu_ema: Optional[float] = None
    _tie_seen: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

        # 每實例的 prior（避免外部動態變更時相互污染）
        self.prior = np.array([self.prior_b, self.prior_p, self.prior_t], dtype=np.float64)
        self.prior = _safe_normalize(self.prior)

        # 累積計數（learn 模式的長期計數，帶衰減）
        self.counts = np.zeros(3, dtype=np.float64)

        # 歷史窗口（最近 N 手 outcome；0=莊,1=閒,2=和）
        self.history_window: List[int] = []

        # streak 狀態（只記非和）
        self._streak_len = 0
        self._streak_side = None  # 0=莊, 1=閒

        # 每實例的 env 開關快照
        self.dynamic_tie = DEF_DYNAMIC_TIE_RANGE

        # 權重/視窗
        self.hist_win = DEF_HIST_WIN
        self.hist_pseudo = DEF_HIST_PSEUDO
        self.hist_weight_max = DEF_HIST_WEIGHT_MAX

        self.pf_win = DEF_PF_WIN
        self.pf_alpha = DEF_PF_ALPHA
        self.pf_weight_max = DEF_PF_WEIGHT_MAX
        self.pf_weight_k = DEF_PF_WEIGHT_K

        # tie 估計參數
        self.tie_min = DEF_TIE_MIN
        self.tie_max = DEF_TIE_MAX
        self.tie_beta_a = DEF_TIE_BETA_A
        self.tie_beta_b = DEF_TIE_BETA_B
        self.tie_ema_alpha = DEF_TIE_EMA_ALPHA
        self.tie_min_samples = DEF_TIE_MIN_SAMPLES
        self.tie_delta = DEF_TIE_DELTA
        self.tie_cap = DEF_TIE_MAX_CAP
        self.tie_floor = DEF_TIE_MIN_FLOOR

        # 抖動
        self.jitter = DEF_PROB_JITTER

        # streak 偏壓
        self.streak_enable = DEF_STREAK_ENABLE
        self.streak_min = DEF_STREAK_MIN
        self.streak_gain = DEF_STREAK_GAIN
        self.streak_maxw = DEF_STREAK_MAXW
        self.streak_relief = DEF_STREAK_RELIEF

        # 模式：以實例優先，否則讀 ENV
        mm = (self.model_mode or os.getenv("MODEL_MODE", "indep")).strip().lower()
        if mm not in ("indep", "learn"):
            warnings.warn(f"[pfilter] MODEL_MODE='{mm}' 非法，改用 indep")
            mm = "indep"
        self.model_mode = mm

        # 其他
        self.decay = DEF_PF_DECAY

    # ===== 基本 API =====
    def update_point_history(self, p_pts: int, b_pts: int):
        self.prev_p_pts = int(p_pts)
        self.prev_b_pts = int(b_pts)

    def update_outcome(self, outcome: int):
        """outcome: 0=莊, 1=閒, 2=和"""
        if outcome not in (0, 1, 2):
            return

        # learn：長期計數帶衰減
        if self.model_mode == "learn":
            self.counts *= self.decay
            self.counts[outcome] += 1.0

        # 滾動窗口
        self.history_window.append(outcome)
        if len(self.history_window) > max(self.hist_win, self.pf_win, 100):
            self.history_window.pop(0)

        # tie 統計（僅用於動態 tie）
        if outcome == 2:
            self._tie_seen += 1

        # 更新 streak（不含和）
        if outcome in (0, 1):
            if self._streak_side is None or self._streak_side != outcome:
                self._streak_side = outcome
                self._streak_len = 1
            else:
                self._streak_len += 1

        # 更新動態 tie 範圍
        if self.dynamic_tie:
            self._update_tie_bounds()

    # ===== 動態和局：Beta + EMA + 樣本門檻 =====
    def _update_tie_bounds(self):
        n = len(self.history_window)
        if n < self.tie_min_samples:
            # 樣本不足，維持固定範圍
            return

        tie_count = sum(1 for o in self.history_window[-self.tie_min_samples:] if o == 2)
        # Beta 後驗期望（帶先驗）
        post_a = self.tie_beta_a + tie_count
        post_b = self.tie_beta_b + (self.tie_min_samples - tie_count)
        mu = post_a / max(post_a + post_b, EPS)

        # EMA 平滑
        if self._tie_mu_ema is None:
            self._tie_mu_ema = float(mu)
        else:
            a = self.tie_ema_alpha
            self._tie_mu_ema = a * float(mu) + (1 - a) * float(self._tie_mu_ema)

        # 以 EMA 均值決定上下限（±delta）
        base = float(self._tie_mu_ema)
        lo = max(self.tie_floor, base * (1.0 - self.tie_delta))
        hi = min(self.tie_cap,   base * (1.0 + self.tie_delta))

        # 落回安全範圍
        lo = max(lo, DEF_TIE_MIN)
        hi = min(hi, DEF_TIE_MAX)

        self.tie_min = lo
        self.tie_max = hi

    # ===== 抖動（Dirichlet 優先；必要時小幅 Gaussian 再正規化） =====
    def _dirichlet_jitter(self, probs: np.ndarray, strength: float = 100.0) -> np.ndarray:
        alpha = np.clip(probs, EPS, 1.0) * float(strength) + 1.0
        return _safe_normalize(self.rng.dirichlet(alpha))

    # ===== 獨立模式：輕量歷史混合（Dirichlet/Laplace 平滑） =====
    def _light_historical_update(self, probs: np.ndarray) -> np.ndarray:
        hw = min(self.hist_win, len(self.history_window))
        if hw < DEF_MIN_HISTORY_FOR_LEARNING:
            return probs

        recent = self.history_window[-hw:]
        # Dirichlet/Laplace 平滑計數
        cB = recent.count(0) + self.hist_pseudo
        cP = recent.count(1) + self.hist_pseudo
        cT = recent.count(2) + self.hist_pseudo
        hist_probs = _safe_normalize(np.array([cB, cP, cT], dtype=np.float64))

        # 依視窗長度決定混合權重，上限 hist_weight_max
        w = min(self.hist_weight_max, hw / 100.0)
        out = _safe_normalize((1 - w) * probs + w * hist_probs)
        return out

    # ===== 學習模式：簡化 PF 後驗（帶 pseudo-counts） =====
    def _particle_filter_predict(self, sims_scale: float) -> np.ndarray:
        """
        sims_scale：由 sims_per_particle 推得的尺度（越大代表越穩定，權重↑、抖動↓）
        """
        pw = min(1.0, self.pf_win / (self.pf_win + 10.0))  # 避免小視窗過強
        wlen = min(self.pf_win, len(self.history_window))
        if wlen == 0:
            return self.prior.copy()

        recent = self.history_window[-wlen:]
        cnt = np.array([
            recent.count(0),
            recent.count(1),
            recent.count(2),
        ], dtype=np.float64)

        # Dirichlet pseudo-counts（alpha）
        alpha = np.array([self.pf_alpha, self.pf_alpha, self.pf_alpha], dtype=np.float64)
        post = cnt + alpha
        pf_probs = _safe_normalize(post)

        # 視窗大小與 sims_scale → 權重映射（平滑至 pf_weight_max）
        # sims_scale 取自 sims_per_particle（越大越信任 PF）
        base = min(1.0, wlen / max(self.pf_weight_k, 1.0))
        weight = min(self.pf_weight_max, 0.5 * base + 0.5 * min(1.0, sims_scale))
        # 與 prior 作柔性混合，避免過度短期化
        mixed = _safe_normalize((1 - weight) * self.prior + weight * pf_probs)
        return mixed

    # ===== 連莊偏壓（在 simplex 內柔性偏壓，不會把對側歸零） =====
    def _apply_streak_bias(self, probs: np.ndarray) -> np.ndarray:
        if not self.streak_enable:
            return probs
        if self._streak_len < self.streak_min or self._streak_side not in (0, 1):
            return probs

        # 權重：隨 streak_len 緩增，並限制上限
        extra = max(0, self._streak_len - self.streak_min)
        gain = np.log1p(extra) * self.streak_gain
        w = float(min(self.streak_maxw, max(0.0, gain)))

        # 對齊莊/閒兩類（不偏和）
        p = probs.copy().astype(np.float64)
        target_idx = self._streak_side  # 0 or 1
        opp_idx = 1 - target_idx
        tie_idx = 2

        # 先保留對側一定比例，再把剩餘質量往 target 混
        p_opp = p[opp_idx]
        keep_opp = self.streak_relief * p_opp
        move_opp = p_opp - keep_opp

        # 從 tie 也抽一點（依 w 比例）
        take_tie = w * p[tie_idx] * 0.5
        p[tie_idx] = max(EPS, p[tie_idx] - take_tie)

        p[target_idx] += (move_opp + take_tie)
        p[opp_idx] = keep_opp

        return _safe_normalize(p)

    # ===== 後驗平均（長期計數 + 先驗） =====
    def _posterior_mean(self) -> np.ndarray:
        post = self.prior * self.prior_strength + self.counts
        return _safe_normalize(post)

    # ===== 對外預測 =====
    def predict(self, sims_per_particle: int = 30) -> np.ndarray:
        """
        回傳 [pB, pP, pT]（float32）
        - sims_per_particle：實際用來調控抖動強度與 PF 權重
        """
        # sims 尺度（限制 1~200 之間，再映射到 0.2~1.0）
        spp = max(1, int(sims_per_particle))
        sims_scale = min(1.0, max(0.2, np.log1p(spp) / np.log1p(200.0)))

        if self.model_mode == "indep":
            # 先從 prior 出發 → 輕量歷史混合
            probs = self.prior.copy()
            probs = self._light_historical_update(probs)

            # 抖動：小抖動用 Gaussian，較大抖動用 Dirichlet（且強度隨 sims_scale 反向）
            if self.jitter > 0:
                if self.jitter > 0.01:
                    # Dirichlet 抖動強度與 sims_scale 反向（sims 大 → 抖動小）
                    strength = 60.0 + 140.0 * sims_scale  # 60~200
                    probs = self._dirichlet_jitter(probs, strength=strength)
                else:
                    noise = self.rng.normal(0.0, self.jitter * (1.2 - sims_scale), size=3)
                    probs = _safe_normalize(probs + noise)

        elif self.model_mode == "learn":
            # PF 近似 + 長期貝氏後驗 混合（視窗權重與 sims_scale 決定）
            pf_probs = self._particle_filter_predict(sims_scale=sims_scale)
            bayes_probs = self._posterior_mean()
            # 權重：與 PF 邏輯一致（這裡再做一次柔性融合）
            pf_w = min(self.pf_weight_max, 0.6 * sims_scale + 0.4 * (len(self.history_window) / max(self.pf_weight_k, 1.0)))
            probs = _safe_normalize(pf_w * pf_probs + (1 - pf_w) * bayes_probs)
        else:
            # fallback：保守 prior
            probs = self.prior.copy()

        # 動態和局上下限（若開啟）
        if self.dynamic_tie:
            p = probs.copy().astype(np.float64)
            # 只裁 tie，剩餘按比例縮放回 B/P
            tie = float(np.clip(p[2], self.tie_min, self.tie_max))
            rest = max(EPS, 1.0 - tie)
            bp_sum = max(EPS, p[0] + p[1])
            p[0] = p[0] / bp_sum * rest
            p[1] = p[1] / bp_sum * rest
            p[2] = tie
            probs = _safe_normalize(p)

        # 連莊偏壓（可關掉）
        probs = self._apply_streak_bias(probs)

        return probs.astype(np.float32)
