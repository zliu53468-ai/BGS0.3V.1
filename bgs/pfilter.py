diff --git a/bgs/pfilter.py b/bgs/pfilter.py
index f7b0f73a2dab6c5e768c720979397f56c1a96744..1267a972a052ce25fb0ff5709466717539783bd0 100644
--- a/bgs/pfilter.py
+++ b/bgs/pfilter.py
@@ -46,73 +46,89 @@ PF_DECAY = _env_float("PF_DECAY", "0.985")
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
 
 # 是否排除「上一手」對下一手的影響（建議預設開啟）
 EXCLUDE_LAST_OUTCOME = _env_flag("EXCLUDE_LAST_OUTCOME", "1")
 
+# KF head smoothing（Kalman + window 平滑）
+KF_HEAD_ON = _env_flag("KF_HEAD_ON", "1")
+KF_HEAD_WINDOW = max(1, _env_int("KF_HEAD_WINDOW", "10"))
+KF_PROCESS_NOISE = _env_float("KF_PROCESS_NOISE", "1e-4")
+KF_MEAS_NOISE = _env_float("KF_MEAS_NOISE", "2.5e-3")
+KF_INIT_VARIANCE = _env_float("KF_INIT_VARIANCE", "0.0025")
+KF_MIN_VARIANCE = _env_float("KF_MIN_VARIANCE", "1e-6")
+
 # Particle filter（learn 分支混合用權重）
 PF_WIN = _env_int("PF_WIN", "50")
 PF_ALPHA = _env_float("PF_ALPHA", "0.5")
 PF_WEIGHT_MAX = _env_float("PF_WEIGHT_MAX", "0.7")
 PF_WEIGHT_K = _env_float("PF_WEIGHT_K", "80.0")
 
 # ===== 真 PF 參數（可從環境注入；若未用也相容）=====
 PF_N = _env_int("PF_N", "120")
 PF_RESAMPLE = _env_float("PF_RESAMPLE", "0.73")      # ESS/N 門檻
 PF_KAPPA = _env_float("PF_KAPPA", "220.0")           # 狀態轉移濃度（越大越穩）
 PF_REJUV = _env_float("PF_REJUV", "220.0")           # 重採樣後再活化濃度
 DIRICHLET_EPS = _env_float("PF_DIR_EPS", "0.012")
 STABILITY_FACTOR = _env_float("PF_STAB_FACTOR", "0.8")
 
 # ===== 點差偏置（只在 B/P 子空間）=====
 POINT_BIAS_ON          = _env_flag("POINT_BIAS_ON", "1")
 POINT_BIAS_K           = _env_float("POINT_BIAS_K", "0.35")          # tanh 斜率
 POINT_BIAS_MAX_SHIFT   = _env_float("POINT_BIAS_MAX_SHIFT", "0.04")  # 建議 0.04~0.06
 POINT_BIAS_MIN_GAP     = _env_int("POINT_BIAS_MIN_GAP", "3")         # 至少差 3 才啟動
 POINT_BIAS_DAMP_N      = _env_int("POINT_BIAS_DAMP_N", "20")         # 小樣本衰減門檻
 POINT_BIAS_TIE_DAMP_AT = _env_float("POINT_BIAS_TIE_DAMP_AT", "0.14")# 和局高時衰減
 POINT_BIAS_TIE_DAMP    = _env_float("POINT_BIAS_TIE_DAMP", "0.5")    # 衰減係數
 
+# CUSUM guard（避免單邊追太久）
+CUSUM_ON         = _env_flag("CUSUM_ON", "1")
+CUSUM_DECAY      = _env_float("CUSUM_DECAY", "0.92")
+CUSUM_DRIFT      = _env_float("CUSUM_DRIFT", "0.18")
+CUSUM_THRESHOLD  = _env_float("CUSUM_THRESHOLD", "2.4")
+CUSUM_GAIN       = _env_float("CUSUM_GAIN", "0.06")
+CUSUM_MAX_SHIFT  = _env_float("CUSUM_MAX_SHIFT", "0.1")
+
 EPS = 1e-9
 
 
 class SimpleDirichletPF:
     """輕量級 PF：粒子為 (B,P,T) simplex。用 Dirichlet 抖動保持在 simplex，
     觀測時以對應 outcome 的機率加權，ESS 低於門檻則行系統式重採樣。
     * 兼容 kappa / rejuvenate 兩個參數（目前僅儲存，未強制使用）。
     """
 
     def __init__(
         self,
         prior: np.ndarray,
         prior_strength: float,
         n_particles: int,
         dirichlet_eps: float,
         stability_factor: float,
         resample_thr: float,
         rng: np.random.Generator,
         # 相容 server/env 的兩個參數（可選）
         kappa: float | None = None,
         rejuvenate: float | None = None,
     ):
         self.rng = rng
         self.n_particles = max(1, int(n_particles))
         self.dirichlet_eps = float(max(EPS, dirichlet_eps))
diff --git a/bgs/pfilter.py b/bgs/pfilter.py
index f7b0f73a2dab6c5e768c720979397f56c1a96744..1267a972a052ce25fb0ff5709466717539783bd0 100644
--- a/bgs/pfilter.py
+++ b/bgs/pfilter.py
@@ -193,50 +209,59 @@ class OutcomePF:
     prev_b_pts: Optional[int] = None
 
     # internal states
     def __post_init__(self):
         self.rng = np.random.default_rng(self.seed)
 
         # base prior
         self.prior = np.array([PRIOR_B, PRIOR_P, PRIOR_T], dtype=np.float64)
         self.prior = self.prior / self.prior.sum()
 
         # long-term counts for bayes posterior (learn 混合用)
         self.counts = np.zeros(3, dtype=np.float64)
         self.counts_pre_update = self.counts.copy()   # 用於 EXCLUDE_LAST_OUTCOME
 
         # rolling window outcomes (0:B, 1:P, 2:T)
         self.history_window: List[int] = []
         self.max_hist_len = max(HIST_WIN, PF_WIN, 100)
         self.last_outcome: Optional[int] = None
 
         # dynamic tie trackers
         self.t_ema = None
         self.tie_samples = 0
         self.adaptive_tie_min = TIE_MIN
         self.adaptive_tie_max = TIE_MAX
 
+        # KF head state
+        self.kf_state: Optional[np.ndarray] = None
+        self.kf_cov: Optional[np.ndarray] = None
+        self.kf_measurements: List[np.ndarray] = []
+
+        # CUSUM trackers
+        self.cusum_pos = 0.0
+        self.cusum_neg = 0.0
+
         # 真 PF（僅 learn 使用）
         self.pf: Optional[SimpleDirichletPF] = None
         self._pf_prediction_before_last: Optional[np.ndarray] = None
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
             # 預先填充一個 PF 預測，給 EXCLUDE_LAST_OUTCOME 使用
             self._pf_prediction_before_last = self.pf.predict()
 
     # -------- external updates --------
     def update_point_history(self, p_pts: int, b_pts: int):
         self.prev_p_pts = int(p_pts)
         self.prev_b_pts = int(b_pts)
 
     def update_outcome(self, outcome: int):
         if outcome not in (0, 1, 2):
diff --git a/bgs/pfilter.py b/bgs/pfilter.py
index f7b0f73a2dab6c5e768c720979397f56c1a96744..1267a972a052ce25fb0ff5709466717539783bd0 100644
--- a/bgs/pfilter.py
+++ b/bgs/pfilter.py
@@ -244,98 +269,139 @@ class OutcomePF:
 
         self.last_outcome = outcome
 
         if MODEL_MODE == "learn":
             # 長期貝氏計數（先衰減再加一）
             self.counts *= PF_DECAY
             self.counts_pre_update = self.counts.copy()  # 記錄「尚未含本手」的狀態
             self.counts[outcome] += 1.0
 
             # PF 更新（若要排除上一手在下一手的預測，先取一份 predict 緩存）
             if self.pf is not None:
                 if EXCLUDE_LAST_OUTCOME:
                     # 這份快取代表「寫入本手 outcome 之前」PF 的看法
                     self._pf_prediction_before_last = self.pf.predict()
                 self.pf.update(outcome)
 
         # 滾動視窗（此處照常寫入，但預測階段可選擇不取最後一筆）
         self.history_window.append(outcome)
         if len(self.history_window) > self.max_hist_len:
             self.history_window.pop(0)
 
         # 動態和局跟蹤
         if DYNAMIC_TIE_RANGE:
             self._update_tie_trackers()
 
+        self._update_cusum_trackers(outcome)
+
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
 
+    def _update_cusum_trackers(self, outcome: int):
+        if not CUSUM_ON:
+            return
+        decay = float(np.clip(CUSUM_DECAY, 0.0, 1.0))
+        drift = float(max(0.0, CUSUM_DRIFT))
+        self.cusum_pos *= decay
+        self.cusum_neg *= decay
+        if outcome not in (0, 1):
+            return
+        x = 1.0 if outcome == 1 else -1.0
+        self.cusum_pos = max(0.0, self.cusum_pos + x - drift)
+        self.cusum_neg = min(0.0, self.cusum_neg + x + drift)
+
     # -------- smoothing utilities (indep 用) --------
     def _dirichlet_jitter(self, probs: np.ndarray, strength: float = 120.0) -> np.ndarray:
         s = float(max(EPS, strength))
         alpha = np.clip(probs, EPS, 1.0) * s
         return self.rng.dirichlet(alpha)
 
     def _compute_jitter_strength(self) -> Optional[float]:
         """
         依樣本量自動調整擾動強度：樣本越多 → 抖動越小。回傳 alpha strength；
         若不需要抖動則回傳 None。
         """
         if PROB_JITTER <= 0:
             return None
         base = max(50.0, min(PROB_JITTER_STRENGTH_MAX, 1.0 / max(1e-6, PROB_JITTER)))
         n = len(self.history_window)
         if n <= 0 or PROB_JITTER_SCALE <= 0:
             return base
         growth = 1.0 + min(4.0, n / PROB_JITTER_SCALE)  # 最多放大到 ~5x
         strength = base * growth
         return float(np.clip(strength, 50.0, PROB_JITTER_STRENGTH_MAX))
 
+    def _apply_kf_head(self, probs: np.ndarray) -> np.ndarray:
+        if not KF_HEAD_ON:
+            return probs
+        measurement = np.clip(probs, EPS, None)
+        measurement = measurement / measurement.sum()
+        self.kf_measurements.append(measurement)
+        if len(self.kf_measurements) > KF_HEAD_WINDOW:
+            self.kf_measurements.pop(0)
+        stacked = np.stack(self.kf_measurements, axis=0)
+        measurement_avg = np.mean(stacked, axis=0)
+        measurement_avg = np.clip(measurement_avg, EPS, None)
+        measurement_avg = measurement_avg / measurement_avg.sum()
+        if self.kf_state is None or self.kf_cov is None:
+            self.kf_state = measurement_avg.copy()
+            self.kf_cov = np.full(3, KF_INIT_VARIANCE, dtype=np.float64)
+            return measurement_avg
+        pred_state = self.kf_state
+        pred_cov = self.kf_cov + KF_PROCESS_NOISE
+        gain = pred_cov / (pred_cov + KF_MEAS_NOISE)
+        updated_state = pred_state + gain * (measurement_avg - pred_state)
+        updated_cov = (1.0 - gain) * pred_cov
+        self.kf_state = np.clip(updated_state, EPS, None)
+        self.kf_cov = np.clip(updated_cov, KF_MIN_VARIANCE, None)
+        smoothed = self.kf_state / self.kf_state.sum()
+        return smoothed
+
     # 小工具：可控是否包含「最後一手」的視窗
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
             sum(1 for x in window if x == 2)
         ], dtype=np.float64)
         hist_probs = (counts + HIST_PSEUDO) / max(EPS, (n + 3.0 * HIST_PSEUDO))
         w_dyn = n / (n + 80.0)  # 緩升
         w = min(HISTORICAL_WEIGHT, HIST_WEIGHT_MAX, w_dyn)
         mixed = probs * (1 - w) + hist_probs * w
diff --git a/bgs/pfilter.py b/bgs/pfilter.py
index f7b0f73a2dab6c5e768c720979397f56c1a96744..1267a972a052ce25fb0ff5709466717539783bd0 100644
--- a/bgs/pfilter.py
+++ b/bgs/pfilter.py
@@ -381,62 +447,95 @@ class OutcomePF:
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
 
+    def _apply_cusum_guard(self, probs: np.ndarray) -> np.ndarray:
+        if not CUSUM_ON:
+            return probs
+        t = float(probs[2])
+        b = float(probs[0])
+        p = float(probs[1])
+        if (b + p) <= EPS:
+            return probs
+        if self.cusum_pos > CUSUM_THRESHOLD:
+            excess = self.cusum_pos - CUSUM_THRESHOLD
+            shift = min(CUSUM_MAX_SHIFT, excess * CUSUM_GAIN)
+            shift = min(shift, p - EPS)
+            if shift > 0:
+                b += shift
+                p -= shift
+        elif (-self.cusum_neg) > CUSUM_THRESHOLD:
+            excess = (-self.cusum_neg) - CUSUM_THRESHOLD
+            shift = min(CUSUM_MAX_SHIFT, excess * CUSUM_GAIN)
+            shift = min(shift, b - EPS)
+            if shift > 0:
+                b -= shift
+                p += shift
+        adjusted = np.array([b, p, t], dtype=np.float64)
+        adjusted = np.clip(adjusted, EPS, None)
+        adjusted /= adjusted.sum()
+        return adjusted
+
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
                 # 若排除上一手：優先使用「上一手更新前」的PF預測快取
                 if EXCLUDE_LAST_OUTCOME and (self._pf_prediction_before_last is not None):
                     pf_probs = self._pf_prediction_before_last.copy()
                 else:
                     pf_probs = self.pf.predict()
             else:
                 pf_probs = self._particle_filter_predict()
             bayes_probs = self._posterior_mean()
             n_pf = min(len(self.history_window), PF_WIN)
             w_pf = min(PF_WEIGHT_MAX, n_pf / (n_pf + PF_WEIGHT_K))
             probs = pf_probs * w_pf + bayes_probs * (1 - w_pf)
         else:
             probs = self.prior.copy()
 
-        # 2) Tie 夾取（guard rails）
+        # 2) KF head 平滑，降低尖峰
+        probs = self._apply_kf_head(probs)
+
+        # 3) Tie 夾取（guard rails）
         if DYNAMIC_TIE_RANGE and self.tie_samples >= TIE_MIN_SAMPLES:
             tie_min, tie_max = self.adaptive_tie_min, self.adaptive_tie_max
         else:
             tie_min, tie_max = TIE_MIN, TIE_MAX
         probs = np.clip(probs, EPS, None); probs /= probs.sum()
         probs[2] = np.clip(probs[2], tie_min, tie_max)
         probs = np.clip(probs, EPS, None); probs /= probs.sum()
 
-        # 3) 點數差偏置（只動 B/P；在夾 T 之後施加）
+        # 4) CUSUM guard：避免單邊追過久
+        probs = self._apply_cusum_guard(probs)
+
+        # 5) 點數差偏置（只動 B/P；在夾 T 之後施加）
         probs = self._apply_point_bias(probs)
 
         return probs.astype(np.float32)
