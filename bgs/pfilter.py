# -*- coding: utf-8 -*-
"""
DirichletFeaturePF — 以「點數證據」為主、趨勢為輔的獨立機率模型
適用：Baccarat（莊/閒/和）。

核心概念：
- 以理論機率 THEO 作為 Dirichlet 先驗 alpha。
- 每局輸入【上一局點數 (p_pts, b_pts)】轉成證據權重 multipliers，
  以貝式方式調整 alpha（不必依賴上一局勝負）。
- 是否納入勝負 outcome 可用 OUTCOME_WEIGHT 控制（預設 0 → 完全獨立於趨勢）。
- 輕度衰減 DECAY，避免歷史證據永久堆積。

環境變數（可選）：
- PF_DIR_EPS        ：float，先驗強度比例，預設 0.08（越大越靠近理論值）
- PF_DECAY          ：float，衰減係數，預設 0.98（越小越容易忘記歷史）
- OUTCOME_WEIGHT    ：float，勝負寫入 alpha 的權重，預設 0.0（不吃趨勢）
- FEAT_TIE_BOOST    ：float，點數差 |Δ| 小時提升「和」的幅度，預設 0.35
- FEAT_EDGE_SCALE   ：float，點數高者對該方提升幅度比例，預設 0.15
- TIE_MIN/TIE_MAX   ：float，夾緊和局輸出範圍，預設 0.02/0.22
"""
from __future__ import annotations
import os
import numpy as np
from dataclasses import dataclass

THEO = np.array([0.4586, 0.4462, 0.0952], dtype=np.float64)  # [B, P, T]

# ---- 讀環境變數工具 ----
def _env_float(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default).strip())
    except Exception:
        return float(default)

@dataclass
class _Cfg:
    dirichlet_eps: float = _env_float("PF_DIR_EPS", "0.08")
    decay: float = _env_float("PF_DECAY", "0.98")
    outcome_w: float = _env_float("OUTCOME_WEIGHT", "0.0")
    feat_tie_boost: float = _env_float("FEAT_TIE_BOOST", "0.35")
    feat_edge_scale: float = _env_float("FEAT_EDGE_SCALE", "0.15")
    tie_min: float = _env_float("TIE_MIN", "0.02")
    tie_max: float = _env_float("TIE_MAX", "0.22")

class OutcomePF:
    """介面與既有 server.py 相容：
    - update_outcome(outcome)
    - predict(sims_per_particle=...)
    - 可選：update_points(p_pts, b_pts) 供 server 呼叫
    """
    def __init__(self, decks: int = 8, seed: int = 42, n_particles: int = 50,
                 sims_lik: int = 30, resample_thr: float = 0.5, backend: str = "bayes",
                 dirichlet_eps: float = None):
        self.cfg = _Cfg()
        if dirichlet_eps is not None:
            self.cfg.dirichlet_eps = float(dirichlet_eps)
        self.rng = np.random.default_rng(int(seed))

        prior_strength = max(1.0, 100.0 * float(self.cfg.dirichlet_eps))
        self.alpha = THEO * prior_strength  # Dirichlet 先驗
        self._backend = f"dirichlet-feature({backend})"

        # 兼容 server.py 會讀取的屬性
        self.n_particles = int(n_particles)
        self.decks = int(decks)
        self.sims_lik = int(sims_lik)
        self.resample_thr = float(resample_thr)

        # 暫存最近一次點數證據（只影響當下 predict 前的 tilt）
        self._last_feat_mult = np.ones(3, dtype=np.float64)

    # ---- 供 server 使用：點數→證據 ----
    def update_points(self, p_pts: int, b_pts: int):
        # 以點數差塑形：差越小，越偏向和；高分一方略增該側
        gap = abs(int(p_pts) - int(b_pts))
        # tie boost：Δ 越小越大；Δ=0 時最大
        tie_boost = self.cfg.feat_tie_boost * np.exp(-0.8 * gap)
        # edge boost：高分方獲得小幅乘數
        edge_scale = self.cfg.feat_edge_scale
        mB = 1.0
        mP = 1.0
        if b_pts > p_pts:
            mB += edge_scale * ((b_pts - p_pts) / 9.0)
        elif p_pts > b_pts:
            mP += edge_scale * ((p_pts - b_pts) / 9.0)
        mT = 1.0 + tie_boost
        self._last_feat_mult = np.array([mB, mP, mT], dtype=np.float64)

        # 將乘數寫入 alpha（作為柔性的證據），再做輕度衰減
        self.alpha = np.maximum(self.alpha * self.cfg.decay, 1e-6)
        self.alpha *= self._last_feat_mult

    # ---- 可選：納入勝負結果（預設權重 0） ----
    def update_outcome(self, outcome: int):
        if outcome not in (0, 1, 2):
            return
        w = float(self.cfg.outcome_w)
        if w <= 0:
            return  # 完全不吃趨勢
        self.alpha = np.maximum(self.alpha * self.cfg.decay, 1e-6)
        self.alpha[outcome] += w

    def predict(self, sims_per_particle: int = 5) -> np.ndarray:
        # 基於 alpha 取機率
        probs = self.alpha / (self.alpha.sum() + 1e-12)
        # 夾緊 Tie，避免極端
        pB, pP, pT = float(probs[0]), float(probs[1]), float(probs[2])
        pT = float(np.clip(pT, self.cfg.tie_min, self.cfg.tie_max))
        scale = (1.0 - pT) / (pB + pP + 1e-12)
        pB *= scale; pP *= scale
        out = np.array([pB, pP, pT], dtype=np.float32)
        out /= out.sum()
        return out

    @property
    def backend(self) -> str:
        return self._backend
