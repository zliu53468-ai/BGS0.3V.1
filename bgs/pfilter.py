# -*- coding: utf-8 -*-
"""
DirichletFeaturePF — 以「點數證據」為主、可切換記憶模式的獨立機率模型
適用：Baccarat（莊/閒/和）。

【本版重點（含你的補丁需求）】
- 支援 HISTORY_MODE：
  - off：完全無歷史（Stateless）。每局僅以先驗 × 當局點數證據做推斷。
  - ema（預設）：指數衰減記憶（PF_DECAY）。
  - cap：EMA + alpha 總量封頂（ALPHA_CAP）。
- 支援 ALPHA_CAP：限制分佈總量，避免歷史累積過大綁死模型。
- 預設 OUTCOME_WEIGHT=0.0 → 不吃趨勢（勝負不會進入學習）。
- 新增「均值回歸」MEAN_REVERT/REVERT_RATE：每次更新後，將分佈朝先驗微量拉回，抑制連龍黏著。
- 新增「輸出溫度軟化」SOFT_TAU：預測輸出做溫度平滑，避免極端百分比。
- 以理論機率 THEO 為 Dirichlet 先驗；點數差與高分方轉為 multiplicative 證據。
- predict() 會夾緊 Tie 機率到 [TIE_MIN, TIE_MAX]。

環境變數（皆可不設，採用預設值）：
- HISTORY_MODE      ：off | ema | cap（預設 ema）
- PF_DIR_EPS       ：先驗強度比例，預設 0.08（越大越靠理論值）
- PF_DECAY         ：衰減係數，預設 0.98（ema/cap 有效）
- ALPHA_CAP        ：總量封頂（>0 啟用；cap 模式有效）
- OUTCOME_WEIGHT   ：勝負寫入權重，預設 0.0（0=不吃趨勢）
- FEAT_TIE_BOOST   ：小差距時提升「和」的幅度，預設 0.35
- FEAT_EDGE_SCALE  ：高分方加權比例，預設 0.15
- TIE_MIN/TIE_MAX  ：Tie 輸出夾緊範圍，預設 0.02/0.22
- MEAN_REVERT      ：1=啟用均值回歸；0=關（預設 1）
- REVERT_RATE      ：均值回歸強度（0~1），預設 0.05
- SOFT_TAU         ：輸出軟化溫度（>1 越平滑），預設 1.15
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

def _env_str(name: str, default: str) -> str:
    try:
        return str(os.getenv(name, default)).strip().lower()
    except Exception:
        return default

@dataclass
class _Cfg:
    # 先驗/衰減/趨勢權重
    dirichlet_eps: float = _env_float("PF_DIR_EPS", "0.08")
    decay: float = _env_float("PF_DECAY", "0.98")
    outcome_w: float = _env_float("OUTCOME_WEIGHT", "0.0")
    # 點數→證據
    feat_tie_boost: float = _env_float("FEAT_TIE_BOOST", "0.35")
    feat_edge_scale: float = _env_float("FEAT_EDGE_SCALE", "0.15")
    # Tie 夾緊
    tie_min: float = _env_float("TIE_MIN", "0.02")
    tie_max: float = _env_float("TIE_MAX", "0.22")
    # 記憶模式與封頂
    history_mode: str = _env_str("HISTORY_MODE", "ema")   # off | ema | cap
    alpha_cap: float = _env_float("ALPHA_CAP", "0")        # <=0 表示不封頂
    # 均值回歸與輸出軟化
    mean_revert: int = int(_env_float("MEAN_REVERT", "1"))
    revert_rate: float = _env_float("REVERT_RATE", "0.05")
    soft_tau: float = _env_float("SOFT_TAU", "1.15")

class OutcomePF:
    """介面與 server.py 相容：
    - update_points(p_pts, b_pts)
    - update_outcome(outcome)
    - predict(sims_per_particle=...)
    - 屬性：n_particles、decks、sims_lik、resample_thr、backend
    """
    def __init__(self, decks: int = 8, seed: int = 42, n_particles: int = 50,
                 sims_lik: int = 30, resample_thr: float = 0.5, backend: str = "bayes",
                 dirichlet_eps: float = None):
        self.cfg = _Cfg()
        if dirichlet_eps is not None:
            self.cfg.dirichlet_eps = float(dirichlet_eps)
        self.rng = np.random.default_rng(int(seed))

        # 先驗 alpha（保存快照供 stateless 路徑使用）
        prior_strength = max(1.0, 100.0 * float(self.cfg.dirichlet_eps))
        self.alpha = THEO * prior_strength
        self._prior_alpha = self.alpha.copy()

        # 兼容屬性
        self.n_particles = int(n_particles)
        self.decks = int(decks)
        self.sims_lik = int(sims_lik)
        self.resample_thr = float(resample_thr)
        self._backend = f"dirichlet-feature({backend})"

        # 暫存最近一次點數證據乘數（predict 時也會用到）
        self._last_feat_mult = np.ones(3, dtype=np.float64)

    @property
    def backend(self) -> str:
        return self._backend

    # ---- 點數→證據 ----
    def update_points(self, p_pts: int, b_pts: int):
        # 計算乘數：差越小越偏向 T，高分方略增該側
        p_pts = int(p_pts); b_pts = int(b_pts)
        gap = abs(p_pts - b_pts)
        tie_boost = self.cfg.feat_tie_boost * np.exp(-0.8 * gap)
        edge_scale = self.cfg.feat_edge_scale
        mB = 1.0; mP = 1.0
        if b_pts > p_pts:
            mB += edge_scale * ((b_pts - p_pts) / 9.0)
        elif p_pts > b_pts:
            mP += edge_scale * ((p_pts - b_pts) / 9.0)
        mT = 1.0 + tie_boost
        self._last_feat_mult = np.array([mB, mP, mT], dtype=np.float64)

        mode = self.cfg.history_mode
        if mode == "off":
            # 無歷史：不持久化到 self.alpha；predict 時臨時套用
            return

        # ema / cap：持久化（帶衰減）
        self.alpha = np.maximum(self.alpha * self.cfg.decay, 1e-6)
        self.alpha *= self._last_feat_mult

        # mean reversion：往先驗拉回，避免連龍鎖死
        if self.cfg.mean_revert and 0.0 < self.cfg.revert_rate < 1.0:
            self.alpha = (1.0 - self.cfg.revert_rate) * self.alpha + self.cfg.revert_rate * self._prior_alpha

        # cap：對總量封頂
        if mode == "cap" and self.cfg.alpha_cap > 0:
            s = float(self.alpha.sum())
            if s > self.cfg.alpha_cap:
                self.alpha *= (self.cfg.alpha_cap / s)

    # ---- 可選：納入勝負（預設 outcome_w=0 → 不吃趨勢） ----
    def update_outcome(self, outcome: int):
        if outcome not in (0, 1, 2):
            return
        w = float(self.cfg.outcome_w)
        if w <= 0:
            return  # 完全不吃趨勢
        mode = self.cfg.history_mode
        if mode == "off":
            return  # 無歷史模式忽略勝負

        self.alpha = np.maximum(self.alpha * self.cfg.decay, 1e-6)
        self.alpha[outcome] += w

        # mean reversion：勝負更新後也做一次回拉
        if self.cfg.mean_revert and 0.0 < self.cfg.revert_rate < 1.0:
            self.alpha = (1.0 - self.cfg.revert_rate) * self.alpha + self.cfg.revert_rate * self._prior_alpha

        if mode == "cap" and self.cfg.alpha_cap > 0:
            s = float(self.alpha.sum())
            if s > self.cfg.alpha_cap:
                self.alpha *= (self.cfg.alpha_cap / s)

    # ---- 輸出機率 ----
    def predict(self, sims_per_particle: int = 5) -> np.ndarray:
        if self.cfg.history_mode == "off":
            # 以先驗為底，僅套用當局點數 multiplicative tilt
            tmp = self._prior_alpha.copy()
            tmp *= self._last_feat_mult
            probs = tmp / (tmp.sum() + 1e-12)
        else:
            probs = self.alpha / (self.alpha.sum() + 1e-12)

        # 夾緊 T，避免極端
        pB, pP, pT = float(probs[0]), float(probs[1]), float(probs[2])
        pT = float(np.clip(pT, self.cfg.tie_min, self.cfg.tie_max))
        scale = (1.0 - pT) / (pB + pP + 1e-12)
        pB *= scale; pP *= scale
        out = np.array([pB, pP, pT], dtype=np.float32)
        out /= out.sum()

        # softmax temperature（>1 越平、<1 越尖；我們只做 >1）：
        if self.cfg.soft_tau and self.cfg.soft_tau > 1.0:
            t = float(self.cfg.soft_tau)
            out = (out ** (1.0 / t)).astype(np.float32)
            out /= out.sum()

        return out
