# -*- coding: utf-8 -*-
"""
組成依賴蒙地卡羅（Baccarat Shoe Monte Carlo）
 - 模擬次數預設 1000（原 500 → 1000）
 - 耗損程度調整 (deplete_factor=1.0)，默認完全耗損
 - 若出現自然牌型時減少牌桶耗損程度
 - 供 PF 或外層邏輯在需要時以『剩餘牌桶』推估下一局的 B/P/T 機率

注意：此檔獨立於 PF，可單獨呼叫。
"""
from __future__ import annotations
import numpy as np
import os
from dataclasses import dataclass
from typing import Tuple

# Initialize a global random number generator for simulations
RNG = np.random.default_rng(int(os.getenv("SEED", "42")))

CARD_IDX = list(range(10))  # 0..9 ; 0 桶代表 10/J/Q/K = 0點
TEN_BUCKET = 0

# --- 三段工具（只有在未明確傳 sims 時才會用到） ---
def _stage_bounds():
    early_end = int(os.getenv("EARLY_HANDS", "20"))
    mid_end   = int(os.getenv("MID_HANDS",   os.getenv("LATE_HANDS", "56")))
    return early_end, mid_end

def _stage_prefix(rounds_seen: int | None) -> str:
    if rounds_seen is None: return ""
    e_end, m_end = _stage_bounds()
    if rounds_seen < e_end: return "EARLY_"
    elif rounds_seen < m_end: return "MID_"
    else: return "LATE_"

# --- Shoe 初始化/拷貝 ---
def init_counts(decks: int = 8) -> np.ndarray:
    counts = np.zeros(10, dtype=np.int32)
    counts[1:10] = 4 * decks
    counts[TEN_BUCKET] = 16 * decks
    return counts

# --- 基本抽牌（不放回） ---
def draw_card(counts: np.ndarray, rng: np.random.Generator) -> int:
    tot = int(counts.sum())
    if tot <= 0:
        raise RuntimeError("Shoe empty")
    r = int(rng.integers(0, tot))
    acc = 0
    for v in CARD_IDX:
        acc += int(counts[v])
        if r < acc:
            counts[v] -= 1
            return v
    counts[9] -= 1
    return 9

# --- 百家樂點數規則 ---
@np.vectorize
def _pt(x: int) -> int:
    return 0 if x == 0 else (x % 10)

def points_add(a: int, b: int) -> int:
    return (a + b) % 10

@dataclass
class Hand:
    p1: int; p2: int; b1: int; b2: int
    p3: int = -1; b3: int = -1

# 完整百家樂補牌規則
def deal_hand(counts: np.ndarray, rng: np.random.Generator) -> Tuple[int,int]:
    p1 = draw_card(counts, rng); b1 = draw_card(counts, rng)
    p2 = draw_card(counts, rng); b2 = draw_card(counts, rng)
    p = (_pt(p1) + _pt(p2)) % 10
    b = (_pt(b1) + _pt(b2)) % 10
    if p >= 8 or b >= 8:
        return p, b
    if p <= 5:
        p3 = draw_card(counts, rng)
        p3_val = _pt(p3)
        p = (p + p3_val) % 10
        player_draw = True
    else:
        player_draw = False
        p3_val = -1
    if not player_draw:
        if b <= 5:
            b3 = draw_card(counts, rng)
            b = (b + _pt(b3)) % 10
    else:
        if b <= 2:
            b3 = draw_card(counts, rng); b = (b + _pt(b3)) % 10
        elif b == 3 and p3_val != 8:
            b3 = draw_card(counts, rng); b = (b + _pt(b3)) % 10
        elif b == 4 and 2 <= p3_val <= 7:
            b3 = draw_card(counts, rng); b = (b + _pt(b3)) % 10
        elif b == 5 and 4 <= p3_val <= 7:
            b3 = draw_card(counts, rng); b = (b + _pt(b3)) % 10
        elif b == 6 and 6 <= p3_val <= 7:
            b3 = draw_card(counts, rng); b = (b + _pt(b3)) % 10
    return p, b

# --- 模擬下一局機率 ---
def simulate_probs(base_counts: np.ndarray, sims: int = 1000) -> np.ndarray:
    win = np.zeros(3, dtype=np.int32)  # [B, P, T]
    for _ in range(int(sims)):
        c = base_counts.copy()
        try:
            p, b = deal_hand(c, RNG)
        except RuntimeError:
            return np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)
        if p == b:   win[2] += 1
        elif p > b:  win[1] += 1
        else:        win[0] += 1
    probs = (win / max(1, win.sum())).astype(np.float32)
    return probs

# --- 依觀察點數對 Shoe 做半耗損（保留隨機性） ---
def soften_deplete(counts: np.ndarray, p_pts: int, b_pts: int, factor: float = 0.5):
    total_est = 5
    gap = abs(int(p_pts) - int(b_pts))
    w = np.ones(10, dtype=np.float64)
    w[TEN_BUCKET] += 0.6 * np.exp(-0.6 * gap)
    for v in range(1, 10):
        w[v] += 0.15 * (v / 9.0) * (1 - np.exp(-0.6 * gap))
    p = w / w.sum()
    est = RNG.multinomial(total_est, p)
    dec = np.floor(est * float(factor)).astype(int)
    for v in CARD_IDX:
        if dec[v] > 0:
            counts[v] = max(0, int(counts[v]) - int(dec[v]))

# --- pipeline：依最近點數更新後再模擬（新增 rounds_seen 可選三段） ---
def probs_after_points(base_counts: np.ndarray, p_pts: int, b_pts: int,
                       sims: int = 1000, deplete_factor: float = 1.0,
                       rounds_seen: int | None = None) -> np.ndarray:
    # 若外層沒明確給 sims，才自動吃三段 DEPLETEMC_SIMS
    if sims is None:
        try:
            sims = int(float(os.getenv("DEPLETEMC_SIMS", "1000")))
        except:
            sims = 1000
        prefix = _stage_prefix(rounds_seen)
        if prefix:
            s = os.getenv(prefix + "DEPLETEMC_SIMS")
            if s not in (None, ""):
                try: sims = int(float(s))
                except: pass
            if prefix == "LATE_":
                late_dep = os.getenv("LATE_DEPLETEMC_SIMS")
                if late_dep not in (None, ""):
                    try: sims = int(float(late_dep))
                    except: pass

    c = base_counts.copy()
    if p_pts >= 8 or b_pts >= 8:
        deplete_factor = deplete_factor * 0.5
    soften_deplete(c, p_pts, b_pts, factor=deplete_factor)
    return simulate_probs(c, sims=int(sims))
