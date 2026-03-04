# -*- coding: utf-8 -*-
"""
優化版 deplete.py — 高效蒙地卡羅 + 階段 sims + 自然牌耗損調整
"""
import numpy as np
import os
from typing import Tuple

RNG = np.random.default_rng(int(os.getenv("SEED", "42")))
CARD_IDX = np.arange(10, dtype=np.int32)

def init_counts(decks: int = 8) -> np.ndarray:
    counts = np.zeros(10, dtype=np.int32)
    counts[1:10] = 4 * decks
    counts[0] = 16 * decks
    return counts

def draw_card(counts: np.ndarray) -> int:
    tot = counts.sum()
    if tot <= 0:
        raise RuntimeError("Shoe empty")
    cum = np.cumsum(counts)
    r = RNG.integers(0, tot)
    idx = np.searchsorted(cum, r, side='right')
    counts[idx] -= 1
    return idx

def points(cards: np.ndarray) -> int:
    return np.sum(cards % 10) % 10

def deal_hand(counts: np.ndarray) -> Tuple[int, int]:
    p_cards = np.array([draw_card(counts), draw_card(counts)])
    b_cards = np.array([draw_card(counts), draw_card(counts)])
    p = points(p_cards)
    b = points(b_cards)
    if p >= 8 or b >= 8:
        return p, b
    player_draw = p <= 5
    p3_val = -1
    if player_draw:
        p3 = draw_card(counts)
        p3_val = p3 % 10
        p = points(np.append(p_cards, p3))
    banker_draw = False
    if b <= 2:
        banker_draw = True
    elif b == 3 and p3_val != 8:
        banker_draw = True
    elif b == 4 and 2 <= p3_val <= 7:
        banker_draw = True
    elif b == 5 and 4 <= p3_val <= 7:
        banker_draw = True
    elif b == 6 and 6 <= p3_val <= 7:
        banker_draw = True
    if banker_draw:
        b3 = draw_card(counts)
        b = points(np.append(b_cards, b3))
    return p, b

def simulate_probs(base_counts: np.ndarray, sims: int = 2000) -> np.ndarray:
    outcomes = np.zeros(3, dtype=np.int32)
    for _ in range(sims):
        c = base_counts.copy()
        try:
            p, b = deal_hand(c)
        except RuntimeError:
            return np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)
        if p == b: outcomes[2] += 1
        elif p > b: outcomes[1] += 1
        else: outcomes[0] += 1
    probs = outcomes / outcomes.sum()
    return probs.astype(np.float32)

def deplete_adjust(counts: np.ndarray, p_pts: int, b_pts: int, factor: float = 0.7):
    gap = abs(p_pts - b_pts)
    natural = p_pts >= 8 or b_pts >= 8
    eff_factor = factor * (0.3 if natural else 1.0)
    weights = np.ones(10, dtype=np.float64)
    weights[0] += 0.8 * (1 - np.exp(-gap / 3.0))
    weights[8:10] += 0.6
    weights /= weights.sum()
    remove_n = int(RNG.integers(4, 9) * eff_factor)
    remove = RNG.multinomial(remove_n, weights)
    counts -= remove
    counts = np.maximum(counts, 0)

def probs_after_points(base_counts: np.ndarray, p_pts: int, b_pts: int,
                       sims: int = None, deplete_factor: float = 0.7,
                       rounds_seen: int | None = None) -> np.ndarray:
    if sims is None:
        sims = int(os.getenv("DEPLETEMC_SIMS", "2000"))
        prefix = _stage_prefix(rounds_seen)
        if prefix:
            s = os.getenv(prefix + "DEPLETEMC_SIMS")
            if s: sims = int(float(s))
        if rounds_seen is not None and rounds_seen > 50:
            sims = max(sims, 3000)
    c = base_counts.copy()
    deplete_adjust(c, p_pts, b_pts, factor=deplete_factor)
    return simulate_probs(c, sims)

def _stage_bounds():
    early_end = int(os.getenv("EARLY_HANDS", "20"))
    mid_end = int(os.getenv("MID_HANDS", os.getenv("LATE_HANDS", "56")))
    return early_end, mid_end

def _stage_prefix(rounds_seen: int | None) -> str:
    if rounds_seen is None: return ""
    e_end, m_end = _stage_bounds()
    if rounds_seen < e_end: return "EARLY_"
    elif rounds_seen < m_end: return "MID_"
    else: return "LATE_"
