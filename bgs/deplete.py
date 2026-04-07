# -*- coding: utf-8 -*-
"""
REAL DEPLETE.py — 真實扣牌 + 定期重置 + 統一隨機源
"""
import numpy as np
import os
from typing import Tuple
RNG = np.random.default_rng(int(os.getenv("SEED", "42")))
CARD_IDX = np.arange(10, dtype=np.int32)
GLOBAL_COUNTS = None
def init_counts(decks: int = 8) -> np.ndarray:
    global GLOBAL_COUNTS
    if GLOBAL_COUNTS is None:
        counts = np.zeros(10, dtype=np.int32)
        counts[1:10] = 4 * decks
        counts[0] = 16 * decks
        GLOBAL_COUNTS = counts.copy()
    return GLOBAL_COUNTS
def maybe_reset_shoe(rounds_seen: int | None, reset_every: int = 60):
    global GLOBAL_COUNTS
    if rounds_seen is not None and rounds_seen > 0 and rounds_seen % reset_every == 0:
        GLOBAL_COUNTS = None
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
    banker_draw = b <= 2
    if player_draw:
        if b == 3 and p3_val != 8:
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
        if p == b:
            outcomes[2] += 1
        elif p > b:
            outcomes[1] += 1
        else:
            outcomes[0] += 1
    probs = outcomes / outcomes.sum()
    return probs.astype(np.float32)
def update_counts_from_points(counts: np.ndarray, p_pts: int, b_pts: int):
    remove_cards = 4
    if p_pts <= 5:
        remove_cards += 1
    if b_pts <= 5:
        remove_cards += 1
    weights = np.ones(10)
    weights[0] = 4
    weights = weights / weights.sum()
    for _ in range(remove_cards):
        idx = RNG.choice(10, p=weights)
        if counts[idx] > 0:
            counts[idx] -= 1
def probs_after_points(base_counts: np.ndarray,
                       p_pts: int,
                       b_pts: int,
                       sims: int = None,
                       deplete_factor: float = None,
                       rounds_seen: int | None = None) -> np.ndarray:
    global GLOBAL_COUNTS
    maybe_reset_shoe(rounds_seen, reset_every=60)
    if GLOBAL_COUNTS is None:
        init_counts()
    counts = GLOBAL_COUNTS if base_counts is None else base_counts
    update_counts_from_points(counts, p_pts, b_pts)
    c = counts.copy()
    if sims is None:
        sims = int(os.getenv("DEPLETEMC_SIMS", "2000"))
        if rounds_seen is not None:
            if rounds_seen > 50:
                sims = max(sims, 3000)
            elif rounds_seen > 30:
                sims = max(sims, 2500)
    result = simulate_probs(c, sims)
    SAFE = np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)
    if rounds_seen is None:
        alpha = 0.2
    elif rounds_seen < 20:
        alpha = 0.3
    elif rounds_seen < 50:
        alpha = 0.2
    else:
        alpha = 0.15
    result = result * (1 - alpha) + SAFE * alpha
    result = result / result.sum()
    return result.astype(np.float32)
def _stage_bounds():
    early_end = int(os.getenv("EARLY_HANDS", "20"))
    mid_end = int(os.getenv("MID_HANDS", "56"))
    return early_end, mid_end
def _stage_prefix(rounds_seen: int | None) -> str:
    if rounds_seen is None:
        return ""
    e_end, m_end = _stage_bounds()
    if rounds_seen < e_end:
        return "EARLY_"
    elif rounds_seen < m_end:
        return "MID_"
    else:
        return "LATE_"
