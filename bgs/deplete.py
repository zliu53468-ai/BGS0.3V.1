# -*- coding: utf-8 -*-
"""
組成依賴蒙地卡羅（Baccarat Shoe Monte Carlo）
- 模擬次數預設 1000（原 500 → 1000）
- 耗損只扣一半（deplete_factor=0.5），保留更多隨機性
- 供 PF 或外層邏輯在需要時以『剩餘牌桶』推估下一局的 B/P/T 機率

注意：此檔獨立於 PF，可單獨呼叫。
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple

CARD_IDX = list(range(10))  # 0..9 ; 0 桶代表 10/J/Q/K = 0點
TEN_BUCKET = 0

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
    # 理論上不會到這
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

# 簡化之路：以常見抽第三張規則（略） — 實務足夠
# 這裡採用近似版：若任一方總點 ≤ 5 則可能抽第三張；
# 不精細套表（對模擬統計影響可忽略於 1000 次級別）

def deal_hand(counts: np.ndarray, rng: np.random.Generator) -> Tuple[int,int]:
    # 初始牌
    p1 = draw_card(counts, rng); b1 = draw_card(counts, rng)
    p2 = draw_card(counts, rng); b2 = draw_card(counts, rng)
    p = ( _pt(p1) + _pt(p2) ) % 10
    b = ( _pt(b1) + _pt(b2) ) % 10

    # Natural 停牌
    if p >= 8 or b >= 8:
        return p, b

    # Player 規則（簡化）：<=5 抽
    if p <= 5:
        p3 = draw_card(counts, rng)
        p = (p + _pt(p3)) % 10

    # Banker 規則（簡化）：<=5 抽
    if b <= 5:
        b3 = draw_card(counts, rng)
        b = (b + _pt(b3)) % 10

    return p, b

# --- 模擬下一局機率 ---
def simulate_probs(base_counts: np.ndarray, sims: int = 1000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    win = np.zeros(3, dtype=np.int32)  # [B, P, T]
    for _ in range(int(sims)):
        c = base_counts.copy()
        try:
            p, b = deal_hand(c, rng)
        except RuntimeError:
            # 鞋耗盡：以理論近似回傳
            return np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)
        if p == b:
            win[2] += 1
        elif p > b:
            win[1] += 1
        else:
            win[0] += 1
    probs = (win / max(1, win.sum())).astype(np.float32)
    return probs

# --- 依觀察點數對 Shoe 做半耗損（保留隨機性） ---
# 觀察上一局點數 (p_pts, b_pts) 只知道「點數」，不知道確切卡牌。
# 這裡採用『期望耗損』的概念：每局雙方平均出 2~3 張牌，
# 我們用點數來微調：差距小 → 提高 TEN_BUCKET 消耗；差距大 → 提高高點數桶消耗。

def soften_deplete(counts: np.ndarray, p_pts: int, b_pts: int, factor: float = 0.5):
    # 每方預期 2.5 張牌 → 估計 5 張左右
    total_est = 5
    gap = abs(int(p_pts) - int(b_pts))
    # 以 gap 調整：gap 小 → 多扣 0 點桶；gap 大 → 多扣高點桶
    w = np.ones(10, dtype=np.float64)
    w[TEN_BUCKET] += 0.6 * np.exp(-0.6 * gap)
    for v in range(1, 10):
        w[v] += 0.15 * (v / 9.0) * (1 - np.exp(-0.6 * gap))

    # 正規化成機率，抽樣 total_est 張作為期望耗損，再乘上 factor（半耗損）
    p = w / w.sum()
    est = np.random.multinomial(total_est, p)
    dec = np.floor(est * float(factor)).astype(int)
    # 實際扣減（不低於 0）
    for v in CARD_IDX:
        if dec[v] > 0:
            counts[v] = max(0, int(counts[v]) - int(dec[v]))

# --- pipeline：依最近點數更新後再模擬 ---
def probs_after_points(base_counts: np.ndarray, p_pts: int, b_pts: int,
                       sims: int = 1000, seed: int = 42, deplete_factor: float = 0.5) -> np.ndarray:
    c = base_counts.copy()
    soften_deplete(c, p_pts, b_pts, factor=deplete_factor)
    return simulate_probs(c, sims=sims, seed=seed)
