# -*- coding: utf-8 -*-
"""
REAL DEPLETE.py — 真實扣牌 + 合理牌局反推軟扣牌 + 定期重置 + 統一隨機源

輸出機率順序：
    [Banker, Player, Tie]
    [莊, 閒, 和]

本版重點：
1. 修正百家樂莊家補牌規則：
   - 閒未補牌時，莊 0~5 補，6~7 停
   - 閒有補牌時，依第三張牌規則補牌

2. update_counts_from_points 改成「多候選合理牌局 + 軟扣牌」
   - 不再找到第一組就硬扣
   - 會收集多組符合上一局點數的合理牌局
   - 隨機挑一組後，再依 DEPLETEMC_SOFT_DEDUCT_RATE 軟扣
   - 避免模型太黏上一局結果 / 上一局點數

3. 保留原函式名稱，方便 server.py 不用改
"""

import os
from typing import Tuple, Optional, List

import numpy as np


# ---------------------------------------------------------
# Global
# ---------------------------------------------------------

SAFE_PROBS = np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)

RNG = np.random.default_rng(int(os.getenv("SEED", "42")))

GLOBAL_COUNTS: Optional[np.ndarray] = None


# ---------------------------------------------------------
# Env helpers
# ---------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")


# ---------------------------------------------------------
# Shoe / counts
# ---------------------------------------------------------

def _make_counts(decks: int = 8) -> np.ndarray:
    counts = np.zeros(10, dtype=np.int32)

    # 百家樂點數：
    # A=1, 2~9=2~9, 10/J/Q/K=0
    counts[1:10] = 4 * int(decks)
    counts[0] = 16 * int(decks)

    return counts


def init_counts(decks: int = 8) -> np.ndarray:
    """
    初始化全域牌靴。

    注意：
    回傳的是 GLOBAL_COUNTS 本體，會被扣牌邏輯持續更新。
    """
    global GLOBAL_COUNTS

    if GLOBAL_COUNTS is None:
        GLOBAL_COUNTS = _make_counts(decks)

    return GLOBAL_COUNTS


def maybe_reset_shoe(rounds_seen: int | None, reset_every: int | None = None) -> None:
    """
    定期重置牌靴。

    預設 60 局重置，可用環境變數：
        SHOE_RESET_EVERY=60
    """
    global GLOBAL_COUNTS

    if reset_every is None:
        reset_every = _env_int("SHOE_RESET_EVERY", 60)

    try:
        reset_every = int(reset_every)
    except Exception:
        reset_every = 60

    if reset_every <= 0:
        return

    if rounds_seen is not None and rounds_seen > 0 and rounds_seen % reset_every == 0:
        GLOBAL_COUNTS = None


# ---------------------------------------------------------
# Baccarat core rules
# ---------------------------------------------------------

def draw_card(counts: np.ndarray) -> int:
    """
    從目前 counts 抽一張牌，並扣除。
    """
    counts = np.asarray(counts, dtype=np.int32)

    total = int(counts.sum())

    if total <= 0:
        raise RuntimeError("Shoe empty")

    cum = np.cumsum(counts)
    r = int(RNG.integers(0, total))
    idx = int(np.searchsorted(cum, r, side="right"))

    if idx < 0 or idx > 9 or counts[idx] <= 0:
        positive = np.where(counts > 0)[0]
        if len(positive) <= 0:
            raise RuntimeError("Shoe empty")
        idx = int(RNG.choice(positive))

    counts[idx] -= 1
    return idx


def points(cards: np.ndarray | List[int]) -> int:
    """
    百家樂點數總和 mod 10。
    傳入的 card 已經是 0~9 點數。
    """
    return int(np.sum(np.asarray(cards, dtype=np.int32) % 10) % 10)


def deal_hand(counts: np.ndarray, return_cards: bool = False):
    """
    模擬一局百家樂。

    回傳：
        return_cards=False:
            p, b

        return_cards=True:
            p, b, used_cards

    used_cards 為實際抽到的點數牌，用於反推扣牌。
    """

    used_cards: List[int] = []

    def _draw() -> int:
        c = draw_card(counts)
        used_cards.append(c)
        return c

    # 初始兩張
    p_cards = [_draw(), _draw()]
    b_cards = [_draw(), _draw()]

    p = points(p_cards)
    b = points(b_cards)

    # 天牌：任一方 8 / 9，雙方不補
    if p >= 8 or b >= 8:
        if return_cards:
            return p, b, used_cards
        return p, b

    # 閒家補牌規則
    player_draw = p <= 5
    p3_val = None

    if player_draw:
        p3 = _draw()
        p_cards.append(p3)
        p3_val = p3 % 10
        p = points(p_cards)

    # 莊家補牌規則
    if not player_draw:
        # 閒家未補牌時，莊家 0~5 補牌，6~7 停牌
        banker_draw = b <= 5
    else:
        # 閒家有補第三張時，莊家依第三張牌規則
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
        b3 = _draw()
        b_cards.append(b3)
        b = points(b_cards)

    if return_cards:
        return p, b, used_cards

    return p, b


# ---------------------------------------------------------
# Simulation
# ---------------------------------------------------------

def simulate_probs(base_counts: np.ndarray, sims: int = 2000) -> np.ndarray:
    """
    從目前扣牌後的 counts 模擬下一局機率。

    回傳：
        [莊, 閒, 和]
    """

    if base_counts is None:
        return SAFE_PROBS.copy()

    base_counts = np.asarray(base_counts, dtype=np.int32)

    if base_counts.sum() <= 0:
        return SAFE_PROBS.copy()

    sims = max(50, int(sims))

    outcomes = np.zeros(3, dtype=np.int32)

    for _ in range(sims):
        c = base_counts.copy()

        try:
            p, b = deal_hand(c)
        except Exception:
            return SAFE_PROBS.copy()

        if p == b:
            outcomes[2] += 1
        elif p > b:
            outcomes[1] += 1
        else:
            outcomes[0] += 1

    total = int(outcomes.sum())

    if total <= 0:
        return SAFE_PROBS.copy()

    probs = outcomes.astype(np.float64) / float(total)

    if (not np.isfinite(probs).all()) or probs.sum() <= 0:
        return SAFE_PROBS.copy()

    probs = probs / probs.sum()
    return probs.astype(np.float32)


# ---------------------------------------------------------
# Smarter deplete from known final points
# ---------------------------------------------------------

def _remove_used_cards(counts: np.ndarray, used_cards: List[int]) -> bool:
    """
    軟扣牌版本：
    找到符合上一局點數的合理牌局後，不是 100% 全扣，
    而是依照 DEPLETEMC_SOFT_DEDUCT_RATE 做機率扣牌。

    這樣可以避免模型太黏上一局點數 / 上一局結果。

    建議參數：
        DEPLETEMC_SOFT_DEDUCT_RATE=0.55

    若還是太黏上一局：
        改成 0.40

    若想讓扣牌影響更強：
        改成 0.70
    """

    if counts is None:
        return False

    soft_rate = _env_float("DEPLETEMC_SOFT_DEDUCT_RATE", 0.55)
    soft_rate = max(0.0, min(1.0, float(soft_rate)))

    tmp = np.asarray(counts, dtype=np.int32).copy()

    deducted_any = False

    for card in used_cards:
        card = int(card) % 10

        if tmp[card] <= 0:
            continue

        # 軟扣牌：不是每張都扣，避免過度相信上一局
        if float(RNG.random()) <= soft_rate:
            tmp[card] -= 1
            deducted_any = True

    if deducted_any:
        counts[:] = tmp
        return True

    return False


def _fallback_remove_cards(counts: np.ndarray, p_pts: int, b_pts: int) -> None:
    """
    最後備援扣牌：
    如果反推合理牌局失敗，才用保守權重扣牌。

    這不是主邏輯，只是避免極端情況無法扣牌。
    """

    if counts is None or counts.sum() <= 0:
        return

    remove_cards = 4

    try:
        p_pts = int(p_pts)
        b_pts = int(b_pts)
    except Exception:
        p_pts, b_pts = 0, 0

    if p_pts <= 5:
        remove_cards += 1
    if b_pts <= 5:
        remove_cards += 1

    remove_cards = max(4, min(6, remove_cards))

    # fallback 也做得更保守，避免找不到候選時突然重扣
    fallback_rate = _env_float("DEPLETEMC_FALLBACK_DEDUCT_RATE", 0.45)
    fallback_rate = max(0.0, min(1.0, float(fallback_rate)))

    available = np.maximum(np.asarray(counts, dtype=np.int32), 0).astype(np.float64)

    if available.sum() <= 0:
        return

    for _ in range(remove_cards):
        if available.sum() <= 0:
            break

        if float(RNG.random()) > fallback_rate:
            continue

        probs = available / available.sum()
        idx = int(RNG.choice(10, p=probs))

        if counts[idx] > 0:
            counts[idx] -= 1
            available[idx] -= 1


def update_counts_from_points(
    counts: np.ndarray,
    p_pts: int,
    b_pts: int,
    match_tries: int | None = None,
) -> None:
    """
    根據上一局最終點數，反推合理牌局並做「軟扣牌」。

    這版避免太黏上一局結果：
    1. 不只找第一組符合牌局
    2. 會收集多組候選牌局
    3. 從候選裡隨機挑一組
    4. 再用軟扣牌比例扣牌
    """

    if counts is None:
        return

    try:
        p_pts = int(p_pts)
        b_pts = int(b_pts)
    except Exception:
        return

    if not (0 <= p_pts <= 9 and 0 <= b_pts <= 9):
        return

    if match_tries is None:
        match_tries = _env_int("DEPLETEMC_MATCH_TRIES", 600)

    match_tries = max(50, int(match_tries))

    max_candidates = _env_int("DEPLETEMC_MATCH_CANDIDATES", 12)
    max_candidates = max(1, min(50, int(max_candidates)))

    base = np.asarray(counts, dtype=np.int32)

    if base.sum() <= 0:
        return

    candidates: List[List[int]] = []

    # 收集多組合理牌局，不要第一組就直接扣
    for _ in range(match_tries):
        c = base.copy()

        try:
            sim_p, sim_b, used_cards = deal_hand(c, return_cards=True)
        except Exception:
            continue

        if sim_p == p_pts and sim_b == b_pts:
            candidates.append(list(used_cards))

            if len(candidates) >= max_candidates:
                break

    if candidates:
        pick_idx = int(RNG.integers(0, len(candidates)))
        used_cards = candidates[pick_idx]

        ok = _remove_used_cards(counts, used_cards)
        if ok:
            return

    # 找不到合理牌局時，才走保守 fallback
    _fallback_remove_cards(counts, p_pts, b_pts)


# ---------------------------------------------------------
# Main public function
# ---------------------------------------------------------

def _get_sims_by_stage(rounds_seen: int | None) -> int:
    """
    依牌靴階段提高模擬次數。
    """

    sims = _env_int("DEPLETEMC_SIMS", 2000)

    if rounds_seen is None:
        return max(200, sims)

    try:
        r = int(rounds_seen)
    except Exception:
        return max(200, sims)

    if r > 50:
        return max(sims, _env_int("DEPLETEMC_SIMS_LATE", 3000))
    elif r > 30:
        return max(sims, _env_int("DEPLETEMC_SIMS_MID", 2500))
    else:
        return max(sims, _env_int("DEPLETEMC_SIMS_EARLY", 2000))


def _get_shrink_alpha(rounds_seen: int | None) -> float:
    """
    SAFE_PROBS 收斂權重。
    alpha 越高，越保守，越不容易出現誇張機率差距。
    alpha 越低，越相信扣牌模擬結果。

    預設：
        early: 0.30
        mid:   0.20
        late:  0.15
    """

    if rounds_seen is None:
        return _env_float("DEPLETEMC_SHRINK_ALPHA_DEFAULT", 0.20)

    try:
        r = int(rounds_seen)
    except Exception:
        return _env_float("DEPLETEMC_SHRINK_ALPHA_DEFAULT", 0.20)

    if r < 20:
        return _env_float("DEPLETEMC_SHRINK_ALPHA_EARLY", 0.30)
    elif r < 50:
        return _env_float("DEPLETEMC_SHRINK_ALPHA_MID", 0.20)
    else:
        return _env_float("DEPLETEMC_SHRINK_ALPHA_LATE", 0.15)


def _normalize_probs(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)

    if arr.shape[0] < 3:
        return SAFE_PROBS.copy()

    arr = arr[:3]
    arr = np.maximum(arr, 0.0)

    s = float(arr.sum())

    if s <= 0 or (not np.isfinite(arr).all()):
        return SAFE_PROBS.copy()

    arr = arr / s
    return arr.astype(np.float32)


def probs_after_points(
    base_counts: np.ndarray,
    p_pts: int,
    b_pts: int,
    sims: int = None,
    deplete_factor: float = None,
    rounds_seen: int | None = None,
) -> np.ndarray:
    """
    Server 主要呼叫函式。

    參數保留：
        base_counts
        p_pts
        b_pts
        sims
        deplete_factor
        rounds_seen

    回傳：
        np.ndarray([banker, player, tie], dtype=np.float32)
    """

    global GLOBAL_COUNTS

    # 定期重置牌靴
    maybe_reset_shoe(rounds_seen)

    # 初始化全域牌靴
    if GLOBAL_COUNTS is None:
        decks = _env_int("DECKS", 8)
        init_counts(decks)

    # 選擇要扣哪一份 counts
    # base_counts 如果由外部傳入，就沿用外部牌靴；
    # 否則使用 GLOBAL_COUNTS。
    if base_counts is None:
        counts = GLOBAL_COUNTS
    else:
        counts = base_counts

    if counts is None:
        counts = init_counts(_env_int("DECKS", 8))

    counts = np.asarray(counts, dtype=np.int32)

    # 根據上一局點數扣牌
    update_counts_from_points(counts, p_pts, b_pts)

    # 讓 GLOBAL_COUNTS 跟著更新
    if base_counts is None:
        GLOBAL_COUNTS = counts

    # 模擬次數
    if sims is None:
        sims = _get_sims_by_stage(rounds_seen)
    else:
        sims = max(50, int(sims))

    # 用扣牌後的 counts 模擬下一局
    result = simulate_probs(counts.copy(), sims=sims)
    result = _normalize_probs(result)

    # 收斂到長期安全分布，避免單局扣牌造成假大差距
    alpha = _get_shrink_alpha(rounds_seen)
    alpha = max(0.0, min(0.80, float(alpha)))

    result = result * (1.0 - alpha) + SAFE_PROBS * alpha
    result = _normalize_probs(result)

    return result.astype(np.float32)


# ---------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------

def _stage_bounds():
    early_end = _env_int("EARLY_HANDS", 20)
    mid_end = _env_int("MID_HANDS", 56)
    return early_end, mid_end


def _stage_prefix(rounds_seen: int | None) -> str:
    if rounds_seen is None:
        return ""

    e_end, m_end = _stage_bounds()

    try:
        r = int(rounds_seen)
    except Exception:
        return ""

    if r < e_end:
        return "EARLY_"
    elif r < m_end:
        return "MID_"
    else:
        return "LATE_"
