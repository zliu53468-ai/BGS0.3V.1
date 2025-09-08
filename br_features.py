# -*- coding: utf-8 -*-
"""
Shared Big-Road (6x20) & stats utilities for B/P/T sequences.
De-dup source of truth used by server/train scripts.
"""

import os
from typing import Any, Dict, List, Tuple

CLASS_ORDER = ("B", "P", "T")

def recent_freq(seq: List[str], win: int) -> List[float]:
    """Smoothed frequency over last win hands."""
    if not seq: return [1/3, 1/3, 1/3]
    cut = seq[-win:] if win > 0 else seq
    a = float(os.getenv("LAPLACE", "0.5"))
    nB = cut.count("B") + a
    nP = cut.count("P") + a
    nT = cut.count("T") + a
    tot = max(1, len(cut)) + 3*a
    return [nB/tot, nP/tot, nT/tot]

def conditional_markov_next_prob(seq: List[str], decay: float = None) -> List[float]:
    """
    True conditional Markov:
    P(X_{t+1}=· | X_t=last) using EW counts (decay toward past).
    """
    if not seq or len(seq) < 2:
        return [1/3, 1/3, 1/3]
    if decay is None:
        decay = float(os.getenv("MKV_DECAY", "0.98"))

    idx = {"B":0, "P":1, "T":2}
    T = [[0.0]*3 for _ in range(3)]  # rows: from, cols: to

    w = 1.0
    for a, b in zip(seq[:-1], seq[1:]):
        if a in idx and b in idx:
            T[idx[a]][idx[b]] += w
        w *= decay

    # row-normalize only *last* row
    last = seq[-1]
    row = T[idx[last]]
    a = float(os.getenv("MKV_LAPLACE", "0.5"))
    row = [x + a for x in row]
    s = sum(row) if sum(row) > 1e-12 else 1.0
    row = [x/s for x in row]
    return row

# ---------- Big-Road (6x20) ----------
def _last_run_len(s: List[str]) -> int:
    if not s: return 0
    ch = s[-1]; i = len(s)-2; n = 1
    while i >= 0 and s[i] == ch:
        n += 1; i -= 1
    return n

def _features_like_early_dragon(seq: List[str]) -> bool:
    k = min(6, len(seq))
    if k < 4: return False
    tail = seq[-k:]
    most = max(tail.count("B"), tail.count("P"))
    return (most >= k-1)

def map_to_big_road(seq: List[str], rows: int = 6, cols: int = 20) -> Tuple[List[List[str]], Dict[str, Any]]:
    """
    Simplified Big-Road (6x20):
    - Same result: try go down; if bottom or below occupied, move right (stay same row).
    - Different result: move right, place from row 0.
    Returns (grid, features)
    """
    grid = [["" for _ in range(cols)] for _ in range(rows)]
    if not seq:
        return grid, {"cur_run":0, "col_depth":0, "blocked":False, "r":0, "c":0, "early_dragon_hint":False}

    r = c = 0
    last = None
    for ch in seq:
        if last is None:
            grid[r][c] = ch; last = ch; continue
        if ch == last:
            if r+1 < rows and grid[r+1][c] == "":
                r += 1
            else:
                c = min(cols-1, c+1)
                while c < cols and grid[r][c] != "":
                    c = min(cols-1, c+1)
                if c >= cols: c = cols-1
        else:
            last = ch
            c = min(cols-1, c+1)
            r = 0
            while c < cols and grid[r][c] != "":
                c = min(cols-1, c+1)
            if c >= cols: c = cols-1

        if grid[r][c] == "":
            grid[r][c] = ch

    # column depth at c
    cur_depth = 0
    for rr in range(rows):
        if grid[rr][c] != "": cur_depth = rr+1

    blocked = (cur_depth >= rows) or (r == rows-1) or (r+1 < rows and grid[r+1][c] != "" and last == grid[r][c])
    feats = {
        "cur_run": _last_run_len(seq),
        "col_depth": cur_depth,
        "blocked": blocked,
        "r": r, "c": c,
        "early_dragon_hint": (cur_depth >= 3 and _features_like_early_dragon(seq))
    }
    return grid, feats

# ---------- Run/hazard utils ----------
def bp_only(seq: List[str]) -> List[str]:
    return [x for x in seq if x in ("B", "P")]

def run_hist(seq_bp: List[str]) -> Dict[int, int]:
    hist: Dict[int,int] = {}
    if not seq_bp: return hist
    cur = 1
    for i in range(1, len(seq_bp)):
        if seq_bp[i] == seq_bp[i-1]:
            cur += 1
        else:
            hist[cur] = hist.get(cur, 0) + 1
            cur = 1
    hist[cur] = hist.get(cur, 0) + 1
    return hist

def hazard_from_hist(L: int, hist: Dict[int,int], alpha: float = 0.5) -> float:
    """P(end exactly at L | length >= L) ≈ count(L) / sum_{k>=L} count(k) with smoothing."""
    if L <= 0: return 0.0
    ge = sum(v for k, v in hist.items() if k >= L)
    end = hist.get(L, 0)
    return (end + alpha) / (ge + alpha * max(1, len(hist)))

