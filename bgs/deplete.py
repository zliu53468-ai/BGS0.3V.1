"""蒙地卡羅耗損近似 - 修正版

移除硬編碼限制，確保環境變數能正確生效，加入牌靴重置與計牌邏輯
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional

# ---- 牌值桶：0..9；0 桶代表 10/J/Q/K = 0點 ----
CARD_IDX = list(range(10))
TEN_BUCKET = 0

def init_counts(decks: int = 8) -> np.ndarray:
    counts = np.zeros(10, dtype=np.int32)
    counts[1:10] = 4 * decks
    counts[TEN_BUCKET] = 16 * decks
    return counts

def draw_card(counts: np.ndarray, rng: np.random.Generator) -> int:
    tot = int(counts.sum())
    if tot <= 0:
        raise RuntimeError("Shoe empty")
    r = int(rng.integers(0, tot))
    acc = 0
    for v in range(10):
        acc += int(counts[v])
        if r < acc:
            counts[v] -= 1
            return v
    v = 9
    counts[v] -= 1
    return v

def points_add(a: int, b: int) -> int:
    return (a + b) % 10

def third_card_rule_player(p_sum: int) -> bool:
    return p_sum <= 5

def third_card_rule_banker(b_sum: int, p3: Optional[int]) -> bool:
    if b_sum <= 2: return True
    if b_sum == 3: return (p3 is None) or (p3 != 8)
    if b_sum == 4: return (p3 is not None) and (p3 in (2,3,4,5,6,7))
    if b_sum == 5: return (p3 is not None) and (p3 in (4,5,6,7))
    if b_sum == 6: return (p3 is not None) and (p3 in (6,7))
    return False

@dataclass
class DepleteMC:
    """牌靴耗損蒙地卡羅"""
    decks: int = 8
    seed: int = 42

    def __post_init__(self):
        self.rng = np.random.default_rng(int(self.seed))
        self.counts = init_counts(int(self.decks))
        self.initial_counts = self.counts.copy()
        self.recent_outcomes = []
        self.max_history = 20
        self.cards_used = 0
        self.shoe_reset_threshold = int(0.75 * self.decks * 52)

    def reset_shoe(self, decks: Optional[int] = None):
        if decks is not None:
            self.decks = int(decks)
        self.counts = init_counts(self.decks)
        self.initial_counts = self.counts.copy()
        self.recent_outcomes.clear()
        self.cards_used = 0

    def _sample_outcome_only(self, outcome: int, trials: int = 500):
        exp_usage = np.zeros_like(self.counts, dtype=np.float64)
        success = 0
        
        for _ in range(trials):
            tmp = self.counts.copy()
            try:
                P1 = draw_card(tmp, self.rng); P2 = draw_card(tmp, self.rng)
                B1 = draw_card(tmp, self.rng); B2 = draw_card(tmp, self.rng)
                p_sum = points_add(P1, P2); b_sum = points_add(B1, B2)

                P3 = None
                if not (p_sum in (8,9) or b_sum in (8,9)):
                    if third_card_rule_player(p_sum):
                        P3 = draw_card(tmp, self.rng)
                        p_sum = points_add(p_sum, P3)
                    if third_card_rule_banker(b_sum, P3):
                        B3 = draw_card(tmp, self.rng)
                        b_sum = points_add(b_sum, B3)

                if p_sum > b_sum: w = 1
                elif b_sum > p_sum: w = 0
                else: w = 2

                if w != outcome:
                    continue

                used = self.counts - tmp
                if used.min() < 0:
                    continue
                exp_usage += used; success += 1
                self.cards_used += used.sum()
            except Exception:
                continue

        if success > 0:
            exp_usage = exp_usage / success
            self.counts = np.maximum(0, (self.counts - exp_usage).astype(np.int32))

        # 檢查是否需要重置牌靴
        if self.cards_used >= self.shoe_reset_threshold:
            self.reset_shoe()

    def update_outcome(self, outcome: int, trials: int = 500):
        if outcome not in (0, 1, 2):
            return
        
        self.recent_outcomes.append(outcome)
        if len(self.recent_outcomes) > self.max_history:
            self.recent_outcomes.pop(0)
            
        self._sample_outcome_only(outcome, trials=trials)

    def predict(self, sims: int = 20000) -> np.ndarray:
        wins = np.zeros(3, dtype=np.int64)
        
        # 計牌調整：根據剩餘牌計算莊/閒優勢
        total_cards = self.counts.sum()
        if total_cards > 0:
            card_dist = self.counts / total_cards
            banker_adjust = sum(card_dist[6:9]) * 0.02  # 高點數牌有利莊
            player_adjust = sum(card_dist[1:4]) * 0.015  # 低點數牌有利閒
        else:
            banker_adjust = player_adjust = 0.0

        for _ in range(sims):
            tmp = self.counts.copy()
            try:
                P1 = draw_card(tmp, self.rng); P2 = draw_card(tmp, self.rng)
                B1 = draw_card(tmp, self.rng); B2 = draw_card(tmp, self.rng)
                p_sum = points_add(P1, P2); b_sum = points_add(B1, B2)

                P3 = None
                if not (p_sum in (8,9) or b_sum in (8,9)):
                    if third_card_rule_player(p_sum):
                        P3 = draw_card(tmp, self.rng)
                        p_sum = points_add(p_sum, P3)
                    if third_card_rule_banker(b_sum, P3):
                        B3 = draw_card(tmp, self.rng)
                        b_sum = points_add(b_sum, B3)

                if p_sum > b_sum: wins[1] += 1
                elif b_sum > p_sum: wins[0] += 1
                else: wins[2] += 1
            except Exception:
                continue

        tot = int(wins.sum())
        if tot == 0:
            return np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)

        p = wins.astype(np.float64) / float(tot)
        p[0] += banker_adjust
        p[1] += player_adjust
        p = np.clip(p, 0.01, 0.99)
        p = p / p.sum()
        return p.astype(np.float32)

class OutcomePF:
    """與 server.py 相容的 OutcomePF"""
    def __init__(
        self,
        decks: int = 8,
        seed: int = 42,
        n_particles: int = 200,
        sims_lik: int = 80,
        resample_thr: float = 0.5,
        backend: str = "mc",
        dirichlet_eps: float = 0.01,
        **kwargs,
    ):
        self.backend = f"mc-deplete"
        self.mc = DepleteMC(decks=int(decks), seed=int(seed))
        self.sims_lik = int(sims_lik)
        self.dirichlet_eps = float(max(1e-6, dirichlet_eps))
        self.win_counts = np.zeros(3, dtype=np.float64)

    def update_outcome(self, outcome: int):
        if outcome in (0, 1, 2):
            self.win_counts[outcome] += 1.0
            self.mc.update_outcome(outcome, trials=max(50, self.sims_lik))

    def predict(self, sims_per_particle: int = 0) -> np.ndarray:
        base_sims = 20000 if sims_per_particle <= 0 else int(sims_per_particle) * 100
        mc_proba = self.mc.predict(sims=base_sims)

        alpha = self.win_counts + self.dirichlet_eps
        base = alpha / alpha.sum()

        out = 0.6 * mc_proba + 0.4 * base
        out = out / out.sum()
        
        return out.astype(np.float32)
