# bgs/deplete.py — 組成依賴蒙地卡羅（百家樂）
import numpy as np
from dataclasses import dataclass

CARD_IDX = list(range(10))  # 0..9 ; 0 桶代表 10/J/Q/K = 0點
TEN_BUCKET = 0

def init_counts(decks=8):
    counts = np.zeros(10, dtype=np.int32)
    counts[1:10] = 4 * decks
    counts[TEN_BUCKET] = 16 * decks
    return counts

def draw_card(counts, rng):
    tot = counts.sum()
    if tot <= 0: raise RuntimeError("Shoe empty")
    r = rng.integers(0, tot)
    acc = 0
    for v in range(10):
        acc += counts[v]
        if r < acc:
            counts[v] -= 1
            return v
    v = 9; counts[v] -= 1; return v

def points_add(a, b): return (a + b) % 10
def third_card_rule_player(p_sum): return p_sum <= 5
def third_card_rule_banker(b_sum, p3):
    if b_sum <= 2: return True
    if b_sum == 3: return p3 != 8
    if b_sum == 4: return p3 in (2,3,4,5,6,7)
    if b_sum == 5: return p3 in (4,5,6,7)
    if b_sum == 6: return p3 in (6,7)
    return False

@dataclass
class DepleteMC:
    decks: int = 8
    seed: int = 42

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.counts = init_counts(self.decks)

    def reset_shoe(self, decks=None):
        if decks is not None: self.decks = decks
        self.counts = init_counts(self.decks)

    def _sample_hand_conditional(self, p_total=None, b_total=None, p3_drawn=None, b3_drawn=None, p3_val=None, b3_val=None, trials=300):
        exp_usage = np.zeros_like(self.counts, dtype=np.float64)
        success = 0
        for _ in range(trials):
            tmp = self.counts.copy()
            try:
                P1 = draw_card(tmp, self.rng); P2 = draw_card(tmp, self.rng)
                B1 = draw_card(tmp, self.rng); B2 = draw_card(tmp, self.rng)
                p_sum = points_add(P1, P2); b_sum = points_add(B1, B2)
                if p_total is not None and p_sum != (p_total % 10): continue
                if b_total is not None and b_sum != (b_total % 10): continue
                if p_sum in (8,9) or b_sum in (8,9):
                    pass
                else:
                    if p3_drawn is None: p3_do = third_card_rule_player(p_sum)
                    else: p3_do = bool(p3_drawn)
                    P3 = None
                    if p3_do:
                        if p3_val is None:
                            P3 = draw_card(tmp, self.rng)
                        else:
                            if tmp[p3_val] > 0: tmp[p3_val] -= 1; P3 = p3_val
                            else: P3 = draw_card(tmp, self.rng)
                        p_sum = points_add(p_sum, P3)
                    if b3_drawn is None:
                        b3_do = third_card_rule_banker(b_sum, P3 if P3 is not None else 10)
                    else:
                        b3_do = bool(b3_drawn)
                    if b3_do:
                        if b3_val is None:
                            B3 = draw_card(tmp, self.rng)
                        else:
                            if tmp[b3_val] > 0: tmp[b3_val] -= 1; B3 = b3_val
                            else: B3 = draw_card(tmp, self.rng)
                        b_sum = points_add(b_sum, B3)
                used = self.counts - tmp
                if used.min() < 0: continue
                exp_usage += used; success += 1
            except:
                continue
        if success > 0:
            exp_usage = exp_usage / success
            self.counts = np.maximum(0, (self.counts - exp_usage).astype(np.int32))

    def update_hand(self, obs: dict):
        self._sample_hand_conditional(
            p_total=obs.get("p_total"),
            b_total=obs.get("b_total"),
            p3_drawn=obs.get("p3_drawn"),
            b3_drawn=obs.get("b3_drawn"),
            p3_val=obs.get("p3_val"),
            b3_val=obs.get("b3_val"),
            trials=int(obs.get("trials", 300))
        )

    def predict(self, sims=20000):
        wins = np.zeros(3, dtype=np.int64)
        for _ in range(sims):
            tmp = self.counts.copy()
            try:
                P1 = draw_card(tmp, self.rng); P2 = draw_card(tmp, self.rng)
                B1 = draw_card(tmp, self.rng); B2 = draw_card(tmp, self.rng)
                p_sum = points_add(P1, P2); b_sum = points_add(B1, B2)
                if p_sum in (8,9) or b_sum in (8,9):
                    pass
                else:
                    if third_card_rule_player(p_sum):
                        P3 = draw_card(tmp, self.rng)
                        p_sum = points_add(p_sum, P3)
                    else:
                        P3 = None
                    if third_card_rule_banker(b_sum, P3 if P3 is not None else 10):
                        B3 = draw_card(tmp, self.rng)
                        b_sum = points_add(b_sum, B3)
                if p_sum > b_sum: wins[1] += 1
                elif b_sum > p_sum: wins[0] += 1
                else: wins[2] += 1
            except:
                continue
        tot = wins.sum()
        if tot == 0: return np.array([0.45,0.45,0.10], dtype=np.float32)
        p = wins / tot
        p[2] = np.clip(p[2], 0.06, 0.20)
        p = p / p.sum()
        return p.astype(np.float32)
