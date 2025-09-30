"""蒙地卡羅耗損近似。

這份模組僅在擁有充足 CPU 資源時才建議啟用，因為完整的
``predict`` 會跑上萬局模擬。Render 免費環境預設仍使用
`bgs.pfilter` 內的輕量 OutcomePF，只保留此檔以供進階實驗者
在資源允許的情況下切換。
"""

# Author: 親愛的 x GPT-5 Thinking

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional

# ---- 牌值桶：0..9；0 桶代表 10/J/Q/K = 0點 ----
CARD_IDX = list(range(10))
TEN_BUCKET = 0

def init_counts(decks: int = 8) -> np.ndarray:
    counts = np.zeros(10, dtype=np.int32)
    counts[1:10] = 4 * decks              # 1..9 各 4*decks
    counts[TEN_BUCKET] = 16 * decks       # 10/J/Q/K 合計 16*decks
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
    # p3: 若玩家未第三張，傳入 None；等同於 10（不影響莊規則的條件分支）
    if b_sum <= 2: return True
    if b_sum == 3: return (p3 is None) or (p3 != 8)
    if b_sum == 4: return (p3 is not None) and (p3 in (2,3,4,5,6,7))
    if b_sum == 5: return (p3 is not None) and (p3 in (4,5,6,7))
    if b_sum == 6: return (p3 is not None) and (p3 in (6,7))
    return False

@dataclass
class DepleteMC:
    """牌靴耗損蒙地卡羅。可單獨用，也被 OutcomePF 包裝後供 server.py 呼叫。"""
    decks: int = 8
    seed: int = 42

    def __post_init__(self):
        self.rng = np.random.default_rng(int(self.seed))
        self.counts = init_counts(int(self.decks))

    def reset_shoe(self, decks: Optional[int] = None):
        if decks is not None:
            self.decks = int(decks)
        self.counts = init_counts(self.decks)

    # ---- 只知道勝方(outcome)時的條件耗損近似 ----
    def _sample_outcome_only(self, outcome: int, trials: int = 300):
        """
        outcome: 0=莊、1=閒、2=和
        用接受-拒絕抽樣：僅統計模擬結果勝方==outcome 的局，取用牌期望扣到牌靴。
        """
        exp_usage = np.zeros_like(self.counts, dtype=np.float64)
        success = 0
        for _ in range(trials):
            tmp = self.counts.copy()
            try:
                # 發兩張
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

                # 判定勝方
                if p_sum > b_sum: w = 1
                elif b_sum > p_sum: w = 0
                else: w = 2

                if w != outcome:
                    continue  # 只接受與觀測勝方一致的樣本

                used = self.counts - tmp
                if used.min() < 0:
                    continue
                exp_usage += used; success += 1
            except Exception:
                continue

        if success > 0:
            exp_usage = exp_usage / success
            self.counts = np.maximum(0, (self.counts - exp_usage).astype(np.int32))

    # 舊版：若掌握更細的點數/第三張資訊，可做更嚴格的條件耗損
    def _sample_hand_conditional(
        self,
        p_total: Optional[int] = None,
        b_total: Optional[int] = None,
        p3_drawn: Optional[bool] = None,
        b3_drawn: Optional[bool] = None,
        p3_val: Optional[int] = None,
        b3_val: Optional[int] = None,
        trials: int = 300,
    ):
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

                if not (p_sum in (8,9) or b_sum in (8,9)):
                    # 玩家第三張
                    if p3_drawn is None:
                        p3_do = third_card_rule_player(p_sum)
                    else:
                        p3_do = bool(p3_drawn)
                    P3 = None
                    if p3_do:
                        if (p3_val is not None) and (tmp[p3_val] > 0):
                            tmp[p3_val] -= 1; P3 = p3_val
                        else:
                            P3 = draw_card(tmp, self.rng)
                        p_sum = points_add(p_sum, P3)

                    # 莊家第三張
                    if b3_drawn is None:
                        b3_do = third_card_rule_banker(b_sum, P3)
                    else:
                        b3_do = bool(b3_drawn)
                    if b3_do:
                        if (b3_val is not None) and (tmp[b3_val] > 0):
                            tmp[b3_val] -= 1; B3 = b3_val
                        else:
                            B3 = draw_card(tmp, self.rng)
                        b_sum = points_add(b_sum, B3)

                used = self.counts - tmp
                if used.min() < 0: continue
                exp_usage += used; success += 1
            except Exception:
                continue

        if success > 0:
            exp_usage = exp_usage / success
            self.counts = np.maximum(0, (self.counts - exp_usage).astype(np.int32))

    # 你原本的 API（保留）
    def update_hand(self, obs: dict):
        self._sample_hand_conditional(
            p_total=obs.get("p_total"),
            b_total=obs.get("b_total"),
            p3_drawn=obs.get("p3_drawn"),
            b3_drawn=obs.get("b3_drawn"),
            p3_val=obs.get("p3_val"),
            b3_val=obs.get("b3_val"),
            trials=int(obs.get("trials", 300)),
        )

    # 供 OutcomePF 包裝器呼叫：只憑勝方耗損
    def update_outcome(self, outcome: int, trials: int = 300):
        if outcome not in (0, 1, 2):
            return
        self._sample_outcome_only(outcome, trials=trials)

    def predict(self, sims: int = 20000) -> np.ndarray:
        wins = np.zeros(3, dtype=np.int64)  # [B,P,T]
        for _ in range(int(sims)):
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
            return np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)  # 修正為標準百家樂理論值

        p = wins.astype(np.float64) / float(tot)
        # 綁定和局到合理區間，避免極端估計
        p[2] = float(np.clip(p[2], 0.06, 0.12))  # 縮小和局範圍
        p = p / p.sum()
        
        # 確保莊閒機率不會過度偏斜，加入平衡機制
        bp_ratio = p[0] / p[1] if p[1] > 0 else 1.0
        if bp_ratio > 1.15:  # 莊機率過高
            adjustment = (bp_ratio - 1.0) * 0.3
            p[0] -= adjustment * 0.5
            p[1] += adjustment * 0.5
        elif bp_ratio < 0.87:  # 閒機率過高
            adjustment = (1.0 - bp_ratio) * 0.3
            p[0] += adjustment * 0.5
            p[1] -= adjustment * 0.5
            
        p = p / p.sum()
        return p.astype(np.float32)

# ---- 與 server.py 相容的包裝器：OutcomePF ----
class OutcomePF:
    """
    與 server.py 的建構/方法完全對齊：
      OutcomePF(decks, seed, n_particles, sims_lik, resample_thr,
                backend='mc', dirichlet_eps=0.003, **kwargs)

      update_outcome(outcome:int)
      predict(sims_per_particle:int=0) -> np.ndarray([pB,pP,pT], float32)
      backend 屬性（供 log 顯示）
    """
    def __init__(
        self,
        decks: int = 8,
        seed: int = 42,
        n_particles: int = 200,   # 這裡不使用，但保留參數相容
        sims_lik: int = 80,       # 用於 update 時的條件抽樣次數
        resample_thr: float = 0.5, # 不使用，保留參數相容
        backend: str = "mc",
        dirichlet_eps: float = 0.003,
        **kwargs,
    ):
        self.backend = f"mc-deplete"
        self.mc = DepleteMC(decks=int(decks), seed=int(seed))
        self.sims_lik = int(sims_lik)
        # 用於平滑/穩定的 Dirichlet 先驗 + 簡單的勝負計數
        self.dirichlet_eps = float(max(1e-6, dirichlet_eps))
        self.win_counts = np.zeros(3, dtype=np.float64)  # [B,P,T]

    def update_outcome(self, outcome: int):
        # 記錄勝負（做平滑）
        if outcome in (0, 1, 2):
            self.win_counts[outcome] += 1.0
            # 用 outcome-only 的條件抽樣來近似耗損
            self.mc.update_outcome(outcome, trials=max(50, self.sims_lik))

    def predict(self, sims_per_particle: int = 0) -> np.ndarray:
        # 1) 蒙地卡羅估計（依目前靴狀態）
        mc_proba = self.mc.predict(sims=20000 if sims_per_particle <= 0 else int(sims_per_particle)*100)

        # 2) Dirichlet 平滑：用 win_counts + eps 當作外部資訊
        alpha = self.win_counts + self.dirichlet_eps
        base = alpha / alpha.sum()

        # 3) 混合，避免過度自信：MC 為主、外部Count為輔，但加入平衡機制
        out = 0.75 * mc_proba + 0.25 * base
        out = out / out.sum()
        
        # 最終平衡檢查：確保莊閒機率不會極端偏斜
        if out[0] > 0.55:  # 莊機率過高
            adjustment = (out[0] - 0.5) * 0.4
            out[0] -= adjustment
            out[1] += adjustment
        elif out[1] > 0.55:  # 閒機率過高
            adjustment = (out[1] - 0.5) * 0.4
            out[1] -= adjustment
            out[0] += adjustment
            
        out = out / out.sum()
        return out.astype(np.float32)
