# bgs/deplete.py — 組成依賴蒙地卡羅（百家樂）
# 保持原 API/邏輯，強化健壯性與可調參：
# - predict() 預設模擬次數可由環境變數 DEPL_MC_SIMS 控制（預設 20000）
# - clamp tie 機率範圍可由 DEPL_TIE_MIN / DEPL_TIE_MAX 設定（預設 0.06~0.20）
# - 新增 set_seed() / get_state() / set_state() 方便復現與快照
# - 內部抽牌/條件採樣全包 try/except，避免極端鞋況崩潰
# - 嚴格保留回傳順序：p = [Banker, Player, Tie] (np.float32)

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np

# ----- 牌桶設定：0..9；0 桶代表 10/J/Q/K = 0 點 -----
CARD_IDX = list(range(10))  # 0..9
TEN_BUCKET = 0

def init_counts(decks: int = 8) -> np.ndarray:
    """建立各點數剩餘張數（10 桶=16*decks；其餘 1~9 各 4*decks）。"""
    counts = np.zeros(10, dtype=np.int32)
    counts[1:10] = 4 * decks
    counts[TEN_BUCKET] = 16 * decks
    return counts

def draw_card(counts: np.ndarray, rng: np.random.Generator) -> int:
    """依剩餘張數機率抽一張，並自 counts 扣除；若鞋空則拋錯。"""
    tot = int(counts.sum())
    if tot <= 0:
        raise RuntimeError("Shoe empty")
    r = int(rng.integers(0, tot))
    acc = 0
    # 線性走訪 10 桶，對 20000 sims 規模已足夠
    for v in range(10):
        acc += int(counts[v])
        if r < acc:
            counts[v] -= 1
            return v
    # 安全落地（理論上不會到這裡）
    v = 9
    counts[v] -= 1
    return v

def points_add(a: int, b: int) -> int:
    return (a + b) % 10

def third_card_rule_player(p_sum: int) -> bool:
    return p_sum <= 5

def third_card_rule_banker(b_sum: int, p3: Optional[int]) -> bool:
    # p3=None 表玩家未補牌；此時 p3 視作 10（= 0 點）以便條件
    if b_sum <= 2:
        return True
    if b_sum == 3:
        return (p3 is None) or (p3 != 8)
    if b_sum == 4:
        return (p3 is not None) and (p3 in (2, 3, 4, 5, 6, 7))
    if b_sum == 5:
        return (p3 is not None) and (p3 in (4, 5, 6, 7))
    if b_sum == 6:
        return (p3 is not None) and (p3 in (6, 7))
    return False

@dataclass
class DepleteMC:
    """組成依賴蒙地卡羅（依觀測消耗期望張數）"""
    decks: int = 8
    seed: int = 42

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.counts = init_counts(self.decks)

        # 可由環境變數調整的預設
        self._default_sims = int(os.getenv("DEPL_MC_SIMS", "20000"))
        self._tie_min = float(os.getenv("DEPL_TIE_MIN", "0.06"))
        self._tie_max = float(os.getenv("DEPL_TIE_MAX", "0.20"))

    # ---------- 實用輔助 ----------
    def set_seed(self, seed: int) -> None:
        """重設亂數種子（方便復現）。"""
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def get_state(self) -> Dict[str, Any]:
        """取得目前鞋況快照（可自行持久化）。"""
        return {
            "decks": int(self.decks),
            "seed": int(self.seed),
            "counts": self.counts.astype(np.int32).tolist(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """從快照還原鞋況。"""
        self.decks = int(state.get("decks", self.decks))
        self.seed = int(state.get("seed", self.seed))
        self.rng = np.random.default_rng(self.seed)
        counts = state.get("counts", None)
        if counts is not None:
            arr = np.asarray(counts, dtype=np.int32)
            if arr.shape == (10,):
                self.counts = arr.copy()
            else:
                self.counts = init_counts(self.decks)
        else:
            self.counts = init_counts(self.decks)

    # ---------- 鞋重置 ----------
    def reset_shoe(self, decks: Optional[int] = None) -> None:
        """重置鞋；decks 不傳則沿用原設定。"""
        if decks is not None:
            self.decks = int(decks)
        self.counts = init_counts(self.decks)

    # ---------- 依觀測「期望消耗」更新 ----------
    def _sample_hand_conditional(
        self,
        p_total: Optional[int] = None,
        b_total: Optional[int] = None,
        p3_drawn: Optional[bool] = None,
        b3_drawn: Optional[bool] = None,
        p3_val: Optional[int] = None,
        b3_val: Optional[int] = None,
        trials: int = 300,
    ) -> None:
        trials = max(10, int(trials))  # 最低 10 次避免除以 0
        exp_usage = np.zeros_like(self.counts, dtype=np.float64)
        success = 0

        for _ in range(trials):
            tmp = self.counts.copy()
            try:
                # 發頭兩張
                P1 = draw_card(tmp, self.rng); P2 = draw_card(tmp, self.rng)
                B1 = draw_card(tmp, self.rng); B2 = draw_card(tmp, self.rng)
                p_sum = points_add(P1, P2); b_sum = points_add(B1, B2)

                # 條件過濾（若指定 p_total / b_total）
                if p_total is not None and (p_sum != (int(p_total) % 10)):
                    continue
                if b_total is not None and (b_sum != (int(b_total) % 10)):
                    continue

                # 天生贏 8/9
                if (p_sum in (8, 9)) or (b_sum in (8, 9)):
                    pass
                else:
                    # 玩家第三張是否抽牌（由條件 p3_drawn 覆蓋，否則用規則）
                    if p3_drawn is None:
                        p3_do = third_card_rule_player(p_sum)
                    else:
                        p3_do = bool(p3_drawn)

                    P3 = None
                    if p3_do:
                        if p3_val is None:
                            P3 = draw_card(tmp, self.rng)
                        else:
                            val = int(p3_val)
                            if 0 <= val <= 9 and tmp[val] > 0:
                                tmp[val] -= 1; P3 = val
                            else:
                                P3 = draw_card(tmp, self.rng)
                        p_sum = points_add(p_sum, P3)

                    # 莊家第三張（由條件 b3_drawn 覆蓋，否則用規則）
                    if b3_drawn is None:
                        b3_do = third_card_rule_banker(b_sum, P3)
                    else:
                        b3_do = bool(b3_drawn)

                    if b3_do:
                        if b3_val is None:
                            B3 = draw_card(tmp, self.rng)
                        else:
                            val = int(b3_val)
                            if 0 <= val <= 9 and tmp[val] > 0:
                                tmp[val] -= 1; B3 = val
                            else:
                                B3 = draw_card(tmp, self.rng)
                        b_sum = points_add(b_sum, B3)

                used = self.counts - tmp
                if used.min() < 0:
                    continue  # 防呆：抽超張（理論上不會）
                exp_usage += used
                success += 1

            except Exception:
                # 任一模擬步驟出錯就丟棄此樣本，繼續下一次
                continue

        if success > 0:
            exp_usage = exp_usage / float(success)
            # 從剩餘牌數扣除期望消耗（四捨五入趨保守，且下界為 0）
            dec = np.rint(exp_usage).astype(np.int32)
            self.counts = np.maximum(0, self.counts - dec)

    def update_hand(self, obs: Dict[str, Any]) -> None:
        """依觀測（如：p_total/b_total、是否抽第三張、指定第三張點數等）更新鞋況。
        允許鍵：
          - p_total, b_total: 0..9
          - p3_drawn, b3_drawn: bool
          - p3_val, b3_val: 0..9
          - trials: int（條件採樣次數，預設 300）
        """
        self._sample_hand_conditional(
            p_total=obs.get("p_total"),
            b_total=obs.get("b_total"),
            p3_drawn=obs.get("p3_drawn"),
            b3_drawn=obs.get("b3_drawn"),
            p3_val=obs.get("p3_val"),
            b3_val=obs.get("b3_val"),
            trials=int(obs.get("trials", 300)),
        )

    # ---------- 模擬預測 ----------
    def predict(self, sims: Optional[int] = None) -> np.ndarray:
        """回傳 [Banker, Player, Tie] 機率 (np.float32)，內含 tie clamp 與歸一化。"""
        sims = int(sims if sims is not None else self._default_sims)
        sims = max(1000, sims)  # 底線以避免過度噪聲

        wins = np.zeros(3, dtype=np.int64)  # 0=B,1=P,2=T
        for _ in range(sims):
            tmp = self.counts.copy()
            try:
                P1 = draw_card(tmp, self.rng); P2 = draw_card(tmp, self.rng)
                B1 = draw_card(tmp, self.rng); B2 = draw_card(tmp, self.rng)
                p_sum = points_add(P1, P2); b_sum = points_add(B1, B2)

                if (p_sum in (8, 9)) or (b_sum in (8, 9)):
                    pass
                else:
                    if third_card_rule_player(p_sum):
                        P3 = draw_card(tmp, self.rng)
                        p_sum = points_add(p_sum, P3)
                    else:
                        P3 = None

                    if third_card_rule_banker(b_sum, P3):
                        B3 = draw_card(tmp, self.rng)
                        b_sum = points_add(b_sum, B3)

                if p_sum > b_sum:
                    wins[1] += 1  # Player
                elif b_sum > p_sum:
                    wins[0] += 1  # Banker
                else:
                    wins[2] += 1  # Tie
            except Exception:
                # 鞋不夠或抽牌失敗就丟棄此樣本
                continue

        tot = int(wins.sum())
        if tot <= 0:
            # 極端錯誤保底
            return np.array([0.45, 0.45, 0.10], dtype=np.float32)

        p = wins.astype(np.float64) / float(tot)
        # Tie 機率夾緊（避免極端估計）
        p[2] = float(np.clip(p[2], self._tie_min, self._tie_max))
        p = p / p.sum()
        return p.astype(np.float32)

__all__ = [
    "DepleteMC",
    "init_counts",
    "draw_card",
    "points_add",
    "third_card_rule_player",
    "third_card_rule_banker",
    "CARD_IDX",
    "TEN_BUCKET",
]
