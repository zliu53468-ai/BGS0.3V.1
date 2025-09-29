# -*- coding: utf-8 -*-
# bgs/deplete.py — 組成依賴蒙地卡羅（百家樂）
# 平衡預測版本：與 server.py 和 pfilter.py 對齊
import os
import numpy as np
from dataclasses import dataclass

# ---- 參數（與平衡預測版本對齊） ----
DECKS_DEFAULT     = int(os.getenv("DECKS", "6"))              # 與 server/pfilter 對齊
SEED_DEFAULT      = int(os.getenv("SEED",  "42"))
DEPL_SIMS_DEFAULT = int(os.getenv("DEPLETION_SIMS", "15000")) # 減少模擬次數提高響應速度
DIR_EPS           = float(os.getenv("DEPL_DIR_EPS", "1e-6"))

# 與平衡預測版本對齊的和局範圍
TIE_MIN           = float(os.getenv("TIE_MIN", "0.04"))       
TIE_MAX           = float(os.getenv("TIE_MAX", "0.20"))

# 輕度先驗混合（與平衡版本對齊）
CALIB_MIX         = float(os.getenv("DEPL_CALIB_MIX", "0.08"))

# 對稱先驗（與 pfilter 對齊）
PRIOR_B           = float(os.getenv("PRIOR_B", "0.458"))
PRIOR_P           = float(os.getenv("PRIOR_P", "0.446"))
PRIOR_T           = float(os.getenv("PRIOR_T", "0.096"))

CARD_IDX = list(range(10))  # 0..9 ; 0 桶代表 10/J/Q/K = 0點
TEN_BUCKET = 0

def init_counts(decks=6):
    """初始化牌組計數"""
    counts = np.zeros(10, dtype=np.int32)
    counts[1:10] = 4 * decks
    counts[TEN_BUCKET] = 16 * decks
    return counts

def draw_card(counts, rng):
    """抽牌函數"""
    tot = counts.sum()
    if tot <= 0:
        raise RuntimeError("Shoe empty")
    r = rng.integers(0, tot)
    acc = 0
    for v in range(10):
        acc += counts[v]
        if r < acc:
            counts[v] -= 1
            return v
    # 理論到不了這行；防呆
    v = 9
    counts[v] -= 1
    return v

def points_add(a, b): 
    """點數相加（取個位數）"""
    return (a + b) % 10

def third_card_rule_player(p_sum): 
    """閒家第三張牌規則"""
    return p_sum <= 5

def third_card_rule_banker(b_sum, p3):
    """莊家第三張牌規則"""
    if b_sum <= 2: return True
    if b_sum == 3: return p3 != 8
    if b_sum == 4: return p3 in (2,3,4,5,6,7)
    if b_sum == 5: return p3 in (4,5,6,7)
    if b_sum == 6: return p3 in (6,7)
    return False

def _balanced_calibration(prob: np.ndarray) -> np.ndarray:
    """
    平衡校準：
    - 正規化概率
    - 夾住和局區間到 [TIE_MIN, TIE_MAX]
    - 輕度先驗混合，避免極端偏移
    - 保持概率合理性
    回傳：np.float32, sum=1
    """
    p = np.array(prob, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if s <= 0:
        # 退回平衡先驗
        p = np.array([PRIOR_B, PRIOR_P, PRIOR_T], dtype=np.float64)
        s = p.sum()
    p /= s

    # 夾 tie 機率（與平衡版本對齊）
    p[2] = np.clip(p[2], TIE_MIN, TIE_MAX)
    
    # 重新正規化
    p = np.clip(p, DIR_EPS, None)
    p /= p.sum()

    # 輕度先驗混合（避免極端）
    if CALIB_MIX > 0.0:
        prior = np.array([PRIOR_B, PRIOR_P, PRIOR_T], dtype=np.float64)
        prior /= prior.sum()
        p = (1.0 - CALIB_MIX) * p + CALIB_MIX * prior
        p = np.clip(p, DIR_EPS, None)
        p /= p.sum()

    # 確保機率差異性（避免過於接近）
    min_diff = 0.015
    if abs(p[0] - p[1]) < min_diff:
        # 輕微放大差異
        diff = p[0] - p[1]
        if abs(diff) > 0:
            scale = min(1.3, 1.0 + min_diff / abs(diff))
            if diff > 0:
                p[0] = (p[0] + p[1] * (scale - 1)) / scale
                p[1] = p[1] / scale
            else:
                p[1] = (p[1] + p[0] * (scale - 1)) / scale
                p[0] = p[0] / scale
            p = np.clip(p, 0.01, 0.98)
            p /= p.sum()

    return p.astype(np.float32)

@dataclass
class DepleteMC:
    """平衡蒙地卡羅預測器"""
    decks: int = DECKS_DEFAULT
    seed: int = SEED_DEFAULT

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.counts = init_counts(self.decks)
        self.hand_count = 0

    def reset_shoe(self, decks=None):
        """重置牌靴"""
        if decks is not None: 
            self.decks = int(decks)
        self.counts = init_counts(self.decks)
        self.hand_count = 0

    def _sample_hand_conditional(
        self,
        p_total=None, b_total=None,
        p3_drawn=None, b3_drawn=None,
        p3_val=None, b3_val=None,
        trials=200  # 減少試驗次數提高響應
    ):
        """
        以觀測條件近似估計「平均牌耗」，將期望耗牌量扣回 self.counts。
        平衡版本：使用較少試驗，提高響應速度。
        """
        exp_usage = np.zeros_like(self.counts, dtype=np.float64)
        success = 0
        
        for _ in range(int(trials)):
            tmp = self.counts.copy()
            try:
                P1 = draw_card(tmp, self.rng)
                P2 = draw_card(tmp, self.rng)
                B1 = draw_card(tmp, self.rng)
                B2 = draw_card(tmp, self.rng)
                p_sum = points_add(P1, P2)
                b_sum = points_add(B1, B2)

                # 條件匹配
                if p_total is not None and p_sum != (p_total % 10): 
                    continue
                if b_total is not None and b_sum != (b_total % 10): 
                    continue

                # 自然 8/9
                if p_sum in (8,9) or b_sum in (8,9):
                    pass
                else:
                    # 閒第三張
                    if p3_drawn is None:
                        p3_do = third_card_rule_player(p_sum)
                    else:
                        p3_do = bool(p3_drawn)
                    P3 = None
                    if p3_do:
                        if p3_val is None:
                            P3 = draw_card(tmp, self.rng)
                        else:
                            if tmp[p3_val] > 0:
                                tmp[p3_val] -= 1
                                P3 = p3_val
                            else:
                                P3 = draw_card(tmp, self.rng)
                        p_sum = points_add(p_sum, P3)
                    
                    # 莊第三張
                    if b3_drawn is None:
                        b3_do = third_card_rule_banker(b_sum, P3 if P3 is not None else 10)
                    else:
                        b3_do = bool(b3_drawn)
                    if b3_do:
                        if b3_val is None:
                            B3 = draw_card(tmp, self.rng)
                        else:
                            if tmp[b3_val] > 0:
                                tmp[b3_val] -= 1
                                B3 = b3_val
                            else:
                                B3 = draw_card(tmp, self.rng)
                        b_sum = points_add(b_sum, B3)

                used = self.counts - tmp
                if used.min() < 0:  # 防呆
                    continue
                exp_usage += used
                success += 1
            except Exception:
                continue

        if success > 0:
            exp_usage = exp_usage / success
            # 用期望耗牌量扣回（仍保留整數庫存）
            self.counts = np.maximum(0, (self.counts - exp_usage).astype(np.int32))
            
        # 更新手數計數
        self.hand_count += 1
        
        # 每52手自動重置牌靴（約1副牌）
        if self.hand_count >= 52:
            self.reset_shoe()

    def update_hand(self, obs: dict):
        """更新手牌資訊"""
        # 簡化更新邏輯，提高響應性
        trials = int(obs.get("trials", 150))  # 進一步減少試驗次數
        
        self._sample_hand_conditional(
            p_total=obs.get("p_total"),
            b_total=obs.get("b_total"),
            p3_drawn=obs.get("p3_drawn"),
            b3_drawn=obs.get("b3_drawn"),
            p3_val=obs.get("p3_val"),
            b3_val=obs.get("b3_val"),
            trials=trials
        )

    def update_from_points(self, p_pts: int, b_pts: int):
        """從點數更新（簡化接口）"""
        if p_pts == 0 and b_pts == 0:  # 和局
            obs = {
                "p_total": 0,
                "b_total": 0, 
                "p3_drawn": False,
                "b3_drawn": False,
                "trials": 100  # 和局時使用更少試驗
            }
        else:
            # 簡化點數分析
            p_total = p_pts
            b_total = b_pts
            
            # 判斷是否有第三張牌（簡化邏輯）
            p3_drawn = (p_pts > 7)  # 簡化判斷
            b3_drawn = (b_pts > 7)  # 簡化判斷
            
            obs = {
                "p_total": p_total,
                "b_total": b_total,
                "p3_drawn": p3_drawn,
                "b3_drawn": b3_drawn,
                "trials": 150
            }
        
        self.update_hand(obs)

    def predict(self, sims: int = None):
        """
        回傳順序 [B, P, T] 機率；平衡校準；型別 float32。
        使用較少模擬次數提高響應速度。
        """
        if sims is None:
            sims = DEPL_SIMS_DEFAULT

        wins = np.zeros(3, dtype=np.int64)
        valid_sims = 0
        
        for _ in range(int(sims)):
            tmp = self.counts.copy()
            try:
                P1 = draw_card(tmp, self.rng)
                P2 = draw_card(tmp, self.rng)
                B1 = draw_card(tmp, self.rng)
                B2 = draw_card(tmp, self.rng)
                p_sum = points_add(P1, P2)
                b_sum = points_add(B1, B2)

                # 自然 8/9
                if p_sum in (8,9) or b_sum in (8,9):
                    pass
                else:
                    # 閒第三張
                    if third_card_rule_player(p_sum):
                        P3 = draw_card(tmp, self.rng)
                        p_sum = points_add(p_sum, P3)
                    else:
                        P3 = None
                    # 莊第三張
                    if third_card_rule_banker(b_sum, (P3 if P3 is not None else 10)):
                        B3 = draw_card(tmp, self.rng)
                        b_sum = points_add(b_sum, B3)

                if p_sum > b_sum: 
                    wins[1] += 1  # 閒贏
                elif b_sum > p_sum: 
                    wins[0] += 1  # 莊贏
                else: 
                    wins[2] += 1  # 和局
                    
                valid_sims += 1
            except Exception:
                continue

        # 確保有有效模擬
        if valid_sims <= 0:
            # 回退：用平衡先驗
            base = np.array([PRIOR_B, PRIOR_P, PRIOR_T], dtype=np.float64)
            base = base / base.sum()
            return base.astype(np.float32)

        # 使用有效模擬次數計算概率
        p = wins.astype(np.float64) / float(valid_sims)
        p = _balanced_calibration(p)
        return p.astype(np.float32)

    def get_remaining_cards(self):
        """獲取剩餘牌數（用於調試）"""
        return self.counts.sum()

    def get_shoe_penetration(self):
        """獲取牌靴穿透率"""
        total_cards = 52 * self.decks
        remaining = self.counts.sum()
        return (total_cards - remaining) / total_cards
