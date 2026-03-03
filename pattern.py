# -*- coding: utf-8 -*-
from typing import List
import numpy as np
from collections import deque

class PatternModel:
    def __init__(self, lookback: int = 40):
        self.lookback = max(20, int(lookback))
        self.history: deque = deque(maxlen=self.lookback)

    def load_history(self, seq: List[int]):
        self.history.clear()
        for s in seq[-self.lookback:]:
            self.history.append(s)

    def update(self, outcome: int):
        self.history.append(outcome)

    def predict(self) -> np.ndarray:
        if len(self.history) < 6:
            return np.array([0.458, 0.446, 0.096], dtype=np.float32)
        
        h = np.array(self.history)
        n = len(h)
        
        # 全局分布 (平滑)
        counts = np.bincount(h, minlength=3) + 0.6
        global_p = counts / counts.sum()
        
        # 一階轉移矩陣 (針對最後一個結果)
        last1 = h[-1]
        trans1 = np.ones(3) * 0.5
        for i in range(n-1):
            if h[i] == last1:
                trans1[h[i+1]] += 1.0
        trans1 /= trans1.sum()
        
        # 連勝長度 (streak)
        streak = 1
        for i in range(2, min(9, n)+1):
            if h[-i] == last1:
                streak += 1
            else:
                break
        
        # 動態連勝強化
        streak_bonus = np.zeros(3)
        if streak >= 6:
            streak_bonus[last1] = 0.075
        elif streak >= 4:
            streak_bonus[last1] = 0.05
        elif streak == 3:
            streak_bonus[last1] = 0.03
        
        # 局部型態獎勵 (避免與長連勝疊加，並防止逆勢干擾)
        pattern_bonus = np.zeros(3)
        if n >= 3:
            last3 = h[-3:]
            # 連續三次相同（只在非長連勝時啟用）
            if streak < 4 and np.all(last3 == last1):
                pattern_bonus[last1] += 0.035
            # 型態如 B P P 或 P B B（只在非長連勝時啟用）
            if streak < 4 and (last3[0] != last3[1] == last3[2]):
                pattern_bonus[0 if last1 == 1 else 1] += 0.025
        
        # 趨勢濃度檢測 (最近10局，排除和局)
        recent_window = h[-10:]
        recent_counts = np.bincount(recent_window, minlength=3)
        valid = recent_counts[0] + recent_counts[1]  # 莊+閒的有效局數
        trend_bonus = np.zeros(3)
        if valid > 0:
            dominant = np.argmax(recent_counts[:2])  # 只看莊(0)和閒(1)
            dominance_ratio = recent_counts[dominant] / valid
            if dominance_ratio >= 0.75:
                trend_bonus[dominant] = 0.06
            elif dominance_ratio >= 0.65:
                trend_bonus[dominant] = 0.035
            # 若最後一局不是 dominant，衰減趨勢強度（避免延遲反應）
            if last1 != dominant:
                trend_bonus *= 0.6

        # ========== 單槍跳保護機制（修正版：排除和局） ==========
        single_jump_protect = False
        if valid > 0:
            # 取最近三局並過濾和局
            recent3 = recent_window[-3:]
            recent3_nontie = recent3[recent3 != 2]
            # 計算有效局中與 dominant 不同的次數
            reverse_count = np.sum(recent3_nontie != dominant)
            if (
                streak >= 5 and
                dominance_ratio >= 0.7 and
                last1 != dominant and
                reverse_count == 1
            ):
                single_jump_protect = True
        
        # 若為單槍跳，壓制 trans1 與 pattern_bonus，並動態強化 trend_bonus
        if single_jump_protect:
            trans1 = np.zeros(3)          # 忽略轉移矩陣
            pattern_bonus = np.zeros(3)   # 忽略型態獎勵
            # 動態增量：dominance_ratio 越高，增量越大
            trend_bonus[dominant] += 0.05 * dominance_ratio

        # ========== 雙跳確認反轉機制（修正版：排除和局干擾） ==========
        if valid > 0:
            # 取最近兩局並過濾和局
            recent2 = recent_window[-2:]
            recent2_nontie = recent2[recent2 != 2]
            if (
                streak <= 2 and
                len(recent2_nontie) == 2 and          # 確保兩局均為有效局
                np.all(recent2_nontie != dominant)    # 且都與趨勢方向不同
            ):
                trend_bonus *= 0.5   # 減半趨勢獎勵，開始懷疑反轉

        # 調整後的權重
        p = (
            0.36 * global_p +
            0.28 * trans1 +
            0.18 * streak_bonus +
            0.10 * pattern_bonus +
            0.08 * trend_bonus
        )
        
        # 歸一化
        p = p / p.sum()
        
        # 和局下限調整
        if p[2] < 0.07:
            p[2] = 0.07
            p[:2] = p[:2] / p[:2].sum() * 0.93
        
        # 震盪盤降溫機制：檢測近期莊閒交替率，若過高則壓低主勝率
        bp_sequence = recent_window[recent_window != 2]
        if len(bp_sequence) >= 6:
            alternations = np.sum(bp_sequence[:-1] != bp_sequence[1:])
            alt_ratio = alternations / (len(bp_sequence) - 1)
            if alt_ratio > 0.7:
                p[:2] *= 0.85
                p = p / p.sum()
        
        return np.clip(p, 0.015, 0.97).astype(np.float32)
