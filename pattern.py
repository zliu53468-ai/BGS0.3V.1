# -*- coding: utf-8 -*-
"""
Advanced Pattern Model v2.0
大幅精修版：多階轉移 + Streak + 百家樂Pattern加成 + 動態融合
專為降低連輸、提升玩家體驗設計
"""

from typing import List
import numpy as np
from collections import deque, defaultdict

class PatternModel:
    def __init__(self, lookback: int = 40):
        self.lookback = max(20, int(lookback))   # 加大記憶長度
        self.history: deque = deque(maxlen=self.lookback)

    def load_history(self, seq: List[int]):
        """seq: 0=莊, 1=閒, 2=和"""
        self.history.clear()
        for s in seq[-self.lookback:]:
            self.history.append(s)

    def update(self, outcome: int):
        self.history.append(outcome)

    def predict(self) -> np.ndarray:
        """
        回傳 [pB, pP, pT]
        """
        if len(self.history) < 4:
            return np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)

        h = np.array(self.history)
        n = len(h)

        # 1. 整體頻率（長期基準）
        counts = np.bincount(h, minlength=3) + 0.5  # Laplace平滑
        global_p = counts / counts.sum()

        # 2. 最後1手轉移
        last1 = h[-1]
        trans1 = np.ones(3) * 0.5
        for i in range(n-1):
            if h[i] == last1:
                trans1[h[i+1]] += 1.0
        trans1 /= trans1.sum()

        # 3. 最後2手轉移（更高階）
        last2 = tuple(h[-2:]) if n >= 2 else (last1, last1)
        trans2 = np.ones(3) * 0.5
        for i in range(n-2):
            if tuple(h[i:i+2]) == last2:
                trans2[h[i+2]] += 1.0
        trans2 /= trans2.sum()

        # 4. Streak (連勝) 動態加權
        streak = 1
        for i in range(2, min(10, n)+1):
            if h[-i] == last1:
                streak += 1
            else:
                break
        streak_bonus = np.zeros(3)
        if streak >= 3:
            streak_bonus[last1] = 0.12 * (streak - 2) * 0.3   # 連越長越敢追
        elif streak == 2:
            streak_bonus[last1] = 0.06

        # 5. 常見Pattern加成
        pattern_bonus = np.zeros(3)
        if n >= 3:
            last3 = h[-3:]
            if np.all(last3 == last1):           # BBB / PPP
                pattern_bonus[last1] += 0.11
            if (last3[0] != last3[1] == last3[2]):  # 砍後追（BPB, PBP）
                pattern_bonus[0 if last1 == 1 else 1] += 0.08

        # 6. 最終融合（可調權重）
        p = (
            0.35 * global_p +
            0.30 * trans1 +
            0.20 * trans2 +
            0.10 * streak_bonus +
            0.05 * pattern_bonus
        )
        p = p / p.sum()

        # Tie 長遠回歸修正
        if p[2] < 0.085:
            p[2] = 0.092
            p[:2] /= p[:2].sum() * (1 - 0.092)   # 重新歸一

        return np.clip(p, 0.01, 0.98).astype(np.float32)
