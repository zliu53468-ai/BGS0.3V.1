# -*- coding: utf-8 -*-
"""
Lightweight Pattern Model
不使用深度學習，只做序列轉移統計
"""

from typing import List
import numpy as np


class PatternModel:
    def __init__(self, lookback: int = 12):
        self.lookback = max(4, int(lookback))
        self.history: List[int] = []

    def load_history(self, seq: List[int]):
        """seq: 0=莊, 1=閒, 2=和"""
        self.history = seq[-self.lookback :]

    def update(self, outcome: int):
        self.history.append(outcome)
        if len(self.history) > self.lookback:
            self.history.pop(0)

    def predict(self) -> np.ndarray:
        """
        回傳 [pB, pP, pT]
        """
        if len(self.history) < 2:
            return np.array([0.46, 0.44, 0.10], dtype=np.float32)

        last = self.history[-1]

        # 統計 last → next
        counts = [1.0, 1.0, 0.5]  # 平滑

        for i in range(len(self.history) - 1):
            if self.history[i] == last:
                nxt = self.history[i + 1]
                counts[nxt] += 1.0

        arr = np.array(counts, dtype=np.float32)
        arr /= arr.sum()
        return arr
