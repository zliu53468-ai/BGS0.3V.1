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
        counts = np.bincount(h, minlength=3) + 0.6
        global_p = counts / counts.sum()
        last1 = h[-1]
        trans1 = np.ones(3) * 0.5
        for i in range(n-1):
            if h[i] == last1:
                trans1[h[i+1]] += 1.0
        trans1 /= trans1.sum()
        streak = 1
        for i in range(2, min(9, n)+1):
            if h[-i] == last1:
                streak += 1
            else:
                break
        streak_bonus = np.zeros(3)
        if streak >= 4:
            streak_bonus[last1] = 0.04
        elif streak == 3:
            streak_bonus[last1] = 0.022
        pattern_bonus = np.zeros(3)
        if n >= 3:
            last3 = h[-3:]
            if np.all(last3 == last1):
                pattern_bonus[last1] += 0.042
            if (last3[0] != last3[1] == last3[2]):
                pattern_bonus[0 if last1 == 1 else 1] += 0.028
        p = (
            0.48 * global_p +
            0.32 * trans1 +
            0.12 * streak_bonus +
            0.08 * pattern_bonus
        )
        p = p / p.sum()
        if p[2] < 0.088:
            p[2] = 0.092
            p[:2] /= p[:2].sum() * (1 - 0.092)
        return np.clip(p, 0.015, 0.97).astype(np.float32)
