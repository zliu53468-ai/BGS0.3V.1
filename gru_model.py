# -*- coding: utf-8 -*-
import numpy as np
from collections import deque

class HybridGRUModel:
    def __init__(self, lookback=40):
        self.history = deque(maxlen=lookback)
        # ============== 可調參數（重點實驗區） ==============
        self.trend_threshold = 2.45
        self.chop_threshold = 0.64
        self.trend_boost = 0.105
        self.chop_boost = 0.085
        self.chaos_boost = 0.045

    def load_from_string(self, history_str: str):
        mapping = {'B': 0, 'P': 1, 'T': 2, 'b': 0, 'p': 1, 't': 2}
        seq = [mapping[c] for c in history_str if c.upper() in mapping]
        self.load_history(seq)

    def load_history(self, seq):
        self.history.clear()
        for s in seq[-self.history.maxlen:]:
            self.history.append(s)

    def update(self, outcome):
        if isinstance(outcome, str):
            mapping = {'B':0, 'P':1, 'T':2, 'b':0, 'p':1, 't':2}
            outcome = mapping.get(outcome.upper(), outcome)
        self.history.append(outcome)

    def undo(self):
        if len(self.history) > 0:
            self.history.pop()

    def meta_state(self):
        if len(self.history) < 12:
            return "CHAOS"
        h = list(self.history)
        switches = sum(1 for i in range(1, len(h)) if h[i] != h[i-1])
        switch_rate = switches / len(h)
        runs = []
        cur = 1
        for i in range(1, len(h)):
            if h[i] == h[i-1]:
                cur += 1
            else:
                runs.append(cur)
                cur = 1
        runs.append(cur)
        avg_run = np.mean(runs)
        if avg_run >= self.trend_threshold:
            return "TREND"
        if switch_rate > self.chop_threshold:
            return "CHOP"
        return "CHAOS"

    def predict(self):
        if len(self.history) < 8:
            return np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)

        # 三模型 ensemble
        meta_probs = self._meta_predict()
        pattern_probs = self.detect_pattern()
        pf_probs = self.particle_predict()

        final = (meta_probs + pattern_probs + pf_probs) / 3.0
        if final[2] < 0.085:
            final[2] = 0.092
        final = final / final.sum()
        return final.astype(np.float32)

    # 內部方法（保持乾淨）
    def _meta_predict(self):
        h = np.array(self.history)
        total = len(h)
        pB = np.sum(h == 0) / total
        pP = np.sum(h == 1) / total
        pT = np.sum(h == 2) / total

        streak = 1
        last = h[-1]
        for i in range(2, min(12, len(h))+1):
            if h[-i] == last:
                streak += 1
            else:
                break

        state = self.meta_state()
        if state == "TREND":
            boost = self.trend_boost * (1 + (streak-2)*0.12)
            if last == 0: pB += boost
            elif last == 1: pP += boost
            else: pB += boost*0.6
        elif state == "CHOP":
            if last == 0: pP += self.chop_boost
            elif last == 1: pB += self.chop_boost
            else: pT += self.chop_boost*0.8
        else:
            if last == 0: pB += self.chaos_boost
            elif last == 1: pP += self.chaos_boost*0.9

        probs = np.array([pB, pP, pT])
        probs = np.clip(probs, 0.01, 0.98)
        return probs / probs.sum()

    def detect_pattern(self):
        if len(self.history) < 8:
            return np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)
        h = np.array(self.history)
        last3 = h[-3:]
        pB, pP, pT = 0.45, 0.44, 0.11
        if np.all(last3 == 0):
            pB += 0.12
        elif np.all(last3 == 1):
            pP += 0.12
        elif np.all(last3 == [0,1,0]) or np.all(last3 == [1,0,1]):
            pB, pP = pP + 0.08, pB + 0.08
        elif len(h) >= 5 and np.sum(h[-5:] == h[-1]) >= 4:
            if h[-1] == 0:
                pB += 0.09
            elif h[-1] == 1:
                pP += 0.09
        probs = np.array([pB, pP, pT])
        return probs / probs.sum()

    def particle_predict(self):
        particles = []
        for _ in range(10):
            bias = np.random.normal(0, 0.03)
            particles.append([0.46 + bias, 0.44 - bias, 0.10])
        weights = np.ones(10)
        if len(self.history) >= 5:
            recent = np.array(self.history)[-5:]
            for i, p in enumerate(particles):
                pred = np.argmax(p)
                weights[i] = np.sum(recent == pred) / 5 + 0.1
        weights /= weights.sum()
        final = np.average(particles, axis=0, weights=weights)
        return np.clip(final, 0.01, 0.98)
