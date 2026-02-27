# -*- coding: utf-8 -*-
import numpy as np
from collections import deque

class HybridGRUModel:
    def __init__(self, lookback=40):
        self.history = deque(maxlen=lookback)
        # 調低boost避免過度追莊
        self.trend_threshold = 2.6
        self.chop_threshold = 0.62
        self.trend_boost = 0.065
        self.chop_boost = 0.055
        self.chaos_boost = 0.028

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

    # === 加強版階段檢測 ===
    def meta_state(self):
        if len(self.history) < 20:
            return "CHAOS"
        h = np.array(self.history)
        n = len(h)
        # CUSUM 轉折檢測
        switches = np.where(np.diff(h) != 0)[0]
        recent_sw = len([s for s in switches if s > n-18])
        switch_rate = recent_sw / 18
        # 自適應閾值
        trend_th = 2.6 + 0.015*(n//60)
        chop_th = 0.62 * (0.92 + 0.015*(n//60))
        # 近期平均run length
        diff = np.diff(h)
        runs = np.diff(np.concatenate(([0], np.where(diff != 0)[0]+1, [n])))
        avg_run = np.mean(runs[-12:]) if len(runs) > 8 else np.mean(runs)
        if avg_run >= trend_th:
            return "TREND"
        if switch_rate > chop_th:
            return "CHOP"
        return "CHAOS"

    def predict(self):
        if len(self.history) < 8:
            return np.array([0.458, 0.446, 0.096], dtype=np.float32)
        meta_probs = self._meta_predict()
        pattern_probs = self.detect_pattern()
        pf_probs = self.particle_predict()
        final = (meta_probs * 0.4 + pattern_probs * 0.35 + pf_probs * 0.25)
        if final[2] < 0.088:
            final[2] = 0.092
        final = final / final.sum()
        return final.astype(np.float32)

    def _meta_predict(self):
        h = np.array(self.history)
        total = len(h)
        pB = np.sum(h == 0) / total
        pP = np.sum(h == 1) / total
        pT = np.sum(h == 2) / total
        # streak
        streak = 1
        last = h[-1]
        for i in range(2, min(12, len(h))+1):
            if h[-i] == last:
                streak += 1
            else:
                break
        state = self.meta_state()
        if state == "TREND":
            boost = self.trend_boost * (1 + (streak-2)*0.08)
            if last == 0: pB += boost
            elif last == 1: pP += boost
        elif state == "CHOP":
            if last == 0: pP += self.chop_boost
            elif last == 1: pB += self.chop_boost
        else:
            if last == 0: pB += self.chaos_boost
            elif last == 1: pP += self.chaos_boost
        probs = np.array([pB, pP, pT])
        probs = np.clip(probs, 0.01, 0.97)
        return probs / probs.sum()

    def detect_pattern(self):
        if len(self.history) < 8:
            return np.array([0.458, 0.446, 0.096], dtype=np.float32)
        h = np.array(self.history)
        last3 = h[-3:]
        pB, pP, pT = 0.455, 0.445, 0.10
        if np.all(last3 == 0):
            pB += 0.075
        elif np.all(last3 == 1):
            pP += 0.075
        elif np.all(last3 == [0,1,0]) or np.all(last3 == [1,0,1]):
            pB, pP = pP + 0.055, pB + 0.055
        probs = np.array([pB, pP, pT])
        return probs / probs.sum()

    def particle_predict(self):
        particles = []
        for _ in range(12):
            bias = np.random.normal(0, 0.025)
            particles.append([0.46 + bias, 0.44 - bias, 0.10])
        weights = np.ones(12)
        if len(self.history) >= 6:
            recent = np.array(self.history)[-6:]
            for i, p in enumerate(particles):
                pred = np.argmax(p)
                weights[i] = np.sum(recent == pred) / 6 + 0.12
        weights /= weights.sum()
        final = np.average(particles, axis=0, weights=weights)
        return np.clip(final, 0.02, 0.96)
