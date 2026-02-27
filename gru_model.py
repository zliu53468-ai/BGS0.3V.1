# -*- coding: utf-8 -*-
import numpy as np
from collections import deque

class HybridGRUModel:
    def __init__(self, lookback=40):
        self.history = deque(maxlen=lookback)
        self.trend_threshold = 2.85
        self.chop_threshold = 0.68
        self.trend_boost = 0.032
        self.chop_boost = 0.072
        self.chaos_boost = 0.018

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
        if len(self.history) < 20:
            return "CHAOS"
        h = np.array(self.history)
        n = len(h)
        switches = np.where(np.diff(h) != 0)[0]
        recent_sw = len([s for s in switches if s > n-20])
        switch_rate = recent_sw / 20
        runs = np.diff(np.concatenate(([0], np.where(np.diff(h) != 0)[0]+1, [n])))
        avg_run = np.mean(runs[-10:]) if len(runs)>8 else np.mean(runs)
        if avg_run >= 2.85:
            return "TREND"
        if switch_rate > 0.68:
            return "CHOP"
        return "CHAOS"

    def predict(self):
        if len(self.history) < 8:
            return np.array([0.458, 0.446, 0.096], dtype=np.float32)
        meta_probs = self._meta_predict()
        pattern_probs = self.detect_pattern()
        pf_probs = self.particle_predict()

        # PF防守觸發：當最近5局出現4局相同，放大PF權重（倍率1.4，且對Tie略為壓抑）
        if len(self.history) >= 5:
            recent = list(self.history)[-5:]
            if recent.count(recent[-1]) >= 4:
                pf_probs = pf_probs * 1.4
                pf_probs[2] *= 0.9          # 不放大Tie
                pf_probs = pf_probs / pf_probs.sum()

        # 判斷是否為真正的尾盤混沌狀態（需同時滿足高切換率）
        state = self.meta_state()
        tail_mode = False
        if state == "CHAOS" and len(self.history) >= 30:
            # 檢查最近12局的切換率是否高於0.60（原0.55，優化後提高至0.60）
            recent_arr = np.array(self.history)[-12:]
            switches = np.sum(np.diff(recent_arr) != 0)
            switch_rate = switches / 11
            if switch_rate > 0.60:
                tail_mode = True

        if tail_mode:
            # 尾盤模式：PF權重50%，其餘各25%
            meta_weight = 0.25
            pattern_weight = 0.25
            pf_weight = 0.50
        else:
            # 一般模式
            meta_weight = 0.35
            pattern_weight = 0.30
            pf_weight = 0.35

        final = (
            meta_probs * meta_weight +
            pattern_probs * pattern_weight +
            pf_probs * pf_weight
        )

        # 優化Tie機率強制調整：確保B/P比例正確歸一化
        if final[2] < 0.088:
            scale = (1 - 0.092) / (final[0] + final[1])
            final[0] *= scale
            final[1] *= scale
            final[2] = 0.092

        final = final / final.sum()
        return final.astype(np.float32)

    def _meta_predict(self):
        h = np.array(self.history)
        total = len(h)
        pB = np.sum(h == 0) / total
        pP = np.sum(h == 1) / total
        pT = np.sum(h == 2) / total
        last = h[-1]
        streak = 1
        for i in range(2, min(10, len(h))+1):
            if h[-i] == last:
                streak += 1
            else:
                break
        state = self.meta_state()
        if state == "TREND" and streak >= 3:
            boost = self.trend_boost * (streak-2)
            if last == 0: pB += boost
            elif last == 1: pP += boost
        elif state == "CHOP":
            if last == 0: pP += self.chop_boost
            elif last == 1: pB += self.chop_boost
        else:
            if last == 0: pB += self.chaos_boost
            elif last == 1: pP += self.chaos_boost
        if streak >= 4:  # 反追機制
            if last == 0: pP += 0.038
            elif last == 1: pB += 0.038
        probs = np.array([pB, pP, pT])
        probs = np.clip(probs, 0.02, 0.96)
        return probs / probs.sum()

    def detect_pattern(self):
        if len(self.history) < 8:
            return np.array([0.458, 0.446, 0.096], dtype=np.float32)
        h = np.array(self.history)
        last3 = h[-3:]
        pB, pP, pT = 0.455, 0.445, 0.10
        if np.all(last3 == 0):
            pB += 0.045
        elif np.all(last3 == 1):
            pP += 0.045
        elif np.all(last3 == [0,1,0]) or np.all(last3 == [1,0,1]):
            pB, pP = pP + 0.035, pB + 0.035
        probs = np.array([pB, pP, pT])
        return probs / probs.sum()

    def particle_predict(self):
        particles = []
        # PF粒子生成：粒子數18，且對每個粒子的pB/pP進行clip防止極端值
        for _ in range(18):
            bias = np.random.normal(0, 0.06)
            pB = np.clip(0.45 + bias, 0.05, 0.90)
            pP = np.clip(0.45 - bias, 0.05, 0.90)
            particles.append([pB, pP, 0.10])

        weights = np.ones(18)
        if len(self.history) >= 6:
            recent = np.array(self.history)[-6:]
            for i, p in enumerate(particles):
                pred = np.argmax(p)
                match_rate = np.sum(recent == pred) / 6
                weights[i] = (1 - match_rate)**1.6 + 0.08
        weights /= weights.sum()
        final = np.average(particles, axis=0, weights=weights)
        return np.clip(final, 0.02, 0.96)
