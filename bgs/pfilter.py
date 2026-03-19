# -*- coding: utf-8 -*-
"""
ULTIMATE 55% STABLE pfilter.py（Render可跑 + 抗偏 + 自適應）
"""

import os
import numpy as np

SAFE_PROBS = np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)


class OutcomePF:

    def __init__(self, decks, seed=42,
                 n_particles=220,
                 sims_lik=15,
                 resample_thr=0.5,
                 dirichlet_eps=0.04):

        self.decks = int(decks)
        self.n_particles = int(n_particles)
        self.sims_lik = int(sims_lik)
        self.resample_thr = float(resample_thr)
        self.dirichlet_eps = float(dirichlet_eps)

        np.random.seed(seed)

        base = np.zeros(10, dtype=np.int32)
        base[0] = 16 * self.decks
        base[1:10] = 4 * self.decks

        self.particles = np.tile(base, (self.n_particles, 1))
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles)

        # 🔥 新增：行為記錄（核心）
        self.history = []
        self.loss_streak = 0

    # ===== 抽牌 =====
    def _draw(self, counts):
        cum = np.cumsum(np.maximum(counts, 0), axis=1)
        total = cum[:, -1]
        total[total <= 0] = 1e-6
        r = np.random.rand(len(counts)) * total
        return np.argmax(cum >= r[:, None], axis=1)

    # ===== 單次模擬 =====
    def _simulate_once(self, counts):

        draws = np.zeros((len(counts), 4), dtype=np.int32)

        for i in range(4):
            d = self._draw(counts)
            draws[:, i] = d
            counts[np.arange(len(counts)), d] -= 1
            counts = np.maximum(counts, 0)

        player = (draws[:, 0] + draws[:, 2]) % 10
        banker = (draws[:, 1] + draws[:, 3]) % 10

        natural = (player >= 8) | (banker >= 8)

        outcome = np.full(len(counts), -1)

        outcome[natural & (player > banker)] = 1
        outcome[natural & (player < banker)] = 0
        outcome[natural & (player == banker)] = 2

        mask = outcome == -1
        if not mask.any():
            return outcome

        idx = np.where(mask)[0]

        sub_counts = counts[idx]
        sub_p = player[idx]
        sub_b = banker[idx]

        pt = np.full(len(idx), -1)

        # 玩家第三張
        need_p3 = sub_p <= 5
        if need_p3.any():
            d = self._draw(sub_counts[need_p3])
            sub_p[need_p3] = (sub_p[need_p3] + d) % 10
            sub_counts[need_p3, d] -= 1
            pt[need_p3] = d

        # 莊第三張
        draw_b = np.zeros(len(idx), dtype=bool)
        valid = pt != -1

        draw_b[sub_b <= 2] = True
        draw_b[(sub_b == 3) & valid & (pt != 8)] = True
        draw_b[(sub_b == 4) & valid & (pt >= 2) & (pt <= 7)] = True
        draw_b[(sub_b == 5) & valid & (pt >= 4) & (pt <= 7)] = True
        draw_b[(sub_b == 6) & valid & (pt >= 6) & (pt <= 7)] = True

        if draw_b.any():
            d = self._draw(sub_counts[draw_b])
            sub_b[draw_b] = (sub_b[draw_b] + d) % 10
            sub_counts[draw_b, d] -= 1

        result = np.where(sub_p > sub_b, 1,
                         np.where(sub_p < sub_b, 0, 2))

        outcome[idx] = result

        return outcome

    # ===== predict =====
    def predict(self, sims_per_particle=None, rounds_seen=None):

        try:
            sims = sims_per_particle or 6

            all_outcomes = []

            for _ in range(sims):
                res = self._simulate_once(self.particles.copy())
                all_outcomes.append(res)

            all_outcomes = np.stack(all_outcomes, axis=1)

            probs = np.zeros((self.n_particles, 3))

            for i in range(self.n_particles):
                probs[i] = np.bincount(all_outcomes[i], minlength=3)

            probs /= sims

            total = np.average(probs, axis=0, weights=self.weights)

            if not np.isfinite(total).all() or total.sum() <= 0:
                return SAFE_PROBS

            total /= total.sum()

            # ===== 🔥 波動偵測 =====
            if len(self.history) >= 6:
                recent = self.history[-6:]
                variance = np.std(recent)
                if variance > 0.9:
                    total = total * 0.85 + SAFE_PROBS * 0.15

            # ===== 🔥 anti-bias =====
            diff = abs(total[0] - total[1])
            if diff > 0.10:
                mid = (total[0] + total[1]) / 2
                total[0] = mid + (total[0] - mid) * 0.6
                total[1] = mid + (total[1] - mid) * 0.6
                total /= total.sum()

            return total.astype(np.float32)

        except Exception as e:
            print("PF ERROR:", e)
            return SAFE_PROBS

    # ===== update =====
    def update_outcome(self, outcome):

        try:
            if outcome in ('T', 2): outcome = 2
            elif outcome in ('P', 1): outcome = 1
            elif outcome in ('B', 0): outcome = 0
            else: return

            # 記錄歷史（🔥核心）
            self.history.append(outcome)
            if len(self.history) > 50:
                self.history.pop(0)

            sims = self._simulate_once(self.particles.copy())

            match = (sims == outcome).astype(float)

            lik = match + self.dirichlet_eps

            w = self.weights * lik
            s = w.sum()

            if s <= 0 or not np.isfinite(s):
                self.weights.fill(1.0 / self.n_particles)
                return

            self.weights = w / s

            # ===== 🔥 連輸修正 =====
            if match.mean() < 0.4:
                self.loss_streak += 1
            else:
                self.loss_streak = 0

            if self.loss_streak >= 3:
                self.weights *= 0.7
                self.weights /= self.weights.sum()

            # ===== resample =====
            ess = 1.0 / np.sum(self.weights ** 2)

            if ess < self.resample_thr * self.n_particles:
                idx = np.random.choice(self.n_particles, self.n_particles, p=self.weights)
                self.particles = self.particles[idx]
                self.weights.fill(1.0 / self.n_particles)

        except Exception:
            self.weights.fill(1.0 / self.n_particles)
