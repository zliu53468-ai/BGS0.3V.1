# -*- coding: utf-8 -*-
"""
CLEAN STABLE PFILTER (No anti-bias, No loss streak, No variance hack)
"""

import numpy as np

SAFE_PROBS = np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)


class OutcomePF:

    def __init__(self, decks, seed=42,
                 n_particles=200,
                 sims_lik=10,
                 resample_thr=0.5,
                 dirichlet_eps=0.03):

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

    # ===== 抽牌輔助 =====
    def _draw(self, counts):
        cum = np.cumsum(np.maximum(counts, 0), axis=1)
        total = cum[:, -1]
        total[total <= 0] = 1e-6
        r = np.random.rand(len(counts)) * total
        return np.argmax(cum >= r[:, None], axis=1)

    # ===== 核心模擬引擎 =====
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

        # 玩家補牌
        need_p3 = sub_p <= 5
        if need_p3.any():
            d = self._draw(sub_counts[need_p3])
            sub_p[need_p3] = (sub_p[need_p3] + d) % 10
            sub_counts[need_p3, d] -= 1
            pt[need_p3] = d

        # 莊補牌規則
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

    # ===== 預測機率（無任何干預）=====
    def predict(self, sims_per_particle=6):

        try:
            all_outcomes = []

            for _ in range(sims_per_particle):
                res = self._simulate_once(self.particles.copy())
                all_outcomes.append(res)

            all_outcomes = np.stack(all_outcomes, axis=1)

            probs = np.zeros((self.n_particles, 3))

            for i in range(self.n_particles):
                probs[i] = np.bincount(all_outcomes[i], minlength=3)

            probs /= sims_per_particle

            total = np.average(probs, axis=0, weights=self.weights)

            if not np.isfinite(total).all() or total.sum() <= 0:
                return SAFE_PROBS

            return (total / total.sum()).astype(np.float32)

        except Exception:
            return SAFE_PROBS

    # ===== 更新粒子權重（純 PF）=====
    def update_outcome(self, outcome):

        try:
            if outcome in ('T', 2): outcome = 2
            elif outcome in ('P', 1): outcome = 1
            elif outcome in ('B', 0): outcome = 0
            else: return

            sims = self._simulate_once(self.particles.copy())

            match = (sims == outcome).astype(float)

            lik = match + self.dirichlet_eps

            w = self.weights * lik
            s = w.sum()

            if s <= 0 or not np.isfinite(s):
                self.weights.fill(1.0 / self.n_particles)
                return

            self.weights = w / s

            # 有效樣本數重採樣
            ess = 1.0 / np.sum(self.weights ** 2)

            if ess < self.resample_thr * self.n_particles:
                idx = np.random.choice(self.n_particles, self.n_particles, p=self.weights)
                self.particles = self.particles[idx]
                self.weights.fill(1.0 / self.n_particles)

        except Exception:
            self.weights.fill(1.0 / self.n_particles)
