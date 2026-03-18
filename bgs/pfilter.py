# -*- coding: utf-8 -*-
"""
ULTIMATE SAFE pfilter.py（完全防爆版）
"""

import os
import numpy as np


SAFE_PROBS = np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)


class OutcomePF:

    def __init__(self, decks, seed=42,
                 n_particles=300,
                 sims_lik=10,
                 resample_thr=0.5,
                 dirichlet_eps=0.02):

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

        noise = np.random.dirichlet(np.ones(10), size=self.n_particles)
        noise = (noise * 0.01 * base.sum()).astype(np.int32)

        self.particles += noise
        self.particles = np.maximum(0, self.particles)

        self.weights = np.full(self.n_particles, 1.0 / self.n_particles)

    # ======================
    # SAFE 抽牌
    # ======================
    def _draw_cards(self, counts, k):

        cum = np.cumsum(np.maximum(counts, 0), axis=1)
        total = cum[:, -1]

        total[total <= 0] = 1e-6

        r = np.random.rand(len(counts), k) * total[:, None]
        draws = np.argmax(cum[:, None, :] >= r[:, :, None], axis=-1)

        return draws

    # ======================
    # SAFE 模擬
    # ======================
    def _fast_simulate(self, particles, n_sims):

        N = particles.shape[0]
        outcomes = np.zeros((N, n_sims), dtype=np.int8)

        for s in range(n_sims):

            counts = particles.copy()

            draws = self._draw_cards(counts, 4)

            for d in range(4):
                counts[np.arange(N), draws[:, d]] = np.maximum(
                    0, counts[np.arange(N), draws[:, d]] - 1
                )

            player = (draws[:, 0] + draws[:, 2]) % 10
            banker = (draws[:, 1] + draws[:, 3]) % 10

            natural = (player >= 8) | (banker >= 8)

            outcome = np.select(
                [natural & (player > banker),
                 natural & (player < banker),
                 natural],
                [1, 0, 2],
                default=-1
            )

            mask = outcome == -1
            if not mask.any():
                outcomes[:, s] = outcome
                continue

            idx = np.nonzero(mask)[0]

            sub_p = player[idx]
            sub_b = banker[idx]
            sub_N = len(idx)

            pt_full = np.full(sub_N, -1, dtype=np.int8)

            # ===== 玩家第三張 =====
            draw_p3 = sub_p <= 5
            idx_p3 = np.nonzero(draw_p3)[0]

            if idx_p3.size > 0:

                sub_counts = counts[idx]

                draws_p3 = self._draw_cards(sub_counts[idx_p3], 1).flatten()

                sub_p[idx_p3] = (sub_p[idx_p3] + draws_p3) % 10

                rows = idx[idx_p3]
                counts[rows, draws_p3] = np.maximum(
                    0, counts[rows, draws_p3] - 1
                )

                pt_full[idx_p3] = draws_p3

            # ===== 莊第三張 =====
            b_now = sub_b
            draw_b = np.zeros(sub_N, dtype=bool)

            valid = pt_full != -1

            draw_b[b_now <= 2] = True
            draw_b[(b_now == 3) & valid & (pt_full != 8)] = True
            draw_b[(b_now == 4) & valid & (pt_full >= 2) & (pt_full <= 7)] = True
            draw_b[(b_now == 5) & valid & (pt_full >= 4) & (pt_full <= 7)] = True
            draw_b[(b_now == 6) & valid & (pt_full >= 6) & (pt_full <= 7)] = True

            idx_b3 = np.nonzero(draw_b)[0]

            if idx_b3.size > 0:

                sub_counts = counts[idx]

                draws_b3 = self._draw_cards(sub_counts[idx_b3], 1).flatten()

                sub_b[idx_b3] = (sub_b[idx_b3] + draws_b3) % 10

                rows = idx[idx_b3]
                counts[rows, draws_b3] = np.maximum(
                    0, counts[rows, draws_b3] - 1
                )

            outcome[idx] = np.where(
                sub_p > sub_b, 1,
                np.where(sub_p < sub_b, 0, 2)
            )

            outcomes[:, s] = outcome

        return outcomes

    # ======================
    # SAFE predict
    # ======================
    def predict(self, sims_per_particle=None, rounds_seen=None):

        try:

            if sims_per_particle is None:
                sims_per_particle = int(os.getenv("PF_PRED_SIMS", "2"))

            if self.n_particles == 0:
                return SAFE_PROBS

            outcomes = self._fast_simulate(self.particles, sims_per_particle)

            counts = np.zeros((self.n_particles, 3))

            for i in range(self.n_particles):
                counts[i] = np.bincount(outcomes[i], minlength=3)

            probs = counts / max(1, sims_per_particle)

            total = np.average(probs, axis=0, weights=self.weights)

            if not np.isfinite(total).all() or total.sum() <= 0:
                return SAFE_PROBS

            total /= total.sum()

            return total.astype(np.float32)

        except Exception:
            return SAFE_PROBS

    # ======================
    # SAFE update
    # ======================
    def update_outcome(self, outcome):

        try:

            if outcome in ('T', 2): outcome = 2
            elif outcome in ('P', 1): outcome = 1
            elif outcome in ('B', 0): outcome = 0
            else: return

            if outcome == 2 and int(os.getenv('SKIP_TIE_UPD', '1')):
                return

            sims = self._fast_simulate(self.particles, self.sims_lik)

            match = (sims == outcome).sum(axis=1)

            lik = (match + self.dirichlet_eps) / (
                self.sims_lik + 3 * self.dirichlet_eps
            )

            w = self.weights * lik
            s = w.sum()

            if s <= 0 or not np.isfinite(s):
                self.weights.fill(1.0 / self.n_particles)
                return

            self.weights = w / s

            ess = 1.0 / np.sum(self.weights ** 2)

            if ess < self.resample_thr * self.n_particles:

                idx = np.random.choice(
                    self.n_particles, self.n_particles, p=self.weights
                )

                self.particles = self.particles[idx]

                noise = np.random.normal(0, 0.2, self.particles.shape)
                self.particles = np.maximum(0, self.particles + noise).astype(np.int32)

                self.weights.fill(1.0 / self.n_particles)

        except Exception:
            self.weights.fill(1.0 / self.n_particles)
