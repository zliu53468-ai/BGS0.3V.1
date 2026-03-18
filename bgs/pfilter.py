# -*- coding: utf-8 -*-
"""
最終版 pfilter.py — dynamic shoe depletion + 更快 vectorized 模擬 + 防護強化
與 server.py 100% 相容
修正：粒子與條件陣列長度不一致導致廣播錯誤
"""
import os
import numpy as np

class OutcomePF:
    def __init__(self, decks: int, seed: int = 42,
                 n_particles: int = 800,
                 sims_lik: int = 20,
                 resample_thr: float = 0.5,
                 backend: str = 'numpy',
                 dirichlet_eps: float = 0.01):
        self.decks = int(decks)
        self.n_particles = int(n_particles)
        self.sims_lik = int(sims_lik)
        self.resample_thr = float(resample_thr)
        self.dirichlet_eps = float(dirichlet_eps)
        np.random.seed(seed)

        base = np.zeros(10, dtype=np.int32)
        base[0] = 16 * self.decks  # A
        base[1:10] = 4 * self.decks  # 2~K
        self.particles = np.tile(base, (self.n_particles, 1)).astype(np.int32)

        noise = np.random.dirichlet(np.ones(10) * 0.4, size=self.n_particles)
        noise = (noise * 0.015 * base.sum()).astype(np.int32)
        self.particles += noise - noise.mean(axis=0, keepdims=True).astype(np.int32)
        self.particles = np.maximum(0, self.particles).astype(np.int32)

        self.weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=np.float64)
        self.lik_denom = self.sims_lik + 3 * self.dirichlet_eps

    def _fast_simulate(self, particles: np.ndarray, n_sims: int) -> np.ndarray:
        N = particles.shape[0]
        outcomes = np.zeros((N, n_sims), dtype=np.int8)

        for sim_idx in range(n_sims):
            counts = particles.copy()  # (N, 10)

            # 抽牌 indices
            cum = np.cumsum(counts, axis=1)
            total = cum[:, -1] + 1e-12
            r = np.random.rand(N, 4) * total[:, None]
            draws = np.argmax(cum[:, None, :] >= r[:, :, None], axis=-1)  # (N,4)

            # 扣牌 (vectorized)
            for d in range(4):
                np.add.at(counts, (np.arange(N), draws[:, d]), -1)

            player = (draws[:,0] + draws[:,2]) % 10
            banker = (draws[:,1] + draws[:,3]) % 10

            natural = (player >= 8) | (banker >= 8)
            outcome = np.select(
                [natural & (player > banker), natural & (player < banker), natural],
                [1, 0, 2],
                default=-1
            )

            mask_cont = outcome == -1
            if not mask_cont.any():
                outcomes[:, sim_idx] = outcome
                continue

            # 只處理需要繼續的粒子
            idx = np.nonzero(mask_cont)[0]
            if len(idx) == 0:
                outcomes[:, sim_idx] = outcome
                continue

            sub_counts = counts[idx]
            sub_player = player[idx]
            sub_banker = banker[idx]

            # 閒第三張
            draw_p3 = sub_player <= 5
            idx_p3 = np.nonzero(draw_p3)[0]  # 相對於 sub 的索引
            if idx_p3.size > 0:
                cum_p3 = np.cumsum(np.maximum(sub_counts[idx_p3], 0), axis=1)
                tot_p3 = cum_p3[:, -1] + 1e-12
                r_p3 = np.random.rand(len(idx_p3)) * tot_p3
                p3 = np.argmax(cum_p3 >= r_p3[:, None], axis=1)
                sub_player[idx_p3] = (sub_player[idx_p3] + p3) % 10

                # 扣牌
                np.add.at(sub_counts[idx_p3], (np.arange(len(idx_p3)), p3), -1)

                pt = p3 % 10
            else:
                pt = np.full(len(idx), -1, dtype=np.int8)

            # 莊第三張
            b_now = sub_banker
            draw_b = np.zeros(len(idx), dtype=bool)
            draw_b[b_now <= 2] = True
            draw_b[(b_now == 3) & (pt != 8)] = True
            draw_b[(b_now == 4) & (pt >= 2) & (pt <= 7)] = True
            draw_b[(b_now == 5) & (pt >= 4) & (pt <= 7)] = True
            draw_b[(b_now == 6) & (pt >= 6) & (pt <= 7)] = True

            idx_b3 = np.nonzero(draw_b)[0]  # 相對於 sub 的索引
            if idx_b3.size > 0:
                cum_b3 = np.cumsum(np.maximum(sub_counts[idx_b3], 0), axis=1)
                tot_b3 = cum_b3[:, -1] + 1e-12
                r_b3 = np.random.rand(len(idx_b3)) * tot_b3
                b3 = np.argmax(cum_b3 >= r_b3[:, None], axis=1)
                sub_banker[idx_b3] = (sub_banker[idx_b3] + b3) % 10

                # 扣牌
                np.add.at(sub_counts[idx_b3], (np.arange(len(idx_b3)), b3), -1)

            # 更新 outcome
            sub_outcome = np.where(sub_player > sub_banker, 1,
                                   np.where(sub_player < sub_banker, 0, 2))
            outcome[idx] = sub_outcome

            outcomes[:, sim_idx] = outcome

        return outcomes

    def predict(self, sims_per_particle: int = None, rounds_seen=None) -> np.ndarray:
        if sims_per_particle is None:
            sims_per_particle = int(os.getenv("PF_PRED_SIMS", "3"))
        if self.n_particles == 0:
            return np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)

        outcomes = self._fast_simulate(self.particles, sims_per_particle)
        counts = np.zeros((self.n_particles, 3), dtype=np.int32)
        for i in range(self.n_particles):
            counts[i] = np.bincount(outcomes[i], minlength=3)

        probs = counts / sims_per_particle
        total = np.average(probs, axis=0, weights=self.weights)
        total /= total.sum() + 1e-12

        # anti-extreme
        if max(total[0], total[1]) > 0.62:
            theo = np.array([0.4586, 0.4462, 0.0952])
            total[:2] = 0.65 * total[:2] + 0.35 * theo[:2]
            total /= total.sum()

        return total.astype(np.float32)

    def update_outcome(self, outcome: int | str):
        if outcome in ('T', 2, 't'): outcome = 2
        elif outcome in ('P', 1, 'p'): outcome = 1
        elif outcome in ('B', 0, 'b'): outcome = 0
        else: return

        if outcome == 2 and int(os.getenv('SKIP_TIE_UPD', '1')): return

        N = self.n_particles
        if N == 0: return

        sims = self._fast_simulate(self.particles, self.sims_lik)
        match = (sims == outcome).sum(axis=1)
        lik = (match + self.dirichlet_eps) / (self.sims_lik + 3 * self.dirichlet_eps)

        w = self.weights * lik
        s = w.sum()
        self.weights = w / s if s > 1e-12 else np.full(N, 1.0 / N)

        ess = 1.0 / np.sum(self.weights ** 2)
        if ess < self.resample_thr * N:
            idx = np.random.choice(N, N, p=self.weights)
            self.particles = self.particles[idx].copy()
            self.weights.fill(1.0 / N)
