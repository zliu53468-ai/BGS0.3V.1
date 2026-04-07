# -*- coding: utf-8 -*-
"""
CLEAN STABLE PFILTER (No anti-bias, No loss streak, No variance hack)
"""
import numpy as np

SAFE_PROBS = np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)


class OutcomePF:
    def __init__(
        self,
        decks,
        seed=42,
        n_particles=360,
        sims_lik=10,
        resample_thr=0.5,
        dirichlet_eps=0.03,
        backend=None,
        **kwargs,
    ):
        self.decks = int(decks)
        self.n_particles = max(1, int(n_particles))
        self.sims_lik = max(1, int(sims_lik))
        self.resample_thr = float(resample_thr)
        self.dirichlet_eps = float(dirichlet_eps)
        self.backend = backend or "mc"

        np.random.seed(int(seed))

        base = np.zeros(10, dtype=np.int32)
        base[0] = 16 * self.decks
        base[1:10] = 4 * self.decks

        self.particles = np.tile(base, (self.n_particles, 1)).astype(np.int32)
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=np.float64)

    def _draw(self, counts: np.ndarray) -> np.ndarray:
        counts = np.asarray(counts, dtype=np.int32)
        counts = np.maximum(counts, 0)

        if counts.ndim == 1:
            counts = counts.reshape(1, -1)

        cum = np.cumsum(counts, axis=1)
        total = cum[:, -1].astype(np.float64)
        total[total <= 0] = 1e-6

        r = np.random.rand(len(counts)) * total
        return np.argmax(cum >= r[:, None], axis=1).astype(np.int32)

    def _simulate_once(self, counts: np.ndarray) -> np.ndarray:
        counts = np.asarray(counts, dtype=np.int32).copy()
        n = len(counts)

        draws = np.zeros((n, 4), dtype=np.int32)

        for i in range(4):
            d = self._draw(counts)
            draws[:, i] = d
            counts[np.arange(n), d] -= 1
            counts = np.maximum(counts, 0)

        player = (draws[:, 0] + draws[:, 2]) % 10
        banker = (draws[:, 1] + draws[:, 3]) % 10

        natural = (player >= 8) | (banker >= 8)
        outcome = np.full(n, -1, dtype=np.int32)

        outcome[natural & (player > banker)] = 1
        outcome[natural & (player < banker)] = 0
        outcome[natural & (player == banker)] = 2

        mask = outcome == -1
        if not mask.any():
            return outcome

        idx = np.where(mask)[0]
        sub_counts = counts[idx].copy()
        sub_p = player[idx].copy()
        sub_b = banker[idx].copy()

        pt = np.full(len(idx), -1, dtype=np.int32)

        need_p3 = sub_p <= 5
        if need_p3.any():
            d = self._draw(sub_counts[need_p3])
            sub_p[need_p3] = (sub_p[need_p3] + d) % 10
            sub_counts[need_p3, d] -= 1
            sub_counts[need_p3] = np.maximum(sub_counts[need_p3], 0)
            pt[need_p3] = d

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
            sub_counts[draw_b] = np.maximum(sub_counts[draw_b], 0)

        result = np.where(sub_p > sub_b, 1, np.where(sub_p < sub_b, 0, 2)).astype(np.int32)
        outcome[idx] = result
        return outcome

    def predict(self, sims_per_particle=6) -> np.ndarray:
        try:
            sims_per_particle = max(1, int(sims_per_particle))

            all_outcomes = []
            base_particles = self.particles.copy()

            for _ in range(sims_per_particle):
                res = self._simulate_once(base_particles)
                all_outcomes.append(res)

            all_outcomes = np.stack(all_outcomes, axis=1)

            probs = np.zeros((self.n_particles, 3), dtype=np.float64)
            for i in range(self.n_particles):
                binc = np.bincount(all_outcomes[i], minlength=3)
                probs[i] = binc[:3]

            probs /= float(sims_per_particle)

            weights = np.asarray(self.weights, dtype=np.float64)
            wsum = weights.sum()
            if (not np.isfinite(weights).all()) or wsum <= 0:
                weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=np.float64)
            else:
                weights = weights / wsum

            total = np.average(probs, axis=0, weights=weights)

            if (not np.isfinite(total).all()) or total.sum() <= 0:
                return SAFE_PROBS.copy()

            total = total / total.sum()
            return total.astype(np.float32)

        except Exception:
            return SAFE_PROBS.copy()

    def update_outcome(self, outcome) -> None:
        try:
            if outcome in ("T", 2):
                outcome = 2
            elif outcome in ("P", 1):
                outcome = 1
            elif outcome in ("B", 0):
                outcome = 0
            else:
                return

            sims = self._simulate_once(self.particles.copy())
            match = (sims == outcome).astype(np.float64)

            lik = match + self.dirichlet_eps
            w = self.weights * lik
            s = w.sum()

            if s <= 0 or (not np.isfinite(s)):
                self.weights.fill(1.0 / self.n_particles)
                return

            self.weights = w / s

            ess = 1.0 / np.sum(self.weights ** 2)
            if ess < self.resample_thr * self.n_particles:
                idx = np.random.choice(self.n_particles, self.n_particles, p=self.weights)
                self.particles = self.particles[idx].copy()
                self.weights.fill(1.0 / self.n_particles)

        except Exception:
            self.weights.fill(1.0 / self.n_particles)
