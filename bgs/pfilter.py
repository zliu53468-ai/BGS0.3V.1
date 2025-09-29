
# bgs/pfilter.py — Outcome-only Particle Filter (clean version)
import numpy as np
from dataclasses import dataclass

@dataclass
class OutcomePF:
    decks: int = 8
    seed: int = 42
    n_particles: int = 200
    sims_lik: int = 80
    resample_thr: float = 0.5
    dirichlet_alpha: float = 0.8
    use_exact: bool = False  # 保留接口；此版以 MC 為主

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        # 10 桶：牌值 0..9 每值 4*decks*4 張，這裡用簡化：各值同數
        per = 16 * self.decks  # 簡化桶數
        base = np.full(10, per, dtype=np.float64)
        prior = base + float(self.dirichlet_alpha)
        self.p_counts = np.stack([
            np.maximum(0, np.rint(self.rng.normal(loc=prior, scale=np.sqrt(prior)*0.05)).astype(np.int32))
            for _ in range(self.n_particles)
        ], axis=0)
        self.weights = np.ones(self.n_particles, dtype=np.float64) / self.n_particles

    # ---- 抽樣/點數工具（簡化桶；0..9 直接當點數，10 視為 0） ----
    def _draw_card(self, counts, rng):
        tot = counts.sum()
        if tot <= 0: return 0
        i = rng.integers(0, tot)
        # 手寫累積（為速度）
        s = 0
        for v in range(10):
            s += counts[v]
            if i < s:
                counts[v] -= 1
                return v
        return 0

    @staticmethod
    def _points_add(a, b):
        return (int(a) + int(b)) % 10

    def _third_player(self, p_sum):
        return p_sum <= 5

    def _third_banker(self, b_sum, p3):
        # 近似：若玩家第三張存在且值高，莊多抽；否則按標準近似
        if p3 is None:  # 玩家未抽
            return b_sum <= 5
        # 簡化近似（非逐條款枚舉，CPU-friendly）
        if b_sum <= 2: return True
        if b_sum == 3: return p3 != 8
        if b_sum == 4: return p3 in (2,3,4,5,6,7)
        if b_sum == 5: return p3 in (4,5,6,7)
        if b_sum == 6: return p3 in (6,7)
        return False

    def _mc_prob(self, counts, sims):
        wins = np.zeros(3, dtype=np.int64)
        for _ in range(int(max(1, sims))):
            tmp = counts.copy()
            try:
                P1 = self._draw_card(tmp, self.rng); P2 = self._draw_card(tmp, self.rng)
                B1 = self._draw_card(tmp, self.rng); B2 = self._draw_card(tmp, self.rng)
                p_sum = self._points_add(P1, P2); b_sum = self._points_add(B1, B2)
                if (p_sum in (8,9)) or (b_sum in (8,9)):
                    pass
                else:
                    P3 = None
                    if self._third_player(p_sum):
                        P3 = self._draw_card(tmp, self.rng); p_sum = self._points_add(p_sum, P3)
                    if self._third_banker(b_sum, P3 if P3 is not None else None):
                        B3 = self._draw_card(tmp, self.rng); b_sum = self._points_add(b_sum, B3)
                if p_sum > b_sum: wins[1] += 1
                elif b_sum > p_sum: wins[0] += 1
                else: wins[2] += 1
            except Exception:
                continue
        tot = int(wins.sum())
        if tot == 0:
            return np.array([0.45,0.45,0.10], dtype=np.float64)
        p = wins / tot
        p[2] = np.clip(p[2], 0.06, 0.20)
        p = p / p.sum()
        return p

    # ---- 公開 API ----
    def update_outcome(self, outcome: int):
        # 替代真正扣牌：以 Dirichlet-狀態攪拌粒子
        # outcome: 0=莊,1=閒,2=和 （和局：溫和擾動）
        alpha = np.array([1.2, 1.2, 0.6])  # 偏向 B/P，和較小
        if outcome in (0,1):
            drift = 0.03 if outcome==0 else -0.03
        else:
            drift = 0.0
        for i in range(self.n_particles):
            noise = self.rng.normal(0, 0.02, size=10)
            self.p_counts[i] = np.maximum(0, self.p_counts[i] + np.rint(noise*self.p_counts[i]).astype(np.int32))
            # 輕微總量回復
            tot = self.p_counts[i].sum()
            target = int(16*self.decks*10)
            if tot <= 0:
                self.p_counts[i] = np.full(10, 16*self.decks, dtype=np.int32)
            elif tot > 0 and abs(tot-target)/target > 0.1:
                self.p_counts[i] = np.rint(self.p_counts[i] * (target/tot)).astype(np.int32)

        # 權重退火
        self.weights = (self.weights + 1e-12)
        self.weights /= self.weights.sum()

        # 視需要重採樣
        neff = 1.0 / np.sum((self.weights**2))
        if neff / len(self.weights) < self.resample_thr:
            idx = self.rng.choice(len(self.weights), size=len(self.weights), replace=True, p=self.weights/self.weights.sum())
            self.p_counts = self.p_counts[idx]
            self.weights = np.ones_like(self.weights) / len(self.weights)

    def update_point_history(self, p_pts: int, b_pts: int):
        # 本簡化版不使用點數歷史；保留接口相容
        pass

    def predict(self, sims_per_particle: int = 120) -> np.ndarray:
        probs = np.zeros(3, dtype=np.float64)
        wsum = float(self.weights.sum())
        if wsum <= 0: self.weights[:] = 1.0/len(self.weights); wsum = 1.0
        for i in range(self.n_particles):
            p = self._mc_prob(self.p_counts[i], sims_per_particle)
            probs += float(self.weights[i]) * p
        probs = np.clip(probs, 1e-9, None); probs = probs / probs.sum()
        return probs.astype(np.float32)
