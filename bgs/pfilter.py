# bgs/pfilter.py — Outcome-only Particle Filter with RB-Exact forward & Dirichlet smoothing
import numpy as np
from dataclasses import dataclass
from typing import Literal
from .deplete import init_counts

# ---------- 百家樂規則 ----------
def points_add(a, b): 
    return (a + b) % 10

def third_player(p_sum): 
    return p_sum <= 5

def third_banker(b_sum, p3):
    if b_sum <= 2: return True
    if b_sum == 3: return p3 != 8
    if b_sum == 4: return p3 in (2,3,4,5,6,7)
    if b_sum == 5: return p3 in (4,5,6,7)
    if b_sum == 6: return p3 in (6,7)
    return False

# ---------- 4 張精確枚舉（無放回） ----------
def _prob_draw_seq_4(counts: np.ndarray):
    """
    產生四連抽（P1,P2,B1,B2）的 (v1,v2,v3,v4, weight, remaining_counts)。
    counts: shape (10,) 的整數向量；index=0 代表 0 點（10/J/Q/K），1..9 代表 1..9 點。
    """
    N = int(counts.sum())
    if N < 4:
        return
    base = counts.astype(np.int64)

    for v1 in range(10):
        c1 = int(base[v1])
        if c1 <= 0:
            continue
        w1 = c1 / N
        r1 = base.copy(); r1[v1] -= 1
        for v2 in range(10):
            c2 = int(r1[v2])
            if c2 <= 0:
                continue
            w2 = w1 * (c2 / (N - 1))
            r2 = r1.copy(); r2[v2] -= 1
            for v3 in range(10):
                c3 = int(r2[v3])
                if c3 <= 0:
                    continue
                w3 = w2 * (c3 / (N - 2))
                r3 = r2.copy(); r3[v3] -= 1
                for v4 in range(10):
                    c4 = int(r3[v4])
                    if c4 <= 0:
                        continue
                    w4 = w3 * (c4 / (N - 3))
                    r4 = r3.copy(); r4[v4] -= 1
                    yield v1, v2, v3, v4, w4, r4

# ---------- RB-Exact 前向機率 ----------
def _rb_exact_prob(counts: np.ndarray) -> np.ndarray:
    """
    Rao-Blackwellized 'Exact':
    對首四張（P1,P2,B1,B2）做無放回精確枚舉，第三張使用剩餘牌的解析期望，得到近似精確的 (pB,pP,pT)。
    """
    wins = np.zeros(3, dtype=np.float64)
    totw = 0.0

    for P1, P2, B1, B2, w4, r4 in _prob_draw_seq_4(counts):
        p_sum = (P1 + P2) % 10
        b_sum = (B1 + B2) % 10

        # 天牌
        if p_sum in (8, 9) or b_sum in (8, 9):
            if p_sum > b_sum: wins[1] += w4
            elif b_sum > p_sum: wins[0] += w4
            else: wins[2] += w4
            totw += w4
            continue

        rem = r4
        R = int(rem.sum())

        # 玩家第三張期望
        if third_player(p_sum):
            for v in range(10):
                rv = int(rem[v])
                if rv <= 0:
                    continue
                pw = w4 * (rv / R)
                p_new = (p_sum + v) % 10
                rem2 = rem.copy(); rem2[v] -= 1
                R2 = R - 1

                # 莊第三張期望
                if third_banker(b_sum, v):
                    for vb in range(10):
                        rb = int(rem2[vb])
                        if rb <= 0:
                            continue
                        bw = pw * (rb / R2)
                        b_new = (b_sum + vb) % 10
                        if p_new > b_new: wins[1] += bw
                        elif b_new > p_new: wins[0] += bw
                        else: wins[2] += bw
                        totw += bw
                else:
                    if p_new > b_sum: wins[1] += pw
                    elif b_sum > p_new: wins[0] += pw
                    else: wins[2] += pw
                    totw += pw
        else:
            # 玩家不補 → 看莊是否補（p3 視作 None→10）
            if third_banker(b_sum, 10):
                for vb in range(10):
                    rb = int(rem[vb])
                    if rb <= 0:
                        continue
                    bw = w4 * (rb / R)
                    b_new = (b_sum + vb) % 10
                    if p_sum > b_new: wins[1] += bw
                    elif b_new > p_sum: wins[0] += bw
                    else: wins[2] += bw
                    totw += bw
            else:
                if p_sum > b_sum: wins[1] += w4
                elif b_sum > p_sum: wins[0] += w4
                else: wins[2] += w4
                totw += w4

    if totw <= 0.0:
        return np.array([0.45, 0.45, 0.10], dtype=np.float64)

    p = wins / totw
    p[2] = np.clip(p[2], 0.06, 0.20)  # 合理化 tie 區間
    p = p / p.sum()
    return p

# ---------- 備用 MC 前向（只在 backend='mc' 用） ----------
def _mc_prob(counts: np.ndarray, rng: np.random.Generator, sims: int = 200) -> np.ndarray:
    wins = np.zeros(3, dtype=np.int64)

    def draw(tmp: np.ndarray) -> int:
        tot = int(tmp.sum())
        r = rng.integers(0, tot)
        acc = 0
        for v in range(10):
            acc += int(tmp[v])
            if r < acc:
                tmp[v] -= 1
                return v
        # safety
        for v in range(9, -1, -1):
            if tmp[v] > 0:
                tmp[v] -= 1
                return v
        return 0

    for _ in range(sims):
        tmp = counts.copy()
        try:
            P1 = draw(tmp); P2 = draw(tmp); B1 = draw(tmp); B2 = draw(tmp)
            p_sum = (P1 + P2) % 10; b_sum = (B1 + B2) % 10
            if p_sum not in (8, 9) and b_sum not in (8, 9):
                if third_player(p_sum):
                    P3 = draw(tmp); p_sum = (p_sum + P3) % 10
                else:
                    P3 = None
                if third_banker(b_sum, P3 if P3 is not None else 10):
                    B3 = draw(tmp); b_sum = (b_sum + B3) % 10
            if p_sum > b_sum: wins[1] += 1
            elif b_sum > p_sum: wins[0] += 1
            else: wins[2] += 1
        except Exception:
            continue

    tot = wins.sum()
    if tot == 0:
        return np.array([0.45, 0.45, 0.10], dtype=np.float64)
    p = wins / tot
    p[2] = np.clip(p[2], 0.06, 0.20)
    p = p / p.sum()
    return p

# ---------- 粒子濾波主體 ----------
@dataclass
class OutcomePF:
    decks: int = 8
    seed: int = 42
    n_particles: int = 200
    sims_lik: int = 60                 # 只有 backend='mc' 會用到
    resample_thr: float = 0.5
    backend: Literal["exact", "mc"] = "exact"
    dirichlet_eps: float = 0.002       # 對權重做平滑，避免早期退化

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        base = init_counts(self.decks).astype(np.int64)
        self.p_counts = np.stack([base.copy() for _ in range(self.n_particles)], axis=0)
        self.weights = np.ones(self.n_particles, dtype=np.float64) / self.n_particles

    # 前向機率（每個粒子）
    def _forward_prob(self, counts: np.ndarray) -> np.ndarray:
        if self.backend == "exact":
            return _rb_exact_prob(counts)
        else:
            return _mc_prob(counts, self.rng, sims=max(50, int(self.sims_lik)))

    # 只用「輸贏事件」更新權重
    def update_outcome(self, outcome: int):
        """
        outcome: 0=莊勝, 1=閒勝, 2=和
        對每個粒子估 p(outcome | counts) 後做 Bayes 更新；加 Dirichlet/Laplace 平滑避免 0 機率。
        """
        eps = max(1e-6, float(self.dirichlet_eps))
        lik = np.zeros(self.n_particles, dtype=np.float64)

        for i in range(self.n_particles):
            p = self._forward_prob(self.p_counts[i])
            # 平滑避免極小值導致權重崩潰
            lik[i] = eps + (1.0 - 3.0 * eps) * float(p[outcome])

        self.weights *= lik
        s = float(self.weights.sum())
        if s <= 0.0:
            self.weights[:] = 1.0 / self.n_particles
        else:
            self.weights /= s

        # 退化檢查：有效粒子數
        neff = 1.0 / np.sum(np.square(self.weights))
        if neff < self.resample_thr * self.n_particles:
            idx = self.rng.choice(self.n_particles, size=self.n_particles, replace=True, p=self.weights)
            self.p_counts = self.p_counts[idx].copy()
            self.weights[:] = 1.0 / self.n_particles

    # 產生下一局機率（加權平均）
    def predict(self, sims_per_particle: int = 0) -> np.ndarray:
        agg = np.zeros(3, dtype=np.float64)
        for i in range(self.n_particles):
            p = self._forward_prob(self.p_counts[i])
            agg += self.weights[i] * p
        out = agg
        out[2] = np.clip(out[2], 0.06, 0.20)
        out = out / out.sum()
        return out.astype(np.float32)
