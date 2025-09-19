# bgs/pfilter.py — Outcome-only Particle Filter with RB-Exact forward & Dirichlet smoothing
import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional, Dict
# Attempt to import ``init_counts`` from the local ``deplete`` module.  When
# running inside a package this relative import works; when executed as a
# standalone file the relative import fails and we fall back to a top‑level
# import instead.  If both attempts fail the ImportError is re‑raised to
# surface the underlying issue.
try:
    from .deplete import init_counts  # type: ignore
except Exception:
    try:
        from deplete import init_counts  # type: ignore
    except Exception as _exc:
        raise _exc

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

# ---------- 專業機率校準 ----------
def professional_calibration(prob: np.ndarray, prev_p_pts: int = None, prev_b_pts: int = None) -> np.ndarray:
    """
    專業級機率校準 - 考慮點數差和歷史反轉模式
    """
    pB, pP, pT = prob[0], prob[1], prob[2]
    
    # 1. 基礎正規化確保總和為1
    total = pB + pP + pT
    if abs(total - 1.0) > 0.001:
        pB /= total
        pP /= total
        pT /= total
    
    # 2. 點數差智能調整
    if prev_p_pts is not None and prev_b_pts is not None:
        point_diff = abs(prev_p_pts - prev_b_pts)
        
        # 點數接近時的智能調整
        if point_diff <= 2:  # 點數非常接近
            # 降低極端預測，增加不確定性
            uncertainty_factor = 0.7 - (point_diff * 0.1)
            pB = 0.5 + (pB - 0.5) * uncertainty_factor
            pP = 0.5 + (pP - 0.5) * uncertainty_factor
            
            # 點數接近時適度提高和局機率
            pT = max(0.06, min(0.12, pT * 1.3))
        
        # 大點數差時的穩定性調整
        elif point_diff >= 5:
            # 大點數差時稍微強化趨勢
            trend_strength = 1.1
            if pB > pP:
                pB *= trend_strength
            else:
                pP *= trend_strength
    
    # 3. 基於統計學的機率校正
    banker_advantage = 0.008  # 莊家天然優勢
    
    if pB > pP:
        pB += banker_advantage * 0.6
        pP -= banker_advantage * 0.6
    elif pP > pB:
        pP += banker_advantage * 0.4
        pB -= banker_advantage * 0.4
    
    # 4. 和局機率合理化
    pT = max(0.035, min(0.095, pT))
    
    # 5. 機率邊界保護
    pB = max(0.40, min(0.58, pB))
    pP = max(0.40, min(0.58, pP))
    
    # 6. 最終正規化
    total = pB + pP + pT
    return np.array([pB/total, pP/total, pT/total], dtype=np.float64)

# ---------- RB-Exact 前向機率 ----------
def _rb_exact_prob(counts: np.ndarray, prev_p_pts: int = None, prev_b_pts: int = None) -> np.ndarray:
    """
    Rao-Blackwellized 'Exact' with point difference consideration
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
        return np.array([0.458, 0.446, 0.096], dtype=np.float64)

    p = wins / totw
    # 應用專業校準（傳入上局點數）
    return professional_calibration(p, prev_p_pts, prev_b_pts)

# ---------- 備用 MC 前向 ----------
def _mc_prob(counts: np.ndarray, rng: np.random.Generator, sims: int = 200, 
             prev_p_pts: int = None, prev_b_pts: int = None) -> np.ndarray:
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
        return np.array([0.458, 0.446, 0.096], dtype=np.float64)
    p = wins / tot
    # 應用專業校準
    return professional_calibration(p, prev_p_pts, prev_b_pts)

# ---------- 粒子濾波主體 ----------
@dataclass
class OutcomePF:
    decks: int = 8
    seed: int = 42
    n_particles: int = 100
    sims_lik: int = 80
    resample_thr: float = 0.3
    backend: Literal["exact", "mc"] = "exact"
    dirichlet_eps: float = 0.005
    stability_factor: float = 0.8
    prev_p_pts: Optional[int] = None
    prev_b_pts: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        base = init_counts(self.decks).astype(np.int64)
        self.p_counts = np.stack([base.copy() for _ in range(self.n_particles)], axis=0)
        self.weights = np.ones(self.n_particles, dtype=np.float64) / self.n_particles
        self.prediction_history = []
        self.point_diff_history = []

    # 前向機率（每個粒子）
    def _forward_prob(self, counts: np.ndarray) -> np.ndarray:
        if self.backend == "exact":
            return _rb_exact_prob(counts, self.prev_p_pts, self.prev_b_pts)
        else:
            return _mc_prob(counts, self.rng, sims=max(50, int(self.sims_lik)), 
                           prev_p_pts=self.prev_p_pts, prev_b_pts=self.prev_b_pts)

    # 更新點數記錄
    def update_point_history(self, p_pts: int, b_pts: int):
        """記錄點數歷史用於反轉模式分析"""
        self.prev_p_pts = p_pts
        self.prev_b_pts = b_pts
        point_diff = abs(p_pts - b_pts)
        self.point_diff_history.append(point_diff)
        if len(self.point_diff_history) > 20:
            self.point_diff_history.pop(0)

    # 只用「輸贏事件」更新權重
    def update_outcome(self, outcome: int):
        eps = max(1e-6, float(self.dirichlet_eps))
        lik = np.zeros(self.n_particles, dtype=np.float64)

        for i in range(self.n_particles):
            p = self._forward_prob(self.p_counts[i])
            lik[i] = eps + (1.0 - 3.0 * eps) * float(p[outcome]) * self.stability_factor

        self.weights *= lik
        s = float(self.weights.sum())
        if s <= 0.0:
            self.weights[:] = 1.0 / self.n_particles
        else:
            self.weights /= s

        # 退化檢查：有效粒子數
        neff = 1.0 / np.sum(np.square(self.weights))
        if neff < self.resample_thr * self.n_particles:
            # 使用系統性重採樣，更穩定
            cumulative_weights = np.cumsum(self.weights)
            uniform_samples = (np.arange(self.n_particles) + self.rng.random()) / self.n_particles
            new_indices = np.searchsorted(cumulative_weights, uniform_samples)
            self.p_counts = self.p_counts[new_indices].copy()
            self.weights[:] = 1.0 / self.n_particles

    # 產生下一局機率
    def predict(self, sims_per_particle: int = 0) -> np.ndarray:
        agg = np.zeros(3, dtype=np.float64)
        valid_particles = 0
        
        for i in range(self.n_particles):
            try:
                p = self._forward_prob(self.p_counts[i])
                if self.weights[i] > 0.001:
                    agg += self.weights[i] * p
                    valid_particles += 1
            except:
                continue
        
        if valid_particles < 5:
            return np.array([0.458, 0.446, 0.096], dtype=np.float32)
        
        out = professional_calibration(agg, self.prev_p_pts, self.prev_b_pts)
        
        # 記錄預測歷史
        self.prediction_history.append(out.copy())
        if len(self.prediction_history) > 50:
            self.prediction_history.pop(0)
            
        return out.astype(np.float32)

    def get_reversal_probability(self) -> float:
        """計算反轉機率基於點數差歷史"""
        if len(self.point_diff_history) < 3:
            return 0.3
        
        # 分析點數差模式來預測反轉可能性
        recent_diffs = self.point_diff_history[-3:]
        avg_diff = np.mean(recent_diffs)
        
        # 點數差越小，反轉機率越高
        if avg_diff <= 1:
            return 0.6  # 60% 反轉機率
        elif avg_diff <= 2:
            return 0.45  # 45% 反轉機率
        else:
            return 0.25  # 25% 反轉機率

    def get_accuracy_metrics(self) -> Dict[str, float]:
        if len(self.prediction_history) < 5:
            return {"confidence": 0.5, "stability": 0.5, "reversal_risk": 0.3}
        
        recent_preds = np.array(self.prediction_history[-5:])
        std_dev = np.std(recent_preds, axis=0)
        confidence = 1.0 - np.mean(std_dev)
        
        return {
            "confidence": float(confidence),
            "stability": float(1.0 - np.max(std_dev)),
            "reversal_risk": self.get_reversal_probability() * 100
        }
