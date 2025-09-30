# bgs/pfilter.py — Outcome-only Particle Filter for Baccarat
import numpy as np
from dataclasses import dataclass
from .deplete import init_counts, draw_card, points_add, third_card_rule_player, third_card_rule_banker

@dataclass
class OutcomePF:
    decks: int = 8
    seed: int = 42
    n_particles: int = 200
    sims_lik: int = 80             # 更新時每粒子小模擬次數（用來估似然）
    resample_thr: float = 0.5      # 有效樣本比門檻
    dirichlet_alpha: float = 0.8   # Dirichlet 先驗（對 10 桶的平滑）
    use_exact: bool = False        # True: Exact-lite 前向；False: MC 前向

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        base = init_counts(self.decks).astype(np.float64)
        # Dirichlet 先驗：對每桶加 α，再四捨五入成整數初始化（避免極早期退化）
        prior = base + self.dirichlet_alpha
        # 初始化粒子：在 prior 周圍加些微擾
        self.p_counts = np.stack([
            np.maximum(0, np.rint(self.rng.normal(loc=prior, scale=np.sqrt(prior)*0.05)).astype(np.int32))
            for _ in range(self.n_particles)
        ], axis=0)
        self.weights = np.ones(self.n_particles, dtype=np.float64) / self.n_particles

    # --------- 前向模型：回傳在某個 counts 下的 (pB,pP,pT) ----------
    def _simulate_outcome_prob_mc(self, counts, sims):
        wins = np.zeros(3, dtype=np.int64)
        for _ in range(sims):
            tmp = counts.copy()
            try:
                P1 = draw_card(tmp, self.rng); P2 = draw_card(tmp, self.rng)
                B1 = draw_card(tmp, self.rng); B2 = draw_card(tmp, self.rng)
                p_sum = points_add(P1, P2); b_sum = points_add(B1, B2)
                if p_sum in (8,9) or b_sum in (8,9):
                    pass
                else:
                    if third_card_rule_player(p_sum):
                        P3 = draw_card(tmp, self.rng); p_sum = points_add(p_sum, P3)
                    else:
                        P3 = None
                    if third_card_rule_banker(b_sum, P3 if P3 is not None else 10):
                        B3 = draw_card(tmp, self.rng); b_sum = points_add(b_sum, B3)
                if p_sum > b_sum: wins[1] += 1
                elif b_sum > p_sum: wins[0] += 1
                else: wins[2] += 1
            except:
                continue
        tot = wins.sum()
        if tot == 0: 
            return np.array([0.45,0.45,0.10], dtype=np.float64)
        p = wins / tot
        p[2] = np.clip(p[2], 0.06, 0.20)
        p = p / p.sum()
        return p

    def _simulate_outcome_prob_exactlite(self, counts):
        """
        Exact-lite：以超幾何對『起手四張』精確枚舉，對第三張用期望近似。
        速度快、比純MC穩，仍保持 CPU-friendly。
        """
        total = counts.sum()
        if total < 4:
            return np.array([0.45,0.45,0.10], dtype=np.float64)

        # 起手兩兩抽樣的精確概率
        p_acc = np.zeros(3, dtype=np.float64)

        # 對所有 (i,j,k,l) 牌值桶做枚舉（10^4 理論最大，但很多組在 counts 下是 0；用剪枝）
        # 為效率：只列出 counts>0 的桶
        avail = [v for v in range(10) if counts[v] > 0]
        for i in avail:
            ci = counts[i]; if ci == 0: continue
            for j in avail:
                cj = counts[j] - (1 if j==i else 0)
                if cj <= 0: continue
                for k in avail:
                    ck = counts[k] - (1 if k==i else 0) - (1 if k==j else 0)
                    if ck <= 0: continue
                    for l in avail:
                        cl = counts[l] - (1 if l==i else 0) - (1 if l==j else 0) - (1 if l==k else 0)
                        if cl <= 0: continue

                        # 超幾何權重（抽樣順序近似，足夠精準）
                        #
