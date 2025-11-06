import os
import numpy as np

def _stage_bounds():
    early_end = int(os.getenv("EARLY_HANDS", "20"))
    mid_end   = int(os.getenv("MID_HANDS",   os.getenv("LATE_HANDS", "56")))
    return early_end, mid_end

def _stage_prefix(rounds_seen: int | None) -> str:
    if rounds_seen is None: return ""
    e_end, m_end = _stage_bounds()
    if rounds_seen < e_end: return "EARLY_"
    elif rounds_seen < m_end: return "MID_"
    else: return "LATE_"

class OutcomePF:
    def __init__(self, decks: int, seed: int = None, 
                 n_particles: int = 1000, sims_lik: int = 100, 
                 resample_thr: float = 0.5, backend: str = 'numpy', 
                 dirichlet_eps: float = 1e-6):
        # 初始化參數
        self.decks = decks
        self.n_particles = n_particles
        self.sims_lik = sims_lik
        self.resample_thr = resample_thr
        self.backend = backend
        self.dirichlet_eps = dirichlet_eps
        mode = os.getenv('HISTORY_MODE', '0')
        self.history_mode = int(mode) if mode.isdigit() else 0
        if seed is not None:
            np.random.seed(seed)
        # 基礎牌靴
        self.base_counts = self._init_card_counts(decks)
        # 粒子
        self.particles = [self.base_counts.copy() for _ in range(n_particles)]
        self.weights = np.ones(n_particles, dtype=np.float64) / n_particles

    def _init_card_counts(self, decks: int):
        counts = {0: 16 * decks}
        for point in range(1, 10):
            counts[point] = 4 * decks
        return counts

    def _simulate_round(self, counts):
        # 以 counts 展開成列，用於抽樣
        card_values = []
        for value, count in counts.items():
            card_values.extend([value] * count)
        card_values = np.array(card_values, dtype=np.int8)

        draws = np.random.choice(card_values, size=4, replace=False)
        temp_counts = counts.copy()
        for val in draws:
            temp_counts[val] -= 1

        player_total = (draws[0] + draws[2]) % 10
        banker_total = (draws[1] + draws[3]) % 10

        player_natural = player_total in (8, 9)
        banker_natural = banker_total in (8, 9)
        player_third = None
        banker_third = None
        if not (player_natural or banker_natural):
            if player_total <= 5:
                player_third = np.random.choice(
                    np.array([val for val, cnt in temp_counts.items() for _ in range(cnt)]), 
                    replace=False
                )
                temp_counts[player_third] -= 1
                player_total = (player_total + player_third) % 10
            if banker_total <= 5:
                draw_flag = False
                if player_third is None:
                    draw_flag = True
                else:
                    pt = player_third % 10
                    if banker_total <= 2: draw_flag = True
                    elif banker_total == 3 and pt != 8: draw_flag = True
                    elif banker_total == 4 and 2 <= pt <= 7: draw_flag = True
                    elif banker_total == 5 and 4 <= pt <= 7: draw_flag = True
                    elif banker_total == 6 and pt in (6, 7): draw_flag = True
                if draw_flag:
                    banker_third = np.random.choice(
                        np.array([val for val, cnt in temp_counts.items() for _ in range(cnt)]), 
                        replace=False
                    )
                    temp_counts[banker_third] -= 1
                    banker_total = (banker_total + banker_third) % 10

        if player_total > banker_total: return 1
        elif player_total < banker_total: return 0
        else: return 2

    def predict(self, sims_per_particle: int = None, rounds_seen: int | None = None) -> np.ndarray:
        """
        若 sims_per_particle 明確傳入：直接用（與既有 server 呼叫相容）
        否則：依 rounds_seen 自動吃 EARLY_/MID_/LATE_PF_PRED_SIMS，最後退回 PF_PRED_SIMS（預設 5）
        """
        if sims_per_particle is None:
            try:
                sims_per_particle = int(os.getenv("PF_PRED_SIMS", "5"))
            except:
                sims_per_particle = 5
            prefix = _stage_prefix(rounds_seen)
            if prefix:
                sp = os.getenv(prefix + "PF_PRED_SIMS")
                if sp not in (None, ""):
                    try: sims_per_particle = int(float(sp))
                    except: pass

        total_prob = np.zeros(3, dtype=np.float64)
        for i, counts in enumerate(self.particles):
            outcomes = [self._simulate_round(counts) for _ in range(int(sims_per_particle))]
            outcomes = np.array(outcomes, dtype=np.int8)
            total_prob[0] += self.weights[i] * (outcomes == 0).mean()
            total_prob[1] += self.weights[i] * (outcomes == 1).mean()
            total_prob[2] += self.weights[i] * (outcomes == 2).mean()
        total = total_prob.sum()
        if total > 0:
            total_prob /= total
        return total_prob.astype(np.float32)

    def update_outcome(self, outcome: int):
        if self.history_mode == 0:
            return
        if outcome == 2 and bool(int(os.getenv('SKIP_TIE_UPD', '0'))):
            return
        N = self.n_particles
        new_weights = np.zeros(N, dtype=np.float64)
        for i, counts in enumerate(self.particles):
            outcomes = [self._simulate_round(counts) for _ in range(self.sims_lik)]
            match_count = sum(1 for o in outcomes if o == outcome)
            likelihood = (match_count + self.dirichlet_eps) / (self.sims_lik + 3 * self.dirichlet_eps)
            new_weights[i] = self.weights[i] * likelihood
        s = new_weights.sum()
        if s == 0: new_weights[:] = 1.0 / N
        else: new_weights /= s
        self.weights = new_weights
        ess = 1.0 / np.sum(self.weights**2)
        if ess < self.resample_thr * N:
            idx = np.random.choice(np.arange(N), size=N, p=self.weights)
            self.particles = [self.particles[j].copy() for j in idx]
            self.weights.fill(1.0 / N)
