### bgs/pfilter.py
```python
import os
import numpy as np
from typing import Any, List, Tuple

# Environment-configurable parameters
PF_RESAMPLE = float(os.getenv("PF_RESAMPLE", "0.3"))  # 重採樣門檻
PF_DIR_EPS   = float(os.getenv("PF_DIR_EPS",   "0.1"))  # Dirichlet 平滑參數
PF_N         = int(os.getenv("PF_N",             "120"))  # 粒子數量

class ParticleFilter:
    def __init__(self, n: int = PF_N):
        self.n = n
        self.particles = self.sample_from_prior_batch()
        self.weights = np.ones(n) / n
        self.round_count = 0
        self.last_outcome = None
        self.streak_count = 0

    def sample_from_prior(self) -> int:
        # 假設均勻先驗：0=莊,1=閒,2=和
        return np.random.choice([0,1,2])

    def sample_from_prior_batch(self) -> np.ndarray:
        return np.random.choice([0,1,2], size=self.n)

    def _normalize_weights(self):
        total = np.sum(self.weights)
        if total > 0:
            self.weights /= total

    def _resample(self):
        indices = np.random.choice(self.n, p=self.weights, size=self.n)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.n) / self.n

    def update(self, outcome: int):
        # 更新連勝計數
        if outcome == self.last_outcome and outcome in (0,1):
            self.streak_count += 1
        else:
            self.streak_count = 0
        self.last_outcome = outcome

        # 權重更新 (示意)
        likelihood = np.where(self.particles == outcome, 1.0, PF_DIR_EPS)
        self.weights *= likelihood
        self._normalize_weights()

        # 計算 ESS
        ess = 1.0 / np.sum(self.weights**2)
        if ess / self.n < PF_RESAMPLE:
            self._resample()

        # 隨機重啟機制：每隔 10 局，重置 20% 粒子
        self.round_count += 1
        if self.round_count % 10 == 0:
            num_rej = int(self.n * 0.2)
            idx = np.random.choice(self.n, num_rej, replace=False)
            for i in idx:
                self.particles[i] = self.sample_from_prior()
            self._normalize_weights()

    def predict(self) -> Tuple[float, float, float]:
        # 根據權重統計概率
        probs = []
        for outcome in [0,1,2]:
            mask = self.particles == outcome
            probs.append(np.sum(self.weights[mask]))
        return tuple(probs)
```  

### deplete.py
```python
import numpy as np
from dataclasses import dataclass

# Monte Carlo deplete 模組示意
@dataclass
class DepleteMC:
    counts: np.ndarray
    rng: np.random.Generator
    trials: int = 1000  # 增加到 1000

    def simulate(self) -> None:
        # 直接示意：隨機取樣 outcomes
        outcomes = np.random.choice([0,1,2], size=self.trials)
        success = np.sum(outcomes != -1) or 1
        exp_usage = np.zeros_like(self.counts, dtype=float)
        # ... 填入真實模擬邏輯 ...
        # 保留更多隨機性，只扣 50%
        usage = (exp_usage / success) * 0.5
        self.counts = np.maximum(0, self.counts - usage)
```  

### server.py
```python
import os
import math
from flask import Flask, request, jsonify
from bgs.pfilter import ParticleFilter
from deplete import DepleteMC

app = Flask(__name__)

# 平滑與趨勢懲罰設定
SMOOTH_ALPHA = float(os.getenv("SMOOTH_ALPHA", "0.4"))
THEO_ALPHA   = float(os.getenv("THEO_ALPHA",   "0.6"))
STREAK_THRESH = int(os.getenv("STREAK_THRESH", "2"))
STREAK_PENALTY = float(os.getenv("STREAK_PENALTY", "0.08"))
HOUSE_EDGE = float(os.getenv("HOUSE_EDGE", "0.010"))  # 1%

pf = ParticleFilter()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json or {}
    last_outcome = data.get('last_outcome')  # 0,1,2
    pf.update(last_outcome)

    # 粒子濾波器理論概率
    pred_probs = pf.predict()
    # 假設固定理論概率
    theo_probs = (0.4585, 0.4463, 0.0952)

    # 平滑混合
    smoothed = tuple(
        SMOOTH_ALPHA * p + THEO_ALPHA * t for p, t in zip(pred_probs, theo_probs)
    )

    # 計算 EV 並扣趨勢懲罰
    evB = smoothed[0] - HOUSE_EDGE
    evP = smoothed[1]
    if pf.last_outcome == 0 and pf.streak_count >= STREAK_THRESH:
        evB -= STREAK_PENALTY
    if pf.last_outcome == 1 and pf.streak_count >= STREAK_THRESH:
        evP -= STREAK_PENALTY

    # 決策
    if evB > evP and evB > 0:
        decision = 'BANKER'
    elif evP > evB and evP > 0:
        decision = 'PLAYER'
    else:
        decision = 'PASS'

    return jsonify({
        'pred_probs': pred_probs,
        'smoothed_probs': smoothed,
        'evB': evB,
        'evP': evP,
        'decision': decision,
        'streak_count': pf.streak_count
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
