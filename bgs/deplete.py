### deplete.py
```python
import numpy as np
from dataclasses import dataclass
from typing import Optional

# 牌值桶：0..9；0 桶代表 10/J/Q/K = 0點
CARD_IDX = list(range(10))
TEN_BUCKET = 0

def init_counts(decks: int = 8) -> np.ndarray:
    counts = np.zeros(10, dtype=np.int32)
    counts[1:10] = 4 * decks
    counts[TEN_BUCKET] = 16 * decks
    return counts

def draw_card(counts: np.ndarray, rng: np.random.Generator) -> int:
    tot = int(counts.sum())
    if tot <= 0:
        raise RuntimeError("Shoe empty")
    r = int(rng.integers(0, tot))
    acc = 0
    for v in range(10):
        acc += int(counts[v])
        if r < acc:
            counts[v] -= 1
            return v
    counts[9] -= 1
    return 9

def points_add(a: int, b: int) -> int:
    return (a + b) % 10

def third_card_rule_player(p_sum: int) -> bool:
    return p_sum <= 5

def third_card_rule_banker(b_sum: int, p3: Optional[int]) -> bool:
    if b_sum <= 2:
        return True
    if b_sum == 3:
        return (p3 is None) or (p3 != 8)
    if b_sum == 4:
        return (p3 is not None) and (p3 in (2,3,4,5,6,7))
    if b_sum == 5:
        return (p3 is not None) and (p3 in (4,5,6,7))
    if b_sum == 6:
        return (p3 is not None) and (p3 in (6,7))
    return False

@dataclass
class DepleteMC:
    """蒙地卡羅牌靴耗損"""
    decks: int = 8
    seed: int = 42

    def __post_init__(self):
        self.rng = np.random.default_rng(int(self.seed))
        self.counts = init_counts(int(self.decks))
        self.initial_counts = self.counts.copy()
        self.cards_used = 0
        self.shoe_reset_threshold = int(0.75 * self.decks * 52)

    def reset_shoe(self, decks: Optional[int] = None):
        if decks is not None:
            self.decks = int(decks)
        self.counts = init_counts(self.decks)
        self.initial_counts = self.counts.copy()
        self.cards_used = 0

    def update_outcome(self, outcome: int, trials: int = 1000):
        """
        使用蒙地卡羅模擬來估計牌靴耗損，模擬次數提升到 1000，
        並將耗損僅扣除一半，保留隨機性。
        """
        if outcome not in (0,1,2):
            return
        exp_usage = np.zeros_like(self.counts, dtype=np.float64)
        success = 0
        for _ in range(trials):
            tmp = self.counts.copy()
            try:
                # 發牌流程
                P1 = draw_card(tmp, self.rng); P2 = draw_card(tmp, self.rng)
                B1 = draw_card(tmp, self.rng); B2 = draw_card(tmp, self.rng)
                p_sum = points_add(P1, P2); b_sum = points_add(B1, B2)
                P3 = None
                if not (p_sum in (8,9) or b_sum in (8,9)):
                    if third_card_rule_player(p_sum):
                        P3 = draw_card(tmp, self.rng)
                        p_sum = points_add(p_sum, P3)
                    if third_card_rule_banker(b_sum, P3):
                        B3 = draw_card(tmp, self.rng)
                        b_sum = points_add(b_sum, B3)
                if (p_sum > b_sum and outcome==1) or (b_sum > p_sum and outcome==0) or (p_sum==b_sum and outcome==2):
                    used = self.counts - tmp
                    if (used >= 0).all():
                        exp_usage += used
                        success += 1
                        self.cards_used += used.sum()
            except Exception:
                continue
        if success > 0:
            avg_usage = exp_usage / success
            # 扣除一半
            usage = (avg_usage * 0.5).astype(np.int32)
            self.counts = np.maximum(0, self.counts - usage)
        # 牌靴重置
        if self.cards_used >= self.shoe_reset_threshold:
            self.reset_shoe()
```

### server.py
```python
import os
import time
import json
import numpy as np
from flask import Flask, request, jsonify
from deplete import DepleteMC

# 版本
VERSION = "bgs-final-antitrend-2025-10-01"

# 平滑與趨勢懲罰參數
SMOOTH_ALPHA = 0.4
THEO_ALPHA   = 0.6
STREAK_THRESH = 2
STREAK_PENALTY = 0.08
HOUSE_EDGE = 0.010  # 1%

# 初始化 Flask
app = Flask(__name__)

# 初始化 DepleteMC
deplete_pf = DepleteMC(
    decks=int(os.getenv("DECKS", "8")),
    seed=int(os.getenv("SEED", "42"))
)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json or {}
    last_outcome = data.get('last_outcome')  # 0=莊,1=閒,2=和
    # 更新耗損模型
    deplete_pf.update_outcome(last_outcome)
    # 取得理論 & 耗損後機率
    mc_probs = deplete_pf.predict(sims=20000)
    # 假設固定理論概率
    theo_probs = np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)
    # 平滑混合
    smoothed = SMOOTH_ALPHA * mc_probs + THEO_ALPHA * theo_probs
    smoothed /= smoothed.sum()
    # 計算 EV 並扣趨勢懲罰
    evB = smoothed[0] - HOUSE_EDGE
    evP = smoothed[1]
    if last_outcome == 0 and data.get('streak_count', 0) >= STREAK_THRESH:
        evB -= STREAK_PENALTY
    if last_outcome == 1 and data.get('streak_count', 0) >= STREAK_THRESH:
        evP -= STREAK_PENALTY
    # 決策
    if evB > evP and evB > 0:
        decision = 'BANKER'
    elif evP > evB and evP > 0:
        decision = 'PLAYER'
    else:
        decision = 'PASS'
    return jsonify({
        'mc_probs': mc_probs.tolist(),
        'smoothed_probs': smoothed.tolist(),
        'evB': evB,
        'evP': evP,
        'decision': decision
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
```
