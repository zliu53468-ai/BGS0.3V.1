import itertools
import random
from typing import Dict, Any, List, Tuple, Optional

# ============================================================
# 點數組成 / 補牌可能性 Monte Carlo
# ============================================================
# 用途：
# 使用者仍然只輸入「閒點 + 莊點」，例如 65。
# 本模組會反推：
# - 閒點數可能由 2 張或 3 張牌組成
# - 莊點數可能由 2 張或 3 張牌組成
# - 模擬四種情境：雙方不補、閒補、莊補、雙方補
#
# 注意：
# 這不是完整牌靴追蹤，因為使用者沒有輸入實際牌面。
# 它是「點數組成可能性修正層」，適合當輔助權重，不建議權重過高。
# ============================================================

BASE_BANKER_NO_TIE = 0.5068

CARD_VALUES = {
    "A": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 0,
}

# 百家樂點數中 10/J/Q/K 都算 0，合併成 T，權重等同 16 張。
CARD_WEIGHTS = {
    "A": 4,
    "2": 4,
    "3": 4,
    "4": 4,
    "5": 4,
    "6": 4,
    "7": 4,
    "8": 4,
    "9": 4,
    "T": 16,
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def hand_total(cards: List[str]) -> int:
    return sum(CARD_VALUES[str(c)] for c in cards) % 10


def combo_weight(combo: Tuple[str, ...]) -> int:
    """用牌面權重估計該組合形成難度；T 比一般牌更常見。"""
    w = 1
    for c in combo:
        w *= int(CARD_WEIGHTS.get(str(c), 1))
    return w


def possible_hands_for_total(
    total: int,
    card_count: int,
    max_combos: int = 160,
) -> List[Tuple[str, ...]]:
    """
    反推某個點數可能由哪些牌組成。
    card_count = 2 代表沒補牌；card_count = 3 代表有補牌。
    """
    total = int(total) % 10
    card_count = int(card_count)
    max_combos = max(20, int(max_combos))

    cards = list(CARD_VALUES.keys())
    combos: List[Tuple[str, ...]] = []

    for combo in itertools.product(cards, repeat=card_count):
        if hand_total(list(combo)) == total:
            combos.append(combo)
            if len(combos) >= max_combos:
                break

    return combos


def weighted_combo_score(combos: List[Tuple[str, ...]]) -> float:
    if not combos:
        return 0.0
    return float(sum(combo_weight(c) for c in combos))


def scenario_combo_info(
    player_point: int,
    banker_point: int,
    player_draw: bool,
    banker_draw: bool,
    max_combos: int = 160,
) -> Dict[str, Any]:
    player_count = 3 if player_draw else 2
    banker_count = 3 if banker_draw else 2

    player_hands = possible_hands_for_total(player_point, player_count, max_combos)
    banker_hands = possible_hands_for_total(banker_point, banker_count, max_combos)

    player_score = weighted_combo_score(player_hands)
    banker_score = weighted_combo_score(banker_hands)

    return {
        "player_draw": bool(player_draw),
        "banker_draw": bool(banker_draw),
        "player_count": player_count,
        "banker_count": banker_count,
        "player_combo_count": len(player_hands),
        "banker_combo_count": len(banker_hands),
        "player_combo_score": player_score,
        "banker_combo_score": banker_score,
    }


def normalize_pair(banker_prob: float, player_prob: float) -> Tuple[float, float]:
    banker_prob = float(banker_prob)
    player_prob = float(player_prob)
    total = banker_prob + player_prob
    if total <= 0:
        banker_prob = BASE_BANKER_NO_TIE
        player_prob = 1.0 - banker_prob
    else:
        banker_prob /= total
        player_prob /= total
    banker_prob = clamp(banker_prob, 0.38, 0.62)
    player_prob = 1.0 - banker_prob
    return banker_prob, player_prob


def composition_mc_lookup(
    player_point: int,
    banker_point: int,
    n_sim: int = 500,
    max_combos: int = 160,
    seed_key: str = "",
    scenario_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    根據輸入點數，模擬四種補牌情境的可能組合。

    回傳格式會刻意設計成跟其他 layer 類似：
    - available
    - banker_prob
    - player_prob
    - source
    - sample_size
    """
    try:
        player_point = int(player_point)
        banker_point = int(banker_point)
    except Exception:
        return {
            "available": False,
            "banker_prob": BASE_BANKER_NO_TIE,
            "player_prob": 1.0 - BASE_BANKER_NO_TIE,
            "source": "POINT_COMPOSITION_MC_INVALID_POINT",
            "sample_size": 0,
            "scenario_debug": [],
        }

    if not (0 <= player_point <= 9 and 0 <= banker_point <= 9):
        return {
            "available": False,
            "banker_prob": BASE_BANKER_NO_TIE,
            "player_prob": 1.0 - BASE_BANKER_NO_TIE,
            "source": "POINT_COMPOSITION_MC_POINT_OUT_OF_RANGE",
            "sample_size": 0,
            "scenario_debug": [],
        }

    n_sim = max(80, min(int(n_sim or 500), 2000))
    max_combos = max(40, min(int(max_combos or 160), 500))

    # 四種情境：不補/補牌。權重不是絕對機率，只是簡化假設。
    # 後續可用環境變數或實測資料再校正。
    if not scenario_weights:
        scenario_weights = {
            "none_draw": 0.20,
            "player_draw": 0.25,
            "banker_draw": 0.25,
            "both_draw": 0.30,
        }

    scenarios = [
        (False, False, float(scenario_weights.get("none_draw", 0.20)), "none_draw"),
        (True, False, float(scenario_weights.get("player_draw", 0.25)), "player_draw"),
        (False, True, float(scenario_weights.get("banker_draw", 0.25)), "banker_draw"),
        (True, True, float(scenario_weights.get("both_draw", 0.30)), "both_draw"),
    ]

    banker_score_total = 0.0
    player_score_total = 0.0
    used_weight = 0.0
    scenario_debug: List[Dict[str, Any]] = []

    for player_draw, banker_draw, weight, name in scenarios:
        if weight <= 0:
            continue

        info = scenario_combo_info(
            player_point=player_point,
            banker_point=banker_point,
            player_draw=player_draw,
            banker_draw=banker_draw,
            max_combos=max_combos,
        )

        p_score = float(info["player_combo_score"])
        b_score = float(info["banker_combo_score"])

        if p_score <= 0 or b_score <= 0:
            continue

        # 組合形成度：哪邊在該情境下越容易形成該點數，就給哪邊一點修正。
        combo_total = p_score + b_score
        banker_form_rate = b_score / combo_total
        player_form_rate = p_score / combo_total

        # 當局點數方向修正：這層仍然尊重輸入點數強弱，但不拉太滿。
        if banker_point > player_point:
            banker_form_rate += 0.035
            player_form_rate -= 0.035
        elif player_point > banker_point:
            banker_form_rate -= 0.035
            player_form_rate += 0.035

        banker_form_rate = clamp(banker_form_rate, 0.40, 0.60)
        player_form_rate = 1.0 - banker_form_rate

        banker_score_total += banker_form_rate * weight
        player_score_total += player_form_rate * weight
        used_weight += weight

        info.update({
            "scenario": name,
            "scenario_weight": weight,
            "banker_form_rate": banker_form_rate,
            "player_form_rate": player_form_rate,
        })
        scenario_debug.append(info)

    if used_weight <= 0:
        banker_prob = BASE_BANKER_NO_TIE
        player_prob = 1.0 - banker_prob
        available = False
    else:
        banker_prob = banker_score_total / used_weight
        player_prob = player_score_total / used_weight
        banker_prob, player_prob = normalize_pair(banker_prob, player_prob)
        available = True

    # 抽樣只是讓輸出更像 MC 穩定度結果；不重新模擬完整牌靴。
    rng = random.Random(f"point_composition_mc:{player_point}:{banker_point}:{seed_key}")
    banker_wins = 0
    player_wins = 0

    for _ in range(n_sim):
        # 微幅擾動避免完全死板，但限制很小。
        b = clamp(banker_prob + rng.uniform(-0.008, 0.008), 0.38, 0.62)
        if rng.random() < b:
            banker_wins += 1
        else:
            player_wins += 1

    total = banker_wins + player_wins
    if total > 0:
        banker_prob = banker_wins / total
        player_prob = player_wins / total
        banker_prob, player_prob = normalize_pair(banker_prob, player_prob)

    return {
        "available": bool(available),
        "feature_key": f"P{player_point}_B{banker_point}_COMPOSITION_MC",
        "banker_prob": banker_prob,
        "player_prob": player_prob,
        "source": "POINT_COMPOSITION_MC_DRAW_SCENARIO_V1" if available else "POINT_COMPOSITION_MC_FALLBACK",
        "sample_size": int(n_sim),
        "total_simulated_samples": int(n_sim),
        "scenario_debug": scenario_debug,
    }
