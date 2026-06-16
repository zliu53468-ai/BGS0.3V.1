import itertools
import random
from typing import Dict, Any, List, Tuple, Optional

# ============================================================
# 點數組成 / 補牌可能性 Monte Carlo V9
# ============================================================
# 使用者仍然只輸入「閒點 + 莊點」，例如 65。
# 本模組反推四種點數形成情境：
# - NONE_DRAW：雙方都沒補牌
# - PLAYER_DRAW：閒補一張、莊未補
# - BANKER_DRAW：閒未補、莊補一張
# - BOTH_DRAW：雙方都有補牌
#
# 回傳 scenario_debug 會給 combo_db.py 使用，拿「點數 + 補牌情境」
# 去對應 300 萬組條件資料庫。
# ============================================================

BASE_BANKER_NO_TIE = 0.5068

CARD_VALUES = {
    "A": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    "6": 6, "7": 7, "8": 8, "9": 9, "T": 0,
}

# 10/J/Q/K 都算 0，合併為 T，等同 16 張。
CARD_WEIGHTS = {
    "A": 4, "2": 4, "3": 4, "4": 4, "5": 4,
    "6": 4, "7": 4, "8": 4, "9": 4, "T": 16,
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


def possible_hands_for_total(total: int, card_count: int, max_combos: int = 160) -> List[Tuple[str, ...]]:
    """
    反推某個點數可能由哪些牌組成。
    card_count = 2 代表未補牌；card_count = 3 代表有補牌。
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


def scenario_name(player_draw: bool, banker_draw: bool) -> str:
    if player_draw and banker_draw:
        return "BOTH_DRAW"
    if player_draw and not banker_draw:
        return "PLAYER_DRAW"
    if banker_draw and not player_draw:
        return "BANKER_DRAW"
    return "NONE_DRAW"


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
        "scenario": scenario_name(player_draw, banker_draw),
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

    回傳重點：
    - scenario_debug：每種補牌情境與權重，供 combo_db.py 查 300 萬條件資料庫。
    - banker_prob/player_prob：補牌組成層自身的輔助概率。
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
            "top_scenario": "UNKNOWN",
        }

    if not (0 <= player_point <= 9 and 0 <= banker_point <= 9):
        return {
            "available": False,
            "banker_prob": BASE_BANKER_NO_TIE,
            "player_prob": 1.0 - BASE_BANKER_NO_TIE,
            "source": "POINT_COMPOSITION_MC_POINT_OUT_OF_RANGE",
            "sample_size": 0,
            "scenario_debug": [],
            "top_scenario": "UNKNOWN",
        }

    n_sim = max(80, min(int(n_sim or 500), 2000))
    max_combos = max(40, min(int(max_combos or 160), 500))

    if not scenario_weights:
        scenario_weights = {
            "NONE_DRAW": 0.20,
            "PLAYER_DRAW": 0.25,
            "BANKER_DRAW": 0.25,
            "BOTH_DRAW": 0.30,
        }

    scenarios = [
        (False, False, float(scenario_weights.get("NONE_DRAW", scenario_weights.get("none_draw", 0.20)))),
        (True, False, float(scenario_weights.get("PLAYER_DRAW", scenario_weights.get("player_draw", 0.25)))),
        (False, True, float(scenario_weights.get("BANKER_DRAW", scenario_weights.get("banker_draw", 0.25)))),
        (True, True, float(scenario_weights.get("BOTH_DRAW", scenario_weights.get("both_draw", 0.30)))),
    ]

    banker_score_total = 0.0
    player_score_total = 0.0
    used_weight = 0.0
    scenario_debug: List[Dict[str, Any]] = []

    raw_scenario_strength_total = 0.0

    for player_draw, banker_draw, base_weight in scenarios:
        if base_weight <= 0:
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

        combo_total = p_score + b_score
        banker_form_rate = b_score / combo_total
        player_form_rate = p_score / combo_total

        # 尊重當局點數方向，但不拉太滿。
        if banker_point > player_point:
            banker_form_rate += 0.035
            player_form_rate -= 0.035
        elif player_point > banker_point:
            banker_form_rate -= 0.035
            player_form_rate += 0.035

        banker_form_rate = clamp(banker_form_rate, 0.40, 0.60)
        player_form_rate = 1.0 - banker_form_rate

        # 情境可信度：基礎權重 * 雙方組合形成度。
        scenario_strength = base_weight * combo_total
        raw_scenario_strength_total += scenario_strength

        banker_score_total += banker_form_rate * base_weight
        player_score_total += player_form_rate * base_weight
        used_weight += base_weight

        info.update({
            "scenario_weight": base_weight,
            "scenario_strength": scenario_strength,
            "scenario_probability": 0.0,  # 下方正規化後補上
            "banker_form_rate": banker_form_rate,
            "player_form_rate": player_form_rate,
        })
        scenario_debug.append(info)

    if used_weight <= 0 or not scenario_debug:
        banker_prob = BASE_BANKER_NO_TIE
        player_prob = 1.0 - banker_prob
        available = False
        top_scenario = "UNKNOWN"
    else:
        for info in scenario_debug:
            if raw_scenario_strength_total > 0:
                info["scenario_probability"] = float(info.get("scenario_strength", 0.0)) / raw_scenario_strength_total
            else:
                info["scenario_probability"] = float(info.get("scenario_weight", 0.0)) / used_weight

        scenario_debug.sort(key=lambda x: x.get("scenario_probability", 0.0), reverse=True)
        top_scenario = str(scenario_debug[0].get("scenario", "UNKNOWN"))

        banker_prob = banker_score_total / used_weight
        player_prob = player_score_total / used_weight
        banker_prob, player_prob = normalize_pair(banker_prob, player_prob)
        available = True

    # 抽樣驗證此補牌組成層的穩定度。
    rng = random.Random(f"point_composition_mc:{player_point}:{banker_point}:{seed_key}")
    banker_wins = 0
    player_wins = 0

    for _ in range(n_sim):
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
        "source": "POINT_COMPOSITION_MC_DRAW_SCENARIO_V9" if available else "POINT_COMPOSITION_MC_FALLBACK",
        "sample_size": int(n_sim),
        "total_simulated_samples": int(n_sim),
        "scenario_debug": scenario_debug,
        "top_scenario": top_scenario,
        "scenario_count": len(scenario_debug),
    }
