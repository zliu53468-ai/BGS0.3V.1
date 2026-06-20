# -*- coding: utf-8 -*-
"""
point_composition_mc.py - V9.7 補牌情境 Monte Carlo 強化版

設計目標：
- 使用者仍然只輸入「閒點 + 莊點」，例如 65。
- 不吃用戶真實路單歷史，不需要暖機。
- 保留原本四種補牌情境：
  NONE_DRAW / PLAYER_DRAW / BANKER_DRAW / BOTH_DRAW
- 升級重點：
  1. 加入百家樂真實補牌規則過濾，避免不可能的補牌情境亂進來。
  2. scenario_debug 變得更可靠，讓 combo_db.py 更容易對到正確條件資料庫。
  3. 新增補牌情境信心、情境熵、贏方點數區間等欄位，讓 predictor.py 可以更細緻融合。
  4. 保留原本輸出欄位，不破壞原先邏輯。
"""

import itertools
import math
import os
import random
from typing import Dict, Any, List, Tuple, Optional

BASE_BANKER_NO_TIE = 0.5000

CARD_VALUES = {
    "A": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    "6": 6, "7": 7, "8": 8, "9": 9, "T": 0,
}

# 10/J/Q/K 都算 0，合併為 T，等同 16 張。
CARD_WEIGHTS = {
    "A": 4, "2": 4, "3": 4, "4": 4, "5": 4,
    "6": 4, "7": 4, "8": 4, "9": 4, "T": 16,
}

SCENARIO_ZH = {
    "NONE_DRAW": "雙方不補",
    "PLAYER_DRAW": "閒補莊不補",
    "BANKER_DRAW": "莊補閒不補",
    "BOTH_DRAW": "莊閒皆補",
}


def env_bool(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return float(default)


def env_int(name: str, default: str) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return int(default)


# 是否套用真實百家樂補牌規則過濾。
COMPOSITION_USE_REALISTIC_DRAW_RULE_FILTER = env_bool("COMPOSITION_USE_REALISTIC_DRAW_RULE_FILTER", "1")

# 方向強度：仍尊重當前點數，但不讓補牌MC硬拉太滿。
COMPOSITION_POINT_DIRECTION_STRENGTH = env_float("COMPOSITION_POINT_DIRECTION_STRENGTH", "0.030")

# 補牌情境信心越高，predictor 可提高 composition_mc 權重。
COMPOSITION_CONFIDENCE_GAP_SCALE = env_float("COMPOSITION_CONFIDENCE_GAP_SCALE", "0.22")

# MC 上限，避免 Render 過慢。
COMPOSITION_MC_MAX_SIM_CAP = env_int("COMPOSITION_MC_MAX_SIM_CAP", "5000")
COMPOSITION_MC_MAX_COMBO_CAP = env_int("COMPOSITION_MC_MAX_COMBO_CAP", "800")

# 預設情境先驗。可以用 env 微調。
DEFAULT_SCENARIO_WEIGHTS = {
    "NONE_DRAW": env_float("COMPOSITION_PRIOR_NONE_DRAW", "0.20"),
    "PLAYER_DRAW": env_float("COMPOSITION_PRIOR_PLAYER_DRAW", "0.25"),
    "BANKER_DRAW": env_float("COMPOSITION_PRIOR_BANKER_DRAW", "0.25"),
    "BOTH_DRAW": env_float("COMPOSITION_PRIOR_BOTH_DRAW", "0.30"),
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
    return int(w)


def state_weight(cards: Tuple[str, ...]) -> int:
    return combo_weight(cards)


def banker_should_draw(banker_two_total: int, player_third_value: Optional[int] = None, player_drew: bool = False) -> bool:
    """
    百家樂莊家補牌規則。
    """
    b = int(banker_two_total)
    if not player_drew:
        return b <= 5

    pt = int(player_third_value)
    if b <= 2:
        return True
    if b == 3:
        return pt != 8
    if b == 4:
        return 2 <= pt <= 7
    if b == 5:
        return 4 <= pt <= 7
    if b == 6:
        return pt in (6, 7)
    return False


def valid_current_state(player_cards: Tuple[str, ...], banker_cards: Tuple[str, ...]) -> bool:
    """
    判斷這一組最終牌型是否符合百家樂真實補牌規則。
    注意：這裡只是在「已知最終點數與補牌情境」下過濾不可能的牌型。
    """
    p2 = hand_total(list(player_cards[:2]))
    b2 = hand_total(list(banker_cards[:2]))
    p_drew = len(player_cards) == 3
    b_drew = len(banker_cards) == 3

    # 任一方前兩張天牌 8/9，雙方不補。
    if p2 in (8, 9) or b2 in (8, 9):
        return (not p_drew) and (not b_drew)

    expected_p_draw = p2 <= 5
    if p_drew != expected_p_draw:
        return False

    p_third_value = CARD_VALUES[player_cards[2]] if p_drew else None
    expected_b_draw = banker_should_draw(b2, p_third_value, p_drew)
    return b_drew == expected_b_draw


def possible_hands_for_total(total: int, card_count: int, max_combos: int = 220) -> List[Tuple[str, ...]]:
    """
    反推某個最終點數可能由哪些牌組成。
    card_count = 2 代表未補牌；card_count = 3 代表有補牌。
    """
    total = int(total) % 10
    card_count = int(card_count)
    max_combos = max(20, int(max_combos))

    cards = list(CARD_VALUES.keys())
    combos: List[Tuple[str, ...]] = []

    # 先收集全部，排序後再截斷，避免 itertools 前段順序造成偏差。
    for combo in itertools.product(cards, repeat=card_count):
        if hand_total(list(combo)) == total:
            combos.append(combo)

    # 高權重牌型優先，但保留 deterministic。
    combos.sort(key=lambda c: combo_weight(c), reverse=True)
    return combos[:max_combos]


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


def winner_point_zone(player_point: int, banker_point: int) -> Dict[str, Any]:
    if player_point > banker_point:
        side = "PLAYER"
        point = int(player_point)
    elif banker_point > player_point:
        side = "BANKER"
        point = int(banker_point)
    else:
        side = "TIE"
        point = int(player_point)

    if point in (7, 8, 9):
        zone = "HIGH_7_9"
    elif point in (1, 2, 3, 4):
        zone = "LOW_1_4"
    elif point in (5, 6):
        zone = "MID_5_6"
    else:
        zone = "ZERO"

    return {
        "winner_side": side,
        "winner_point": point,
        "winner_point_zone": zone,
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


def scenario_combo_info(
    player_point: int,
    banker_point: int,
    player_draw: bool,
    banker_draw: bool,
    max_combos: int = 220,
) -> Dict[str, Any]:
    player_count = 3 if player_draw else 2
    banker_count = 3 if banker_draw else 2
    scenario = scenario_name(player_draw, banker_draw)

    player_hands = possible_hands_for_total(player_point, player_count, max_combos)
    banker_hands = possible_hands_for_total(banker_point, banker_count, max_combos)

    raw_player_score = weighted_combo_score(player_hands)
    raw_banker_score = weighted_combo_score(banker_hands)

    valid_state_count = 0
    valid_state_weight = 0.0
    banker_state_score = 0.0
    player_state_score = 0.0
    example_states: List[Dict[str, Any]] = []

    for p_cards in player_hands:
        pw = combo_weight(p_cards)
        for b_cards in banker_hands:
            if COMPOSITION_USE_REALISTIC_DRAW_RULE_FILTER and not valid_current_state(p_cards, b_cards):
                continue
            bw = combo_weight(b_cards)
            sw = float(pw * bw)
            valid_state_count += 1
            valid_state_weight += sw
            player_state_score += float(pw) * sw
            banker_state_score += float(bw) * sw
            if len(example_states) < 5:
                example_states.append({
                    "player_cards": "".join(p_cards),
                    "banker_cards": "".join(b_cards),
                    "state_weight": int(sw),
                })

    if valid_state_count <= 0:
        # 若真實規則過濾後沒有資料，回傳 0，讓該 scenario 自然降權。
        valid_state_weight = 0.0
        banker_form_rate = 0.5
        player_form_rate = 0.5
    else:
        combo_total = banker_state_score + player_state_score
        if combo_total <= 0:
            banker_form_rate = 0.5
            player_form_rate = 0.5
        else:
            banker_form_rate = banker_state_score / combo_total
            player_form_rate = player_state_score / combo_total

    # 尊重當局點數方向，但不拉太滿。
    point_strength = COMPOSITION_POINT_DIRECTION_STRENGTH
    if banker_point > player_point:
        banker_form_rate += point_strength
        player_form_rate -= point_strength
    elif player_point > banker_point:
        banker_form_rate -= point_strength
        player_form_rate += point_strength

    banker_form_rate = clamp(banker_form_rate, 0.40, 0.60)
    player_form_rate = 1.0 - banker_form_rate

    wz = winner_point_zone(player_point, banker_point)

    return {
        "scenario": scenario,
        "scenario_zh": SCENARIO_ZH.get(scenario, scenario),
        "player_draw": bool(player_draw),
        "banker_draw": bool(banker_draw),
        "player_count": player_count,
        "banker_count": banker_count,
        "player_combo_count": len(player_hands),
        "banker_combo_count": len(banker_hands),
        "player_combo_score": raw_player_score,
        "banker_combo_score": raw_banker_score,
        "valid_state_count": int(valid_state_count),
        "valid_state_weight": float(valid_state_weight),
        "banker_state_score": float(banker_state_score),
        "player_state_score": float(player_state_score),
        "banker_form_rate": banker_form_rate,
        "player_form_rate": player_form_rate,
        "realistic_rule_filter": bool(COMPOSITION_USE_REALISTIC_DRAW_RULE_FILTER),
        "example_states": example_states,
        **wz,
    }


def scenario_entropy(probs: List[float]) -> float:
    probs = [p for p in probs if p > 0]
    if len(probs) <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in probs)
    return h / math.log(len(probs))


def composition_mc_lookup(
    player_point: int,
    banker_point: int,
    n_sim: int = 500,
    max_combos: int = 220,
    seed_key: str = "",
    scenario_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    根據輸入點數，反推四種補牌情境的可能組合。

    回傳重點：
    - scenario_debug：每種補牌情境與權重，供 combo_db.py 查條件資料庫。
    - banker_prob/player_prob：補牌組成層自身的輔助概率。
    - composition_confidence：補牌情境明確度，供 predictor.py 動態調整 composition_mc 權重。
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
            "composition_confidence": 0.0,
            "scenario_entropy": 1.0,
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
            "composition_confidence": 0.0,
            "scenario_entropy": 1.0,
        }

    n_sim = max(80, min(int(n_sim or 500), int(COMPOSITION_MC_MAX_SIM_CAP)))
    max_combos = max(40, min(int(max_combos or 220), int(COMPOSITION_MC_MAX_COMBO_CAP)))

    if not scenario_weights:
        scenario_weights = dict(DEFAULT_SCENARIO_WEIGHTS)

    scenarios = [
        (False, False, float(scenario_weights.get("NONE_DRAW", scenario_weights.get("none_draw", 0.20)))),
        (True, False, float(scenario_weights.get("PLAYER_DRAW", scenario_weights.get("player_draw", 0.25)))),
        (False, True, float(scenario_weights.get("BANKER_DRAW", scenario_weights.get("banker_draw", 0.25)))),
        (True, True, float(scenario_weights.get("BOTH_DRAW", scenario_weights.get("both_draw", 0.30)))),
    ]

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

        # 真實規則過濾後沒有可行 state，直接略過或極低權重。
        valid_weight = float(info.get("valid_state_weight", 0.0) or 0.0)
        valid_count = int(info.get("valid_state_count", 0) or 0)
        if valid_weight <= 0 or valid_count <= 0:
            continue

        # 情境可信度：基礎權重 * 真實可行 state 權重。
        scenario_strength = float(base_weight) * valid_weight
        raw_scenario_strength_total += scenario_strength

        info.update({
            "scenario_weight": float(base_weight),
            "scenario_strength": float(scenario_strength),
            "scenario_probability": 0.0,
        })
        scenario_debug.append(info)

    if raw_scenario_strength_total <= 0 or not scenario_debug:
        banker_prob = BASE_BANKER_NO_TIE
        player_prob = 1.0 - banker_prob
        available = False
        top_scenario = "UNKNOWN"
        top_prob = 0.0
        second_prob = 0.0
        entropy = 1.0
        comp_conf = 0.0
    else:
        for info in scenario_debug:
            info["scenario_probability"] = float(info.get("scenario_strength", 0.0)) / raw_scenario_strength_total

        scenario_debug.sort(key=lambda x: x.get("scenario_probability", 0.0), reverse=True)
        top_scenario = str(scenario_debug[0].get("scenario", "UNKNOWN"))
        top_prob = float(scenario_debug[0].get("scenario_probability", 0.0) or 0.0)
        second_prob = float(scenario_debug[1].get("scenario_probability", 0.0) or 0.0) if len(scenario_debug) >= 2 else 0.0

        probs = [float(x.get("scenario_probability", 0.0) or 0.0) for x in scenario_debug]
        entropy = scenario_entropy(probs)

        # 情境信心：top 與 second 拉開越多、熵越低，信心越高。
        comp_conf = (top_prob - second_prob) / max(COMPOSITION_CONFIDENCE_GAP_SCALE, 0.0001)
        comp_conf = clamp(comp_conf, 0.0, 1.0)
        comp_conf = clamp((comp_conf * 0.70) + ((1.0 - entropy) * 0.30), 0.0, 1.0)

        banker_score_total = 0.0
        player_score_total = 0.0
        used_weight = 0.0

        for info in scenario_debug:
            w = float(info.get("scenario_probability", 0.0) or 0.0)
            banker_score_total += float(info.get("banker_form_rate", BASE_BANKER_NO_TIE)) * w
            player_score_total += float(info.get("player_form_rate", 1.0 - BASE_BANKER_NO_TIE)) * w
            used_weight += w

        banker_prob = banker_score_total / used_weight if used_weight > 0 else BASE_BANKER_NO_TIE
        player_prob = player_score_total / used_weight if used_weight > 0 else 1.0 - banker_prob
        banker_prob, player_prob = normalize_pair(banker_prob, player_prob)
        available = True

    # 抽樣驗證此補牌組成層的穩定度。
    rng = random.Random(f"point_composition_mc_v9_7:{player_point}:{banker_point}:{seed_key}:{top_scenario}")
    banker_wins = 0
    player_wins = 0

    # 信心越高，noise 越小；信心越低，讓 MC 保守一點。
    noise = 0.010 - 0.004 * float(comp_conf)
    noise = clamp(noise, 0.004, 0.012)

    for _ in range(n_sim):
        b = clamp(float(banker_prob) + rng.uniform(-noise, noise), 0.38, 0.62)
        if rng.random() < b:
            banker_wins += 1
        else:
            player_wins += 1

    total = banker_wins + player_wins
    if total > 0:
        banker_prob = banker_wins / total
        player_prob = player_wins / total
        banker_prob, player_prob = normalize_pair(banker_prob, player_prob)

    gap = abs(banker_prob - player_prob)
    wz = winner_point_zone(player_point, banker_point)

    return {
        "available": bool(available),
        "feature_key": f"P{player_point}_B{banker_point}_COMPOSITION_MC",
        "banker_prob": banker_prob,
        "player_prob": player_prob,
        "source": "POINT_COMPOSITION_MC_REALISTIC_DRAW_SCENARIO_V9_7" if available else "POINT_COMPOSITION_MC_FALLBACK_V9_7",
        "sample_size": int(n_sim),
        "total_simulated_samples": int(n_sim),
        "scenario_debug": scenario_debug,
        "top_scenario": top_scenario,
        "top_scenario_probability": float(top_prob),
        "second_scenario_probability": float(second_prob),
        "scenario_entropy": float(entropy),
        "composition_confidence": float(comp_conf),
        "composition_gap": float(gap),
        "scenario_count": len(scenario_debug),
        "realistic_rule_filter": bool(COMPOSITION_USE_REALISTIC_DRAW_RULE_FILTER),
        "banker_mc_wins": int(banker_wins),
        "player_mc_wins": int(player_wins),
        **wz,
    }
