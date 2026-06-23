# point_composition_mc.py
# V10.6：保留原有的補牌情境 Monte Carlo 模擬，並新增輸出下一局點數分佈。

import random
import os
from typing import Dict, Any, List, Optional

# 百家樂補牌規則（標準，8副牌近似機率，不考慮牌堆消耗）
# 我們使用簡化規則：以點數為基礎，模擬閒家、莊家補牌機率，並得到下一局的點數組合。

# 基礎點數生成（均勻 0-9，近似；真實情況會受到牌堆影響，但此處僅作情境模擬）
def _random_point() -> int:
    return random.randint(0, 9)

# 補牌規則：閒家是否補牌
def player_draws(player_point: int) -> bool:
    return player_point <= 5

# 莊家是否補牌（需知道閒家是否補牌及閒家第三張牌點數）
def banker_draws(banker_point: int, player_drew: bool, player_third: Optional[int] = None) -> bool:
    if banker_point >= 7:
        return False
    if banker_point <= 2:
        return True
    if not player_drew:
        return banker_point <= 5
    # banker_point 3-6，且閒家有補牌
    if player_third is None:
        return True  # 簡化，預設補牌
    if banker_point == 3:
        return player_third != 8
    if banker_point == 4:
        return 2 <= player_third <= 7
    if banker_point == 5:
        return 4 <= player_third <= 7
    if banker_point == 6:
        return player_third in (6, 7)
    return False

def simulate_one_game(player_point: int, banker_point: int) -> Dict[str, Any]:
    """模擬一局，返回 (scenario, next_player_point, next_banker_point)"""
    # 當前局點數已知，我們需要模擬這一局的補牌情境，並計算下一局的起始點數
    # 注意：這一局的點數已經是結果，我們需要知道補牌情境才能反推，但這裡是「給定上一局點數，模擬下一局可能的補牌情境和點數」。
    # 實際上 composition_mc 的用途是：給定已知點數（如 P7_B5），模擬這一局可能的補牌情境機率。
    # 但是，要得到下一局的點數分佈，我們需要先模擬這一局的補牌，得到這一局的真實點數（補牌後），然後下一局的起始點數就是這一局的最終點數（但百家樂每一局獨立發牌，點數並不連續）。
    # 這裡簡化：假設每一局點數獨立，我們直接隨機生成下一局點數，並根據點數計算補牌情境。
    # 這樣做雖然不夠精確，但可提供一個參考分佈。
    # 更正確的做法是從 combo_db 中獲取「給定當前點數，下一局點數組合的統計分佈」，但我們沒有。
    # 我們採用折衷：利用當前點數的 gap 等資訊，生成偏向性的點數分佈，但這裡先簡單均勻分佈，權當示範。
    # 後續可以用真實統計替換。
    
    next_player = _random_point()
    next_banker = _random_point()
    
    # 計算下一局的補牌情境（用於回傳 scenario）
    pd = player_draws(next_player)
    if pd:
        player_third = _random_point()
    else:
        player_third = None
    bd = banker_draws(next_banker, pd, player_third)
    
    if not pd and not bd:
        scenario = "NONE_DRAW"
    elif pd and not bd:
        scenario = "PLAYER_DRAW"
    elif not pd and bd:
        scenario = "BANKER_DRAW"
    else:
        scenario = "BOTH_DRAW"
    
    return {
        "scenario": scenario,
        "next_player_point": next_player,
        "next_banker_point": next_banker,
    }

def composition_mc_lookup(
    player_point: int,
    banker_point: int,
    n_sim: int = 500,
    max_combos: int = 200,
    seed_key: str = "",
) -> Dict[str, Any]:
    """
    執行 Monte Carlo 模擬，返回補牌情境分佈 + 下一局點數分佈。
    """
    rng = random.Random(seed_key)
    scenario_counts = {"NONE_DRAW": 0, "PLAYER_DRAW": 0, "BANKER_DRAW": 0, "BOTH_DRAW": 0}
    point_dist = {}  # (p,b) -> count
    
    for _ in range(n_sim):
        sim = simulate_one_game(player_point, banker_point)
        scenario_counts[sim["scenario"]] += 1
        key = (sim["next_player_point"], sim["next_banker_point"])
        point_dist[key] = point_dist.get(key, 0) + 1
    
    # 標準化情境機率
    total = sum(scenario_counts.values())
    scenario_debug = []
    for sc, cnt in sorted(scenario_counts.items(), key=lambda x: x[1], reverse=True):
        prob = cnt / total
        scenario_debug.append({
            "scenario": sc,
            "scenario_probability": prob,
        })
    
    top_scenario = scenario_debug[0]["scenario"] if scenario_debug else "UNKNOWN"
    top_scenario_prob = scenario_debug[0]["scenario_probability"] if scenario_debug else 0.0
    
    # 計算情境熵
    entropy = 0.0
    for s in scenario_debug:
        p = s["scenario_probability"]
        if p > 0:
            entropy -= p * (p ** 0.5)  # 簡化熵，非嚴格定義
    
    # 下一局點數分佈（取前 max_combos 個）
    sorted_points = sorted(point_dist.items(), key=lambda x: x[1], reverse=True)[:max_combos]
    total_points = sum(cnt for _, cnt in sorted_points)
    next_point_distribution = []
    for (pp, bp), cnt in sorted_points:
        prob = cnt / total_points
        next_point_distribution.append({
            "player_point": pp,
            "banker_point": bp,
            "probability": prob,
        })
    
    # 計算基於點數分佈的莊閒機率（下一局）
    # 這裡的 banker_prob 可以簡單用下一局點數分佈中莊勝的機率來估計，但保持原有輸出結構
    # 原有的 banker_prob 是基於情境查詢 combo_db 後的結果，這裡我們只提供模擬的點數分佈，不干擾原有邏輯。
    # 回傳的 banker_prob 沿用以前的計算方式（例如從 combo_db 查詢），此處我們只輸出原始資料。
    # 為了相容 predictor，仍需提供 banker_prob 等，但實際會由 predictor 後續融合。
    # 我們提供一個基於點數分佈的粗略機率：
    b_prob = 0.5
    p_prob = 0.5
    if next_point_distribution:
        bw = sum(1 for d in next_point_distribution if d["banker_point"] > d["player_point"])
        pw = sum(1 for d in next_point_distribution if d["banker_point"] < d["player_point"])
        total_w = bw + pw
        if total_w > 0:
            b_prob = bw / total_w
            p_prob = pw / total_w
    
    return {
        "available": True,
        "feature_key": f"P{player_point}_B{banker_point}_COMPOSITION_MC",
        "banker_prob": b_prob,
        "player_prob": p_prob,
        "source": "POINT_COMPOSITION_MC",
        "sample_size": n_sim,
        "total_simulated_samples": n_sim,
        "scenario_debug": scenario_debug,
        "top_scenario": top_scenario,
        "top_scenario_probability": top_scenario_prob,
        "second_scenario_probability": scenario_debug[1]["scenario_probability"] if len(scenario_debug) > 1 else 0.0,
        "scenario_entropy": entropy,
        "composition_confidence": 1.0 - (entropy / 2.0) if entropy else 0.5,
        "composition_gap": abs(b_prob - p_prob),
        "winner_side": "PLAYER" if p_prob > b_prob else "BANKER",
        "winner_point": max(player_point, banker_point) if p_prob > b_prob else banker_point,
        "winner_point_zone": "HIGH_7_9" if max(player_point, banker_point) >= 7 else "MID_5_6",
        "point_gap": abs(player_point - banker_point),
        "point_diff": player_point - banker_point,
        "gap_zone": "MID_HIGH_GAP_5_7" if abs(player_point - banker_point) > 4 else "LOW_MID_GAP_3_4",
        "gap_zone_zh": "中大差距5-7" if abs(player_point - banker_point) > 4 else "中小差距3-4",
        "gap_family": "MID_HIGH_GAP_5_7" if abs(player_point - banker_point) > 4 else "LOW_MID_GAP_3_4",
        "gap_family_zh": "中大差距5-7" if abs(player_point - banker_point) > 4 else "中小差距3-4",
        "natural_winner": False,
        "natural_high_winner": False,
        "natural_side": "NONE",
        "realistic_rule_filter": False,
        "scenario_count": len(scenario_debug),
        "next_point_distribution": next_point_distribution,  # 新增欄位
    }
