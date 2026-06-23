# point_composition_mc.py
# V10.6：使用真實點數差距資料庫模擬下一局點數，並輸出補牌情境與點數分佈

import random
import os
import json
from typing import Dict, Any, List, Optional, Tuple

# 嘗試載入 config 以讀取環境變數（若有）
try:
    import config
except Exception:
    config = None

# 環境變數
USE_NEXT_POINT_GAP_DB = os.getenv("USE_NEXT_POINT_GAP_DB", "1") == "1"
NEXT_POINT_GAP_DB_PATH = os.getenv("NEXT_POINT_GAP_DB_PATH", "data/next_point_gap_db.json")
COMPOSITION_MC_SIMULATIONS = int(os.getenv("COMPOSITION_MC_SIMULATIONS", "800"))
COMPOSITION_MC_MAX_COMBOS = int(os.getenv("COMPOSITION_MC_MAX_COMBOS", "300"))

def _load_gap_db() -> Dict[str, Dict[str, float]]:
    """載入點數差距資料庫"""
    if not USE_NEXT_POINT_GAP_DB:
        return {}
    path = NEXT_POINT_GAP_DB_PATH
    if not os.path.exists(path):
        # 嘗試從專案根目錄找
        path = os.path.join(os.getcwd(), path)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

# 快取資料庫（避免每次模擬都讀檔）
_gap_db_cache = None

def _get_gap_probabilities(player_point: int, banker_point: int) -> Optional[Dict[str, float]]:
    """取得給定點數的下一局差距機率分佈"""
    global _gap_db_cache
    if _gap_db_cache is None:
        _gap_db_cache = _load_gap_db()
    key = f"P{player_point}_B{banker_point}"
    return _gap_db_cache.get(key)

def _random_point_by_gap(gap: int, rng: random.Random) -> Tuple[int, int]:
    """根據差距隨機生成一組閒莊點數（0-9）"""
    # 隨機選擇閒點，再計算莊點 = 閒點 ± gap，確保在 0-9 範圍內
    player = rng.randint(0, 9)
    if gap == 0:
        banker = player
    else:
        # 隨機決定莊點比閒點大或小
        if rng.random() < 0.5:
            banker = player + gap
        else:
            banker = player - gap
    # 調整到 0-9 範圍（若超出則鏡像或重新隨機，簡單處理：重新選取）
    attempts = 0
    while not (0 <= banker <= 9):
        player = rng.randint(0, 9)
        if gap == 0:
            banker = player
        else:
            if rng.random() < 0.5:
                banker = player + gap
            else:
                banker = player - gap
        attempts += 1
        if attempts > 100:  # 安全機制
            banker = rng.randint(0, 9)
            break
    return player, banker

def _sample_gap_from_distribution(probs: Dict[str, float], rng: random.Random) -> int:
    """從機率分佈中抽樣一個 gap 值（0-9）"""
    items = []
    weights = []
    for k, v in probs.items():
        if k.startswith("gap_"):
            g = int(k.split("_")[1])
            items.append(g)
            weights.append(v)
    if not items:
        return rng.randint(0, 9)
    # 標準化（可能總和不為1）
    total = sum(weights)
    weights = [w/total for w in weights]
    return rng.choices(items, weights=weights, k=1)[0]

# 補牌相關函數（與先前一致）
def _player_draws(player_point: int) -> bool:
    return player_point <= 5

def _banker_draws(banker_point: int, player_drew: bool, player_third: Optional[int] = None) -> bool:
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

def _determine_scenario(player_point: int, banker_point: int, rng: random.Random) -> str:
    """給定點數，判斷補牌情境（模擬第三張牌）"""
    pd = _player_draws(player_point)
    player_third = None
    if pd:
        player_third = rng.randint(0, 9)  # 簡化第三張牌點數隨機
    bd = _banker_draws(banker_point, pd, player_third)
    
    if not pd and not bd:
        return "NONE_DRAW"
    elif pd and not bd:
        return "PLAYER_DRAW"
    elif not pd and bd:
        return "BANKER_DRAW"
    else:
        return "BOTH_DRAW"

def composition_mc_lookup(
    player_point: int,
    banker_point: int,
    n_sim: int = None,
    max_combos: int = None,
    seed_key: str = "",
) -> Dict[str, Any]:
    """
    執行 Monte Carlo 模擬，返回補牌情境分佈 + 下一局點數分佈。
    使用真實點數差距資料庫（若有）來生成下一局點數。
    """
    if n_sim is None:
        n_sim = COMPOSITION_MC_SIMULATIONS
    if max_combos is None:
        max_combos = COMPOSITION_MC_MAX_COMBOS
    
    rng = random.Random(seed_key)
    
    # 取得該點數的下一局差距機率
    gap_probs = _get_gap_probabilities(player_point, banker_point)
    
    scenario_counts = {"NONE_DRAW": 0, "PLAYER_DRAW": 0, "BANKER_DRAW": 0, "BOTH_DRAW": 0}
    point_dist = {}  # (p,b) -> count
    
    for _ in range(n_sim):
        # 決定下一局點數
        if gap_probs:
            gap = _sample_gap_from_distribution(gap_probs, rng)
            next_player, next_banker = _random_point_by_gap(gap, rng)
        else:
            # 沒有資料庫時，均勻隨機
            next_player = rng.randint(0, 9)
            next_banker = rng.randint(0, 9)
        
        # 記錄點數組合
        key = (next_player, next_banker)
        point_dist[key] = point_dist.get(key, 0) + 1
        
        # 判斷該點數組合的補牌情境
        scenario = _determine_scenario(next_player, next_banker, rng)
        scenario_counts[scenario] += 1
    
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
    
    # 情境熵（簡化）
    entropy = 0.0
    for s in scenario_debug:
        p = s["scenario_probability"]
        if p > 0:
            entropy -= p * (p ** 0.5)  # 非嚴格熵，僅供參考
    
    # 下一局點數分佈
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
    
    # 粗略莊閒機率（基於點數分佈中點數大小判斷）
    b_prob = 0.5
    p_prob = 0.5
    if next_point_distribution:
        bw = sum(1 for d in next_point_distribution if d["banker_point"] > d["player_point"])
        pw = sum(1 for d in next_point_distribution if d["banker_point"] < d["player_point"])
        total_w = bw + pw
        if total_w > 0:
            b_prob = bw / total_w
            p_prob = pw / total_w
    
    # 相容原有輸出欄位
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
        "next_point_distribution": next_point_distribution,  # 新增輸出
    }
