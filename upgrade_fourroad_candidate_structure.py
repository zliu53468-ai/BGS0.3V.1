#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# predictor.py 四路候選模擬 + 結構判斷升級腳本
#
# 使用方式：
# 1. 把這個檔案放在 predictor.py 同一個資料夾。
# 2. 執行：python upgrade_fourroad_candidate_structure.py
# 3. 程式會自動備份原本 predictor.py，然後寫入四路候選模擬 + 齊整/有無/直落結構強化版。
#
# 重要保證：
# - 只替換四路下三路相關區塊：
#   1) Candidate Down-Road Simulation 環境變數區
#   2) 候選模擬 helper 區塊
#   3) _derived_road_score()
#   4) _fuhao_down3_vote()
# - 其他原先模型不動刀：
#   ML / DeepSeek / Walk-forward / Pattern Replay / Road Lifecycle / Adaptive Road Memory /
#   Road Rhythm / Long Anchor / Big Road / NGram / Markov 等都不會被整段替換。
#
# 強化重點：
# - 下三路不再用「紅=跟、藍=反」。
# - 改成先模擬下一口如果開莊 / 開閒，各自會讓大眼仔 / 小路 / 蟑螂路產生什麼紅藍結果。
# - 再加入真正下三路結構判斷：齊整、有無、直落、黏邊橫拖、新欄高度齊整。
# - 最後用「紅藍節奏分 + 結構分」去評估候選莊/閒哪個更合理。

from pathlib import Path
from datetime import datetime
import re
import sys


PREDICTOR_PATH = Path("predictor.py")


ENV_BLOCK = r'''
# Candidate Down-Road Simulation：下三路候選模擬參數
# 目的：不要把紅藍直接等於莊閒，而是先模擬下一口莊/閒各自會產生什麼紅藍，再反推哪邊更合理。
DERIVED_CANDIDATE_LOOKBACK = int(os.getenv("DERIVED_CANDIDATE_LOOKBACK", str(ROAD_ENGINE_DERIVED_LOOKBACK)))
DERIVED_CANDIDATE_MAX_EDGE = float(os.getenv("DERIVED_CANDIDATE_MAX_EDGE", "0.078"))
DERIVED_CANDIDATE_MIN_EDGE = float(os.getenv("DERIVED_CANDIDATE_MIN_EDGE", "0.010"))
DERIVED_COLOR_JUMP_RATE = float(os.getenv("DERIVED_COLOR_JUMP_RATE", "0.68"))
DERIVED_COLOR_STREAK_MIN = int(os.getenv("DERIVED_COLOR_STREAK_MIN", "3"))
DERIVED_COLOR_RATIO_GAP = float(os.getenv("DERIVED_COLOR_RATIO_GAP", "0.22"))
DERIVED_COLOR_NGRAM_MAX = int(os.getenv("DERIVED_COLOR_NGRAM_MAX", "5"))
FUHAO_DOWN3_MIN_DIFF = float(os.getenv("FUHAO_DOWN3_MIN_DIFF", "0.025"))

# Down-Road Structure：下三路齊整 / 有無 / 直落結構分
# 色彩節奏分：看目前紅藍是單跳、連紅/連藍、比例偏態、NGram。
# 結構分：看候選落點是否齊整、有無是否一致、是否直落、是否橫拖。
DERIVED_COLOR_SCORE_WEIGHT = float(os.getenv("DERIVED_COLOR_SCORE_WEIGHT", "0.62"))
DERIVED_STRUCTURE_SCORE_WEIGHT = float(os.getenv("DERIVED_STRUCTURE_SCORE_WEIGHT", "0.38"))
DERIVED_STRUCTURE_NEAT_BONUS = float(os.getenv("DERIVED_STRUCTURE_NEAT_BONUS", "0.055"))
DERIVED_STRUCTURE_MISMATCH_PENALTY = float(os.getenv("DERIVED_STRUCTURE_MISMATCH_PENALTY", "0.045"))
DERIVED_STRUCTURE_DROP_BONUS = float(os.getenv("DERIVED_STRUCTURE_DROP_BONUS", "0.035"))
DERIVED_STRUCTURE_SIDE_DRAG_PENALTY = float(os.getenv("DERIVED_STRUCTURE_SIDE_DRAG_PENALTY", "0.025"))
DERIVED_STRUCTURE_NEWCOL_BONUS = float(os.getenv("DERIVED_STRUCTURE_NEWCOL_BONUS", "0.025"))
DERIVED_STRUCTURE_MAX_EDGE = float(os.getenv("DERIVED_STRUCTURE_MAX_EDGE", "0.090"))
# End Candidate Down-Road Simulation
'''.strip()


HELPER_BLOCK = r'''
def _classify_bigroad_move(before_layout: Dict[str, Any], after_layout: Dict[str, Any], candidate: str) -> Dict[str, Any]:
    # 判斷候選莊/閒落在大路後屬於哪一種大路移動：
    # NEW_COLUMN=換邊開新欄；VERTICAL_DROP=同邊往下直落；SIDE_DRAG=到底/卡位往右橫拖。
    before_positions = before_layout.get("positions", []) or []
    after_positions = after_layout.get("positions", []) or []

    if not after_positions:
        return {"move_type": "NONE", "before": {}, "after": {}}

    after_pos = after_positions[-1]
    before_pos = before_positions[-1] if before_positions else {}

    if not before_pos:
        return {"move_type": "START", "before": before_pos, "after": after_pos}

    before_side = before_pos.get("side", "")
    before_row = int(before_pos.get("row", 0))
    before_col = int(before_pos.get("col", 0))
    after_row = int(after_pos.get("row", 0))
    after_col = int(after_pos.get("col", 0))

    if candidate != before_side:
        move_type = "NEW_COLUMN"
    elif after_col == before_col and after_row == before_row + 1:
        move_type = "VERTICAL_DROP"
    elif after_col > before_col and after_row == before_row:
        move_type = "SIDE_DRAG"
    else:
        move_type = "CONTINUE_OTHER"

    return {
        "move_type": move_type,
        "before": before_pos,
        "after": after_pos,
        "before_row": before_row,
        "before_col": before_col,
        "after_row": after_row,
        "after_col": after_col,
    }


def _candidate_derived_color_info(non_tie: List[str], candidate: str, offset: int) -> Dict[str, Any]:
    # 候選下三路模擬：
    # 模擬下一口若開 candidate=B/P，會在指定下三路 offset 產生什麼顏色與結構。
    # new_color：1=紅，-1=藍，0=無法產生 / 資料不足。
    #
    # 結構判斷：
    # row=0 新欄：比前一欄高度與對照欄高度是否齊整。
    # row>0 直落/橫拖：比對照欄同列有無、上一列有無；有有/無無=齊整，有無/無有=不齊。
    if candidate not in {"B", "P"}:
        return {
            "candidate": candidate,
            "new_color": 0,
            "new_color_text": "N",
            "before_len": 0,
            "after_len": 0,
            "pos": {},
            "move": {},
            "structure": {},
        }

    before_layout = _build_big_road(non_tie)
    before_series = _derived_series(before_layout, offset)

    after_layout = _build_big_road(non_tie + [candidate])
    after_series = _derived_series(after_layout, offset)

    new_color = 0
    if len(after_series) > len(before_series):
        new_color = after_series[-1]

    move_info = _classify_bigroad_move(before_layout, after_layout, candidate)
    pos = move_info.get("after", {}) or {}
    row = int(pos.get("row", 0))
    col = int(pos.get("col", 0))

    grid = after_layout.get("grid", {}) or {}
    heights = after_layout.get("col_heights", {}) or {}

    structure: Dict[str, Any] = {
        "move_type": move_info.get("move_type", "NONE"),
        "row": row,
        "col": col,
        "offset": offset,
        "is_new_column": row == 0,
        "is_vertical_drop": move_info.get("move_type") == "VERTICAL_DROP",
        "is_side_drag": move_info.get("move_type") == "SIDE_DRAG",
        "is_neat": False,
        "has_same_row": None,
        "has_prev_row": None,
        "left_height": None,
        "compare_height": None,
        "relation": "",
    }

    if col <= offset:
        structure["relation"] = "資料不足"
    elif row == 0:
        # 新欄齊整：
        # 大眼仔 offset=1：前一欄 vs 前二欄高度
        # 小路 offset=2：前一欄 vs 前三欄高度
        # 蟑螂路 offset=3：前一欄 vs 前四欄高度
        left_col = col - 1
        compare_col = col - 1 - offset
        left_h = int(heights.get(left_col, 0))
        compare_h = int(heights.get(compare_col, 0))
        is_neat = bool(left_h > 0 and compare_h > 0 and left_h == compare_h)

        structure.update({
            "left_col": left_col,
            "compare_col": compare_col,
            "left_height": left_h,
            "compare_height": compare_h,
            "is_neat": is_neat,
            "relation": f"新欄高度{'齊整' if is_neat else '不齊'}:{left_h}/{compare_h}",
        })
    else:
        # 有無齊整：
        # 同列有無與上一列有無相同 => 齊整
        # 同列有無與上一列有無不同 => 不齊
        compare_col = col - offset
        has_same_row = ((row, compare_col) in grid)
        has_prev_row = ((row - 1, compare_col) in grid)
        is_neat = bool(has_same_row == has_prev_row)
        relation = (
            ("有" if has_same_row else "無")
            + "/"
            + ("有" if has_prev_row else "無")
        )

        structure.update({
            "compare_col": compare_col,
            "has_same_row": has_same_row,
            "has_prev_row": has_prev_row,
            "is_neat": is_neat,
            "relation": f"有無{relation}:{'齊整' if is_neat else '不齊'}",
        })

    return {
        "candidate": candidate,
        "new_color": new_color,
        "new_color_text": "R" if new_color == 1 else "B" if new_color == -1 else "N",
        "before_len": len(before_series),
        "after_len": len(after_series),
        "pos": pos,
        "move": move_info,
        "structure": structure,
    }


def _score_candidate_color_pattern(series: List[int], candidate_color: int, lookback: Optional[int] = None) -> Dict[str, Any]:
    # 評分候選顏色是否符合目前下三路紅藍節奏。
    # 這裡只判斷紅藍節奏，不直接把紅藍等於莊閒。
    if lookback is None:
        lookback = DERIVED_CANDIDATE_LOOKBACK

    if candidate_color not in {1, -1}:
        return {
            "score": 0.5,
            "confidence": 0.0,
            "expected_color": 0,
            "expected_color_text": "N",
            "candidate_color_text": "N",
            "label": "候選無新色",
        }

    tail = series[-lookback:] if series else []
    if len(tail) < 3:
        return {
            "score": 0.5,
            "confidence": 0.0,
            "expected_color": 0,
            "expected_color_text": "N",
            "candidate_color_text": "R" if candidate_color == 1 else "B",
            "label": "紅藍樣本不足",
        }

    last_color = tail[-1]

    color_streak = 1
    for x in reversed(tail[:-1]):
        if x == last_color:
            color_streak += 1
        else:
            break

    switches = sum(1 for a, b in zip(tail, tail[1:]) if a != b)
    switch_rate = _safe_div(switches, max(1, len(tail) - 1), 0.5)

    red_rate = tail.count(1) / len(tail)
    blue_rate = tail.count(-1) / len(tail)

    expected_color = 0
    edge = 0.0
    label = "紅藍中性"

    if switch_rate >= DERIVED_COLOR_JUMP_RATE and len(tail) >= 6:
        expected_color = -last_color
        edge = min(0.16, 0.09 + (switch_rate - DERIVED_COLOR_JUMP_RATE) * 0.28)
        label = "下三路紅藍單跳"
    elif color_streak >= DERIVED_COLOR_STREAK_MIN:
        expected_color = last_color
        edge = min(0.17, 0.09 + (color_streak - DERIVED_COLOR_STREAK_MIN) * 0.025)
        label = f"下三路{'紅' if last_color == 1 else '藍'}連{color_streak}"
    elif abs(red_rate - blue_rate) >= DERIVED_COLOR_RATIO_GAP:
        expected_color = 1 if red_rate > blue_rate else -1
        edge = min(0.11, abs(red_rate - blue_rate) * 0.22)
        label = "下三路紅藍比例偏態"
    else:
        found = False
        max_k = min(max(2, DERIVED_COLOR_NGRAM_MAX), len(tail) - 1)
        for k in range(max_k, 1, -1):
            key = tail[-k:]
            follows = []
            for i in range(0, len(tail) - k):
                if tail[i:i + k] == key and i + k < len(tail):
                    follows.append(tail[i + k])

            if len(follows) >= 2:
                red_follow = follows.count(1)
                blue_follow = follows.count(-1)
                if red_follow != blue_follow:
                    expected_color = 1 if red_follow > blue_follow else -1
                    edge = min(0.12, abs(red_follow - blue_follow) / len(follows) * 0.16)
                    label = f"下三路紅藍NGram{k}"
                    found = True
                    break

        if not found:
            expected_color = last_color
            edge = 0.035
            label = "下三路弱續勢"

    score = 0.5 + edge if candidate_color == expected_color else 0.5 - edge
    confidence = min(1.0, abs(score - 0.5) * 2.8)

    return {
        "score": round(score, 5),
        "confidence": round(confidence, 4),
        "expected_color": expected_color,
        "expected_color_text": "R" if expected_color == 1 else "B" if expected_color == -1 else "N",
        "candidate_color_text": "R" if candidate_color == 1 else "B" if candidate_color == -1 else "N",
        "label": label,
        "switch_rate": round(switch_rate, 4),
        "color_streak": color_streak,
        "red_rate": round(red_rate, 4),
        "blue_rate": round(blue_rate, 4),
        "tail": "".join("R" if x == 1 else "B" for x in tail),
    }


def _score_candidate_structure(info: Dict[str, Any], series: List[int]) -> Dict[str, Any]:
    # 結構分：判斷候選落點是否符合「齊整 / 有無 / 直落 / 橫拖」。
    # score > 0.5 支持該候選；score < 0.5 降低該候選。
    structure = info.get("structure", {}) or {}
    move_type = structure.get("move_type", "NONE")
    is_neat = bool(structure.get("is_neat", False))
    new_color = int(info.get("new_color", 0) or 0)

    edge = 0.0
    reasons = []

    # 齊整是下三路核心。
    if is_neat:
        edge += DERIVED_STRUCTURE_NEAT_BONUS
        reasons.append("齊整")
    else:
        edge -= DERIVED_STRUCTURE_MISMATCH_PENALTY
        reasons.append("不齊")

    # 直落代表大路同欄規律仍在延伸。
    if move_type == "VERTICAL_DROP":
        edge += DERIVED_STRUCTURE_DROP_BONUS
        reasons.append("直落")
    elif move_type == "SIDE_DRAG":
        # 橫拖代表到底/卡位後黏邊，容易是龍尾或形變，先降一點。
        edge -= DERIVED_STRUCTURE_SIDE_DRAG_PENALTY
        reasons.append("黏邊橫拖")
    elif move_type == "NEW_COLUMN":
        edge += DERIVED_STRUCTURE_NEWCOL_BONUS
        reasons.append("新欄")

    # 結構與新色矛盾時降分：
    # 正常：齊整多半對應紅，不齊多半對應藍。
    if new_color in {1, -1}:
        if is_neat and new_color == -1:
            edge -= DERIVED_STRUCTURE_MISMATCH_PENALTY * 0.55
            reasons.append("齊整卻出藍")
        elif (not is_neat) and new_color == 1:
            edge -= DERIVED_STRUCTURE_MISMATCH_PENALTY * 0.35
            reasons.append("不齊卻出紅")

    edge = _clamp(edge, -DERIVED_STRUCTURE_MAX_EDGE, DERIVED_STRUCTURE_MAX_EDGE)
    score = 0.5 + edge

    return {
        "score": round(score, 5),
        "edge": round(edge, 5),
        "label": "+".join(reasons) if reasons else "結構中性",
        "move_type": move_type,
        "is_neat": is_neat,
        "relation": structure.get("relation", ""),
        "structure": structure,
    }


def _combine_candidate_scores(color_score: float, structure_score: float) -> float:
    # 合併候選分數：
    # color_score=紅藍節奏分；structure_score=齊整 / 有無 / 直落結構分。
    total_w = max(0.0001, DERIVED_COLOR_SCORE_WEIGHT + DERIVED_STRUCTURE_SCORE_WEIGHT)
    cw = DERIVED_COLOR_SCORE_WEIGHT / total_w
    sw = DERIVED_STRUCTURE_SCORE_WEIGHT / total_w
    return float(color_score) * cw + float(structure_score) * sw


def _candidate_scores_to_side_prob(b_score: float, p_score: float, max_edge: Optional[float] = None) -> Tuple[float, float, float]:
    # 把候選莊 / 閒評分轉成 B/P 機率。
    # 分數差越大，B/P 邊際越大；但限制最大邊際，避免過度自信。
    if max_edge is None:
        max_edge = DERIVED_CANDIDATE_MAX_EDGE

    diff = float(b_score) - float(p_score)
    edge = _clamp(diff * 0.18, -max_edge, max_edge)
    b = 0.5 + edge
    p = 1.0 - b
    return b, p, abs(edge)
'''.strip()


DERIVED_ROAD_SCORE_BLOCK = r'''
def _derived_road_score(non_tie: List[str], offset: int, road_key: str, display_name: str) -> Dict[str, Any]:
    # 下三路候選模擬 + 結構判斷版。
    # 不再用「紅=跟、藍=反」這種簡化邏輯。
    #
    # 流程：
    # 1. 模擬下一口開莊，會讓該下三路產生什麼顏色與結構。
    # 2. 模擬下一口開閒，會讓該下三路產生什麼顏色與結構。
    # 3. 用紅藍節奏分 + 齊整/有無/直落結構分合成候選分數。
    # 4. 回推下一口大路比較偏莊或閒。
    default = {
        "B": 0.5,
        "P": 0.5,
        "label": f"{display_name}資料不足",
        "strength": 0.0,
        "road_key": road_key,
        "stats": {"last": 0, "red_rate": 0.5, "blue_rate": 0.5, "count": 0, "tail": ""},
        "red_pressure": 0.5,
        "blue_pressure": 0.5,
        "candidate": {},
    }

    if not USE_ROAD_ENGINE or len(non_tie) < ROAD_ENGINE_MIN_HISTORY:
        return default

    layout = _build_big_road(non_tie)
    series = _derived_series(layout, offset=offset)
    stats = _color_stats(series)
    count = int(stats.get("count", 0))

    if count < DERIVED_ROAD_MIN_COUNT:
        return {
            **default,
            "stats": stats,
            "label": f"{display_name}樣本不足",
        }

    # 候選模擬：下一口莊 / 閒各自會讓這條下三路出現什麼顏色與結構。
    b_info = _candidate_derived_color_info(non_tie, "B", offset)
    p_info = _candidate_derived_color_info(non_tie, "P", offset)

    b_color_eval = _score_candidate_color_pattern(series, int(b_info.get("new_color", 0)))
    p_color_eval = _score_candidate_color_pattern(series, int(p_info.get("new_color", 0)))

    b_struct_eval = _score_candidate_structure(b_info, series)
    p_struct_eval = _score_candidate_structure(p_info, series)

    b_score = _combine_candidate_scores(
        float(b_color_eval.get("score", 0.5)),
        float(b_struct_eval.get("score", 0.5)),
    )
    p_score = _combine_candidate_scores(
        float(p_color_eval.get("score", 0.5)),
        float(p_struct_eval.get("score", 0.5)),
    )

    b, p, edge = _candidate_scores_to_side_prob(b_score, p_score, max_edge=DERIVED_CANDIDATE_MAX_EDGE)

    if edge < DERIVED_CANDIDATE_MIN_EDGE:
        label = f"{display_name}候選接近"
        strength = 0.06
    else:
        pick = "莊" if b > p else "閒"
        label = f"{display_name}結構候選偏{pick}"
        strength = 0.10 + min(0.15, edge * 2.0)

    red_rate = float(stats.get("red_rate", 0.5))
    blue_rate = float(stats.get("blue_rate", 0.5))

    return {
        "B": round(b, 5),
        "P": round(p, 5),
        "label": label,
        "strength": round(strength, 4),
        "road_key": road_key,
        "stats": stats,
        "red_pressure": round(red_rate, 4),
        "blue_pressure": round(blue_rate, 4),
        "tail": stats.get("tail", ""),
        "candidate": {
            "B": {
                "new_color": b_info.get("new_color_text", "N"),
                "color_score": round(float(b_color_eval.get("score", 0.5)), 5),
                "structure_score": round(float(b_struct_eval.get("score", 0.5)), 5),
                "score": round(b_score, 5),
                "color_eval": b_color_eval,
                "structure_eval": b_struct_eval,
                "pos": b_info.get("pos", {}),
                "structure": b_info.get("structure", {}),
            },
            "P": {
                "new_color": p_info.get("new_color_text", "N"),
                "color_score": round(float(p_color_eval.get("score", 0.5)), 5),
                "structure_score": round(float(p_struct_eval.get("score", 0.5)), 5),
                "score": round(p_score, 5),
                "color_eval": p_color_eval,
                "structure_eval": p_struct_eval,
                "pos": p_info.get("pos", {}),
                "structure": p_info.get("structure", {}),
            },
            "edge": round(edge, 5),
            "diff": round(b_score - p_score, 5),
        },
    }
'''.strip()


FUHAO_DOWN3_VOTE_BLOCK = r'''
def _fuhao_down3_vote(non_tie: List[str], offset: int, name: str) -> Dict[str, Any]:
    # 富濠式下三路候選模擬 + 結構判斷版。
    # 不再用紅=續、藍=反。
    # 直接模擬下一口莊/閒各會讓下三路出現什麼顏色與結構，
    # 再用紅藍節奏分 + 齊整/有無/直落結構分評分。
    if len(non_tie) < FUHAO_MIN_VALID_ROUNDS:
        return {
            "pick": "",
            "label": f"{name}資料不足",
            "confidence": 0.0,
            "stats": {},
            "candidate": {},
        }

    layout = _build_big_road(non_tie)
    series = _derived_series(layout, offset=offset)
    stats = _color_stats(series)
    count = int(stats.get("count", 0))

    if count < DERIVED_ROAD_MIN_COUNT:
        return {
            "pick": "",
            "label": f"{name}樣本不足",
            "confidence": 0.0,
            "stats": stats,
            "candidate": {},
        }

    b_info = _candidate_derived_color_info(non_tie, "B", offset)
    p_info = _candidate_derived_color_info(non_tie, "P", offset)

    b_color_eval = _score_candidate_color_pattern(series, int(b_info.get("new_color", 0)))
    p_color_eval = _score_candidate_color_pattern(series, int(p_info.get("new_color", 0)))

    b_struct_eval = _score_candidate_structure(b_info, series)
    p_struct_eval = _score_candidate_structure(p_info, series)

    b_score = _combine_candidate_scores(
        float(b_color_eval.get("score", 0.5)),
        float(b_struct_eval.get("score", 0.5)),
    )
    p_score = _combine_candidate_scores(
        float(p_color_eval.get("score", 0.5)),
        float(p_struct_eval.get("score", 0.5)),
    )

    diff = b_score - p_score

    if abs(diff) < FUHAO_DOWN3_MIN_DIFF:
        pick = ""
        label = f"{name}候選差距不足"
        confidence = 0.42
    else:
        pick = "B" if diff > 0 else "P"
        label = f"{name}結構候選偏{_fuhao_side_name(pick)}"
        confidence = min(0.80, 0.50 + abs(diff) * 1.35 + min(0.08, count * 0.006))

    return {
        "pick": pick,
        "label": label,
        "confidence": round(confidence, 4),
        "stats": stats,
        "candidate": {
            "B": {
                "new_color": b_info.get("new_color_text", "N"),
                "color_score": round(float(b_color_eval.get("score", 0.5)), 5),
                "structure_score": round(float(b_struct_eval.get("score", 0.5)), 5),
                "score": round(b_score, 5),
                "color_eval": b_color_eval,
                "structure_eval": b_struct_eval,
                "pos": b_info.get("pos", {}),
                "structure": b_info.get("structure", {}),
            },
            "P": {
                "new_color": p_info.get("new_color_text", "N"),
                "color_score": round(float(p_color_eval.get("score", 0.5)), 5),
                "structure_score": round(float(p_struct_eval.get("score", 0.5)), 5),
                "score": round(p_score, 5),
                "color_eval": p_color_eval,
                "structure_eval": p_struct_eval,
                "pos": p_info.get("pos", {}),
                "structure": p_info.get("structure", {}),
            },
            "diff": round(diff, 5),
        },
    }
'''.strip()


def fail(message: str) -> None:
    print(f"[ERROR] {message}")
    sys.exit(1)


def replace_between(text: str, start_marker: str, end_marker: str, new_block: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        fail(f"找不到起點：{start_marker}")

    end = text.find(end_marker, start)
    if end < 0:
        fail(f"找不到終點：{end_marker}")

    return text[:start] + new_block.rstrip() + "\n\n\n" + text[end:]


def upsert_env_block(text: str) -> str:
    start_marker = "# Candidate Down-Road Simulation：下三路候選模擬參數"
    end_marker = "# End Candidate Down-Road Simulation"

    if start_marker in text and end_marker in text:
        pattern = re.compile(
            re.escape(start_marker) + r".*?" + re.escape(end_marker),
            flags=re.DOTALL,
        )
        return pattern.sub(ENV_BLOCK, text, count=1)

    target = 'DERIVED_ROAD_MIN_COUNT = int(os.getenv("DERIVED_ROAD_MIN_COUNT", "3"))'
    idx = text.find(target)
    if idx < 0:
        fail("找不到 DERIVED_ROAD_MIN_COUNT，請確認 predictor.py 是否為你貼給我的版本。")

    line_end = text.find("\n", idx)
    if line_end < 0:
        line_end = idx + len(target)

    return text[:line_end + 1] + ENV_BLOCK + "\n" + text[line_end + 1:]


def upsert_helper_block(text: str) -> str:
    helper_start = "def _candidate_derived_color_info("
    big_road_start = "def _big_road_score("

    if helper_start in text:
        return replace_between(text, helper_start, big_road_start, HELPER_BLOCK)

    idx = text.find(big_road_start)
    if idx < 0:
        fail("找不到 def _big_road_score，無法插入候選模擬 helper。")

    return text[:idx] + HELPER_BLOCK.rstrip() + "\n\n\n" + text[idx:]


def main() -> None:
    if not PREDICTOR_PATH.exists():
        fail("目前資料夾找不到 predictor.py，請把本腳本放在 predictor.py 同一層。")

    text = PREDICTOR_PATH.read_text(encoding="utf-8")

    backup_path = PREDICTOR_PATH.with_name(f"predictor.backup-{datetime.now():%Y%m%d-%H%M%S}.py")
    backup_path.write_text(text, encoding="utf-8")

    text = upsert_env_block(text)
    text = upsert_helper_block(text)
    text = replace_between(
        text,
        "def _derived_road_score(",
        "def _big_eye_score(",
        DERIVED_ROAD_SCORE_BLOCK,
    )
    text = replace_between(
        text,
        "def _fuhao_down3_vote(",
        "def _fuhao_deep_parity_vote(",
        FUHAO_DOWN3_VOTE_BLOCK,
    )

    PREDICTOR_PATH.write_text(text, encoding="utf-8")

    print("[OK] predictor.py 已完成四路候選模擬 + 結構判斷升級。")
    print(f"[OK] 原檔備份：{backup_path.name}")
    print("[OK] 本腳本只替換四路下三路相關區塊，其餘模型沒有動刀。")
    print("[OK] 請執行：python -m py_compile predictor.py")
    print("[OK] 若沒有錯誤，再部署到 Render。")


if __name__ == "__main__":
    main()
