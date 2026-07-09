#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime
import re
import sys

PREDICTOR_PATH = Path('predictor.py')

ENV_BLOCK = r'''
# Candidate Down-Road Simulation：下三路候選模擬參數
DERIVED_CANDIDATE_LOOKBACK = int(os.getenv("DERIVED_CANDIDATE_LOOKBACK", str(ROAD_ENGINE_DERIVED_LOOKBACK)))
DERIVED_CANDIDATE_MAX_EDGE = float(os.getenv("DERIVED_CANDIDATE_MAX_EDGE", "0.078"))
DERIVED_CANDIDATE_MIN_EDGE = float(os.getenv("DERIVED_CANDIDATE_MIN_EDGE", "0.008"))
DERIVED_COLOR_JUMP_RATE = float(os.getenv("DERIVED_COLOR_JUMP_RATE", "0.68"))
DERIVED_COLOR_STREAK_MIN = int(os.getenv("DERIVED_COLOR_STREAK_MIN", "3"))
DERIVED_COLOR_RATIO_GAP = float(os.getenv("DERIVED_COLOR_RATIO_GAP", "0.22"))
DERIVED_COLOR_NGRAM_MAX = int(os.getenv("DERIVED_COLOR_NGRAM_MAX", "5"))
FUHAO_DOWN3_MIN_DIFF = float(os.getenv("FUHAO_DOWN3_MIN_DIFF", "0.020"))

# Down-Road Structure：下三路齊整 / 有無 / 直落結構分
DERIVED_COLOR_SCORE_WEIGHT = float(os.getenv("DERIVED_COLOR_SCORE_WEIGHT", "0.55"))
DERIVED_STRUCTURE_SCORE_WEIGHT = float(os.getenv("DERIVED_STRUCTURE_SCORE_WEIGHT", "0.45"))
DERIVED_STRUCTURE_NEAT_BONUS = float(os.getenv("DERIVED_STRUCTURE_NEAT_BONUS", "0.055"))
DERIVED_STRUCTURE_MISMATCH_PENALTY = float(os.getenv("DERIVED_STRUCTURE_MISMATCH_PENALTY", "0.045"))
DERIVED_STRUCTURE_DROP_BONUS = float(os.getenv("DERIVED_STRUCTURE_DROP_BONUS", "0.035"))
DERIVED_STRUCTURE_SIDE_DRAG_PENALTY = float(os.getenv("DERIVED_STRUCTURE_SIDE_DRAG_PENALTY", "0.020"))
DERIVED_STRUCTURE_NEWCOL_BONUS = float(os.getenv("DERIVED_STRUCTURE_NEWCOL_BONUS", "0.025"))
DERIVED_STRUCTURE_MAX_EDGE = float(os.getenv("DERIVED_STRUCTURE_MAX_EDGE", "0.090"))
# End Candidate Down-Road Simulation
'''.strip()

BUILD_BIG_ROAD_BLOCK = r'''
def _build_big_road(non_tie: List[str], rows: int = ROAD_ENGINE_ROWS) -> Dict[str, Any]:
    # 百家樂大路盤面：同邊直落；到底/卡位橫拖；換邊開新欄。
    rows = max(3, int(rows or 6))
    sequence = [x for x in non_tie if x in {"B", "P"}]
    grid: Dict[Tuple[int, int], str] = {}
    positions: List[Dict[str, Any]] = []
    last_side = ""
    row = 0
    col = 0

    for idx, side in enumerate(sequence):
        if idx == 0:
            row, col, move_type = 0, 0, "START"
        elif side != last_side:
            target_row, target_col = 0, col + 1
            while (target_row, target_col) in grid:
                target_col += 1
            row, col, move_type = target_row, target_col, "NEW_COLUMN"
        else:
            target_row, target_col = row + 1, col
            if target_row < rows and (target_row, target_col) not in grid:
                row, col, move_type = target_row, target_col, "VERTICAL_DROP"
            else:
                target_row, target_col = row, col + 1
                while (target_row, target_col) in grid:
                    target_col += 1
                row, col, move_type = target_row, target_col, "SIDE_DRAG"

        grid[(row, col)] = side
        positions.append({"i": idx, "side": side, "row": row, "col": col, "move_type": move_type})
        last_side = side

    col_heights = Counter()
    col_sides: Dict[int, str] = {}
    for (r, c), side in grid.items():
        col_heights[c] += 1
        if r == 0:
            col_sides[c] = side

    max_col = max([p["col"] for p in positions], default=0)
    last_pos = positions[-1] if positions else {"i": -1, "side": "", "row": 0, "col": 0, "move_type": "NONE"}
    return {
        "rows": rows,
        "sequence": sequence,
        "grid": grid,
        "positions": positions,
        "col_heights": dict(col_heights),
        "col_sides": col_sides,
        "max_col": max_col,
        "last": last_pos,
    }
'''.strip()

DERIVED_COLOR_AT_BLOCK = r'''
def _derived_color_at(layout: Dict[str, Any], pos: Dict[str, Any], offset: int) -> int:
    # 下三路紅藍：offset=1 大眼仔；2 小路；3 蟑螂路。1=紅，-1=藍，0=資料不足。
    col = int(pos.get("col", 0))
    row = int(pos.get("row", 0))
    offset = int(offset)
    grid = layout.get("grid", {}) or {}
    heights = layout.get("col_heights", {}) or {}
    if col <= offset:
        return 0
    if row == 0:
        left_h = int(heights.get(col - 1, 0))
        compare_h = int(heights.get(col - 1 - offset, 0))
        if left_h <= 0 or compare_h <= 0:
            return 0
        return 1 if left_h == compare_h else -1
    compare_col = col - offset
    has_same_row = ((row, compare_col) in grid)
    has_prev_row = ((row - 1, compare_col) in grid)
    return 1 if has_same_row == has_prev_row else -1
'''.strip()

DERIVED_SERIES_BLOCK = r'''
def _derived_series(layout: Dict[str, Any], offset: int) -> List[int]:
    # 下三路必須逐局生成，不能用最後版面倒回去重算，避免未來格子污染過去紅藍。
    seq = layout.get("sequence")
    if seq:
        clean_seq = [x for x in seq if x in {"B", "P"}]
        cache_key = (int(offset), "".join(clean_seq))
        cache = getattr(_derived_series, "_cache", None)
        if cache is None:
            cache = {}
            setattr(_derived_series, "_cache", cache)
        if cache_key in cache:
            return list(cache[cache_key])
        series: List[int] = []
        for i in range(1, len(clean_seq) + 1):
            partial_layout = _build_big_road(clean_seq[:i])
            positions = partial_layout.get("positions", []) or []
            if not positions:
                continue
            color = _derived_color_at(partial_layout, positions[-1], offset)
            if color != 0:
                series.append(color)
        if len(cache) > 500:
            cache.clear()
        cache[cache_key] = list(series)
        return series

    series = []
    for pos in layout.get("positions", []):
        color = _derived_color_at(layout, pos, offset)
        if color != 0:
            series.append(color)
    return series
'''.strip()

HELPER_BLOCK = r'''
def _classify_bigroad_move(before_layout: Dict[str, Any], after_layout: Dict[str, Any], candidate: str) -> Dict[str, Any]:
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
        move_type = after_pos.get("move_type", "CONTINUE_OTHER")
    return {"move_type": move_type, "before": before_pos, "after": after_pos,
            "before_row": before_row, "before_col": before_col, "after_row": after_row, "after_col": after_col}


def _candidate_derived_color_info(non_tie: List[str], candidate: str, offset: int) -> Dict[str, Any]:
    if candidate not in {"B", "P"}:
        return {"candidate": candidate, "new_color": 0, "new_color_text": "N", "before_len": 0, "after_len": 0, "pos": {}, "move": {}, "structure": {}}

    before_layout = _build_big_road(non_tie)
    before_series = _derived_series(before_layout, offset)
    after_layout = _build_big_road(non_tie + [candidate])
    after_series = _derived_series(after_layout, offset)
    new_color = after_series[-1] if len(after_series) > len(before_series) else 0
    move_info = _classify_bigroad_move(before_layout, after_layout, candidate)
    pos = move_info.get("after", {}) or {}
    row = int(pos.get("row", 0))
    col = int(pos.get("col", 0))
    grid = after_layout.get("grid", {}) or {}
    heights = after_layout.get("col_heights", {}) or {}
    structure: Dict[str, Any] = {
        "move_type": move_info.get("move_type", "NONE"), "row": row, "col": col, "offset": offset,
        "is_new_column": row == 0, "is_vertical_drop": move_info.get("move_type") == "VERTICAL_DROP",
        "is_side_drag": move_info.get("move_type") == "SIDE_DRAG", "is_neat": False,
        "has_same_row": None, "has_prev_row": None, "left_height": None, "compare_height": None, "relation": "",
    }
    if col <= offset:
        structure["relation"] = "資料不足"
    elif row == 0:
        left_col = col - 1
        compare_col = col - 1 - offset
        left_h = int(heights.get(left_col, 0))
        compare_h = int(heights.get(compare_col, 0))
        is_neat = bool(left_h > 0 and compare_h > 0 and left_h == compare_h)
        structure.update({"left_col": left_col, "compare_col": compare_col, "left_height": left_h,
                          "compare_height": compare_h, "is_neat": is_neat,
                          "relation": f"新欄高度{'齊整' if is_neat else '不齊'}:{left_h}/{compare_h}"})
    else:
        compare_col = col - offset
        has_same_row = ((row, compare_col) in grid)
        has_prev_row = ((row - 1, compare_col) in grid)
        is_neat = bool(has_same_row == has_prev_row)
        relation = ("有" if has_same_row else "無") + "/" + ("有" if has_prev_row else "無")
        structure.update({"compare_col": compare_col, "has_same_row": has_same_row, "has_prev_row": has_prev_row,
                          "is_neat": is_neat, "relation": f"有無{relation}:{'齊整' if is_neat else '不齊'}"})
    return {"candidate": candidate, "new_color": new_color, "new_color_text": "R" if new_color == 1 else "B" if new_color == -1 else "N",
            "before_len": len(before_series), "after_len": len(after_series), "pos": pos, "move": move_info, "structure": structure}


def _score_candidate_color_pattern(series: List[int], candidate_color: int, lookback: Optional[int] = None) -> Dict[str, Any]:
    if lookback is None:
        lookback = DERIVED_CANDIDATE_LOOKBACK
    if candidate_color not in {1, -1}:
        return {"score": 0.5, "confidence": 0.0, "expected_color": 0, "expected_color_text": "N", "candidate_color_text": "N", "label": "候選無新色"}
    tail = series[-lookback:] if series else []
    if len(tail) < 3:
        return {"score": 0.5, "confidence": 0.0, "expected_color": 0, "expected_color_text": "N", "candidate_color_text": "R" if candidate_color == 1 else "B", "label": "紅藍樣本不足"}
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
    expected_color, edge, label = 0, 0.0, "紅藍中性"
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
            follows = [tail[i + k] for i in range(0, len(tail) - k) if tail[i:i + k] == key and i + k < len(tail)]
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
            expected_color, edge, label = last_color, 0.035, "下三路弱續勢"
    score = 0.5 + edge if candidate_color == expected_color else 0.5 - edge
    return {"score": round(score, 5), "confidence": round(min(1.0, abs(score - 0.5) * 2.8), 4),
            "expected_color": expected_color, "expected_color_text": "R" if expected_color == 1 else "B" if expected_color == -1 else "N",
            "candidate_color_text": "R" if candidate_color == 1 else "B", "label": label,
            "switch_rate": round(switch_rate, 4), "color_streak": color_streak, "red_rate": round(red_rate, 4),
            "blue_rate": round(blue_rate, 4), "tail": "".join("R" if x == 1 else "B" for x in tail)}


def _score_candidate_structure(info: Dict[str, Any], series: List[int]) -> Dict[str, Any]:
    structure = info.get("structure", {}) or {}
    move_type = structure.get("move_type", "NONE")
    is_neat = bool(structure.get("is_neat", False))
    new_color = int(info.get("new_color", 0) or 0)
    edge, reasons = 0.0, []
    if is_neat:
        edge += DERIVED_STRUCTURE_NEAT_BONUS; reasons.append("齊整")
    else:
        edge -= DERIVED_STRUCTURE_MISMATCH_PENALTY; reasons.append("不齊")
    if move_type == "VERTICAL_DROP":
        edge += DERIVED_STRUCTURE_DROP_BONUS; reasons.append("直落")
    elif move_type == "SIDE_DRAG":
        edge -= DERIVED_STRUCTURE_SIDE_DRAG_PENALTY; reasons.append("黏邊橫拖")
    elif move_type == "NEW_COLUMN":
        edge += DERIVED_STRUCTURE_NEWCOL_BONUS; reasons.append("新欄")
    if new_color in {1, -1}:
        if is_neat and new_color == -1:
            edge -= DERIVED_STRUCTURE_MISMATCH_PENALTY * 0.55; reasons.append("齊整卻出藍")
        elif (not is_neat) and new_color == 1:
            edge -= DERIVED_STRUCTURE_MISMATCH_PENALTY * 0.35; reasons.append("不齊卻出紅")
    edge = _clamp(edge, -DERIVED_STRUCTURE_MAX_EDGE, DERIVED_STRUCTURE_MAX_EDGE)
    return {"score": round(0.5 + edge, 5), "edge": round(edge, 5), "label": "+".join(reasons),
            "move_type": move_type, "is_neat": is_neat, "relation": structure.get("relation", ""), "structure": structure}


def _combine_candidate_scores(color_score: float, structure_score: float) -> float:
    total_w = max(0.0001, DERIVED_COLOR_SCORE_WEIGHT + DERIVED_STRUCTURE_SCORE_WEIGHT)
    return float(color_score) * (DERIVED_COLOR_SCORE_WEIGHT / total_w) + float(structure_score) * (DERIVED_STRUCTURE_SCORE_WEIGHT / total_w)


def _candidate_scores_to_side_prob(b_score: float, p_score: float, max_edge: Optional[float] = None) -> Tuple[float, float, float]:
    if max_edge is None:
        max_edge = DERIVED_CANDIDATE_MAX_EDGE
    edge = _clamp((float(b_score) - float(p_score)) * 0.18, -max_edge, max_edge)
    return 0.5 + edge, 0.5 - edge, abs(edge)


def _roadmap_ask_road_debug(non_tie: List[str]) -> Dict[str, Any]:
    layout = _build_big_road(non_tie)
    result: Dict[str, Any] = {"current_big_road": {"last": layout.get("last", {}), "max_col": layout.get("max_col", 0), "col_heights": layout.get("col_heights", {})}}
    for candidate in ["B", "P"]:
        result[f"ask_{candidate}"] = {"candidate": candidate, "candidate_text": "莊" if candidate == "B" else "閒", "roads": {}}
        for offset, road_key in {1: "big_eye", 2: "small_road", 3: "cockroach"}.items():
            info = _candidate_derived_color_info(non_tie, candidate, offset)
            result[f"ask_{candidate}"]["roads"][road_key] = {"color": info.get("new_color_text", "N"), "pos": info.get("pos", {}), "move_type": info.get("move", {}).get("move_type", ""), "structure": info.get("structure", {})}
    return result
'''.strip()

DERIVED_ROAD_SCORE_BLOCK = r'''
def _derived_road_score(non_tie: List[str], offset: int, road_key: str, display_name: str) -> Dict[str, Any]:
    default = {"B": 0.5, "P": 0.5, "label": f"{display_name}資料不足", "strength": 0.0, "road_key": road_key,
               "stats": {"last": 0, "red_rate": 0.5, "blue_rate": 0.5, "count": 0, "tail": ""},
               "red_pressure": 0.5, "blue_pressure": 0.5, "candidate": {}}
    if not USE_ROAD_ENGINE or len(non_tie) < ROAD_ENGINE_MIN_HISTORY:
        return default
    layout = _build_big_road(non_tie)
    series = _derived_series(layout, offset=offset)
    stats = _color_stats(series)
    count = int(stats.get("count", 0))
    if count < DERIVED_ROAD_MIN_COUNT:
        return {**default, "stats": stats, "label": f"{display_name}樣本不足"}
    b_info = _candidate_derived_color_info(non_tie, "B", offset)
    p_info = _candidate_derived_color_info(non_tie, "P", offset)
    b_color_eval = _score_candidate_color_pattern(series, int(b_info.get("new_color", 0)))
    p_color_eval = _score_candidate_color_pattern(series, int(p_info.get("new_color", 0)))
    b_struct_eval = _score_candidate_structure(b_info, series)
    p_struct_eval = _score_candidate_structure(p_info, series)
    b_score = _combine_candidate_scores(float(b_color_eval.get("score", 0.5)), float(b_struct_eval.get("score", 0.5)))
    p_score = _combine_candidate_scores(float(p_color_eval.get("score", 0.5)), float(p_struct_eval.get("score", 0.5)))
    b, p, edge = _candidate_scores_to_side_prob(b_score, p_score, max_edge=DERIVED_CANDIDATE_MAX_EDGE)
    label = f"{display_name}候選接近" if edge < DERIVED_CANDIDATE_MIN_EDGE else f"{display_name}路單候選偏{'莊' if b > p else '閒'}"
    strength = 0.06 if edge < DERIVED_CANDIDATE_MIN_EDGE else 0.10 + min(0.15, edge * 2.0)
    red_rate = float(stats.get("red_rate", 0.5)); blue_rate = float(stats.get("blue_rate", 0.5))
    return {"B": round(b, 5), "P": round(p, 5), "label": label, "strength": round(strength, 4), "road_key": road_key,
            "stats": stats, "red_pressure": round(red_rate, 4), "blue_pressure": round(blue_rate, 4), "tail": stats.get("tail", ""),
            "candidate": {
                "B": {"new_color": b_info.get("new_color_text", "N"), "color_score": round(float(b_color_eval.get("score", 0.5)), 5), "structure_score": round(float(b_struct_eval.get("score", 0.5)), 5), "score": round(b_score, 5), "color_eval": b_color_eval, "structure_eval": b_struct_eval, "pos": b_info.get("pos", {}), "structure": b_info.get("structure", {})},
                "P": {"new_color": p_info.get("new_color_text", "N"), "color_score": round(float(p_color_eval.get("score", 0.5)), 5), "structure_score": round(float(p_struct_eval.get("score", 0.5)), 5), "score": round(p_score, 5), "color_eval": p_color_eval, "structure_eval": p_struct_eval, "pos": p_info.get("pos", {}), "structure": p_info.get("structure", {})},
                "edge": round(edge, 5), "diff": round(b_score - p_score, 5)}}
'''.strip()

FUHAO_DOWN3_VOTE_BLOCK = r'''
def _fuhao_down3_vote(non_tie: List[str], offset: int, name: str) -> Dict[str, Any]:
    if len(non_tie) < FUHAO_MIN_VALID_ROUNDS:
        return {"pick": "", "label": f"{name}資料不足", "confidence": 0.0, "stats": {}, "candidate": {}}
    layout = _build_big_road(non_tie)
    series = _derived_series(layout, offset=offset)
    stats = _color_stats(series)
    count = int(stats.get("count", 0))
    if count < DERIVED_ROAD_MIN_COUNT:
        return {"pick": "", "label": f"{name}樣本不足", "confidence": 0.0, "stats": stats, "candidate": {}}
    b_info = _candidate_derived_color_info(non_tie, "B", offset)
    p_info = _candidate_derived_color_info(non_tie, "P", offset)
    b_color_eval = _score_candidate_color_pattern(series, int(b_info.get("new_color", 0)))
    p_color_eval = _score_candidate_color_pattern(series, int(p_info.get("new_color", 0)))
    b_struct_eval = _score_candidate_structure(b_info, series)
    p_struct_eval = _score_candidate_structure(p_info, series)
    b_score = _combine_candidate_scores(float(b_color_eval.get("score", 0.5)), float(b_struct_eval.get("score", 0.5)))
    p_score = _combine_candidate_scores(float(p_color_eval.get("score", 0.5)), float(p_struct_eval.get("score", 0.5)))
    diff = b_score - p_score
    if abs(diff) < FUHAO_DOWN3_MIN_DIFF:
        pick, label, confidence = "", f"{name}候選差距不足", 0.42
    else:
        pick = "B" if diff > 0 else "P"
        label = f"{name}路單候選偏{_fuhao_side_name(pick)}"
        confidence = min(0.80, 0.50 + abs(diff) * 1.35 + min(0.08, count * 0.006))
    return {"pick": pick, "label": label, "confidence": round(confidence, 4), "stats": stats,
            "candidate": {
                "B": {"new_color": b_info.get("new_color_text", "N"), "color_score": round(float(b_color_eval.get("score", 0.5)), 5), "structure_score": round(float(b_struct_eval.get("score", 0.5)), 5), "score": round(b_score, 5), "color_eval": b_color_eval, "structure_eval": b_struct_eval, "pos": b_info.get("pos", {}), "structure": b_info.get("structure", {})},
                "P": {"new_color": p_info.get("new_color_text", "N"), "color_score": round(float(p_color_eval.get("score", 0.5)), 5), "structure_score": round(float(p_struct_eval.get("score", 0.5)), 5), "score": round(p_score, 5), "color_eval": p_color_eval, "structure_eval": p_struct_eval, "pos": p_info.get("pos", {}), "structure": p_info.get("structure", {})},
                "diff": round(diff, 5)}}
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
        pattern = re.compile(re.escape(start_marker) + r".*?" + re.escape(end_marker), flags=re.DOTALL)
        return pattern.sub(ENV_BLOCK, text, count=1)
    target = 'DERIVED_ROAD_MIN_COUNT = int(os.getenv("DERIVED_ROAD_MIN_COUNT", "3"))'
    idx = text.find(target)
    if idx < 0:
        fail("找不到 DERIVED_ROAD_MIN_COUNT，請確認 predictor.py 是否為你的主程式版本。")
    line_end = text.find("\n", idx)
    if line_end < 0:
        line_end = idx + len(target)
    return text[:line_end + 1] + ENV_BLOCK + "\n" + text[line_end + 1:]


def upsert_helper_block(text: str) -> str:
    idx_big = text.find("def _big_road_score(")
    if idx_big < 0:
        fail("找不到 def _big_road_score，無法插入候選 helper。")
    starts = []
    for marker in ["def _classify_bigroad_move(", "def _candidate_derived_color_info("]:
        idx = text.find(marker)
        if 0 <= idx < idx_big:
            starts.append(idx)
    if starts:
        start = min(starts)
        return text[:start] + HELPER_BLOCK.rstrip() + "\n\n\n" + text[idx_big:]
    return text[:idx_big] + HELPER_BLOCK.rstrip() + "\n\n\n" + text[idx_big:]


def main() -> None:
    if not PREDICTOR_PATH.exists():
        fail("目前資料夾找不到 predictor.py，請把本腳本放在 predictor.py 同一層。")
    text = PREDICTOR_PATH.read_text(encoding="utf-8")
    backup_path = PREDICTOR_PATH.with_name(f"predictor.backup-roadmap-engine-{datetime.now():%Y%m%d-%H%M%S}.py")
    backup_path.write_text(text, encoding="utf-8")

    text = upsert_env_block(text)
    text = replace_between(text, "def _build_big_road(", "def _derived_color_at(", BUILD_BIG_ROAD_BLOCK)
    text = replace_between(text, "def _derived_color_at(", "def _derived_series(", DERIVED_COLOR_AT_BLOCK)
    text = replace_between(text, "def _derived_series(", "def _color_stats(", DERIVED_SERIES_BLOCK)
    text = upsert_helper_block(text)
    text = replace_between(text, "def _derived_road_score(", "def _big_eye_score(", DERIVED_ROAD_SCORE_BLOCK)
    text = replace_between(text, "def _fuhao_down3_vote(", "def _fuhao_deep_parity_vote(", FUHAO_DOWN3_VOTE_BLOCK)

    PREDICTOR_PATH.write_text(text, encoding="utf-8")
    print("[OK] predictor.py 已完成路單盤面引擎 + 下三路候選結構判斷升級。")
    print(f"[OK] 原檔備份：{backup_path.name}")
    print("[OK] 修改範圍：_build_big_road / _derived_color_at / _derived_series / 候選 helper / _derived_road_score / _fuhao_down3_vote / 環境變數區。")
    print("[OK] 其餘原先模型與 predict 主流程沒有動刀。")
    print("[OK] 請執行：python -m py_compile predictor.py")


if __name__ == "__main__":
    main()
