#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# upgrade_ask_road_memory_micro.py
#
# 最新「看路命中率微調」一鍵升級腳本。
#
# 作用：
# 1. 新增 Ask Road Hit Memory：記錄大眼仔 / 小路 / 蟑螂路近期命中率，動態微調可信度。
# 2. 新增 Column Shape Score：加入前排大路欄型分，讓候選判斷更像實際路單看法。
# 3. 替換 _derived_road_score() 與 _fuhao_down3_vote()，把欄型分整合進候選分數。
# 4. 微調 predict() 與 _fuhao_clone_predict()，新增問路記憶更新與 pending 結算。
#
# 注意：
# - 這支是升級腳本，不是 predictor.py。
# - 請放在 predictor.py 同一層執行。
# - 如果你的 predictor.py 尚未有「路單盤面引擎 / 下三路候選 helper」，
#   請直接用我回傳的 predictor_ask_road_memory_full.py 覆蓋 predictor.py。
#
# 使用：
#   python upgrade_ask_road_memory_micro.py
#   python -m py_compile predictor.py

from pathlib import Path
from datetime import datetime
import re
import sys

PREDICTOR_PATH = Path("predictor.py")

ASK_COLUMN_ENV = '# Ask Road Hit Memory：問路命中率記憶\n# 目的：讓每一靴依照最近實際命中率，微調大眼仔 / 小路 / 蟑螂路的可信度。\nUSE_ASK_ROAD_MEMORY = os.getenv("USE_ASK_ROAD_MEMORY", "1") == "1"\nASK_ROAD_MEMORY_WINDOW = int(os.getenv("ASK_ROAD_MEMORY_WINDOW", "24"))\nASK_ROAD_MEMORY_MIN_COUNT = int(os.getenv("ASK_ROAD_MEMORY_MIN_COUNT", "4"))\nASK_ROAD_MEMORY_ALPHA = float(os.getenv("ASK_ROAD_MEMORY_ALPHA", "0.35"))\nASK_ROAD_MEMORY_BAYES_ALPHA = float(os.getenv("ASK_ROAD_MEMORY_BAYES_ALPHA", "2.0"))\nASK_ROAD_MEMORY_MIN_FACTOR = float(os.getenv("ASK_ROAD_MEMORY_MIN_FACTOR", "0.72"))\nASK_ROAD_MEMORY_MAX_FACTOR = float(os.getenv("ASK_ROAD_MEMORY_MAX_FACTOR", "1.28"))\nASK_ROAD_MEMORY_DISABLE_BELOW = float(os.getenv("ASK_ROAD_MEMORY_DISABLE_BELOW", "0.43"))\nASK_ROAD_MEMORY_BOOST_ABOVE = float(os.getenv("ASK_ROAD_MEMORY_BOOST_ABOVE", "0.57"))\nASK_ROAD_MEMORY_APPLY_TO_HYBRID = os.getenv("ASK_ROAD_MEMORY_APPLY_TO_HYBRID", "1") == "1"\nASK_ROAD_MEMORY_APPLY_TO_FUHAO = os.getenv("ASK_ROAD_MEMORY_APPLY_TO_FUHAO", "1") == "1"\nASK_ROAD_MEMORY_DROP_BAD_VOTE = os.getenv("ASK_ROAD_MEMORY_DROP_BAD_VOTE", "1") == "1"\nASK_ROAD_MEMORY_BAD_VOTE_ACC = float(os.getenv("ASK_ROAD_MEMORY_BAD_VOTE_ACC", "0.40"))\nASK_ROAD_MEMORY_DEBUG = os.getenv("ASK_ROAD_MEMORY_DEBUG", "0") == "1"\n\n# Column Shape Score：前排大路欄型分\n# 目的：讓問路不只看紅藍與有無，也看候選落點是否符合前排欄高節奏。\nUSE_DERIVED_COLUMN_SHAPE = os.getenv("USE_DERIVED_COLUMN_SHAPE", "1") == "1"\nDERIVED_COLUMN_SHAPE_WEIGHT = float(os.getenv("DERIVED_COLUMN_SHAPE_WEIGHT", "0.18"))\nDERIVED_COLUMN_SHAPE_LOOKBACK = int(os.getenv("DERIVED_COLUMN_SHAPE_LOOKBACK", "5"))\nDERIVED_COLUMN_NEAT_BONUS = float(os.getenv("DERIVED_COLUMN_NEAT_BONUS", "0.040"))\nDERIVED_COLUMN_BREAK_PENALTY = float(os.getenv("DERIVED_COLUMN_BREAK_PENALTY", "0.035"))\nDERIVED_COLUMN_DRAG_PENALTY = float(os.getenv("DERIVED_COLUMN_DRAG_PENALTY", "0.020"))\nDERIVED_COLUMN_MAX_EDGE = float(os.getenv("DERIVED_COLUMN_MAX_EDGE", "0.070"))'

ASK_ROAD_MEMORY_BLOCK = '# ============ Ask Road Hit Memory：問路命中率記憶 ============\ndef _ask_road_state(training_key: str) -> Dict[str, Any]:\n    key = training_key or "anonymous|ask_road"\n    state = _ASK_ROAD_STATE.get(key)\n    if state is None:\n        state = {"pending": None, "records": []}\n        _ASK_ROAD_STATE[key] = state\n    return state\n\n\ndef _update_ask_road_truth(training_key: str, non_tie: List[str]) -> None:\n    # 將上一輪問路票用本輪新增結果結算。\n    # pending 的 non_tie_len=N，代表上一輪在 N 口時預測第 N+1 口。\n    # 本輪 len(non_tie)>N 時，truth=non_tie[N]，完全不偷看未來。\n    if not USE_ASK_ROAD_MEMORY:\n        return\n\n    state = _ask_road_state(training_key)\n    pending = state.get("pending")\n    if not pending:\n        return\n\n    pred_len = int(pending.get("non_tie_len", -1))\n    if pred_len < 0 or len(non_tie) <= pred_len:\n        return\n\n    truth = non_tie[pred_len]\n    if truth not in {"B", "P"}:\n        state["pending"] = None\n        return\n\n    predictions = pending.get("predictions", {}) or {}\n    record = {"truth": truth, "at_len": pred_len, "models": {}}\n    for name, pick in predictions.items():\n        if pick in {"B", "P"}:\n            record["models"][name] = 1 if pick == truth else 0\n\n    if record["models"]:\n        records = state.setdefault("records", [])\n        records.append(record)\n        max_keep = max(20, ASK_ROAD_MEMORY_WINDOW * 4)\n        if len(records) > max_keep:\n            del records[:-max_keep]\n\n    state["pending"] = None\n\n\ndef _get_ask_road_performance(training_key: str) -> Dict[str, Any]:\n    # 回傳每條問路最近命中率與動態 factor。\n    default_models = ["big_eye", "small_road", "cockroach", "road_majority", "final"]\n    result: Dict[str, Any] = {\n        "enabled": USE_ASK_ROAD_MEMORY,\n        "window": ASK_ROAD_MEMORY_WINDOW,\n        "models": {\n            name: {\n                "acc": 0.5,\n                "raw_acc": 0.5,\n                "count": 0,\n                "correct": 0,\n                "factor": 1.0,\n            }\n            for name in default_models\n        },\n        "label": "問路記憶尚未啟用" if not USE_ASK_ROAD_MEMORY else "問路記憶暖機中",\n    }\n\n    if not USE_ASK_ROAD_MEMORY:\n        return result\n\n    state = _ask_road_state(training_key)\n    records = state.get("records", [])[-max(1, ASK_ROAD_MEMORY_WINDOW):]\n    alpha = max(0.0001, ASK_ROAD_MEMORY_BAYES_ALPHA)\n\n    model_names = set(default_models)\n    for rec in records:\n        model_names.update((rec.get("models") or {}).keys())\n\n    models: Dict[str, Any] = {}\n    best_name = ""\n    best_acc = 0.5\n\n    for name in sorted(model_names):\n        vals = [\n            int((rec.get("models") or {}).get(name))\n            for rec in records\n            if name in (rec.get("models") or {})\n        ]\n        count = len(vals)\n        correct = sum(vals)\n        raw_acc = correct / count if count else 0.5\n        acc = (correct + alpha) / (count + 2 * alpha) if count else 0.5\n\n        factor = 1.0\n        if count >= ASK_ROAD_MEMORY_MIN_COUNT:\n            factor = 1.0 + (acc - 0.5) * 2.0 * ASK_ROAD_MEMORY_ALPHA\n            if acc <= ASK_ROAD_MEMORY_DISABLE_BELOW:\n                factor = min(factor, 0.90)\n            elif acc >= ASK_ROAD_MEMORY_BOOST_ABOVE:\n                factor = max(factor, 1.05)\n            factor = _clamp(factor, ASK_ROAD_MEMORY_MIN_FACTOR, ASK_ROAD_MEMORY_MAX_FACTOR)\n\n        models[name] = {\n            "acc": round(acc, 4),\n            "raw_acc": round(raw_acc, 4),\n            "count": count,\n            "correct": correct,\n            "factor": round(factor, 4),\n        }\n\n        if count >= ASK_ROAD_MEMORY_MIN_COUNT and acc > best_acc:\n            best_acc = acc\n            best_name = name\n\n    result["models"] = models\n    if best_name:\n        result["label"] = f"問路記憶:{best_name}較準 {int(best_acc * 100)}%"\n    else:\n        result["label"] = f"問路記憶暖機中 樣本{len(records)}"\n\n    return result\n\n\ndef _ask_road_factor(performance: Optional[Dict[str, Any]], road_key: str, default: float = 1.0) -> float:\n    try:\n        if not (USE_ASK_ROAD_MEMORY and performance):\n            return default\n        return float((performance.get("models") or {}).get(road_key, {}).get("factor", default))\n    except Exception:\n        return default\n\n\ndef _apply_ask_road_factor_to_score(score: Dict[str, Any], road_key: str, performance: Optional[Dict[str, Any]]) -> Dict[str, Any]:\n    # 用問路近期命中率微調 HYBRID 下三路 B/P 邊際。\n    # factor>1：放大這條路的邊際；factor<1：縮小這條路的邊際。\n    if not (USE_ASK_ROAD_MEMORY and ASK_ROAD_MEMORY_APPLY_TO_HYBRID and performance and isinstance(score, dict)):\n        return score\n\n    factor = _ask_road_factor(performance, road_key, 1.0)\n    if abs(factor - 1.0) < 0.0001:\n        return score\n\n    try:\n        b = float(score.get("B", 0.5))\n        p = float(score.get("P", 0.5))\n        side_total = max(0.0001, b + p)\n        b_side = b / side_total\n        edge = b_side - 0.5\n        new_b_side = _clamp(0.5 + edge * factor, SIDE_CLAMP_MIN, SIDE_CLAMP_MAX)\n        new_score = dict(score)\n        new_score["B"] = round(new_b_side, 5)\n        new_score["P"] = round(1.0 - new_b_side, 5)\n        new_score["ask_road_memory_factor"] = round(factor, 4)\n        new_score["ask_road_memory"] = (performance.get("models") or {}).get(road_key, {})\n        old_label = str(new_score.get("label", ""))\n        new_score["label"] = f"{old_label}|問路記憶x{factor:.2f}" if old_label else f"問路記憶x{factor:.2f}"\n        return new_score\n    except Exception:\n        return score\n\n\ndef _apply_ask_road_factor_to_vote(vote: Dict[str, Any], road_key: str, performance: Optional[Dict[str, Any]]) -> Dict[str, Any]:\n    # 用問路近期命中率微調 FUHAO 下三路票。\n    # 若某條路近期明顯失準，且設定允許，會暫時清掉該票，避免壞路拖累多數決。\n    if not (USE_ASK_ROAD_MEMORY and ASK_ROAD_MEMORY_APPLY_TO_FUHAO and performance and isinstance(vote, dict)):\n        return vote\n\n    models = performance.get("models") or {}\n    stat = models.get(road_key, {})\n    if not stat:\n        return vote\n\n    factor = float(stat.get("factor", 1.0))\n    count = int(stat.get("count", 0))\n    acc = float(stat.get("acc", 0.5))\n\n    new_vote = dict(vote)\n    old_conf = float(new_vote.get("confidence", 0.0) or 0.0)\n    new_vote["confidence"] = round(_clamp(old_conf * factor, 0.0, 0.88), 4)\n    new_vote["ask_road_memory_factor"] = round(factor, 4)\n    new_vote["ask_road_memory"] = stat\n\n    if (\n        ASK_ROAD_MEMORY_DROP_BAD_VOTE\n        and count >= ASK_ROAD_MEMORY_MIN_COUNT\n        and acc <= ASK_ROAD_MEMORY_BAD_VOTE_ACC\n        and new_vote.get("pick") in {"B", "P"}\n    ):\n        old_pick = new_vote.get("pick", "")\n        new_vote["pick"] = ""\n        new_vote["label"] = f"{new_vote.get(\'label\', \'\')}|問路記憶暫停{_fuhao_side_name(old_pick)}票 acc{acc:.2f}"\n    else:\n        new_vote["label"] = f"{new_vote.get(\'label\', \'\')}|問路記憶x{factor:.2f}"\n\n    return new_vote\n\n\ndef _store_ask_road_pending(training_key: str, non_tie: List[str], predictions: Dict[str, str]) -> None:\n    if not (USE_ASK_ROAD_MEMORY and predictions):\n        return\n\n    clean = {str(k): v for k, v in predictions.items() if v in {"B", "P"}}\n    if not clean:\n        return\n\n    state = _ask_road_state(training_key)\n    state["pending"] = {\n        "non_tie_len": len(non_tie),\n        "predictions": clean,\n    }\n\n\ndef get_ask_road_state_info() -> Dict[str, Any]:\n    return {\n        "enabled": USE_ASK_ROAD_MEMORY,\n        "size": len(_ASK_ROAD_STATE),\n        "keys": list(_ASK_ROAD_STATE.keys())[-30:],\n    }\n\n\ndef clear_ask_road_state() -> Dict[str, Any]:\n    removed = len(_ASK_ROAD_STATE)\n    _ASK_ROAD_STATE.clear()\n    return {"ok": True, "removed": removed}'

COLUMN_SHAPE_FUNC = 'def _score_column_shape(non_tie: List[str], candidate: str) -> Dict[str, Any]:\n    # 前排大路欄型分：\n    # 讓候選判斷不只看下三路紅藍，也看下一口落點是否符合最近欄高節奏。\n    if not USE_DERIVED_COLUMN_SHAPE or candidate not in {"B", "P"} or len(non_tie) < ROAD_ENGINE_MIN_HISTORY:\n        return {"score": 0.5, "edge": 0.0, "label": "欄型關閉或資料不足", "tail_heights": []}\n\n    try:\n        before_layout = _build_big_road(non_tie)\n        after_layout = _build_big_road(non_tie + [candidate])\n        move = _classify_bigroad_move(before_layout, after_layout, candidate)\n        move_type = move.get("move_type", "NONE")\n\n        after_last = after_layout.get("last", {}) or {}\n        col = int(after_last.get("col", 0))\n        row = int(after_last.get("row", 0))\n        heights = after_layout.get("col_heights", {}) or {}\n\n        lookback = max(3, DERIVED_COLUMN_SHAPE_LOOKBACK)\n        start_col = max(0, col - lookback)\n        prev_heights = [int(heights.get(c, 0)) for c in range(start_col, col) if int(heights.get(c, 0)) > 0]\n        tail_heights = prev_heights[-lookback:]\n        current_height = int(heights.get(col, 0))\n\n        if not tail_heights:\n            return {"score": 0.5, "edge": 0.0, "label": "欄型樣本不足", "tail_heights": [], "current_height": current_height, "move_type": move_type}\n\n        avg_h = sum(tail_heights) / len(tail_heights)\n        max_h = max(tail_heights)\n        min_h = min(tail_heights)\n        last_h = tail_heights[-1]\n        repeated = tail_heights.count(last_h) >= max(2, len(tail_heights) // 2)\n\n        edge = 0.0\n        reasons = []\n\n        if move_type == "VERTICAL_DROP":\n            # 直落如果仍在前排欄高範圍內，視為欄型延續；若超出太多，視為可能疲乏。\n            if current_height <= max_h + 1:\n                edge += DERIVED_COLUMN_NEAT_BONUS\n                reasons.append("直落貼近前排欄高")\n            else:\n                edge -= DERIVED_COLUMN_BREAK_PENALTY\n                reasons.append("直落超出前排欄高")\n        elif move_type == "NEW_COLUMN":\n            # 換邊開新欄：如果前一欄高度接近近期欄型，代表上一欄完成得較漂亮。\n            prev_col_h = int(heights.get(col - 1, 0))\n            if repeated and prev_col_h == last_h:\n                edge += DERIVED_COLUMN_NEAT_BONUS * 0.85\n                reasons.append("新欄承接重複欄型")\n            elif abs(prev_col_h - avg_h) <= 1.0:\n                edge += DERIVED_COLUMN_NEAT_BONUS * 0.55\n                reasons.append("新欄承接平均欄高")\n            else:\n                edge -= DERIVED_COLUMN_BREAK_PENALTY * 0.60\n                reasons.append("新欄前欄破欄型")\n        elif move_type == "SIDE_DRAG":\n            # 橫拖代表到底/卡位，通常要保守一點；若近期欄高本來就很高，扣分較小。\n            if max_h >= ROAD_ENGINE_ROWS - 1:\n                edge -= DERIVED_COLUMN_DRAG_PENALTY * 0.55\n                reasons.append("高欄橫拖")\n            else:\n                edge -= DERIVED_COLUMN_DRAG_PENALTY\n                reasons.append("黏邊橫拖")\n        else:\n            if abs(current_height - avg_h) <= 1.0:\n                edge += DERIVED_COLUMN_NEAT_BONUS * 0.35\n                reasons.append("欄高接近平均")\n\n        # 如果最近欄型很整齊，候選造成明顯偏離就扣分。\n        if repeated and current_height not in {last_h, last_h + 1, 1}:\n            edge -= DERIVED_COLUMN_BREAK_PENALTY * 0.45\n            reasons.append("偏離重複欄型")\n\n        edge = _clamp(edge, -DERIVED_COLUMN_MAX_EDGE, DERIVED_COLUMN_MAX_EDGE)\n\n        return {\n            "score": round(0.5 + edge, 5),\n            "edge": round(edge, 5),\n            "label": "+".join(reasons) if reasons else "欄型中性",\n            "move_type": move_type,\n            "row": row,\n            "col": col,\n            "current_height": current_height,\n            "tail_heights": tail_heights,\n            "avg_height": round(avg_h, 3),\n            "max_height": max_h,\n            "min_height": min_h,\n        }\n    except Exception as e:\n        return {"score": 0.5, "edge": 0.0, "label": f"欄型錯誤:{e}", "tail_heights": []}'

COMBINE_FUNC = 'def _combine_candidate_scores(color_score: float, structure_score: float, column_score: Optional[float] = None) -> float:\n    # 合併候選分數：\n    # color_score=紅藍節奏分；structure_score=齊整 / 有無 / 直落結構分；column_score=前排大路欄型分。\n    cw_raw = DERIVED_COLOR_SCORE_WEIGHT\n    sw_raw = DERIVED_STRUCTURE_SCORE_WEIGHT\n    col_raw = DERIVED_COLUMN_SHAPE_WEIGHT if (USE_DERIVED_COLUMN_SHAPE and column_score is not None) else 0.0\n\n    total_w = max(0.0001, cw_raw + sw_raw + col_raw)\n    cw = cw_raw / total_w\n    sw = sw_raw / total_w\n    colw = col_raw / total_w\n\n    result = float(color_score) * cw + float(structure_score) * sw\n    if column_score is not None and colw > 0:\n        result += float(column_score) * colw\n    return result'

DERIVED_ROAD_SCORE_BLOCK = 'def _derived_road_score(non_tie: List[str], offset: int, road_key: str, display_name: str) -> Dict[str, Any]:\n    # 下三路候選模擬 + 路單結構 + 前排欄型判斷版。\n    # 不再用「紅=跟、藍=反」這種簡化邏輯。\n    default = {\n        "B": 0.5,\n        "P": 0.5,\n        "label": f"{display_name}資料不足",\n        "strength": 0.0,\n        "road_key": road_key,\n        "stats": {"last": 0, "red_rate": 0.5, "blue_rate": 0.5, "count": 0, "tail": ""},\n        "red_pressure": 0.5,\n        "blue_pressure": 0.5,\n        "candidate": {},\n    }\n\n    if not USE_ROAD_ENGINE or len(non_tie) < ROAD_ENGINE_MIN_HISTORY:\n        return default\n\n    layout = _build_big_road(non_tie)\n    series = _derived_series(layout, offset=offset)\n    stats = _color_stats(series)\n    count = int(stats.get("count", 0))\n\n    if count < DERIVED_ROAD_MIN_COUNT:\n        return {\n            **default,\n            "stats": stats,\n            "label": f"{display_name}樣本不足",\n        }\n\n    b_info = _candidate_derived_color_info(non_tie, "B", offset)\n    p_info = _candidate_derived_color_info(non_tie, "P", offset)\n\n    b_color_eval = _score_candidate_color_pattern(series, int(b_info.get("new_color", 0)))\n    p_color_eval = _score_candidate_color_pattern(series, int(p_info.get("new_color", 0)))\n\n    b_struct_eval = _score_candidate_structure(b_info, series)\n    p_struct_eval = _score_candidate_structure(p_info, series)\n\n    b_column_eval = _score_column_shape(non_tie, "B")\n    p_column_eval = _score_column_shape(non_tie, "P")\n\n    b_score = _combine_candidate_scores(\n        float(b_color_eval.get("score", 0.5)),\n        float(b_struct_eval.get("score", 0.5)),\n        float(b_column_eval.get("score", 0.5)),\n    )\n    p_score = _combine_candidate_scores(\n        float(p_color_eval.get("score", 0.5)),\n        float(p_struct_eval.get("score", 0.5)),\n        float(p_column_eval.get("score", 0.5)),\n    )\n\n    b, p, edge = _candidate_scores_to_side_prob(b_score, p_score, max_edge=DERIVED_CANDIDATE_MAX_EDGE)\n\n    if edge < DERIVED_CANDIDATE_MIN_EDGE:\n        label = f"{display_name}候選接近"\n        strength = 0.06\n    else:\n        pick = "莊" if b > p else "閒"\n        label = f"{display_name}路單候選偏{pick}"\n        strength = 0.10 + min(0.15, edge * 2.0)\n\n    red_rate = float(stats.get("red_rate", 0.5))\n    blue_rate = float(stats.get("blue_rate", 0.5))\n\n    return {\n        "B": round(b, 5),\n        "P": round(p, 5),\n        "label": label,\n        "strength": round(strength, 4),\n        "road_key": road_key,\n        "stats": stats,\n        "red_pressure": round(red_rate, 4),\n        "blue_pressure": round(blue_rate, 4),\n        "tail": stats.get("tail", ""),\n        "candidate": {\n            "B": {\n                "new_color": b_info.get("new_color_text", "N"),\n                "color_score": round(float(b_color_eval.get("score", 0.5)), 5),\n                "structure_score": round(float(b_struct_eval.get("score", 0.5)), 5),\n                "column_score": round(float(b_column_eval.get("score", 0.5)), 5),\n                "score": round(b_score, 5),\n                "color_eval": b_color_eval,\n                "structure_eval": b_struct_eval,\n                "column_eval": b_column_eval,\n                "pos": b_info.get("pos", {}),\n                "structure": b_info.get("structure", {}),\n            },\n            "P": {\n                "new_color": p_info.get("new_color_text", "N"),\n                "color_score": round(float(p_color_eval.get("score", 0.5)), 5),\n                "structure_score": round(float(p_struct_eval.get("score", 0.5)), 5),\n                "column_score": round(float(p_column_eval.get("score", 0.5)), 5),\n                "score": round(p_score, 5),\n                "color_eval": p_color_eval,\n                "structure_eval": p_struct_eval,\n                "column_eval": p_column_eval,\n                "pos": p_info.get("pos", {}),\n                "structure": p_info.get("structure", {}),\n            },\n            "edge": round(edge, 5),\n            "diff": round(b_score - p_score, 5),\n        },\n    }'

FUHAO_DOWN3_VOTE_BLOCK = 'def _fuhao_down3_vote(non_tie: List[str], offset: int, name: str) -> Dict[str, Any]:\n    # 富濠式下三路候選模擬 + 路單結構 + 前排欄型判斷版。\n    if len(non_tie) < FUHAO_MIN_VALID_ROUNDS:\n        return {\n            "pick": "",\n            "label": f"{name}資料不足",\n            "confidence": 0.0,\n            "stats": {},\n            "candidate": {},\n        }\n\n    layout = _build_big_road(non_tie)\n    series = _derived_series(layout, offset=offset)\n    stats = _color_stats(series)\n    count = int(stats.get("count", 0))\n\n    if count < DERIVED_ROAD_MIN_COUNT:\n        return {\n            "pick": "",\n            "label": f"{name}樣本不足",\n            "confidence": 0.0,\n            "stats": stats,\n            "candidate": {},\n        }\n\n    b_info = _candidate_derived_color_info(non_tie, "B", offset)\n    p_info = _candidate_derived_color_info(non_tie, "P", offset)\n\n    b_color_eval = _score_candidate_color_pattern(series, int(b_info.get("new_color", 0)))\n    p_color_eval = _score_candidate_color_pattern(series, int(p_info.get("new_color", 0)))\n\n    b_struct_eval = _score_candidate_structure(b_info, series)\n    p_struct_eval = _score_candidate_structure(p_info, series)\n\n    b_column_eval = _score_column_shape(non_tie, "B")\n    p_column_eval = _score_column_shape(non_tie, "P")\n\n    b_score = _combine_candidate_scores(\n        float(b_color_eval.get("score", 0.5)),\n        float(b_struct_eval.get("score", 0.5)),\n        float(b_column_eval.get("score", 0.5)),\n    )\n    p_score = _combine_candidate_scores(\n        float(p_color_eval.get("score", 0.5)),\n        float(p_struct_eval.get("score", 0.5)),\n        float(p_column_eval.get("score", 0.5)),\n    )\n\n    diff = b_score - p_score\n\n    if abs(diff) < FUHAO_DOWN3_MIN_DIFF:\n        pick = ""\n        label = f"{name}候選差距不足"\n        confidence = 0.42\n    else:\n        pick = "B" if diff > 0 else "P"\n        label = f"{name}路單候選偏{_fuhao_side_name(pick)}"\n        confidence = min(0.80, 0.50 + abs(diff) * 1.35 + min(0.08, count * 0.006))\n\n    return {\n        "pick": pick,\n        "label": label,\n        "confidence": round(confidence, 4),\n        "stats": stats,\n        "candidate": {\n            "B": {\n                "new_color": b_info.get("new_color_text", "N"),\n                "color_score": round(float(b_color_eval.get("score", 0.5)), 5),\n                "structure_score": round(float(b_struct_eval.get("score", 0.5)), 5),\n                "column_score": round(float(b_column_eval.get("score", 0.5)), 5),\n                "score": round(b_score, 5),\n                "color_eval": b_color_eval,\n                "structure_eval": b_struct_eval,\n                "column_eval": b_column_eval,\n                "pos": b_info.get("pos", {}),\n                "structure": b_info.get("structure", {}),\n            },\n            "P": {\n                "new_color": p_info.get("new_color_text", "N"),\n                "color_score": round(float(p_color_eval.get("score", 0.5)), 5),\n                "structure_score": round(float(p_struct_eval.get("score", 0.5)), 5),\n                "column_score": round(float(p_column_eval.get("score", 0.5)), 5),\n                "score": round(p_score, 5),\n                "color_eval": p_color_eval,\n                "structure_eval": p_struct_eval,\n                "column_eval": p_column_eval,\n                "pos": p_info.get("pos", {}),\n                "structure": p_info.get("structure", {}),\n            },\n            "diff": round(diff, 5),\n        },\n    }'


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


def upsert_env(text: str) -> str:
    if "USE_ASK_ROAD_MEMORY" in text and "USE_DERIVED_COLUMN_SHAPE" in text:
        return text

    marker = "# End Candidate Down-Road Simulation"
    if marker in text:
        return text.replace(marker, marker + "\n" + ASK_COLUMN_ENV, 1)

    target = 'DERIVED_ROAD_MIN_COUNT = int(os.getenv("DERIVED_ROAD_MIN_COUNT", "3"))'
    idx = text.find(target)
    if idx < 0:
        fail("找不到 DERIVED_ROAD_MIN_COUNT，無法插入環境變數。")
    line_end = text.find("\n", idx)
    if line_end < 0:
        line_end = idx + len(target)
    return text[:line_end + 1] + ASK_COLUMN_ENV + "\n" + text[line_end + 1:]


def upsert_global_state(text: str) -> str:
    if "_ASK_ROAD_STATE" in text:
        return text
    marker = "_WALK_FORWARD_STATE: Dict[str, Dict[str, Any]] = {}\n"
    if marker not in text:
        fail("找不到 _WALK_FORWARD_STATE，無法插入 _ASK_ROAD_STATE。")
    insert = """\n# Ask Road Hit Memory：每個 LINE UID / 場館 / 房間 / 靴號 的問路命中率記憶。\n# 只保存上一輪問路票 pending，以及最近 N 次問路票是否命中的紀錄。\n_ASK_ROAD_STATE: Dict[str, Dict[str, Any]] = {}\n"""
    return text.replace(marker, marker + insert, 1)


def upsert_ask_memory_block(text: str) -> str:
    if "# ============ Ask Road Hit Memory：問路命中率記憶 ===========" in text:
        return text
    marker = "# ============ Pattern Replay Memory"
    if marker not in text:
        fail("找不到 Pattern Replay Memory 區塊，無法插入 Ask Road Memory。")
    return text.replace(marker, ASK_ROAD_MEMORY_BLOCK + "\n\n\n" + marker, 1)


def upsert_column_shape(text: str) -> str:
    if "def _candidate_derived_color_info(" not in text:
        fail("目前 predictor.py 還沒有下三路候選 helper。請先使用完整 predictor_ask_road_memory_full.py 覆蓋，或先執行路單盤面引擎升級。")

    if "def _score_column_shape(" not in text:
        marker = "def _combine_candidate_scores("
        if marker not in text:
            fail("找不到 _combine_candidate_scores，無法插入 Column Shape Score。")
        text = text.replace(marker, COLUMN_SHAPE_FUNC + "\n\n\n" + marker, 1)

    pattern = re.compile(r"def _combine_candidate_scores\(.*?\n(?=def _candidate_scores_to_side_prob\()", flags=re.DOTALL)
    text, n = pattern.subn(COMBINE_FUNC.rstrip() + "\n\n\n", text, count=1)
    if n == 0:
        fail("找不到可替換的 _combine_candidate_scores。")
    return text


def patch_predict_flow(text: str) -> str:
    old = """    _update_walk_forward_truth(training_key, non_tie)\n    live_walk_forward_performance = _get_walk_forward_performance(training_key)\n"""
    new = """    _update_walk_forward_truth(training_key, non_tie)\n    _update_ask_road_truth(training_key, non_tie)\n    live_walk_forward_performance = _get_walk_forward_performance(training_key)\n    ask_road_performance = _get_ask_road_performance(training_key)\n"""
    if old in text and "ask_road_performance = _get_ask_road_performance(training_key)" not in text:
        text = text.replace(old, new, 1)

    old = """    road_family = _road_family_scores(non_tie)\n    big_road = road_family.get("big_road", {"B": 0.5, "P": 0.5, "label": "大路資料不足"})\n    big_eye = road_family.get("big_eye", {"B": 0.5, "P": 0.5, "label": "大眼仔資料不足"})\n    small_road = road_family.get("small_road", {"B": 0.5, "P": 0.5, "label": "小路資料不足"})\n    cockroach = road_family.get("cockroach", {"B": 0.5, "P": 0.5, "label": "蟑螂路資料不足"})\n    road_consensus = road_family.get("consensus", {"B": 0.5, "P": 0.5, "label": "四路共識資料不足"})\n"""
    new = """    road_family = _road_family_scores(non_tie)\n\n    # Ask Road Hit Memory：依本靴最近問路命中率，微調大眼仔 / 小路 / 蟑螂路邊際。\n    if USE_ASK_ROAD_MEMORY and ASK_ROAD_MEMORY_APPLY_TO_HYBRID:\n        for _rk in ["big_eye", "small_road", "cockroach"]:\n            if _rk in road_family:\n                road_family[_rk] = _apply_ask_road_factor_to_score(road_family[_rk], _rk, ask_road_performance)\n        try:\n            road_family["consensus"] = _road_consensus_score({\n                "big_road": road_family.get("big_road", {}),\n                "big_eye": road_family.get("big_eye", {}),\n                "small_road": road_family.get("small_road", {}),\n                "cockroach": road_family.get("cockroach", {}),\n            })\n        except Exception:\n            pass\n\n    big_road = road_family.get("big_road", {"B": 0.5, "P": 0.5, "label": "大路資料不足"})\n    big_eye = road_family.get("big_eye", {"B": 0.5, "P": 0.5, "label": "大眼仔資料不足"})\n    small_road = road_family.get("small_road", {"B": 0.5, "P": 0.5, "label": "小路資料不足"})\n    cockroach = road_family.get("cockroach", {"B": 0.5, "P": 0.5, "label": "蟑螂路資料不足"})\n    road_consensus = road_family.get("consensus", {"B": 0.5, "P": 0.5, "label": "四路共識資料不足"})\n"""
    if old in text and "# Ask Road Hit Memory：依本靴最近問路命中率" not in text:
        text = text.replace(old, new, 1)

    old = """        f"四路:{road_consensus.get('label', '')}",\n        f"生命周期:{lifecycle.get('label', '')}",\n"""
    new = """        f"四路:{road_consensus.get('label', '')}",\n        f"問路記憶:{ask_road_performance.get('label', '')}",\n        f"生命周期:{lifecycle.get('label', '')}",\n"""
    if old in text and "ask_road_performance.get('label'" not in text:
        text = text.replace(old, new, 1)

    old = """    if recommend in {"B", "P"}:\n        current_model_picks["final"] = recommend\n    _store_walk_forward_pending(training_key, non_tie, current_model_picks)\n"""
    new = """    if recommend in {"B", "P"}:\n        current_model_picks["final"] = recommend\n    _store_walk_forward_pending(training_key, non_tie, current_model_picks)\n\n    ask_road_pending = {\n        "big_eye": _pick_from_score(big_eye, min_edge=0.002),\n        "small_road": _pick_from_score(small_road, min_edge=0.002),\n        "cockroach": _pick_from_score(cockroach, min_edge=0.002),\n        "road_majority": road_consensus.get("pick", ""),\n        "final": recommend if recommend in {"B", "P"} else "",\n    }\n    _store_ask_road_pending(training_key, non_tie, ask_road_pending)\n"""
    if old in text and "ask_road_pending = {" not in text:
        text = text.replace(old, new, 1)

    old = """        "live_walk_forward_performance": live_walk_forward_performance,\n        "walk_forward_enabled": USE_WALK_FORWARD_LEARNING,\n"""
    new = """        "live_walk_forward_performance": live_walk_forward_performance,\n        "ask_road_memory": ask_road_performance,\n        "ask_road_memory_label": ask_road_performance.get("label", ""),\n        "ask_road_memory_enabled": USE_ASK_ROAD_MEMORY,\n        "walk_forward_enabled": USE_WALK_FORWARD_LEARNING,\n"""
    if old in text and '"ask_road_memory": ask_road_performance' not in text:
        text = text.replace(old, new, 1)

    return text


def patch_fuhao_flow(text: str) -> str:
    old = """    tie_count = history.count("T") if FUHAO_KEEP_TIE_COUNT else 0\n    valid_len = len(non_tie)\n    recommend_text_map = {"B": "莊", "P": "閒", "T": "和", "NONE": "觀望"}\n"""
    new = """    tie_count = history.count("T") if FUHAO_KEEP_TIE_COUNT else 0\n    valid_len = len(non_tie)\n    training_key = f"{user_id or 'anonymous'}|FUHAO_CLONE|{venue}|{room}|{shoe_id}"\n    _update_ask_road_truth(training_key, non_tie)\n    ask_road_performance = _get_ask_road_performance(training_key)\n    recommend_text_map = {"B": "莊", "P": "閒", "T": "和", "NONE": "觀望"}\n"""
    if old in text and "ask_road_performance = _get_ask_road_performance(training_key)" not in text:
        text = text.replace(old, new, 1)

    old = """            "training_key": f"{user_id or 'anonymous'}|FUHAO_CLONE",\n            "model_cache_size": len(_MODEL_CACHE),\n"""
    new = """            "training_key": training_key,\n            "ask_road_memory": ask_road_performance,\n            "ask_road_memory_label": ask_road_performance.get("label", ""),\n            "ask_road_memory_enabled": USE_ASK_ROAD_MEMORY,\n            "model_cache_size": len(_MODEL_CACHE),\n"""
    if old in text:
        text = text.replace(old, new, 1)

    patches = [
        (
            """        road_models["big_eye"] = _fuhao_down3_vote(non_tie, 1, "大眼仔")\n        road_votes.append(road_models["big_eye"].get("pick", ""))\n""",
            """        road_models["big_eye"] = _fuhao_down3_vote(non_tie, 1, "大眼仔")\n        road_models["big_eye"] = _apply_ask_road_factor_to_vote(road_models["big_eye"], "big_eye", ask_road_performance)\n        road_votes.append(road_models["big_eye"].get("pick", ""))\n""",
        ),
        (
            """        road_models["small_road"] = _fuhao_down3_vote(non_tie, 2, "小路")\n        road_votes.append(road_models["small_road"].get("pick", ""))\n""",
            """        road_models["small_road"] = _fuhao_down3_vote(non_tie, 2, "小路")\n        road_models["small_road"] = _apply_ask_road_factor_to_vote(road_models["small_road"], "small_road", ask_road_performance)\n        road_votes.append(road_models["small_road"].get("pick", ""))\n""",
        ),
        (
            """        road_models["cockroach"] = _fuhao_down3_vote(non_tie, 3, "蟑螂路")\n        road_votes.append(road_models["cockroach"].get("pick", ""))\n""",
            """        road_models["cockroach"] = _fuhao_down3_vote(non_tie, 3, "蟑螂路")\n        road_models["cockroach"] = _apply_ask_road_factor_to_vote(road_models["cockroach"], "cockroach", ask_road_performance)\n        road_votes.append(road_models["cockroach"].get("pick", ""))\n""",
        ),
    ]
    for old, new in patches:
        if old in text and new not in text:
            text = text.replace(old, new, 1)

    old = """    reason_parts = [\n        road_consensus_label,\n        advanced_label,\n"""
    new = """    reason_parts = [\n        road_consensus_label,\n        advanced_label,\n        f"問路記憶:{ask_road_performance.get('label', '')}",\n"""
    if old in text and "ask_road_performance.get('label'" not in text:
        text = text.replace(old, new, 1)

    old = """    training_key = f"{user_id or 'anonymous'}|FUHAO_CLONE|{venue}|{room}|{shoe_id}"\n\n    result = {\n"""
    new = """    # training_key 已於函數前段建立，供 Ask Road Memory 使用。\n\n    ask_road_pending = {\n        "big_road": road_models.get("big_road", {}).get("pick", ""),\n        "big_eye": road_models.get("big_eye", {}).get("pick", ""),\n        "small_road": road_models.get("small_road", {}).get("pick", ""),\n        "cockroach": road_models.get("cockroach", {}).get("pick", ""),\n        "road_majority": road_majority.get("pick", ""),\n        "final": recommend if recommend in {"B", "P"} else "",\n    }\n    _store_ask_road_pending(training_key, non_tie, ask_road_pending)\n\n    result = {\n"""
    if old in text:
        text = text.replace(old, new, 1)

    old = """        "dynamic_weights": dynamic_weights,\n        "online_model_performance": {},\n"""
    new = """        "dynamic_weights": dynamic_weights,\n        "ask_road_memory": ask_road_performance,\n        "ask_road_memory_label": ask_road_performance.get("label", ""),\n        "ask_road_memory_enabled": USE_ASK_ROAD_MEMORY,\n        "online_model_performance": {},\n"""
    if old in text and '"ask_road_memory": ask_road_performance' not in text:
        text = text.replace(old, new, 1)

    return text


def main() -> None:
    if not PREDICTOR_PATH.exists():
        fail("目前資料夾找不到 predictor.py，請把本腳本放在 predictor.py 同一層。")

    text = PREDICTOR_PATH.read_text(encoding="utf-8")
    backup_path = PREDICTOR_PATH.with_name(f"predictor.backup-ask-road-memory-{datetime.now():%Y%m%d-%H%M%S}.py")
    backup_path.write_text(text, encoding="utf-8")

    text = upsert_env(text)
    text = upsert_global_state(text)
    text = upsert_ask_memory_block(text)
    text = upsert_column_shape(text)
    text = replace_between(text, "def _derived_road_score(", "def _big_eye_score(", DERIVED_ROAD_SCORE_BLOCK)
    text = replace_between(text, "def _fuhao_down3_vote(", "def _fuhao_deep_parity_vote(", FUHAO_DOWN3_VOTE_BLOCK)
    text = patch_predict_flow(text)
    text = patch_fuhao_flow(text)

    PREDICTOR_PATH.write_text(text, encoding="utf-8")

    print("[OK] predictor.py 已加入 Ask Road Hit Memory + Column Shape Score。")
    print(f"[OK] 原檔備份：{backup_path.name}")
    print("[OK] 修改範圍：問路記憶、欄型分、_derived_road_score、_fuhao_down3_vote、predict/FUHAO 小幅接線。")
    print("[OK] 其餘原先模型主流程沒有動刀。")
    print("[OK] 請執行：python -m py_compile predictor.py")


if __name__ == "__main__":
    main()
