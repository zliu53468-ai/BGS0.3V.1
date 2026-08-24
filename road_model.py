"""BGS V10.8 牌路模型入口。

本檔保留六個牌路專家，並完全移除一至三階條件轉移模型：
- full_road：完整歷史牌路規劃模型
- short／mid／long／pattern／analogue：其餘近期牌路專家

主引擎可以分別取得 ``planning_probability`` 與 ``recent_probability``，
避免把全部牌路訊號壓成一個只追近期的群組。
"""
from __future__ import annotations

from hashlib import sha256
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import math

import numpy as np

from full_road_pattern_model import analyze_full_road_pattern


# 正式牌路特徵固定在程式碼內，避免 Render 的舊 ROAD_* 值在重啟後使同一張
# 路紙得到不同 context。這些是特徵窗口，不是勝率或下注參數。
ROAD_HISTORY_LIMIT = 500
ROAD_SHORT_WINDOW = 8
ROAD_MID_WINDOW = 18
ROAD_LONG_WINDOW = 36
ROAD_RECENT_SIMULATIONS = 3000
ROAD_RECENT_MAX_MODEL_WEIGHT = 0.22
ROAD_RECENT_MAX_DISAGREEMENT = 0.145


def normalize_raw_outcomes(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for item in values:
        if isinstance(item, Mapping):
            raw = item.get("outcome") or item.get("actual") or item.get("virtual_outcome")
        else:
            raw = item
        value = str(raw or "").upper().strip()
        if value in {"B", "P", "T"}:
            result.append(value)
    return result[-max(ROAD_HISTORY_LIMIT * 2, 200):]


def normalize_road_sequence(values: Iterable[Any]) -> List[str]:
    return [
        value for value in normalize_raw_outcomes(values)
        if value in {"B", "P"}
    ][-ROAD_HISTORY_LIMIT:]


def _clip_probability(value: Any) -> float:
    try:
        return max(0.02, min(0.98, float(value)))
    except Exception:
        return 0.5


def _runs(sequence: Sequence[str]) -> List[Tuple[str, int]]:
    result: List[Tuple[str, int]] = []
    for value in sequence:
        if result and result[-1][0] == value:
            result[-1] = (value, result[-1][1] + 1)
        else:
            result.append((value, 1))
    return result


def _continuation_probability(
    sequence: Sequence[str],
    decay: float,
) -> tuple[float, float]:
    """只計算「延續或反轉」，完全不統計莊／閒各自出現次數。

    舊版直接把最近窗口的 B/P 數量做指數加權；例如最近莊多，短期模型便
    自動偏莊，即使牌路結構已經轉折。這裡改成對相鄰兩手的關係做 Beta
    平滑：``P(continue) = (2 + Σw·I[x_t=x_{t-1}]) / (4 + Σw)``。
    最後才依目前最後一手把「續／反」映射為 B 或 P，故整個模型對交換
    B/P 標籤保持對稱，不會因哪一邊累積較多而偏移。
    """
    values = list(sequence)
    if len(values) < 2:
        return 0.5, 0.0
    continue_mass = 2.0
    reverse_mass = 2.0
    for reverse_index, (previous, current) in enumerate(
        reversed(list(zip(values, values[1:])))
    ):
        weight = float(decay) ** reverse_index
        if previous == current:
            continue_mass += weight
        else:
            reverse_mass += weight
    total = continue_mass + reverse_mass
    return continue_mass / max(1e-12, total), total - 4.0


def _window_model(sequence: Sequence[str], size: int, decay: float) -> Dict[str, Any]:
    window = list(sequence[-size:])
    continuation_probability, effective_transitions = _continuation_probability(
        window, decay
    )
    support = len(window)
    last = window[-1] if window else "B"
    probability = (
        continuation_probability
        if last == "B"
        else 1.0 - continuation_probability
    )
    reliability = min(0.82, support / max(1.0, float(size)))
    return {
        "active": support >= min(4, size),
        "support": support,
        "banker_probability": probability,
        "direction": "B" if probability >= 0.5 else "P",
        "edge": abs(probability - 0.5) * 2.0,
        "reliability": reliability,
        "window": size,
        "decay": decay,
        "continuation_probability": continuation_probability,
        "effective_transition_support": effective_transitions,
        "direction_semantics": "continuation_or_reversal_not_bp_frequency",
    }


def _analogue_model(sequence: Sequence[str]) -> Dict[str, Any]:
    transitions = [
        "C" if current == previous else "R"
        for previous, current in zip(sequence, sequence[1:])
    ]
    for order in (6, 5, 4, 3, 2):
        if len(transitions) < order + 1:
            continue
        suffix = tuple(transitions[-order:])
        continued = 2.5
        reversed_mass = 2.5
        support = 0
        for next_index in range(order + 1, len(sequence)):
            prefix_transitions = transitions[:next_index - 1]
            if tuple(prefix_transitions[-order:]) != suffix:
                continue
            support += 1
            if sequence[next_index] == sequence[next_index - 1]:
                continued += 1.0
            else:
                reversed_mass += 1.0
        if support >= 2:
            continuation_probability = continued / (continued + reversed_mass)
            probability = (
                continuation_probability
                if sequence[-1] == "B"
                else 1.0 - continuation_probability
            )
            return {
                "active": True,
                "support": support,
                "banker_probability": probability,
                "direction": "B" if probability >= 0.5 else "P",
                "edge": abs(probability - 0.5) * 2.0,
                "reliability": min(0.76, support / 10.0),
                "order": order,
                "continuation_probability": continuation_probability,
                "direction_semantics": "relative_transition_analogue",
            }
    return {
        "active": False,
        "support": 0,
        "banker_probability": 0.5,
        "direction": "B",
        "reliability": 0.0,
        "order": 0,
        "direction_semantics": "relative_transition_analogue",
    }


def _pattern_features(sequence: Sequence[str]) -> np.ndarray:
    seq = list(sequence)
    runs = _runs(seq)
    lengths = [length for _, length in runs]
    recent_runs = lengths[-10:]

    def change_rate(size: int) -> float:
        window = seq[-size:]
        if len(window) < 2:
            return 0.5
        return sum(a != b for a, b in zip(window, window[1:])) / (len(window) - 1)

    mean_run = float(np.mean(recent_runs)) if recent_runs else 0.0
    variance = float(np.var(recent_runs)) if recent_runs else 0.0
    return np.asarray([
        min(1.0, (recent_runs[-1] if recent_runs else 0) / 6.0),
        change_rate(8),
        change_rate(18),
        change_rate(36),
        min(1.0, mean_run / 5.0),
        min(1.0, variance / 8.0),
        sum(value == 1 for value in recent_runs) / max(1, len(recent_runs)),
        sum(value == 2 for value in recent_runs) / max(1, len(recent_runs)),
        sum(value >= 4 for value in recent_runs) / max(1, len(recent_runs)),
    ], dtype=np.float64)


def _pattern_model(sequence: Sequence[str]) -> Dict[str, Any]:
    if len(sequence) < 10:
        return {
            "active": False,
            "support": 0,
            "banker_probability": 0.5,
            "direction": "B",
            "reliability": 0.0,
            "hard_pattern_rule": False,
        }
    target = _pattern_features(sequence)
    candidates: List[Tuple[float, bool]] = []
    for index in range(10, len(sequence)):
        prefix = sequence[:index]
        vector = _pattern_features(prefix)
        distance = float(np.sqrt(np.mean(np.square(target - vector))))
        similarity = math.exp(-4.4 * distance)
        candidates.append((similarity, sequence[index] == sequence[index - 1]))
    candidates.sort(key=lambda item: item[0], reverse=True)
    neighbors = candidates[:24]
    continued = 3.0
    reversed_mass = 3.0
    support = 0
    similarity_total = 0.0
    for similarity, did_continue in neighbors:
        similarity_total += similarity
        if similarity >= 0.42:
            support += 1
        if did_continue:
            continued += similarity
        else:
            reversed_mass += similarity
    continuation_probability = continued / (continued + reversed_mass)
    probability = (
        continuation_probability
        if sequence[-1] == "B"
        else 1.0 - continuation_probability
    )
    mean_similarity = similarity_total / max(1, len(neighbors))
    reliability = min(0.78, min(1.0, support / 12.0) * (0.45 + 0.55 * mean_similarity))
    return {
        "active": support >= 2,
        "support": support,
        "banker_probability": probability,
        "direction": "B" if probability >= 0.5 else "P",
        "edge": abs(probability - 0.5) * 2.0,
        "reliability": reliability if support >= 2 else 0.0,
        "mean_similarity": mean_similarity,
        "hard_pattern_rule": False,
        "continuation_probability": continuation_probability,
        "direction_semantics": "relative_pattern_walk_forward",
    }


def _detect_regime(sequence: Sequence[str]) -> Dict[str, Any]:
    recent = list(sequence[-18:])
    runs = _runs(recent)
    current_run = runs[-1][1] if runs else 0
    alternation = (
        sum(a != b for a, b in zip(recent, recent[1:])) / max(1, len(recent) - 1)
        if len(recent) >= 2 else 0.0
    )
    pair_rate = sum(length == 2 for _, length in runs[-8:]) / max(1, len(runs[-8:]))
    if current_run >= 4:
        name = "streak"
    elif alternation >= 0.78:
        name = "alternating"
    elif pair_rate >= 0.60:
        name = "double"
    elif 0.42 <= alternation <= 0.68:
        name = "chaotic"
    else:
        name = "transition"
    return {
        "name": name,
        "current_run": current_run,
        "alternation_rate": alternation,
        "pair_rate": pair_rate,
        "descriptor_only": True,
        "controls_weight": False,
    }


def _structural_regime_component(sequence: Sequence[str]) -> Dict[str, Any]:
    """將可重複的大路結構轉為 Adaptive 可用的正式成員。

    short/mid/long 是近期比例模型，最後一顆天然權重最高；它們不能單獨
    代表「單跳」或「雙跳」。此元件只在 run-length 結構已重複確認時啟用：

    - 長龍：末段至少三顆同向。
    - 單跳：最近四個 run 都是 1。
    - 雙跳：最近三個完整 run 都是 2；若目前只走到 pair 的第一顆，先補
      同邊，pair 完成才切到另一邊。
    - 跳跳龍：最近四個 run 的長度交替為 1/2 或 2/1。

    未確認時固定中性且 inactive，避免把單一最新結果偽裝成規律。
    """
    values = list(sequence[-36:])
    runs = _runs(values)
    if not runs:
        return {
            "active": False, "name": "mixed", "direction": "",
            "banker_probability": 0.5, "reliability": 0.0,
            "support": len(runs), "edge": 0.0, "reason": "run 不足",
        }

    last_side, last_length = runs[-1]
    opposite = "P" if last_side == "B" else "B"
    lengths = [length for _, length in runs]

    def _active(name: str, direction: str, reliability: float, reason: str) -> Dict[str, Any]:
        probability = 0.5 + (0.5 * reliability if direction == "B" else -0.5 * reliability)
        return {
            "active": True,
            "name": name,
            "direction": direction,
            "banker_probability": float(max(0.02, min(0.98, probability))),
            "reliability": float(reliability),
            "support": len(runs),
            "edge": float(reliability),
            "run_lengths": lengths[-6:],
            "reason": reason,
        }

    if last_length >= 3:
        return _active(
            "dragon",
            last_side,
            min(0.82, 0.64 + 0.045 * min(4, last_length - 3)),
            f"末段 {last_length} 顆同向長龍",
        )

    if len(runs) < 3:
        return {
            "active": False, "name": "mixed", "direction": "",
            "banker_probability": 0.5, "reliability": 0.0,
            "support": len(runs), "edge": 0.0,
            "run_lengths": lengths[-6:], "reason": "run 不足",
        }

    if len(lengths) >= 4 and all(length == 1 for length in lengths[-4:]):
        return _active("single_chop", opposite, 0.76, "最近四個 run 均為單跳")

    # 雙跳的最後一組若只出現一顆，代表該 pair 尚未完成，方向應補同邊；
    # 若 pair 已完成，才切換到另一邊。這可避免 BBPPBBP 被誤判為反打。
    if len(lengths) >= 3 and all(length == 2 for length in lengths[-3:]):
        return _active("double_chop", opposite, 0.74, "最近三個完整 run 均為雙跳")
    if (
        last_length == 1
        and len(lengths) >= 3
        and all(length == 2 for length in lengths[-3:-1])
    ):
        return _active("double_chop_building", last_side, 0.70, "雙跳結構中，正在補足 pair")

    if len(lengths) >= 4:
        tail = lengths[-4:]
        if tail in ([1, 2, 1, 2], [2, 1, 2, 1]):
            expected = tail[-1]
            direction = last_side if last_length < expected else opposite
            return _active(
                "alternating_run_pattern",
                direction,
                0.70,
                f"run 長度重複 {tail} 的跳跳龍節奏",
            )

    return {
        "active": False,
        "name": "mixed",
        "direction": "",
        "banker_probability": 0.5,
        "reliability": 0.0,
        "support": len(runs),
        "edge": 0.0,
        "run_lengths": lengths[-6:],
        "reason": "沒有通過長龍／單跳／雙跳／跳跳龍確認",
    }


def _bounded_recent_weights(models: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    priors = {
        "short": 0.19,
        "mid": 0.20,
        "long": 0.22,
        "pattern": 0.19,
        "analogue": 0.20,
    }
    raw: Dict[str, float] = {}
    for name, model in models.items():
        if not bool(model.get("active", True)):
            raw[name] = 0.0
            continue
        reliability = max(0.0, min(1.0, float(model.get("reliability", 0.0) or 0.0)))
        support = max(0, int(model.get("support", 0) or 0))
        support_score = min(1.0, support / 18.0)
        probability = _clip_probability(model.get("banker_probability", 0.5))
        edge_score = min(1.0, abs(probability - 0.5) / 0.12)
        raw[name] = priors[name] * (0.48 + 0.34 * reliability + 0.12 * support_score + 0.06 * edge_score)

    total = sum(raw.values()) or 1.0
    weights = {name: value / total for name, value in raw.items()}
    # 單一近期模型不得過度主導。
    for _ in range(8):
        excess = 0.0
        free: List[str] = []
        for name, value in weights.items():
            if value > ROAD_RECENT_MAX_MODEL_WEIGHT:
                excess += value - ROAD_RECENT_MAX_MODEL_WEIGHT
                weights[name] = ROAD_RECENT_MAX_MODEL_WEIGHT
            else:
                free.append(name)
        if excess <= 1e-12 or not free:
            break
        free_total = sum(max(1e-12, raw[name]) for name in free)
        for name in free:
            weights[name] += excess * max(1e-12, raw[name]) / free_total
    total = sum(weights.values()) or 1.0
    return {name: value / total for name, value in weights.items()}


def calculate_road_probabilities(
    values: Iterable[Any],
    seed: int | None = None,
    *,
    grid_cells: Sequence[Mapping[str, Any]] | None = None,
    initial_image_count: int = 0,
    manual_count: int = 0,
) -> Dict[str, Any]:
    raw_outcomes = normalize_raw_outcomes(values)
    sequence = [value for value in raw_outcomes if value in {"B", "P"}][-ROAD_HISTORY_LIMIT:]
    length = len(sequence)
    tie_count = sum(value == "T" for value in raw_outcomes)
    if seed is None:
        seed_payload = (
            "".join(raw_outcomes)
            + f"|{max(0, int(initial_image_count or 0))}"
            + f"|{max(0, int(manual_count or 0))}"
        )
        run_seed = int.from_bytes(
            sha256(seed_payload.encode("utf-8")).digest()[:4],
            byteorder="big",
            signed=False,
        )
    else:
        run_seed = int(seed) & 0xFFFFFFFF

    planning = analyze_full_road_pattern(
        sequence,
        grid_cells=grid_cells,
        initial_image_count=initial_image_count,
        manual_count=manual_count,
    )
    structural_regime = _structural_regime_component(sequence)
    recent_models: Dict[str, Dict[str, Any]] = {
        "short": _window_model(sequence, ROAD_SHORT_WINDOW, 0.84),
        "mid": _window_model(sequence, ROAD_MID_WINDOW, 0.91),
        "long": _window_model(sequence, ROAD_LONG_WINDOW, 0.965),
        "pattern": _pattern_model(sequence),
        "analogue": _analogue_model(sequence),
    }
    recent_weights = _bounded_recent_weights(recent_models)
    recent_components = {
        name: _clip_probability(model.get("banker_probability", 0.5))
        for name, model in recent_models.items()
    }
    recent_center = sum(recent_weights.get(name, 0.0) * probability for name, probability in recent_components.items())
    recent_disagreement = math.sqrt(sum(
        recent_weights.get(name, 0.0) * (probability - recent_center) ** 2
        for name, probability in recent_components.items()
    ))

    # 這裡只需要加權 Beta 分布的均值與標準差，不需要真的抽樣。
    # 舊版每次使用新的隨機 seed，會讓完全相同的牌路產生不同 context，
    # 進而污染 cMAB 的方差基準。解析式同時更快且完全可重現。
    recent_probability = 0.0
    recent_variance = 0.0
    for name, model in recent_models.items():
        weight = recent_weights.get(name, 0.0)
        if weight <= 0:
            continue
        probability = recent_components[name]
        reliability = max(0.05, float(model.get("reliability", 0.0) or 0.0))
        concentration = 12.0 + 30.0 * reliability
        alpha = max(0.5, probability * concentration)
        beta = max(0.5, (1.0 - probability) * concentration)
        total_concentration = alpha + beta
        beta_mean = alpha / total_concentration
        beta_variance = (
            alpha * beta
            / (
                total_concentration * total_concentration
                * (total_concentration + 1.0)
            )
        )
        recent_probability += weight * beta_mean
        recent_variance += weight * weight * beta_variance
    if length:
        recent_probability = float(recent_probability)
        recent_uncertainty = float(math.sqrt(max(0.0, recent_variance)))
    else:
        recent_probability = 0.5
        recent_uncertainty = 0.5
    recent_reliability = max(
        0.0,
        min(
            0.82,
            0.34 * min(1.0, length / 36.0)
            + 0.34 * (1.0 - min(1.0, recent_disagreement / ROAD_RECENT_MAX_DISAGREEMENT))
            + 0.32 * (1.0 - min(1.0, recent_uncertainty / 0.16)),
        ),
    )

    planning_probability = _clip_probability(planning.get("banker_probability", 0.5))
    planning_reliability = max(0.0, min(1.0, float(planning.get("reliability", 0.0) or 0.0)))

    # 相容舊欄位的 road aggregate。完整截圖歷史一旦達到可用長度，就以
    # Full Road 的全盤相位作為主幹；近期元件只負責確認目前尾段是否延續或
    # 轉折。這裡的比例不是機率勝率，而是「全盤與近期資訊」的資料涵蓋率。
    whole_shoe_evidence = max(
        0.0, min(1.0, float(planning.get("whole_shoe_evidence", 0.0) or 0.0))
    )
    if bool(planning.get("active")):
        planning_share = 0.70 + 0.15 * whole_shoe_evidence
    elif bool(planning.get("ok")):
        planning_share = 0.58
    else:
        planning_share = 0.0
    recent_share = 1.0 - planning_share
    banker_probability = planning_probability * planning_share + recent_probability * recent_share
    direction = "B" if banker_probability >= 0.5 else "P"
    combined_reliability = planning_reliability * planning_share + recent_reliability * recent_share

    model_outputs: Dict[str, Dict[str, Any]] = {
        **{
            name: {
                **model,
                "banker_probability": round(recent_components[name], 8),
                "effective_weight": round(recent_weights.get(name, 0.0), 8),
                "group": "recent_road",
            }
            for name, model in recent_models.items()
        },
        "full_road": {
            **planning,
            "banker_probability": round(planning_probability, 8),
            "effective_weight": 1.0,
            "group": "road_planning",
        },
        "structural_regime": {
            **structural_regime,
            "effective_weight": 1.0 if structural_regime.get("active") else 0.0,
            "group": "confirmed_run_length_structure",
        },
    }

    return {
        "ok": bool(sequence),
        "engine": "ROAD_FULL_SCREENSHOT_HISTORY_PRIMARY_V11",
        "pipeline_stage": "full_screenshot_history_planning_then_recent_confirmation",
        "run_seed": run_seed,
        "sequence": sequence,
        "raw_outcomes": raw_outcomes,
        "sample_count": length,
        "raw_sample_count": len(raw_outcomes),
        "tie_count": tie_count,
        "observed_tie_rate": tie_count / max(1, len(raw_outcomes)),
        "banker_probability": float(banker_probability),
        "player_probability": float(1.0 - banker_probability),
        "direction": direction,
        "direction_text": "莊" if direction == "B" else "閒",
        "action": direction,
        "action_text": "莊" if direction == "B" else "閒",
        "signal_allowed": bool(sequence),
        "signal_status_text": "完整截圖歷史已納入全盤規劃" if length >= 12 else "完整牌路樣本累積中",
        "signal_reason": "先以完整截圖 B/P/T 歷史建立全盤相位，再以近期專家確認下一局方向。",
        "confidence_score": float(combined_reliability),
        "confidence_label": "較高" if combined_reliability >= 0.72 else "中等" if combined_reliability >= 0.50 else "偏低",
        "uncertainty": float(recent_uncertainty),
        "model_disagreement": float(recent_disagreement),
        "planning_probability": float(planning_probability),
        "planning_player_probability": float(1.0 - planning_probability),
        "planning_reliability": float(planning_reliability),
        "planning_available": bool(planning.get("active")),
        "planning_share": float(planning_share),
        "whole_shoe_evidence": float(whole_shoe_evidence),
        "whole_shoe_regime": dict(planning.get("whole_shoe_regime") or {}),
        "recent_probability": float(recent_probability),
        "recent_player_probability": float(1.0 - recent_probability),
        "recent_reliability": float(recent_reliability),
        "recent_uncertainty": float(recent_uncertainty),
        "recent_uncertainty_method": "analytic_independent_beta_moments",
        "recent_model_disagreement": float(recent_disagreement),
        "full_road_analysis": planning,
        "structural_regime": dict(structural_regime),
        "models": model_outputs,
        "component_probabilities": {
            name: {
                "B": float(model_outputs[name].get("banker_probability", 0.5)),
                "P": 1.0 - float(model_outputs[name].get("banker_probability", 0.5)),
                "T": 0.0,
            }
            for name in model_outputs
        },
        "weights": {name: float(value) for name, value in recent_weights.items()},
        "regime": _detect_regime(sequence),
        "supports": {
            "short": int(recent_models["short"].get("support", 0)),
            "mid": int(recent_models["mid"].get("support", 0)),
            "long": int(recent_models["long"].get("support", 0)),
            "pattern": int(recent_models["pattern"].get("support", 0)),
            "analogue": int(recent_models["analogue"].get("support", 0)),
            "planning": int(planning.get("support", 0) or 0),
            "structural_regime": int(structural_regime.get("support", 0) or 0),
        },
        "removed_transition_chain_orders": [1, 2, 3],
        "eligible_for_core": length >= 10,
        "suggested_core_weight": 0.0,
        "max_core_weight": 0.0,
        "data_scope": "all_recognized_screenshot_history_primary_then_bounded_recent_confirmation",
        "full_history_used_count": length,
        "initial_image_count": max(0, int(initial_image_count or 0)),
        "manual_count": max(0, int(manual_count or 0)),
        "grid_cell_count": len(list(grid_cells or [])),
    }


def build_road_context(
    values: Iterable[Any],
    seed: int | None = None,
    *,
    grid_cells: Sequence[Mapping[str, Any]] | None = None,
    initial_image_count: int = 0,
    manual_count: int = 0,
) -> Dict[str, Any]:
    return calculate_road_probabilities(
        values,
        seed=seed,
        grid_cells=grid_cells,
        initial_image_count=initial_image_count,
        manual_count=manual_count,
    )


def fuse_road_with_main_prediction(
    main_prediction: Mapping[str, Any],
    road_analysis: Mapping[str, Any],
) -> Dict[str, Any]:
    """相容舊版呼叫；V10.8 主引擎內已完成 Stacking，不再做第二次融合。"""
    result = dict(main_prediction or {})
    result["road_support"] = dict(road_analysis or {})
    internal = dict(result.get("road_integration") or {})
    result["road_fusion"] = internal or {
        "processed_inside_core": False,
        "applied": False,
        "reason": "V10.8 建議由主引擎內部統整",
    }
    return result


__all__ = [
    "build_road_context",
    "calculate_road_probabilities",
    "fuse_road_with_main_prediction",
    "normalize_raw_outcomes",
    "normalize_road_sequence",
]
