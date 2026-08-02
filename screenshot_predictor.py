"""遊戲截圖的牌路先行預測轉接層。

流程固定為：
1. 清理截圖辨識出的 B/P 序列。
2. 先執行 road_model 建立牌路 context。
3. 估計截圖未提供的 0~9 點剩餘張數。
4. 將牌路 context 與估計牌組一起交給統一主引擎。

截圖沒有每個點值的真實剩餘張數，因此牌組組成仍屬估計資料。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import secrets

from particle_filter_points import fresh_counts
from predictor import predict
from road_model import build_road_context


def _clean_sequence(values: Iterable[Any]) -> List[str]:
    return [
        str(item).upper().strip()
        for item in values
        if str(item).upper().strip() in {"B", "P"}
    ][-500:]


def _largest_remainder_allocation(weights: Sequence[float], total: int) -> List[int]:
    if total <= 0:
        return [0] * len(weights)
    positive = [max(0.0, float(value)) for value in weights]
    weight_sum = sum(positive)
    if weight_sum <= 0:
        positive = [1.0] * len(weights)
        weight_sum = float(len(weights))
    exact = [value / weight_sum * total for value in positive]
    floors = [int(value) for value in exact]
    remainder = total - sum(floors)
    order = sorted(
        range(len(exact)),
        key=lambda index: (exact[index] - floors[index], positive[index]),
        reverse=True,
    )
    for index in order[:remainder]:
        floors[index] += 1
    return floors


def estimate_point_counts(
    remaining_cards: int,
    *,
    prior_counts: Optional[Sequence[int]] = None,
    decks: int = 8,
) -> Tuple[List[int], str]:
    """把剩餘總張數轉成 0~9 點值的估計張數。"""
    maximum = 52 * max(1, min(16, int(decks)))
    total = max(6, min(maximum, int(remaining_cards)))
    if (
        isinstance(prior_counts, Sequence)
        and len(prior_counts) == 10
        and sum(max(0, int(value)) for value in prior_counts) >= 6
    ):
        weights = [max(0, int(value)) for value in prior_counts]
        source = "session_scaled"
    else:
        weights = fresh_counts(decks)
        source = "fresh_shoe_scaled"
    return _largest_remainder_allocation(weights, total), source


def predict_from_screenshot(
    sequence: Iterable[Any],
    *,
    remaining_cards: Optional[int],
    prior_counts: Optional[Sequence[int]] = None,
    venue: str = "",
    room: str = "",
    user_id: str = "",
    run_seed: Optional[int] = None,
) -> Dict[str, Any]:
    cleaned = _clean_sequence(sequence)
    fallback_total = sum(int(value) for value in prior_counts or [] if int(value) >= 0)
    total = int(remaining_cards or fallback_total or 416)
    counts, source = estimate_point_counts(total, prior_counts=prior_counts, decks=8)

    seed = int(run_seed if run_seed is not None else secrets.randbits(32)) & 0xFFFFFFFF
    road_seed = (seed ^ 0x9E3779B9) & 0xFFFFFFFF

    # 關鍵順序：牌路先分析，再把 road context 交給主引擎。
    road_context = build_road_context(cleaned, seed=road_seed)
    result = predict(
        history=cleaned,
        venue=venue,
        room=room,
        user_id=user_id,
        run_seed=seed,
        shoe_context={"remaining_counts": counts},
        road_context=road_context,
    )
    result.update({
        "model_version": "V9.5-SCREEN-ROAD-FIRST-UNIFIED",
        "mode": "screen_estimated_composition_road_first",
        "model_core": "牌路先行＋有限牌組超幾何＋粒子／蒙地卡羅統一判斷",
        "screen_remaining_cards": total,
        "estimated_remaining_counts": counts,
        "composition_source": source,
        "composition_quality": "estimated",
        "road_sequence_length": len(cleaned),
        "road_support": road_context,
        "road_pipeline_completed": True,
        "virtual_only": False,
        "external_screen_input": True,
        "disclaimer": (
            "截圖只有剩餘總張數，未包含每個點值的真實剩餘張數；"
            "系統會先分析已辨識牌路，再以估計牌組執行機率模型。"
        ),
    })
    # 相容舊面板欄位；內容實際來自主引擎內部整合，不是 app 後置融合。
    result["road_fusion"] = dict(result.get("road_integration") or {})
    return result


__all__ = ["estimate_point_counts", "predict_from_screenshot"]
