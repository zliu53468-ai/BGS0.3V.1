"""遊戲截圖用的機率估計轉接層。

重要限制：截圖只提供「剩餘總張數」與莊閒路紙，沒有每個點值實際剩餘
張數，因此無法重建真人桌的精確牌組。本模組會依既有 Session 牌值比例，
或全新八副牌的理論比例，建立一組可重現的估計 counts，再交給既有
超幾何＋蒙地卡羅引擎。輸出會明確標示 estimated composition。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import secrets

from particle_filter_points import fresh_counts
from predictor import predict


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
    counts, source = estimate_point_counts(
        total,
        prior_counts=prior_counts,
        decks=8,
    )
    seed = int(run_seed if run_seed is not None else secrets.randbits(32)) & 0xFFFFFFFF
    result = predict(
        history=cleaned,
        venue=venue,
        room=room,
        user_id=user_id,
        run_seed=seed,
        shoe_context={"remaining_counts": counts},
    )
    result.update(
        {
            "model_version": "V9.0-SCREEN-OCR-HYPERGEOMETRIC-MC",
            "mode": "screen_estimated_composition",
            "model_core": "超幾何分布＋粒子/蒙地卡羅驗證",
            "screen_remaining_cards": total,
            "estimated_remaining_counts": counts,
            "composition_source": source,
            "road_sequence_length": len(cleaned),
            "virtual_only": False,
            "external_screen_input": True,
            "disclaimer": (
                "截圖只有剩餘總張數，未包含每個點值的真實剩餘張數；"
                "本結果使用估計牌值組成，不等同真人桌精確牌靴。"
            ),
        }
    )
    return result

