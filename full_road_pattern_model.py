"""BGS V10.8 全盤牌路模型相容層。

保留舊版 ``analyze_full_road_pattern``／``build_big_road`` 介面，
內部改由 ``road_planning_model`` 的完整歷史 walk-forward 規劃模型執行。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence

from road_planning_model import analyze_road_planning, build_big_road


def analyze_full_road_pattern(
    values: Iterable[Any],
    *,
    grid_cells: Sequence[Mapping[str, Any]] | None = None,
    initial_image_count: int = 0,
    manual_count: int = 0,
) -> Dict[str, Any]:
    result = analyze_road_planning(
        values,
        grid_cells=grid_cells,
        initial_image_count=initial_image_count,
        manual_count=manual_count,
    )
    return {
        **result,
        "continuation_score": (
            float(result.get("continuation_probability", 0.5) or 0.5) - 0.5
        ) * 2.0,
        "lookback_mode": "entire_history_walk_forward_road_planning",
    }


__all__ = ["analyze_full_road_pattern", "build_big_road"]
