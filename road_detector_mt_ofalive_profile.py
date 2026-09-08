"""MT / ofalive99 手機直式畫面專用前置 Profile。

只處理已選 MT、完整直式畫面且符合既有 ofalive Android 白色路紙特徵的圖片。
若專用 ROI 未通過既有 ring-grid 品質閘門，會完整回退原本 road_detector，
因此不改動 DG / DB / SA / OB / T9 或既有 MT 格式的原先辨識流程。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import road_detector as _base


MT_OFALIVE99_MOBILE_ROIS = (
    (0.315, 0.715, 0.630, 0.080),
    (0.315, 0.724, 0.630, 0.080),
)


def _candidate_summary(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": item.get("region_name"),
        "recognized_count": int(item.get("recognized_count", 0) or 0),
        "unknown_candidates": int(
            item.get("unknown_candidates", item.get("uncertain_count", 0)) or 0
        ),
        "score": float(item.get("selection_score", -9999) or -9999),
        "elapsed_ms": float(item.get("elapsed_ms", 0) or 0),
        "method": str(item.get("method") or ""),
        "fixed_grid": bool(item.get("fixed_grid")),
        "ring_grid": bool(item.get("ring_grid")),
        "quality_ok": bool(item.get("quality_ok", True)),
        "reconstructed_all": bool(item.get("reconstructed_all", True)),
        "fallback_reason": str(item.get("fallback_reason") or ""),
        "alignment_score": float(
            dict(item.get("effective_grid") or {}).get("score", 0.0) or 0.0
        ),
        "grid_columns": int(item.get("grid_columns", 0) or 0),
        "layout_profile": str(item.get("layout_profile") or ""),
        "roi": dict(item.get("roi") or {}),
    }


def _try_mt_ofalive99(
    image_path: str | Path,
    *,
    venue: str = "",
    input_type: str = "auto",
) -> Dict[str, Any] | None:
    venue_code = str(venue or "").upper().strip()
    requested = str(input_type or "auto").lower().strip()
    if venue_code != "MT" or requested not in {"auto", "full_screen"}:
        return None

    image = _base._read_image(image_path)
    image_height, image_width = image.shape[:2]
    portrait = image_height > image_width * 1.15
    if not portrait or not _base._looks_like_ofalive_android_fullscreen(image):
        return None

    attempted: List[str] = []
    candidates: List[Dict[str, Any]] = []
    errors: List[str] = []

    for index, roi in enumerate(MT_OFALIVE99_MOBILE_ROIS):
        name = f"mt_ofalive99_mobile_big_road_{index}"
        attempted.append(name)
        try:
            current = _base._run_region(
                image,
                roi,
                name,
                72.0 - index * 0.4,
                fixed_grid=False,
                ring_grid=True,
                grid_columns=None,
                layout_profile="mt_ofalive99_mobile_full_screen",
            )
            current = dict(current)
            candidates.append(current)
            if not _base._acceptable(current):
                continue

            result = dict(current)
            result.update(
                {
                    "ok": bool(result.get("sequence"))
                    and bool(result.get("quality_ok", True)),
                    "input_type": "full_screen",
                    "selected_region": str(result.get("region_name") or name),
                    "venue_hint": "MT",
                    "errors": errors,
                    "attempted_regions": attempted,
                    "fast_early_exit": True,
                    "wide_layout_detected": False,
                    "road_crop_signature": False,
                    "dream_compact_mobile_profile_detected": False,
                    "ofalive_android_profile_detected": True,
                    "mt_ofalive99_mobile_profile_detected": True,
                    "image_size": {
                        "width": image_width,
                        "height": image_height,
                    },
                    "candidate_regions": [
                        _candidate_summary(item) for item in candidates
                    ],
                }
            )
            return result
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    return None


def detect_road_sequence_detailed(
    image_path: str | Path,
    *,
    venue: str = "",
    input_type: str = "auto",
) -> Dict[str, Any]:
    """先嘗試新增 MT/ofalive99 Profile；未命中就原封不動走既有 detector。"""
    try:
        preferred = _try_mt_ofalive99(
            image_path,
            venue=venue,
            input_type=input_type,
        )
        if preferred is not None:
            return preferred
    except Exception:
        pass

    return _base.detect_road_sequence_detailed(
        image_path,
        venue=venue,
        input_type=input_type,
    )


def detect_road_sequence(image_path: str | Path) -> List[str]:
    return list(detect_road_sequence_detailed(image_path).get("sequence") or [])


__all__ = ["detect_road_sequence", "detect_road_sequence_detailed"]
