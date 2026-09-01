"""Road detector V11.8 adaptive-grid auto-locate wrapper.

Keeps the proven V11.6 detector intact in road_detector_base_v11_6.py.
Only when the primary detector fails quality gates, this wrapper:
1. locates bright road-paper regions,
2. estimates the real horizontal grid pitch from neutral gray grid-line periodicity,
3. retries the unchanged 6xN fixed-grid recognizer with explicit column counts.

It does not loosen chronology / uncertainty / confidence gates and never promotes
partial reconstruction into a valid road.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple
import os

import cv2
import numpy as np

import road_detector_base_v11_6 as _base

AUTOLOCATE_ENABLED = os.getenv("ROAD_AUTOLOCATE_ENABLED", "1").strip() == "1"
AUTOLOCATE_SEARCH_TOP = max(0.45, min(0.85, float(os.getenv("ROAD_AUTOLOCATE_SEARCH_TOP", "0.62") or "0.62")))
AUTOLOCATE_SEARCH_BOTTOM = max(AUTOLOCATE_SEARCH_TOP + 0.05, min(1.0, float(os.getenv("ROAD_AUTOLOCATE_SEARCH_BOTTOM", "0.96") or "0.96")))
AUTOLOCATE_MIN_BOARD_WIDTH = max(0.12, min(0.80, float(os.getenv("ROAD_AUTOLOCATE_MIN_BOARD_WIDTH", "0.24") or "0.24")))
AUTOLOCATE_MIN_BOARD_HEIGHT = max(0.02, min(0.30, float(os.getenv("ROAD_AUTOLOCATE_MIN_BOARD_HEIGHT", "0.035") or "0.035")))
AUTOLOCATE_MIN_BRIGHT_FRACTION = max(0.15, min(0.90, float(os.getenv("ROAD_AUTOLOCATE_MIN_BRIGHT_FRACTION", "0.30") or "0.30")))
AUTOLOCATE_MAX_CANDIDATES = max(4, min(24, int(os.getenv("ROAD_AUTOLOCATE_MAX_CANDIDATES", "14") or "14")))
AUTOLOCATE_WHOLE_CROP_MIN_ASPECT = max(1.4, min(6.0, float(os.getenv("ROAD_AUTOLOCATE_WHOLE_CROP_MIN_ASPECT", "1.75") or "1.75")))
AUTOLOCATE_GRID_PERIOD_MIN_CORR = max(0.05, min(0.80, float(os.getenv("ROAD_AUTOLOCATE_GRID_PERIOD_MIN_CORR", "0.12") or "0.12")))
AUTOLOCATE_GRID_TRIALS = max(2, min(6, int(os.getenv("ROAD_AUTOLOCATE_GRID_TRIALS", "4") or "4")))


def __getattr__(name: str) -> Any:
    return getattr(_base, name)


def _clip_roi(roi: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, width, height = [float(v) for v in roi]
    x = max(0.0, min(0.99, x))
    y = max(0.0, min(0.99, y))
    width = max(0.01, min(1.0 - x, width))
    height = max(0.01, min(1.0 - y, height))
    return x, y, width, height


def _roi_key(roi: Sequence[float]) -> Tuple[int, int, int, int]:
    return tuple(int(round(float(v) * 10000.0)) for v in roi)  # type: ignore[return-value]


def _bright_neutral_mask(image: np.ndarray) -> np.ndarray:
    pixels = image.astype(np.int16, copy=False)
    channel_min = np.min(pixels, axis=2)
    channel_span = np.max(pixels, axis=2) - channel_min
    return ((channel_min >= 165) & (channel_span <= 90)).astype(np.uint8)


def _candidate_board_components(image: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
    height, width = image.shape[:2]
    aspect = width / max(1.0, float(height))
    y0 = 0 if aspect >= AUTOLOCATE_WHOLE_CROP_MIN_ASPECT else int(round(height * AUTOLOCATE_SEARCH_TOP))
    y1 = height if aspect >= AUTOLOCATE_WHOLE_CROP_MIN_ASPECT else int(round(height * AUTOLOCATE_SEARCH_BOTTOM))
    y1 = max(y0 + 1, min(height, y1))
    search = image[y0:y1]
    mask = _bright_neutral_mask(search) * 255

    close_w = max(5, width // 70)
    close_h = max(3, height // 300)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h)), iterations=2)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (max(7, width // 40), max(1, height // 640))), iterations=1)

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boards: List[Tuple[float, float, float, float, float]] = []
    for index in range(1, count):
        x, local_y, box_w, box_h, _area = [int(v) for v in stats[index]]
        y = local_y + y0
        if box_w / max(1.0, float(width)) < AUTOLOCATE_MIN_BOARD_WIDTH:
            continue
        if box_h / max(1.0, float(height)) < AUTOLOCATE_MIN_BOARD_HEIGHT:
            continue
        if aspect < AUTOLOCATE_WHOLE_CROP_MIN_ASPECT and y / max(1.0, float(height)) < AUTOLOCATE_SEARCH_TOP - 0.03:
            continue
        crop = image[y:min(height, y + box_h), x:min(width, x + box_w)]
        if crop.size == 0:
            continue
        bright_fraction = float(np.mean(_bright_neutral_mask(crop)))
        if bright_fraction < AUTOLOCATE_MIN_BRIGHT_FRACTION:
            continue
        boards.append((x / width, y / height, box_w / width, box_h / height, bright_fraction))

    boards.sort(key=lambda item: (item[1], item[2] * item[3], item[4]), reverse=True)
    return boards[:4]


def _autolocate_candidate_rois(image: np.ndarray) -> List[Dict[str, Any]]:
    height, width = image.shape[:2]
    aspect = width / max(1.0, float(height))
    output: List[Dict[str, Any]] = []
    seen: set[Tuple[int, int, int, int]] = set()

    def add(name: str, roi: Sequence[float], preference: float, source: str) -> None:
        normalized = _clip_roi(roi)
        key = _roi_key(normalized)
        if key in seen:
            return
        seen.add(key)
        output.append({"name": name, "roi": normalized, "preference": float(preference), "source": source})

    if aspect >= AUTOLOCATE_WHOLE_CROP_MIN_ASPECT:
        add("autolocate_whole_image_6xN", (0.0, 0.0, 1.0, 1.0), 78.0, "whole_road_crop")

    for board_index, (bx, by, bw, bh, bright_fraction) in enumerate(_candidate_board_components(image)):
        presets = (
            (0.26, 0.06, 0.58, 0.42),
            (0.29, 0.08, 0.54, 0.38),
            (0.31, 0.10, 0.50, 0.34),
            (0.24, 0.10, 0.64, 0.36),
            (0.30, 0.02, 0.50, 0.48),
        )
        for preset_index, (px, py, pw, ph) in enumerate(presets):
            add(
                f"autolocate_board_{board_index}_bigroad_{preset_index}",
                (bx + bw * px, by + bh * py, bw * pw, bh * ph),
                76.0 - preset_index * 0.4 + bright_fraction * 2.0,
                "bright_board_upper_middle",
            )

        board_aspect = bw / max(1e-6, bh)
        if board_aspect >= 7.0 or bh <= 0.10:
            add(f"autolocate_board_{board_index}_full", (bx, by, bw, bh), 70.0 + bright_fraction * 2.0, "bright_board_full")

    if aspect < AUTOLOCATE_WHOLE_CROP_MIN_ASPECT:
        if height > width * 1.15:
            generic = (
                (0.29, 0.715, 0.54, 0.095),
                (0.27, 0.700, 0.60, 0.120),
                (0.31, 0.725, 0.50, 0.090),
                (0.24, 0.690, 0.66, 0.135),
            )
        else:
            generic = (
                (0.20, 0.700, 0.64, 0.120),
                (0.24, 0.750, 0.58, 0.120),
                (0.18, 0.790, 0.68, 0.150),
                (0.28, 0.650, 0.55, 0.130),
            )
        for index, roi in enumerate(generic):
            add(f"autolocate_geometry_{index}", roi, 60.0 - index * 0.4, "generic_bottom_geometry")

    return output[:AUTOLOCATE_MAX_CANDIDATES]


def _grid_column_trials(crop: np.ndarray) -> List[Dict[str, Any]]:
    """Estimate real column count from neutral gray vertical-grid periodicity."""
    height, width = crop.shape[:2]
    if height <= 0 or width <= 0:
        return [{"grid_columns": max(5, int(getattr(_base, "ROAD_GRID_COLS", 15))), "source": "fallback", "period": 0.0, "correlation": 0.0}]

    trials: List[Dict[str, Any]] = []
    seen: set[int] = set()

    def add(columns: int, source: str, period: float, correlation: float) -> None:
        columns = max(5, min(60, int(columns)))
        if columns in seen:
            return
        seen.add(columns)
        trials.append({
            "grid_columns": columns,
            "source": source,
            "period": round(float(period), 6),
            "correlation": round(float(correlation), 6),
        })

    rows = max(3, int(getattr(_base, "ROAD_GRID_ROWS", 6)))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    neutral = ((saturation < 80) & (value > 80)).astype(np.float32)
    darkness = (255.0 - gray.astype(np.float32)) * neutral
    projection = np.mean(darkness, axis=0)
    centered = projection - float(np.mean(projection))

    row_pitch = height / max(1.0, float(rows))
    min_lag = max(5, int(round(row_pitch * 0.50)))
    max_lag = min(80, max(min_lag + 2, int(round(row_pitch * 1.90))), max(min_lag + 2, width // 3))

    scored: List[Tuple[float, int]] = []
    if width >= 40 and max_lag > min_lag:
        for lag in range(min_lag, max_lag + 1):
            left = centered[:-lag]
            right = centered[lag:]
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denominator <= 1e-6:
                continue
            correlation = float(np.dot(left, right) / denominator)
            scored.append((correlation, lag))
        scored.sort(reverse=True)

    used_periods: List[int] = []
    for correlation, lag in scored:
        if correlation < AUTOLOCATE_GRID_PERIOD_MIN_CORR:
            break
        if any(abs(lag - prior) <= 1 for prior in used_periods):
            continue
        if any(abs(lag - 2 * prior) <= 2 for prior in used_periods):
            continue
        used_periods.append(lag)
        estimated_columns = int(round(width / max(1.0, float(lag))))
        add(estimated_columns, "grid_period", float(lag), correlation)
        add(estimated_columns - 1, "grid_period_neighbor", float(lag), correlation - 0.01)
        add(estimated_columns + 1, "grid_period_neighbor", float(lag), correlation - 0.01)
        break

    square_estimate = int(round((width / max(1.0, float(height))) * rows))
    add(square_estimate, "aspect_fallback", row_pitch, 0.0)
    return trials[:AUTOLOCATE_GRID_TRIALS]


def _candidate_summary(result: Mapping[str, Any], source: str) -> Dict[str, Any]:
    return {
        "name": str(result.get("region_name") or ""),
        "source": source,
        "recognized_count": int(result.get("recognized_count", 0) or 0),
        "uncertain_count": int(result.get("uncertain_count", result.get("unknown_candidates", 0)) or 0),
        "quality_ok": bool(result.get("quality_ok", False)),
        "reconstructed_all": bool(result.get("reconstructed_all", False)),
        "fallback_reason": str(result.get("fallback_reason") or ""),
        "selection_score": float(result.get("selection_score", -9999.0) or -9999.0),
        "grid_columns_requested": int(result.get("autolocate_grid_columns_requested", 0) or 0),
        "grid_period_x": float(result.get("autolocate_grid_period_x", 0.0) or 0.0),
        "grid_period_correlation": float(result.get("autolocate_grid_period_correlation", 0.0) or 0.0),
        "grid_trial_source": str(result.get("autolocate_grid_trial_source") or ""),
        "roi": list(result.get("normalized_roi") or []),
    }


def _run_autolocate(image: np.ndarray) -> Tuple[Dict[str, Any] | None, List[Dict[str, Any]], List[str]]:
    best: Dict[str, Any] | None = None
    diagnostics: List[Dict[str, Any]] = []
    errors: List[str] = []

    for item in _autolocate_candidate_rois(image):
        try:
            candidate_crop, _pixels = _base._crop(image, item["roi"])
            column_trials = _grid_column_trials(candidate_crop)
        except Exception as exc:
            errors.append(f"{item['name']}: grid-period-estimation: {exc}")
            continue

        for trial in column_trials:
            columns = int(trial["grid_columns"])
            trial_name = f"{item['name']}_c{columns}"
            try:
                current = _base._run_region(
                    image,
                    item["roi"],
                    trial_name,
                    float(item["preference"]),
                    fixed_grid=True,
                    ring_grid=False,
                    grid_columns=columns,
                    layout_profile="auto_white_board_adaptive_grid_v11_8",
                )
                current = dict(current)
                current.update({
                    "autolocate_grid_columns_requested": columns,
                    "autolocate_grid_period_x": float(trial.get("period", 0.0) or 0.0),
                    "autolocate_grid_period_correlation": float(trial.get("correlation", 0.0) or 0.0),
                    "autolocate_grid_trial_source": str(trial.get("source") or ""),
                })
                diagnostics.append(_candidate_summary(current, str(item.get("source") or "")))
                valid = bool(current.get("quality_ok")) and bool(current.get("reconstructed_all")) and bool(current.get("sequence"))
                if not valid:
                    continue
                if best is None or float(current.get("selection_score", -9999.0)) > float(best.get("selection_score", -9999.0)):
                    best = dict(current)
                if str(trial.get("source") or "").startswith("grid_period"):
                    diagnostics.sort(key=lambda d: float(d.get("selection_score", -9999.0)), reverse=True)
                    return best, diagnostics, errors
            except Exception as exc:
                errors.append(f"{trial_name}: {exc}")

    diagnostics.sort(key=lambda item: float(item.get("selection_score", -9999.0)), reverse=True)
    return best, diagnostics, errors


def detect_road_sequence_detailed(image_path: str, *, venue: str = "", input_type: str = "auto") -> Dict[str, Any]:
    primary = dict(_base.detect_road_sequence_detailed(image_path, venue=venue, input_type=input_type) or {})
    primary_ok = bool(primary.get("ok")) and bool(primary.get("quality_ok", True)) and bool(primary.get("sequence"))
    if primary_ok or not AUTOLOCATE_ENABLED:
        primary.setdefault("autolocate_fallback_used", False)
        primary.setdefault("autolocate_candidates", [])
        return primary

    try:
        image = _base._read_image(image_path)
        best, diagnostics, auto_errors = _run_autolocate(image)
    except Exception as exc:
        primary["autolocate_fallback_used"] = False
        primary["autolocate_error"] = str(exc)
        primary["autolocate_candidates"] = []
        return primary

    if best is None:
        primary["autolocate_fallback_used"] = False
        primary["autolocate_attempted"] = True
        primary["autolocate_candidates"] = diagnostics
        primary["autolocate_errors"] = auto_errors
        primary["detector_version"] = "ROAD-DETECTOR-V11.8-ADAPTIVE-GRID"
        return primary

    original_reason = str(primary.get("fallback_reason") or "")
    original_region = str(primary.get("selected_region") or "")
    result = dict(best)
    result.update({
        "ok": True,
        "quality_ok": True,
        "input_type": str(primary.get("input_type") or input_type or "auto"),
        "venue_hint": str(venue or "").upper().strip(),
        "selected_region": str(best.get("region_name") or "autolocate"),
        "layout_profile": "auto_white_board_adaptive_grid_v11_8",
        "autolocate_fallback_used": True,
        "autolocate_attempted": True,
        "autolocate_primary_region": original_region,
        "autolocate_primary_fallback_reason": original_reason,
        "autolocate_candidates": diagnostics,
        "autolocate_errors": auto_errors,
        "detector_version": "ROAD-DETECTOR-V11.8-ADAPTIVE-GRID",
    })
    result["candidate_regions"] = list(primary.get("candidate_regions") or []) + diagnostics
    return result


def detect_road_sequence(image_path: str) -> List[str]:
    return list(detect_road_sequence_detailed(image_path).get("sequence") or [])


__all__ = ["detect_road_sequence", "detect_road_sequence_detailed", "_autolocate_candidate_rois", "_grid_column_trials"]
