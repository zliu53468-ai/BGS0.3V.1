"""遊戲畫面大路偵測模組 V10.6（MT 固定 6×15 大路掃描版）。

重點：
1. MT 完整畫面只裁切右下方的大路區塊，不再掃描珠盤路、下三路或整張圖片。
2. 固定把目標區塊切成 6 列 × 15 欄，共 90 格。
3. 每格分別統計紅、藍、綠 HSV 像素；紅=莊、藍=閒、綠=和局標記。
4. 依標準大路落點規則反推時間序列，避免單純 x/y 排序在長龍黏邊時錯序。
5. 其他館別仍保留原本 OpenCV/YOLO 區域辨識流程。
"""
from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import os
import time

import cv2
import numpy as np

from baccarat_vision import analyze_baccarat_array_detailed

_YOLO_MODEL: Any = None
_YOLO_LOCK = Lock()


def _parse_roi(
    raw: str,
    default: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    try:
        values = [float(part.strip()) for part in str(raw).split(",")]
        if len(values) != 4:
            raise ValueError
        x, y, width, height = values
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        width = max(0.01, min(1.0 - x, width))
        height = max(0.01, min(1.0 - y, height))
        return x, y, width, height
    except Exception:
        return default


def _env_roi(
    name: str,
    default: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    return _parse_roi(
        os.getenv(name, ",".join(str(value) for value in default)),
        default,
    )


# 一般館別備援區域。
ROAD_ROI = _env_roi("ROAD_ROI", (0.0, 0.58, 1.0, 0.42))

# MT 1728×903 範例中的實際大路區塊：x=1071, y=647, w=348, h=134。
# 使用比例座標後，畫面同比例縮放時仍可沿用。
MT_FIXED_ROAD_ROI = _env_roi(
    "MT_FIXED_ROAD_ROI",
    (0.619791667, 0.716500554, 0.201388889, 0.148394241),
)

# 橫向「珠盤路＋大路＋下三路」裁圖中的右上第一區塊。
WIDE_TOP_ROAD_ROI = _env_roi(
    "WIDE_TOP_ROAD_ROI",
    (0.265, 0.00, 0.735, 0.64),
)

VENUE_ROIS: Dict[str, Tuple[float, float, float, float]] = {
    "DG": _env_roi("DG_ROAD_ROI", (0.00, 0.80, 0.66, 0.20)),
    "MT": MT_FIXED_ROAD_ROI,
    "DB": _env_roi("DB_ROAD_ROI", (0.00, 0.58, 1.00, 0.42)),
    "SA": _env_roi("SA_ROAD_ROI", (0.00, 0.58, 1.00, 0.42)),
    "OB": _env_roi("OB_ROAD_ROI", (0.00, 0.58, 1.00, 0.42)),
    "T9": _env_roi("T9_ROAD_ROI", (0.00, 0.58, 1.00, 0.42)),
}

ROAD_GRID_ROWS = max(3, min(12, int(os.getenv("ROAD_GRID_ROWS", "6") or "6")))
ROAD_GRID_COLS = max(5, min(60, int(os.getenv("ROAD_GRID_COLS", "15") or "15")))
ROAD_GRID_INNER_MARGIN = max(
    0.0,
    min(0.25, float(os.getenv("ROAD_GRID_INNER_MARGIN", "0.08") or "0.08")),
)
ROAD_GRID_MIN_COLOR_PIXELS = max(
    5,
    int(os.getenv("ROAD_GRID_MIN_COLOR_PIXELS", "20") or "20"),
)
ROAD_GRID_COLOR_DOMINANCE = max(
    1.05,
    min(3.0, float(os.getenv("ROAD_GRID_COLOR_DOMINANCE", "1.25") or "1.25")),
)
ROAD_GRID_TIE_MIN_PIXELS = max(
    3,
    int(os.getenv("ROAD_GRID_TIE_MIN_PIXELS", "8") or "8"),
)
ROAD_GRID_MIN_RECOGNIZED = max(
    1,
    int(os.getenv("ROAD_GRID_MIN_RECOGNIZED", "4") or "4"),
)
ROAD_GRID_MAX_UNCERTAIN_RATIO = max(
    0.0,
    min(0.8, float(os.getenv("ROAD_GRID_MAX_UNCERTAIN_RATIO", "0.12") or "0.12")),
)

WIDE_LAYOUT_MIN_ASPECT = max(
    3.2,
    float(os.getenv("WIDE_LAYOUT_MIN_ASPECT", "4.0") or "4.0"),
)
ROAD_AUTO_FULL_FALLBACK = os.getenv("ROAD_AUTO_FULL_FALLBACK", "0").strip() == "1"
ROAD_USE_YOLO = os.getenv("ROAD_USE_YOLO", "0").strip() == "1"
ROAD_FAST_EARLY_EXIT = os.getenv("ROAD_FAST_EARLY_EXIT", "1").strip() == "1"
ROAD_FAST_MIN_RECOGNIZED = max(
    4,
    int(os.getenv("ROAD_FAST_MIN_RECOGNIZED", "8") or "8"),
)
ROAD_FAST_MAX_UNKNOWN_RATIO = max(
    0.0,
    min(0.8, float(os.getenv("ROAD_FAST_MAX_UNKNOWN_RATIO", "0.18") or "0.18")),
)
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "").strip()
YOLO_CONFIDENCE = max(
    0.05,
    min(0.95, float(os.getenv("YOLO_CONFIDENCE", "0.35") or "0.35")),
)
YOLO_IMAGE_SIZE = max(
    320,
    min(1536, int(os.getenv("YOLO_IMAGE_SIZE", "960") or "960")),
)


def _read_image(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(Path(path)), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("無法讀取遊戲畫面。")
    return image


def _crop(
    image: np.ndarray,
    roi: Sequence[float],
) -> Tuple[np.ndarray, Dict[str, int]]:
    height, width = image.shape[:2]
    x, y, roi_width, roi_height = [float(value) for value in roi]
    x1 = max(0, min(width - 1, int(round(x * width))))
    y1 = max(0, min(height - 1, int(round(y * height))))
    x2 = max(x1 + 1, min(width, int(round((x + roi_width) * width))))
    y2 = max(y1 + 1, min(height, int(round((y + roi_height) * height))))
    return (
        image[y1:y2, x1:x2].copy(),
        {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
    )


def _grid_bounds(length: int, index: int, count: int) -> Tuple[int, int]:
    start = int(round(index * length / count))
    end = int(round((index + 1) * length / count))
    return max(0, start), max(start + 1, min(length, end))


def _reconstruct_big_road_order(
    cells: Sequence[Mapping[str, Any]],
) -> List[Tuple[int, int]]:
    """依標準六列大路規則，從最終格位反推出非和局結果順序。"""
    grid: Dict[Tuple[int, int], str] = {
        (int(item.get("column", 0)), int(item.get("row", 0))): str(
            item.get("outcome") or ""
        ).upper()
        for item in cells
        if str(item.get("outcome") or "").upper() in {"B", "P"}
    }
    if not grid or (0, 0) not in grid:
        return sorted(grid, key=lambda position: (position[0], position[1]))

    target_count = len(grid)
    first_outcome = grid[(0, 0)]
    visited = {(0, 0)}
    ordered = [(0, 0)]

    def search(
        column: int,
        row: int,
        start_column: int,
        previous: str,
    ) -> bool:
        if len(visited) == target_count:
            return True

        options: List[Tuple[Tuple[int, int], int, str]] = []

        # 同色：能往下就往下；到底或下方已被占用才往右黏邊。
        if row < ROAD_GRID_ROWS - 1 and (column, row + 1) not in visited:
            same_position = (column, row + 1)
        else:
            next_column = column + 1
            while (next_column, row) in visited:
                next_column += 1
            same_position = (next_column, row)

        if (
            same_position in grid
            and same_position not in visited
            and grid[same_position] == previous
        ):
            options.append((same_position, start_column, previous))

        # 變色：從下一個主欄最上方開始。
        opposite = "P" if previous == "B" else "B"
        next_start_column = start_column + 1
        while (next_start_column, 0) in visited:
            next_start_column += 1
        opposite_position = (next_start_column, 0)
        if (
            opposite_position in grid
            and opposite_position not in visited
            and grid[opposite_position] == opposite
        ):
            options.append((opposite_position, next_start_column, opposite))

        for position, candidate_start, candidate_outcome in options:
            visited.add(position)
            ordered.append(position)
            if search(
                position[0],
                position[1],
                candidate_start,
                candidate_outcome,
            ):
                return True
            ordered.pop()
            visited.remove(position)
        return False

    if search(0, 0, 0, first_outcome):
        return list(ordered)

    # 無法完整反推時，仍回傳穩定的欄優先順序，並讓 quality_ok 反映異常。
    return sorted(grid, key=lambda position: (position[0], position[1]))


def _detect_fixed_grid(crop: np.ndarray) -> Dict[str, Any]:
    """固定掃描 6×15 格，只辨識每格的紅、藍與綠色和局記號。"""
    if crop is None or crop.size == 0:
        raise ValueError("固定大路裁圖為空。")

    image_height, image_width = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    red_mask = (
        (((hue <= 12) | (hue >= 168)) & (saturation >= 95) & (value >= 90))
    ).astype(np.uint8)
    blue_mask = (
        ((hue >= 90) & (hue <= 135) & (saturation >= 75) & (value >= 75))
    ).astype(np.uint8)
    green_mask = (
        ((hue >= 35) & (hue <= 90) & (saturation >= 75) & (value >= 75))
    ).astype(np.uint8)

    cells: List[Dict[str, Any]] = []
    uncertain_cells: List[Dict[str, Any]] = []

    for row in range(ROAD_GRID_ROWS):
        y1, y2 = _grid_bounds(image_height, row, ROAD_GRID_ROWS)
        for column in range(ROAD_GRID_COLS):
            x1, x2 = _grid_bounds(image_width, column, ROAD_GRID_COLS)

            margin_x = max(1, int(round((x2 - x1) * ROAD_GRID_INNER_MARGIN)))
            margin_y = max(1, int(round((y2 - y1) * ROAD_GRID_INNER_MARGIN)))
            inner_x1 = min(x2 - 1, x1 + margin_x)
            inner_x2 = max(inner_x1 + 1, x2 - margin_x)
            inner_y1 = min(y2 - 1, y1 + margin_y)
            inner_y2 = max(inner_y1 + 1, y2 - margin_y)

            red_pixels = int(
                red_mask[inner_y1:inner_y2, inner_x1:inner_x2].sum()
            )
            blue_pixels = int(
                blue_mask[inner_y1:inner_y2, inner_x1:inner_x2].sum()
            )
            green_pixels = int(
                green_mask[inner_y1:inner_y2, inner_x1:inner_x2].sum()
            )

            dominant_pixels = max(red_pixels, blue_pixels)
            outcome = ""
            if dominant_pixels >= ROAD_GRID_MIN_COLOR_PIXELS:
                if red_pixels >= blue_pixels * ROAD_GRID_COLOR_DOMINANCE:
                    outcome = "B"
                elif blue_pixels >= red_pixels * ROAD_GRID_COLOR_DOMINANCE:
                    outcome = "P"
                else:
                    uncertain_cells.append(
                        {
                            "row": row,
                            "column": column,
                            "red_pixels": red_pixels,
                            "blue_pixels": blue_pixels,
                        }
                    )

            if not outcome:
                continue

            inner_area = max(
                1,
                (inner_x2 - inner_x1) * (inner_y2 - inner_y1),
            )
            cells.append(
                {
                    "index": -1,
                    "outcome": outcome,
                    "column": column,
                    "row": row,
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "cx": round((x1 + x2) / 2.0, 2),
                    "cy": round((y1 + y2) / 2.0, 2),
                    "red_pixels": red_pixels,
                    "blue_pixels": blue_pixels,
                    "green_pixels": green_pixels,
                    "tie_count": 1
                    if green_pixels >= ROAD_GRID_TIE_MIN_PIXELS
                    else 0,
                    "confidence": round(dominant_pixels / inner_area, 6),
                }
            )

    cell_lookup = {
        (int(item["column"]), int(item["row"])): item for item in cells
    }
    ordered_positions = _reconstruct_big_road_order(cells)
    ordered_cells: List[Dict[str, Any]] = []
    sequence: List[str] = []
    raw_outcomes: List[str] = []
    tie_markers: Dict[str, int] = {}

    for index, position in enumerate(ordered_positions):
        source = dict(cell_lookup[position])
        source["index"] = index
        ordered_cells.append(source)
        outcome = str(source["outcome"])
        sequence.append(outcome)
        raw_outcomes.append(outcome)
        tie_count = int(source.get("tie_count", 0) or 0)
        if tie_count > 0:
            tie_markers[str(index)] = tie_count
            raw_outcomes.extend(["T"] * tie_count)

    uncertain_count = len(uncertain_cells)
    candidate_total = len(sequence) + uncertain_count
    uncertain_ratio = uncertain_count / max(1, candidate_total)
    reconstructed_all = len(ordered_positions) == len(cells)
    quality_ok = bool(
        len(sequence) >= ROAD_GRID_MIN_RECOGNIZED
        and uncertain_ratio <= ROAD_GRID_MAX_UNCERTAIN_RATIO
        and reconstructed_all
    )

    return {
        "ok": bool(sequence),
        "quality_ok": quality_ok,
        "sequence": sequence,
        "raw_outcomes": raw_outcomes,
        "tie_markers": tie_markers,
        "grid_cells": ordered_cells,
        "recognized_count": len(sequence),
        "confirmed_round_count": len(raw_outcomes),
        "uncertain_count": uncertain_count,
        "unknown_candidates": uncertain_count,
        "unknown_ratio": round(uncertain_ratio, 6),
        "raw_contours": 0,
        "candidates": ordered_cells,
        "method": "fixed_hsv_grid_6x15_v10_6",
        "grid_rows": ROAD_GRID_ROWS,
        "grid_columns": ROAD_GRID_COLS,
        "grid_size": {
            "width": image_width,
            "height": image_height,
        },
        "reconstructed_all": reconstructed_all,
        "uncertain_cells": uncertain_cells,
    }


def _sort_big_road(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: (
            float(item.get("cx", 0)),
            float(item.get("cy", 0)),
        ),
    )


def _get_yolo_model() -> Any:
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    if (
        not ROAD_USE_YOLO
        or not YOLO_MODEL_PATH
        or not Path(YOLO_MODEL_PATH).is_file()
    ):
        return None
    with _YOLO_LOCK:
        if _YOLO_MODEL is None:
            from ultralytics import YOLO

            _YOLO_MODEL = YOLO(YOLO_MODEL_PATH)
    return _YOLO_MODEL


def _normalize_yolo_label(label: str) -> str:
    value = str(label or "").strip().lower()
    if value in {"b", "banker", "red", "莊", "庄", "banker_circle"}:
        return "B"
    if value in {"p", "player", "blue", "閒", "闲", "player_circle"}:
        return "P"
    return ""


def _detect_yolo(crop: np.ndarray) -> Dict[str, Any]:
    model = _get_yolo_model()
    if model is None:
        return {"ok": False, "sequence": [], "method": "yolo_unavailable"}

    results = model.predict(
        source=crop,
        conf=YOLO_CONFIDENCE,
        imgsz=YOLO_IMAGE_SIZE,
        verbose=False,
    )
    detections: List[Dict[str, Any]] = []
    for result in results:
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for coordinates, class_id, confidence in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.cls.cpu().numpy(),
            boxes.conf.cpu().numpy(),
        ):
            label = (
                names.get(int(class_id), str(int(class_id)))
                if isinstance(names, Mapping)
                else str(int(class_id))
            )
            outcome = _normalize_yolo_label(label)
            if not outcome:
                continue
            x1, y1, x2, y2 = [float(value) for value in coordinates]
            detections.append(
                {
                    "outcome": outcome,
                    "label": str(label),
                    "confidence": round(float(confidence), 6),
                    "x": round(x1, 2),
                    "y": round(y1, 2),
                    "width": round(max(1.0, x2 - x1), 2),
                    "height": round(max(1.0, y2 - y1), 2),
                    "cx": round((x1 + x2) / 2.0, 2),
                    "cy": round((y1 + y2) / 2.0, 2),
                }
            )

    ordered = _sort_big_road(detections)
    return {
        "ok": bool(ordered),
        "sequence": [item["outcome"] for item in ordered],
        "raw_outcomes": [item["outcome"] for item in ordered],
        "recognized_count": len(ordered),
        "candidates": ordered,
        "method": "custom_yolo",
        "unknown_candidates": 0,
        "raw_contours": 0,
        "quality_ok": bool(ordered),
    }


def _score_result(result: Mapping[str, Any], preference: float = 0.0) -> float:
    recognized = int(result.get("recognized_count", 0) or 0)
    unknown = int(
        result.get("unknown_candidates", result.get("uncertain_count", 0)) or 0
    )
    raw = int(result.get("raw_contours", 0) or 0)
    if recognized <= 0:
        return -9999.0
    noise = max(0, raw - recognized * 8)
    fixed_bonus = 30.0 if str(result.get("method") or "").startswith("fixed_") else 0.0
    return recognized * 5.0 - unknown * 1.5 - noise * 0.04 + preference + fixed_bonus


def _run_region(
    image: np.ndarray,
    roi: Tuple[float, float, float, float],
    name: str,
    preference: float,
    *,
    fixed_grid: bool = False,
) -> Dict[str, Any]:
    crop, pixels = _crop(image, roi)
    started = time.perf_counter()

    if fixed_grid:
        result = _detect_fixed_grid(crop)
    elif ROAD_USE_YOLO and _get_yolo_model() is not None:
        result = _detect_yolo(crop)
    else:
        result = analyze_baccarat_array_detailed(crop)

    result = dict(result or {})
    result.update(
        {
            "region_name": name,
            "roi": pixels,
            "normalized_roi": list(roi),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "fixed_grid": fixed_grid,
        }
    )
    result["selection_score"] = round(_score_result(result, preference), 4)
    return result


def _acceptable(result: Mapping[str, Any]) -> bool:
    recognized = int(result.get("recognized_count", 0) or 0)
    unknown = int(
        result.get("unknown_candidates", result.get("uncertain_count", 0)) or 0
    )
    ratio = unknown / max(1, recognized + unknown)
    return bool(
        recognized >= ROAD_FAST_MIN_RECOGNIZED
        and ratio <= ROAD_FAST_MAX_UNKNOWN_RATIO
        and result.get("quality_ok", True)
    )


def detect_road_sequence_detailed(
    image_path: str | Path,
    *,
    venue: str = "",
    input_type: str = "auto",
) -> Dict[str, Any]:
    image = _read_image(image_path)
    venue_code = str(venue or "").upper().strip()
    requested = str(input_type or "auto").lower().strip()
    aspect = image.shape[1] / max(1.0, float(image.shape[0]))

    wide_multi_road = requested == "auto" and aspect >= WIDE_LAYOUT_MIN_ASPECT
    likely_crop = requested == "road_crop" or (
        requested == "auto" and 2.40 <= aspect < WIDE_LAYOUT_MIN_ASPECT
    )
    detected_type = (
        "wide_multi_road"
        if wide_multi_road
        else "road_crop"
        if likely_crop
        else "full_screen"
    )

    errors: List[str] = []
    candidates: List[Dict[str, Any]] = []
    attempted: List[str] = []
    plan: List[
        Tuple[
            str,
            Tuple[float, float, float, float],
            float,
            bool,
        ]
    ] = []

    if venue_code == "MT" and not likely_crop and not wide_multi_road:
        # MT 完整畫面：只掃右下方固定 6×15 大路，不做其他 ROI 備援。
        plan = [("mt_fixed_6x15", MT_FIXED_ROAD_ROI, 20.0, True)]
    elif likely_crop:
        # 使用者直接上傳牌路裁切圖時，整張圖視為固定 6×15 大路。
        plan = [("road_crop_fixed_6x15", (0.0, 0.0, 1.0, 1.0), 15.0, True)]
    elif wide_multi_road:
        plan = [("wide_top_fixed_6x15", WIDE_TOP_ROAD_ROI, 10.0, True)]
    else:
        if venue_code in VENUE_ROIS:
            plan.append(
                (
                    f"{venue_code.lower()}_venue_roi",
                    VENUE_ROIS[venue_code],
                    3.0,
                    False,
                )
            )
        plan.append(("generic_road_roi", ROAD_ROI, 1.5, False))
        if ROAD_AUTO_FULL_FALLBACK:
            plan.append(("full_image", (0.0, 0.0, 1.0, 1.0), 0.0, False))

    seen = set()
    best: Dict[str, Any] | None = None

    for name, roi, preference, fixed_grid in plan:
        key = (tuple(round(value, 6) for value in roi), fixed_grid)
        if key in seen:
            continue
        seen.add(key)
        attempted.append(name)
        try:
            current = _run_region(
                image,
                roi,
                name,
                preference,
                fixed_grid=fixed_grid,
            )
            candidates.append(current)
            if best is None or float(current.get("selection_score", -9999)) > float(
                best.get("selection_score", -9999)
            ):
                best = current
            if ROAD_FAST_EARLY_EXIT and _acceptable(current):
                best = current
                break
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    if best is None:
        return {
            "ok": False,
            "sequence": [],
            "recognized_count": 0,
            "method": "failed",
            "input_type": "unknown",
            "errors": errors,
            "attempted_regions": attempted,
        }

    result = dict(best)
    result.update(
        {
            "ok": bool(result.get("sequence")),
            "input_type": detected_type,
            "selected_region": str(best.get("region_name") or ""),
            "venue_hint": venue_code,
            "errors": errors,
            "attempted_regions": attempted,
            "fast_early_exit": len(candidates) < len(plan),
            "wide_layout_detected": bool(wide_multi_road),
            "candidate_regions": [
                {
                    "name": item.get("region_name"),
                    "recognized_count": int(item.get("recognized_count", 0) or 0),
                    "unknown_candidates": int(
                        item.get(
                            "unknown_candidates",
                            item.get("uncertain_count", 0),
                        )
                        or 0
                    ),
                    "score": float(item.get("selection_score", -9999)),
                    "elapsed_ms": float(item.get("elapsed_ms", 0) or 0),
                    "method": str(item.get("method") or ""),
                    "fixed_grid": bool(item.get("fixed_grid")),
                }
                for item in candidates
            ],
        }
    )
    return result


def detect_road_sequence(image_path: str | Path) -> List[str]:
    return list(detect_road_sequence_detailed(image_path).get("sequence") or [])


__all__ = ["detect_road_sequence", "detect_road_sequence_detailed"]
