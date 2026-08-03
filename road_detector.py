"""遊戲畫面大路偵測模組 V10.9.1（自適應固定 6×15；非 MT 完整畫面也優先固定格）。

重點：
1. MT 完整畫面只裁切右下方的大路區塊。
2. DG / DB / SA / OB / T9 完整畫面同樣優先固定 6×15（venue ROI → generic ROI），
   找圓 / YOLO 僅作 quality 失敗後的備援，避免再出現「未偵測到大路圓圈」誤導。
3. 固定維持 6 列 × 15 欄，但先在 ROI 內微調有效格線範圍與內縮距離。
4. 每格分別統計紅、藍、綠 HSV 像素；雙色接近標記 uncertain，不硬選。
5. 依標準大路落點規則（含長龍右黏狀態）反推時間序列；失敗不再把欄排序當正確答案。
6. ROAD_GRID_DEBUG=1 可輸出疊格線除錯圖。
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


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(str(os.getenv(name, default)).strip())
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(str(os.getenv(name, default)).strip())
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


# 一般館別備援區域（下半大路常見區，較寬以利手機截圖）。
ROAD_ROI = _env_roi("ROAD_ROI", (0.0, 0.55, 1.0, 0.45))

# MT 1728×903 範例中的實際大路區塊。
MT_FIXED_ROAD_ROI = _env_roi(
    "MT_FIXED_ROAD_ROI",
    (0.619791667, 0.716500554, 0.201388889, 0.148394241),
)

# 橫向「珠盤路＋大路＋下三路」裁圖中的右上第一區塊。
WIDE_TOP_ROAD_ROI = _env_roi(
    "WIDE_TOP_ROAD_ROI",
    (0.265, 0.00, 0.735, 0.64),
)

# DG 等館：預設改為較寬的下半／中下大路區，避免舊版 y=0.80 窄帶裁空。
# 格式：x, y, w, h（正規化 0~1）。可用環境變數覆寫。
VENUE_ROIS: Dict[str, Tuple[float, float, float, float]] = {
    "DG": _env_roi("DG_ROAD_ROI", (0.00, 0.58, 1.00, 0.42)),
    "MT": MT_FIXED_ROAD_ROI,
    "DB": _env_roi("DB_ROAD_ROI", (0.00, 0.55, 1.00, 0.45)),
    "SA": _env_roi("SA_ROAD_ROI", (0.00, 0.55, 1.00, 0.45)),
    "OB": _env_roi("OB_ROAD_ROI", (0.00, 0.55, 1.00, 0.45)),
    "T9": _env_roi("T9_ROAD_ROI", (0.00, 0.55, 1.00, 0.45)),
}

# 同一館完整畫面可再試的備援 ROI（仍走固定格）。
VENUE_FALLBACK_ROIS: Dict[str, List[Tuple[str, Tuple[float, float, float, float]]]] = {
    "DG": [
        ("dg_mid_road", _env_roi("DG_ROAD_ROI_ALT1", (0.00, 0.48, 1.00, 0.40))),
        ("dg_lower_road", _env_roi("DG_ROAD_ROI_ALT2", (0.00, 0.62, 0.85, 0.35))),
    ],
    "DB": [
        ("db_mid_road", _env_roi("DB_ROAD_ROI_ALT1", (0.00, 0.50, 1.00, 0.42))),
    ],
    "SA": [
        ("sa_mid_road", _env_roi("SA_ROAD_ROI_ALT1", (0.00, 0.50, 1.00, 0.42))),
    ],
    "OB": [
        ("ob_mid_road", _env_roi("OB_ROAD_ROI_ALT1", (0.00, 0.50, 1.00, 0.42))),
    ],
    "T9": [
        ("t9_mid_road", _env_roi("T9_ROAD_ROI_ALT1", (0.00, 0.50, 1.00, 0.42))),
    ],
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

ROAD_GRID_ALIGN_MAX_TRIM = _env_float("ROAD_GRID_ALIGN_MAX_TRIM", 0.12, 0.0, 0.25)
ROAD_GRID_ALIGN_SEARCH_STEPS = _env_int("ROAD_GRID_ALIGN_SEARCH_STEPS", 17, 5, 41)
ROAD_GRID_MIN_ALIGNMENT_SCORE = _env_float("ROAD_GRID_MIN_ALIGNMENT_SCORE", 0.46, 0.0, 1.0)
ROAD_GRID_MIN_COLOR_RATIO = _env_float("ROAD_GRID_MIN_COLOR_RATIO", 0.018, 0.001, 0.20)
ROAD_GRID_INNER_MARGIN_MAX = _env_float("ROAD_GRID_INNER_MARGIN_MAX", 0.18, 0.02, 0.35)
ROAD_GRID_BOUNDARY_GUARD_PX = _env_float("ROAD_GRID_BOUNDARY_GUARD_PX", 1.4, 0.0, 6.0)
ROAD_GRID_MIN_MEDIAN_CONFIDENCE = _env_float(
    "ROAD_GRID_MIN_MEDIAN_CONFIDENCE", 0.42, 0.0, 1.0
)
ROAD_GRID_RED_MIN_S = _env_int("ROAD_GRID_RED_MIN_S", 58, 0, 255)
ROAD_GRID_RED_MIN_V = _env_int("ROAD_GRID_RED_MIN_V", 48, 0, 255)
ROAD_GRID_BLUE_MIN_S = _env_int("ROAD_GRID_BLUE_MIN_S", 52, 0, 255)
ROAD_GRID_BLUE_MIN_V = _env_int("ROAD_GRID_BLUE_MIN_V", 45, 0, 255)
ROAD_GRID_GREEN_MIN_S = _env_int("ROAD_GRID_GREEN_MIN_S", 62, 0, 255)
ROAD_GRID_GREEN_MIN_V = _env_int("ROAD_GRID_GREEN_MIN_V", 48, 0, 255)
ROAD_GRID_TIE_MIN_AREA_RATIO = _env_float(
    "ROAD_GRID_TIE_MIN_AREA_RATIO", 0.006, 0.001, 0.15
)
ROAD_GRID_TIE_MIN_COMPONENT_RATIO = _env_float(
    "ROAD_GRID_TIE_MIN_COMPONENT_RATIO", 0.45, 0.10, 1.0
)
ROAD_GRID_TIE_MAX_SPAN_RATIO = _env_float(
    "ROAD_GRID_TIE_MAX_SPAN_RATIO", 0.78, 0.20, 1.0
)
ROAD_GRID_DEBUG = os.getenv("ROAD_GRID_DEBUG", "0").strip() == "1"
ROAD_GRID_DEBUG_DIR = os.getenv("ROAD_GRID_DEBUG_DIR", "/tmp/bgs_road_debug").strip()
ROAD_CROP_MIN_ASPECT = _env_float("ROAD_CROP_MIN_ASPECT", 2.05, 1.2, 4.0)

WIDE_LAYOUT_MIN_ASPECT = max(
    3.2,
    float(os.getenv("WIDE_LAYOUT_MIN_ASPECT", "4.0") or "4.0"),
)
ROAD_AUTO_FULL_FALLBACK = os.getenv("ROAD_AUTO_FULL_FALLBACK", "0").strip() == "1"
ROAD_USE_YOLO = os.getenv("ROAD_USE_YOLO", "0").strip() == "1"
# 固定格失敗後是否嘗試找圓備援（預設開；僅 quality 全失敗時才會用到）。
ROAD_CONTOUR_FALLBACK = os.getenv("ROAD_CONTOUR_FALLBACK", "1").strip() == "1"
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


def _color_masks(crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    red = (
        ((hue <= 15) | (hue >= 165))
        & (saturation >= ROAD_GRID_RED_MIN_S)
        & (value >= ROAD_GRID_RED_MIN_V)
    ).astype(np.uint8)
    blue = (
        (hue >= 88)
        & (hue <= 142)
        & (saturation >= ROAD_GRID_BLUE_MIN_S)
        & (value >= ROAD_GRID_BLUE_MIN_V)
    ).astype(np.uint8)
    green = (
        (hue >= 34)
        & (hue <= 87)
        & (saturation >= ROAD_GRID_GREEN_MIN_S)
        & (value >= ROAD_GRID_GREEN_MIN_V)
    ).astype(np.uint8)
    union = ((red | blue | green) > 0).astype(np.uint8)
    return red, blue, green, union


def _axis_alignment(
    color_coordinates: np.ndarray,
    edge_projection: np.ndarray,
    length: int,
    divisions: int,
) -> Dict[str, float]:
    """在不改變列欄數的前提下，搜尋 ROI 內最合理的格線起訖。"""
    length = max(1, int(length))
    max_trim = int(round(length * ROAD_GRID_ALIGN_MAX_TRIM))
    steps = max(5, ROAD_GRID_ALIGN_SEARCH_STEPS)
    starts = np.unique(np.rint(np.linspace(0, max_trim, steps)).astype(int))
    ends = np.unique(np.rint(np.linspace(length - max_trim, length, steps)).astype(int))
    projection = np.asarray(edge_projection, dtype=np.float64).reshape(-1)
    if projection.size != length:
        projection = np.resize(projection, length)
    denominator = float(np.percentile(projection, 92)) if projection.size else 0.0
    denominator = max(1e-9, denominator)
    coordinates = np.asarray(color_coordinates, dtype=np.float64).reshape(-1)

    def score_candidate(start: int, end: int) -> Tuple[float, float, float, float]:
        width = float(end - start)
        if width < max(6.0, divisions * 2.0):
            return -1.0, 0.0, 0.0, 0.0
        inside = coordinates[(coordinates >= start) & (coordinates < end)]
        coverage = float(inside.size / max(1, coordinates.size)) if coordinates.size else 0.0
        if inside.size:
            pitch = width / divisions
            phase = np.mod((inside - start) / pitch, 1.0)
            center_score = float(np.mean(np.clip(1.0 - np.abs(phase - 0.5) / 0.5, 0.0, 1.0)))
        else:
            center_score = 0.0
        boundaries = np.rint(np.linspace(start, end, divisions + 1)).astype(int)
        edge_samples: List[float] = []
        for boundary in boundaries:
            left = max(0, boundary - 1)
            right = min(length, boundary + 2)
            if right > left:
                edge_samples.append(float(np.max(projection[left:right])) / denominator)
        edge_score = min(1.0, float(np.mean(edge_samples)) if edge_samples else 0.0)
        trim_ratio = (start + (length - end)) / max(1.0, float(length))
        score = (
            0.58 * center_score
            + 0.27 * edge_score
            + 0.15 * coverage
            - 0.08 * (trim_ratio / max(1e-9, ROAD_GRID_ALIGN_MAX_TRIM * 2.0))
        )
        return score, center_score, edge_score, coverage

    nominal_score, nominal_center, nominal_edge, nominal_coverage = score_candidate(0, length)
    best = {
        "start": 0.0,
        "end": float(length),
        "score": float(max(0.0, nominal_score)),
        "center_score": float(nominal_center),
        "edge_score": float(nominal_edge),
        "coverage": float(nominal_coverage),
        "nominal_score": float(max(0.0, nominal_score)),
    }
    for start in starts:
        for end in ends:
            if end <= start:
                continue
            score, center_score, edge_score, coverage = score_candidate(int(start), int(end))
            if score > best["score"] + 1e-12:
                best.update(
                    {
                        "start": float(start),
                        "end": float(end),
                        "score": float(max(0.0, score)),
                        "center_score": float(center_score),
                        "edge_score": float(edge_score),
                        "coverage": float(coverage),
                    }
                )
    best["gain"] = float(best["score"] - best["nominal_score"])
    best["scale"] = float((best["end"] - best["start"]) / max(1.0, float(length)))
    best["offset"] = float(best["start"])
    return best


def _effective_grid_bounds(crop: np.ndarray, union_mask: np.ndarray) -> Dict[str, Any]:
    height, width = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edge_x = np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)), axis=0)
    edge_y = np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)), axis=1)
    ys, xs = np.nonzero(union_mask)
    x_alignment = _axis_alignment(xs, edge_x, width, ROAD_GRID_COLS)
    y_alignment = _axis_alignment(ys, edge_y, height, ROAD_GRID_ROWS)
    x1 = int(round(x_alignment["start"]))
    x2 = int(round(x_alignment["end"]))
    y1 = int(round(y_alignment["start"]))
    y2 = int(round(y_alignment["end"]))
    x1 = max(0, min(width - 1, x1))
    x2 = max(x1 + 1, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(y1 + 1, min(height, y2))
    score = float((x_alignment["score"] + y_alignment["score"]) / 2.0)
    coverage = float((x_alignment["coverage"] + y_alignment["coverage"]) / 2.0)
    return {
        "x": x1,
        "y": y1,
        "width": x2 - x1,
        "height": y2 - y1,
        "score": score,
        "coverage": coverage,
        "offset_x": x1,
        "offset_y": y1,
        "scale_x": (x2 - x1) / max(1.0, float(width)),
        "scale_y": (y2 - y1) / max(1.0, float(height)),
        "gain_x": float(x_alignment["gain"]),
        "gain_y": float(y_alignment["gain"]),
        "x_axis": x_alignment,
        "y_axis": y_alignment,
    }


def _integral(mask: np.ndarray) -> np.ndarray:
    return cv2.integral(mask.astype(np.uint8), sdepth=cv2.CV_32S)


def _rect_sum(integral: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> int:
    return int(integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1])


def _green_component_stats(mask: np.ndarray) -> Tuple[int, float, float, float]:
    if mask.size == 0 or int(mask.sum()) <= 0:
        return 0, 0.0, 0.0, 0.0
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return 0, 0.0, 0.0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    index = int(np.argmax(areas)) + 1
    largest = int(stats[index, cv2.CC_STAT_AREA])
    width = int(stats[index, cv2.CC_STAT_WIDTH])
    height = int(stats[index, cv2.CC_STAT_HEIGHT])
    total = max(1, int(mask.sum()))
    return largest, largest / total, width / max(1, mask.shape[1]), height / max(1, mask.shape[0])


def _classify_grid(
    crop: np.ndarray,
    red_mask: np.ndarray,
    blue_mask: np.ndarray,
    green_mask: np.ndarray,
    bounds: Mapping[str, Any],
) -> Dict[str, Any]:
    red_integral = _integral(red_mask)
    blue_integral = _integral(blue_mask)
    green_integral = _integral(green_mask)
    grid_x = int(bounds["x"])
    grid_y = int(bounds["y"])
    grid_width = int(bounds["width"])
    grid_height = int(bounds["height"])
    recognized: List[Dict[str, Any]] = []
    uncertain: List[Dict[str, Any]] = []
    all_cells: List[Dict[str, Any]] = []

    for row in range(ROAD_GRID_ROWS):
        local_y1, local_y2 = _grid_bounds(grid_height, row, ROAD_GRID_ROWS)
        y1, y2 = grid_y + local_y1, grid_y + local_y2
        for column in range(ROAD_GRID_COLS):
            local_x1, local_x2 = _grid_bounds(grid_width, column, ROAD_GRID_COLS)
            x1, x2 = grid_x + local_x1, grid_x + local_x2
            cell_width = max(1, x2 - x1)
            cell_height = max(1, y2 - y1)
            margin_ratio_x = min(
                ROAD_GRID_INNER_MARGIN_MAX,
                max(ROAD_GRID_INNER_MARGIN, ROAD_GRID_BOUNDARY_GUARD_PX / cell_width),
            )
            margin_ratio_y = min(
                ROAD_GRID_INNER_MARGIN_MAX,
                max(ROAD_GRID_INNER_MARGIN, ROAD_GRID_BOUNDARY_GUARD_PX / cell_height),
            )
            margin_x = max(1, int(round(cell_width * margin_ratio_x)))
            margin_y = max(1, int(round(cell_height * margin_ratio_y)))
            inner_x1 = min(x2 - 1, x1 + margin_x)
            inner_x2 = max(inner_x1 + 1, x2 - margin_x)
            inner_y1 = min(y2 - 1, y1 + margin_y)
            inner_y2 = max(inner_y1 + 1, y2 - margin_y)
            inner_area = max(1, (inner_x2 - inner_x1) * (inner_y2 - inner_y1))
            red_pixels = _rect_sum(red_integral, inner_x1, inner_y1, inner_x2, inner_y2)
            blue_pixels = _rect_sum(blue_integral, inner_x1, inner_y1, inner_x2, inner_y2)
            green_pixels = _rect_sum(green_integral, inner_x1, inner_y1, inner_x2, inner_y2)
            minimum_pixels = max(
                ROAD_GRID_MIN_COLOR_PIXELS,
                int(round(inner_area * ROAD_GRID_MIN_COLOR_RATIO)),
            )
            dominant_pixels = max(red_pixels, blue_pixels)
            secondary_pixels = min(red_pixels, blue_pixels)
            dominance = (dominant_pixels + 1.0) / (secondary_pixels + 1.0)
            outcome = ""
            is_uncertain = False
            if dominant_pixels >= minimum_pixels:
                if dominance >= ROAD_GRID_COLOR_DOMINANCE:
                    outcome = "B" if red_pixels > blue_pixels else "P"
                else:
                    is_uncertain = True

            largest_green, green_concentration, green_span_x, green_span_y = _green_component_stats(
                green_mask[inner_y1:inner_y2, inner_x1:inner_x2]
            )
            tie_minimum = max(
                ROAD_GRID_TIE_MIN_PIXELS,
                int(round(inner_area * ROAD_GRID_TIE_MIN_AREA_RATIO)),
            )
            tie_confident = bool(
                outcome
                and largest_green >= tie_minimum
                and green_concentration >= ROAD_GRID_TIE_MIN_COMPONENT_RATIO
                and max(green_span_x, green_span_y) <= ROAD_GRID_TIE_MAX_SPAN_RATIO
            )
            separation = max(
                0.0,
                min(
                    1.0,
                    (dominant_pixels - secondary_pixels)
                    / max(1.0, dominant_pixels + secondary_pixels),
                ),
            )
            pixel_strength = max(
                0.0, min(1.0, dominant_pixels / max(1.0, minimum_pixels * 2.0))
            )
            confidence = 0.55 * pixel_strength + 0.45 * separation if outcome else 0.0
            cell = {
                "index": -1,
                "outcome": outcome,
                "uncertain": bool(is_uncertain),
                "empty": bool(not outcome and not is_uncertain),
                "column": column,
                "row": row,
                "x": x1,
                "y": y1,
                "width": cell_width,
                "height": cell_height,
                "inner_x": inner_x1,
                "inner_y": inner_y1,
                "inner_width": inner_x2 - inner_x1,
                "inner_height": inner_y2 - inner_y1,
                "cx": round((x1 + x2) / 2.0, 2),
                "cy": round((y1 + y2) / 2.0, 2),
                "red_pixels": int(red_pixels),
                "blue_pixels": int(blue_pixels),
                "green_pixels": int(green_pixels),
                "minimum_color_pixels": int(minimum_pixels),
                "dominance": round(float(dominance), 6),
                "green_largest_component": int(largest_green),
                "green_component_ratio": round(float(green_concentration), 6),
                "green_span_x_ratio": round(float(green_span_x), 6),
                "green_span_y_ratio": round(float(green_span_y), 6),
                "tie_count": 1 if tie_confident else 0,
                "confidence": round(float(confidence), 6),
            }
            all_cells.append(cell)
            if outcome:
                recognized.append(dict(cell))
            elif is_uncertain:
                uncertain.append(dict(cell))

    return {"cells": recognized, "uncertain_cells": uncertain, "all_grid_cells": all_cells}


def _reconstruct_big_road_order(
    cells: Sequence[Mapping[str, Any]],
    *,
    return_details: bool = False,
) -> Any:
    """依六列大路規則反推時間序；長龍轉右後會保持右黏，不再錯誤往下。"""
    grid: Dict[Tuple[int, int], str] = {
        (int(item.get("column", 0)), int(item.get("row", 0))): str(
            item.get("outcome") or ""
        ).upper()
        for item in cells
        if str(item.get("outcome") or "").upper() in {"B", "P"}
    }
    fallback_preview = sorted(grid, key=lambda position: (position[0], position[1]))
    if not grid:
        details = {
            "positions": [],
            "reconstructed_all": False,
            "fallback_reason": "no_recognized_cells",
            "partial_positions": [],
            "fallback_preview": [],
            "solution_count": 0,
        }
        return details if return_details else []
    if (0, 0) not in grid:
        details = {
            "positions": [],
            "reconstructed_all": False,
            "fallback_reason": "missing_big_road_origin_0_0",
            "partial_positions": [],
            "fallback_preview": fallback_preview,
            "solution_count": 0,
        }
        return details if return_details else []

    target_count = len(grid)
    first_outcome = grid[(0, 0)]
    best_partial: List[Tuple[int, int]] = [(0, 0)]
    solutions: List[List[Tuple[int, int]]] = []

    def search(
        current: Tuple[int, int],
        run_start_column: int,
        previous: str,
        tailing_right: bool,
        visited: set[Tuple[int, int]],
        ordered: List[Tuple[int, int]],
    ) -> None:
        nonlocal best_partial
        if len(ordered) > len(best_partial):
            best_partial = list(ordered)
        if len(ordered) == target_count:
            solutions.append(list(ordered))
            return
        if len(solutions) >= 2:
            return

        column, row = current
        options: List[Tuple[Tuple[int, int], int, str, bool]] = []
        if not tailing_right and row < ROAD_GRID_ROWS - 1 and (column, row + 1) not in visited:
            same_position = (column, row + 1)
            same_tailing = False
        else:
            next_column = column + 1
            while (next_column, row) in visited:
                next_column += 1
            same_position = (next_column, row)
            same_tailing = True
        if grid.get(same_position) == previous and same_position not in visited:
            options.append((same_position, run_start_column, previous, same_tailing))

        opposite = "P" if previous == "B" else "B"
        next_start_column = run_start_column + 1
        while (next_start_column, 0) in visited:
            next_start_column += 1
        opposite_position = (next_start_column, 0)
        if grid.get(opposite_position) == opposite and opposite_position not in visited:
            options.append((opposite_position, next_start_column, opposite, False))

        for position, candidate_start, candidate_outcome, candidate_tailing in options:
            visited.add(position)
            ordered.append(position)
            search(
                position,
                candidate_start,
                candidate_outcome,
                candidate_tailing,
                visited,
                ordered,
            )
            ordered.pop()
            visited.remove(position)
            if len(solutions) >= 2:
                return

    search((0, 0), 0, first_outcome, False, {(0, 0)}, [(0, 0)])
    unique = len(solutions) == 1
    positions = solutions[0] if unique else []
    if unique:
        fallback_reason = ""
    elif len(solutions) > 1:
        fallback_reason = "ambiguous_big_road_reconstruction"
    else:
        fallback_reason = (
            f"incomplete_big_road_reconstruction_{len(best_partial)}_of_{target_count}"
        )
    details = {
        "positions": positions,
        "reconstructed_all": unique and len(positions) == target_count,
        "fallback_reason": fallback_reason,
        "partial_positions": best_partial,
        "fallback_preview": fallback_preview,
        "solution_count": len(solutions),
    }
    return details if return_details else positions


def _debug_overlay(
    crop: np.ndarray,
    all_cells: Sequence[Mapping[str, Any]],
    grid_bounds: Mapping[str, Any],
    quality_ok: bool,
    fallback_reason: str,
) -> str:
    if not ROAD_GRID_DEBUG:
        return ""
    overlay = crop.copy()
    x, y = int(grid_bounds["x"]), int(grid_bounds["y"])
    width, height = int(grid_bounds["width"]), int(grid_bounds["height"])
    cv2.rectangle(overlay, (x, y), (x + width - 1, y + height - 1), (255, 255, 255), 1)
    for column in range(ROAD_GRID_COLS + 1):
        line_x = int(round(x + column * width / ROAD_GRID_COLS))
        cv2.line(overlay, (line_x, y), (line_x, y + height), (160, 160, 160), 1)
    for row in range(ROAD_GRID_ROWS + 1):
        line_y = int(round(y + row * height / ROAD_GRID_ROWS))
        cv2.line(overlay, (x, line_y), (x + width, line_y), (160, 160, 160), 1)
    for cell in all_cells:
        label = str(cell.get("outcome") or "")
        if bool(cell.get("uncertain")):
            label = "?"
        if int(cell.get("tie_count", 0) or 0) > 0:
            label += "T"
        if not label:
            continue
        cx = int(round(float(cell.get("cx", 0))))
        cy = int(round(float(cell.get("cy", 0))))
        cv2.putText(
            overlay,
            label,
            (max(0, cx - 7), max(10, cy + 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    status = "OK" if quality_ok else f"RETAKE:{fallback_reason or 'quality'}"
    cv2.putText(
        overlay,
        status[:80],
        (4, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    directory = Path(ROAD_GRID_DEBUG_DIR or "/tmp/bgs_road_debug")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"road_grid_{time.time_ns()}.png"
    cv2.imwrite(str(path), overlay)
    return str(path)


def _detect_fixed_grid(crop: np.ndarray) -> Dict[str, Any]:
    """固定 6×15，先微調有效區，再以像素量、dominance 與集中綠點分類。"""
    if crop is None or crop.size == 0:
        raise ValueError("固定大路裁圖為空。")
    image_height, image_width = crop.shape[:2]
    red_mask, blue_mask, green_mask, union_mask = _color_masks(crop)
    grid_bounds = _effective_grid_bounds(crop, union_mask)
    classified = _classify_grid(crop, red_mask, blue_mask, green_mask, grid_bounds)
    cells = list(classified["cells"])
    uncertain_cells = list(classified["uncertain_cells"])
    all_grid_cells = list(classified["all_grid_cells"])
    reconstruction = _reconstruct_big_road_order(cells, return_details=True)
    ordered_positions = list(reconstruction["positions"])
    cell_lookup = {(int(item["column"]), int(item["row"])): item for item in cells}
    ordered_cells: List[Dict[str, Any]] = []
    sequence: List[str] = []
    raw_outcomes: List[str] = []
    tie_markers: Dict[str, int] = {}
    if reconstruction["reconstructed_all"]:
        for index, position in enumerate(ordered_positions):
            source = dict(cell_lookup[position])
            source["index"] = index
            source["chronology_confirmed"] = True
            ordered_cells.append(source)
            outcome = str(source["outcome"])
            sequence.append(outcome)
            raw_outcomes.append(outcome)
            tie_count = int(source.get("tie_count", 0) or 0)
            if tie_count > 0:
                tie_markers[str(index)] = tie_count
                raw_outcomes.extend(["T"] * tie_count)
    else:
        for source in sorted(cells, key=lambda item: (int(item["column"]), int(item["row"]))):
            item = dict(source)
            item["chronology_confirmed"] = False
            ordered_cells.append(item)

    uncertain_count = len(uncertain_cells)
    recognized_count = len(cells)
    candidate_total = recognized_count + uncertain_count
    uncertain_ratio = uncertain_count / max(1, candidate_total)
    confidences = [float(item.get("confidence", 0.0) or 0.0) for item in cells]
    median_confidence = float(np.median(confidences)) if confidences else 0.0
    alignment_ok = bool(
        float(grid_bounds["score"]) >= ROAD_GRID_MIN_ALIGNMENT_SCORE
        and float(grid_bounds["coverage"]) >= 0.80
    )
    quality_ok = bool(
        recognized_count >= ROAD_GRID_MIN_RECOGNIZED
        and uncertain_ratio <= ROAD_GRID_MAX_UNCERTAIN_RATIO
        and bool(reconstruction["reconstructed_all"])
        and alignment_ok
        and median_confidence >= ROAD_GRID_MIN_MEDIAN_CONFIDENCE
    )
    fallback_reason = str(reconstruction.get("fallback_reason") or "")
    if not fallback_reason and not alignment_ok:
        fallback_reason = "grid_alignment_not_confident"
    if not fallback_reason and median_confidence < ROAD_GRID_MIN_MEDIAN_CONFIDENCE:
        fallback_reason = "cell_color_confidence_too_low"
    if not fallback_reason and recognized_count < ROAD_GRID_MIN_RECOGNIZED:
        fallback_reason = "recognized_count_below_minimum"
    if not fallback_reason and uncertain_ratio > ROAD_GRID_MAX_UNCERTAIN_RATIO:
        fallback_reason = "too_many_uncertain_cells"
    debug_overlay_path = _debug_overlay(
        crop, all_grid_cells, grid_bounds, quality_ok, fallback_reason
    )

    return {
        "ok": bool(sequence) and quality_ok,
        "quality_ok": quality_ok,
        "sequence": sequence,
        "raw_outcomes": raw_outcomes,
        "tie_markers": tie_markers,
        "grid_cells": ordered_cells,
        "all_grid_cells": all_grid_cells,
        "recognized_count": recognized_count,
        "sequence_count": len(sequence),
        "confirmed_round_count": len(raw_outcomes),
        "uncertain_count": uncertain_count,
        "unknown_candidates": uncertain_count,
        "unknown_ratio": round(uncertain_ratio, 6),
        "raw_contours": 0,
        "candidates": ordered_cells,
        "method": "fixed_hsv_grid_6x15_adaptive_v10_9_1",
        "grid_rows": ROAD_GRID_ROWS,
        "grid_columns": ROAD_GRID_COLS,
        "grid_size": {"width": image_width, "height": image_height},
        "effective_grid": {
            key: grid_bounds[key]
            for key in (
                "x",
                "y",
                "width",
                "height",
                "score",
                "coverage",
                "offset_x",
                "offset_y",
                "scale_x",
                "scale_y",
                "gain_x",
                "gain_y",
            )
        },
        "grid_alignment": grid_bounds,
        "alignment_ok": alignment_ok,
        "median_cell_confidence": round(median_confidence, 6),
        "reconstructed_all": bool(reconstruction["reconstructed_all"]),
        "reconstruction_solution_count": int(reconstruction["solution_count"]),
        "partial_reconstruction": list(reconstruction["partial_positions"]),
        "fallback_preview_positions": list(reconstruction["fallback_preview"]),
        "fallback_reason": fallback_reason,
        "uncertain_cells": uncertain_cells,
        "count_is_confirmed": bool(quality_ok and uncertain_count == 0),
        "debug_overlay_path": debug_overlay_path,
        "debug_enabled": ROAD_GRID_DEBUG,
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
        "reconstructed_all": bool(ordered),
        "fallback_reason": "" if ordered else "yolo_no_detections",
    }


def _score_result(result: Mapping[str, Any], preference: float = 0.0) -> float:
    recognized = int(result.get("recognized_count", 0) or 0)
    unknown = int(result.get("unknown_candidates", result.get("uncertain_count", 0)) or 0)
    raw = int(result.get("raw_contours", 0) or 0)
    if recognized <= 0:
        return -9999.0
    noise = max(0, raw - recognized * 8)
    fixed = str(result.get("method") or "").startswith("fixed_")
    quality_bonus = 55.0 if bool(result.get("quality_ok")) else -35.0
    reconstruction_bonus = 25.0 if bool(result.get("reconstructed_all", not fixed)) else -45.0
    alignment = float(dict(result.get("effective_grid") or {}).get("score", 0.0) or 0.0)
    # 固定格成功時額外加分，避免被找圓備援搶過。
    fixed_success_bonus = 40.0 if fixed and bool(result.get("quality_ok")) else 0.0
    return (
        recognized * 3.0
        - unknown * 4.0
        - noise * 0.04
        + preference
        + quality_bonus
        + reconstruction_bonus
        + alignment * 20.0
        + fixed_success_bonus
    )


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
        result.setdefault("reconstructed_all", bool(result.get("quality_ok")))
        result.setdefault(
            "fallback_reason",
            ""
            if result.get("quality_ok")
            else "contour_circle_detection_failed",
        )

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
    """只有固定格且 quality 通過才允許 early exit，避免找圓半成品提前結束。"""
    recognized = int(result.get("recognized_count", 0) or 0)
    unknown = int(
        result.get("unknown_candidates", result.get("uncertain_count", 0)) or 0
    )
    ratio = unknown / max(1, recognized + unknown)
    return bool(
        bool(result.get("fixed_grid"))
        and bool(result.get("quality_ok"))
        and bool(result.get("reconstructed_all", True))
        and recognized >= ROAD_FAST_MIN_RECOGNIZED
        and ratio <= ROAD_FAST_MAX_UNKNOWN_RATIO
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

    if requested not in {"auto", "full_screen", "road_crop", "wide_multi_road"}:
        requested = "auto"
    wide_multi_road = requested == "wide_multi_road" or (
        requested == "auto" and aspect >= WIDE_LAYOUT_MIN_ASPECT
    )
    likely_crop = requested == "road_crop" or (
        requested == "auto" and ROAD_CROP_MIN_ASPECT <= aspect < WIDE_LAYOUT_MIN_ASPECT
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
        # MT 完整畫面：只掃右下方固定 6×15。
        plan = [("mt_fixed_6x15", MT_FIXED_ROAD_ROI, 20.0, True)]
    elif likely_crop:
        plan = [("road_crop_fixed_6x15", (0.0, 0.0, 1.0, 1.0), 15.0, True)]
    elif wide_multi_road:
        plan = [("wide_top_fixed_6x15", WIDE_TOP_ROAD_ROI, 10.0, True)]
    else:
        # 完整畫面（含 DG）：一律先走固定 6×15，找圓僅備援。
        if venue_code in VENUE_ROIS:
            plan.append(
                (
                    f"{venue_code.lower()}_venue_fixed_6x15",
                    VENUE_ROIS[venue_code],
                    18.0,
                    True,
                )
            )
            for alt_name, alt_roi in VENUE_FALLBACK_ROIS.get(venue_code, []):
                plan.append((f"{alt_name}_fixed_6x15", alt_roi, 12.0, True))
        plan.append(("generic_road_fixed_6x15", ROAD_ROI, 10.0, True))
        if ROAD_CONTOUR_FALLBACK:
            # 備援：同一 ROI 走找圓（分數較低，不會蓋過成功固定格）。
            if venue_code in VENUE_ROIS:
                plan.append(
                    (
                        f"{venue_code.lower()}_venue_contour",
                        VENUE_ROIS[venue_code],
                        2.0,
                        False,
                    )
                )
            plan.append(("generic_road_contour", ROAD_ROI, 1.0, False))
        if ROAD_AUTO_FULL_FALLBACK:
            plan.append(("full_image_fixed_6x15", (0.0, 0.0, 1.0, 1.0), 4.0, True))

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
            "quality_ok": False,
            "sequence": [],
            "recognized_count": 0,
            "method": "failed",
            "input_type": "unknown",
            "fallback_reason": "all_regions_failed",
            "errors": errors,
            "attempted_regions": attempted,
            "reconstructed_all": False,
        }

    result = dict(best)
    result.update(
        {
            "ok": bool(result.get("sequence")) and bool(result.get("quality_ok", True)),
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
                    "quality_ok": bool(item.get("quality_ok", True)),
                    "reconstructed_all": bool(item.get("reconstructed_all", True)),
                    "fallback_reason": str(item.get("fallback_reason") or ""),
                    "alignment_score": float(
                        dict(item.get("effective_grid") or {}).get("score", 0.0) or 0.0
                    ),
                }
                for item in candidates
            ],
        }
    )
    return result


def detect_road_sequence(image_path: str | Path) -> List[str]:
    return list(detect_road_sequence_detailed(image_path).get("sequence") or [])


__all__ = ["detect_road_sequence", "detect_road_sequence_detailed"]
