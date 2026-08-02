"""百家樂大路圖片辨識模組（完整畫面／牌路裁切圖共用，支援綠色和局標記）。

改良重點：
1. 先以幾何輪廓定位圓圈，再以「外框環形區域」判定紅莊／藍閒。
2. 空心圓、中心數字與白色圓心不再直接造成顏色判定失敗。
3. 支援直接傳入 OpenCV ndarray，避免路紙裁圖寫入暫存檔的磁碟延遲。
4. 圖片會先限制分析尺寸，再進行輪廓與局部顏色運算。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import math
import os

import cv2
import numpy as np


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(str(os.getenv(name, default)).strip())
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(str(os.getenv(name, default)).strip())
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


MAX_IMAGE_SIDE = _env_int("VISION_MAX_IMAGE_SIDE", 1200, 480, 3000)
ADAPTIVE_BLOCK_SIZE = _env_int("VISION_ADAPTIVE_BLOCK_SIZE", 31, 11, 101)
if ADAPTIVE_BLOCK_SIZE % 2 == 0:
    ADAPTIVE_BLOCK_SIZE += 1
ADAPTIVE_C = _env_int("VISION_ADAPTIVE_C", 7, -30, 30)
MIN_CIRCLE_AREA = _env_float("VISION_MIN_CIRCLE_AREA", 24.0, 4.0, 5000.0)
MAX_CIRCLE_AREA_RATIO = _env_float("VISION_MAX_CIRCLE_AREA_RATIO", 0.025, 0.001, 0.20)
MIN_CIRCULARITY = _env_float("VISION_MIN_CIRCULARITY", 0.30, 0.05, 0.95)
MIN_SATURATION = _env_float("VISION_MIN_SATURATION", 34.0, 0.0, 255.0)
CENTER_PATCH_RADIUS = _env_int("VISION_CENTER_PATCH_RADIUS", 2, 1, 8)
RING_INNER_RATIO = _env_float("VISION_RING_INNER_RATIO", 0.22, 0.08, 0.42)
RING_OUTER_RATIO = _env_float("VISION_RING_OUTER_RATIO", 0.52, 0.30, 0.78)
RING_MIN_COLOR_RATIO = _env_float("VISION_RING_MIN_COLOR_RATIO", 0.10, 0.02, 0.60)
TIE_GREEN_MIN_HUE = _env_float("VISION_TIE_GREEN_MIN_HUE", 32.0, 20.0, 80.0)
TIE_GREEN_MAX_HUE = _env_float("VISION_TIE_GREEN_MAX_HUE", 92.0, 50.0, 120.0)
TIE_GREEN_MIN_SATURATION = _env_float("VISION_TIE_GREEN_MIN_SATURATION", 55.0, 10.0, 255.0)
TIE_GREEN_MIN_RATIO = _env_float("VISION_TIE_GREEN_MIN_RATIO", 0.025, 0.003, 0.30)
COLUMN_TOLERANCE_RATIO = _env_float("VISION_COLUMN_TOLERANCE_RATIO", 0.62, 0.25, 1.50)


@dataclass(frozen=True)
class CircleCandidate:
    x: int
    y: int
    width: int
    height: int
    area: float
    circularity: float
    fill_ratio: float
    outcome: str = ""
    hue: float = 0.0
    saturation: float = 0.0
    value: float = 0.0
    red_ratio: float = 0.0
    blue_ratio: float = 0.0
    green_ratio: float = 0.0
    tie_count: int = 0
    color_method: str = ""

    @property
    def diameter(self) -> float:
        return (float(self.width) + float(self.height)) / 2.0


def _read_image(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(Path(path)), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("無法讀取圖片，請確認格式為 JPG、PNG 或 WEBP。")
    return image


def _resize_for_analysis(image: np.ndarray) -> Tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= MAX_IMAGE_SIDE:
        return image, 1.0
    scale = MAX_IMAGE_SIDE / float(longest)
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _circular_hsv_mean(patch_hsv: np.ndarray) -> Tuple[float, float, float]:
    pixels = patch_hsv.reshape(-1, 3).astype(np.float64)
    if pixels.size == 0:
        return 0.0, 0.0, 0.0
    hues = pixels[:, 0]
    saturation = float(np.mean(pixels[:, 1]))
    value = float(np.mean(pixels[:, 2]))
    angles = hues / 180.0 * 2.0 * math.pi
    angle = math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))
    if angle < 0:
        angle += 2.0 * math.pi
    return angle / (2.0 * math.pi) * 180.0, saturation, value


def _center_patch_hsv(hsv: np.ndarray, cx: int, cy: int) -> Tuple[float, float, float]:
    height, width = hsv.shape[:2]
    radius = CENTER_PATCH_RADIUS
    x1, x2 = max(0, cx - radius), min(width, cx + radius + 1)
    y1, y2 = max(0, cy - radius), min(height, cy + radius + 1)
    return _circular_hsv_mean(hsv[y1:y2, x1:x2])


def _classify_local_color(hue: float, saturation: float, value: float) -> str:
    if saturation < MIN_SATURATION or value < 30.0:
        return ""
    red_distance = min(abs(hue), abs(hue - 180.0))
    blue_distance = abs(hue - 110.0)
    red_allowed = hue <= 20.0 or hue >= 160.0
    blue_allowed = 78.0 <= hue <= 145.0
    if red_allowed and (not blue_allowed or red_distance <= blue_distance):
        return "B"
    if blue_allowed:
        return "P"
    return ""


def _ring_color_stats(hsv: np.ndarray, candidate: CircleCandidate) -> Tuple[str, float, float, float, float, float]:
    """只讀取圓圈外框環形區域，適用紅／藍空心圓。"""
    height, width = hsv.shape[:2]
    # RING_* 比例以候選框直徑為基準；0.52 約等於圓半徑。
    diameter = max(8.0, candidate.diameter)
    outer = max(3, int(round(diameter * RING_OUTER_RATIO)))
    inner = max(1, int(round(diameter * RING_INNER_RATIO)))
    x1, x2 = max(0, candidate.x - outer), min(width, candidate.x + outer + 1)
    y1, y2 = max(0, candidate.y - outer), min(height, candidate.y + outer + 1)
    patch = hsv[y1:y2, x1:x2]
    if patch.size == 0:
        return "", 0.0, 0.0, 0.0, 0.0, 0.0

    yy, xx = np.ogrid[:patch.shape[0], :patch.shape[1]]
    cx = candidate.x - x1
    cy = candidate.y - y1
    distance2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = (distance2 <= outer ** 2) & (distance2 >= inner ** 2)
    pixels = patch[mask]
    if pixels.size == 0:
        return "", 0.0, 0.0, 0.0, 0.0, 0.0

    h = pixels[:, 0].astype(np.float64)
    s = pixels[:, 1].astype(np.float64)
    v = pixels[:, 2].astype(np.float64)
    valid = (s >= MIN_SATURATION) & (v >= 30.0)
    denominator = max(1, int(np.count_nonzero(mask)))
    red = valid & ((h <= 22.0) | (h >= 158.0))
    blue = valid & (h >= 75.0) & (h <= 148.0)
    red_ratio = float(np.count_nonzero(red)) / denominator
    blue_ratio = float(np.count_nonzero(blue)) / denominator

    colored = pixels[red | blue]
    hue, saturation, value = _circular_hsv_mean(colored.reshape(-1, 1, 3)) if colored.size else (0.0, 0.0, 0.0)
    best_ratio = max(red_ratio, blue_ratio)
    if best_ratio < RING_MIN_COLOR_RATIO:
        return "", red_ratio, blue_ratio, hue, saturation, value
    if red_ratio > blue_ratio * 1.10:
        return "B", red_ratio, blue_ratio, hue, saturation, value
    if blue_ratio > red_ratio * 1.10:
        return "P", red_ratio, blue_ratio, hue, saturation, value
    return "", red_ratio, blue_ratio, hue, saturation, value



def _tie_marker_stats(hsv: np.ndarray, candidate: CircleCandidate) -> Tuple[int, float]:
    """偵測大路圓圈內外的綠色和局標記。

    圖片只能可靠判定「此格曾出現和局」；若平台以綠色數字表示多次和局，
    精確次數仍以使用者後續按下 T 為準。
    """
    height, width = hsv.shape[:2]
    radius = max(4, int(round(candidate.diameter * 0.58)))
    x1, x2 = max(0, candidate.x - radius), min(width, candidate.x + radius + 1)
    y1, y2 = max(0, candidate.y - radius), min(height, candidate.y + radius + 1)
    patch = hsv[y1:y2, x1:x2]
    if patch.size == 0:
        return 0, 0.0
    yy, xx = np.ogrid[:patch.shape[0], :patch.shape[1]]
    cx, cy = candidate.x - x1, candidate.y - y1
    circle_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    h = patch[:, :, 0].astype(np.float64)
    sat = patch[:, :, 1].astype(np.float64)
    val = patch[:, :, 2].astype(np.float64)
    green = (
        circle_mask
        & (h >= TIE_GREEN_MIN_HUE)
        & (h <= TIE_GREEN_MAX_HUE)
        & (sat >= TIE_GREEN_MIN_SATURATION)
        & (val >= 35.0)
    )
    denominator = max(1, int(np.count_nonzero(circle_mask)))
    ratio = float(np.count_nonzero(green)) / denominator
    return (1 if ratio >= TIE_GREEN_MIN_RATIO else 0), ratio

def _preprocess_geometry(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    smallest_side = min(gray.shape[:2])
    if smallest_side < 5:
        raise ValueError("圖片尺寸太小，無法辨識路紙。")
    block_size = min(ADAPTIVE_BLOCK_SIZE, smallest_side if smallest_side % 2 else smallest_side - 1)
    block_size = max(3, block_size)
    white_background = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, ADAPTIVE_C
    )
    contour_map = cv2.bitwise_not(white_background)
    contour_map = cv2.morphologyEx(
        contour_map, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    )
    contour_map = cv2.morphologyEx(
        contour_map, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    return contour_map


def _geometry_candidates(image: np.ndarray, contour_map: np.ndarray) -> List[CircleCandidate]:
    contours, _ = cv2.findContours(contour_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(image.shape[0] * image.shape[1])
    max_area = max(MIN_CIRCLE_AREA * 4.0, image_area * MAX_CIRCLE_AREA_RATIO)
    candidates: List[CircleCandidate] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < MIN_CIRCLE_AREA or area > max_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if width < 6 or height < 6:
            continue
        aspect_ratio = width / float(max(1, height))
        if not 0.76 <= aspect_ratio <= 1.28:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            continue
        fill_ratio = area / float(max(1, width * height))
        if not 0.08 <= fill_ratio <= 0.96:
            continue
        candidates.append(CircleCandidate(
            x=int(x + width / 2), y=int(y + height / 2), width=int(width), height=int(height),
            area=area, circularity=circularity, fill_ratio=fill_ratio,
        ))
    return candidates


def _deduplicate_candidates(candidates: Sequence[CircleCandidate]) -> List[CircleCandidate]:
    if not candidates:
        return []
    median_diameter = float(np.median([item.diameter for item in candidates]))
    center_threshold = max(3.0, median_diameter * 0.38)
    ordered = sorted(
        candidates,
        key=lambda item: (item.circularity, -abs(item.diameter - median_diameter), item.area),
        reverse=True,
    )
    accepted: List[CircleCandidate] = []
    for item in ordered:
        if any(math.hypot(item.x - old.x, item.y - old.y) <= center_threshold for old in accepted):
            continue
        accepted.append(item)
    return accepted


def _sort_big_road(candidates: Sequence[CircleCandidate]) -> List[CircleCandidate]:
    if not candidates:
        return []
    median_diameter = float(np.median([item.diameter for item in candidates]))
    x_tolerance = max(4.0, median_diameter * COLUMN_TOLERANCE_RATIO)
    columns: List[List[CircleCandidate]] = []
    for item in sorted(candidates, key=lambda value: (value.x, value.y)):
        target = None
        best = float("inf")
        for column in columns:
            center_x = float(np.mean([member.x for member in column]))
            distance = abs(item.x - center_x)
            if distance <= x_tolerance and distance < best:
                target, best = column, distance
        if target is None:
            columns.append([item])
        else:
            target.append(item)
    columns.sort(key=lambda column: float(np.mean([item.x for item in column])))
    ordered: List[CircleCandidate] = []
    for column in columns:
        ordered.extend(sorted(column, key=lambda item: (item.y, item.x)))
    return ordered


def analyze_baccarat_array_detailed(source: np.ndarray) -> Dict[str, Any]:
    if source is None or not isinstance(source, np.ndarray) or source.size == 0:
        raise ValueError("無法讀取路紙圖片。")
    image, resize_scale = _resize_for_analysis(source.copy())
    contour_map = _preprocess_geometry(image)
    raw_candidates = _geometry_candidates(image, contour_map)
    unique_candidates = _deduplicate_candidates(raw_candidates)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    colored: List[CircleCandidate] = []
    unknown = 0
    for candidate in unique_candidates:
        outcome, red_ratio, blue_ratio, hue, saturation, value = _ring_color_stats(hsv, candidate)
        method = "ring_hsv"
        if not outcome:
            hue, saturation, value = _center_patch_hsv(hsv, candidate.x, candidate.y)
            outcome = _classify_local_color(hue, saturation, value)
            method = "center_hsv_fallback" if outcome else "unknown"
        if not outcome:
            unknown += 1
            continue
        tie_count, green_ratio = _tie_marker_stats(hsv, candidate)
        colored.append(CircleCandidate(
            x=candidate.x, y=candidate.y, width=candidate.width, height=candidate.height,
            area=candidate.area, circularity=candidate.circularity, fill_ratio=candidate.fill_ratio,
            outcome=outcome, hue=round(hue, 3), saturation=round(saturation, 3), value=round(value, 3),
            red_ratio=round(red_ratio, 4), blue_ratio=round(blue_ratio, 4),
            green_ratio=round(green_ratio, 4), tie_count=tie_count, color_method=method,
        ))

    ordered = _sort_big_road(colored)
    sequence = [item.outcome for item in ordered]
    tie_markers = {
        str(index): int(item.tie_count)
        for index, item in enumerate(ordered)
        if int(item.tie_count) > 0
    }
    raw_outcomes: List[str] = []
    for index, item in enumerate(ordered):
        raw_outcomes.append(item.outcome)
        raw_outcomes.extend(["T"] * int(tie_markers.get(str(index), 0)))
    return {
        "ok": bool(sequence),
        "sequence": sequence,
        "raw_outcomes": raw_outcomes,
        "tie_markers": tie_markers,
        "tie_count": sum(tie_markers.values()),
        "tie_count_estimated_from_image": bool(tie_markers),
        "recognized_count": len(sequence),
        "unknown_candidates": unknown,
        "raw_contours": len(raw_candidates),
        "unique_geometry_candidates": len(unique_candidates),
        "image_size": {
            "width": int(image.shape[1]), "height": int(image.shape[0]),
            "resize_scale": round(float(resize_scale), 6),
        },
        "candidates": [asdict(item) for item in ordered],
        "method": "adaptive_contour_ring_hsv_green_tie_marker",
    }


def analyze_baccarat_image_detailed(image_path: str | Path) -> Dict[str, Any]:
    return analyze_baccarat_array_detailed(_read_image(image_path))


def analyze_baccarat_image(image_path: str | Path) -> List[str]:
    return list(analyze_baccarat_image_detailed(image_path).get("sequence") or [])


__all__ = [
    "analyze_baccarat_array_detailed",
    "analyze_baccarat_image",
    "analyze_baccarat_image_detailed",
]
