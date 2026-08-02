"""百家樂大路圖片辨識模組 V10.1（格位重建修正版）。

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
    radius = max(4.0, candidate.diameter / 2.0)
    outer = max(3, int(round(radius * RING_OUTER_RATIO)))
    inner = max(1, int(round(radius * RING_INNER_RATIO)))
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


def _cluster_axis(values: Sequence[int], tolerance: float) -> List[float]:
    centers: List[List[float]] = []
    for value in sorted(float(v) for v in values):
        target = None
        for cluster in centers:
            if abs(value - float(np.mean(cluster))) <= tolerance:
                target = cluster
                break
        if target is None:
            centers.append([value])
        else:
            target.append(value)
    return [float(np.mean(cluster)) for cluster in centers]


def _nearest_index(value: float, centers: Sequence[float]) -> Tuple[int, float]:
    index = min(range(len(centers)), key=lambda i: abs(float(value) - centers[i]))
    return index, abs(float(value) - centers[index])


def _next_big_road_cell(
    current: Tuple[int, int],
    start_column: int,
    same_side: bool,
    occupied: set[Tuple[int, int]],
) -> Tuple[Tuple[int, int], int]:
    column, row = current
    if same_side:
        below = (column, row + 1)
        if row < 5 and below not in occupied:
            return below, start_column
        right = (column + 1, row)
        while right in occupied:
            right = (right[0] + 1, row)
        return right, start_column
    new_column = start_column + 1
    while (new_column, 0) in occupied:
        new_column += 1
    return (new_column, 0), new_column


def _reconstruct_big_road(
    grid: Dict[Tuple[int, int], CircleCandidate],
) -> List[CircleCandidate]:
    if not grid or (0, 0) not in grid:
        return []
    first = grid[(0, 0)]
    best: List[CircleCandidate] = []

    def walk(
        ordered: List[CircleCandidate],
        occupied: set[Tuple[int, int]],
        current: Tuple[int, int],
        start_column: int,
    ) -> None:
        nonlocal best
        if len(ordered) > len(best):
            best = list(ordered)
        if len(ordered) == len(grid):
            return
        last_side = ordered[-1].outcome
        options = []
        for next_side in (last_side, 'P' if last_side == 'B' else 'B'):
            cell, next_start = _next_big_road_cell(
                current, start_column, next_side == last_side, occupied
            )
            item = grid.get(cell)
            if item is not None and cell not in occupied and item.outcome == next_side:
                options.append((item, cell, next_start))
        for item, cell, next_start in options:
            occupied.add(cell)
            ordered.append(item)
            walk(ordered, occupied, cell, next_start)
            ordered.pop()
            occupied.remove(cell)

    walk([first], {(0, 0)}, (0, 0), 0)
    return best if len(best) == len(grid) else []


def _sort_big_road(candidates: Sequence[CircleCandidate]) -> List[CircleCandidate]:
    """依六列大路格位重建時間順序，而不是單純逐欄由上往下讀。"""
    if not candidates:
        return []
    diameters = [item.diameter for item in candidates]
    median_diameter = float(np.median(diameters))
    size_filtered = [
        item for item in candidates
        if 0.58 * median_diameter <= item.diameter <= 1.55 * median_diameter
    ] or list(candidates)

    x_centers = _cluster_axis([item.x for item in size_filtered], max(3.0, median_diameter * 0.48))
    y_centers = _cluster_axis([item.y for item in size_filtered], max(3.0, median_diameter * 0.48))
    x_centers.sort(); y_centers.sort()
    # 大路固定最多六列；全圖誤抓到其他區域時，只採最密集的六個水平格位。
    if len(y_centers) > 6:
        counts = []
        for center in y_centers:
            counts.append(sum(abs(item.y-center) <= median_diameter*0.48 for item in size_filtered))
        keep = sorted(range(len(y_centers)), key=lambda i: counts[i], reverse=True)[:6]
        y_centers = sorted(y_centers[i] for i in keep)
    if not x_centers or not y_centers:
        return []

    grid: Dict[Tuple[int, int], CircleCandidate] = {}
    max_residual = median_diameter * 0.46
    min_x = min(x_centers)
    for item in size_filtered:
        col, dx = _nearest_index(item.x, x_centers)
        row, dy = _nearest_index(item.y, y_centers)
        if dx > max_residual or dy > max_residual or row > 5:
            continue
        cell = (col, row)
        old = grid.get(cell)
        quality = max(item.red_ratio, item.blue_ratio, item.saturation / 255.0) + item.circularity
        old_quality = -1.0 if old is None else max(old.red_ratio, old.blue_ratio, old.saturation / 255.0) + old.circularity
        if old is None or quality > old_quality:
            grid[cell] = item

    if not grid:
        return []
    min_col = min(column for column, _ in grid)
    normalized = {(column-min_col, row): item for (column,row), item in grid.items()}
    reconstructed = _reconstruct_big_road(normalized)
    if reconstructed:
        return reconstructed

    # 無法唯一重建時保守回退；仍以格位而非原始輪廓座標排序。
    return [normalized[cell] for cell in sorted(normalized, key=lambda p: (p[0], p[1]))]


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
        colored.append(CircleCandidate(
            x=candidate.x, y=candidate.y, width=candidate.width, height=candidate.height,
            area=candidate.area, circularity=candidate.circularity, fill_ratio=candidate.fill_ratio,
            outcome=outcome, hue=round(hue, 3), saturation=round(saturation, 3), value=round(value, 3),
            red_ratio=round(red_ratio, 4), blue_ratio=round(blue_ratio, 4), color_method=method,
        ))

    ordered = _sort_big_road(colored)
    sequence = [item.outcome for item in ordered]
    return {
        "ok": bool(sequence),
        "sequence": sequence,
        "recognized_count": len(sequence),
        "unknown_candidates": unknown,
        "raw_contours": len(raw_candidates),
        "unique_geometry_candidates": len(unique_candidates),
        "image_size": {
            "width": int(image.shape[1]), "height": int(image.shape[0]),
            "resize_scale": round(float(resize_scale), 6),
        },
        "candidates": [asdict(item) for item in ordered],
        "method": "adaptive_contour_ring_hsv_with_center_fallback",
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
