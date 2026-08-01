"""百家樂大路截圖辨識模組。

設計原則：
1. 幾何定位與顏色判定完全分離。
2. 先以灰階、自適應二值化與輪廓幾何尋找近似圓形位置。
3. 不建立整張圖片的紅藍遮罩；只在候選圓心附近讀取極小區域顏色。
4. 回傳 B（莊）與 P（閒）序列，以及可供除錯的診斷資訊。

注意：不同平台的路紙尺寸、縮放與顏色可能不同，因此所有門檻皆可用
環境變數微調。此模組不保證每張截圖都能完全辨識，LINE 端仍應保留
手動補輸與清除功能。
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


MAX_IMAGE_SIDE = _env_int("VISION_MAX_IMAGE_SIDE", 1800, 480, 4000)
ADAPTIVE_BLOCK_SIZE = _env_int("VISION_ADAPTIVE_BLOCK_SIZE", 31, 11, 101)
if ADAPTIVE_BLOCK_SIZE % 2 == 0:
    ADAPTIVE_BLOCK_SIZE += 1
ADAPTIVE_C = _env_int("VISION_ADAPTIVE_C", 7, -30, 30)
MIN_CIRCLE_AREA = _env_float("VISION_MIN_CIRCLE_AREA", 35.0, 5.0, 5000.0)
MAX_CIRCLE_AREA_RATIO = _env_float(
    "VISION_MAX_CIRCLE_AREA_RATIO", 0.025, 0.001, 0.20
)
MIN_CIRCULARITY = _env_float("VISION_MIN_CIRCULARITY", 0.34, 0.05, 0.95)
MIN_SATURATION = _env_float("VISION_MIN_SATURATION", 38.0, 0.0, 255.0)
CENTER_PATCH_RADIUS = _env_int("VISION_CENTER_PATCH_RADIUS", 2, 1, 6)  # 2 => 5x5
COLUMN_TOLERANCE_RATIO = _env_float(
    "VISION_COLUMN_TOLERANCE_RATIO", 0.62, 0.25, 1.50
)


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

    @property
    def diameter(self) -> float:
        return (float(self.width) + float(self.height)) / 2.0


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
    """計算 HSV 平均值，Hue 使用圓形平均避免紅色跨越 0/179。"""
    pixels = patch_hsv.reshape(-1, 3).astype(np.float64)
    if pixels.size == 0:
        return 0.0, 0.0, 0.0

    hues = pixels[:, 0]
    saturation = float(np.mean(pixels[:, 1]))
    value = float(np.mean(pixels[:, 2]))

    angles = hues / 180.0 * 2.0 * math.pi
    sin_mean = float(np.mean(np.sin(angles)))
    cos_mean = float(np.mean(np.cos(angles)))
    angle = math.atan2(sin_mean, cos_mean)
    if angle < 0:
        angle += 2.0 * math.pi
    hue = angle / (2.0 * math.pi) * 180.0
    return hue, saturation, value


def _center_patch_hsv(hsv: np.ndarray, cx: int, cy: int) -> Tuple[float, float, float]:
    """只提取圓心周圍 5x5（預設半徑 2）局部區域。"""
    height, width = hsv.shape[:2]
    radius = CENTER_PATCH_RADIUS
    x1 = max(0, cx - radius)
    x2 = min(width, cx + radius + 1)
    y1 = max(0, cy - radius)
    y2 = min(height, cy + radius + 1)
    return _circular_hsv_mean(hsv[y1:y2, x1:x2])


def _classify_local_color(hue: float, saturation: float, value: float) -> str:
    """依圓心局部 HSV 判定紅色 B 或藍色 P。"""
    if saturation < MIN_SATURATION or value < 35.0:
        return ""

    # OpenCV Hue 範圍為 0~179。紅色位於兩端，藍色通常位於 85~135。
    red_distance = min(abs(hue - 0.0), abs(hue - 180.0))
    blue_distance = abs(hue - 110.0)

    red_allowed = hue <= 18.0 or hue >= 165.0
    blue_allowed = 78.0 <= hue <= 142.0

    if red_allowed and (not blue_allowed or red_distance <= blue_distance):
        return "B"
    if blue_allowed:
        return "P"
    return ""


def _preprocess_geometry(image: np.ndarray) -> np.ndarray:
    """產生供 findContours 使用的幾何圖，不進行全圖紅藍遮罩。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # OpenCV 要求 block size 為大於 1 的奇數；小尺寸裁圖時自動縮小。
    smallest_side = min(gray.shape[:2])
    if smallest_side < 5:
        raise ValueError("圖片尺寸太小，無法辨識路紙。")
    block_size = min(ADAPTIVE_BLOCK_SIZE, smallest_side if smallest_side % 2 else smallest_side - 1)
    block_size = max(3, block_size)

    # THRESH_BINARY 讓亮背景保持白色；再反相供輪廓搜尋。
    white_background = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        ADAPTIVE_C,
    )
    contour_map = cv2.bitwise_not(white_background)

    # 小型開運算削弱細格線，閉運算補齊被光暈切斷的圓邊。
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    contour_map = cv2.morphologyEx(contour_map, cv2.MORPH_OPEN, open_kernel)
    contour_map = cv2.morphologyEx(contour_map, cv2.MORPH_CLOSE, close_kernel)
    return contour_map


def _geometry_candidates(image: np.ndarray, contour_map: np.ndarray) -> List[CircleCandidate]:
    contours, _ = cv2.findContours(
        contour_map,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    image_area = float(image.shape[0] * image.shape[1])
    max_area = max(MIN_CIRCLE_AREA * 4.0, image_area * MAX_CIRCLE_AREA_RATIO)
    candidates: List[CircleCandidate] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < MIN_CIRCLE_AREA or area > max_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        if width < 7 or height < 7:
            continue
        aspect_ratio = width / float(max(1, height))
        if not 0.8 <= aspect_ratio <= 1.2:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0.0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            continue

        fill_ratio = area / float(max(1, width * height))
        if not 0.12 <= fill_ratio <= 0.95:
            continue

        candidates.append(
            CircleCandidate(
                x=int(x + width / 2),
                y=int(y + height / 2),
                width=int(width),
                height=int(height),
                area=area,
                circularity=circularity,
                fill_ratio=fill_ratio,
            )
        )
    return candidates


def _deduplicate_candidates(candidates: Sequence[CircleCandidate]) -> List[CircleCandidate]:
    """移除同一圓圈內外邊緣造成的重複輪廓。"""
    if not candidates:
        return []
    median_diameter = float(np.median([item.diameter for item in candidates]))
    center_threshold = max(3.0, median_diameter * 0.38)

    # 優先保留圓度較高且尺寸接近中位數的輪廓。
    ordered = sorted(
        candidates,
        key=lambda item: (
            item.circularity,
            -abs(item.diameter - median_diameter),
            item.area,
        ),
        reverse=True,
    )
    accepted: List[CircleCandidate] = []
    for item in ordered:
        if any(
            math.hypot(item.x - old.x, item.y - old.y) <= center_threshold
            for old in accepted
        ):
            continue
        accepted.append(item)
    return accepted


def _sort_big_road(candidates: Sequence[CircleCandidate]) -> List[CircleCandidate]:
    """依大路常見排列：由左到右分欄，每欄由上到下。"""
    if not candidates:
        return []
    median_diameter = float(np.median([item.diameter for item in candidates]))
    x_tolerance = max(4.0, median_diameter * COLUMN_TOLERANCE_RATIO)

    columns: List[List[CircleCandidate]] = []
    for item in sorted(candidates, key=lambda value: (value.x, value.y)):
        target_column: List[CircleCandidate] | None = None
        best_distance = float("inf")
        for column in columns:
            center_x = float(np.mean([member.x for member in column]))
            distance = abs(item.x - center_x)
            if distance <= x_tolerance and distance < best_distance:
                target_column = column
                best_distance = distance
        if target_column is None:
            columns.append([item])
        else:
            target_column.append(item)

    columns.sort(key=lambda column: float(np.mean([item.x for item in column])))
    ordered: List[CircleCandidate] = []
    for column in columns:
        ordered.extend(sorted(column, key=lambda item: (item.y, item.x)))
    return ordered


def analyze_baccarat_image_detailed(image_path: str | Path) -> Dict[str, Any]:
    """分析百家樂大路截圖並回傳序列與診斷資訊。

    回傳欄位：
    - sequence: 依座標排序後的 ["B", "P", ...]
    - candidates: 已成功判色的候選圓圈資訊
    - unknown_candidates: 找到幾何圓圈但中心顏色不明的數量
    - image_size: 分析時影像尺寸
    """
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"找不到圖片：{path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("無法讀取圖片，請確認格式為 JPG、PNG 或 WEBP。")

    image, resize_scale = _resize_for_analysis(image)
    contour_map = _preprocess_geometry(image)
    raw_candidates = _geometry_candidates(image, contour_map)
    unique_candidates = _deduplicate_candidates(raw_candidates)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    colored: List[CircleCandidate] = []
    unknown = 0
    for candidate in unique_candidates:
        hue, saturation, value = _center_patch_hsv(hsv, candidate.x, candidate.y)
        outcome = _classify_local_color(hue, saturation, value)
        if not outcome:
            unknown += 1
            continue
        colored.append(
            CircleCandidate(
                x=candidate.x,
                y=candidate.y,
                width=candidate.width,
                height=candidate.height,
                area=candidate.area,
                circularity=candidate.circularity,
                fill_ratio=candidate.fill_ratio,
                outcome=outcome,
                hue=round(hue, 3),
                saturation=round(saturation, 3),
                value=round(value, 3),
            )
        )

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
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "resize_scale": round(float(resize_scale), 6),
        },
        "candidates": [asdict(item) for item in ordered],
        "method": "adaptive_threshold_contour_center_5x5_hsv",
    }


def analyze_baccarat_image(image_path: str | Path) -> List[str]:
    """題目指定的簡化入口：只回傳辨識後的 B/P List。"""
    return list(analyze_baccarat_image_detailed(image_path)["sequence"])


__all__ = [
    "analyze_baccarat_image",
    "analyze_baccarat_image_detailed",
]
