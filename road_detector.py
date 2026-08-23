"""遊戲畫面大路偵測模組 V11.5（保留既有館別，新增兩種手機全畫面格式）。

重點：
1. DG 手機／電腦 ROI、滑動搜尋與固定格辨識流程完整保留 V11.2，不改動原有邏輯。
2. 新增 MT／DB 手機直式全畫面專用 ROI、附近滑動追焦與彩色圓環格位辨識。
3. 固定維持 6 列，MT／DB 允許非正方形格位；欄距與列距由彩色圓環中心自動估計。
4. 每個圓環分別統計紅、藍、綠 HSV 像素；雙色接近或偏離格位者標記 uncertain。
5. 依標準大路落點規則（含長龍右黏狀態）反推時間序列；失敗不把欄排序當正確答案。
6. 可用 ROAD_GRID_DEBUG=1 輸出追焦疊圖；版型不吻合時仍以品質閘門阻擋錯序列。
7. 新增 ofalive99 類 Android Chrome 直式全畫面候選；僅在畫面比例與底部白色大路特徵吻合時啟用。
8. 新增 Dream Gaming 緊湊手機版候選；獨立辨識中間大路，不將右側下三路混入。
"""
from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import math
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


# 一般館別備援區域。
ROAD_ROI = _env_roi("ROAD_ROI", (0.0, 0.58, 1.0, 0.42))

# MT 1728×903 範例中的實際大路區塊：x=1071, y=647, w=348, h=134。
# 使用比例座標後，畫面同比例縮放時仍可沿用。
MT_FIXED_ROAD_ROI = _env_roi(
    "MT_FIXED_ROAD_ROI",
    (0.619791667, 0.716500554, 0.201388889, 0.148394241),
)

# MT／DB iPhone Safari 直式完整桌廳：只截取右側大路本體，
# 不包含左側珠盤路、上方統計列、下三路與右側問路按鈕。
# 這兩組是新增候選，不取代 MT 電腦版與 DG 原有候選。
MT_MOBILE_BIG_ROAD_ROI = _env_roi(
    "MT_MOBILE_BIG_ROAD_ROI",
    (0.315, 0.733, 0.630, 0.080),
)
DB_MOBILE_BIG_ROAD_ROI = _env_roi(
    "DB_MOBILE_BIG_ROAD_ROI",
    (0.350, 0.775, 0.650, 0.086),
)

# 橫向「珠盤路＋大路＋下三路」裁圖中的右上第一區塊。
WIDE_TOP_ROAD_ROI = _env_roi(
    "WIDE_TOP_ROAD_ROI",
    (0.265, 0.00, 0.735, 0.64),
)

# DG 網頁版全畫面：只取「大路」本體，不包含左側珠盤與下方衍生路。
# 兩組比例分別以使用者提供的 iPhone 直式瀏覽器與 16:9 電腦版校正；
# 實際執行仍會在附近小範圍滑動搜尋，避免瀏覽器工具列與 UI 縮放造成位移。
DG_MOBILE_BIG_ROAD_ROI = _env_roi(
    "DG_MOBILE_BIG_ROAD_ROI",
    # 新版 iPhone Safari 完整桌廳：大路位於白色路紙右上區。
    # x 向左保留第一格，避免首顆莊被裁掉；執行時仍會在附近滑動搜尋。
    (0.302, 0.660, 0.653, 0.104),
)

# DG 942×2048 手機版新增兩種較低路紙位置。
# 這兩組只作為額外候選，原本 DG 手機／電腦 ROI 與辨識邏輯完全保留。
DG_MOBILE_LOWER_FULL_VIEW_ROI = _env_roi(
    "DG_MOBILE_LOWER_FULL_VIEW_ROI",
    (0.302, 0.700, 0.653, 0.104),
)
DG_MOBILE_LOWER_BROWSER_VIEW_ROI = _env_roi(
    "DG_MOBILE_LOWER_BROWSER_VIEW_ROI",
    (0.302, 0.720, 0.653, 0.104),
)

# Android Chrome（例如 ofalive99）直式全畫面：瀏覽器工具列與底部導覽列
# 會讓大路落在比既有 DG 手機版更低的位置。此設定是「新增候選」，
# 不覆寫 DG／MT／DB／其他館別的既有 ROI、HSV 門檻或品質閘門。
# 以 858×1907 範例換算：約為 x=259~819、y=1350~1598，只保留右側大路六列，
# 排除左側珠盤路、下三路與右側問路按鈕。
OFALIVE_ANDROID_PROFILE_ENABLED = (
    os.getenv("OFALIVE_ANDROID_PROFILE_ENABLED", "1").strip() == "1"
)
OFALIVE_ANDROID_BIG_ROAD_ROI = _env_roi(
    "OFALIVE_ANDROID_BIG_ROAD_ROI",
    (0.302, 0.708, 0.653, 0.130),
)
OFALIVE_ANDROID_SIGNATURE_ROI = _env_roi(
    "OFALIVE_ANDROID_SIGNATURE_ROI",
    (0.280, 0.675, 0.680, 0.195),
)
OFALIVE_ANDROID_PROFILE_SEARCH_Y = _env_float(
    "OFALIVE_ANDROID_PROFILE_SEARCH_Y", 0.030, 0.0, 0.12
)
OFALIVE_ANDROID_MIN_TALL_RATIO = _env_float(
    "OFALIVE_ANDROID_MIN_TALL_RATIO", 1.85, 1.20, 4.00
)
OFALIVE_ANDROID_MAX_TALL_RATIO = _env_float(
    "OFALIVE_ANDROID_MAX_TALL_RATIO", 2.65, 1.20, 4.00
)
OFALIVE_ANDROID_MIN_BRIGHT_FRACTION = _env_float(
    "OFALIVE_ANDROID_MIN_BRIGHT_FRACTION", 0.60, 0.10, 0.95
)

# Dream Gaming 緊湊手機版（例如 new-dd-cn.ahsy114.com）：珠盤路在左、中間為
# 六列大路、右側另有下三路。其大路寬度遠小於 Android Chrome 的 ofalive99 版型，
# 必須獨立取中間區塊，否則會把右側下三路誤當成同一張大路。
# 以 591×1280 範例換算：約為 x=161~427、y=928~1114；app.py 放大圖片後
# 仍以等比例座標辨識，不能依賴原始像素寬度。
DREAM_COMPACT_MOBILE_PROFILE_ENABLED = (
    os.getenv("DREAM_COMPACT_MOBILE_PROFILE_ENABLED", "1").strip() == "1"
)
DREAM_COMPACT_MOBILE_BIG_ROAD_ROI = _env_roi(
    "DREAM_COMPACT_MOBILE_BIG_ROAD_ROI",
    (0.272, 0.725, 0.450, 0.145),
)
DREAM_COMPACT_MOBILE_PROFILE_SEARCH_Y = _env_float(
    "DREAM_COMPACT_MOBILE_PROFILE_SEARCH_Y", 0.020, 0.0, 0.10
)
DREAM_COMPACT_MOBILE_MIN_WIDTH = _env_int(
    "DREAM_COMPACT_MOBILE_MIN_WIDTH", 320, 240, 1600
)
DREAM_COMPACT_MOBILE_MAX_WIDTH = _env_int(
    "DREAM_COMPACT_MOBILE_MAX_WIDTH", 1200, 240, 2400
)
DREAM_COMPACT_MOBILE_MIN_TALL_RATIO = _env_float(
    "DREAM_COMPACT_MOBILE_MIN_TALL_RATIO", 2.10, 1.20, 4.00
)
DREAM_COMPACT_MOBILE_MAX_TALL_RATIO = _env_float(
    "DREAM_COMPACT_MOBILE_MAX_TALL_RATIO", 2.19, 1.20, 4.00
)
DREAM_COMPACT_MOBILE_MIN_BRIGHT_FRACTION = _env_float(
    "DREAM_COMPACT_MOBILE_MIN_BRIGHT_FRACTION", 0.62, 0.10, 0.95
)

DG_DESKTOP_BIG_ROAD_ROI = _env_roi(
    "DG_DESKTOP_BIG_ROAD_ROI",
    (0.240, 0.803, 0.145, 0.135),
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

# 固定格內容自適應與顏色可信度。
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
    "ROAD_GRID_TIE_MIN_COMPONENT_RATIO", 0.30, 0.10, 1.0
)
ROAD_GRID_TIE_MAX_SPAN_RATIO = _env_float(
    "ROAD_GRID_TIE_MAX_SPAN_RATIO", 0.78, 0.20, 1.0
)
ROAD_GRID_DEBUG = os.getenv("ROAD_GRID_DEBUG", "0").strip() == "1"
ROAD_GRID_DEBUG_DIR = os.getenv("ROAD_GRID_DEBUG_DIR", "/tmp/bgs_road_debug").strip()
ROAD_CROP_MIN_ASPECT = _env_float("ROAD_CROP_MIN_ASPECT", 2.05, 1.2, 4.0)
ROAD_GRID_AUTO_COLUMNS = os.getenv("ROAD_GRID_AUTO_COLUMNS", "1").strip() == "1"
ROAD_GRID_AUTO_COL_MIN = _env_int("ROAD_GRID_AUTO_COL_MIN", 8, 5, 60)
ROAD_GRID_AUTO_COL_MAX = _env_int("ROAD_GRID_AUTO_COL_MAX", 32, 8, 60)
ROAD_GRID_AUTO_COL_RADIUS = _env_int("ROAD_GRID_AUTO_COL_RADIUS", 3, 1, 8)
ROAD_GRID_MIN_COMPONENT_AREA_RATIO = _env_float(
    "ROAD_GRID_MIN_COMPONENT_AREA_RATIO", 0.018, 0.002, 0.20
)
ROAD_GRID_MIN_COMPONENT_SPAN_RATIO = _env_float(
    "ROAD_GRID_MIN_COMPONENT_SPAN_RATIO", 0.18, 0.05, 0.70
)
ROAD_GRID_TIE_PIXELS_PER_MARK_RATIO = _env_float(
    "ROAD_GRID_TIE_PIXELS_PER_MARK_RATIO", 0.075, 0.025, 0.25
)
ROAD_GRID_TIE_MAX_COUNT = _env_int("ROAD_GRID_TIE_MAX_COUNT", 4, 1, 9)
ROAD_PROFILE_SEARCH_X = _env_float("ROAD_PROFILE_SEARCH_X", 0.012, 0.0, 0.08)
ROAD_PROFILE_SEARCH_Y_MOBILE = _env_float(
    "ROAD_PROFILE_SEARCH_Y_MOBILE", 0.035, 0.0, 0.12
)
ROAD_PROFILE_SEARCH_Y_DESKTOP = _env_float(
    "ROAD_PROFILE_SEARCH_Y_DESKTOP", 0.020, 0.0, 0.08
)
ROAD_PROFILE_SEARCH_STEPS = _env_int("ROAD_PROFILE_SEARCH_STEPS", 5, 1, 9)
ROAD_CROP_BRIGHT_FRACTION = _env_float(
    "ROAD_CROP_BRIGHT_FRACTION", 0.45, 0.10, 0.95
)


# MT／DB 手機大路的圓環追焦參數。DG 不會進入此分支。
MT_PROFILE_SEARCH_Y_MOBILE = _env_float(
    "MT_PROFILE_SEARCH_Y_MOBILE", 0.018, 0.0, 0.08
)
DB_PROFILE_SEARCH_Y_MOBILE = _env_float(
    "DB_PROFILE_SEARCH_Y_MOBILE", 0.018, 0.0, 0.08
)
MOBILE_RING_HOUGH_PARAM1 = _env_float(
    "MOBILE_RING_HOUGH_PARAM1", 80.0, 20.0, 240.0
)
MT_MOBILE_RING_HOUGH_PARAM2 = _env_float(
    "MT_MOBILE_RING_HOUGH_PARAM2", 14.0, 4.0, 40.0
)
DB_MOBILE_RING_HOUGH_PARAM2 = _env_float(
    "DB_MOBILE_RING_HOUGH_PARAM2", 12.0, 4.0, 40.0
)
MOBILE_RING_MIN_COLOR_PIXELS = _env_int(
    "MOBILE_RING_MIN_COLOR_PIXELS", 18, 5, 300
)
MOBILE_RING_COLOR_DOMINANCE = _env_float(
    "MOBILE_RING_COLOR_DOMINANCE", 1.35, 1.05, 5.0
)
MOBILE_RING_MAX_UNCERTAIN_RATIO = _env_float(
    "MOBILE_RING_MAX_UNCERTAIN_RATIO", 0.20, 0.0, 0.80
)
MOBILE_RING_MAX_MEDIAN_FIT_ERROR = _env_float(
    "MOBILE_RING_MAX_MEDIAN_FIT_ERROR", 0.18, 0.05, 0.45
)
MOBILE_RING_MAX_SINGLE_FIT_ERROR = _env_float(
    "MOBILE_RING_MAX_SINGLE_FIT_ERROR", 0.30, 0.10, 0.60
)

WIDE_LAYOUT_MIN_ASPECT = max(
    3.2,
    float(os.getenv("WIDE_LAYOUT_MIN_ASPECT", "4.0") or "4.0"),
)
ROAD_AUTO_FULL_FALLBACK = os.getenv("ROAD_AUTO_FULL_FALLBACK", "1").strip() == "1"
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


def _effective_grid_bounds(
    crop: np.ndarray,
    union_mask: np.ndarray,
    *,
    grid_columns: int,
    grid_rows: int = ROAD_GRID_ROWS,
) -> Dict[str, Any]:
    height, width = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edge_x = np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)), axis=0)
    edge_y = np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)), axis=1)
    ys, xs = np.nonzero(union_mask)
    x_alignment = _axis_alignment(xs, edge_x, width, grid_columns)
    y_alignment = _axis_alignment(ys, edge_y, height, grid_rows)
    x1 = int(round(x_alignment["start"]))
    x2 = int(round(x_alignment["end"]))
    y1 = int(round(y_alignment["start"]))
    y2 = int(round(y_alignment["end"]))
    x1 = max(0, min(width - 1, x1))
    x2 = max(x1 + 1, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(y1 + 1, min(height, y2))
    pitch_x = (x2 - x1) / max(1.0, float(grid_columns))
    pitch_y = (y2 - y1) / max(1.0, float(grid_rows))
    square_score = max(0.0, 1.0 - abs(pitch_x - pitch_y) / max(1.0, pitch_x, pitch_y))
    score = float(0.44 * x_alignment["score"] + 0.44 * y_alignment["score"] + 0.12 * square_score)
    coverage = float((x_alignment["coverage"] + y_alignment["coverage"]) / 2.0)
    return {
        "x": x1,
        "y": y1,
        "width": x2 - x1,
        "height": y2 - y1,
        "score": score,
        "coverage": coverage,
        "square_cell_score": float(square_score),
        "cell_pitch_x": float(pitch_x),
        "cell_pitch_y": float(pitch_y),
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


def _largest_component_stats(mask: np.ndarray) -> Tuple[int, float, float]:
    if mask.size == 0 or int(mask.sum()) <= 0:
        return 0, 0.0, 0.0
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return 0, 0.0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    index = int(np.argmax(areas)) + 1
    largest = int(stats[index, cv2.CC_STAT_AREA])
    width = int(stats[index, cv2.CC_STAT_WIDTH])
    height = int(stats[index, cv2.CC_STAT_HEIGHT])
    return largest, width / max(1, mask.shape[1]), height / max(1, mask.shape[0])


def _classify_grid(
    crop: np.ndarray,
    red_mask: np.ndarray,
    blue_mask: np.ndarray,
    green_mask: np.ndarray,
    bounds: Mapping[str, Any],
    *,
    grid_columns: int,
    grid_rows: int = ROAD_GRID_ROWS,
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

    for row in range(grid_rows):
        local_y1, local_y2 = _grid_bounds(grid_height, row, grid_rows)
        y1, y2 = grid_y + local_y1, grid_y + local_y2
        for column in range(grid_columns):
            local_x1, local_x2 = _grid_bounds(grid_width, column, grid_columns)
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
            inner_width = max(1, inner_x2 - inner_x1)
            inner_height = max(1, inner_y2 - inner_y1)
            inner_area = max(1, inner_width * inner_height)
            red_pixels = _rect_sum(red_integral, inner_x1, inner_y1, inner_x2, inner_y2)
            blue_pixels = _rect_sum(blue_integral, inner_x1, inner_y1, inner_x2, inner_y2)
            green_pixels = _rect_sum(green_integral, inner_x1, inner_y1, inner_x2, inner_y2)
            red_component, red_span_x, red_span_y = _largest_component_stats(
                red_mask[inner_y1:inner_y2, inner_x1:inner_x2]
            )
            blue_component, blue_span_x, blue_span_y = _largest_component_stats(
                blue_mask[inner_y1:inner_y2, inner_x1:inner_x2]
            )
            minimum_pixels = max(
                ROAD_GRID_MIN_COLOR_PIXELS,
                int(round(inner_area * ROAD_GRID_MIN_COLOR_RATIO)),
            )
            minimum_component = max(
                4, int(round(inner_area * ROAD_GRID_MIN_COMPONENT_AREA_RATIO))
            )
            red_shape_ok = bool(
                red_component >= minimum_component
                and min(red_span_x, red_span_y) >= ROAD_GRID_MIN_COMPONENT_SPAN_RATIO
            )
            blue_shape_ok = bool(
                blue_component >= minimum_component
                and min(blue_span_x, blue_span_y) >= ROAD_GRID_MIN_COMPONENT_SPAN_RATIO
            )
            qualified_red = red_pixels if red_shape_ok else 0
            qualified_blue = blue_pixels if blue_shape_ok else 0
            dominant_pixels = max(qualified_red, qualified_blue)
            secondary_pixels = min(qualified_red, qualified_blue)
            dominance = (dominant_pixels + 1.0) / (secondary_pixels + 1.0)
            outcome = ""
            is_uncertain = False
            if dominant_pixels >= minimum_pixels:
                if dominance >= ROAD_GRID_COLOR_DOMINANCE:
                    outcome = "B" if qualified_red > qualified_blue else "P"
                else:
                    is_uncertain = True
            elif max(red_pixels, blue_pixels) >= minimum_pixels and (red_shape_ok or blue_shape_ok):
                is_uncertain = True

            largest_green, green_concentration, green_span_x, green_span_y = _green_component_stats(
                green_mask[inner_y1:inner_y2, inner_x1:inner_x2]
            )
            tie_minimum = max(
                ROAD_GRID_TIE_MIN_PIXELS,
                int(round(inner_area * ROAD_GRID_TIE_MIN_AREA_RATIO)),
            )
            green_area_ratio = green_pixels / max(1.0, float(inner_area))
            tie_confident = bool(
                outcome
                and green_pixels >= tie_minimum
                and largest_green >= max(3, int(round(tie_minimum * 0.35)))
                and green_concentration >= ROAD_GRID_TIE_MIN_COMPONENT_RATIO
                and max(green_span_x, green_span_y) <= ROAD_GRID_TIE_MAX_SPAN_RATIO
            )
            tie_count = 1 if tie_confident else 0
            separation = max(
                0.0,
                min(
                    1.0,
                    (dominant_pixels - secondary_pixels)
                    / max(1.0, dominant_pixels + secondary_pixels),
                ),
            )
            pixel_strength = max(
                0.0,
                min(1.0, dominant_pixels / max(1.0, minimum_pixels * 2.0)),
            )
            shape_strength = max(
                min(red_span_x, red_span_y) if outcome == "B" else 0.0,
                min(blue_span_x, blue_span_y) if outcome == "P" else 0.0,
            )
            confidence = (
                0.45 * pixel_strength + 0.35 * separation + 0.20 * min(1.0, shape_strength / 0.45)
                if outcome
                else 0.0
            )
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
                "inner_width": inner_width,
                "inner_height": inner_height,
                "cx": round((x1 + x2) / 2.0, 2),
                "cy": round((y1 + y2) / 2.0, 2),
                "red_pixels": int(red_pixels),
                "blue_pixels": int(blue_pixels),
                "green_pixels": int(green_pixels),
                "red_largest_component": int(red_component),
                "blue_largest_component": int(blue_component),
                "red_component_span_x": round(float(red_span_x), 6),
                "red_component_span_y": round(float(red_span_y), 6),
                "blue_component_span_x": round(float(blue_span_x), 6),
                "blue_component_span_y": round(float(blue_span_y), 6),
                "minimum_color_pixels": int(minimum_pixels),
                "minimum_component_pixels": int(minimum_component),
                "dominance": round(float(dominance), 6),
                "green_largest_component": int(largest_green),
                "green_component_ratio": round(float(green_concentration), 6),
                "green_area_ratio": round(float(green_area_ratio), 6),
                "green_span_x_ratio": round(float(green_span_x), 6),
                "green_span_y_ratio": round(float(green_span_y), 6),
                "tie_count": int(tie_count),
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
    grid_rows: int = ROAD_GRID_ROWS,
) -> Any:
    """依六列大路規則反推時間序；長龍轉右後會保持右黏，不再錯誤往下。"""
    grid: Dict[Tuple[int, int], str] = {
        (int(item.get("column", 0)), int(item.get("row", 0))): str(item.get("outcome") or "").upper()
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
        if not tailing_right and row < grid_rows - 1 and (column, row + 1) not in visited:
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
        fallback_reason = f"incomplete_big_road_reconstruction_{len(best_partial)}_of_{target_count}"
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
    *,
    grid_columns: int,
    grid_rows: int = ROAD_GRID_ROWS,
    profile: str = "",
) -> str:
    if not ROAD_GRID_DEBUG:
        return ""
    overlay = crop.copy()
    x, y = int(grid_bounds["x"]), int(grid_bounds["y"])
    width, height = int(grid_bounds["width"]), int(grid_bounds["height"])
    cv2.rectangle(overlay, (x, y), (x + width - 1, y + height - 1), (255, 255, 255), 1)
    for column in range(grid_columns + 1):
        line_x = int(round(x + column * width / grid_columns))
        cv2.line(overlay, (line_x, y), (line_x, y + height), (160, 160, 160), 1)
    for row in range(grid_rows + 1):
        line_y = int(round(y + row * height / grid_rows))
        cv2.line(overlay, (x, line_y), (x + width, line_y), (160, 160, 160), 1)
    for cell in all_cells:
        label = str(cell.get("outcome") or "")
        if bool(cell.get("uncertain")):
            label = "?"
        tie_count = int(cell.get("tie_count", 0) or 0)
        if tie_count > 0:
            label += f"T{tie_count}"
        if not label:
            continue
        cx, cy = int(round(float(cell.get("cx", 0)))), int(round(float(cell.get("cy", 0))))
        cv2.putText(overlay, label, (max(0, cx - 8), max(10, cy + 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
    status = "OK" if quality_ok else f"RETAKE:{fallback_reason or 'quality'}"
    header = f"{status} cols={grid_columns} {profile}".strip()
    cv2.putText(overlay, header[:100], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    directory = Path(ROAD_GRID_DEBUG_DIR or "/tmp/bgs_road_debug")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"road_grid_{time.time_ns()}.png"
    cv2.imwrite(str(path), overlay)
    return str(path)


def _column_candidates(crop: np.ndarray, requested: Optional[int] = None) -> List[int]:
    if requested is not None:
        return [max(5, min(60, int(requested)))]
    height, width = crop.shape[:2]
    estimated = int(round((width / max(1.0, float(height))) * ROAD_GRID_ROWS))
    estimated = max(ROAD_GRID_AUTO_COL_MIN, min(ROAD_GRID_AUTO_COL_MAX, estimated))
    if not ROAD_GRID_AUTO_COLUMNS:
        return [ROAD_GRID_COLS]
    values = {ROAD_GRID_COLS, estimated}
    for delta in range(-ROAD_GRID_AUTO_COL_RADIUS, ROAD_GRID_AUTO_COL_RADIUS + 1):
        values.add(estimated + delta)
    return sorted(
        value for value in values
        if ROAD_GRID_AUTO_COL_MIN <= value <= ROAD_GRID_AUTO_COL_MAX
    )


def _detect_fixed_grid_for_columns(
    crop: np.ndarray,
    grid_columns: int,
    *,
    profile: str = "",
) -> Dict[str, Any]:
    image_height, image_width = crop.shape[:2]
    red_mask, blue_mask, green_mask, union_mask = _color_masks(crop)
    grid_bounds = _effective_grid_bounds(
        crop, union_mask, grid_columns=grid_columns, grid_rows=ROAD_GRID_ROWS
    )
    classified = _classify_grid(
        crop, red_mask, blue_mask, green_mask, grid_bounds,
        grid_columns=grid_columns, grid_rows=ROAD_GRID_ROWS,
    )
    cells = list(classified["cells"])
    uncertain_cells = list(classified["uncertain_cells"])
    all_grid_cells = list(classified["all_grid_cells"])
    reconstruction = _reconstruct_big_road_order(
        cells, return_details=True, grid_rows=ROAD_GRID_ROWS
    )
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
        and float(grid_bounds.get("square_cell_score", 0.0)) >= 0.72
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

    pitch_x = float(grid_bounds.get("cell_pitch_x", 0.0) or 0.0)
    pitch_y = float(grid_bounds.get("cell_pitch_y", 0.0) or 0.0)
    geometry_score = float(grid_bounds.get("square_cell_score", 0.0) or 0.0)
    candidate_score = (
        (1000.0 if quality_ok else 0.0)
        + (300.0 if reconstruction["reconstructed_all"] else 0.0)
        + float(grid_bounds["score"]) * 120.0
        + float(grid_bounds["coverage"]) * 45.0
        + geometry_score * 100.0
        + recognized_count * 2.0
        - uncertain_count * 8.0
        - abs(pitch_x - pitch_y) * 2.0
    )
    debug_overlay_path = _debug_overlay(
        crop, all_grid_cells, grid_bounds, quality_ok, fallback_reason,
        grid_columns=grid_columns, grid_rows=ROAD_GRID_ROWS, profile=profile,
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
        "method": "fixed_hsv_grid_6xN_adaptive_v11_0",
        "grid_rows": ROAD_GRID_ROWS,
        "grid_columns": int(grid_columns),
        "grid_size": {"width": image_width, "height": image_height},
        "effective_grid": {
            key: grid_bounds[key]
            for key in (
                "x", "y", "width", "height", "score", "coverage",
                "square_cell_score", "cell_pitch_x", "cell_pitch_y",
                "offset_x", "offset_y", "scale_x", "scale_y", "gain_x", "gain_y"
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
        "layout_profile": profile,
        "column_candidate_score": round(candidate_score, 6),
    }


def _detect_fixed_grid(
    crop: np.ndarray,
    *,
    grid_columns: Optional[int] = None,
    profile: str = "",
) -> Dict[str, Any]:
    """固定六列、欄數自動；逐一驗證格線幾何與完整大路反推後選最佳候選。"""
    if crop is None or crop.size == 0:
        raise ValueError("固定大路裁圖為空。")
    results = [
        _detect_fixed_grid_for_columns(crop, columns, profile=profile)
        for columns in _column_candidates(crop, grid_columns)
    ]
    best = max(
        results,
        key=lambda item: (
            float(item.get("column_candidate_score", -9999.0)),
            float(dict(item.get("effective_grid") or {}).get("score", 0.0)),
            -abs(
                int(item.get("grid_columns", ROAD_GRID_COLS))
                - int(round((crop.shape[1] / max(1.0, float(crop.shape[0]))) * ROAD_GRID_ROWS))
            ),
        ),
    )
    output = dict(best)
    output["column_candidates"] = [
        {
            "grid_columns": int(item.get("grid_columns", 0) or 0),
            "quality_ok": bool(item.get("quality_ok")),
            "recognized_count": int(item.get("recognized_count", 0) or 0),
            "reconstructed_all": bool(item.get("reconstructed_all")),
            "alignment_score": float(dict(item.get("effective_grid") or {}).get("score", 0.0) or 0.0),
            "square_cell_score": float(dict(item.get("effective_grid") or {}).get("square_cell_score", 0.0) or 0.0),
            "score": float(item.get("column_candidate_score", -9999.0) or -9999.0),
            "fallback_reason": str(item.get("fallback_reason") or ""),
        }
        for item in results
    ]
    return output



def _median_float(values: Sequence[float], default: float) -> float:
    if not values:
        return float(default)
    return float(np.median(np.asarray(list(values), dtype=np.float64)))


def _nearest_ring_pitch(
    items: Sequence[Mapping[str, Any]],
    *,
    axis: str,
    expected: float,
) -> float:
    """由每顆圓環的最近同列／同欄鄰居估計實際欄距或列距。

    MT 手機版格位約為寬 30、高 22 像素，不能沿用 DG 的近正方形假設；
    DB 手機版則約為寬 22、高 20 像素。因此 x/y pitch 必須分開估計。
    """
    nearest: List[float] = []
    for index, first in enumerate(items):
        best: Optional[float] = None
        for other_index, second in enumerate(items):
            if index == other_index:
                continue
            dx = abs(float(first.get("cx", 0.0)) - float(second.get("cx", 0.0)))
            dy = abs(float(first.get("cy", 0.0)) - float(second.get("cy", 0.0)))
            if axis == "x":
                if dy > max(4.0, expected * 0.30):
                    continue
                distance = dx
                minimum = max(6.0, expected * 0.55)
                maximum = expected * 2.20
            else:
                if dx > max(5.0, expected * 0.35):
                    continue
                distance = dy
                minimum = max(6.0, expected * 0.50)
                maximum = expected * 1.65
            if minimum <= distance <= maximum and (
                best is None or distance < best
            ):
                best = distance
        if best is not None:
            nearest.append(float(best))

    if not nearest:
        return float(expected)
    median = _median_float(nearest, expected)
    trimmed = [
        value
        for value in nearest
        if abs(value - median) <= max(2.5, median * 0.18)
    ]
    return _median_float(trimmed, median)


def _debug_ring_overlay(
    crop: np.ndarray,
    cells: Sequence[Mapping[str, Any]],
    uncertain_cells: Sequence[Mapping[str, Any]],
    *,
    quality_ok: bool,
    fallback_reason: str,
    pitch_x: float,
    pitch_y: float,
    origin_x: float,
    origin_y: float,
    profile: str,
) -> str:
    if not ROAD_GRID_DEBUG:
        return ""
    overlay = crop.copy()
    for item in list(cells) + list(uncertain_cells):
        cx = int(round(float(item.get("cx", 0.0))))
        cy = int(round(float(item.get("cy", 0.0))))
        radius = max(3, int(item.get("radius", 6) or 6))
        uncertain = bool(item.get("uncertain"))
        outcome = str(item.get("outcome") or "?")
        color = (
            (0, 255, 255)
            if uncertain
            else (0, 0, 255)
            if outcome == "B"
            else (255, 0, 0)
        )
        cv2.circle(overlay, (cx, cy), radius, color, 1, cv2.LINE_AA)
        label = "?" if uncertain else outcome
        tie_count = int(item.get("tie_count", 0) or 0)
        if tie_count > 0:
            label += f"T{tie_count}"
        cv2.putText(
            overlay,
            label,
            (max(0, cx - 7), min(overlay.shape[0] - 2, cy + 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    max_column = max(
        [int(item.get("column", 0) or 0) for item in cells],
        default=0,
    )
    for column in range(max_column + 2):
        x = int(round(origin_x - pitch_x / 2.0 + column * pitch_x))
        cv2.line(overlay, (x, 0), (x, overlay.shape[0] - 1), (120, 120, 120), 1)
    for row in range(ROAD_GRID_ROWS + 1):
        y = int(round(origin_y - pitch_y / 2.0 + row * pitch_y))
        cv2.line(overlay, (0, y), (overlay.shape[1] - 1, y), (120, 120, 120), 1)

    status = "OK" if quality_ok else f"RETAKE:{fallback_reason or 'quality'}"
    cv2.putText(
        overlay,
        f"{status} px={pitch_x:.1f} py={pitch_y:.1f} {profile}"[:110],
        (4, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    directory = Path(ROAD_GRID_DEBUG_DIR or "/tmp/bgs_road_debug")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"road_ring_{time.time_ns()}.png"
    cv2.imwrite(str(path), overlay)
    return str(path)


def _detect_mobile_ring_grid(
    crop: np.ndarray,
    *,
    profile: str,
) -> Dict[str, Any]:
    """MT／DB 手機全畫面專用彩色圓環大路偵測。

    只由新增的 MT／DB 手機候選呼叫。DG 仍完整使用原本的
    ``_detect_fixed_grid``，不會進入本函式。
    """
    if crop is None or crop.size == 0:
        raise ValueError("MT/DB 手機大路裁圖為空。")

    image_height, image_width = crop.shape[:2]
    expected_pitch_y = image_height / max(1.0, float(ROAD_GRID_ROWS))
    minimum_radius = max(3, int(round(expected_pitch_y * 0.24)))
    maximum_radius = max(minimum_radius + 2, int(round(expected_pitch_y * 0.68)))
    profile_key = str(profile or "").lower()
    hough_param2 = (
        MT_MOBILE_RING_HOUGH_PARAM2
        if profile_key.startswith("mt_")
        else DB_MOBILE_RING_HOUGH_PARAM2
    )

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=max(7.0, expected_pitch_y * 0.55),
        param1=MOBILE_RING_HOUGH_PARAM1,
        param2=hough_param2,
        minRadius=minimum_radius,
        maxRadius=maximum_radius,
    )

    red_mask, blue_mask, green_mask, _ = _color_masks(crop)
    yy, xx = np.ogrid[:image_height, :image_width]
    colored: List[Dict[str, Any]] = []
    uncertain_cells: List[Dict[str, Any]] = []

    raw_circles = [] if circles is None else np.round(circles[0]).astype(int)
    for raw_circle in raw_circles:
        cx, cy, radius = [int(value) for value in raw_circle]
        if not (0 <= cx < image_width and 0 <= cy < image_height):
            continue
        squared_distance = (xx - cx) ** 2 + (yy - cy) ** 2
        annulus = (
            (squared_distance <= float(radius + 2) ** 2)
            & (squared_distance >= float(max(1, radius - 3)) ** 2)
        )
        disk = squared_distance <= float(radius + 2) ** 2
        red_pixels = int(red_mask[annulus].sum())
        blue_pixels = int(blue_mask[annulus].sum())
        green_pixels = int(green_mask[disk].sum())
        dominant = max(red_pixels, blue_pixels)
        secondary = min(red_pixels, blue_pixels)
        dominance = dominant / max(1.0, float(secondary))
        minimum_color = max(
            MOBILE_RING_MIN_COLOR_PIXELS,
            int(round(math.pi * radius * radius * 0.12)),
        )
        base = {
            "cx": float(cx),
            "cy": float(cy),
            "radius": int(radius),
            "red_pixels": red_pixels,
            "blue_pixels": blue_pixels,
            "green_pixels": green_pixels,
            "color_dominance": round(dominance, 6),
        }
        if dominant < minimum_color or dominance < MOBILE_RING_COLOR_DOMINANCE:
            uncertain_cells.append({**base, "uncertain": True})
            continue

        outcome = "B" if red_pixels > blue_pixels else "P"
        colored_fraction = dominant / max(1.0, math.pi * float(radius + 2) ** 2)
        confidence = min(
            1.0,
            0.55 * colored_fraction
            + 0.45 * min(1.0, (dominance - 1.0) / 3.0),
        )
        tie_threshold = max(8, int(round(math.pi * radius * radius * 0.035)))
        colored.append(
            {
                **base,
                "outcome": outcome,
                "tie_count": 1 if green_pixels >= tie_threshold else 0,
                "confidence": round(confidence, 6),
                "uncertain": False,
            }
        )

    if not colored:
        return {
            "ok": False,
            "quality_ok": False,
            "sequence": [],
            "raw_outcomes": [],
            "recognized_count": 0,
            "unknown_candidates": len(uncertain_cells),
            "uncertain_count": len(uncertain_cells),
            "raw_contours": len(raw_circles),
            "method": "fixed_ring_grid_6xN_mobile_v11_3",
            "reconstructed_all": False,
            "fallback_reason": "no_colored_big_road_rings",
            "layout_profile": profile,
        }

    pitch_y = _nearest_ring_pitch(
        colored, axis="y", expected=expected_pitch_y
    )
    pitch_x = _nearest_ring_pitch(
        colored, axis="x", expected=expected_pitch_y
    )
    origin_x = min(float(item["cx"]) for item in colored)
    origin_y = min(float(item["cy"]) for item in colored)

    mapped: List[Dict[str, Any]] = []
    for source in colored:
        column = int(round((float(source["cx"]) - origin_x) / max(1.0, pitch_x)))
        row = int(round((float(source["cy"]) - origin_y) / max(1.0, pitch_y)))
        expected_x = origin_x + column * pitch_x
        expected_y = origin_y + row * pitch_y
        fit_x = abs(float(source["cx"]) - expected_x) / max(1.0, pitch_x)
        fit_y = abs(float(source["cy"]) - expected_y) / max(1.0, pitch_y)
        item = {
            **source,
            "column": column,
            "row": row,
            "fit_error_x": round(fit_x, 6),
            "fit_error_y": round(fit_y, 6),
        }
        if (
            column < 0
            or row < 0
            or row >= ROAD_GRID_ROWS
            or fit_x > MOBILE_RING_MAX_SINGLE_FIT_ERROR
            or fit_y > MOBILE_RING_MAX_SINGLE_FIT_ERROR
        ):
            item["uncertain"] = True
            uncertain_cells.append(item)
        else:
            mapped.append(item)

    # Hough 偶爾會在同一圓環產生兩個候選；同格只保留顏色信心較高者。
    by_cell: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for item in mapped:
        key = (int(item["column"]), int(item["row"]))
        score = float(item.get("confidence", 0.0)) + 0.01 * float(
            item.get("radius", 0) or 0
        )
        previous = by_cell.get(key)
        previous_score = float(previous.get("_dedup_score", -1.0)) if previous else -1.0
        if previous is None or score > previous_score:
            if previous is not None:
                rejected = dict(previous)
                rejected.pop("_dedup_score", None)
                rejected["uncertain"] = True
                uncertain_cells.append(rejected)
            by_cell[key] = {**item, "_dedup_score": score}
        else:
            rejected = dict(item)
            rejected["uncertain"] = True
            uncertain_cells.append(rejected)

    cells: List[Dict[str, Any]] = []
    for value in by_cell.values():
        item = dict(value)
        item.pop("_dedup_score", None)
        cells.append(item)

    reconstruction = _reconstruct_big_road_order(
        cells,
        return_details=True,
        grid_rows=ROAD_GRID_ROWS,
    )
    cell_lookup = {
        (int(item["column"]), int(item["row"])): item
        for item in cells
    }
    ordered_cells: List[Dict[str, Any]] = []
    sequence: List[str] = []
    raw_outcomes: List[str] = []
    tie_markers: Dict[str, int] = {}
    if bool(reconstruction.get("reconstructed_all")):
        for index, position in enumerate(list(reconstruction.get("positions") or [])):
            source = dict(cell_lookup[position])
            source["index"] = index
            source["chronology_confirmed"] = True
            ordered_cells.append(source)
            outcome = str(source.get("outcome") or "")
            sequence.append(outcome)
            raw_outcomes.append(outcome)
            tie_count = int(source.get("tie_count", 0) or 0)
            if tie_count > 0:
                tie_markers[str(index)] = tie_count
                raw_outcomes.extend(["T"] * tie_count)
    else:
        for source in sorted(
            cells,
            key=lambda item: (int(item["column"]), int(item["row"])),
        ):
            item = dict(source)
            item["chronology_confirmed"] = False
            ordered_cells.append(item)

    fit_errors = [
        max(
            float(item.get("fit_error_x", 1.0) or 0.0),
            float(item.get("fit_error_y", 1.0) or 0.0),
        )
        for item in cells
    ]
    median_fit_error = _median_float(fit_errors, 1.0)
    maximum_fit_error = max(fit_errors or [1.0])
    recognized_count = len(cells)
    uncertain_count = len(uncertain_cells)
    uncertain_ratio = uncertain_count / max(1, recognized_count + uncertain_count)
    geometry_score = max(
        0.0,
        min(1.0, 1.0 - median_fit_error / max(1e-9, MOBILE_RING_MAX_SINGLE_FIT_ERROR)),
    )
    reconstruction_ok = bool(reconstruction.get("reconstructed_all"))
    quality_ok = bool(
        recognized_count >= ROAD_GRID_MIN_RECOGNIZED
        and reconstruction_ok
        and uncertain_ratio <= MOBILE_RING_MAX_UNCERTAIN_RATIO
        and median_fit_error <= MOBILE_RING_MAX_MEDIAN_FIT_ERROR
    )

    fallback_reason = str(reconstruction.get("fallback_reason") or "")
    if not fallback_reason and uncertain_ratio > MOBILE_RING_MAX_UNCERTAIN_RATIO:
        fallback_reason = "too_many_uncertain_mobile_rings"
    if not fallback_reason and median_fit_error > MOBILE_RING_MAX_MEDIAN_FIT_ERROR:
        fallback_reason = "mobile_ring_grid_geometry_not_confident"
    if not fallback_reason and recognized_count < ROAD_GRID_MIN_RECOGNIZED:
        fallback_reason = "recognized_count_below_minimum"

    maximum_column = max(
        [int(item.get("column", 0) or 0) for item in cells],
        default=-1,
    )
    grid_columns = maximum_column + 1
    grid_x = max(0, int(round(origin_x - pitch_x / 2.0)))
    grid_y = max(0, int(round(origin_y - pitch_y / 2.0)))
    grid_width = min(
        image_width - grid_x,
        max(1, int(round(max(1, grid_columns) * pitch_x))),
    )
    grid_height = min(
        image_height - grid_y,
        max(1, int(round(ROAD_GRID_ROWS * pitch_y))),
    )
    effective_grid = {
        "x": grid_x,
        "y": grid_y,
        "width": grid_width,
        "height": grid_height,
        "score": round(geometry_score, 6),
        "coverage": round(
            recognized_count / max(1, ROAD_GRID_ROWS * max(1, grid_columns)),
            6,
        ),
        "square_cell_score": round(min(pitch_x, pitch_y) / max(pitch_x, pitch_y), 6),
        "cell_pitch_x": round(pitch_x, 6),
        "cell_pitch_y": round(pitch_y, 6),
        "offset_x": round(origin_x, 6),
        "offset_y": round(origin_y, 6),
        "scale_x": 1.0,
        "scale_y": 1.0,
        "gain_x": 0.0,
        "gain_y": 0.0,
    }
    debug_overlay_path = _debug_ring_overlay(
        crop,
        cells,
        uncertain_cells,
        quality_ok=quality_ok,
        fallback_reason=fallback_reason,
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        origin_x=origin_x,
        origin_y=origin_y,
        profile=profile,
    )

    return {
        "ok": bool(sequence) and quality_ok,
        "quality_ok": quality_ok,
        "sequence": sequence,
        "raw_outcomes": raw_outcomes,
        "tie_markers": tie_markers,
        "grid_cells": ordered_cells,
        "all_grid_cells": cells + uncertain_cells,
        "recognized_count": recognized_count,
        "sequence_count": len(sequence),
        "confirmed_round_count": len(raw_outcomes),
        "uncertain_count": uncertain_count,
        "unknown_candidates": uncertain_count,
        "unknown_ratio": round(uncertain_ratio, 6),
        "raw_contours": len(raw_circles),
        "candidates": ordered_cells,
        "method": "fixed_ring_grid_6xN_mobile_v11_3",
        "grid_rows": ROAD_GRID_ROWS,
        "grid_columns": grid_columns,
        "grid_size": {"width": image_width, "height": image_height},
        "effective_grid": effective_grid,
        "grid_alignment": effective_grid,
        "alignment_ok": bool(quality_ok),
        "median_cell_confidence": round(
            _median_float(
                [float(item.get("confidence", 0.0) or 0.0) for item in cells],
                0.0,
            ),
            6,
        ),
        "reconstructed_all": reconstruction_ok,
        "reconstruction_solution_count": int(
            reconstruction.get("solution_count", 0) or 0
        ),
        "partial_reconstruction": list(
            reconstruction.get("partial_positions") or []
        ),
        "fallback_preview_positions": list(
            reconstruction.get("fallback_preview") or []
        ),
        "fallback_reason": fallback_reason,
        "uncertain_cells": uncertain_cells,
        "count_is_confirmed": bool(quality_ok and uncertain_count == 0),
        "debug_overlay_path": debug_overlay_path,
        "debug_enabled": ROAD_GRID_DEBUG,
        "layout_profile": profile,
        "ring_pitch_x": round(pitch_x, 6),
        "ring_pitch_y": round(pitch_y, 6),
        "ring_origin": {"x": round(origin_x, 6), "y": round(origin_y, 6)},
        "ring_fit_median_error": round(median_fit_error, 6),
        "ring_fit_max_error": round(maximum_fit_error, 6),
        "column_candidate_score": round(
            (1000.0 if quality_ok else 0.0)
            + (300.0 if reconstruction_ok else 0.0)
            + geometry_score * 120.0
            + recognized_count * 2.0
            - uncertain_count * 8.0,
            6,
        ),
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
    unknown = int(result.get("unknown_candidates", result.get("uncertain_count", 0)) or 0)
    raw = int(result.get("raw_contours", 0) or 0)
    if recognized <= 0:
        return -9999.0
    noise = max(0, raw - recognized * 8)
    fixed = str(result.get("method") or "").startswith("fixed_")
    quality_bonus = 55.0 if bool(result.get("quality_ok")) else -35.0
    reconstruction_bonus = 25.0 if bool(result.get("reconstructed_all", not fixed)) else -45.0
    alignment = float(dict(result.get("effective_grid") or {}).get("score", 0.0) or 0.0)
    return (
        recognized * 3.0
        - unknown * 4.0
        - noise * 0.04
        + preference
        + quality_bonus
        + reconstruction_bonus
        + alignment * 20.0
    )

def _run_region(
    image: np.ndarray,
    roi: Tuple[float, float, float, float],
    name: str,
    preference: float,
    *,
    fixed_grid: bool = False,
    ring_grid: bool = False,
    grid_columns: Optional[int] = None,
    layout_profile: str = "",
) -> Dict[str, Any]:
    crop, pixels = _crop(image, roi)
    started = time.perf_counter()

    if ring_grid:
        result = _detect_mobile_ring_grid(
            crop, profile=layout_profile or name
        )
    elif fixed_grid:
        result = _detect_fixed_grid(
            crop, grid_columns=grid_columns, profile=layout_profile or name
        )
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
            "fixed_grid": bool(fixed_grid or ring_grid),
            "ring_grid": bool(ring_grid),
            "layout_profile": str(result.get("layout_profile") or layout_profile or ""),
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


def _looks_like_road_crop(image: np.ndarray) -> bool:
    height, width = image.shape[:2]
    if width / max(1.0, float(height)) < ROAD_CROP_MIN_ASPECT:
        return False
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    bright_neutral = ((value >= 185) & (saturation <= 70)).astype(np.uint8)
    bright_fraction = float(np.mean(bright_neutral))
    red, blue, _, _ = _color_masks(image)
    color_fraction = float(np.mean((red | blue) > 0))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_x = np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)), axis=0)
    edge_y = np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)), axis=1)
    periodic_energy = float(
        min(1.0, (np.percentile(edge_x, 90) + np.percentile(edge_y, 90)) / 120.0)
    )
    return bool(
        bright_fraction >= ROAD_CROP_BRIGHT_FRACTION
        and color_fraction >= 0.001
        and periodic_energy >= 0.25
    )


def _looks_like_ofalive_android_fullscreen(image: np.ndarray) -> bool:
    """判斷是否為新增支援的 Android Chrome 直式大路版型。

    不依賴館別名稱，避免使用者選擇既有館別時漏掉 Android 版型；但必須同時
    滿足高直式比例與畫面下方指定區域的大面積白色路紙特徵，才會新增候選。
    因此不會改變一般桌面、橫向裁圖或其他不符合此畫面特徵的既有流程。
    """
    if not OFALIVE_ANDROID_PROFILE_ENABLED:
        return False

    height, width = image.shape[:2]
    if height < 360 or width < 240:
        return False

    tall_ratio = height / max(1.0, float(width))
    if not (OFALIVE_ANDROID_MIN_TALL_RATIO <= tall_ratio <= OFALIVE_ANDROID_MAX_TALL_RATIO):
        return False

    sample, _ = _crop(image, OFALIVE_ANDROID_SIGNATURE_ROI)
    if sample.size == 0:
        return False

    # BGR 不需要先轉 HSV：白色路紙同時具有高亮度、低色差，可避免額外耗時。
    pixels = sample.astype(np.int16, copy=False)
    channel_min = np.min(pixels, axis=2)
    channel_span = np.max(pixels, axis=2) - channel_min
    bright_neutral = (channel_min >= 175) & (channel_span <= 75)
    return bool(
        float(np.mean(bright_neutral)) >= OFALIVE_ANDROID_MIN_BRIGHT_FRACTION
    )


def _looks_like_dream_compact_mobile_fullscreen(image: np.ndarray) -> bool:
    """判斷珠盤路／大路／下三路橫向並列的 Dream 緊湊手機版。"""
    if not DREAM_COMPACT_MOBILE_PROFILE_ENABLED:
        return False

    height, width = image.shape[:2]
    if height < 500 or not (
        DREAM_COMPACT_MOBILE_MIN_WIDTH <= width <= DREAM_COMPACT_MOBILE_MAX_WIDTH
    ):
        return False

    tall_ratio = height / max(1.0, float(width))
    if not (
        DREAM_COMPACT_MOBILE_MIN_TALL_RATIO
        <= tall_ratio
        <= DREAM_COMPACT_MOBILE_MAX_TALL_RATIO
    ):
        return False

    sample, _ = _crop(image, DREAM_COMPACT_MOBILE_BIG_ROAD_ROI)
    if sample.size == 0:
        return False

    pixels = sample.astype(np.int16, copy=False)
    channel_min = np.min(pixels, axis=2)
    channel_span = np.max(pixels, axis=2) - channel_min
    bright_neutral = (channel_min >= 175) & (channel_span <= 75)
    return bool(
        float(np.mean(bright_neutral))
        >= DREAM_COMPACT_MOBILE_MIN_BRIGHT_FRACTION
    )


def _shifted_profile_rois(
    base: Tuple[float, float, float, float],
    *,
    y_radius: float,
) -> List[Tuple[float, float, float, float]]:
    x, y, width, height = base
    steps = max(1, ROAD_PROFILE_SEARCH_STEPS)
    x_shifts = [0.0] if ROAD_PROFILE_SEARCH_X <= 0 or steps == 1 else list(
        np.linspace(-ROAD_PROFILE_SEARCH_X, ROAD_PROFILE_SEARCH_X, min(3, steps))
    )
    y_shifts = [0.0] if y_radius <= 0 or steps == 1 else list(
        np.linspace(-y_radius, y_radius, steps)
    )
    variants: List[Tuple[float, float, float, float]] = []
    for dy in y_shifts:
        for dx in x_shifts:
            nx = max(0.0, min(1.0 - width, x + float(dx)))
            ny = max(0.0, min(1.0 - height, y + float(dy)))
            roi = (nx, ny, width, height)
            if not any(all(abs(a - b) < 1e-8 for a, b in zip(roi, prior)) for prior in variants):
                variants.append(roi)
    variants.sort(key=lambda roi: abs(roi[0] - x) + abs(roi[1] - y))
    return variants


def detect_road_sequence_detailed(
    image_path: str | Path,
    *,
    venue: str = "",
    input_type: str = "auto",
) -> Dict[str, Any]:
    image = _read_image(image_path)
    image_height, image_width = image.shape[:2]
    venue_code = str(venue or "").upper().strip()
    requested = str(input_type or "auto").lower().strip()
    aspect = image_width / max(1.0, float(image_height))

    if requested not in {"auto", "full_screen", "road_crop", "wide_multi_road"}:
        requested = "auto"
    crop_signature = _looks_like_road_crop(image) if requested == "auto" else False
    likely_crop = requested == "road_crop" or crop_signature
    wide_multi_road = requested == "wide_multi_road" or (
        requested == "auto" and aspect >= WIDE_LAYOUT_MIN_ASPECT and not likely_crop
    )
    detected_type = (
        "road_crop" if likely_crop else "wide_multi_road" if wide_multi_road else "full_screen"
    )

    errors: List[str] = []
    candidates: List[Dict[str, Any]] = []
    attempted: List[str] = []
    plan: List[Dict[str, Any]] = []
    dream_compact_mobile_layout = False
    ofalive_android_layout = False

    if likely_crop:
        plan.append({
            "name": "road_crop_dynamic_6xN",
            "roi": (0.0, 0.0, 1.0, 1.0),
            "preference": 18.0,
            "fixed_grid": True,
            "grid_columns": None,
            "profile": "road_crop_dynamic",
        })
    elif wide_multi_road:
        plan.append({
            "name": "wide_top_dynamic_6xN",
            "roi": WIDE_TOP_ROAD_ROI,
            "preference": 10.0,
            "fixed_grid": True,
            "grid_columns": None,
            "profile": "wide_multi_road",
        })
    else:
        portrait = image_height > image_width * 1.15
        landscape = image_width > image_height * 1.25
        dream_compact_mobile_layout = bool(
            portrait
            and venue_code in {"", "DG"}
            and _looks_like_dream_compact_mobile_fullscreen(image)
        )
        ofalive_android_layout = bool(
            portrait
            and not dream_compact_mobile_layout
            and _looks_like_ofalive_android_fullscreen(image)
        )

        if dream_compact_mobile_layout:
            # 必須先於 ofalive 與既有 DG 手機候選執行；這張版型的右側是下三路，
            # 只有中間白色六列區塊可作為大路反推。候選失敗時仍會繼續原有流程。
            for index, roi in enumerate(
                _shifted_profile_rois(
                    DREAM_COMPACT_MOBILE_BIG_ROAD_ROI,
                    y_radius=DREAM_COMPACT_MOBILE_PROFILE_SEARCH_Y,
                )
            ):
                plan.append({
                    "name": f"dream_compact_mobile_big_road_{index}",
                    "roi": roi,
                    "preference": 64.0 - index * 0.4,
                    "fixed_grid": True,
                    "grid_columns": None,
                    "profile": "dream_compact_mobile_full_screen",
                })

        if ofalive_android_layout:
            # 這批候選排在既有版型之前，確保 Android Chrome 的完整六列大路
            # 不會先被較淺的舊手機 ROI 裁掉；若本批未通過，下面所有原有流程
            # 仍會依原順序繼續執行。
            for index, roi in enumerate(
                _shifted_profile_rois(
                    OFALIVE_ANDROID_BIG_ROAD_ROI,
                    y_radius=OFALIVE_ANDROID_PROFILE_SEARCH_Y,
                )
            ):
                plan.append({
                    "name": f"ofalive_android_big_road_{index}",
                    "roi": roi,
                    "preference": 60.0 - index * 0.4,
                    "fixed_grid": True,
                    "grid_columns": None,
                    "profile": "ofalive_android_chrome_full_screen",
                })

        if portrait and venue_code == "DG":
            # 新增兩種 DG 942×2048 手機畫面；先嘗試精準 ROI，
            # 未通過品質閘門時才繼續執行原本 dg_mobile_big_road_* 流程。
            plan.append({
                "name": "dg_mobile_lower_full_view",
                "roi": DG_MOBILE_LOWER_FULL_VIEW_ROI,
                "preference": 56.0,
                "fixed_grid": True,
                "grid_columns": None,
                "profile": "dg_mobile_full_screen",
            })
            plan.append({
                "name": "dg_mobile_lower_browser_view",
                "roi": DG_MOBILE_LOWER_BROWSER_VIEW_ROI,
                "preference": 55.5,
                "fixed_grid": True,
                "grid_columns": None,
                "profile": "dg_mobile_full_screen",
            })

        if portrait and venue_code in {"", "DG"}:
            for index, roi in enumerate(
                _shifted_profile_rois(
                    DG_MOBILE_BIG_ROAD_ROI, y_radius=ROAD_PROFILE_SEARCH_Y_MOBILE
                )
            ):
                plan.append({
                    "name": f"dg_mobile_big_road_{index}",
                    "roi": roi,
                    "preference": 40.0 - index * 0.4 if venue_code == "DG" else 24.0 - index * 0.4,
                    "fixed_grid": True,
                    "grid_columns": None,
                    "profile": "dg_mobile_full_screen",
                })
        if portrait and venue_code == "MT":
            for index, roi in enumerate(
                _shifted_profile_rois(
                    MT_MOBILE_BIG_ROAD_ROI,
                    y_radius=MT_PROFILE_SEARCH_Y_MOBILE,
                )
            ):
                plan.append({
                    "name": f"mt_mobile_big_road_{index}",
                    "roi": roi,
                    "preference": 52.0 - index * 0.4,
                    "fixed_grid": False,
                    "ring_grid": True,
                    "grid_columns": None,
                    "profile": "mt_mobile_full_screen",
                })
        if portrait and venue_code == "DB":
            for index, roi in enumerate(
                _shifted_profile_rois(
                    DB_MOBILE_BIG_ROAD_ROI,
                    y_radius=DB_PROFILE_SEARCH_Y_MOBILE,
                )
            ):
                plan.append({
                    "name": f"db_mobile_big_road_{index}",
                    "roi": roi,
                    "preference": 52.0 - index * 0.4,
                    "fixed_grid": False,
                    "ring_grid": True,
                    "grid_columns": None,
                    "profile": "db_mobile_full_screen",
                })
        if landscape and venue_code in {"", "DG"}:
            for index, roi in enumerate(
                _shifted_profile_rois(
                    DG_DESKTOP_BIG_ROAD_ROI, y_radius=ROAD_PROFILE_SEARCH_Y_DESKTOP
                )
            ):
                plan.append({
                    "name": f"dg_desktop_big_road_{index}",
                    "roi": roi,
                    "preference": 40.0 - index * 0.4 if venue_code == "DG" else 24.0 - index * 0.4,
                    "fixed_grid": True,
                    "grid_columns": None,
                    "profile": "dg_desktop_full_screen",
                })
        if venue_code == "MT":
            plan.append({
                "name": "mt_fixed_dynamic_6xN",
                "roi": MT_FIXED_ROAD_ROI,
                "preference": 20.0,
                "fixed_grid": True,
                "grid_columns": None,
                "profile": "mt_full_screen",
            })
        if venue_code in VENUE_ROIS and venue_code != "MT":
            plan.append({
                "name": f"{venue_code.lower()}_venue_roi",
                "roi": VENUE_ROIS[venue_code],
                "preference": 3.0,
                "fixed_grid": False,
                "grid_columns": None,
                "profile": "legacy_venue_roi",
            })
        plan.append({
            "name": "generic_road_roi",
            "roi": ROAD_ROI,
            "preference": 1.5,
            "fixed_grid": False,
            "grid_columns": None,
            "profile": "legacy_generic",
        })
        if ROAD_AUTO_FULL_FALLBACK:
            plan.append({
                "name": "full_image",
                "roi": (0.0, 0.0, 1.0, 1.0),
                "preference": 0.0,
                "fixed_grid": False,
                "grid_columns": None,
                "profile": "legacy_full_image",
            })

    seen = set()
    best: Optional[Dict[str, Any]] = None
    for item in plan:
        roi = tuple(float(value) for value in item["roi"])
        fixed_grid = bool(item["fixed_grid"])
        ring_grid = bool(item.get("ring_grid", False))
        key = (
            tuple(round(value, 6) for value in roi),
            fixed_grid,
            ring_grid,
            item.get("grid_columns"),
        )
        if key in seen:
            continue
        seen.add(key)
        name = str(item["name"])
        attempted.append(name)
        try:
            current = _run_region(
                image,
                roi,
                name,
                float(item["preference"]),
                fixed_grid=fixed_grid,
                ring_grid=ring_grid,
                grid_columns=item.get("grid_columns"),
                layout_profile=str(item.get("profile") or ""),
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
            "input_type": detected_type,
            "errors": errors,
            "attempted_regions": attempted,
        }

    result = dict(best)
    result.update({
        "ok": bool(result.get("sequence")) and bool(result.get("quality_ok", True)),
        "input_type": detected_type,
        "selected_region": str(best.get("region_name") or ""),
        "venue_hint": venue_code,
        "errors": errors,
        "attempted_regions": attempted,
        "fast_early_exit": len(candidates) < len(plan),
        "wide_layout_detected": bool(wide_multi_road),
        "road_crop_signature": bool(crop_signature),
        "dream_compact_mobile_profile_detected": bool(dream_compact_mobile_layout),
        "ofalive_android_profile_detected": bool(ofalive_android_layout),
        "image_size": {"width": image_width, "height": image_height},
        "candidate_regions": [
            {
                "name": item.get("region_name"),
                "recognized_count": int(item.get("recognized_count", 0) or 0),
                "unknown_candidates": int(item.get("unknown_candidates", item.get("uncertain_count", 0)) or 0),
                "score": float(item.get("selection_score", -9999)),
                "elapsed_ms": float(item.get("elapsed_ms", 0) or 0),
                "method": str(item.get("method") or ""),
                "fixed_grid": bool(item.get("fixed_grid")),
                "ring_grid": bool(item.get("ring_grid")),
                "quality_ok": bool(item.get("quality_ok", True)),
                "reconstructed_all": bool(item.get("reconstructed_all", True)),
                "fallback_reason": str(item.get("fallback_reason") or ""),
                "alignment_score": float(dict(item.get("effective_grid") or {}).get("score", 0.0) or 0.0),
                "grid_columns": int(item.get("grid_columns", 0) or 0),
                "layout_profile": str(item.get("layout_profile") or ""),
                "roi": dict(item.get("roi") or {}),
            }
            for item in candidates
        ],
    })
    return result


def detect_road_sequence(image_path: str | Path) -> List[str]:
    return list(detect_road_sequence_detailed(image_path).get("sequence") or [])


__all__ = ["detect_road_sequence", "detect_road_sequence_detailed"]
