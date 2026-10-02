"""遊戲畫面大路偵測模組 V11.6（保留既有館別，補強手機全畫面容錯）。

重點：
1. DG 手機／電腦 ROI、滑動搜尋與固定格辨識流程完整保留 V11.2，不改動原有邏輯。
2. 新增 MT／DB 手機直式全畫面專用 ROI、附近滑動追焦與彩色圓環格位辨識。
3. 固定維持 6 列，MT／DB 允許非正方形格位；欄距與列距由彩色圓環中心自動估計。
4. 每個圓環分別統計紅、藍、綠 HSV 像素；雙色接近或偏離格位者標記 uncertain。
5. 依標準大路落點規則（含長龍右黏狀態）反推時間序列；失敗不把欄排序當正確答案。
6. 可用 ROAD_GRID_DEBUG=1 輸出追焦疊圖；版型不吻合時仍以品質閘門阻擋錯序列。
7. 新增 ofalive99 類 Android Chrome 直式全畫面候選；僅在畫面比例與底部白色大路特徵吻合時啟用。
8. 新增 Dream Gaming 緊湊手機版候選；獨立辨識中間大路，不將右側下三路混入。
9. 專用版型未命中時，最後追加兩個低優先級手機大路候選；不覆蓋既有成功格式。
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
    # DB 深色手機版實圖校準；真正大路位於右側上層，避免把下三路一起裁入。
    (0.360, 0.755, 0.640, 0.075),
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
    # DG Dream Gaming 實圖校準；只取白色面板上半部大路，不包含下三路。
    (0.330, 0.730, 0.650, 0.092),
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
    # 869x1884 supplied screenshot target ~= (0.326, 0.732, 0.647, 0.091).
    # Keep padding so browser chrome / safe-area shifts still leave all six rows visible.
    (0.305, 0.710, 0.685, 0.125),
)
DREAM_COMPACT_MOBILE_PROFILE_SEARCH_Y = _env_float(
    "DREAM_COMPACT_MOBILE_PROFILE_SEARCH_Y", 0.065, 0.0, 0.14
)
DREAM_COMPACT_MOBILE_MIN_WIDTH = _env_int(
    "DREAM_COMPACT_MOBILE_MIN_WIDTH", 280, 240, 1600
)
DREAM_COMPACT_MOBILE_MAX_WIDTH = _env_int(
    "DREAM_COMPACT_MOBILE_MAX_WIDTH", 1600, 240, 2400
)
DREAM_COMPACT_MOBILE_MIN_TALL_RATIO = _env_float(
    "DREAM_COMPACT_MOBILE_MIN_TALL_RATIO", 1.55, 1.20, 4.00
)
DREAM_COMPACT_MOBILE_MAX_TALL_RATIO = _env_float(
    "DREAM_COMPACT_MOBILE_MAX_TALL_RATIO", 2.75, 1.20, 4.00
)
DREAM_COMPACT_MOBILE_MIN_BRIGHT_FRACTION = _env_float(
    "DREAM_COMPACT_MOBILE_MIN_BRIGHT_FRACTION", 0.38, 0.10, 0.95
)

# 專用手機 Profile 只先嘗試最接近的兩個 ROI，避免某一張全圖因為
# Safari／Chrome 工具列不同而把 5×3 個滑動 ROI 與舊備援全數疊加，
# 造成 LINE 圖片回覆逾時。
MOBILE_PROFILE_MAX_CANDIDATES = _env_int(
    "MOBILE_PROFILE_MAX_CANDIDATES", 2, 1, 2
)
MOBILE_AUTO_FOCUS_ENABLED = os.getenv("MOBILE_AUTO_FOCUS_ENABLED", "1").strip() == "1"
DG_DB_FEATURE_LOCATORS_ENABLED = (
    os.getenv("DG_DB_FEATURE_LOCATORS_ENABLED", "1").strip() == "1"
)
DG_FEATURE_MAX_CANDIDATES = _env_int(
    "DG_FEATURE_MAX_CANDIDATES", 2, 1, 3
)
DB_FEATURE_MAX_CANDIDATES = _env_int(
    "DB_FEATURE_MAX_CANDIDATES", 2, 1, 3
)
DB_FEATURE_PREVIEW_SIDE = _env_int(
    "DB_FEATURE_PREVIEW_SIDE", 820, 640, 1080
)
DB_FEATURE_PROBE_WINDOWS = _env_int(
    "DB_FEATURE_PROBE_WINDOWS", 18, 8, 32
)
DB_RING_MIN_DOMINANCE = _env_float(
    "DB_RING_MIN_DOMINANCE", 1.08, 1.02, 1.80
)
DB_RING_MIN_COLOR_RATIO = _env_float(
    "DB_RING_MIN_COLOR_RATIO", 0.045, 0.020, 0.12
)
MOBILE_AUTO_FOCUS_MAX_CANDIDATES = _env_int(
    "MOBILE_AUTO_FOCUS_MAX_CANDIDATES", 3, 1, 4
)
MOBILE_AUTO_FOCUS_PREVIEW_SIDE = _env_int(
    "MOBILE_AUTO_FOCUS_PREVIEW_SIDE", 640, 420, 960
)
MOBILE_AUTO_FOCUS_MIN_AREA_RATIO = _env_float(
    "MOBILE_AUTO_FOCUS_MIN_AREA_RATIO", 0.012, 0.004, 0.20
)
MOBILE_AUTO_FOCUS_MAX_AREA_RATIO = _env_float(
    "MOBILE_AUTO_FOCUS_MAX_AREA_RATIO", 0.42, 0.10, 0.80
)
MOBILE_AUTO_FOCUS_MIN_ASPECT = _env_float(
    "MOBILE_AUTO_FOCUS_MIN_ASPECT", 1.12, 1.02, 4.00
)
MOBILE_AUTO_FOCUS_MIN_WHITE = _env_float(
    "MOBILE_AUTO_FOCUS_MIN_WHITE", 0.48, 0.20, 0.90
)
ROAD_ADAPTIVE_COLOR = os.getenv("ROAD_ADAPTIVE_COLOR", "1").strip() == "1"
ROAD_FAST_MAX_COLUMN_CANDIDATES = _env_int(
    "ROAD_FAST_MAX_COLUMN_CANDIDATES", 3, 1, 8
)
ROAD_GENERIC_AUTO_COL_MAX = _env_int(
    "ROAD_GENERIC_AUTO_COL_MAX", 48, 24, 60
)
ROAD_GENERIC_MAX_COLUMN_CANDIDATES = _env_int(
    "ROAD_GENERIC_MAX_COLUMN_CANDIDATES", 3, 2, 6
)
ROAD_GENERIC_MIN_SQUARE_SCORE = _env_float(
    "ROAD_GENERIC_MIN_SQUARE_SCORE", 0.46, 0.25, 0.80
)
ROAD_FAST_EARLY_EXIT_MIN_CANDIDATES = _env_int(
    "ROAD_FAST_EARLY_EXIT_MIN_CANDIDATES", 2, 1, 4
)
ROAD_RECONSTRUCTION_ONE_CELL_REPAIR = (
    os.getenv("ROAD_RECONSTRUCTION_ONE_CELL_REPAIR", "1").strip() == "1"
)
ROAD_REPAIR_MAX_CANDIDATES = _env_int(
    "ROAD_REPAIR_MAX_CANDIDATES", 4, 1, 8
)
ROAD_DETECTOR_HARD_TIMEOUT_SECONDS = _env_float(
    "ROAD_DETECTOR_HARD_TIMEOUT_SECONDS", 7.0, 2.0, 12.0
)
ROAD_RECONSTRUCT_MAX_SECONDS = _env_float(
    "ROAD_RECONSTRUCT_MAX_SECONDS", 0.35, 0.03, 1.0
)
ROAD_RECONSTRUCT_MAX_NODES = _env_int(
    "ROAD_RECONSTRUCT_MAX_NODES", 300000, 200000, 2000000
)

# 不依賴手機型號、畫面比例或館別的最後一層全圖容錯。它們只會排在
# 所有既有專用 ROI 與 legacy generic ROI 之後；因此原本成功的格式
# 會先照原邏輯返回，新候選僅處理「原本沒有任何可信結果」的版型。
MOBILE_FULLSCREEN_FALLBACK_ENABLED = (
    os.getenv("MOBILE_FULLSCREEN_FALLBACK_ENABLED", "1").strip() == "1"
)
MOBILE_FULLSCREEN_FALLBACK_ROIS = (
    _env_roi("MOBILE_FULLSCREEN_COMPACT_ROI", (0.250, 0.690, 0.500, 0.190)),
    _env_roi("MOBILE_FULLSCREEN_WIDE_ROI", (0.280, 0.670, 0.680, 0.200)),
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
ROAD_GRID_ALIGN_SEARCH_STEPS = _env_int("ROAD_GRID_ALIGN_SEARCH_STEPS", 9, 5, 41)
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
    "MOBILE_RING_MIN_COLOR_PIXELS", 10, 4, 300
)
MOBILE_RING_COLOR_DOMINANCE = _env_float(
    "MOBILE_RING_COLOR_DOMINANCE", 1.20, 1.05, 5.0
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


def _circular_hue_distance(hue: np.ndarray, center: float) -> np.ndarray:
    delta = np.abs(hue.astype(np.float32) - float(center))
    return np.minimum(delta, 180.0 - delta)


def _compact_component_mask(mask: np.ndarray) -> np.ndarray:
    """只保留圓環/小標記型彩色元件，排除大按鈕、橫幅、整片 UI。"""
    source = (mask > 0).astype(np.uint8)
    if source.size == 0 or int(source.sum()) == 0:
        return source
    count, labels, stats, _ = cv2.connectedComponentsWithStats(source, 8)
    out = np.zeros_like(source)
    image_area = float(source.shape[0] * source.shape[1])
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        if area < 3:
            continue
        area_ratio = area / max(1.0, image_area)
        span_x = width / max(1.0, float(source.shape[1]))
        span_y = height / max(1.0, float(source.shape[0]))
        aspect = max(width, height) / max(1.0, float(min(width, height)))
        if area_ratio > 0.035 or span_x > 0.42 or span_y > 0.42 or aspect > 7.0:
            continue
        out[labels == index] = 1
    return out


def _slice_bounds(
    array: np.ndarray,
    bounds: Optional[Mapping[str, Any]],
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    height, width = array.shape[:2]
    if not bounds:
        return array, (0, 0, width, height)
    x1 = max(0, min(width - 1, int(bounds.get("x", 0) or 0)))
    y1 = max(0, min(height - 1, int(bounds.get("y", 0) or 0)))
    x2 = max(x1 + 1, min(width, x1 + int(bounds.get("width", width) or width)))
    y2 = max(y1 + 1, min(height, y1 + int(bounds.get("height", height) or height)))
    return array[y1:y2, x1:x2], (x1, y1, x2, y2)


def _adaptive_hsv_profile(
    hsv: np.ndarray,
    calibration_bounds: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    sample, _ = _slice_bounds(hsv, calibration_bounds)
    hue, saturation, value = cv2.split(sample)
    valid = (saturation >= 16) & (value >= 25)

    # 藍色起點提高到 82，避免綠色和局/青綠 UI 進入藍色校準池。
    red_broad = valid & ((hue <= 28) | (hue >= 152))
    blue_broad = valid & (hue >= 82) & (hue <= 155)
    red_seed = _compact_component_mask(red_broad.astype(np.uint8)) > 0
    blue_seed = _compact_component_mask(blue_broad.astype(np.uint8)) > 0

    # 若圓環被壓縮得太碎，才退回 grid bounds 內的 broad seed；仍不使用整張 UI。
    if int(red_seed.sum()) < 8:
        red_seed = red_broad
    if int(blue_seed.sum()) < 8:
        blue_seed = blue_broad

    def _channel_floor(values: np.ndarray, fallback: int, low: float) -> float:
        if values.size < 8:
            return float(fallback)
        return float(np.clip(np.percentile(values, 18) * 0.60, low, fallback))

    red_hues = hue[red_seed].astype(np.float32)
    if red_hues.size >= 8:
        signed = np.where(red_hues > 90.0, red_hues - 180.0, red_hues)
        center_signed = float(np.median(signed))
        red_center = center_signed % 180.0
        red_mad = float(np.median(np.abs(signed - center_signed)))
        red_tol = float(np.clip(12.0 + red_mad * 2.0, 14.0, 30.0))
    else:
        red_center, red_tol = 0.0, 20.0

    blue_hues = hue[blue_seed].astype(np.float32)
    if blue_hues.size >= 8:
        blue_center = float(np.median(blue_hues))
        blue_mad = float(np.median(np.abs(blue_hues - blue_center)))
        blue_tol = float(np.clip(14.0 + blue_mad * 1.8, 16.0, 32.0))
    else:
        blue_center, blue_tol = 112.0, 28.0

    return {
        "red_center": red_center,
        "red_tol": red_tol,
        "blue_center": blue_center,
        "blue_tol": blue_tol,
        "red_s": _channel_floor(saturation[red_seed], ROAD_GRID_RED_MIN_S, 20.0),
        "red_v": _channel_floor(value[red_seed], ROAD_GRID_RED_MIN_V, 24.0),
        "blue_s": _channel_floor(saturation[blue_seed], ROAD_GRID_BLUE_MIN_S, 18.0),
        "blue_v": _channel_floor(value[blue_seed], ROAD_GRID_BLUE_MIN_V, 24.0),
    }


def _broad_color_masks(
    crop: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """格線對齊第一階段只用寬鬆色罩，不做整張 UI 的自適應校色。"""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    valid = (saturation >= 18) & (value >= 25)
    red = (valid & ((hue <= 30) | (hue >= 150))).astype(np.uint8)
    blue = (valid & (hue >= 82) & (hue <= 158)).astype(np.uint8)
    green = (
        (hue >= 30)
        & (hue <= 92)
        & (saturation >= 24)
        & (value >= 25)
    ).astype(np.uint8)
    union = ((red | blue | green) > 0).astype(np.uint8)
    return red, blue, green, union


def _color_masks(
    crop: np.ndarray,
    *,
    calibration_bounds: Optional[Mapping[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    if ROAD_ADAPTIVE_COLOR:
        profile = _adaptive_hsv_profile(hsv, calibration_bounds)
        red_hue = _circular_hue_distance(hue, profile["red_center"]) <= profile["red_tol"]
        blue_hue = _circular_hue_distance(hue, profile["blue_center"]) <= profile["blue_tol"]
        red = (
            red_hue
            & (saturation >= profile["red_s"])
            & (value >= profile["red_v"])
        ).astype(np.uint8)
        blue = (
            blue_hue
            & (saturation >= profile["blue_s"])
            & (value >= profile["blue_v"])
        ).astype(np.uint8)
    else:
        _, _, _, _ = _broad_color_masks(crop)
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
        (hue >= 30)
        & (hue <= 92)
        & (saturation >= max(24, int(ROAD_GRID_GREEN_MIN_S * 0.68)))
        & (value >= max(25, int(ROAD_GRID_GREEN_MIN_V * 0.68)))
    ).astype(np.uint8)
    union = ((red | blue | green) > 0).astype(np.uint8)
    return red, blue, green, union


def _db_color_masks(
    crop: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """DB 真人專用自適應顏色。

    DB 的紅環大多落在 HSV 高紅端，藍環落在約 90~140。
    這裡刻意讓紅/藍 hue 區間不重疊，避免深色 UI / 瀏覽器 icon
    同時被算成紅與藍。飽和度則依每張圖自動下修，兼容手機色溫、
    LINE 壓縮與低飽和截圖。
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    valid = (value >= 22) & (saturation >= 8)

    red_hue = (hue <= 24) | (hue >= 156)
    blue_hue = (hue >= 86) & (hue <= 146)

    def _adaptive_s_floor(seed: np.ndarray, fallback: float) -> float:
        values = saturation[seed & valid]
        if values.size < 12:
            return float(fallback)
        return float(np.clip(np.percentile(values, 20) * 0.58, 16.0, 44.0))

    red_s = _adaptive_s_floor(red_hue, 28.0)
    blue_s = _adaptive_s_floor(blue_hue, 26.0)

    red = (
        red_hue
        & (saturation >= red_s)
        & (value >= 24)
    ).astype(np.uint8)
    blue = (
        blue_hue
        & (saturation >= blue_s)
        & (value >= 24)
    ).astype(np.uint8)
    green = (
        (hue >= 30)
        & (hue <= 92)
        & (saturation >= 18)
        & (value >= 22)
    ).astype(np.uint8)

    union = ((red | blue | green) > 0).astype(np.uint8)
    return red, blue, green, union, {
        "red_s": red_s,
        "blue_s": blue_s,
    }


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
    profile: str = "",
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
            profile_key = str(profile or "").lower()
            mobile_like_profile = (
                "mobile" in profile_key
                or profile_key.startswith("road_crop")
                or profile_key.startswith("dg_feature_white_grid")
                or profile_key.startswith("dream_compact")
            )
            if mobile_like_profile:
                minimum_pixels = max(
                    5,
                    int(round(inner_area * min(ROAD_GRID_MIN_COLOR_RATIO, 0.010))),
                )
                minimum_component = max(
                    3,
                    int(round(inner_area * min(ROAD_GRID_MIN_COMPONENT_AREA_RATIO, 0.010))),
                )
            else:
                minimum_pixels = max(
                    ROAD_GRID_MIN_COLOR_PIXELS,
                    int(round(inner_area * ROAD_GRID_MIN_COLOR_RATIO)),
                )
                minimum_component = max(
                    4, int(round(inner_area * ROAD_GRID_MIN_COMPONENT_AREA_RATIO))
                )
            component_probe = max(3, int(round(minimum_pixels * 0.35)))
            if red_pixels >= component_probe:
                red_component, red_span_x, red_span_y = _largest_component_stats(
                    red_mask[inner_y1:inner_y2, inner_x1:inner_x2]
                )
            else:
                red_component, red_span_x, red_span_y = 0, 0.0, 0.0
            if blue_pixels >= component_probe:
                blue_component, blue_span_x, blue_span_y = _largest_component_stats(
                    blue_mask[inner_y1:inner_y2, inner_x1:inner_x2]
                )
            else:
                blue_component, blue_span_x, blue_span_y = 0, 0.0, 0.0
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

            tie_minimum = max(
                ROAD_GRID_TIE_MIN_PIXELS,
                int(round(inner_area * ROAD_GRID_TIE_MIN_AREA_RATIO)),
            )
            if outcome and green_pixels >= max(3, int(round(tie_minimum * 0.35))):
                largest_green, green_concentration, green_span_x, green_span_y = _green_component_stats(
                    green_mask[inner_y1:inner_y2, inner_x1:inner_x2]
                )
            else:
                largest_green, green_concentration, green_span_x, green_span_y = 0, 0.0, 0.0, 0.0
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
    """依大路規則反推時間序；歧義搜尋有硬 node/time budget，禁止 DFS 卡死。"""
    grid: Dict[Tuple[int, int], str] = {
        (int(item.get("column", 0)), int(item.get("row", 0))): str(item.get("outcome") or "").upper()
        for item in cells
        if str(item.get("outcome") or "").upper() in {"B", "P"}
    }
    fallback_preview = sorted(grid, key=lambda position: (position[0], position[1]))
    base_details = {
        "positions": [],
        "reconstructed_all": False,
        "partial_positions": [],
        "fallback_preview": fallback_preview,
        "solution_count": 0,
        "search_nodes": 0,
        "search_ms": 0.0,
        "budget_exhausted": False,
    }
    if not grid:
        details = {**base_details, "fallback_reason": "no_recognized_cells", "fallback_preview": []}
        return details if return_details else []
    if (0, 0) not in grid:
        details = {**base_details, "fallback_reason": "missing_big_road_origin_0_0"}
        return details if return_details else []

    target_count = len(grid)
    first_outcome = grid[(0, 0)]
    best_partial: List[Tuple[int, int]] = [(0, 0)]
    solutions: List[List[Tuple[int, int]]] = []
    search_started = time.perf_counter()
    search_deadline = search_started + ROAD_RECONSTRUCT_MAX_SECONDS
    search_nodes = 0
    budget_exhausted = False

    def search(
        current: Tuple[int, int],
        run_start_column: int,
        previous: str,
        tailing_right: bool,
        visited: set[Tuple[int, int]],
        ordered: List[Tuple[int, int]],
    ) -> None:
        nonlocal best_partial, search_nodes, budget_exhausted
        search_nodes += 1
        if search_nodes > ROAD_RECONSTRUCT_MAX_NODES:
            budget_exhausted = True
            return
        if (search_nodes & 31) == 0 and time.perf_counter() >= search_deadline:
            budget_exhausted = True
            return
        if len(ordered) > len(best_partial):
            best_partial = list(ordered)
        if len(ordered) == target_count:
            solutions.append(list(ordered))
            return
        if len(solutions) >= 2 or budget_exhausted:
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
            if budget_exhausted:
                break
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
    search_ms = (time.perf_counter() - search_started) * 1000.0
    unique = len(solutions) == 1 and not budget_exhausted
    positions = solutions[0] if unique else []
    if unique:
        fallback_reason = ""
    elif budget_exhausted:
        fallback_reason = f"big_road_reconstruction_budget_exhausted_{len(best_partial)}_of_{target_count}"
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
        "search_nodes": int(search_nodes),
        "search_ms": round(float(search_ms), 3),
        "budget_exhausted": bool(budget_exhausted),
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


def _column_candidates(
    crop: np.ndarray,
    requested: Optional[int] = None,
    *,
    profile: str = "",
) -> List[int]:
    if requested is not None:
        return [max(5, min(60, int(requested)))]
    height, width = crop.shape[:2]
    aspect = width / max(1.0, float(height))
    profile_key = str(profile or "").lower()
    generic_auto = (
        profile_key.startswith("mobile_auto_general")
        or profile_key.startswith("dg_feature_white_grid")
    )
    maximum = max(ROAD_GRID_AUTO_COL_MAX, ROAD_GENERIC_AUTO_COL_MAX) if generic_auto else ROAD_GRID_AUTO_COL_MAX

    if not ROAD_GRID_AUTO_COLUMNS:
        return [ROAD_GRID_COLS]

    values = {ROAD_GRID_COLS}
    if generic_auto:
        square_estimate = int(round(aspect * ROAD_GRID_ROWS))
        square_estimate = max(
            ROAD_GRID_AUTO_COL_MIN,
            min(maximum, square_estimate),
        )

        # DG 白色大路格線在手機與電腦版都接近正方形。
        # 先測幾何上最直接的欄數與 ±1，避免正確欄數被其他矩形比例假設排到第 4 名之後。
        if profile_key.startswith("dg_feature_white_grid"):
            ordered: List[int] = []
            for candidate in (
                square_estimate,
                square_estimate - 1,
                square_estimate + 1,
                ROAD_GRID_COLS,
            ):
                candidate = max(
                    ROAD_GRID_AUTO_COL_MIN,
                    min(maximum, int(candidate)),
                )
                if candidate not in ordered:
                    ordered.append(candidate)
            return ordered[:ROAD_GENERIC_MAX_COLUMN_CANDIDATES]

        # 通用 Auto Focus 仍保留多種矩形 cell 比例假設。
        for cell_ratio in (0.68, 0.82, 1.0, 1.22, 1.45):
            estimated = int(round(aspect * ROAD_GRID_ROWS / cell_ratio))
            estimated = max(ROAD_GRID_AUTO_COL_MIN, min(maximum, estimated))
            values.add(estimated)
            for delta in (-1, 1):
                candidate = estimated + delta
                if ROAD_GRID_AUTO_COL_MIN <= candidate <= maximum:
                    values.add(candidate)
        values.add(square_estimate)
        ordered = sorted(
            values,
            key=lambda value: (
                min(
                    abs(value - aspect * ROAD_GRID_ROWS / ratio)
                    for ratio in (0.68, 0.82, 1.0, 1.22, 1.45)
                ),
                abs(value - square_estimate),
            ),
        )
        return ordered[:ROAD_GENERIC_MAX_COLUMN_CANDIDATES]

    estimated = int(round(aspect * ROAD_GRID_ROWS))
    estimated = max(ROAD_GRID_AUTO_COL_MIN, min(maximum, estimated))
    values.add(estimated)
    for delta in range(-ROAD_GRID_AUTO_COL_RADIUS, ROAD_GRID_AUTO_COL_RADIUS + 1):
        candidate = estimated + delta
        if ROAD_GRID_AUTO_COL_MIN <= candidate <= maximum:
            values.add(candidate)
    ordered = sorted(
        values,
        key=lambda value: (
            abs(value - estimated),
            0 if value == ROAD_GRID_COLS else 1,
            value,
        ),
    )
    return ordered[:ROAD_FAST_MAX_COLUMN_CANDIDATES]


def _nms_projection_peaks(
    values: np.ndarray,
    *,
    threshold: float,
    min_distance: int,
) -> List[int]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    candidates = np.where(data >= float(threshold))[0]
    if candidates.size == 0:
        return []
    ordered = candidates[np.argsort(data[candidates])[::-1]]
    selected: List[int] = []
    distance = max(2, int(min_distance))
    for raw in ordered:
        index = int(raw)
        if any(abs(index - prior) < distance for prior in selected):
            continue
        selected.append(index)
    return sorted(selected)


def _dg_line_grid_bounds(crop: np.ndarray) -> Optional[Dict[str, Any]]:
    """DG / Dream Gaming 白色大路：直接由連續灰格線鎖定真正 6×N 區域。

    手機縮圖後，候選框可能同時包含左側珠盤路、真正大路、以及下方衍生路。
    因此不能只用「同相位」格線聚類，否則遠處珠盤格線也可能被混入。
    這裡改成：
      1) 垂直軸找最長「連續等距」格線序列；
      2) 水平軸以紅/藍圓中心對齊決定真正 6 列起點；
      3) 第 7 條額外水平線不加分，避免把下三路第一條格線當成大路底線。
    """
    if crop is None or crop.size == 0:
        return None
    height, width = crop.shape[:2]
    if height < 54 or width < 150:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    # 先估白底亮度，再只取比白底暗的低彩度灰線。
    white_reference = float(np.percentile(value, 78))
    neutral_upper = int(np.clip(white_reference - 7.0, 165.0, 244.0))
    neutral_grid = (
        (saturation <= 52)
        & (value >= 72)
        & (value <= neutral_upper)
    ).astype(np.uint8)

    column_strength = neutral_grid.sum(axis=0).astype(np.float64)
    row_strength = neutral_grid.sum(axis=1).astype(np.float64)
    expected_pitch = max(6.0, height / float(ROAD_GRID_ROWS))
    nms_distance = max(3, int(round(expected_pitch * 0.40)))

    vertical_peaks = _nms_projection_peaks(
        column_strength,
        threshold=max(6.0, height * 0.42),
        min_distance=nms_distance,
    )
    horizontal_peaks = _nms_projection_peaks(
        row_strength,
        threshold=max(18.0, width * 0.40),
        min_distance=nms_distance,
    )
    if len(vertical_peaks) < ROAD_GRID_AUTO_COL_MIN + 1 or len(horizontal_peaks) < 4:
        return None

    def _adjacent_pitch(
        peaks: Sequence[int],
        expected: float,
        *,
        low_ratio: float,
        high_ratio: float,
        minimum_count: int,
    ) -> Optional[float]:
        diffs = [
            float(right - left)
            for left, right in zip(peaks, peaks[1:])
            if expected * low_ratio <= (right - left) <= expected * high_ratio
        ]
        if len(diffs) < minimum_count:
            return None
        pitch = float(np.median(np.asarray(diffs, dtype=np.float64)))
        close = [
            item for item in diffs
            if abs(item - pitch) <= max(2.5, pitch * 0.20)
        ]
        if len(close) >= minimum_count:
            pitch = float(np.median(np.asarray(close, dtype=np.float64)))
        return pitch

    # 先由水平 6-row 幾何估 pitch，再用它約束垂直格距。
    pitch_y_seed = _adjacent_pitch(
        horizontal_peaks,
        expected_pitch,
        low_ratio=0.55,
        high_ratio=1.45,
        minimum_count=3,
    )
    if pitch_y_seed is None:
        return None
    pitch_x_seed = _adjacent_pitch(
        vertical_peaks,
        pitch_y_seed,
        low_ratio=0.70,
        high_ratio=1.30,
        minimum_count=5,
    )
    if pitch_x_seed is None:
        return None

    seed_square_score = max(
        0.0,
        1.0
        - abs(pitch_x_seed - pitch_y_seed)
        / max(1.0, pitch_x_seed, pitch_y_seed),
    )
    if seed_square_score < 0.62:
        return None

    # 垂直：只接受連續相鄰格線；最多容許一條格線被圓環/壓縮遮掉。
    x_tolerance = max(2.5, pitch_x_seed * 0.18)
    best_run: Optional[Dict[str, Any]] = None
    for start_index in range(len(vertical_peaks)):
        first = float(vertical_peaks[start_index])
        last = first
        matched: List[int] = [int(vertical_peaks[start_index])]
        grid_steps = 0
        missing_lines = 0

        for next_index in range(start_index + 1, len(vertical_peaks)):
            peak = float(vertical_peaks[next_index])
            gap = peak - last
            if gap < pitch_x_seed * 0.55:
                continue

            one_error = abs(gap - pitch_x_seed)
            two_error = abs(gap - 2.0 * pitch_x_seed)
            if one_error <= x_tolerance:
                grid_steps += 1
                matched.append(int(round(peak)))
                last = peak
                continue
            if (
                missing_lines < 1
                and two_error <= x_tolerance * 1.5
            ):
                grid_steps += 2
                missing_lines += 1
                matched.append(int(round(peak)))
                last = peak
                continue
            break

        if grid_steps < ROAD_GRID_AUTO_COL_MIN:
            continue
        effective_pitch = (last - first) / max(1.0, float(grid_steps))
        coverage = len(matched) / max(1.0, float(grid_steps + 1))
        square_score = max(
            0.0,
            1.0
            - abs(effective_pitch - pitch_y_seed)
            / max(1.0, effective_pitch, pitch_y_seed),
        )
        run_score = (
            float(grid_steps)
            + 2.2 * coverage
            + 0.8 * square_score
            - 0.35 * missing_lines
        )
        candidate = {
            "score": run_score,
            "x0": first,
            "x_last": last,
            "grid_columns": int(grid_steps),
            "pitch_x": float(effective_pitch),
            "coverage": float(coverage),
            "missing_lines": int(missing_lines),
            "matched": matched,
        }
        if (
            best_run is None
            or float(candidate["score"]) > float(best_run["score"])
        ):
            best_run = candidate

    if best_run is None:
        return None

    grid_columns = int(best_run["grid_columns"])
    pitch_x = float(best_run["pitch_x"])
    x0_raw = float(best_run["x0"])
    x2_raw = x0_raw + grid_columns * pitch_x

    # 先不要因為 crop 右緣還剩約一格白底就直接「腦補」一欄。
    # Dream/DG 全畫面在最右側常有 0.6~1.0 格寬的純白 UI 邊界；
    # 舊邏輯會把 20 欄誤推成 21 欄，造成整條大路重建失敗。
    # 是否真的需要補最後一欄，延後到 6-row band 決定後，再用該欄的
    # 真實紅/藍像素證據判斷。
    right_remainder = float(width) - float(best_run["x_last"])
    right_boundary_inferred = False
    right_inference_color_pixels = 0
    right_inference_center_alignment = 0.0

    if not (ROAD_GRID_AUTO_COL_MIN <= grid_columns <= ROAD_GENERIC_AUTO_COL_MAX):
        return None

    # 水平起點不能只看格線數：下三路就在大路下方，常多出第 7 條規律線。
    # 用真正紅/藍圓像素在每格「中央」的對齊程度來選 6-row band。
    valid_color = (saturation >= 14) & (value >= 22)
    red_blue = valid_color & (
        (hue <= 32)
        | (hue >= 148)
        | ((hue >= 82) & (hue <= 160))
    )
    color_y, color_x = np.nonzero(red_blue)
    x_low = max(0, int(math.floor(x0_raw)))
    x_high = min(width, int(math.ceil(x2_raw)))
    within_x = (color_x >= x_low) & (color_x < x_high)
    road_color_y = color_y[within_x]

    y_tolerance = max(2.5, pitch_y_seed * 0.18)
    y_candidates: List[Tuple[float, float, int, float, int, float]] = []
    for peak in horizontal_peaks:
        for boundary_index in range(ROAD_GRID_ROWS + 1):
            y0 = float(peak) - boundary_index * pitch_y_seed
            if y0 < -pitch_y_seed * 0.45:
                continue
            if (
                y0 + ROAD_GRID_ROWS * pitch_y_seed
                > height + pitch_y_seed * 0.45
            ):
                continue

            boundary_errors: List[float] = []
            matches = 0
            for row_boundary in range(ROAD_GRID_ROWS + 1):
                predicted = y0 + row_boundary * pitch_y_seed
                error = min(
                    (abs(float(item) - predicted) for item in horizontal_peaks),
                    default=9999.0,
                )
                if error <= y_tolerance:
                    matches += 1
                    boundary_errors.append(error)
            if matches < 4:
                continue

            band_low = max(0.0, y0)
            band_high = min(
                float(height),
                y0 + ROAD_GRID_ROWS * pitch_y_seed,
            )
            colors = road_color_y[
                (road_color_y >= band_low)
                & (road_color_y < band_high)
            ]
            if colors.size:
                phase = np.mod(
                    (colors.astype(np.float64) - y0) / pitch_y_seed,
                    1.0,
                )
                center_alignment = float(
                    np.mean(
                        np.clip(
                            1.0 - np.abs(phase - 0.5) / 0.5,
                            0.0,
                            1.0,
                        )
                    )
                )
            else:
                center_alignment = 0.0

            # 最多只獎勵 6 條可見邊界；第 7 條可能是下三路第一條線。
            line_coverage = min(matches, ROAD_GRID_ROWS) / float(ROAD_GRID_ROWS)
            mean_error = (
                float(np.mean(boundary_errors))
                if boundary_errors
                else pitch_y_seed
            )
            color_strength = min(1.0, colors.size / 5000.0)
            y_score = (
                1.35 * center_alignment
                + 0.70 * line_coverage
                + 0.12 * color_strength
                - 0.03 * (mean_error / max(1.0, pitch_y_seed))
            )
            y_candidates.append(
                (
                    y_score,
                    y0,
                    matches,
                    center_alignment,
                    int(colors.size),
                    mean_error,
                )
            )

    if not y_candidates:
        return None
    y_score, y0_raw, matches_y, center_alignment, color_count, _ = max(
        y_candidates,
        key=lambda item: item[0],
    )

    # 只有「幾何上像一格」且該額外欄在真正 6-row 大路 band 內有
    # 足夠紅/藍像素、並且像素集中在格子中央時，才允許補右側欄位。
    # 這可保留被截掉最後邊界線的容錯，同時排除 Safari/Chrome/Dream
    # 畫面右側的純白餘邊。
    if (
        0.55 * pitch_x <= right_remainder <= 1.85 * pitch_x
        and grid_columns < ROAD_GENERIC_AUTO_COL_MAX
    ):
        extra_x1 = max(0, min(width - 1, int(round(float(best_run["x_last"])))))
        extra_x2 = max(
            extra_x1 + 1,
            min(width, int(round(float(best_run["x_last"]) + pitch_x))),
        )
        band_y1 = max(0, min(height - 1, int(math.floor(y0_raw))))
        band_y2 = max(
            band_y1 + 1,
            min(height, int(math.ceil(y0_raw + ROAD_GRID_ROWS * pitch_y_seed))),
        )
        extra_mask = red_blue[band_y1:band_y2, extra_x1:extra_x2]
        local_y, local_x = np.nonzero(extra_mask)
        right_inference_color_pixels = int(local_x.size)
        if local_x.size:
            normalized_x = local_x.astype(np.float64) / max(1.0, float(extra_x2 - extra_x1))
            right_inference_center_alignment = float(
                np.mean(
                    np.clip(
                        1.0 - np.abs(normalized_x - 0.5) / 0.5,
                        0.0,
                        1.0,
                    )
                )
            )
        minimum_extra_color = max(
            5,
            int(round(max(1.0, pitch_x * pitch_y_seed) * 0.025)),
        )
        if (
            right_inference_color_pixels >= minimum_extra_color
            and right_inference_center_alignment >= 0.30
        ):
            grid_columns += 1
            x2_raw = min(float(width), x0_raw + grid_columns * pitch_x)
            right_boundary_inferred = True

    x1 = max(0, min(width - 1, int(round(x0_raw))))
    x2 = max(
        x1 + 1,
        min(width, int(round(x2_raw))),
    )
    y1 = max(0, min(height - 1, int(round(y0_raw))))
    y2 = max(
        y1 + 1,
        min(
            height,
            int(round(y0_raw + ROAD_GRID_ROWS * pitch_y_seed)),
        ),
    )

    pitch_x_effective = (x2 - x1) / max(1.0, float(grid_columns))
    pitch_y_effective = (y2 - y1) / float(ROAD_GRID_ROWS)
    square_effective = max(
        0.0,
        1.0
        - abs(pitch_x_effective - pitch_y_effective)
        / max(1.0, pitch_x_effective, pitch_y_effective),
    )
    vertical_coverage = min(
        1.0,
        float(best_run["coverage"]),
    )
    # 頂線可在 crop 外，因此 6/7 條實際邊界已視為完整可用。
    horizontal_coverage = min(
        1.0,
        min(matches_y, ROAD_GRID_ROWS) / float(ROAD_GRID_ROWS),
    )
    coverage = float((vertical_coverage + horizontal_coverage) / 2.0)
    score = float(
        0.38 * vertical_coverage
        + 0.34 * horizontal_coverage
        + 0.18 * square_effective
        + 0.10 * min(1.0, center_alignment / 0.55)
    )

    if (
        vertical_coverage < 0.72
        or horizontal_coverage < 0.66
        or square_effective < 0.62
        or center_alignment < 0.30
    ):
        return None

    return {
        "x": x1,
        "y": y1,
        "width": x2 - x1,
        "height": y2 - y1,
        "score": score,
        "coverage": coverage,
        "square_cell_score": float(square_effective),
        "cell_pitch_x": float(pitch_x_effective),
        "cell_pitch_y": float(pitch_y_effective),
        "offset_x": x1,
        "offset_y": y1,
        "scale_x": (x2 - x1) / max(1.0, float(width)),
        "scale_y": (y2 - y1) / max(1.0, float(height)),
        "gain_x": 0.0,
        "gain_y": 0.0,
        "grid_columns": int(grid_columns),
        "line_grid": True,
        "line_vertical_peaks": int(len(best_run["matched"])),
        "line_horizontal_peaks": int(matches_y),
        "line_pitch_x": round(float(pitch_x), 6),
        "line_pitch_y": round(float(pitch_y_seed), 6),
        "line_vertical_coverage": round(float(vertical_coverage), 6),
        "line_horizontal_coverage": round(float(horizontal_coverage), 6),
        "line_color_center_alignment": round(float(center_alignment), 6),
        "line_color_pixel_count": int(color_count),
        "line_missing_verticals": int(best_run["missing_lines"]),
        "line_right_boundary_inferred": bool(right_boundary_inferred),
        "line_right_inference_color_pixels": int(right_inference_color_pixels),
        "line_right_inference_center_alignment": round(
            float(right_inference_center_alignment), 6
        ),
        "line_y_score": round(float(y_score), 6),
    }


def _repair_one_cell_reconstruction(
    cells: Sequence[Mapping[str, Any]],
    uncertain_cells: Sequence[Mapping[str, Any]],
    all_grid_cells: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """只允許唯一解的一格修復；多解或兩格以上缺失一律不修。"""
    if not ROAD_RECONSTRUCTION_ONE_CELL_REPAIR:
        return None

    occupied = {
        (int(item.get("column", -1)), int(item.get("row", -1)))
        for item in cells
    }
    candidates: List[Tuple[float, Dict[str, Any]]] = []

    def _add_candidate(source: Mapping[str, Any], bonus: float = 0.0) -> None:
        item = dict(source)
        position = (int(item.get("column", -1)), int(item.get("row", -1)))
        if position in occupied or position[0] < 0 or position[1] < 0:
            return
        red = float(item.get("red_pixels", 0) or 0)
        blue = float(item.get("blue_pixels", 0) or 0)
        minimum = max(1.0, float(item.get("minimum_color_pixels", 1) or 1))
        dominance = float(item.get("dominance", 1.0) or 1.0)
        evidence = max(red, blue) / minimum
        score = evidence + 0.15 * min(3.0, dominance) + bonus
        candidates.append((score, item))

    for item in uncertain_cells:
        _add_candidate(item, 0.35)

    # 低飽和壓縮後可能被標為 empty；只收有明顯殘留紅/藍像素的弱格。
    for item in all_grid_cells:
        if not bool(item.get("empty")):
            continue
        red = float(item.get("red_pixels", 0) or 0)
        blue = float(item.get("blue_pixels", 0) or 0)
        minimum = max(1.0, float(item.get("minimum_color_pixels", 1) or 1))
        dominance = float(item.get("dominance", 1.0) or 1.0)
        if max(red, blue) >= minimum * 0.28 and dominance >= 1.08:
            _add_candidate(item, 0.0)

    # 左上原點是大路重建的硬需求；但仍要求至少有少量紅/藍殘留，
    # 禁止在完全空白格憑空補出第一局。
    for item in all_grid_cells:
        if int(item.get("column", -1)) == 0 and int(item.get("row", -1)) == 0:
            red = float(item.get("red_pixels", 0) or 0)
            blue = float(item.get("blue_pixels", 0) or 0)
            minimum = max(1.0, float(item.get("minimum_color_pixels", 1) or 1))
            dominance = float(item.get("dominance", 1.0) or 1.0)
            if max(red, blue) >= minimum * 0.12 and dominance >= 1.03:
                _add_candidate(item, 0.50)
            break

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    successes: List[Dict[str, Any]] = []
    seen = set()
    for _, source in candidates[:ROAD_REPAIR_MAX_CANDIDATES]:
        position = (int(source.get("column", -1)), int(source.get("row", -1)))
        if position in seen:
            continue
        seen.add(position)
        for outcome in ("B", "P"):
            repaired_cell = dict(source)
            repaired_cell["outcome"] = outcome
            repaired_cell["uncertain"] = False
            repaired_cell["empty"] = False
            repaired_cell["repaired_from_weak_cell"] = True
            repaired_cell["confidence"] = max(
                0.30,
                min(0.49, float(source.get("confidence", 0.0) or 0.0)),
            )
            repaired_cells = [dict(item) for item in cells] + [repaired_cell]
            reconstruction = _reconstruct_big_road_order(
                repaired_cells,
                return_details=True,
                grid_rows=ROAD_GRID_ROWS,
            )
            if (
                bool(reconstruction.get("reconstructed_all"))
                and len(reconstruction.get("positions") or []) == len(repaired_cells)
                and int(reconstruction.get("solution_count", 0) or 0) == 1
            ):
                successes.append({
                    "cells": repaired_cells,
                    "reconstruction": reconstruction,
                    "repaired_cell": repaired_cell,
                })

    # 只有全域唯一解才接受，否則維持原本 fail-safe。
    if len(successes) != 1:
        return None
    return successes[0]


def _detect_fixed_grid_for_columns(
    crop: np.ndarray,
    grid_columns: int,
    *,
    profile: str = "",
    grid_bounds_override: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    image_height, image_width = crop.shape[:2]
    _, _, _, broad_union_mask = _broad_color_masks(crop)
    if grid_bounds_override:
        grid_bounds = dict(grid_bounds_override)
    else:
        grid_bounds = _effective_grid_bounds(
            crop,
            broad_union_mask,
            grid_columns=grid_columns,
            grid_rows=ROAD_GRID_ROWS,
        )
    red_mask, blue_mask, green_mask, _ = _color_masks(
        crop,
        calibration_bounds=grid_bounds,
    )
    classified = _classify_grid(
        crop,
        red_mask,
        blue_mask,
        green_mask,
        grid_bounds,
        grid_columns=grid_columns,
        grid_rows=ROAD_GRID_ROWS,
        profile=profile,
    )
    cells = list(classified["cells"])
    uncertain_cells = list(classified["uncertain_cells"])
    all_grid_cells = list(classified["all_grid_cells"])
    reconstruction = _reconstruct_big_road_order(
        cells, return_details=True, grid_rows=ROAD_GRID_ROWS
    )
    repaired_cell: Optional[Dict[str, Any]] = None
    if not bool(reconstruction.get("reconstructed_all")):
        repair = _repair_one_cell_reconstruction(
            cells,
            uncertain_cells,
            all_grid_cells,
        )
        if repair is not None:
            cells = list(repair["cells"])
            reconstruction = dict(repair["reconstruction"])
            repaired_cell = dict(repair["repaired_cell"])
            repaired_position = (
                int(repaired_cell.get("column", -1)),
                int(repaired_cell.get("row", -1)),
            )
            uncertain_cells = [
                item for item in uncertain_cells
                if (
                    int(item.get("column", -1)),
                    int(item.get("row", -1)),
                ) != repaired_position
            ]

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
    profile_key = str(profile or "").lower()
    generic_auto_profile = profile_key.startswith("mobile_auto_general")
    dg_feature_profile = profile_key.startswith("dg_feature_white_grid")
    if dg_feature_profile:
        minimum_alignment_score = max(0.28, ROAD_GRID_MIN_ALIGNMENT_SCORE - 0.14)
        minimum_coverage = 0.62
        minimum_square_score = max(0.34, ROAD_GENERIC_MIN_SQUARE_SCORE - 0.08)
        maximum_uncertain_ratio = max(ROAD_GRID_MAX_UNCERTAIN_RATIO, 0.22)
        minimum_median_confidence = max(0.28, ROAD_GRID_MIN_MEDIAN_CONFIDENCE - 0.12)
    elif generic_auto_profile:
        minimum_alignment_score = max(0.30, ROAD_GRID_MIN_ALIGNMENT_SCORE - 0.08)
        minimum_coverage = 0.70
        minimum_square_score = ROAD_GENERIC_MIN_SQUARE_SCORE
        maximum_uncertain_ratio = ROAD_GRID_MAX_UNCERTAIN_RATIO
        minimum_median_confidence = ROAD_GRID_MIN_MEDIAN_CONFIDENCE
    else:
        minimum_alignment_score = ROAD_GRID_MIN_ALIGNMENT_SCORE
        minimum_coverage = 0.80
        minimum_square_score = 0.72
        maximum_uncertain_ratio = ROAD_GRID_MAX_UNCERTAIN_RATIO
        minimum_median_confidence = ROAD_GRID_MIN_MEDIAN_CONFIDENCE
    alignment_ok = bool(
        float(grid_bounds["score"]) >= minimum_alignment_score
        and float(grid_bounds["coverage"]) >= minimum_coverage
        and float(grid_bounds.get("square_cell_score", 0.0)) >= minimum_square_score
    )
    quality_ok = bool(
        recognized_count >= ROAD_GRID_MIN_RECOGNIZED
        and uncertain_ratio <= maximum_uncertain_ratio
        and bool(reconstruction["reconstructed_all"])
        and alignment_ok
        and median_confidence >= minimum_median_confidence
    )
    fallback_reason = str(reconstruction.get("fallback_reason") or "")
    if not fallback_reason and not alignment_ok:
        fallback_reason = "grid_alignment_not_confident"
    if not fallback_reason and median_confidence < minimum_median_confidence:
        fallback_reason = "cell_color_confidence_too_low"
    if not fallback_reason and recognized_count < ROAD_GRID_MIN_RECOGNIZED:
        fallback_reason = "recognized_count_below_minimum"
    if not fallback_reason and uncertain_ratio > maximum_uncertain_ratio:
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
        "reconstruction_search_nodes": int(reconstruction.get("search_nodes", 0) or 0),
        "reconstruction_search_ms": float(reconstruction.get("search_ms", 0.0) or 0.0),
        "reconstruction_budget_exhausted": bool(reconstruction.get("budget_exhausted")),
        "reconstruction_repaired": bool(repaired_cell),
        "repaired_cell": repaired_cell or {},
        "partial_reconstruction": list(reconstruction["partial_positions"]),
        "fallback_preview_positions": list(reconstruction["fallback_preview"]),
        "fallback_reason": fallback_reason,
        "uncertain_cells": uncertain_cells,
        "count_is_confirmed": bool(quality_ok and uncertain_count == 0 and not repaired_cell),
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
    deadline: Optional[float] = None,
    cancel_event: Any = None,
) -> Dict[str, Any]:
    """固定六列、欄數自動；先測最接近幾何估計的少量欄數，可信即早停。"""
    if crop is None or crop.size == 0:
        raise ValueError("固定大路裁圖為空。")

    results: List[Dict[str, Any]] = []
    profile_key = str(profile or "").lower()
    generic_auto = (
        profile_key.startswith("mobile_auto_general")
        or profile_key.startswith("dg_feature_white_grid")
    )
    minimum_trials = 1
    hard_timeout_reached = False

    # DG 專用：白色路紙有清楚灰色格線時，直接從實際格線求 6×N，
    # 不再先用 crop 寬高比猜欄數。這條路徑失敗才回到原本候選搜尋。
    if profile_key.startswith("dg_feature_white_grid") and grid_columns is None:
        line_bounds = _dg_line_grid_bounds(crop)
        if line_bounds is not None:
            line_columns = int(line_bounds.get("grid_columns", 0) or 0)
            if ROAD_GRID_AUTO_COL_MIN <= line_columns <= ROAD_GENERIC_AUTO_COL_MAX:
                line_item = _detect_fixed_grid_for_columns(
                    crop,
                    line_columns,
                    profile=profile,
                    grid_bounds_override=line_bounds,
                )
                line_item["method"] = "dg_direct_gridline_6xN_v1"
                line_item["line_grid_used"] = True
                results.append(line_item)
                if bool(line_item.get("quality_ok")):
                    output = dict(line_item)
                    output["hard_timeout_reached"] = False
                    output["column_candidates"] = [{
                        "grid_columns": int(line_item.get("grid_columns", 0) or 0),
                        "quality_ok": True,
                        "recognized_count": int(line_item.get("recognized_count", 0) or 0),
                        "reconstructed_all": bool(line_item.get("reconstructed_all")),
                        "alignment_score": float(
                            dict(line_item.get("effective_grid") or {}).get("score", 0.0) or 0.0
                        ),
                        "square_cell_score": float(
                            dict(line_item.get("effective_grid") or {}).get("square_cell_score", 0.0) or 0.0
                        ),
                        "score": float(line_item.get("column_candidate_score", -9999.0) or -9999.0),
                        "fallback_reason": "",
                    }]
                    return output
    for columns in _column_candidates(crop, grid_columns, profile=profile):
        try:
            _deadline_guard(deadline, cancel_event, min_remaining=0.35)
            item = _detect_fixed_grid_for_columns(crop, columns, profile=profile)
        except TimeoutError:
            if results:
                hard_timeout_reached = True
                break
            raise
        results.append(item)
        effective = dict(item.get("effective_grid") or {})
        if (
            ROAD_FAST_EARLY_EXIT
            and len(results) >= minimum_trials
            and bool(item.get("quality_ok"))
            and int(item.get("recognized_count", 0) or 0) >= ROAD_FAST_MIN_RECOGNIZED
            and float(effective.get("score", 0.0) or 0.0) >= (0.46 if generic_auto else 0.52)
            and float(item.get("median_cell_confidence", 0.0) or 0.0) >= 0.46
        ):
            break

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
    output["hard_timeout_reached"] = bool(hard_timeout_reached)
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


def _db_select_six_row_band(
    items: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], float]:
    """從 DB 候選圓中選出真正的六列大路 band。

    不再固定刪除 crop 上方 30%。改由圓環自身半徑推估 row pitch，
    再找能容納最多圓、水平跨度最大的 6-row 區帶，因而可排除
    問路按鈕、瀏覽器 icon 與上方投注 UI。
    """
    if not items:
        return [], 0.0

    radii = [
        float(item.get("radius", 0.0) or 0.0)
        for item in items
        if float(item.get("radius", 0.0) or 0.0) > 0.0
    ]
    median_radius = _median_float(radii, 4.0)
    expected_pitch = max(7.0, median_radius * 2.55)
    pitch_y = _nearest_ring_pitch(
        items,
        axis="y",
        expected=expected_pitch,
    )
    pitch_y = max(6.0, float(pitch_y))

    best: Optional[Tuple[float, float, List[Dict[str, Any]]]] = None
    for source in items:
        source_y = float(source.get("cy", 0.0) or 0.0)
        # 真正第一列圓心附近作為 band 上緣；允許少量 AA / Hough 誤差。
        band_top = source_y - pitch_y * 0.38
        band_bottom = band_top + pitch_y * 6.15
        selected = [
            dict(item)
            for item in items
            if band_top <= float(item.get("cy", 0.0) or 0.0) <= band_bottom
        ]
        if not selected:
            continue
        xs = [float(item.get("cx", 0.0) or 0.0) for item in selected]
        span_x = max(xs) - min(xs) if len(xs) >= 2 else 0.0
        score = (
            len(selected) * 3.0
            + min(12.0, span_x / max(1.0, pitch_y)) * 0.6
        )
        if best is None or score > best[0]:
            best = (score, band_top, selected)

    if best is None:
        return [dict(item) for item in items], pitch_y

    selected = best[2]
    return selected, pitch_y


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


def _repair_one_ring_grid_cell(
    cells: Sequence[Mapping[str, Any]],
    *,
    red_mask: np.ndarray,
    blue_mask: np.ndarray,
    green_mask: np.ndarray,
    pitch_x: float,
    pitch_y: float,
    origin_x: float,
    origin_y: float,
) -> Optional[Dict[str, Any]]:
    """DB 圓環路只補一個有真實像素證據、且可形成唯一大路解的漏格。"""
    if not ROAD_RECONSTRUCTION_ONE_CELL_REPAIR or not cells:
        return None

    height, width = red_mask.shape[:2]
    occupied = {
        (int(item.get("column", -1)), int(item.get("row", -1)))
        for item in cells
    }
    max_column = max((position[0] for position in occupied), default=-1)
    if max_column < 0:
        return None

    radii = [
        float(item.get("radius", 0.0) or 0.0)
        for item in cells
        if float(item.get("radius", 0.0) or 0.0) > 0.0
    ]
    radius = max(
        3.0,
        _median_float(
            radii,
            max(3.0, min(float(pitch_x), float(pitch_y)) * 0.32),
        ),
    )
    yy, xx = np.ogrid[:height, :width]
    successes: List[Dict[str, Any]] = []

    for column in range(max_column + 2):
        for row in range(ROAD_GRID_ROWS):
            position = (column, row)
            if position in occupied:
                continue

            neighbors = sum(
                neighbor in occupied
                for neighbor in (
                    (column - 1, row),
                    (column + 1, row),
                    (column, row - 1),
                    (column, row + 1),
                )
            )
            if neighbors <= 0:
                continue

            cx = float(origin_x + column * pitch_x)
            cy = float(origin_y + row * pitch_y)
            if not (0 <= cx < width and 0 <= cy < height):
                continue

            outer = max(3.0, radius + 3.0)
            inner = max(1.0, radius - 4.0)
            distance2 = (xx - cx) ** 2 + (yy - cy) ** 2
            annulus = (distance2 <= outer ** 2) & (distance2 >= inner ** 2)
            disk = distance2 <= outer ** 2
            red_pixels = int(red_mask[annulus].sum())
            blue_pixels = int(blue_mask[annulus].sum())
            green_pixels = int(green_mask[disk].sum())
            dominant = max(red_pixels, blue_pixels)
            secondary = min(red_pixels, blue_pixels)
            dominance = dominant / max(1.0, float(secondary))

            # 必須真的看到殘留紅/藍像素；不能只靠路規則憑空補牌。
            if dominant < max(4, int(round(MOBILE_RING_MIN_COLOR_PIXELS * 0.45))):
                continue
            if dominance < 1.08:
                continue

            outcome = "B" if red_pixels > blue_pixels else "P"
            candidate = {
                "cx": round(cx, 3),
                "cy": round(cy, 3),
                "radius": int(round(radius)),
                "red_pixels": red_pixels,
                "blue_pixels": blue_pixels,
                "green_pixels": green_pixels,
                "color_dominance": round(dominance, 6),
                "outcome": outcome,
                "tie_count": 0,
                "confidence": min(
                    0.55,
                    0.28 + 0.12 * min(2.0, dominance - 1.0),
                ),
                "uncertain": False,
                "column": column,
                "row": row,
                "fit_error_x": 0.0,
                "fit_error_y": 0.0,
                "repaired_from_ring_pixels": True,
            }
            repaired_cells = [dict(item) for item in cells] + [candidate]
            reconstruction = _reconstruct_big_road_order(
                repaired_cells,
                return_details=True,
                grid_rows=ROAD_GRID_ROWS,
            )
            if (
                bool(reconstruction.get("reconstructed_all"))
                and int(reconstruction.get("solution_count", 0) or 0) == 1
                and len(reconstruction.get("positions") or []) == len(repaired_cells)
            ):
                successes.append(
                    {
                        "cells": repaired_cells,
                        "reconstruction": reconstruction,
                        "repaired_cell": candidate,
                    }
                )

    # 只接受全域唯一的一個補洞位置／顏色。
    if len(successes) != 1:
        return None
    return successes[0]


def _detect_mobile_ring_grid(
    crop: np.ndarray,
    *,
    profile: str,
    deadline: Optional[float] = None,
    cancel_event: Any = None,
) -> Dict[str, Any]:
    """MT／DB 手機全畫面專用彩色圓環大路偵測。

    只由新增的 MT／DB 手機候選呼叫。DG 仍完整使用原本的
    ``_detect_fixed_grid``，不會進入本函式。
    """
    if crop is None or crop.size == 0:
        raise ValueError("MT/DB 手機大路裁圖為空。")
    _deadline_guard(deadline, cancel_event, min_remaining=0.30)

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

    _deadline_guard(deadline, cancel_event, min_remaining=0.20)
    red_mask, blue_mask, green_mask, _ = _color_masks(crop)
    yy, xx = np.ogrid[:image_height, :image_width]
    colored: List[Dict[str, Any]] = []
    uncertain_cells: List[Dict[str, Any]] = []

    raw_circles = [] if circles is None else np.round(circles[0]).astype(int)
    for circle_index, raw_circle in enumerate(raw_circles):
        if (circle_index & 15) == 0:
            _deadline_guard(deadline, cancel_event, min_remaining=0.12)
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
            int(round(math.pi * radius * radius * 0.065)),
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

    if profile_key.startswith("db_feature_dark_ring"):
        # DB feature locator 為了容錯會在大路上方保留一些 padding；
        # 這裡移除上方問路/投注 UI 的彩色圓形按鈕，避免 origin_y 被拉到 UI。
        lower_cut = image_height * 0.30
        road_colored = [
            item for item in colored
            if float(item.get("cy", 0.0) or 0.0) >= lower_cut
        ]
        if len(road_colored) >= ROAD_GRID_MIN_RECOGNIZED:
            colored = road_colored
            uncertain_cells = [
                item for item in uncertain_cells
                if float(item.get("cy", image_height) or image_height) >= lower_cut
            ]

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
    repaired_cell: Optional[Dict[str, Any]] = None
    if (
        not bool(reconstruction.get("reconstructed_all"))
        and profile_key.startswith("db_")
    ):
        repair = _repair_one_ring_grid_cell(
            cells,
            red_mask=red_mask,
            blue_mask=blue_mask,
            green_mask=green_mask,
            pitch_x=pitch_x,
            pitch_y=pitch_y,
            origin_x=origin_x,
            origin_y=origin_y,
        )
        if repair is not None:
            cells = list(repair["cells"])
            reconstruction = dict(repair["reconstruction"])
            repaired_cell = dict(repair["repaired_cell"])
            repaired_cx = float(repaired_cell.get("cx", 0.0) or 0.0)
            repaired_cy = float(repaired_cell.get("cy", 0.0) or 0.0)
            uncertain_cells = [
                item for item in uncertain_cells
                if (
                    abs(float(item.get("cx", -9999.0) or -9999.0) - repaired_cx)
                    > max(3.0, pitch_x * 0.40)
                    or abs(float(item.get("cy", -9999.0) or -9999.0) - repaired_cy)
                    > max(3.0, pitch_y * 0.40)
                )
            ]

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
    effective_grid = dict(result.get("effective_grid") or {})
    alignment = float(effective_grid.get("score", 0.0) or 0.0)
    geometry = float(effective_grid.get("square_cell_score", 0.0) or 0.0)
    median_confidence = float(result.get("median_cell_confidence", 0.0) or 0.0)
    unknown_ratio = unknown / max(1.0, float(recognized + unknown))
    preference_bonus = math.copysign(
        min(18.0, abs(float(preference)) * 0.30),
        float(preference),
    )
    return (
        recognized * 3.0
        - unknown * 4.0
        - unknown_ratio * 45.0
        - noise * 0.04
        + preference_bonus
        + quality_bonus
        + reconstruction_bonus
        + alignment * 28.0
        + geometry * 10.0
        + median_confidence * 25.0
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
    deadline: Optional[float] = None,
    cancel_event: Any = None,
) -> Dict[str, Any]:
    crop, pixels = _crop(image, roi)
    started = time.perf_counter()

    if ring_grid:
        result = _detect_mobile_ring_grid(
            crop,
            profile=layout_profile or name,
            deadline=deadline,
            cancel_event=cancel_event,
        )
    elif fixed_grid:
        result = _detect_fixed_grid(
            crop,
            grid_columns=grid_columns,
            profile=layout_profile or name,
            deadline=deadline,
            cancel_event=cancel_event,
        )
    elif ROAD_USE_YOLO and _get_yolo_model() is not None:
        result = _detect_yolo(crop)
    else:
        result = analyze_baccarat_array_detailed(
            crop,
            deadline=deadline,
            cancel_event=cancel_event,
        )

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


def _strong_acceptable(result: Mapping[str, Any]) -> bool:
    if not _acceptable(result):
        return False
    recognized = int(result.get("recognized_count", 0) or 0)
    unknown = int(result.get("unknown_candidates", result.get("uncertain_count", 0)) or 0)
    ratio = unknown / max(1, recognized + unknown)
    effective = dict(result.get("effective_grid") or {})
    alignment = float(effective.get("score", 0.0) or 0.0)
    median_confidence = float(result.get("median_cell_confidence", 0.0) or 0.0)
    fixed = bool(result.get("fixed_grid")) or str(result.get("method") or "").startswith("fixed_")
    layout_profile = str(result.get("layout_profile") or "").lower()
    venue_feature = layout_profile in {"dg_feature_white_grid", "db_feature_dark_ring"}
    max_ratio = 0.18 if venue_feature else min(0.10, ROAD_FAST_MAX_UNKNOWN_RATIO)
    min_alignment = 0.32 if layout_profile == "dg_feature_white_grid" else 0.48
    min_confidence = 0.30 if layout_profile == "dg_feature_white_grid" else 0.46
    return bool(
        recognized >= max(8 if venue_feature else 10, ROAD_FAST_MIN_RECOGNIZED)
        and ratio <= max_ratio
        and bool(result.get("reconstructed_all", True))
        and (not fixed or alignment >= min_alignment)
        and (not fixed or median_confidence >= min_confidence or bool(result.get("ring_grid")))
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
    """Dream/DG 手機全畫面：用白色路紙 + 紅藍環 + 格線證據判斷，不綁單一瀏覽器比例。"""
    if not DREAM_COMPACT_MOBILE_PROFILE_ENABLED or image is None or image.size == 0:
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

    # 只測 3 個低成本 Y 位移；瀏覽器網址列 / 底部導覽列改變時仍可命中。
    x, y, roi_w, roi_h = DREAM_COMPACT_MOBILE_BIG_ROAD_ROI
    for dy in (0.0, -DREAM_COMPACT_MOBILE_PROFILE_SEARCH_Y, DREAM_COMPACT_MOBILE_PROFILE_SEARCH_Y):
        shifted = (
            x,
            max(0.0, min(1.0 - roi_h, y + float(dy))),
            roi_w,
            roi_h,
        )
        sample, _ = _crop(image, shifted)
        if sample.size == 0 or min(sample.shape[:2]) < 24:
            continue

        pixels = sample.astype(np.int16, copy=False)
        channel_min = np.min(pixels, axis=2)
        channel_span = np.max(pixels, axis=2) - channel_min
        bright_neutral = (channel_min >= 155) & (channel_span <= 96)
        bright_fraction = float(np.mean(bright_neutral))
        if bright_fraction < DREAM_COMPACT_MOBILE_MIN_BRIGHT_FRACTION:
            continue

        hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        red_blue = (
            (saturation >= 16)
            & (value >= 25)
            & (
                (hue <= 32)
                | (hue >= 148)
                | ((hue >= 80) & (hue <= 160))
            )
        )
        color_fraction = float(np.mean(red_blue))
        if color_fraction < 0.0010:
            continue

        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        periodicity = _grid_periodicity_score(gray)
        if periodicity >= 0.14:
            return True

    return False


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


def _deadline_guard(
    deadline: Optional[float],
    cancel_event: Any = None,
    *,
    min_remaining: float = 0.0,
) -> None:
    if cancel_event is not None and bool(getattr(cancel_event, "is_set", lambda: False)()):
        raise TimeoutError("大路辨識已取消。")
    if deadline is not None and float(deadline) - time.perf_counter() <= float(min_remaining):
        raise TimeoutError("大路辨識已達處理時限。")


def _grid_periodicity_score(gray: np.ndarray) -> float:
    if gray is None or gray.size == 0 or min(gray.shape[:2]) < 12:
        return 0.0
    edge_x = np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)), axis=0)
    edge_y = np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)), axis=1)

    def _boundary_score(projection: np.ndarray, divisions: int) -> float:
        if projection.size < divisions + 1:
            return 0.0
        denominator = max(1e-6, float(np.percentile(projection, 92)))
        best = 0.0
        for trim in (0.0, 0.025, 0.05, 0.075):
            start = trim * projection.size
            end = (1.0 - trim) * projection.size
            samples: List[float] = []
            for position in np.linspace(start, end, divisions + 1):
                center = int(round(position))
                left = max(0, center - 2)
                right = min(projection.size, center + 3)
                if right > left:
                    samples.append(min(1.0, float(np.max(projection[left:right])) / denominator))
            if samples:
                best = max(best, float(np.mean(samples)))
        return best

    row_score = _boundary_score(edge_y, ROAD_GRID_ROWS)
    aspect = gray.shape[1] / max(1.0, float(gray.shape[0]))
    col_scores = []
    for cell_ratio in (0.72, 0.88, 1.0, 1.20, 1.40):
        columns = int(round(aspect * ROAD_GRID_ROWS / cell_ratio))
        if 5 <= columns <= ROAD_GENERIC_AUTO_COL_MAX:
            col_scores.append(_boundary_score(edge_x, columns))
    column_score = max(col_scores or [0.0])
    return float(0.56 * row_score + 0.44 * column_score)


def _general_road_candidates(
    image: np.ndarray,
    *,
    deadline: Optional[float] = None,
    cancel_event: Any = None,
) -> List[Tuple[Tuple[float, float, float, float], float]]:
    """跨直/橫式、白/米黃/深色背景，以路紙材質 + 格線週期 + 紅藍環排序候選。"""
    if not MOBILE_AUTO_FOCUS_ENABLED or image is None or image.size == 0:
        return []
    height, width = image.shape[:2]
    if min(height, width) < 220:
        return []

    scale = min(1.0, MOBILE_AUTO_FOCUS_PREVIEW_SIDE / max(height, width))
    preview = (
        cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        if scale < 0.999
        else image
    )
    ph, pw = preview.shape[:2]
    hsv = cv2.cvtColor(preview, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    pixels = preview.astype(np.int16, copy=False)
    channel_min = np.min(pixels, axis=2)
    channel_span = np.max(pixels, axis=2) - channel_min

    bright_floor = int(np.clip(np.percentile(channel_min, 70) * 0.86, 135, 205))
    bright_neutral = (channel_min >= bright_floor) & (channel_span <= 82)
    warm_paper = (
        (val >= 105)
        & (sat >= 5)
        & (sat <= 115)
        & (hue >= 3)
        & (hue <= 38)
    )
    dark_neutral = (
        (val >= 24)
        & (val <= 155)
        & (sat <= 58)
    )
    paper_masks = [
        bright_neutral.astype(np.uint8),
        warm_paper.astype(np.uint8),
        dark_neutral.astype(np.uint8),
    ]

    red_blue = (
        (sat >= 20)
        & (val >= 28)
        & (
            (hue <= 30)
            | (hue >= 150)
            | ((hue >= 82) & (hue <= 158))
        )
    ).astype(np.uint8)

    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    edge_x_full = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    edge_y_full = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    edge_mag = np.clip((edge_x_full + edge_y_full) / 160.0, 0.0, 1.0).astype(np.float32)

    paper_union = np.maximum.reduce(paper_masks).astype(np.uint8)
    paper_integral = cv2.integral(paper_union, sdepth=cv2.CV_32S)
    color_integral = cv2.integral(red_blue, sdepth=cv2.CV_32S)
    edge_integral = cv2.integral(edge_mag, sdepth=cv2.CV_64F)

    def _rect_mean(integral: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
        area = max(1, (x2 - x1) * (y2 - y1))
        total = integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
        return float(total) / float(area)

    boxes: List[Tuple[int, int, int, int, float]] = []
    image_area = float(ph * pw)

    # 先從三種路紙材質的連通區塊提出候選。
    for source_mask in paper_masks:
        _deadline_guard(deadline, cancel_event, min_remaining=0.15)
        mask = (source_mask * 255).astype(np.uint8)
        close_w = max(7, int(round(pw * 0.025)))
        close_h = max(3, int(round(ph * 0.006)))
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h)),
            iterations=1,
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area_ratio = (w * h) / max(1.0, image_area)
            aspect = w / max(1.0, float(h))
            if (
                MOBILE_AUTO_FOCUS_MIN_AREA_RATIO <= area_ratio <= MOBILE_AUTO_FOCUS_MAX_AREA_RATIO
                and aspect >= MOBILE_AUTO_FOCUS_MIN_ASPECT
                and w >= pw * 0.18
                and h >= ph * 0.025
            ):
                x1 = max(0, x - int(round(w * 0.03)))
                x2 = min(pw, x + w + int(round(w * 0.03)))
                y1 = max(0, y - int(round(h * 0.12)))
                y2 = min(ph, y + h + int(round(h * 0.12)))
                boxes.append((x1, y1, x2, y2, 1.0))

    # 紙色被 UI 切碎時，用低成本滑窗做最後保險；只用 integral image 粗排。
    coarse: List[Tuple[float, Tuple[int, int, int, int]]] = []
    for height_fraction in (0.07, 0.10, 0.14, 0.20, 0.28):
        _deadline_guard(deadline, cancel_event, min_remaining=0.15)
        box_h = max(24, int(round(ph * height_fraction)))
        step_y = max(12, int(round(box_h * 0.55)))
        for width_fraction in (0.44, 0.64, 0.84, 1.0):
            box_w = max(48, int(round(pw * width_fraction)))
            x_positions = sorted(set((0, max(0, (pw - box_w) // 2), max(0, pw - box_w))))
            for y1 in range(0, max(1, ph - box_h + 1), step_y):
                y2 = min(ph, y1 + box_h)
                if y2 - y1 < 20:
                    continue
                for x1 in x_positions:
                    x2 = min(pw, x1 + box_w)
                    aspect = (x2 - x1) / max(1.0, float(y2 - y1))
                    if aspect < MOBILE_AUTO_FOCUS_MIN_ASPECT:
                        continue
                    paper_mean = _rect_mean(paper_integral, x1, y1, x2, y2)
                    color_mean = _rect_mean(color_integral, x1, y1, x2, y2)
                    edge_mean = _rect_mean(edge_integral, x1, y1, x2, y2)
                    if color_mean < 0.0005 or edge_mean < 0.035:
                        continue
                    coarse_score = (
                        1.0 * paper_mean
                        + 1.5 * min(1.0, color_mean / 0.030)
                        + 1.2 * min(1.0, edge_mean / 0.18)
                    )
                    coarse.append((coarse_score, (x1, y1, x2, y2)))
    coarse.sort(key=lambda item: item[0], reverse=True)
    for score, box in coarse[:18]:
        boxes.append((*box, score))

    scored: List[Tuple[Tuple[float, float, float, float], float]] = []
    seen_boxes = set()
    for index, (x1, y1, x2, y2, source_score) in enumerate(boxes):
        if (index & 7) == 0:
            _deadline_guard(deadline, cancel_event, min_remaining=0.10)
        key = (int(x1 // 4), int(y1 // 4), int(x2 // 4), int(y2 // 4))
        if key in seen_boxes:
            continue
        seen_boxes.add(key)
        if x2 <= x1 or y2 <= y1:
            continue
        area_ratio = ((x2 - x1) * (y2 - y1)) / max(1.0, image_area)
        if not (MOBILE_AUTO_FOCUS_MIN_AREA_RATIO <= area_ratio <= MOBILE_AUTO_FOCUS_MAX_AREA_RATIO):
            continue

        paper_fraction = _rect_mean(paper_integral, x1, y1, x2, y2)
        color_fraction = _rect_mean(color_integral, x1, y1, x2, y2)
        if color_fraction < 0.0006:
            continue
        roi_gray = gray[y1:y2, x1:x2]
        periodicity = _grid_periodicity_score(roi_gray)
        if periodicity < 0.14:
            continue

        aspect = (x2 - x1) / max(1.0, float(y2 - y1))
        geometry_prior = min(1.0, max(0.0, (aspect - 1.0) / 3.0))
        score = (
            2.8 * periodicity
            + 1.5 * min(1.0, color_fraction / 0.035)
            + 0.75 * paper_fraction
            + 0.25 * geometry_prior
            + 0.10 * min(1.0, source_score)
        )
        roi = (
            x1 / float(pw),
            y1 / float(ph),
            (x2 - x1) / float(pw),
            (y2 - y1) / float(ph),
        )
        scored.append((roi, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    selected: List[Tuple[Tuple[float, float, float, float], float]] = []
    for roi, score in scored:
        x, y, w, h = roi
        duplicate = False
        for prior, _ in selected:
            px, py, pw0, ph0 = prior
            ix1, iy1 = max(x, px), max(y, py)
            ix2, iy2 = min(x + w, px + pw0), min(y + h, py + ph0)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = w * h + pw0 * ph0 - inter
            if union > 0 and inter / union >= 0.58:
                duplicate = True
                break
        if not duplicate:
            selected.append((roi, score))
        if len(selected) >= MOBILE_AUTO_FOCUS_MAX_CANDIDATES:
            break
    return selected



def _roi_iou(
    first: Tuple[float, float, float, float],
    second: Tuple[float, float, float, float],
) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0.0 else 0.0


def _dg_white_grid_feature_candidates(
    image: np.ndarray,
    *,
    deadline: Optional[float] = None,
    cancel_event: Any = None,
) -> List[Tuple[Tuple[float, float, float, float], float]]:
    """DG 全畫面白色格線搜尋：不限制手機/電腦，也不依賴固定 Y 軸。"""
    _deadline_guard(deadline, cancel_event, min_remaining=0.25)
    height, width = image.shape[:2]
    scale = min(1.0, 760.0 / max(height, width))
    preview = (
        cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        if scale < 0.999
        else image
    )
    ph, pw = preview.shape[:2]
    pixels = preview.astype(np.int16, copy=False)
    channel_min = np.min(pixels, axis=2)
    channel_span = np.max(pixels, axis=2) - channel_min
    bright_floor = int(np.clip(np.percentile(channel_min, 78) * 0.90, 145, 190))
    white_bool = (channel_min >= bright_floor) & (channel_span <= 96)
    white = white_bool.astype(np.uint8) * 255

    hsv = cv2.cvtColor(preview, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    color_valid = (sat >= 16) & (val >= 25)
    red_bool = color_valid & ((hue <= 32) | (hue >= 148))
    blue_bool = color_valid & (hue >= 82) & (hue <= 160)
    red_blue = red_bool | blue_bool

    white_integral = cv2.integral(white_bool.astype(np.uint8), sdepth=cv2.CV_32S)
    red_integral = cv2.integral(red_bool.astype(np.uint8), sdepth=cv2.CV_32S)
    blue_integral = cv2.integral(blue_bool.astype(np.uint8), sdepth=cv2.CV_32S)
    color_integral = cv2.integral(red_blue.astype(np.uint8), sdepth=cv2.CV_32S)

    def _rect_mean(integral: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
        area = max(1, (x2 - x1) * (y2 - y1))
        total = integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
        return float(total) / float(area)

    # 先由白色路紙連通區提出候選；不再砍掉畫面上半部。
    close_w = max(7, int(round(pw * 0.018)))
    close_h = max(3, int(round(ph * 0.005)))
    white_closed = cv2.morphologyEx(
        white,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h)),
        iterations=1,
    )
    contours, _ = cv2.findContours(white_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate_boxes: List[Tuple[int, int, int, int, float]] = []

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:18]:
        x, y, w, h = cv2.boundingRect(contour)
        if (
            w < pw * 0.42
            or h < max(24, ph * 0.035)
            or (w * h) < ph * pw * 0.012
        ):
            continue
        candidate_boxes.append((x, y, x + w, y + h, 1.0))

    # Fast path：大型白色路紙一旦找到，直接在其上半部鎖 6×N 真實格線。
    # 這比後面的多尺度 coarse-window 搜尋快非常多，也能讓 LINE 先在 6~9 秒內完成。
    fast_scored: List[Tuple[Tuple[float, float, float, float], float]] = []
    for box_index, (x1, y1, x2, y2, source_score) in enumerate(candidate_boxes[:8]):
        if (box_index & 3) == 0:
            _deadline_guard(deadline, cancel_event, min_remaining=0.18)
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w < pw * 0.50 or box_h < ph * 0.055:
            continue

        # DG/Dream 大路位於白色路紙的上半部偏右；只測少量寬鬆子區，
        # 讓 _dg_line_grid_bounds() 自己排除左珠盤與下三路。
        for x_fraction in (0.12, 0.20, 0.28):
            sx1 = max(0, int(round(x1 + box_w * x_fraction)))
            sx2 = min(pw, x2)
            if sx2 - sx1 < 120:
                continue
            for height_fraction in (0.72, 0.82):
                sy1 = max(0, y1)
                sy2 = min(ph, int(round(y1 + box_h * height_fraction)))
                if sy2 - sy1 < 48:
                    continue
                sub = preview[sy1:sy2, sx1:sx2]
                bounds = _dg_line_grid_bounds(sub)
                if bounds is None:
                    continue

                columns = int(bounds.get("grid_columns", 0) or 0)
                coverage = float(bounds.get("coverage", 0.0) or 0.0)
                square = float(bounds.get("square_cell_score", 0.0) or 0.0)
                center_alignment = float(
                    bounds.get("line_color_center_alignment", 0.0) or 0.0
                )
                if (
                    columns < 10
                    or coverage < 0.78
                    or square < 0.76
                    or center_alignment < 0.40
                ):
                    continue

                gx1 = sx1 + int(bounds["x"])
                gy1 = sy1 + int(bounds["y"])
                gx2 = gx1 + int(bounds["width"])
                gy2 = gy1 + int(bounds["height"])
                gx1 = max(0, min(pw - 1, gx1))
                gy1 = max(0, min(ph - 1, gy1))
                gx2 = max(gx1 + 1, min(pw, gx2))
                gy2 = max(gy1 + 1, min(ph, gy2))

                red_fraction = _rect_mean(red_integral, gx1, gy1, gx2, gy2)
                blue_fraction = _rect_mean(blue_integral, gx1, gy1, gx2, gy2)
                white_fraction = _rect_mean(white_integral, gx1, gy1, gx2, gy2)
                if (
                    red_fraction < 0.00035
                    or blue_fraction < 0.00035
                    or white_fraction < 0.42
                ):
                    continue

                aspect = (gx2 - gx1) / max(1.0, float(gy2 - gy1))
                if aspect < 2.4:
                    continue
                score = (
                    4.0
                    + 1.25 * coverage
                    + 1.10 * square
                    + 0.90 * center_alignment
                    + 0.35 * min(1.0, columns / 20.0)
                    + 0.20 * min(1.0, source_score)
                )
                fast_scored.append(
                    (
                        (
                            gx1 / float(pw),
                            gy1 / float(ph),
                            (gx2 - gx1) / float(pw),
                            (gy2 - gy1) / float(ph),
                        ),
                        score,
                    )
                )

    if fast_scored:
        fast_scored.sort(key=lambda item: item[1], reverse=True)
        selected_fast: List[Tuple[Tuple[float, float, float, float], float]] = []
        for roi, score in fast_scored:
            if any(_roi_iou(roi, prior) >= 0.72 for prior, _ in selected_fast):
                continue
            selected_fast.append((roi, score))
            if len(selected_fast) >= DG_FEATURE_MAX_CANDIDATES:
                break
        if selected_fast:
            return selected_fast

    # Contour 被格線切碎時，用全畫面 integral-image coarse scan 補候選。
    coarse: List[Tuple[float, Tuple[int, int, int, int]]] = []
    for height_fraction in (0.065, 0.085, 0.11, 0.145, 0.19):
        _deadline_guard(deadline, cancel_event, min_remaining=0.15)
        box_h = max(28, int(round(ph * height_fraction)))
        step_y = max(14, int(round(box_h * 0.48)))
        for width_fraction in (0.46, 0.60, 0.74, 0.88):
            box_w = max(100, int(round(pw * width_fraction)))
            step_x = max(24, int(round(box_w * 0.34)))
            for y1 in range(0, max(1, ph - box_h + 1), step_y):
                y2 = min(ph, y1 + box_h)
                for x1 in range(0, max(1, pw - box_w + 1), step_x):
                    x2 = min(pw, x1 + box_w)
                    aspect = (x2 - x1) / max(1.0, float(y2 - y1))
                    if aspect < 2.1:
                        continue
                    white_fraction = _rect_mean(white_integral, x1, y1, x2, y2)
                    color_fraction = _rect_mean(color_integral, x1, y1, x2, y2)
                    if white_fraction < 0.48 or color_fraction < 0.0012:
                        continue
                    score = 1.5 * white_fraction + 1.8 * min(1.0, color_fraction / 0.035)
                    coarse.append((score, (x1, y1, x2, y2)))
    coarse.sort(key=lambda item: item[0], reverse=True)
    for score, box in coarse[:16]:
        candidate_boxes.append((*box, score))

    scored: List[Tuple[Tuple[float, float, float, float], float]] = []
    seen = set()
    for box_index, (x1, y1, x2, y2, source_score) in enumerate(candidate_boxes):
        if (box_index & 7) == 0:
            _deadline_guard(deadline, cancel_event, min_remaining=0.10)
        key = (x1 // 5, y1 // 5, x2 // 5, y2 // 5)
        if key in seen:
            continue
        seen.add(key)
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 0 or box_h <= 0:
            continue

        # 白色面板可能包含珠盤路/下三路/統計列；在面板內再搜尋 6-row 大路子區。
        for y_offset_fraction in (0.0, 0.04, 0.08, 0.14, 0.20):
            for height_fraction in (0.34, 0.42, 0.50, 0.58, 0.66):
                ry1 = int(round(y1 + box_h * y_offset_fraction))
                ry2 = int(round(ry1 + box_h * height_fraction))
                if ry2 > y2:
                    continue
                for x_fraction, width_fraction in (
                    (0.18, 0.80),
                    (0.24, 0.74),
                    (0.28, 0.70),
                    (0.32, 0.66),
                    (0.05, 0.92),
                ):
                    rx1 = max(0, int(round(x1 + box_w * x_fraction)))
                    rx2 = min(pw, int(round(x1 + box_w * min(0.995, x_fraction + width_fraction))))
                    if rx2 - rx1 < 90 or ry2 - ry1 < 28:
                        continue
                    aspect = (rx2 - rx1) / max(1.0, float(ry2 - ry1))
                    if aspect < 2.15:
                        continue
                    color_fraction = _rect_mean(color_integral, rx1, ry1, rx2, ry2)
                    white_fraction = _rect_mean(white_integral, rx1, ry1, rx2, ry2)
                    if color_fraction < 0.0010 or white_fraction < 0.44:
                        continue
                    roi_gray = cv2.cvtColor(preview[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2GRAY)
                    periodicity = _grid_periodicity_score(roi_gray)
                    if periodicity < 0.18:
                        continue
                    split_y = ry1 + max(1, int(round((ry2 - ry1) * 0.68)))
                    upper_color_fraction = _rect_mean(
                        color_integral, rx1, ry1, rx2, min(ry2, split_y)
                    )
                    width_score = min(1.0, (rx2 - rx1) / max(1.0, pw * 0.64))
                    score = (
                        3.4 * periodicity
                        + 1.30 * min(1.0, color_fraction / 0.035)
                        + 0.72 * white_fraction
                        + 0.42 * min(1.0, upper_color_fraction / 0.28)
                        + 0.28 * width_score
                        + 0.08 * min(1.0, source_score)
                    )
                    scored.append(
                        (
                            (
                                rx1 / float(pw),
                                ry1 / float(ph),
                                (rx2 - rx1) / float(pw),
                                (ry2 - ry1) / float(ph),
                            ),
                            score,
                        )
                    )

    scored.sort(key=lambda item: item[1], reverse=True)
    selected: List[Tuple[Tuple[float, float, float, float], float]] = []
    for roi, score in scored:
        if any(_roi_iou(roi, prior) >= 0.66 for prior, _ in selected):
            continue
        selected.append((roi, score))
        if len(selected) >= DG_FEATURE_MAX_CANDIDATES:
            break
    return selected


def _estimate_ring_pitch_free(
    items: Sequence[Tuple[float, float, float]],
    *,
    axis: str,
    median_radius: float,
) -> float:
    """不依賴 crop 高度估計 DB lattice pitch。"""
    diffs: List[float] = []
    orthogonal_limit = max(4.0, median_radius * 1.9)
    minimum = max(5.0, median_radius * 1.45)
    maximum = max(minimum + 1.0, median_radius * 5.0)
    for index, first in enumerate(items):
        for second in items[index + 1:]:
            dx = abs(float(first[0]) - float(second[0]))
            dy = abs(float(first[1]) - float(second[1]))
            if axis == "x":
                if dy > orthogonal_limit:
                    continue
                distance = dx
            else:
                if dx > orthogonal_limit:
                    continue
                distance = dy
            if minimum <= distance <= maximum:
                diffs.append(distance)
    if not diffs:
        return max(8.0, median_radius * 2.7)
    diffs.sort()
    # 最近鄰群通常就是一格 pitch；取較小一半再用中位數抗離群。
    take = diffs[: max(1, min(len(diffs), max(4, len(diffs) // 2)))]
    median = float(np.median(np.asarray(take, dtype=np.float64)))
    trimmed = [value for value in take if abs(value - median) <= max(2.0, median * 0.22)]
    return _median_float(trimmed, median)


def _db_dark_ring_feature_candidates(
    image: np.ndarray,
    *,
    deadline: Optional[float] = None,
    cancel_event: Any = None,
) -> List[Tuple[Tuple[float, float, float, float], float]]:
    """DB 全畫面深色紅藍環搜尋：手機/電腦皆不依賴固定 Y 軸。"""
    _deadline_guard(deadline, cancel_event, min_remaining=0.30)
    height, width = image.shape[:2]
    scale = min(1.0, 760.0 / max(height, width))
    preview = (
        cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        if scale < 0.999
        else image
    )
    ph, pw = preview.shape[:2]
    hsv = cv2.cvtColor(preview, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    # DB 路紙背景很暗，低彩度像素的 hue 不可靠；粗搜尋只看高彩度真紅/真藍。
    red = (
        (sat >= 52)
        & (val >= 30)
        & ((hue <= 28) | (hue >= 152))
    )
    blue = (
        (sat >= 52)
        & (val >= 30)
        & (hue >= 86)
        & (hue <= 154)
    )
    color_union = (red | blue).astype(np.uint8)
    dark_neutral = ((val <= 188) & (sat <= 120)).astype(np.uint8)
    red_integral = cv2.integral(red.astype(np.uint8), sdepth=cv2.CV_32S)
    blue_integral = cv2.integral(blue.astype(np.uint8), sdepth=cv2.CV_32S)
    color_integral = cv2.integral(color_union, sdepth=cv2.CV_32S)
    dark_integral = cv2.integral(dark_neutral, sdepth=cv2.CV_32S)

    def _rect_mean(integral: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
        area = max(1, (x2 - x1) * (y2 - y1))
        total = integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
        return float(total) / float(area)

    # 全畫面 coarse scan 只做 integral-image 計算；不先假設道路在下半部。
    coarse: List[Tuple[float, Tuple[int, int, int, int]]] = []
    for height_fraction in (0.055, 0.075, 0.095, 0.125, 0.16):
        _deadline_guard(deadline, cancel_event, min_remaining=0.18)
        box_h = max(26, int(round(ph * height_fraction)))
        step_y = max(12, int(round(box_h * 0.48)))
        for width_fraction in (0.44, 0.58, 0.72, 0.88):
            box_w = max(90, int(round(pw * width_fraction)))
            step_x = max(22, int(round(box_w * 0.30)))
            for y1 in range(0, max(1, ph - box_h + 1), step_y):
                y2 = min(ph, y1 + box_h)
                for x1 in range(0, max(1, pw - box_w + 1), step_x):
                    x2 = min(pw, x1 + box_w)
                    aspect = (x2 - x1) / max(1.0, float(y2 - y1))
                    if aspect < 2.35:
                        continue
                    red_fraction = _rect_mean(red_integral, x1, y1, x2, y2)
                    blue_fraction = _rect_mean(blue_integral, x1, y1, x2, y2)
                    color_fraction = _rect_mean(color_integral, x1, y1, x2, y2)
                    dark_fraction = _rect_mean(dark_integral, x1, y1, x2, y2)
                    balanced_color = min(red_fraction, blue_fraction)
                    # 必須同時有紅、藍；可排除 Safari/Chrome 藍色工具列與整片紅桌面 UI。
                    if (
                        red_fraction < 0.00045
                        or blue_fraction < 0.00045
                        or balanced_color < 0.00045
                        or dark_fraction < 0.28
                    ):
                        continue
                    score = (
                        2.5 * min(1.0, balanced_color / 0.025)
                        + 0.75 * min(1.0, color_fraction / 0.055)
                        + 0.72 * dark_fraction
                        + 0.10 * min(1.0, aspect / 5.0)
                    )
                    coarse.append((score, (x1, y1, x2, y2)))
    coarse.sort(key=lambda item: item[0], reverse=True)

    scored: List[Tuple[Tuple[float, float, float, float], float]] = []
    yy, xx = np.ogrid[:ph, :pw]
    seen_windows = set()
    for window_index, (coarse_score, (x1, y1, x2, y2)) in enumerate(coarse[:14]):
        if (window_index & 3) == 0:
            _deadline_guard(deadline, cancel_event, min_remaining=0.14)
        key = (x1 // 6, y1 // 6, x2 // 6, y2 // 6)
        if key in seen_windows:
            continue
        seen_windows.add(key)

        zone = preview[y1:y2, x1:x2]
        if zone.size == 0:
            continue
        gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        zone_h, zone_w = gray.shape[:2]
        expected_row_pitch = zone_h / float(ROAD_GRID_ROWS)
        min_radius = max(2, int(round(expected_row_pitch * 0.20)))
        max_radius = max(min_radius + 2, int(round(expected_row_pitch * 0.58)))
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=max(4.0, expected_row_pitch * 0.48),
            param1=85,
            param2=9,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is None:
            continue

        colored: List[Tuple[float, float, float]] = []
        for raw in np.round(circles[0]).astype(int):
            cx_local, cy_local, radius = [int(value) for value in raw]
            cx = cx_local + x1
            cy = cy_local + y1
            if not (0 <= cx < pw and 0 <= cy < ph):
                continue
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            annulus = (
                (d2 <= float(radius + 2) ** 2)
                & (d2 >= float(max(1, radius - 3)) ** 2)
            )
            outer = (
                (d2 <= float(radius + 5) ** 2)
                & (d2 >= float(radius + 2) ** 2)
            )
            red_pixels = int(np.count_nonzero(red[annulus]))
            blue_pixels = int(np.count_nonzero(blue[annulus]))
            dominant = max(red_pixels, blue_pixels)
            secondary = min(red_pixels, blue_pixels)
            if dominant < max(4, int(round(radius * 1.0))):
                continue
            if dominant / max(1.0, float(secondary)) < 1.08:
                continue
            outer_v = val[outer]
            outer_s = sat[outer]
            if outer_v.size:
                background_v = float(np.median(outer_v))
                background_s = float(np.median(outer_s))
                # 排除投注區的大型彩色 UI；DB 路紙背景通常偏暗/低彩度。
                if background_v > 205 and background_s > 120:
                    continue
            outcome = "B" if red_pixels > blue_pixels else "P"
            colored.append((float(cx), float(cy), float(radius), outcome))

        if len(colored) < 5:
            continue
        banker_count = sum(1 for item in colored if item[3] == "B")
        player_count = sum(1 for item in colored if item[3] == "P")
        if len(colored) >= 7 and (banker_count == 0 or player_count == 0):
            continue

        radii = np.asarray([item[2] for item in colored], dtype=np.float64)
        median_radius = max(2.5, float(np.median(radii)))
        radius_filtered = [
            item for item in colored
            if 0.55 * median_radius <= item[2] <= 1.65 * median_radius
        ]
        if len(radius_filtered) >= 5:
            colored = radius_filtered

        pitch_items = [(item[0], item[1], item[2]) for item in colored]
        pitch_y = _estimate_ring_pitch_free(pitch_items, axis="y", median_radius=median_radius)
        pitch_x = _estimate_ring_pitch_free(pitch_items, axis="x", median_radius=median_radius)
        xs = [item[0] for item in colored]
        ys = [item[1] for item in colored]
        span_x = max(xs) - min(xs)
        if span_x < max(pw * 0.12, pitch_x * 3.5):
            continue

        left = max(0.0, min(xs) - pitch_x * 0.75)
        top = max(0.0, min(ys) - pitch_y * 0.60)
        right = min(float(pw), max(xs) + pitch_x * 1.20)
        bottom = min(float(ph), top + pitch_y * 6.25)
        crop_w = right - left
        crop_h = bottom - top
        if crop_h <= 0 or crop_w / max(1.0, crop_h) < 2.25:
            continue

        roi_pixels = preview[int(top):int(bottom), int(left):int(right)]
        hsv_roi = cv2.cvtColor(roi_pixels, cv2.COLOR_BGR2HSV)
        dark_fraction = float(np.mean(hsv_roi[:, :, 2] <= 185))
        geometry_bonus = min(1.0, len(colored) / 18.0)
        score = (
            2.1 * geometry_bonus
            + 1.0 * min(1.0, span_x / max(1.0, pw * 0.36))
            + 0.75 * dark_fraction
            + 0.22 * min(1.0, coarse_score)
            - 0.15 * abs((crop_h / 6.0) - pitch_y) / max(1.0, pitch_y)
        )
        scored.append(
            (
                (
                    left / float(pw),
                    top / float(ph),
                    crop_w / float(pw),
                    crop_h / float(ph),
                ),
                score,
            )
        )

    scored.sort(key=lambda item: item[1], reverse=True)
    selected: List[Tuple[Tuple[float, float, float, float], float]] = []
    for roi, score in scored:
        if any(_roi_iou(roi, prior) >= 0.62 for prior, _ in selected):
            continue
        selected.append((roi, score))
        if len(selected) >= DB_FEATURE_MAX_CANDIDATES:
            break
    return selected


def detect_road_sequence_detailed(
    image_path: str | Path,
    *,
    venue: str = "",
    input_type: str = "auto",
    deadline: Optional[float] = None,
    cancel_event: Any = None,
) -> Dict[str, Any]:
    detector_started = time.perf_counter()
    caller_deadline = deadline
    internal_deadline = detector_started + ROAD_DETECTOR_HARD_TIMEOUT_SECONDS
    detector_deadline = (
        min(float(caller_deadline), internal_deadline)
        if caller_deadline is not None
        else internal_deadline
    )
    _deadline_guard(detector_deadline, cancel_event)
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
            and _looks_like_dream_compact_mobile_fullscreen(image)
        )
        ofalive_android_layout = bool(
            portrait
            and not dream_compact_mobile_layout
            and _looks_like_ofalive_android_fullscreen(image)
        )

        # DG / Dream 優先使用內容定位：白色路紙 + 6列格線 + 紅藍空心圓。
        # Dream 網址/瀏覽器不同時不依賴固定 ROI；同一 locator 同時覆蓋 DG 與 Dream Gaming。
        dream_feature_candidates: List[Tuple[Tuple[float, float, float, float], float]] = []
        if (
            DG_DB_FEATURE_LOCATORS_ENABLED
            and (venue_code == "DG" or dream_compact_mobile_layout)
        ):
            try:
                dream_feature_candidates = _dg_white_grid_feature_candidates(
                    image,
                    deadline=detector_deadline,
                    cancel_event=cancel_event,
                )
                for index, (roi, feature_score) in enumerate(
                    dream_feature_candidates[:DG_FEATURE_MAX_CANDIDATES]
                ):
                    plan.append({
                        "name": (
                            f"dg_feature_white_grid_{index}"
                            if venue_code == "DG"
                            else f"dream_feature_white_grid_{index}"
                        ),
                        "roi": roi,
                        "preference": (
                            88.0 if dream_compact_mobile_layout else 76.0
                        ) + min(8.0, feature_score),
                        "fixed_grid": True,
                        "ring_grid": False,
                        "grid_columns": None,
                        # Reuse DG direct-gridline geometry because Dream/DG road paper is the same 6-row style.
                        "profile": "dg_feature_white_grid",
                    })
            except TimeoutError:
                dream_feature_candidates = []

        if DG_DB_FEATURE_LOCATORS_ENABLED and venue_code == "DB":
            try:
                for index, (roi, feature_score) in enumerate(
                    _db_dark_ring_feature_candidates(
                        image,
                        deadline=detector_deadline,
                        cancel_event=cancel_event,
                    )
                ):
                    plan.append({
                        "name": f"db_feature_dark_ring_{index}",
                        "roi": roi,
                        "preference": 78.0 + min(8.0, feature_score),
                        "fixed_grid": False,
                        "ring_grid": True,
                        "grid_columns": None,
                        "profile": "db_feature_dark_ring",
                    })
            except TimeoutError:
                pass

        if dream_compact_mobile_layout:
            # 內容式 locator 是第一順位；固定 ROI 只留 1 個校準後 fallback，
            # 避免 2~6 個滑動 ROI 把 7 秒 budget 吃掉或混入珠盤/下三路。
            fallback_rois = _shifted_profile_rois(
                DREAM_COMPACT_MOBILE_BIG_ROAD_ROI,
                y_radius=DREAM_COMPACT_MOBILE_PROFILE_SEARCH_Y,
            )
            if fallback_rois:
                plan.append({
                    "name": "dream_compact_mobile_big_road_fallback",
                    "roi": fallback_rois[0],
                    "preference": 34.0,
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
                )[:MOBILE_PROFILE_MAX_CANDIDATES]
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
                )[:MOBILE_PROFILE_MAX_CANDIDATES]
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
                )[:MOBILE_PROFILE_MAX_CANDIDATES]
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
                )[:MOBILE_PROFILE_MAX_CANDIDATES]
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
        # DG / Dream 已經找到內容式白色 6×N 候選時，不要在評估它之前
        # 再花數秒生成 generic auto-focus。這是 LINE 手機 timeout 的主要來源之一。
        # 專用候選完全找不到時，才啟用 generic auto 作為 fallback。
        skip_general_auto = bool(
            (venue_code == "DG" or dream_compact_mobile_layout)
            and dream_feature_candidates
        )
        if MOBILE_AUTO_FOCUS_ENABLED and not skip_general_auto:
            _deadline_guard(detector_deadline, cancel_event, min_remaining=0.6)
            for index, (roi, auto_score) in enumerate(
                _general_road_candidates(
                    image,
                    deadline=detector_deadline,
                    cancel_event=cancel_event,
                )
            ):
                plan.append({
                    "name": f"mobile_auto_general_{index}",
                    "roi": roi,
                    "preference": 22.0 + min(15.0, auto_score * 3.0),
                    "fixed_grid": True,
                    "grid_columns": None,
                    "profile": "mobile_auto_general",
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
        if portrait and MOBILE_FULLSCREEN_FALLBACK_ENABLED:
            # 只有前面所有已知版型與 generic ROI 都沒有達到品質門檻時才會
            # 執行到這裡。兩個候選分別覆蓋 Dream 的中間窄大路與 Android／
            # Safari 較寬、較低的大路，不以裝置比例作為前置條件。
            for index, roi in enumerate(MOBILE_FULLSCREEN_FALLBACK_ROIS):
                plan.append({
                    "name": f"mobile_fullscreen_fallback_{index}",
                    "roi": roi,
                    "preference": 0.8 - index * 0.1,
                    "fixed_grid": True,
                    "grid_columns": None,
                    "profile": "mobile_fullscreen_fallback",
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

    general_auto_items = [
        item for item in plan
        if str(item.get("profile") or "") == "mobile_auto_general"
    ]
    specific_items = [
        item for item in plan
        if str(item.get("profile") or "") != "mobile_auto_general"
    ]
    if venue_code in {"DG", "DB"} and not likely_crop:
        venue_feature_items = [
            item for item in specific_items
            if str(item.get("profile") or "") in {
                "dg_feature_white_grid",
                "db_feature_dark_ring",
            }
        ]
        other_specific_items = [
            item for item in specific_items
            if str(item.get("profile") or "") not in {
                "dg_feature_white_grid",
                "db_feature_dark_ring",
            }
        ]
        plan = venue_feature_items + other_specific_items + general_auto_items
    elif general_auto_items and specific_items and not likely_crop:
        plan = (
            [specific_items[0], general_auto_items[0]]
            + specific_items[1:]
            + general_auto_items[1:]
        )

    seen = set()
    best: Optional[Dict[str, Any]] = None
    has_general_auto = any(
        str(item.get("profile") or "") == "mobile_auto_general"
        for item in plan
    )
    evaluated_general_auto = False
    detector_hard_timeout_reached = False
    for item in plan:
        try:
            _deadline_guard(detector_deadline, cancel_event, min_remaining=0.20)
        except TimeoutError as exc:
            if best is not None:
                errors.append(f"detector_hard_timeout: {exc}")
                detector_hard_timeout_reached = True
                break
            raise
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
                deadline=detector_deadline,
                cancel_event=cancel_event,
            )
            candidates.append(current)
            if str(item.get("profile") or "") == "mobile_auto_general":
                evaluated_general_auto = True
            if best is None or float(current.get("selection_score", -9999)) > float(
                best.get("selection_score", -9999)
            ):
                best = current

            # DB / DG 內容式 locator 已通過強品質閘門時直接採用；
            # 不再為了 generic auto 多跑一輪昂貴候選。
            if (
                str(item.get("profile") or "") in {
                    "dg_feature_white_grid",
                    "db_feature_dark_ring",
                }
                and _strong_acceptable(current)
            ):
                best = current
                break

            minimum_trials = 1 if (likely_crop or not has_general_auto) else 2
            if (
                ROAD_FAST_EARLY_EXIT
                and len(candidates) >= minimum_trials
                and (likely_crop or not has_general_auto or evaluated_general_auto)
            ):
                current_best = max(
                    candidates,
                    key=lambda item: float(item.get("selection_score", -9999.0) or -9999.0),
                )
                if _strong_acceptable(current_best):
                    best = current_best
                    break
        except TimeoutError as exc:
            if best is not None:
                errors.append(f"{name}: {exc}")
                detector_hard_timeout_reached = True
                break
            raise
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
        "mobile_auto_focus_used": str(best.get("region_name") or "").startswith("mobile_auto_general_"),
        "venue_feature_locator_used": str(best.get("region_name") or "").startswith(("dg_feature_", "db_feature_")),
        "detector_hard_timeout_reached": bool(detector_hard_timeout_reached),
        "detector_elapsed_ms": round((time.perf_counter() - detector_started) * 1000.0, 2),
        "detector_budget_ms": round(ROAD_DETECTOR_HARD_TIMEOUT_SECONDS * 1000.0, 2),
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
