"""遊戲畫面大路偵測模組。

預設使用 OpenCV 幾何輪廓＋圓心 HSV 判色；若設定 YOLO_MODEL_PATH 且
環境已安裝 ultralytics，則優先使用自訂 YOLO 權重。沒有自訂訓練權重時，
一般物件偵測模型不會認得百家樂莊閒圓圈，因此程式會自動回退 OpenCV。
"""
from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import os
import tempfile

import cv2
import numpy as np

from baccarat_vision import analyze_baccarat_image_detailed


_YOLO_MODEL: Any = None
_YOLO_LOCK = Lock()


def _parse_roi(raw: str, default: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    try:
        values = [float(part.strip()) for part in str(raw).split(",")]
        if len(values) != 4:
            raise ValueError
        x, y, width, height = values
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        width = max(0.05, min(1.0 - x, width))
        height = max(0.05, min(1.0 - y, height))
        return x, y, width, height
    except Exception:
        return default


ROAD_ROI = _parse_roi(os.getenv("ROAD_ROI", "0,0.24,1,0.62"), (0.0, 0.24, 1.0, 0.62))
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "").strip()
YOLO_CONFIDENCE = max(0.05, min(0.95, float(os.getenv("YOLO_CONFIDENCE", "0.35") or "0.35")))
YOLO_IMAGE_SIZE = max(320, min(1536, int(os.getenv("YOLO_IMAGE_SIZE", "960") or "960")))


def _read_image(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(Path(path)), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("無法讀取遊戲畫面。")
    return image


def _crop(image: np.ndarray, roi: Sequence[float]) -> Tuple[np.ndarray, Dict[str, int]]:
    height, width = image.shape[:2]
    x, y, roi_width, roi_height = [float(value) for value in roi]
    x1 = max(0, min(width - 1, int(round(x * width))))
    y1 = max(0, min(height - 1, int(round(y * height))))
    x2 = max(x1 + 1, min(width, int(round((x + roi_width) * width))))
    y2 = max(y1 + 1, min(height, int(round((y + roi_height) * height))))
    return image[y1:y2, x1:x2].copy(), {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}


def _sort_big_road(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not items:
        return []
    heights = [max(1.0, float(item["height"])) for item in items]
    tolerance = max(4.0, float(np.median(heights)) * 0.65)
    columns: List[List[Dict[str, Any]]] = []
    for item in sorted(items, key=lambda value: (value["cx"], value["cy"])):
        selected = None
        selected_distance = float("inf")
        for column in columns:
            center_x = float(np.mean([member["cx"] for member in column]))
            distance = abs(float(item["cx"]) - center_x)
            if distance <= tolerance and distance < selected_distance:
                selected = column
                selected_distance = distance
        if selected is None:
            columns.append([item])
        else:
            selected.append(item)
    columns.sort(key=lambda column: float(np.mean([item["cx"] for item in column])))
    ordered: List[Dict[str, Any]] = []
    for column in columns:
        ordered.extend(sorted(column, key=lambda item: (item["cy"], item["cx"])))
    return ordered


def _get_yolo_model() -> Any:
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    if not YOLO_MODEL_PATH or not Path(YOLO_MODEL_PATH).is_file():
        return None
    with _YOLO_LOCK:
        if _YOLO_MODEL is None:
            from ultralytics import YOLO
            _YOLO_MODEL = YOLO(YOLO_MODEL_PATH)
    return _YOLO_MODEL


def _normalize_yolo_label(label: str) -> str:
    value = str(label or "").strip().lower()
    banker_aliases = {"b", "banker", "red", "莊", "庄", "banker_circle"}
    player_aliases = {"p", "player", "blue", "閒", "闲", "player_circle"}
    if value in banker_aliases:
        return "B"
    if value in player_aliases:
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
        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        for coordinates, class_id, confidence in zip(xyxy, classes, confidences):
            label = names.get(int(class_id), str(int(class_id))) if isinstance(names, Mapping) else str(int(class_id))
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
        "recognized_count": len(ordered),
        "candidates": ordered,
        "method": "custom_yolo",
    }


def _write_temp_crop(crop: np.ndarray) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="road_roi_", suffix=".png", delete=False)
    handle.close()
    path = Path(handle.name)
    if not cv2.imwrite(str(path), crop):
        path.unlink(missing_ok=True)
        raise RuntimeError("無法建立路紙暫存裁圖。")
    return path


def _detect_opencv(crop: np.ndarray) -> Dict[str, Any]:
    path = _write_temp_crop(crop)
    try:
        return analyze_baccarat_image_detailed(path)
    finally:
        path.unlink(missing_ok=True)


def detect_road_sequence_detailed(image_path: str | Path) -> Dict[str, Any]:
    image = _read_image(image_path)
    crop, roi_pixels = _crop(image, ROAD_ROI)
    errors: List[str] = []

    if YOLO_MODEL_PATH:
        try:
            yolo_result = _detect_yolo(crop)
            if yolo_result.get("sequence"):
                yolo_result.update({"roi": roi_pixels, "normalized_roi": list(ROAD_ROI)})
                return yolo_result
        except Exception as exc:
            errors.append(f"yolo: {exc}")

    try:
        result = _detect_opencv(crop)
        if not result.get("sequence"):
            # ROI 設定不符某些平台時，以整張圖片再跑一次作為備援。
            full_result = analyze_baccarat_image_detailed(image_path)
            if full_result.get("sequence"):
                result = full_result
                roi_pixels = {"x": 0, "y": 0, "width": int(image.shape[1]), "height": int(image.shape[0])}
        result.update(
            {
                "roi": roi_pixels,
                "normalized_roi": list(ROAD_ROI),
                "errors": errors,
            }
        )
        return result
    except Exception as exc:
        errors.append(f"opencv: {exc}")
        return {
            "ok": False,
            "sequence": [],
            "recognized_count": 0,
            "method": "failed",
            "roi": roi_pixels,
            "normalized_roi": list(ROAD_ROI),
            "errors": errors,
        }


def detect_road_sequence(image_path: str | Path) -> List[str]:
    """題目指定入口：回傳依大路座標排序的 B/P List。"""
    return list(detect_road_sequence_detailed(image_path).get("sequence") or [])


__all__ = ["detect_road_sequence", "detect_road_sequence_detailed"]
