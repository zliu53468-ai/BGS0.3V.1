"""遊戲畫面大路偵測模組 V10.1（ROI 與格位品質修正版）。

會依使用者已選館別嘗試平台專用 ROI、通用 ROI 與整張圖片，並以辨識數量、
未知候選與幾何雜訊評分選出最佳結果。牌路裁切圖會直接分析整張圖片。
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


def _env_roi(name: str, default: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    return _parse_roi(os.getenv(name, ",".join(str(v) for v in default)), default)


ROAD_ROI = _env_roi("ROAD_ROI", (0.0, 0.58, 1.0, 0.42))
VENUE_ROIS: Dict[str, Tuple[float, float, float, float]] = {
    "DG": _env_roi("DG_ROAD_ROI", (0.00, 0.80, 0.66, 0.20)),
    "MT": _env_roi("MT_ROAD_ROI", (0.00, 0.58, 1.00, 0.42)),
    "DB": _env_roi("DB_ROAD_ROI", (0.00, 0.58, 1.00, 0.42)),
    "SA": _env_roi("SA_ROAD_ROI", (0.00, 0.58, 1.00, 0.42)),
    "OB": _env_roi("OB_ROAD_ROI", (0.00, 0.58, 1.00, 0.42)),
    "T9": _env_roi("T9_ROAD_ROI", (0.00, 0.58, 1.00, 0.42)),
}
ROAD_AUTO_FULL_FALLBACK = os.getenv("ROAD_AUTO_FULL_FALLBACK", "1").strip() == "1"
ROAD_USE_YOLO = os.getenv("ROAD_USE_YOLO", "0").strip() == "1"
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
    return image[y1:y2, x1:x2].copy(), {
        "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1,
    }


def _sort_big_road(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not items:
        return []
    diameter = float(np.median([max(1.0, (float(i['width'])+float(i['height']))/2.0) for i in items]))
    def cluster(values: List[float]) -> List[float]:
        groups: List[List[float]] = []
        for value in sorted(values):
            target = next((g for g in groups if abs(value-float(np.mean(g))) <= diameter*0.48), None)
            (target if target is not None else groups.append([value]) or groups[-1]).append(value) if target is not None else None
        return [float(np.mean(g)) for g in groups]
    xs=cluster([float(i['cx']) for i in items]); ys=cluster([float(i['cy']) for i in items])
    xs.sort(); ys.sort()
    if len(ys)>6: ys=ys[:6]
    grid={}
    for item in items:
        c=min(range(len(xs)),key=lambda k:abs(float(item['cx'])-xs[k]))
        r=min(range(len(ys)),key=lambda k:abs(float(item['cy'])-ys[k]))
        if r<=5: grid[(c,r)]=item
    return [grid[key] for key in sorted(grid,key=lambda p:(p[0],p[1]))]


def _get_yolo_model() -> Any:
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    if not ROAD_USE_YOLO or not YOLO_MODEL_PATH or not Path(YOLO_MODEL_PATH).is_file():
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
    results = model.predict(source=crop, conf=YOLO_CONFIDENCE, imgsz=YOLO_IMAGE_SIZE, verbose=False)
    detections: List[Dict[str, Any]] = []
    for result in results:
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for coordinates, class_id, confidence in zip(
            boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()
        ):
            label = names.get(int(class_id), str(int(class_id))) if isinstance(names, Mapping) else str(int(class_id))
            outcome = _normalize_yolo_label(label)
            if not outcome:
                continue
            x1, y1, x2, y2 = [float(value) for value in coordinates]
            detections.append({
                "outcome": outcome, "label": str(label), "confidence": round(float(confidence), 6),
                "x": round(x1, 2), "y": round(y1, 2),
                "width": round(max(1.0, x2 - x1), 2), "height": round(max(1.0, y2 - y1), 2),
                "cx": round((x1 + x2) / 2.0, 2), "cy": round((y1 + y2) / 2.0, 2),
            })
    ordered = _sort_big_road(detections)
    return {
        "ok": bool(ordered), "sequence": [item["outcome"] for item in ordered],
        "recognized_count": len(ordered), "candidates": ordered, "method": "custom_yolo",
        "unknown_candidates": 0, "raw_contours": 0,
    }


def _score_result(result: Mapping[str, Any], *, preference: float = 0.0) -> float:
    recognized = int(result.get("recognized_count", 0) or 0)
    unknown = int(result.get("unknown_candidates", 0) or 0)
    raw = int(result.get("raw_contours", 0) or 0)
    if recognized <= 0:
        return -9999.0
    noise = max(0, raw - recognized * 8)
    return recognized * 5.0 - unknown * 1.5 - noise * 0.04 + preference


def _run_region(image: np.ndarray, roi: Tuple[float, float, float, float], name: str, preference: float) -> Dict[str, Any]:
    crop, pixels = _crop(image, roi)
    started = time.perf_counter()
    if ROAD_USE_YOLO and _get_yolo_model() is not None:
        result = _detect_yolo(crop)
    else:
        result = analyze_baccarat_array_detailed(crop)
    result = dict(result or {})
    result.update({
        "region_name": name,
        "roi": pixels,
        "normalized_roi": list(roi),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
    })
    result["selection_score"] = round(_score_result(result, preference=preference), 4)
    return result


def detect_road_sequence_detailed(
    image_path: str | Path,
    *,
    venue: str = "",
    input_type: str = "auto",
) -> Dict[str, Any]:
    image = _read_image(image_path)
    venue_code = str(venue or "").upper().strip()
    requested_type = str(input_type or "auto").lower().strip()
    errors: List[str] = []
    candidates: List[Dict[str, Any]] = []

    regions: List[Tuple[str, Tuple[float, float, float, float], float]] = []
    if requested_type == "road_crop":
        regions.append(("full_image", (0.0, 0.0, 1.0, 1.0), 4.0))
    else:
        if venue_code in VENUE_ROIS:
            regions.append((f"{venue_code.lower()}_venue_roi", VENUE_ROIS[venue_code], 3.0))
        regions.append(("generic_road_roi", ROAD_ROI, 1.5))
        if ROAD_AUTO_FULL_FALLBACK or requested_type == "auto":
            regions.append(("full_image", (0.0, 0.0, 1.0, 1.0), 0.0))

    seen = set()
    for name, roi, preference in regions:
        key = tuple(round(value, 4) for value in roi)
        if key in seen:
            continue
        seen.add(key)
        try:
            candidates.append(_run_region(image, roi, name, preference))
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    if not candidates:
        return {
            "ok": False, "sequence": [], "recognized_count": 0, "method": "failed",
            "input_type": "unknown", "errors": errors,
        }

    aspect = image.shape[1] / max(1.0, float(image.shape[0]))
    likely_crop = requested_type == "road_crop" or (requested_type == "auto" and aspect >= 2.40)
    venue_result = next(
        (item for item in candidates if str(item.get("region_name") or "").endswith("_venue_roi")),
        None,
    )
    full_result = next((item for item in candidates if item.get("region_name") == "full_image"), None)

    if requested_type == "full_screen":
        detected_type = "full_screen"
    elif requested_type == "road_crop":
        detected_type = "road_crop"
    else:
        detected_type = "road_crop" if likely_crop else "full_screen"

    # 使用者已選館且上傳完整畫面時，優先採平台專用 ROI，避免荷官、籌碼與
    # 下注按鈕被全圖輪廓誤判為牌路。只有專用 ROI 完全失敗才使用其他區域。
    if detected_type == "full_screen" and venue_result is not None and int(venue_result.get("recognized_count", 0) or 0) >= 4:
        best = venue_result
    elif detected_type == "road_crop" and full_result is not None and int(full_result.get("recognized_count", 0) or 0) > 0:
        best = full_result
    else:
        best = max(candidates, key=lambda item: float(item.get("selection_score", -9999.0)))
    region_name = str(best.get("region_name") or "")

    result = dict(best)
    result.update({
        "ok": bool(result.get("sequence")),
        "input_type": detected_type,
        "selected_region": region_name,
        "venue_hint": venue_code,
        "errors": errors,
        "candidate_regions": [
            {
                "name": item.get("region_name"),
                "recognized_count": int(item.get("recognized_count", 0) or 0),
                "unknown_candidates": int(item.get("unknown_candidates", 0) or 0),
                "score": float(item.get("selection_score", -9999.0)),
                "elapsed_ms": float(item.get("elapsed_ms", 0.0) or 0.0),
            }
            for item in candidates
        ],
    })
    return result


def detect_road_sequence(image_path: str | Path) -> List[str]:
    return list(detect_road_sequence_detailed(image_path).get("sequence") or [])


__all__ = ["detect_road_sequence", "detect_road_sequence_detailed"]
