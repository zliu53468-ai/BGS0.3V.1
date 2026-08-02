"""遊戲畫面大路偵測模組 V10.4（1 CPU 單路徑加速版）。

不再固定把館別 ROI、通用 ROI、完整圖片全部掃完。採逐級策略：
館別 ROI 成功即回傳；不足才掃通用 ROI；仍失敗才掃全圖。
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
        x = max(0.0, min(1.0, x)); y = max(0.0, min(1.0, y))
        width = max(0.05, min(1.0 - x, width)); height = max(0.05, min(1.0 - y, height))
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
ROAD_AUTO_FULL_FALLBACK = os.getenv("ROAD_AUTO_FULL_FALLBACK", "0").strip() == "1"
ROAD_USE_YOLO = os.getenv("ROAD_USE_YOLO", "0").strip() == "1"
ROAD_FAST_EARLY_EXIT = os.getenv("ROAD_FAST_EARLY_EXIT", "1").strip() == "1"
ROAD_FAST_MIN_RECOGNIZED = max(4, int(os.getenv("ROAD_FAST_MIN_RECOGNIZED", "8") or "8"))
ROAD_FAST_MAX_UNKNOWN_RATIO = max(0.0, min(0.8, float(os.getenv("ROAD_FAST_MAX_UNKNOWN_RATIO", "0.18") or "0.18")))
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
    x, y, rw, rh = [float(value) for value in roi]
    x1 = max(0, min(width - 1, int(round(x * width)))); y1 = max(0, min(height - 1, int(round(y * height))))
    x2 = max(x1 + 1, min(width, int(round((x + rw) * width)))); y2 = max(y1 + 1, min(height, int(round((y + rh) * height))))
    return image[y1:y2, x1:x2].copy(), {"x": x1, "y": y1, "width": x2-x1, "height": y2-y1}


def _sort_big_road(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(items, key=lambda item: (float(item.get("cx", 0)), float(item.get("cy", 0))))


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
    if value in {"b", "banker", "red", "莊", "庄", "banker_circle"}: return "B"
    if value in {"p", "player", "blue", "閒", "闲", "player_circle"}: return "P"
    return ""


def _detect_yolo(crop: np.ndarray) -> Dict[str, Any]:
    model = _get_yolo_model()
    if model is None:
        return {"ok": False, "sequence": [], "method": "yolo_unavailable"}
    results = model.predict(source=crop, conf=YOLO_CONFIDENCE, imgsz=YOLO_IMAGE_SIZE, verbose=False)
    detections: List[Dict[str, Any]] = []
    for result in results:
        names = getattr(result, "names", {}) or {}; boxes = getattr(result, "boxes", None)
        if boxes is None: continue
        for coordinates, class_id, confidence in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
            label = names.get(int(class_id), str(int(class_id))) if isinstance(names, Mapping) else str(int(class_id))
            outcome = _normalize_yolo_label(label)
            if not outcome: continue
            x1,y1,x2,y2=[float(v) for v in coordinates]
            detections.append({"outcome":outcome,"label":str(label),"confidence":round(float(confidence),6),"x":round(x1,2),"y":round(y1,2),"width":round(max(1.0,x2-x1),2),"height":round(max(1.0,y2-y1),2),"cx":round((x1+x2)/2,2),"cy":round((y1+y2)/2,2)})
    ordered=_sort_big_road(detections)
    return {"ok":bool(ordered),"sequence":[i["outcome"] for i in ordered],"raw_outcomes":[i["outcome"] for i in ordered],"recognized_count":len(ordered),"candidates":ordered,"method":"custom_yolo","unknown_candidates":0,"raw_contours":0,"quality_ok":bool(ordered)}


def _score_result(result: Mapping[str, Any], preference: float = 0.0) -> float:
    recognized=int(result.get("recognized_count",0) or 0); unknown=int(result.get("unknown_candidates",0) or 0); raw=int(result.get("raw_contours",0) or 0)
    if recognized<=0: return -9999.0
    noise=max(0,raw-recognized*8)
    return recognized*5.0-unknown*1.5-noise*0.04+preference


def _run_region(image: np.ndarray, roi: Tuple[float,float,float,float], name: str, preference: float) -> Dict[str, Any]:
    crop,pixels=_crop(image,roi); started=time.perf_counter()
    result=_detect_yolo(crop) if ROAD_USE_YOLO and _get_yolo_model() is not None else analyze_baccarat_array_detailed(crop)
    result=dict(result or {})
    result.update({"region_name":name,"roi":pixels,"normalized_roi":list(roi),"elapsed_ms":round((time.perf_counter()-started)*1000,2)})
    result["selection_score"]=round(_score_result(result,preference),4)
    return result


def _acceptable(result: Mapping[str, Any]) -> bool:
    recognized=int(result.get("recognized_count",0) or 0)
    unknown=int(result.get("unknown_candidates",result.get("uncertain_count",0)) or 0)
    ratio=unknown/max(1,recognized+unknown)
    return recognized>=ROAD_FAST_MIN_RECOGNIZED and ratio<=ROAD_FAST_MAX_UNKNOWN_RATIO and bool(result.get("quality_ok",True))


def detect_road_sequence_detailed(image_path: str | Path, *, venue: str="", input_type: str="auto") -> Dict[str, Any]:
    image=_read_image(image_path); venue_code=str(venue or "").upper().strip(); requested=str(input_type or "auto").lower().strip()
    aspect=image.shape[1]/max(1.0,float(image.shape[0])); likely_crop=requested=="road_crop" or (requested=="auto" and aspect>=2.40)
    detected_type="road_crop" if likely_crop else "full_screen"
    errors: List[str]=[]; candidates: List[Dict[str,Any]]=[]; attempted=[]

    if likely_crop:
        plan=[("full_image",(0.0,0.0,1.0,1.0),4.0)]
    else:
        plan=[]
        if venue_code in VENUE_ROIS: plan.append((f"{venue_code.lower()}_venue_roi",VENUE_ROIS[venue_code],3.0))
        plan.append(("generic_road_roi",ROAD_ROI,1.5))
        if ROAD_AUTO_FULL_FALLBACK: plan.append(("full_image",(0.0,0.0,1.0,1.0),0.0))

    seen=set(); best=None
    for name,roi,pref in plan:
        key=tuple(round(v,4) for v in roi)
        if key in seen: continue
        seen.add(key); attempted.append(name)
        try:
            current=_run_region(image,roi,name,pref); candidates.append(current)
            if best is None or float(current.get("selection_score",-9999))>float(best.get("selection_score",-9999)): best=current
            if ROAD_FAST_EARLY_EXIT and _acceptable(current):
                best=current
                break
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    if best is None:
        return {"ok":False,"sequence":[],"recognized_count":0,"method":"failed","input_type":"unknown","errors":errors,"attempted_regions":attempted}

    result=dict(best)
    result.update({
        "ok":bool(result.get("sequence")),"input_type":detected_type,"selected_region":str(best.get("region_name") or ""),"venue_hint":venue_code,"errors":errors,"attempted_regions":attempted,
        "fast_early_exit":len(candidates)<len(plan),
        "candidate_regions":[{"name":i.get("region_name"),"recognized_count":int(i.get("recognized_count",0) or 0),"unknown_candidates":int(i.get("unknown_candidates",0) or 0),"score":float(i.get("selection_score",-9999)),"elapsed_ms":float(i.get("elapsed_ms",0) or 0)} for i in candidates],
    })
    return result


def detect_road_sequence(image_path: str | Path) -> List[str]:
    return list(detect_road_sequence_detailed(image_path).get("sequence") or [])

__all__=["detect_road_sequence","detect_road_sequence_detailed"]
