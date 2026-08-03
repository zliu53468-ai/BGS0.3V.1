"""同一張遊戲畫面的 OCR 與路紙偵測協調器 V10.9.1。

路紙辨識優先；session 已有館別／桌號時不讓 OCR 失敗阻塞。只有 quality_ok、
對齊與完整反推都通過時，才把 sequence/raw_outcomes 放到可供預測的輸出與
session_patch。辨識失敗仍保留 detected_* 與 road debug，並依 fallback_reason
產生正確提示（不再誤用「未偵測到大路圓圈」）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import os
import time

from road_detector import detect_road_sequence_detailed
from room_ocr import analyze_room_info

FAST_SCREEN_MODE = os.getenv("FAST_SCREEN_MODE", "1").strip() == "1"
SKIP_OCR_WHEN_SESSION_READY = os.getenv("SKIP_OCR_WHEN_SESSION_READY", "1").strip() == "1"


def _session_is_ready(session: Mapping[str, Any]) -> bool:
    return bool(str(session.get("venue") or "").strip()) and bool(
        str(session.get("room") or "").strip()
    )


def _session_ocr_fallback(
    session: Mapping[str, Any],
    *,
    error: str = "",
) -> Dict[str, Any]:
    remaining = session.get("screen_remaining_cards")
    if remaining is None:
        remaining = len(session.get("virtual_shoe") or []) or None
    return {
        "venue_code": str(session.get("venue") or ""),
        "venue_name": "",
        "room": str(session.get("room") or session.get("last_confirmed_room") or "1"),
        "remaining_cards": int(remaining) if remaining is not None else None,
        "venue_source": "session_selected" if session.get("venue") else "unresolved",
        "room_source": "session_selected" if session.get("room") else "fallback_default",
        "ocr_skipped": True,
        "ocr_error": error,
        "elapsed_ms": 0.0,
    }


def _user_message_for_road(road: Mapping[str, Any], quality_ok: bool) -> str:
    """依 fallback_reason / method 產生正確提示，避免固定格失敗仍寫「大路圓圈」。"""
    if quality_ok:
        return "牌路辨識完成。"

    reason = str(road.get("fallback_reason") or "").strip()
    method = str(road.get("method") or "").lower()
    fixed = bool(road.get("fixed_grid")) or method.startswith("fixed_")

    reason_map = {
        "no_recognized_cells": "未辨識到足夠的紅／藍大路點，請截取更清楚的大路區域。",
        "missing_big_road_origin_0_0": "大路起點格對不準，請重新截取完整大路（含左上起點）。",
        "ambiguous_big_road_reconstruction": "大路時間序無法唯一還原，請換更清晰或完整的大路截圖。",
        "grid_alignment_not_confident": "大路格線對齊不穩，請避免裁切過窄或傾斜，重新截完整大路區。",
        "cell_color_confidence_too_low": "紅藍顏色可信度不足（反光／壓縮／偏色），請用更清楚的截圖。",
        "recognized_count_below_minimum": "辨識到的莊閒點太少，請確認畫面含完整大路。",
        "too_many_uncertain_cells": "有過多格顏色不明，請提高截圖清晰度後重傳。",
        "all_regions_failed": "所有掃描區域皆失敗，請傳送包含完整大路的清晰截圖。",
        "contour_circle_detection_failed": "輪廓找圓備援未成功，請改傳大路區域裁切圖或更完整畫面。",
        "yolo_no_detections": "物件偵測未找到路點，請改傳清楚的大路截圖。",
    }

    if reason in reason_map:
        return reason_map[reason]
    if reason.startswith("incomplete_big_road_reconstruction_"):
        return "大路時間序反推不完整，請重新截取完整清楚的大路區。"

    if fixed or method.startswith("fixed_"):
        return "牌路對齊、顏色或時間序反推未通過，請重新截取完整清楚的大路區。"

    # 僅在真的走找圓／YOLO 且無結果時才提「圓」
    if "yolo" in method or "contour" in method or "circle" in method:
        return "未偵測到可用的大路點。請傳送包含完整大路的畫面或牌路裁切圖。"

    return "牌路辨識未通過，請重新傳送包含完整大路區域的清晰截圖。"


def analyze_game_screen(
    image_path: str | Path,
    existing_session: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """先辨識大路；品質未通過時不把錯序列交給預測或 session。"""
    started = time.perf_counter()
    session = dict(existing_session or {})

    road_started = time.perf_counter()
    road = detect_road_sequence_detailed(
        image_path,
        venue=str(session.get("venue") or ""),
        input_type="auto",
    )
    road_ms = (time.perf_counter() - road_started) * 1000.0

    should_skip_ocr = bool(
        FAST_SCREEN_MODE
        and SKIP_OCR_WHEN_SESSION_READY
        and _session_is_ready(session)
    )
    ocr_error = ""
    if should_skip_ocr:
        ocr = _session_ocr_fallback(session)
        ocr_ms = 0.0
    else:
        ocr_started = time.perf_counter()
        try:
            ocr = analyze_room_info(image_path)
        except Exception as exc:
            ocr_error = str(exc)
            ocr = _session_ocr_fallback(session, error=ocr_error)
        ocr_ms = (time.perf_counter() - ocr_started) * 1000.0

    detected_sequence = [
        str(item).upper()
        for item in list(road.get("sequence") or [])
        if str(item).upper() in {"B", "P"}
    ]
    detected_raw_outcomes = [
        str(value).upper()
        for value in list(road.get("raw_outcomes") or detected_sequence)
        if str(value).upper() in {"B", "P", "T"}
    ]
    quality_ok = bool(
        road.get("quality_ok", False)
        and road.get("reconstructed_all", True)
        and detected_sequence
    )
    # 只有品質通過的資料能進 predictor；錯誤結果仍留在 road/detected_* 供除錯。
    sequence = list(detected_sequence) if quality_ok else []
    raw_outcomes = list(detected_raw_outcomes) if quality_ok else []

    fallback_venue = str(session.get("venue") or "")
    fallback_room = str(session.get("room") or session.get("last_confirmed_room") or "1")
    fallback_remaining = session.get("screen_remaining_cards")
    if fallback_remaining is None:
        fallback_remaining = len(session.get("virtual_shoe") or []) or None
    resolved = {
        "venue_code": str(ocr.get("venue_code") or fallback_venue),
        "venue_name": str(ocr.get("venue_name") or ""),
        "room": str(ocr.get("room") or fallback_room),
        "remaining_cards": (
            int(ocr["remaining_cards"])
            if ocr.get("remaining_cards") is not None
            else (int(fallback_remaining) if fallback_remaining is not None else 416)
        ),
        "ocr_skipped": bool(ocr.get("ocr_skipped")),
        "ocr_error": str(ocr.get("ocr_error") or ocr_error),
    }
    grid_cells = [
        dict(item)
        for item in list(road.get("grid_cells") or [])
        if isinstance(item, Mapping)
    ]
    all_grid_cells = [
        dict(item)
        for item in list(road.get("all_grid_cells") or [])
        if isinstance(item, Mapping)
    ]
    road_context = {
        "quality_ok": quality_ok,
        "recognition_quality_ok": quality_ok,
        "reconstructed_all": bool(road.get("reconstructed_all", quality_ok)),
        "recognized_count": int(road.get("recognized_count", len(detected_sequence)) or 0),
        "uncertain_count": int(
            road.get("uncertain_count", road.get("unknown_candidates", 0)) or 0
        ),
        "unknown_ratio": float(road.get("unknown_ratio", 0.0) or 0.0),
        "sequence": list(sequence),
        "raw_outcomes": list(raw_outcomes),
        "tie_markers": dict(road.get("tie_markers") or {}) if quality_ok else {},
        "fallback_reason": str(road.get("fallback_reason") or ""),
        "input_type": str(road.get("input_type") or ""),
        "selected_region": str(road.get("selected_region") or ""),
        "composition_quality": "estimated",
    }
    session_patch: Dict[str, Any] = {}
    if quality_ok:
        session_patch = {
            "road_sequence": list(sequence),
            "road_raw_outcomes": list(raw_outcomes),
            "road_tie_markers": dict(road.get("tie_markers") or {}),
            "road_context": dict(road_context),
            "road_scan_quality_ok": True,
            "road_scan_source": str(image_path),
        }

    user_message = _user_message_for_road(road, quality_ok)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "ok": quality_ok,
        "prediction_blocked": not quality_ok,
        "user_message": user_message,
        "ocr": ocr,
        "road": road,
        "sequence": sequence,
        "raw_outcomes": raw_outcomes,
        "detected_sequence": detected_sequence,
        "detected_raw_outcomes": detected_raw_outcomes,
        "tie_markers": dict(road.get("tie_markers") or {}) if quality_ok else {},
        "grid_cells": grid_cells,
        "all_grid_cells": all_grid_cells,
        "recognized_count": int(road.get("recognized_count", len(detected_sequence)) or 0),
        "uncertain_count": int(
            road.get("uncertain_count", road.get("unknown_candidates", 0)) or 0
        ),
        "recognition_quality_ok": quality_ok,
        "road_context": road_context,
        "session_patch": session_patch,
        "session_update_allowed": quality_ok,
        "resolved": resolved,
        "timings": {
            "road_ms": round(road_ms, 2),
            "ocr_ms": round(ocr_ms, 2),
            "screen_total_ms": round(elapsed_ms, 2),
            "ocr_skipped": should_skip_ocr,
        },
        "elapsed_ms": round(elapsed_ms, 2),
    }


__all__ = ["analyze_game_screen"]
