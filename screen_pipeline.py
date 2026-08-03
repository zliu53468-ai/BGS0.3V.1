"""同一張遊戲畫面的 OCR 與大路辨識協調器 V10.10。

與 road_detector V10.10 相容：
1. 牌路辨識優先，OCR 不得拖垮已選館別／桌號的 session。
2. 只有 quality_ok、reconstructed_all 與有效 sequence 同時成立，才允許送入預測。
3. 保留 detected_*、candidate_regions、effective_grid 與 debug 資訊供前端除錯。
4. 品質通過後產生 session_patch；之後 B/P/T 回報應只更新歷史，不重新掃圖。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import os
import time

from road_detector import detect_road_sequence_detailed
from room_ocr import analyze_room_info

FAST_SCREEN_MODE = os.getenv("FAST_SCREEN_MODE", "1").strip() == "1"
SKIP_OCR_WHEN_SESSION_READY = (
    os.getenv("SKIP_OCR_WHEN_SESSION_READY", "1").strip() == "1"
)
SCREEN_DEFAULT_REMAINING_CARDS = max(
    1, int(os.getenv("SCREEN_DEFAULT_REMAINING_CARDS", "416") or "416")
)
SCREEN_INPUT_TYPE = os.getenv("SCREEN_INPUT_TYPE", "auto").strip().lower() or "auto"
if SCREEN_INPUT_TYPE not in {"auto", "full_screen", "road_crop", "wide_multi_road"}:
    SCREEN_INPUT_TYPE = "auto"


def _session_is_ready(session: Mapping[str, Any]) -> bool:
    return bool(str(session.get("venue") or "").strip()) and bool(
        str(session.get("room") or "").strip()
    )


def _safe_int(value: Any, fallback: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or value == "":
            return fallback
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _session_remaining_cards(session: Mapping[str, Any]) -> Optional[int]:
    direct = _safe_int(session.get("screen_remaining_cards"))
    if direct is not None:
        return direct
    remaining_counts = session.get("remaining_counts")
    if isinstance(remaining_counts, (list, tuple)) and len(remaining_counts) == 10:
        try:
            return int(sum(int(value) for value in remaining_counts))
        except (TypeError, ValueError):
            pass
    virtual_shoe = session.get("virtual_shoe")
    if isinstance(virtual_shoe, (list, tuple)) and virtual_shoe:
        return len(virtual_shoe)
    return None


def _session_ocr_fallback(
    session: Mapping[str, Any],
    *,
    error: str = "",
    skipped: bool = True,
) -> Dict[str, Any]:
    remaining = _session_remaining_cards(session)
    venue = str(session.get("venue") or "").strip()
    room = str(session.get("room") or session.get("last_confirmed_room") or "1").strip()
    return {
        "venue_code": venue,
        "venue_name": "",
        "room": room,
        "remaining_cards": remaining,
        "venue_source": "session_selected" if venue else "unresolved",
        "room_source": "session_selected" if session.get("room") else "fallback_default",
        "ocr_skipped": bool(skipped),
        "ocr_error": str(error or ""),
        "elapsed_ms": 0.0,
    }


def _normalize_sequence(values: Any) -> list[str]:
    return [
        str(item).upper().strip()
        for item in list(values or [])
        if str(item).upper().strip() in {"B", "P"}
    ]


def _normalize_raw_outcomes(values: Any) -> list[str]:
    return [
        str(item).upper().strip()
        for item in list(values or [])
        if str(item).upper().strip() in {"B", "P", "T"}
    ]


def _mapping_list(values: Any) -> list[Dict[str, Any]]:
    return [dict(item) for item in list(values or []) if isinstance(item, Mapping)]


def analyze_game_screen(
    image_path: str | Path,
    existing_session: Optional[Mapping[str, Any]] = None,
    *,
    input_type: Optional[str] = None,
) -> Dict[str, Any]:
    """辨識完整畫面或牌路裁圖；品質失敗時不污染 session 與 predictor。"""
    started = time.perf_counter()
    session = dict(existing_session or {})

    requested_input_type = str(
        input_type
        or session.get("road_input_type")
        or session.get("screen_input_type")
        or SCREEN_INPUT_TYPE
        or "auto"
    ).strip().lower()
    if requested_input_type not in {
        "auto",
        "full_screen",
        "road_crop",
        "wide_multi_road",
    }:
        requested_input_type = "auto"

    road_started = time.perf_counter()
    road_error = ""
    try:
        road = detect_road_sequence_detailed(
            image_path,
            venue=str(session.get("venue") or ""),
            input_type=requested_input_type,
        )
        if not isinstance(road, Mapping):
            raise TypeError("road_detector 回傳值不是 Mapping")
        road = dict(road)
    except Exception as exc:
        road_error = str(exc)
        road = {
            "ok": False,
            "quality_ok": False,
            "sequence": [],
            "raw_outcomes": [],
            "tie_markers": {},
            "recognized_count": 0,
            "uncertain_count": 0,
            "unknown_candidates": 0,
            "reconstructed_all": False,
            "fallback_reason": "road_detector_exception",
            "errors": [road_error],
            "input_type": requested_input_type,
            "selected_region": "",
        }
    road_ms = (time.perf_counter() - road_started) * 1000.0

    should_skip_ocr = bool(
        FAST_SCREEN_MODE
        and SKIP_OCR_WHEN_SESSION_READY
        and _session_is_ready(session)
    )
    ocr_error = ""
    if should_skip_ocr:
        ocr = _session_ocr_fallback(session, skipped=True)
        ocr_ms = 0.0
    else:
        ocr_started = time.perf_counter()
        try:
            raw_ocr = analyze_room_info(image_path)
            if not isinstance(raw_ocr, Mapping):
                raise TypeError("room_ocr 回傳值不是 Mapping")
            ocr = dict(raw_ocr)
            ocr.setdefault("ocr_skipped", False)
            ocr.setdefault("ocr_error", "")
        except Exception as exc:
            ocr_error = str(exc)
            # OCR 失敗只降級，不能覆蓋或阻塞大路辨識結果。
            ocr = _session_ocr_fallback(
                session,
                error=ocr_error,
                skipped=False,
            )
        ocr_ms = (time.perf_counter() - ocr_started) * 1000.0

    detected_sequence = _normalize_sequence(road.get("sequence"))
    detected_raw_outcomes = _normalize_raw_outcomes(
        road.get("raw_outcomes") or detected_sequence
    )
    reconstructed_all = bool(road.get("reconstructed_all", False))
    detector_quality_ok = bool(road.get("quality_ok", False))
    quality_ok = bool(
        detector_quality_ok
        and reconstructed_all
        and detected_sequence
        and bool(road.get("ok", True))
    )

    # 只有完整、可信的時間序可送入模型。
    sequence = list(detected_sequence) if quality_ok else []
    raw_outcomes = list(detected_raw_outcomes) if quality_ok else []
    tie_markers = dict(road.get("tie_markers") or {}) if quality_ok else {}

    fallback_venue = str(session.get("venue") or "").strip()
    fallback_room = str(
        session.get("room") or session.get("last_confirmed_room") or "1"
    ).strip()
    fallback_remaining = _session_remaining_cards(session)
    ocr_remaining = _safe_int(ocr.get("remaining_cards"))
    resolved = {
        "venue_code": str(ocr.get("venue_code") or fallback_venue).strip(),
        "venue_name": str(ocr.get("venue_name") or "").strip(),
        "room": str(ocr.get("room") or fallback_room).strip(),
        "remaining_cards": (
            ocr_remaining
            if ocr_remaining is not None
            else fallback_remaining
            if fallback_remaining is not None
            else SCREEN_DEFAULT_REMAINING_CARDS
        ),
        "ocr_skipped": bool(ocr.get("ocr_skipped")),
        "ocr_error": str(ocr.get("ocr_error") or ocr_error),
    }

    grid_cells = _mapping_list(road.get("grid_cells"))
    all_grid_cells = _mapping_list(road.get("all_grid_cells"))
    candidate_regions = _mapping_list(road.get("candidate_regions"))

    recognized_count = int(road.get("recognized_count", len(detected_sequence)) or 0)
    uncertain_count = int(
        road.get("uncertain_count", road.get("unknown_candidates", 0)) or 0
    )
    effective_grid = dict(road.get("effective_grid") or {})
    grid_alignment = dict(road.get("grid_alignment") or {})
    layout_profile = str(
        road.get("layout_profile")
        or road.get("selected_region")
        or road.get("input_type")
        or ""
    )

    road_context: Dict[str, Any] = {
        "quality_ok": quality_ok,
        "recognition_quality_ok": quality_ok,
        "detector_quality_ok": detector_quality_ok,
        "reconstructed_all": reconstructed_all,
        "recognized_count": recognized_count,
        "uncertain_count": uncertain_count,
        "unknown_ratio": float(road.get("unknown_ratio", 0.0) or 0.0),
        "sequence": list(sequence),
        "raw_outcomes": list(raw_outcomes),
        "tie_markers": dict(tie_markers),
        "fallback_reason": str(road.get("fallback_reason") or ""),
        "input_type": str(road.get("input_type") or requested_input_type),
        "requested_input_type": requested_input_type,
        "selected_region": str(road.get("selected_region") or ""),
        "layout_profile": layout_profile,
        "method": str(road.get("method") or ""),
        "grid_rows": int(road.get("grid_rows", 0) or 0),
        "grid_columns": int(road.get("grid_columns", 0) or 0),
        "effective_grid": effective_grid,
        "alignment_ok": bool(road.get("alignment_ok", False)),
        "median_cell_confidence": float(
            road.get("median_cell_confidence", 0.0) or 0.0
        ),
        "debug_overlay_path": str(road.get("debug_overlay_path") or ""),
        # 真人截圖沒有真實剩餘牌點組成，預測端必須視為 estimated。
        "composition_quality": "estimated",
        "source_mode": "screenshot_live_table",
    }

    session_patch: Dict[str, Any] = {}
    if quality_ok:
        session_patch = {
            "road_sequence": list(sequence),
            "road_raw_outcomes": list(raw_outcomes),
            "road_tie_markers": dict(tie_markers),
            "road_context": dict(road_context),
            "road_scan_quality_ok": True,
            "road_scan_source": str(image_path),
            "road_scan_selected_region": str(road.get("selected_region") or ""),
            "road_scan_layout_profile": layout_profile,
            "road_scan_grid_columns": int(road.get("grid_columns", 0) or 0),
            "road_scan_updated_at": int(time.time()),
        }

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    fallback_reason = str(road.get("fallback_reason") or "")
    if road_error:
        user_message = "牌路掃描程式發生錯誤，請確認模組版本與圖片格式。"
    elif quality_ok:
        user_message = "牌路辨識完成。"
    elif fallback_reason == "missing_big_road_origin_0_0":
        user_message = "未包含大路左上起點，請重新截取完整大路區域。"
    elif fallback_reason.startswith("incomplete_big_road_reconstruction"):
        user_message = "大路時間序無法完整反推，請重新截取完整清楚的大路區域。"
    elif fallback_reason == "grid_alignment_not_confident":
        user_message = "大路格線對齊不足，請避免裁到珠盤路或其他衍生路。"
    else:
        user_message = "牌路對齊、顏色或時間序未通過，請重新截取完整清楚的大路區域。"

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
        "tie_markers": tie_markers,
        "grid_cells": grid_cells,
        "all_grid_cells": all_grid_cells,
        "candidate_regions": candidate_regions,
        "recognized_count": recognized_count,
        "uncertain_count": uncertain_count,
        "recognition_quality_ok": quality_ok,
        "reconstructed_all": reconstructed_all,
        "selected_region": str(road.get("selected_region") or ""),
        "layout_profile": layout_profile,
        "grid_rows": int(road.get("grid_rows", 0) or 0),
        "grid_columns": int(road.get("grid_columns", 0) or 0),
        "effective_grid": effective_grid,
        "grid_alignment": grid_alignment,
        "debug_overlay_path": str(road.get("debug_overlay_path") or ""),
        "fallback_reason": fallback_reason,
        "road_context": road_context,
        "session_patch": session_patch,
        "session_update_allowed": quality_ok,
        "resolved": resolved,
        "errors": {
            "road": road_error,
            "ocr": str(ocr.get("ocr_error") or ocr_error),
        },
        "timings": {
            "road_ms": round(road_ms, 2),
            "ocr_ms": round(ocr_ms, 2),
            "screen_total_ms": round(elapsed_ms, 2),
            "ocr_skipped": should_skip_ocr,
        },
        "elapsed_ms": round(elapsed_ms, 2),
    }


__all__ = ["analyze_game_screen"]
