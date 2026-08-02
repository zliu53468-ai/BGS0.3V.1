"""同一張遊戲畫面的快速 OCR 與路紙偵測協調器。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import os
import time

from road_detector import detect_road_sequence_detailed
from room_ocr import analyze_room_info


SCREEN_OCR_TIMEOUT = max(0.2, min(10.0, float(os.getenv("SCREEN_OCR_TIMEOUT", "3.0") or "3.0")))
_VISION_EXECUTOR = ThreadPoolExecutor(max_workers=max(2, min(6, int(os.getenv("SCREEN_VISION_WORKERS", "4") or "4"))), thread_name_prefix="screen-vision")


def _fallback_ocr(session: Mapping[str, Any], *, timed_out: bool, elapsed_ms: float) -> Dict[str, Any]:
    venue = str(session.get("venue") or session.get("last_confirmed_venue") or "")
    room = str(session.get("last_confirmed_room") or session.get("room") or "1")
    remaining = int(session.get("screen_remaining_cards", 0) or 0) or None
    return {
        "ok": bool(venue or room),
        "venue_code": venue,
        "venue_name": "",
        "room": room,
        "remaining_cards": remaining,
        "backend": "session_fallback",
        "raw_text": "",
        "confidence": 0.0,
        "tokens": [],
        "errors": ["OCR timeout; session data used"] if timed_out else [],
        "venue_source": "session_selected",
        "room_source": "session_previous" if session.get("last_confirmed_room") else "session_selected",
        "timed_out": timed_out,
        "elapsed_ms": round(elapsed_ms, 2),
    }


def analyze_game_screen(
    image_path: str | Path,
    existing_session: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """完整圖與牌路裁切圖皆可用；OCR 最長等待固定秒數，不再阻塞整體流程。"""
    started = time.perf_counter()
    session = dict(existing_session or {})
    venue_hint = str(session.get("venue") or session.get("last_confirmed_venue") or "")

    road_started = time.perf_counter()
    road_future = _VISION_EXECUTOR.submit(
        detect_road_sequence_detailed, image_path, venue=venue_hint, input_type="auto"
    )
    ocr_future = _VISION_EXECUTOR.submit(
        analyze_room_info, image_path, venue_hint, "auto", True
    )

    road = road_future.result()
    road_ms = (time.perf_counter() - road_started) * 1000.0
    input_type = str(road.get("input_type") or "full_screen")

    elapsed_seconds = time.perf_counter() - started
    remaining_wait = max(0.0, SCREEN_OCR_TIMEOUT - elapsed_seconds)
    ocr_timed_out = False
    ocr_started = time.perf_counter()
    try:
        ocr = ocr_future.result(timeout=remaining_wait)
    except FutureTimeoutError:
        ocr_timed_out = True
        ocr = _fallback_ocr(
            session, timed_out=True, elapsed_ms=(time.perf_counter() - ocr_started) * 1000.0
        )
    except Exception as exc:
        ocr = _fallback_ocr(session, timed_out=False, elapsed_ms=0.0)
        ocr["errors"] = [str(exc)]
    ocr_ms = float(ocr.get("elapsed_ms", 0.0) or 0.0)

    sequence = [
        str(item).upper() for item in list(road.get("sequence") or [])
        if str(item).upper() in {"B", "P"}
    ]

    fallback_venue = str(session.get("venue") or session.get("last_confirmed_venue") or "")
    fallback_room = str(session.get("last_confirmed_room") or session.get("room") or "1")
    fallback_remaining = int(session.get("screen_remaining_cards", 0) or 0) or None

    image_venue = str(ocr.get("venue_code") or "")
    image_room = str(ocr.get("room") or "")
    venue_code = image_venue or fallback_venue
    room = image_room or fallback_room
    remaining_cards = ocr.get("remaining_cards") if ocr.get("remaining_cards") is not None else fallback_remaining

    venue_source = "image_ocr" if image_venue and image_venue != fallback_venue else "session_selected"
    room_source = "image_ocr" if image_room else ("session_previous" if session.get("last_confirmed_room") else "session_selected")
    confidence = float(ocr.get("confidence", 0.0) or 0.0)

    resolved = {
        "venue_code": venue_code,
        "venue_name": str(ocr.get("venue_name") or ""),
        "room": room,
        "remaining_cards": remaining_cards,
        "input_type": input_type,
        "venue_source": venue_source,
        "room_source": room_source,
        "room_confidence": confidence if image_room else 0.0,
        "ocr_timed_out": ocr_timed_out,
    }
    timings = {
        "road_ms": round(road_ms, 2),
        "ocr_ms": round(ocr_ms, 2),
        "total_vision_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "ocr_timed_out": ocr_timed_out,
    }
    return {
        "ok": bool(sequence),
        "ocr": ocr,
        "road": road,
        "sequence": sequence,
        "resolved": resolved,
        "input_type": input_type,
        "timings": timings,
        "elapsed_ms": timings["total_vision_ms"],
    }


__all__ = ["analyze_game_screen"]
