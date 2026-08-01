"""同一張遊戲畫面的 OCR 與路紙偵測協調器。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import time

from road_detector import detect_road_sequence_detailed
from room_ocr import analyze_room_info


def analyze_game_screen(
    image_path: str | Path,
    existing_session: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """平行執行房間 OCR 與大路偵測，降低 LINE 回覆等待時間。"""
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="screen-vision") as executor:
        ocr_future = executor.submit(analyze_room_info, image_path)
        road_future = executor.submit(detect_road_sequence_detailed, image_path)
        ocr = ocr_future.result()
        road = road_future.result()

    sequence = [
        str(item).upper()
        for item in list(road.get("sequence") or [])
        if str(item).upper() in {"B", "P"}
    ]
    session = dict(existing_session or {})
    fallback_venue = str(session.get("venue") or "")
    fallback_room = str(session.get("room") or "1")
    fallback_remaining = len(session.get("virtual_shoe") or []) or None

    resolved = {
        "venue_code": str(ocr.get("venue_code") or fallback_venue),
        "venue_name": str(ocr.get("venue_name") or ""),
        "room": str(ocr.get("room") or fallback_room),
        "remaining_cards": (
            int(ocr["remaining_cards"])
            if ocr.get("remaining_cards") is not None
            else fallback_remaining
        ),
    }
    return {
        "ok": bool(sequence),
        "ocr": ocr,
        "road": road,
        "sequence": sequence,
        "resolved": resolved,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
    }


__all__ = ["analyze_game_screen"]
