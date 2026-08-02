"""同一張遊戲畫面的 OCR 與路紙偵測協調器 V10.4（1 CPU 加速版）。

1 CPU 主機不再讓 OCR 與 OpenCV 同時搶 CPU。若使用者已選館別與桌號，
預設直接沿用 session，只做大路辨識；只有資料不足時才補跑 OCR。
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
    return bool(str(session.get("venue") or "").strip()) and bool(str(session.get("room") or "").strip())


def _session_ocr_fallback(session: Mapping[str, Any]) -> Dict[str, Any]:
    remaining = session.get("screen_remaining_cards")
    if remaining is None:
        remaining = len(session.get("virtual_shoe") or []) or None
    return {
        "venue_code": str(session.get("venue") or ""),
        "venue_name": "",
        "room": str(session.get("room") or session.get("last_confirmed_room") or "1"),
        "remaining_cards": int(remaining) if remaining is not None else None,
        "venue_source": "session_selected",
        "room_source": "session_selected",
        "ocr_skipped": True,
        "elapsed_ms": 0.0,
    }


def analyze_game_screen(
    image_path: str | Path,
    existing_session: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """先做牌路辨識；有既有館別/桌號時跳過 OCR，適合 Render 1 CPU。"""
    started = time.perf_counter()
    session = dict(existing_session or {})

    road_started = time.perf_counter()
    road = detect_road_sequence_detailed(
        image_path,
        venue=str(session.get("venue") or ""),
        input_type="auto",
    )
    road_ms = (time.perf_counter() - road_started) * 1000.0

    should_skip_ocr = FAST_SCREEN_MODE and SKIP_OCR_WHEN_SESSION_READY and _session_is_ready(session)
    if should_skip_ocr:
        ocr = _session_ocr_fallback(session)
        ocr_ms = 0.0
    else:
        ocr_started = time.perf_counter()
        ocr = analyze_room_info(image_path)
        ocr_ms = (time.perf_counter() - ocr_started) * 1000.0

    sequence = [
        str(item).upper()
        for item in list(road.get("sequence") or [])
        if str(item).upper() in {"B", "P"}
    ]
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
    }
    raw_outcomes = [
        str(v).upper()
        for v in list(road.get("raw_outcomes") or sequence)
        if str(v).upper() in {"B", "P", "T"}
    ]
    grid_cells = [dict(item) for item in list(road.get("grid_cells") or []) if isinstance(item, Mapping)]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "ok": bool(sequence) and bool(road.get("quality_ok", True)),
        "ocr": ocr,
        "road": road,
        "sequence": sequence,
        "raw_outcomes": raw_outcomes,
        "tie_markers": dict(road.get("tie_markers") or {}),
        "grid_cells": grid_cells,
        "recognized_count": int(road.get("recognized_count", len(sequence)) or len(sequence)),
        "uncertain_count": int(road.get("uncertain_count", road.get("unknown_candidates", 0)) or 0),
        "recognition_quality_ok": bool(road.get("quality_ok", True)),
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
