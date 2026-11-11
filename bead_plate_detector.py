"""Read DG/MT/DB bead plates (珠盤路), never Big Road.

Six rows per column. Time order: down each column, then one column right.
Red=B, blue=P, green=T (a complete independent round, not an overlay).
v2 adaptive：加強安卓/多解析度珠盤路掃描，固定 ROI 失敗時啟動左下自適應搜尋
The original image normalization/OCR/LINE panel/model live elsewhere unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import math
import time

import cv2
import numpy as np

ROWS = 6
# Normalized target regions calibrated with full-screen DG/MT/DB mobile samples.
PHONE_ROIS = {
    "DG": (0.000, 0.620, 0.320, 0.200),
    "MT": (0.000, 0.730, 0.360, 0.190),
    "DB": (0.000, 0.780, 0.380, 0.190),
}
GENERIC_PHONE_ROI = (0.000, 0.700, 0.380, 0.220)
METHOD_VERSION = "bead_plate_hough_6row_v2_adaptive"


def _guard(deadline: Optional[float], cancel_event: Any) -> None:
    if cancel_event is not None and bool(getattr(cancel_event, "is_set", lambda: False)()):
        raise TimeoutError("珠盤路圖片分析已取消")
    if deadline is not None and time.perf_counter() >= float(deadline):
        raise TimeoutError("珠盤路圖片分析已超時")


def _color(hsv: np.ndarray, x: float, y: float, r: float, dark: bool) -> tuple[str, float]:
    height, width = hsv.shape[:2]
    x1, y1 = max(0, int(x-r-1)), max(0, int(y-r-1))
    x2, y2 = min(width, int(x+r+2)), min(height, int(y+r+2))
    if x2 <= x1 or y2 <= y1:
        return "", 0.0
    view = hsv[y1:y2, x1:x2]
    ys, xs = np.ogrid[y1:y2, x1:x2]
    inside = (xs-x)**2+(ys-y)**2 <= (.87*r)**2
    H, S, V = (view[:,:,i] for i in range(3))
    # DB has a dark red board; avoid classifying the background as a red bead.
    red = (((H <= 13) | (H >= 164)) & (S >= 55) & (V >= (120 if dark else 70)))
    blue = ((H >= 92) & (H <= 135) & (S >= 52) & (V >= 55))
    green = ((H >= 35) & (H <= 95) & (S >= 45) & (V >= 48))
    votes = [int(np.count_nonzero(mask & inside)) for mask in (red, blue, green)]
    winner = int(np.argmax(votes))
    quality = votes[winner]/max(1, sum(votes))
    if votes[winner] < max(9, math.pi*r*r*.14) or quality < .68:
        return "", round(quality, 4)
    return "BPT"[winner], round(quality, 4)


def _group(values: list[float], tolerance: float) -> list[list[float]]:
    groups: list[list[float]] = []
    for v in sorted(values):
        if not groups or v - sum(groups[-1])/len(groups[-1]) > tolerance:
            groups.append([v])
        else:
            groups[-1].append(v)
    return groups




def _adaptive_phone_roi(
    image: np.ndarray,
    deadline: Optional[float],
    cancel_event: Any,
) -> tuple[Optional[tuple[float, float, float, float]], str, int]:
    """Infer a bead-plate crop from colored circles in the lower-left phone area."""
    height, width = image.shape[:2]
    search_x1 = 0
    search_x2 = max(1, min(width, int(round(width * .45))))
    search_y1 = max(0, min(height - 1, int(round(height * .55))))
    search = image[search_y1:height, search_x1:search_x2]
    if search.size == 0 or search.shape[0] < 24 or search.shape[1] < 24:
        return None, "adaptive_search_region_invalid", 0

    _guard(deadline, cancel_event)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    dark = float(np.median(hsv[search_y1:height, search_x1:search_x2, 2])) < 160
    row_guess = search.shape[0] / ROWS
    detections: list[dict[str, Any]] = []

    for scale in (.75, 1.0, 1.25):
        _guard(deadline, cancel_event)
        scaled_width = max(1, int(round(search.shape[1] * scale)))
        scaled_height = max(1, int(round(search.shape[0] * scale)))
        resized = cv2.resize(
            search, (scaled_width, scaled_height),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
        scale_x = scaled_width / search.shape[1]
        scale_y = scaled_height / search.shape[0]
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        local_row_guess = row_guess * (scale_x + scale_y) / 2
        min_radius = max(3, int(round(local_row_guess * .07)))
        max_radius = max(min_radius + 3, int(round(local_row_guess * .50)))
        circles = cv2.HoughCircles(
            cv2.GaussianBlur(gray, (3, 3), 0), cv2.HOUGH_GRADIENT,
            dp=1, minDist=max(6, local_row_guess * .18),
            param1=85, param2=15,
            minRadius=min_radius, maxRadius=max_radius,
        )
        _guard(deadline, cancel_event)
        if circles is None:
            continue
        radius_scale = (scale_x + scale_y) / 2
        for xx, yy, rr in circles[0]:
            _guard(deadline, cancel_event)
            cx = search_x1 + float(xx) / scale_x
            cy = search_y1 + float(yy) / scale_y
            radius = float(rr) / radius_scale
            outcome, confidence = _color(hsv, cx, cy, radius, dark)
            if outcome:
                detections.append({
                    "outcome": outcome, "cx": cx, "cy": cy,
                    "radius": radius, "confidence": confidence,
                })

    if not detections:
        return None, "adaptive_color_circles_not_found", 0

    # The same bead may be returned at all three scales. Keep one best color hit.
    deduplicated: list[dict[str, Any]] = []
    for item in sorted(detections, key=lambda value: value["confidence"], reverse=True):
        duplicate = any(
            math.hypot(item["cx"] - old["cx"], item["cy"] - old["cy"])
            <= max(4.0, min(item["radius"], old["radius"]) * .82)
            for old in deduplicated
        )
        if not duplicate:
            deduplicated.append(item)

    if len(deduplicated) < 8:
        return None, "adaptive_too_few_colored_circles", len(deduplicated)

    radius_guess = float(np.median([item["radius"] for item in deduplicated]))
    row_groups = _group(
        [item["cy"] for item in deduplicated], max(4.0, radius_guess * .72)
    )
    if len(row_groups) < ROWS:
        return None, "adaptive_six_rows_not_found", len(deduplicated)

    best = None
    for start in range(len(row_groups) - ROWS + 1):
        window = row_groups[start:start + ROWS]
        centers = [float(np.median(group)) for group in window]
        diffs = np.diff(centers)
        pitch = float(np.median(diffs))
        if pitch <= 0 or any(abs(float(value) - pitch) > pitch * .28 for value in diffs):
            continue
        if not radius_guess * 1.15 <= pitch <= radius_guess * 4.2:
            continue
        points = []
        for item in deduplicated:
            row = int(np.argmin(np.abs(np.asarray(centers) - item["cy"])))
            if abs(centers[row] - item["cy"]) <= pitch * .38:
                points.append(item)
        if len(points) < 8:
            continue
        supported_rows = sum(1 for group in window if len(group) >= 2)
        if supported_rows < 4:
            continue
        error = float(np.mean(np.abs(diffs - pitch)) / pitch)
        score = (len(points), supported_rows, -error)
        if best is None or score > best[0]:
            best = (score, centers, pitch, points)

    if best is None:
        return None, "adaptive_six_rows_not_aligned", len(deduplicated)

    centers, pitch, points = best[1], best[2], best[3]
    x_groups = _group([item["cx"] for item in points], max(4.0, pitch * .28))
    supported_x = [float(np.median(group)) for group in x_groups if len(group) >= 2]
    if len(supported_x) < 2:
        return None, "adaptive_bead_columns_not_found", len(points)

    x_diffs = np.diff(supported_x)
    x_pitch = float(np.median(x_diffs))
    if x_pitch <= 0:
        return None, "adaptive_bead_columns_not_aligned", len(points)

    x0 = max(float(search_x1), min(supported_x) - x_pitch * .68)
    x2 = min(float(search_x2), max(supported_x) + x_pitch * .68)
    y0 = max(float(search_y1), centers[0] - pitch * .68)
    y2 = min(float(height), centers[-1] + pitch * .68)
    if x2 <= x0 or y2 <= y0:
        return None, "adaptive_region_invalid", len(points)

    roi = (x0 / width, y0 / height, (x2 - x0) / width, (y2 - y0) / height)
    return roi, "", len(points)


def _robust_bead_grid(
    hsv: np.ndarray,
    selected: list[dict[str, Any]],
    y_centers: list[float],
    bpitch: float,
    dark: bool,
) -> tuple[list[float], dict[tuple[int, int], dict[str, Any]]] | None:
    """MT/DB: use supported lattice centers, not every stray Hough circle.

    Contrast-enhanced numerals may look like extra circles at x+/-12px.
    Multiple beads in separate rows must agree on an actual column. After
    fitting a regular column pitch, every expected cell is reclassified from
    the HSV image at its lattice center. A missing interior cell still fails.
    """
    groups = _group([item["cx"] for item in selected], max(5, bpitch*.28))
    supported = [float(np.median(group)) for group in groups if len(group) >= 2]
    if len(supported) < 2:
        return None
    diffs = np.diff(supported)
    adjacent = [float(d) for d in diffs if bpitch*.66 <= d <= bpitch*1.55]
    if not adjacent:
        return None
    xpitch = float(np.median(adjacent))
    if not math.isfinite(xpitch) or xpitch <= 0:
        return None
    origin = supported[0]
    for cx in supported:
        idx = int(round((cx-origin)/xpitch))
        if idx < 0 or abs(cx-(origin+idx*xpitch)) > xpitch*.23:
            return None
    # A first/last column with only one observed bead must not disappear.
    leading = [
        item for item in selected
        if abs(item["cx"]-(origin-xpitch)) <= xpitch*.27
    ]
    if leading:
        origin -= xpitch
    maxcol = max(int(round((cx-origin)/xpitch)) for cx in supported)
    for item in selected:
        idx = int(round((item["cx"]-origin)/xpitch))
        if idx == maxcol+1 and abs(item["cx"]-(origin+idx*xpitch)) <= xpitch*.27:
            maxcol = idx
    if maxcol < 1 or maxcol*ROWS > 72:
        return None
    x_centers = [origin+i*xpitch for i in range(maxcol+1)]
    radius = float(np.median([item["radius"] for item in selected]))
    grid: dict[tuple[int, int], dict[str, Any]] = {}
    for col, cx in enumerate(x_centers):
        for row, cy in enumerate(y_centers):
            outcome, conf = _color(hsv, cx, cy, radius, dark)
            if not outcome:
                continue
            grid[(col, row)] = {
                "outcome": outcome, "cx": float(cx), "cy": float(cy),
                "radius": round(radius,2), "confidence": conf,
                "column": col, "row": row, "index": col*ROWS+row,
                "chronology_confirmed": True,
            }
    return x_centers, grid

def _scan(image: np.ndarray, roi: tuple[float,float,float,float], name: str,
          deadline: Optional[float], cancel_event: Any, *, robust: bool = False) -> Dict[str, Any]:
    _guard(deadline, cancel_event)
    height, width = image.shape[:2]
    x, y, w, h = roi
    x1, y1 = max(0, int(width*x)), max(0, int(height*y))
    x2, y2 = min(width, int(math.ceil(width*(x+w)))), min(height, int(math.ceil(height*(y+h))))
    crop = image[y1:y2, x1:x2]

    def failed(reason: str, **extra: Any) -> Dict[str, Any]:
        return {
            "ok": False, "quality_ok": False, "fallback_reason": reason,
            "selected_region": name, "method": METHOD_VERSION, **extra,
        }

    if crop.size == 0:
        return failed("invalid_bead_region")
    ch, cw = crop.shape[:2]
    row_guess = ch / ROWS
    if height < 1000:
        minr = max(3, int(round(row_guess * .18)))
        maxr = max(minr + 2, int(round(row_guess * .48)))
        param2 = 16
    elif height > 1600:
        radius_scale = min(1.25, height / 1600)
        minr = max(4, int(round(row_guess * .22 * radius_scale)))
        maxr = max(minr + 3, int(round(row_guess * .57 * radius_scale)))
        param2 = 17
    else:
        minr = max(4, int(row_guess * .22))
        maxr = max(6, int(row_guess * .57))
        param2 = 18

    grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(grayscale, (3, 3), 0), cv2.HOUGH_GRADIENT,
        dp=1, minDist=max(9, row_guess * .67),
        param1=90, param2=param2,
        minRadius=minr, maxRadius=max(minr + 2, maxr),
    )
    if circles is None:
        return failed("bead_circles_not_found")
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    dark = float(np.median(hsv[:,:,2])) < 160
    beads = []
    for xx, yy, rr in circles[0]:
        _guard(deadline, cancel_event)
        value, conf = _color(hsv, float(xx), float(yy), float(rr), dark)
        if value:
            beads.append({
                "outcome": value, "cx": float(xx), "cy": float(yy),
                "radius": round(float(rr), 2), "confidence": conf,
            })
    if len(beads) < 8:
        return failed("too_few_beads", recognized_count=len(beads))

    # Require all six consistent rows, rather than accepting a partial crop.
    gy = _group([v["cy"] for v in beads], max(5, row_guess * .29))
    gy = [g for g in gy if len(g) >= 2]
    best = None
    for i in range(max(0, len(gy) - ROWS + 1)):
        window = gy[i:i + ROWS]
        if len(window) != ROWS:
            continue
        centers = [float(np.median(g)) for g in window]
        diffs = np.diff(centers)
        pitch = float(np.median(diffs))
        if pitch <= 0 or any(abs(float(d) - pitch) > pitch * .22 for d in diffs):
            continue
        if not row_guess * .55 <= pitch <= row_guess * 1.42:
            continue
        fitness = (sum(map(len, window)), -float(np.mean(abs(diffs - pitch))))
        if best is None or fitness > best[0]:
            best = (fitness, centers, pitch)
    if best is None:
        return failed("bead_six_rows_not_aligned", recognized_count=len(beads))
    y_centers, bpitch = best[1], best[2]
    selected = []
    for item in beads:
        row = int(np.argmin(abs(np.array(y_centers) - item["cy"])))
        if abs(y_centers[row] - item["cy"]) <= bpitch * .29:
            selected.append({**item, "row": row})
    if robust:
        recovered = _robust_bead_grid(hsv, selected, y_centers, bpitch, dark)
        if recovered is None:
            return failed("bead_column_spacing_invalid")
        x_centers, grid = recovered
        xpitch = float(np.median(np.diff(x_centers)))
    else:
        # DG retains its original column + Hough-cell path.
        gx = _group([v["cx"] for v in selected], max(5, bpitch * .28))
        x_centers = [float(np.median(g)) for g in gx]
        if len(x_centers) < 2:
            return failed("bead_columns_not_aligned")
        xdiff = np.diff(x_centers)
        xpitch = float(np.median(xdiff))
        if (not bpitch * .66 <= xpitch <= bpitch * 1.55
                or any(abs(float(d) - xpitch) > xpitch * .27 for d in xdiff)):
            return failed("bead_column_spacing_invalid")
        grid = {}
        for item in selected:
            col = int(np.argmin(abs(np.array(x_centers) - item["cx"])))
            if abs(x_centers[col] - item["cx"]) > xpitch * .27:
                continue
            position = (col, item["row"])
            if position in grid:
                return failed("ambiguous_bead_grid_cell")
            grid[position] = {
                **item, "column": col, "index": col * ROWS + item["row"],
                "chronology_confirmed": True,
            }

    # Never silently skip an undetected cell: that would shift all later hands.
    last_col = len(x_centers) - 1
    for col in range(len(x_centers)):
        present = [r for r in range(ROWS) if (col, r) in grid]
        expected = list(range(ROWS)) if col < last_col else list(range(len(present)))
        if present != expected:
            return failed(
                "bead_grid_missing_or_uncertain_cell",
                recognized_count=len(grid), grid_columns=len(x_centers), uncertain_count=1,
            )
    raw = [
        grid[(col, row)]["outcome"] for col in range(len(x_centers))
        for row in range(ROWS) if (col, row) in grid
    ]
    if len(raw) > 72:
        return failed("bead_history_over_model_depth")
    if not any(value in {"B", "P"} for value in raw):
        return failed("bead_no_banker_or_player")

    sequence = [value for value in raw if value != "T"]
    markers: Dict[str, int] = {}
    last_bp = -1
    for value in raw:
        if value != "T":
            last_bp += 1
        elif last_bp >= 0:
            markers[str(last_bp)] = markers.get(str(last_bp), 0) + 1
    ordered = [
        grid[(col, row)] for col in range(len(x_centers))
        for row in range(ROWS) if (col, row) in grid
    ]
    quality = float(np.median([value["confidence"] for value in ordered]))
    return {
        "ok": True, "quality_ok": True, "reconstructed_all": True,
        "sequence": sequence, "raw_outcomes": raw, "tie_markers": markers,
        "grid_cells": ordered, "all_grid_cells": ordered,
        "recognized_count": len(ordered), "confirmed_round_count": len(raw),
        "uncertain_count": 0, "unknown_candidates": 0, "unknown_ratio": 0.0,
        "method": METHOD_VERSION, "selected_region": name,
        "layout_profile": name, "grid_rows": ROWS, "grid_columns": len(x_centers),
        "median_cell_confidence": quality, "alignment_ok": True,
        "effective_grid": {
            "x": x1 + float(x_centers[0]) - xpitch / 2,
            "y": y1 + float(y_centers[0]) - bpitch / 2,
            "width": len(x_centers) * xpitch, "height": ROWS * bpitch,
            "score": quality,
        },
        "mobile_auto_focus_used": ("shift_" in name or "adaptive" in name),
        "fallback_reason": "", "grid_alignment": {"score": quality},
        "chronology": "column_major_top_to_bottom",
        "bead_visible_rounds_only": True,
        "visible_window_may_not_include_shoe_start": True,
    }


def _rois(venue: str, image: np.ndarray, input_type: str):
    ih, iw = image.shape[:2]
    portrait = ih > iw * 1.12
    # If an old LINE session persisted road_crop while a full MT/DB phone
    # screenshot is uploaded, do not scan the entire UI as a cropped bead grid.
    if ((input_type == "road_crop" and
         (venue not in {"MT", "DB"} or not portrait or max(iw, ih) < 1000))
        or (max(iw, ih) < 1000 and not portrait)):
        yield "bead_plate_crop", (0.0, 0.0, 1.0, 1.0)
        return
    if portrait:
        # Try the requested venue first, then all mobile layouts. This covers
        # empty and stale venue values without depending on one fixed y offset.
        ordered = []
        if venue in PHONE_ROIS:
            ordered.append(venue)
        for tag in ("DG", "MT", "DB"):
            if tag not in ordered:
                ordered.append(tag)
        shifts = (
            ("bead_phone", 0.000),
            ("bead_phone_shift_up_2", -0.020),
            ("bead_phone_shift_down_2", 0.020),
            ("bead_phone_shift_up_4", -0.040),
            ("bead_phone_shift_down_4", 0.040),
            ("bead_phone_shift_up_6", -0.060),
            ("bead_phone_shift_down_6", 0.060),
            ("bead_phone_shift_up_8", -0.080),
            ("bead_phone_shift_down_8", 0.080),
            ("bead_phone_shift_up_10", -0.100),
            ("bead_phone_shift_down_10", 0.100),
            ("bead_phone_shift_up_12", -0.120),
            ("bead_phone_shift_down_5", 0.050),
        )
        for tag in ordered + ["GENERIC"]:
            x, y, w, h = GENERIC_PHONE_ROI if tag == "GENERIC" else PHONE_ROIS[tag]
            seen = set()
            for name, shift in shifts:
                yy = max(0.0, min(1.0 - h, y + shift))
                key = round(yy, 6)
                if key in seen:
                    continue
                seen.add(key)
                yield f"{tag.lower()}_{name}", (x, yy, w, h)
    else:
        # Desktop: the bead table is left of the Big Road; never scan its right.
        for name, roi in (("bead_desktop", (0, .72, .28, .25)),
                          ("bead_desktop_low", (0, .79, .30, .20))):
            yield f"{venue.lower()}_{name}", roi


def detect_bead_plate_sequence_detailed(image_path: str | Path | np.ndarray, *,
        venue: str = "", input_type: str = "auto", deadline: Optional[float] = None,
        cancel_event: Any = None) -> Dict[str, Any]:
    """Drop-in B/P/T source for existing screen_pipeline and BBB prediction."""
    started = time.perf_counter()
    _guard(deadline, cancel_event)
    if isinstance(image_path, np.ndarray):
        image = image_path
    else:
        # Keep exactly the existing preprocessing, image-size and quality path.
        from road_detector import _read_image
        image = _read_image(image_path, deadline=deadline, cancel_event=cancel_event)

    venue_code = str(venue).upper()
    portrait = image.shape[0] > image.shape[1] * 1.12
    candidates = []
    for name, roi in _rois(venue_code, image, str(input_type).lower()):
        _guard(deadline, cancel_event)
        # DG retains its existing Hough-cell path; MT/DB and the adaptive
        # candidate use the supported six-row lattice recovery.
        robust = (
            ("mt_" in name or "db_" in name)
            or venue_code in {"MT", "DB"}
        )
        if name.startswith("dg_"):
            robust = False
        result = _scan(image, roi, name, deadline, cancel_event, robust=robust)
        candidates.append(result)

    good = [item for item in candidates if item.get("quality_ok")]
    adaptive_search_used = False
    if not good and portrait:
        adaptive_search_used = True
        roi, adaptive_reason, adaptive_count = _adaptive_phone_roi(
            image, deadline, cancel_event
        )
        if roi is None:
            candidates.append({
                "ok": False, "quality_ok": False,
                "selected_region": "phone_adaptive_search",
                "fallback_reason": adaptive_reason,
                "recognized_count": adaptive_count,
                "method": METHOD_VERSION,
            })
        else:
            adaptive = _scan(
                image, roi, "phone_adaptive_6row_roi",
                deadline, cancel_event, robust=True,
            )
            candidates.append(adaptive)
            good = [item for item in candidates if item.get("quality_ok")]

    def summarize(result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": result.get("selected_region", ""),
            "quality_ok": bool(result.get("quality_ok")),
            "recognized_count": int(result.get("recognized_count", 0) or 0),
            "confirmed_round_count": int(result.get("confirmed_round_count", 0) or 0),
            "median_cell_confidence": float(result.get("median_cell_confidence", 0) or 0),
            "fallback_reason": result.get("fallback_reason", ""),
        }

    image_size = {"width": image.shape[1], "height": image.shape[0]}
    candidate_regions = [summarize(item) for item in candidates]
    if good:
        out = dict(max(good, key=lambda item: (
            int(item.get("confirmed_round_count", 0)),
            float(item.get("median_cell_confidence", 0)),
        )))
        out.update({
            "selected_region": out.get("selected_region", ""),
            "input_type": str(input_type),
            "image_size": image_size,
            "candidate_regions": candidate_regions,
            "adaptive_search_used": adaptive_search_used,
            "detector_elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        })
        return out

    reason = str(
        candidates[-1].get("fallback_reason")
        if candidates else "bead_region_not_found"
    )
    return {
        "ok": False, "quality_ok": False, "reconstructed_all": False,
        "sequence": [], "raw_outcomes": [], "tie_markers": {}, "grid_cells": [],
        "recognized_count": 0, "uncertain_count": 0,
        "fallback_reason": reason,
        "errors": ["珠盤路辨識不完整，請提供清楚、完整的珠盤路截圖"],
        "selected_region": "", "candidate_regions": candidate_regions,
        "adaptive_search_used": adaptive_search_used,
        "image_size": image_size, "method": METHOD_VERSION,
        "input_type": str(input_type),
        "detector_elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
    }


__all__=["detect_bead_plate_sequence_detailed"]
