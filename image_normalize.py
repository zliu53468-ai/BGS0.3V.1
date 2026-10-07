"""Bounded image normalization shared by screenshot OCR and road detection.

The helper deliberately keeps one working image only: it reads once, applies an
optional EXIF correction, bounds the longest side, and performs at most one
lightweight contrast pass.  Callers receive a readable error in ``meta``
instead of an exception that could take down an image-analysis worker.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import os
import time

import cv2
import numpy as np


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(str(os.getenv(name, default)).strip())
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


ANALYSIS_MAX_SIDE = _env_int("ANALYSIS_MAX_SIDE", 1600, 800, 2000)
ANALYSIS_SMALL_IMAGE_THRESHOLD = _env_int(
    "ANALYSIS_SMALL_IMAGE_THRESHOLD", 480, 240, 800
)
ANALYSIS_SMALL_IMAGE_TARGET = _env_int(
    "ANALYSIS_SMALL_IMAGE_TARGET", 720, 480, 1200
)
ANALYSIS_LIGHT_ENHANCE = os.getenv("ANALYSIS_LIGHT_ENHANCE", "1").strip() == "1"
ANALYSIS_EXIF_ORIENTATION = (
    os.getenv("ANALYSIS_EXIF_ORIENTATION", "1").strip() == "1"
)


def _meta(
    *,
    original_size: Tuple[int, int] = (0, 0),
    working_size: Tuple[int, int] = (0, 0),
    scale: float = 1.0,
    orientation_applied: bool = False,
    enhanced: bool = False,
    error: str = "",
) -> Dict[str, Any]:
    return {
        "ok": not bool(error),
        "scale": round(float(scale), 6),
        "original_size": {"width": int(original_size[0]), "height": int(original_size[1])},
        "working_size": {"width": int(working_size[0]), "height": int(working_size[1])},
        "orientation_applied": bool(orientation_applied),
        "enhanced": bool(enhanced),
        "error": str(error or ""),
    }


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ValueError("圖片維度不支援。")
    if image.shape[2] == 3:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise ValueError("圖片色彩通道不支援。")


def _apply_exif_orientation(path: Path, image: np.ndarray) -> Tuple[np.ndarray, bool]:
    if not ANALYSIS_EXIF_ORIENTATION:
        return image, False
    try:
        from PIL import Image, ImageOps

        with Image.open(str(path)) as source:
            orientation = int(source.getexif().get(274, 1) or 1)
            if orientation == 1:
                return image, False
            corrected = ImageOps.exif_transpose(source).convert("RGB")
            rgb = np.asarray(corrected)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), True
    except Exception:
        # EXIF is a convenience only; decoding still succeeds for most captures.
        return image, False


def _read_path(path: Path) -> Tuple[np.ndarray | None, bool, str]:
    if not path.exists():
        return None, False, f"找不到圖片：{path}"
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    except Exception as exc:
        return None, False, f"圖片讀取失敗：{exc}"
    if image is None or image.size == 0:
        suffix = path.suffix.lower()
        if suffix in {".heic", ".heif"}:
            return None, False, "HEIC/HEIF 無法解碼；請先轉成 JPG 或 PNG。"
        return None, False, "無法讀取圖片，請確認格式為 JPG、PNG、WEBP 或可解碼的 HEIC。"
    try:
        image = _as_bgr(image)
    except Exception as exc:
        return None, False, f"圖片格式不支援：{exc}"
    image, orientation_applied = _apply_exif_orientation(path, image)
    return image, orientation_applied, ""


def _light_enhance(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(
        cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR
    )


def _deadline_guard(deadline: float | None, cancel_event: Any = None) -> None:
    if cancel_event is not None and bool(getattr(cancel_event, "is_set", lambda: False)()):
        raise TimeoutError("圖片正規化已取消。")
    if deadline is not None and time.perf_counter() >= float(deadline):
        raise TimeoutError("圖片正規化已達截止。")


def normalize_capture(
    image_bgr_or_path: np.ndarray | str | Path,
    *,
    max_side: int | None = None,
    enhance: bool | None = None,
    deadline: float | None = None,
    cancel_event: Any = None,
) -> Tuple[np.ndarray | None, Dict[str, Any]]:
    """Return one bounded BGR working image and descriptive metadata.

    The function never raises for invalid input.  A ``None`` image is paired with
    a non-empty ``meta['error']`` so upstream OCR/vision can downgrade safely.
    """
    try:
        configured_max = ANALYSIS_MAX_SIDE if max_side is None else int(max_side)
        bounded_max = max(800, min(2000, configured_max))
    except Exception:
        bounded_max = ANALYSIS_MAX_SIDE

    orientation_applied = False
    try:
        _deadline_guard(deadline, cancel_event)
        if isinstance(image_bgr_or_path, np.ndarray):
            image = _as_bgr(image_bgr_or_path)
        else:
            image, orientation_applied, error = _read_path(Path(image_bgr_or_path))
            if image is None:
                return None, _meta(error=error)
        _deadline_guard(deadline, cancel_event)
        if image is None or image.size == 0:
            return None, _meta(error="圖片內容為空。")

        original_height, original_width = image.shape[:2]
        original_size = (original_width, original_height)
        longest = max(original_height, original_width)
        scale = 1.0
        if longest > bounded_max:
            _deadline_guard(deadline, cancel_event)
            scale = bounded_max / float(longest)
            image = cv2.resize(
                image,
                (
                    max(1, int(round(original_width * scale))),
                    max(1, int(round(original_height * scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
        elif longest < ANALYSIS_SMALL_IMAGE_THRESHOLD:
            _deadline_guard(deadline, cancel_event)
            target = min(bounded_max, ANALYSIS_SMALL_IMAGE_TARGET)
            scale = target / float(max(1, longest))
            image = cv2.resize(
                image,
                (
                    max(1, int(round(original_width * scale))),
                    max(1, int(round(original_height * scale))),
                ),
                interpolation=cv2.INTER_CUBIC,
            )

        use_enhance = ANALYSIS_LIGHT_ENHANCE if enhance is None else bool(enhance)
        if use_enhance:
            _deadline_guard(deadline, cancel_event)
            image = _light_enhance(image)
        _deadline_guard(deadline, cancel_event)
        working_height, working_width = image.shape[:2]
        return image, _meta(
            original_size=original_size,
            working_size=(working_width, working_height),
            scale=scale,
            orientation_applied=orientation_applied,
            enhanced=use_enhance,
        )
    except TimeoutError as exc:
        return None, _meta(error=str(exc))
    except Exception as exc:
        return None, _meta(error=f"圖片正規化失敗：{exc}")


__all__ = ["normalize_capture"]
