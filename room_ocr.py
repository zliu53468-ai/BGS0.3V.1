"""遊戲畫面館別／桌號快速 OCR。

完整手機截圖會優先掃描平台專用的小型資訊區；牌路裁切圖則掃描上緣與整張圖。
快速模式只使用少量預處理版本，並以英數字 allowlist 降低 EasyOCR 延遲。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import os
import re
import time

import cv2
import numpy as np


_EASY_READER: Any = None
_EASY_READER_LOCK = Lock()


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(str(os.getenv(name, default)).strip())
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(str(os.getenv(name, default)).strip())
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


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


OCR_BACKEND = os.getenv("OCR_BACKEND", "easyocr").strip().lower()
OCR_LANGS = [part.strip() for part in os.getenv("OCR_LANGS", "en").split(",") if part.strip()]
OCR_GPU = os.getenv("OCR_GPU", "0").strip() == "1"
OCR_INFO_ROI = _env_roi("OCR_INFO_ROI", (0.0, 0.0, 1.0, 0.30))
OCR_UPSCALE = _env_float("OCR_UPSCALE", 1.35, 1.0, 3.0)
OCR_MIN_CONFIDENCE = _env_float("OCR_MIN_CONFIDENCE", 0.20, 0.0, 1.0)
OCR_MAX_IMAGE_SIDE = _env_int("OCR_MAX_IMAGE_SIDE", 1400, 640, 3000)
OCR_FAST_VARIANTS = _env_int("OCR_FAST_VARIANTS", 1, 1, 3)
OCR_MAX_ROIS = _env_int("OCR_MAX_ROIS", 3, 1, 6)
OCR_ALLOWLIST = os.getenv(
    "OCR_ALLOWLIST", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:-#"
).strip()
OCR_MODEL_DIR = os.getenv("OCR_MODEL_DIR", "").strip() or None
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()

VENUE_INFO_ROIS: Dict[str, Tuple[float, float, float, float]] = {
    "DG": _env_roi("DG_OCR_ROI", (0.68, 0.00, 0.32, 0.28)),
    "MT": _env_roi("MT_OCR_ROI", (0.00, 0.00, 1.00, 0.26)),
    "DB": _env_roi("DB_OCR_ROI", (0.00, 0.00, 1.00, 0.26)),
    "SA": _env_roi("SA_OCR_ROI", (0.00, 0.00, 1.00, 0.26)),
    "OB": _env_roi("OB_OCR_ROI", (0.00, 0.00, 1.00, 0.26)),
    "T9": _env_roi("T9_OCR_ROI", (0.00, 0.00, 1.00, 0.26)),
}

VENUE_ALIASES: Dict[str, Tuple[str, str]] = {
    "MT": ("MT", "MT真人"), "MT真人": ("MT", "MT真人"),
    "DG": ("DG", "DG真人"), "DG真人": ("DG", "DG真人"),
    "DB": ("DB", "DB真人"), "DB真人": ("DB", "DB真人"),
    "SA": ("SA", "SA真人"), "SA真人": ("SA", "SA真人"),
    "OB": ("OB", "歐博真人"), "OB真人": ("OB", "歐博真人"),
    "歐博": ("OB", "歐博真人"), "歐博真人": ("OB", "歐博真人"),
    "欧博": ("OB", "歐博真人"),
    "T9": ("T9", "T9真人"), "T9真人": ("T9", "T9真人"),
}


@dataclass(frozen=True)
class OCRToken:
    text: str
    confidence: float
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class RoomInfo:
    venue_code: str = ""
    venue_name: str = ""
    room: str = ""
    remaining_cards: Optional[int] = None
    backend: str = ""
    raw_text: str = ""
    confidence: float = 0.0


def _read_image(path: str | Path) -> np.ndarray:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"找不到圖片：{file_path}")
    data = np.fromfile(str(file_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("無法讀取圖片。")
    longest = max(image.shape[:2])
    if longest > OCR_MAX_IMAGE_SIDE:
        scale = OCR_MAX_IMAGE_SIDE / float(longest)
        image = cv2.resize(
            image,
            (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image


def _crop_normalized(image: np.ndarray, roi: Sequence[float]) -> Tuple[np.ndarray, Dict[str, int]]:
    height, width = image.shape[:2]
    x, y, roi_width, roi_height = [float(value) for value in roi]
    x1 = max(0, min(width - 1, int(round(x * width))))
    y1 = max(0, min(height - 1, int(round(y * height))))
    x2 = max(x1 + 1, min(width, int(round((x + roi_width) * width))))
    y2 = max(y1 + 1, min(height, int(round((y + roi_height) * height))))
    return image[y1:y2, x1:x2].copy(), {
        "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1,
    }


def _preprocess_variants(crop: np.ndarray, *, fast: bool = True) -> List[Tuple[str, np.ndarray]]:
    upscaled = cv2.resize(
        crop, None, fx=OCR_UPSCALE, fy=OCR_UPSCALE,
        interpolation=cv2.INTER_CUBIC if OCR_UPSCALE > 1.0 else cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    denoised = cv2.bilateralFilter(clahe, 5, 35, 35)
    variants: List[Tuple[str, np.ndarray]] = [("gray_clahe", denoised)]
    if not fast or OCR_FAST_VARIANTS >= 2:
        variants.append(("color", upscaled))
    if not fast or OCR_FAST_VARIANTS >= 3:
        otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        variants.append(("otsu", otsu))
    return variants[: OCR_FAST_VARIANTS if fast else len(variants)]


def _get_easy_reader() -> Any:
    global _EASY_READER
    if _EASY_READER is not None:
        return _EASY_READER
    with _EASY_READER_LOCK:
        if _EASY_READER is not None:
            return _EASY_READER
        import easyocr
        kwargs: Dict[str, Any] = {"gpu": OCR_GPU, "verbose": False}
        if OCR_MODEL_DIR:
            kwargs["model_storage_directory"] = OCR_MODEL_DIR
            kwargs["user_network_directory"] = OCR_MODEL_DIR
        _EASY_READER = easyocr.Reader(OCR_LANGS, **kwargs)
        return _EASY_READER


def preload_ocr() -> bool:
    """Render 啟動時預載 OCR，避免第一位使用者承擔模型載入時間。"""
    try:
        if OCR_BACKEND in {"easyocr", "auto"}:
            _get_easy_reader()
        return True
    except Exception as exc:
        print("OCR preload failed", exc)
        return False


def _tokens_easyocr(image: np.ndarray) -> List[OCRToken]:
    reader = _get_easy_reader()
    kwargs: Dict[str, Any] = {
        "detail": 1, "paragraph": False, "decoder": "greedy",
        "contrast_ths": 0.05, "adjust_contrast": 0.7,
        "text_threshold": 0.45, "low_text": 0.25, "link_threshold": 0.30,
        "mag_ratio": 1.0,
    }
    if OCR_ALLOWLIST:
        kwargs["allowlist"] = OCR_ALLOWLIST
    results = reader.readtext(image, **kwargs)
    tokens: List[OCRToken] = []
    for box, text, confidence in results:
        confidence = float(confidence or 0.0)
        text = str(text or "").strip()
        if not text or confidence < OCR_MIN_CONFIDENCE:
            continue
        xs = [int(round(point[0])) for point in box]
        ys = [int(round(point[1])) for point in box]
        tokens.append(OCRToken(
            text=text, confidence=confidence, x=min(xs), y=min(ys),
            width=max(1, max(xs) - min(xs)), height=max(1, max(ys) - min(ys)),
        ))
    return tokens


def _tokens_tesseract(image: np.ndarray) -> List[OCRToken]:
    import pytesseract
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    config = "--oem 3 --psm 6"
    if OCR_ALLOWLIST:
        config += f" -c tessedit_char_whitelist={OCR_ALLOWLIST}"
    data = pytesseract.image_to_data(
        image, lang=os.getenv("TESSERACT_LANG", "eng"), config=config,
        output_type=pytesseract.Output.DICT,
    )
    tokens: List[OCRToken] = []
    for index, raw_text in enumerate(data.get("text", [])):
        text = str(raw_text or "").strip()
        try:
            confidence = max(0.0, float(data["conf"][index]) / 100.0)
        except Exception:
            confidence = 0.0
        if not text or confidence < OCR_MIN_CONFIDENCE:
            continue
        tokens.append(OCRToken(
            text=text, confidence=confidence, x=int(data["left"][index]), y=int(data["top"][index]),
            width=int(data["width"][index]), height=int(data["height"][index]),
        ))
    return tokens


def _deduplicate_tokens(tokens: Iterable[OCRToken]) -> List[OCRToken]:
    best: Dict[str, OCRToken] = {}
    for token in tokens:
        normalized = re.sub(r"\s+", "", token.text).upper()
        if not normalized:
            continue
        current = best.get(normalized)
        if current is None or token.confidence > current.confidence:
            best[normalized] = token
    return sorted(best.values(), key=lambda token: (token.y, token.x))


def _normalize_text(value: str) -> str:
    replacements = {"Ｏ": "O", "０": "0", "１": "1", "２": "2", "３": "3", "４": "4", "５": "5", "６": "6", "７": "7", "８": "8", "９": "9"}
    text = str(value or "")
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _room_candidate(compact: str, preferred_venue: str = "") -> str:
    patterns: List[str] = []
    venue = str(preferred_venue or "").upper()
    if venue == "DG":
        patterns.extend([r"\bRB\d{1,2}\b", r"\bS\d{1,2}\b", r"\bQ[CD]\b"])
    elif venue == "MT":
        patterns.extend([r"\b(?:BACCARAT)?\d{1,2}[A-Z]?\b", r"\b[NP]?\d{1,2}[A-Z]?\b"])
    patterns.extend([
        r"(?:TABLE|ROOM|DESK)[:#-]?([A-Z]{0,4}\d{1,3}[A-Z]?)",
        r"\b(RB\d{1,2}|S\d{1,2}|Q[CD]|[A-Z]{1,3}\d{1,3}[A-Z]?)\b",
    ])
    for pattern in patterns:
        match = re.search(pattern, compact, flags=re.IGNORECASE)
        if match:
            value = (match.group(1) if match.lastindex else match.group(0)).upper().strip("-:#")
            if value and value not in {"DG", "MT", "DB", "SA", "OB", "T9"}:
                return value
    return ""


def parse_room_info_text(
    text: str,
    *,
    backend: str = "manual",
    confidence: float = 1.0,
    preferred_venue: str = "",
) -> RoomInfo:
    normalized = _normalize_text(text)
    compact = re.sub(r"\s+", "", normalized).upper()
    venue_code = ""
    venue_name = ""
    for code in ("DG", "MT", "DB", "SA", "OB", "T9"):
        if re.search(rf"\b{code}\b", normalized, flags=re.IGNORECASE):
            venue_code, venue_name = VENUE_ALIASES[code]
            break
    preferred = str(preferred_venue or "").upper().strip()
    if not venue_code and preferred in VENUE_ALIASES:
        venue_code, venue_name = VENUE_ALIASES[preferred]

    room = _room_candidate(compact, preferred or venue_code)
    remaining_cards: Optional[int] = None
    for pattern in [
        r"(?:REMAIN(?:ING)?|CARDS?)[:#-]?(\d{1,3})",
        r"(?:剩餘|餘牌|剩牌)[:#-]?(\d{1,3})",
    ]:
        match = re.search(pattern, compact, flags=re.IGNORECASE)
        if match:
            value = int(match.group(1))
            if 4 <= value <= 416:
                remaining_cards = value
                break

    found = [bool(venue_code), bool(room), remaining_cards is not None]
    field_ratio = sum(found) / 3.0
    return RoomInfo(
        venue_code=venue_code, venue_name=venue_name, room=room,
        remaining_cards=remaining_cards, backend=backend, raw_text=normalized.strip(),
        confidence=round(max(0.0, min(1.0, confidence * 0.70 + field_ratio * 0.30)), 4),
    )


def _candidate_rois(preferred_venue: str, input_type: str) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    venue = str(preferred_venue or "").upper().strip()
    mode = str(input_type or "auto").lower().strip()
    values: List[Tuple[str, Tuple[float, float, float, float]]] = []
    if venue in VENUE_INFO_ROIS:
        values.append((f"{venue.lower()}_info", VENUE_INFO_ROIS[venue]))
    values.append(("top_info", OCR_INFO_ROI))
    if mode in {"auto", "road_crop"}:
        values.append(("road_crop_top", (0.0, 0.0, 1.0, 0.45)))
        values.append(("full_image", (0.0, 0.0, 1.0, 1.0)))
    unique: List[Tuple[str, Tuple[float, float, float, float]]] = []
    seen = set()
    for name, roi in values:
        key = tuple(round(v, 4) for v in roi)
        if key not in seen:
            seen.add(key)
            unique.append((name, roi))
    return unique[:OCR_MAX_ROIS]


def analyze_room_info(
    image_path: str | Path,
    preferred_venue: str = "",
    input_type: str = "auto",
    fast: bool = True,
) -> Dict[str, Any]:
    started = time.perf_counter()
    image = _read_image(image_path)
    backends = [OCR_BACKEND] if OCR_BACKEND in {"easyocr", "tesseract"} else ["easyocr", "tesseract"]
    errors: List[str] = []
    all_tokens: List[OCRToken] = []
    used_backend = ""
    used_roi_name = ""
    used_roi_pixels: Dict[str, int] = {}
    used_roi = OCR_INFO_ROI

    for roi_name, roi in _candidate_rois(preferred_venue, input_type):
        crop, roi_pixels = _crop_normalized(image, roi)
        variants = _preprocess_variants(crop, fast=fast)
        for backend in backends:
            backend_tokens: List[OCRToken] = []
            try:
                for _, variant in variants:
                    backend_tokens.extend(_tokens_easyocr(variant) if backend == "easyocr" else _tokens_tesseract(variant))
                backend_tokens = _deduplicate_tokens(backend_tokens)
                if backend_tokens:
                    merged = "\n".join(token.text for token in backend_tokens)
                    avg = sum(token.confidence for token in backend_tokens) / len(backend_tokens)
                    parsed = parse_room_info_text(
                        merged, backend=backend, confidence=avg, preferred_venue=preferred_venue
                    )
                    all_tokens.extend(backend_tokens)
                    used_backend, used_roi_name, used_roi_pixels, used_roi = backend, roi_name, roi_pixels, roi
                    # 找到桌號即可立即結束；完整資訊不需為剩餘張數額外拖延。
                    if parsed.room:
                        result = asdict(parsed)
                        result.update({
                            "ok": True,
                            "tokens": [asdict(token) for token in _deduplicate_tokens(all_tokens)],
                            "roi": used_roi_pixels, "normalized_roi": list(used_roi),
                            "roi_name": used_roi_name, "errors": errors[-4:],
                            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
                            "venue_source": "image_ocr" if parsed.venue_code and parsed.venue_code != str(preferred_venue or "").upper() else "session_selected",
                            "room_source": "image_ocr",
                        })
                        return result
                    break
            except Exception as exc:
                errors.append(f"{roi_name}/{backend}: {exc}")

    tokens = _deduplicate_tokens(all_tokens)
    merged_text = "\n".join(token.text for token in tokens)
    average_confidence = sum(token.confidence for token in tokens) / len(tokens) if tokens else 0.0
    info = parse_room_info_text(
        merged_text, backend=used_backend or "unavailable", confidence=average_confidence,
        preferred_venue=preferred_venue,
    )
    result = asdict(info)
    result.update({
        "ok": bool(info.venue_code or info.room or info.remaining_cards is not None),
        "tokens": [asdict(token) for token in tokens],
        "roi": used_roi_pixels, "normalized_roi": list(used_roi), "roi_name": used_roi_name,
        "errors": errors[-4:],
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "venue_source": "session_selected" if preferred_venue else "unknown",
        "room_source": "image_ocr" if info.room else "unknown",
    })
    return result


__all__ = ["RoomInfo", "analyze_room_info", "parse_room_info_text", "preload_ocr"]
