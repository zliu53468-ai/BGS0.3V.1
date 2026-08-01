"""遊戲畫面房間資訊 OCR 模組。

目標欄位：
- 館別：MT真人、DG真人、DB真人、SA真人、歐博真人、T9真人
- 桌號
- 剩餘張數

設計重點：
1. 只裁切畫面上方資訊區，避免路紙與按鈕干擾。
2. 同時產生多種 OpenCV 預處理版本，提高深色背景、光暈與小字容錯。
3. OCR 後再用規則解析，不直接相信單一 OCR 行。
4. EasyOCR 預設採延遲載入；也支援 Tesseract 備援。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import os
import re

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


OCR_BACKEND = os.getenv("OCR_BACKEND", "auto").strip().lower()
OCR_LANGS = [part.strip() for part in os.getenv("OCR_LANGS", "ch_tra,en").split(",") if part.strip()]
OCR_GPU = os.getenv("OCR_GPU", "0").strip() == "1"
OCR_INFO_ROI = _parse_roi(os.getenv("OCR_INFO_ROI", "0,0,1,0.32"), (0.0, 0.0, 1.0, 0.32))
OCR_UPSCALE = _env_float("OCR_UPSCALE", 2.2, 1.0, 4.0)
OCR_MIN_CONFIDENCE = _env_float("OCR_MIN_CONFIDENCE", 0.22, 0.0, 1.0)
OCR_MAX_IMAGE_SIDE = _env_int("OCR_MAX_IMAGE_SIDE", 2400, 640, 5000)
OCR_MODEL_DIR = os.getenv("OCR_MODEL_DIR", "").strip() or None
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()


VENUE_ALIASES: Dict[str, Tuple[str, str]] = {
    "MT": ("MT", "MT真人"),
    "MT真人": ("MT", "MT真人"),
    "DG": ("DG", "DG真人"),
    "DG真人": ("DG", "DG真人"),
    "DB": ("DB", "DB真人"),
    "DB真人": ("DB", "DB真人"),
    "SA": ("SA", "SA真人"),
    "SA真人": ("SA", "SA真人"),
    "OB": ("OB", "歐博真人"),
    "OB真人": ("OB", "歐博真人"),
    "歐博": ("OB", "歐博真人"),
    "歐博真人": ("OB", "歐博真人"),
    "欧博": ("OB", "歐博真人"),
    "T9": ("T9", "T9真人"),
    "T9真人": ("T9", "T9真人"),
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
    return image[y1:y2, x1:x2].copy(), {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}


def _preprocess_variants(crop: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    upscaled = cv2.resize(
        crop,
        None,
        fx=OCR_UPSCALE,
        fy=OCR_UPSCALE,
        interpolation=cv2.INTER_CUBIC,
    )
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    denoised = cv2.bilateralFilter(clahe, 7, 45, 45)
    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )
    otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return [
        ("color", upscaled),
        ("gray_clahe", denoised),
        ("adaptive", adaptive),
        ("adaptive_inverse", cv2.bitwise_not(adaptive)),
        ("otsu", otsu),
    ]


def _get_easy_reader() -> Any:
    global _EASY_READER
    if _EASY_READER is not None:
        return _EASY_READER
    with _EASY_READER_LOCK:
        if _EASY_READER is not None:
            return _EASY_READER
        import easyocr  # 延遲匯入，避免每次啟動都載入模型。

        kwargs: Dict[str, Any] = {"gpu": OCR_GPU, "verbose": False}
        if OCR_MODEL_DIR:
            kwargs["model_storage_directory"] = OCR_MODEL_DIR
            kwargs["user_network_directory"] = OCR_MODEL_DIR
        _EASY_READER = easyocr.Reader(OCR_LANGS, **kwargs)
        return _EASY_READER


def _tokens_easyocr(image: np.ndarray) -> List[OCRToken]:
    reader = _get_easy_reader()
    results = reader.readtext(
        image,
        detail=1,
        paragraph=False,
        decoder="greedy",
        contrast_ths=0.05,
        adjust_contrast=0.7,
        text_threshold=0.45,
        low_text=0.25,
        link_threshold=0.30,
        mag_ratio=1.0,
    )
    tokens: List[OCRToken] = []
    for box, text, confidence in results:
        confidence = float(confidence or 0.0)
        text = str(text or "").strip()
        if not text or confidence < OCR_MIN_CONFIDENCE:
            continue
        xs = [int(round(point[0])) for point in box]
        ys = [int(round(point[1])) for point in box]
        tokens.append(
            OCRToken(
                text=text,
                confidence=confidence,
                x=min(xs),
                y=min(ys),
                width=max(1, max(xs) - min(xs)),
                height=max(1, max(ys) - min(ys)),
            )
        )
    return tokens


def _tokens_tesseract(image: np.ndarray) -> List[OCRToken]:
    import pytesseract

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    data = pytesseract.image_to_data(
        image,
        lang=os.getenv("TESSERACT_LANG", "chi_tra+eng"),
        config="--oem 3 --psm 6",
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
        tokens.append(
            OCRToken(
                text=text,
                confidence=confidence,
                x=int(data["left"][index]),
                y=int(data["top"][index]),
                width=int(data["width"][index]),
                height=int(data["height"][index]),
            )
        )
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
    replacements = {
        "賸餘": "剩餘",
        "剩余": "剩餘",
        "桌台": "桌號",
        "房間": "桌號",
        "臺": "台",
        "Ｏ": "O",
        "０": "0",
        "１": "1",
        "２": "2",
        "３": "3",
        "４": "4",
        "５": "5",
        "６": "6",
        "７": "7",
        "８": "8",
        "９": "9",
    }
    text = str(value or "")
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def parse_room_info_text(text: str, *, backend: str = "manual", confidence: float = 1.0) -> RoomInfo:
    """將 OCR 文字解析成館別、桌號與剩餘張數。"""
    normalized = _normalize_text(text)
    compact = re.sub(r"\s+", "", normalized).upper()

    venue_code = ""
    venue_name = ""
    # 先找含「真人」或中文名稱的強特徵，避免 SA/DB 等短字串誤判。
    strong_aliases = [
        alias
        for alias in VENUE_ALIASES
        if len(alias) > 2 or "真人" in alias or alias in {"歐博", "欧博"}
    ]
    for alias in sorted(strong_aliases, key=len, reverse=True):
        if alias.upper() in compact:
            venue_code, venue_name = VENUE_ALIASES[alias]
            break
    if not venue_code:
        for code in ("MT", "DG", "DB", "SA", "OB", "T9"):
            context_pattern = rf"(?:館別|場館|VENUE)\s*[:：#-]?\s*{re.escape(code)}\b"
            line_pattern = rf"(?m)^\s*{re.escape(code)}\s*$"
            if re.search(context_pattern, normalized, flags=re.IGNORECASE) or re.search(
                line_pattern, normalized, flags=re.IGNORECASE
            ):
                venue_code, venue_name = VENUE_ALIASES[code]
                break

    room = ""
    room_patterns = [
        r"(?:桌號|桌台|桌|房號)\s*[:：#-]?\s*([A-Z0-9-]{1,12})",
        r"(?:TABLE|ROOM)\s*[:：#-]?\s*([A-Z0-9-]{1,12})",
    ]
    for pattern in room_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            room = match.group(1).strip()
            break

    remaining_cards: Optional[int] = None
    remaining_patterns = [
        r"(?:剩餘(?:牌數|張數)?|餘牌|剩牌)\s*[:：#-]?\s*(\d{1,3})\s*張?",
        r"(?:REMAIN(?:ING)?|CARDS?)\s*[:：#-]?\s*(\d{1,3})",
    ]
    for pattern in remaining_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            value = int(match.group(1))
            if 4 <= value <= 416:
                remaining_cards = value
                break

    found = [bool(venue_code), bool(room), remaining_cards is not None]
    field_ratio = sum(found) / 3.0
    return RoomInfo(
        venue_code=venue_code,
        venue_name=venue_name,
        room=room,
        remaining_cards=remaining_cards,
        backend=backend,
        raw_text=normalized.strip(),
        confidence=round(max(0.0, min(1.0, confidence * 0.65 + field_ratio * 0.35)), 4),
    )


def analyze_room_info(image_path: str | Path) -> Dict[str, Any]:
    """辨識遊戲畫面上方房間資訊區。"""
    image = _read_image(image_path)
    crop, roi_pixels = _crop_normalized(image, OCR_INFO_ROI)
    variants = _preprocess_variants(crop)

    backends = [OCR_BACKEND] if OCR_BACKEND in {"easyocr", "tesseract"} else ["easyocr", "tesseract"]
    errors: List[str] = []
    all_tokens: List[OCRToken] = []
    used_backend = ""

    for backend in backends:
        backend_tokens: List[OCRToken] = []
        try:
            for _, variant in variants:
                if backend == "easyocr":
                    backend_tokens.extend(_tokens_easyocr(variant))
                else:
                    backend_tokens.extend(_tokens_tesseract(variant))
            backend_tokens = _deduplicate_tokens(backend_tokens)
            if backend_tokens:
                all_tokens = backend_tokens
                used_backend = backend
                break
        except Exception as exc:
            errors.append(f"{backend}: {exc}")

    merged_text = "\n".join(token.text for token in all_tokens)
    average_confidence = (
        sum(token.confidence for token in all_tokens) / len(all_tokens)
        if all_tokens
        else 0.0
    )
    info = parse_room_info_text(
        merged_text,
        backend=used_backend or "unavailable",
        confidence=average_confidence,
    )
    result = asdict(info)
    result.update(
        {
            "ok": bool(info.venue_code or info.room or info.remaining_cards is not None),
            "tokens": [asdict(token) for token in all_tokens],
            "roi": roi_pixels,
            "normalized_roi": list(OCR_INFO_ROI),
            "errors": errors[-4:],
        }
    )
    return result


__all__ = [
    "RoomInfo",
    "analyze_room_info",
    "parse_room_info_text",
]
