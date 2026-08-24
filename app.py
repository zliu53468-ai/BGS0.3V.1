"""BGS V10.3 LINE Bot：手機雙模式快速分析＋B/P/T完整歷史＋保守線上校準。

LINE 主流程：選館 -> 開始分析 -> 設定本金 -> 首次上傳截圖 -> 後續只按莊／閒／和。
收到圖片後直接在 Webhook 內完成房間 OCR、大路偵測與模型分析，最長等待 3.5 秒，
成功時使用同一個 replyToken 回覆分析面板；超時則立即回覆重新上傳提示。
後續按莊／閒／和也改為在 Webhook 內以 deadline 同步完成模型更新，直接用同一個 replyToken 回覆下一局面板，不再依賴 LINE Push API。
虛擬牌靴程式仍保留給既有 API 相容用途，但不再混入 LINE 截圖流程。
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import re
import threading
import time
import tempfile
import traceback
import urllib.parse
import unicodedata
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import requests
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import store
from predictor import run_virtual_round
from screen_pipeline import analyze_game_screen
from screenshot_predictor import predict_from_screenshot
from room_ocr import preload_ocr


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TAIPEI_TZ = timezone(timedelta(hours=8))
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
LIFF_ID = os.getenv("LIFF_ID", "").strip()
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "").strip()
ADMIN_LINE_URL = os.getenv(
    "ADMIN_LINE_URL", "https://line.me/R/ti/p/%40jins888"
).strip()
ALLOW_UNSIGNED_WEBHOOK = os.getenv("ALLOW_UNSIGNED_WEBHOOK", "0").strip() == "1"
MAX_CONCURRENT_PREDICTIONS = max(
    1, min(4, int(os.getenv("APP_MAX_CONCURRENT_PREDICTIONS", "1") or "1"))
)
LINE_IMAGE_ANALYSIS_TIMEOUT = max(
    4.0,
    min(25.0, float(os.getenv("LINE_IMAGE_ANALYSIS_TIMEOUT", "12.0") or "12.0")),
)
PREDICTION_QUEUE_TIMEOUT = max(
    5, min(55, int(os.getenv("APP_PREDICTION_QUEUE_TIMEOUT", "45") or "45"))
)
# LINE replyToken 建議在 webhook 收到後盡快使用；保留模型運算與 Reply API 的安全餘量。
MANUAL_OUTCOME_TIMEOUT = max(
    5.0,
    min(45.0, float(os.getenv("MANUAL_OUTCOME_TIMEOUT", "40") or "40")),
)
PUSH_MAX_RETRIES = max(0, min(5, int(os.getenv("PUSH_MAX_RETRIES", "2") or "2")))
PUSH_RETRY_DELAY_SECONDS = max(
    0.2,
    min(3.0, float(os.getenv("PUSH_RETRY_DELAY_SECONDS", "0.8") or "0.8")),
)
LINE_IMAGE_MAX_BYTES = max(
    1_000_000,
    min(20_000_000, int(os.getenv("LINE_IMAGE_MAX_BYTES", "10000000") or "10000000")),
)
_PREDICTION_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_PREDICTIONS)
_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()
_USER_LOCKS: Dict[str, asyncio.Lock] = {}
_USER_IMAGE_LOCKS: Dict[str, threading.Lock] = {}
_USER_IMAGE_LOCKS_GUARD = threading.Lock()
SCREEN_ESTIMATED_CARDS_PER_ROUND = max(
    0,
    min(6, int(os.getenv("SCREEN_ESTIMATED_CARDS_PER_ROUND", "5") or "5")),
)


VENUES: List[Dict[str, str]] = [
    {"code": "DG", "name": "DG真人", "image": "dg.png"},
    {"code": "MT", "name": "MT真人", "image": "mt.png"},
    {"code": "DB", "name": "DB真人", "image": "db.png"},
    {"code": "SA", "name": "SA真人", "image": "sa.png"},
    {"code": "OB", "name": "歐博真人", "image": "ob.png"},
    {"code": "T9", "name": "T9真人", "image": "t9.png"},
]
VENUE_BY_CODE = {venue["code"]: venue for venue in VENUES}


def _normalize_access_code(value: Any) -> str:
    """Normalize LINE text, including common full-width/hidden/homoglyph input."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    for hidden in ("\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"):
        text = text.replace(hidden, "")
    # Common visually identical characters accidentally produced by IMEs.
    text = text.translate(str.maketrans({
        "а": "a",  # Cyrillic small a
        "ɑ": "a",  # Latin alpha
        "Ａ": "A",
        "ａ": "a",
    }))
    text = re.sub(r"[^0-9A-Za-z]", "", text)
    return text.lower()


def _code_set(env_name: str, defaults: str) -> set[str]:
    """Always keep built-in codes and merge any Render environment codes.

    The previous implementation let an empty/stale Render variable replace all
    built-in codes.  This union-based implementation prevents that failure.
    """
    default_codes = {
        _normalize_access_code(item)
        for item in str(defaults or "").split(",")
        if _normalize_access_code(item)
    }
    environment_codes = {
        _normalize_access_code(item)
        for item in str(os.getenv(env_name, "") or "").split(",")
        if _normalize_access_code(item)
    }
    return default_codes | environment_codes


PERMANENT_CODES = _code_set(
    "PERMANENT_CODES", "aaa1688003,aaa1888007,aaa1000889"
)
MONTHLY_CODES = _code_set("MONTHLY_CODES", "aaa13002,aaa15001,aaa199801")
TEMP_CODES = _code_set("TEMP_CODES", "aaaa1999152,aaa345556,aaa987743")
ALL_CODES = PERMANENT_CODES | MONTHLY_CODES | TEMP_CODES


app = FastAPI(
    title="BGS AI預測系統",
    version="10.5.0",
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_LINE_HTTP = requests.Session()
_LINE_HTTP.mount(
    "https://",
    requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0),
)


class UserRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=160)


class VenueRequest(UserRequest):
    venue: str = Field(min_length=1, max_length=12)
    room: str = Field(default="1", min_length=1, max_length=24)


class RoomRequest(UserRequest):
    room: str = Field(min_length=1, max_length=24)


class ShoeCardsRequest(UserRequest):
    cards: List[Any] = Field(default_factory=list, max_length=832)
    replace: bool = True


class LiffSessionStartRequest(VenueRequest):
    shoe_id: str = Field(default="", max_length=80)


class RoundResultRequest(UserRequest):
    result: str = Field(min_length=1, max_length=1)


class ActivationRequest(UserRequest):
    code: str = Field(min_length=1, max_length=160)


class AccessExpiredError(Exception):
    """Raised only when the user's access is genuinely expired."""


def _now() -> datetime:
    return datetime.now(TAIPEI_TZ)


def _venue_name(code: str) -> str:
    return VENUE_BY_CODE.get(str(code), {}).get("name", str(code or "-"))


def _public_asset(path: str) -> str:
    if PUBLIC_BASE_URL:
        return f"{PUBLIC_BASE_URL}{path}"
    # LINE requires an HTTPS URL. The web UI still works locally without this,
    # while the health endpoint warns that PUBLIC_BASE_URL is missing.
    return f"https://dummyimage.com/600x600/111111/f5c542.png&text={urllib.parse.quote(path)}"


def _verify_signature(body: bytes, signature: Optional[str]) -> bool:
    if ALLOW_UNSIGNED_WEBHOOK and not CHANNEL_SECRET:
        return True
    if not CHANNEL_SECRET or not signature:
        return False
    expected = base64.b64encode(
        hmac.new(CHANNEL_SECRET.encode(), body, hashlib.sha256).digest()
    ).decode()
    return hmac.compare_digest(expected, signature)


def _reply(token: str, messages: List[Dict[str, Any]]) -> bool:
    """使用 replyToken 立即回覆；回傳是否成功。"""
    if not token:
        return False
    if not CHANNEL_ACCESS_TOKEN:
        print("LINE reply preview", json.dumps(messages, ensure_ascii=False))
        return False
    response = _LINE_HTTP.post(
        "https://api.line.me/v2/bot/message/reply",
        headers={
            "Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}",
            "Content-Type": "application/json",
        },
        json={"replyToken": token, "messages": messages[:5]},
        timeout=8,
    )
    if response.status_code >= 300:
        print("LINE reply failed", response.status_code, response.text)
        return False
    return True


def _push(
    to: str,
    messages: List[Dict[str, Any]],
    *,
    max_retries: Optional[int] = None,
    retry_delay_seconds: Optional[float] = None,
) -> bool:
    """背景分析完成後以 Push 傳送；失敗會重試，全部失敗回傳 False。"""
    target = str(to or "").strip()
    if not target:
        return False
    if not CHANNEL_ACCESS_TOKEN:
        print("LINE push preview", target, json.dumps(messages, ensure_ascii=False))
        return False

    retries = PUSH_MAX_RETRIES if max_retries is None else max(0, int(max_retries))
    delay = (
        PUSH_RETRY_DELAY_SECONDS
        if retry_delay_seconds is None
        else max(0.0, float(retry_delay_seconds))
    )
    payload = {"to": target, "messages": messages[:5]}
    headers = {
        "Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    last_status = 0
    last_body = ""
    for attempt in range(retries + 1):
        try:
            response = _LINE_HTTP.post(
                "https://api.line.me/v2/bot/message/push",
                headers=headers,
                json=payload,
                timeout=10,
            )
            last_status = int(response.status_code)
            last_body = (response.text or "")[:500]
            if response.status_code < 300:
                if attempt > 0:
                    print(
                        "LINE push recovered",
                        json.dumps(
                            {
                                "uid": target[-8:],
                                "attempt": attempt,
                                "status": last_status,
                            },
                            ensure_ascii=False,
                        ),
                    )
                return True
            print(
                "LINE push failed",
                json.dumps(
                    {
                        "uid": target[-8:],
                        "attempt": attempt,
                        "status": last_status,
                        "body": last_body,
                    },
                    ensure_ascii=False,
                ),
            )
        except Exception as exc:
            last_status = -1
            last_body = str(exc)[:500]
            print(
                "LINE push exception",
                json.dumps(
                    {
                        "uid": target[-8:],
                        "attempt": attempt,
                        "error": last_body,
                    },
                    ensure_ascii=False,
                ),
            )
        if attempt < retries and delay > 0:
            time.sleep(delay)

    print(
        "LINE push exhausted",
        json.dumps(
            {
                "uid": target[-8:],
                "status": last_status,
                "body": last_body,
            },
            ensure_ascii=False,
        ),
    )
    return False


def _schedule_background(coro: Any) -> None:
    """保存背景 Task 參照，避免工作尚未完成就被回收。"""
    task = asyncio.create_task(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


@app.on_event("startup")
async def _preload_ocr_on_startup() -> None:
    """背景預載 OCR；不阻塞 Web Service 啟動。"""
    if os.getenv("OCR_PRELOAD", "0").strip() == "1":
        _schedule_background(asyncio.to_thread(preload_ocr))


def _user_lock(user_id: str) -> asyncio.Lock:
    """同一 UID 的圖片與莊閒按鈕依序處理，不與其他 UID 互相阻塞。"""
    uid = str(user_id or "").strip()
    lock = _USER_LOCKS.get(uid)
    if lock is None:
        lock = asyncio.Lock()
        _USER_LOCKS[uid] = lock
    return lock


def _user_image_lock(user_id: str) -> threading.Lock:
    """圖片同步分析專用鎖；超時後底層執行緒尚在收尾時，避免同 UID 重疊寫入。"""
    uid = str(user_id or "").strip()
    with _USER_IMAGE_LOCKS_GUARD:
        lock = _USER_IMAGE_LOCKS.get(uid)
        if lock is None:
            lock = threading.Lock()
            _USER_IMAGE_LOCKS[uid] = lock
        return lock


def _remaining_image_time(deadline: float) -> float:
    return max(0.0, float(deadline) - time.perf_counter())


def _raise_if_image_timed_out(
    cancel_event: threading.Event,
    deadline: float,
) -> None:
    if cancel_event.is_set() or _remaining_image_time(deadline) <= 0.0:
        raise TimeoutError("LINE 圖片分析已超過 replyToken 時限。")


def _remaining_manual_time(deadline: float) -> float:
    return max(0.0, float(deadline) - time.perf_counter())


def _raise_if_manual_timed_out(
    cancel_event: threading.Event,
    deadline: float,
) -> None:
    if cancel_event.is_set() or _remaining_manual_time(deadline) <= 0.0:
        raise TimeoutError("LINE 手動結果分析已超過 replyToken 時限。")


def _request_bankroll(user_id: str) -> Dict[str, Any]:
    """相容保護：即使 Render 暫時載到舊 store.py，也不直接噴 AttributeError。"""
    handler = getattr(store, "request_bankroll", None)
    if callable(handler):
        return handler(user_id)
    return store.upsert_session(
        user_id,
        {
            "analysis_mode": "screen",
            "analysis_active": True,
            "awaiting_bankroll": True,
            "awaiting_screenshot": False,
            "status": "等待輸入本金",
        },
    )


def _start_new_uid_analysis(user_id: str) -> Dict[str, Any]:
    handler = getattr(store, "start_new_analysis", None)
    if callable(handler):
        return handler(user_id)
    return store.begin_screen_analysis(user_id, clear_existing=True)


def _end_uid_analysis(user_id: str) -> Dict[str, Any]:
    handler = getattr(store, "end_and_clear_analysis", None)
    if callable(handler):
        return handler(user_id)
    return store.end_session(user_id)


def _text(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": str(text)[:5000]}


def _format_money(value: Any) -> str:
    try:
        return f"{max(0, int(value or 0)):,}"
    except Exception:
        return "0"


def _parse_bankroll(text: str) -> int:
    """接受 10000、本金10000、本金：10,000 等格式。"""
    value = str(text or "").strip()
    value = re.sub(r"^(?:本金|金額|資金)\s*[:：=]?\s*", "", value, flags=re.IGNORECASE)
    value = value.replace(",", "").replace("，", "").replace("$", "").replace("＄", "")
    value = re.sub(r"\s+", "", value)
    if not re.fullmatch(r"\d+", value):
        raise ValueError("請輸入純數字本金，例如：10000")
    bankroll = int(value)
    if bankroll < 100:
        raise ValueError("本金至少需輸入 100 元。")
    if bankroll > 100_000_000:
        raise ValueError("本金數字過大，請重新輸入。")
    return bankroll



def _attach_bankroll_advice(
    prediction: Mapping[str, Any],
    session: Mapping[str, Any],
) -> Dict[str, Any]:
    """保留本金欄位相容性，但不把牌路方向分數偽裝成 EV／Kelly 注碼。"""
    result = dict(prediction or {})
    bankroll = max(0, int(session.get("bankroll", 0) or 0))
    edge = float(result.get("direction_edge", 0.0) or 0.0)
    signal_reason = str(result.get("signal_reason") or "模型正在等待更明確的方向差距")

    if bankroll <= 0:
        level = "尚未設定"
        reason = "請先設定本次分析本金"
    else:
        level = "不自動配置"
        reason = f"{signal_reason}；本版本僅提供牌路方向評估，不使用 EV／Kelly 自動下注。"

    result.update(
        {
            "bankroll": bankroll,
            "suggested_bet_amount": 0,
            "bet_percentage": 0.0,
            "bet_level_text": level,
            "bet_reason": reason,
            "screen_edge": round(edge, 6),
            "selected_expected_return": 0.0,
            "selected_expected_return_percent": 0.0,
            "kelly_percentage_applied": 0.0,
        }
    )
    return result


def _road_quick_reply() -> Dict[str, Any]:
    """首次圖片完成後，後續只保留莊／閒／和三個實際結果按鈕。"""
    return {
        "items": [
            {
                "type": "action",
                "action": {
                    "type": "postback", "label": "🔴 本局：莊",
                    "data": "action=road_append&result=B", "displayText": "🔴 本局：莊",
                },
            },
            {
                "type": "action",
                "action": {
                    "type": "postback", "label": "🔵 本局：閒",
                    "data": "action=road_append&result=P", "displayText": "🔵 本局：閒",
                },
            },
            {
                "type": "action",
                "action": {
                    "type": "postback", "label": "🟢 本局：和",
                    "data": "action=road_append&result=T", "displayText": "🟢 本局：和",
                },
            },
        ]
    }


def _derive_road_state(raw_values: List[str]) -> Dict[str, Any]:
    raw = [str(item).upper().strip() for item in raw_values if str(item).upper().strip() in {"B", "P", "T"}][-1000:]
    road: List[str] = []
    markers: Dict[str, int] = {}
    pending_opening = 0
    for value in raw:
        if value in {"B", "P"}:
            road.append(value)
            if pending_opening:
                key = str(len(road) - 1)
                markers[key] = markers.get(key, 0) + pending_opening
                pending_opening = 0
        elif road:
            key = str(len(road) - 1)
            markers[key] = markers.get(key, 0) + 1
        else:
            pending_opening += 1
    return {
        "raw_outcomes": raw,
        "road_sequence": road[-500:],
        "tie_markers": markers,
        "tie_total": sum(1 for value in raw if value == "T"),
        "pending_opening_ties": pending_opening,
    }

def _road_text_message(
    session: Mapping[str, Any],
    *,
    notice: str = "",
) -> Dict[str, Any]:
    """建立簡潔、正式的牌路狀態訊息。"""
    sequence = [
        str(item).upper()
        for item in list(session.get("road_sequence") or [])
        if str(item).upper() in {"B", "P"}
    ]
    raw_outcomes = [
        str(item).upper()
        for item in list(session.get("raw_outcomes") or sequence)
        if str(item).upper() in {"B", "P", "T"}
    ]
    analysis = dict(session.get("road_last_analysis") or {})
    compact = "".join({"B": "莊", "P": "閒", "T": "和"}[item] for item in raw_outcomes[-24:])
    if len(sequence) > 24:
        compact = "…" + compact
    if not compact:
        compact = "尚未建立"

    lines: List[str] = []
    if notice:
        lines.append(str(notice))
    lines.extend([
        f"完整牌局：{len(raw_outcomes)} 局｜大路樣本：{len(sequence)} 局｜和局：{sum(1 for item in raw_outcomes if item == 'T')} 局",
        f"近期序列：{compact}",
    ])
    if analysis:
        lines.extend([
            f"方向評估：{analysis.get('direction_text') or '-'}",
            f"訊號狀態：{analysis.get('signal_status_text') or analysis.get('action_text') or '暫緩'}",
            f"模型信心：{analysis.get('confidence_label') or '偏低'}",
        ])
    lines.append("請使用下方按鈕回報本局莊／閒／和，系統將結算上一筆預測並更新下一局分析。")
    return {
        "type": "text",
        "text": "\n".join(lines)[:5000],
        "quickReply": _road_quick_reply(),
    }



def _road_error_message(message: str) -> Dict[str, Any]:
    return {
        "type": "text",
        "text": (
            f"⚠️ 畫面資料尚未完整建立\n{message}\n\n"
            "請重新傳送包含完整大路區域的清晰截圖。"
        )[:5000],
    }


def _download_line_image(
    message_id: str,
    *,
    timeout_seconds: Optional[float] = None,
) -> Path:
    """從 LINE Content API 下載圖片到短期暫存檔。"""
    if not CHANNEL_ACCESS_TOKEN:
        raise RuntimeError("尚未設定 LINE_CHANNEL_ACCESS_TOKEN。")
    message_id = str(message_id or "").strip()
    if not message_id:
        raise ValueError("LINE 圖片 messageId 不存在。")

    if timeout_seconds is None:
        request_timeout = (5.0, 25.0)
    else:
        available = max(0.25, float(timeout_seconds))
        request_timeout = (
            max(0.20, min(1.25, available)),
            max(0.20, min(2.50, available)),
        )

    response = _LINE_HTTP.get(
        f"https://api-data.line.me/v2/bot/message/{message_id}/content",
        headers={"Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}"},
        stream=True,
        timeout=request_timeout,
    )
    if response.status_code >= 300:
        raise RuntimeError(
            f"LINE 圖片下載失敗（HTTP {response.status_code}）。"
        )

    content_type = str(response.headers.get("Content-Type") or "").lower()
    suffix = (
        ".png"
        if "png" in content_type
        else ".webp"
        if "webp" in content_type
        else ".jpg"
    )
    expected_length = int(response.headers.get("Content-Length") or 0)
    if expected_length > LINE_IMAGE_MAX_BYTES:
        raise ValueError("圖片檔案過大，請裁切路紙區域後再傳送。")

    temporary = tempfile.NamedTemporaryFile(
        prefix="line_road_",
        suffix=suffix,
        delete=False,
    )
    total = 0
    try:
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > LINE_IMAGE_MAX_BYTES:
                raise ValueError("圖片檔案過大，請裁切路紙區域後再傳送。")
            temporary.write(chunk)
        temporary.flush()
    except Exception:
        temporary.close()
        Path(temporary.name).unlink(missing_ok=True)
        raise
    finally:
        response.close()
    temporary.close()

    if total <= 0:
        Path(temporary.name).unlink(missing_ok=True)
        raise ValueError("LINE 回傳了空白圖片。")
    return Path(temporary.name)



def _prepare_analysis_image(source_path: Path) -> Path:
    """保留原圖比例，只在解析度偏低時放大並轉存 PNG。

    不改變 HSV 色相、不做銳化或強烈增豔，避免破壞紅／藍／綠牌路辨識。
    高解析度原圖直接沿用，避免多餘重編碼。
    """
    path = Path(source_path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return path

    height, width = image.shape[:2]
    long_side = max(height, width)
    short_side = min(height, width)

    # LINE 壓縮後若圖片偏小，等比例放大，讓自動聚焦與格線分析保留更多像素。
    target_long_side = 1800
    target_short_side = 900
    scale = max(
        1.0,
        target_long_side / max(1.0, float(long_side)),
        target_short_side / max(1.0, float(short_side)),
    )
    scale = min(scale, 2.5)

    if scale <= 1.05:
        return path

    resized = cv2.resize(
        image,
        (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        ),
        interpolation=cv2.INTER_CUBIC,
    )
    prepared = path.with_name(f"{path.stem}_analysis.png")
    ok, encoded = cv2.imencode(".png", resized)
    if not ok:
        return path
    encoded.tofile(str(prepared))
    return prepared



def _postback_button(
    label: str,
    action_name: str,
    *,
    style: str = "primary",
    color: str = "#F1C232",
    **kwargs: str,
) -> Dict[str, Any]:
    data = {"action": action_name, **{key: str(value) for key, value in kwargs.items()}}
    return {
        "type": "button",
        "style": style,
        "height": "sm",
        "color": color if style == "primary" else None,
        "action": {
            "type": "postback",
            "label": label[:20],
            "data": urllib.parse.urlencode(data),
            "displayText": label[:300],
        },
    }


def _clean_flex(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _clean_flex(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_clean_flex(item) for item in value]
    return value



def guide_panel() -> Dict[str, Any]:
    """結束分析或首次加入時顯示的 BGS AI 使用指南。"""
    return _clean_flex(
        {
            "type": "flex",
            "altText": "BGS AI預測系統使用指南",
            "contents": {
                "type": "bubble",
                "size": "mega",
                "body": {
                    "type": "box",
                    "layout": "vertical",
                    "backgroundColor": "#FFFFFF",
                    "paddingAll": "18px",
                    "contents": [
                        {
                            "type": "text",
                            "text": "BGS AI預測系統",
                            "weight": "bold",
                            "size": "xl",
                            "color": "#3E3100",
                        },
                        {
                            "type": "text",
                            "text": "BGS AI 預測系統使用指南",
                            "weight": "bold",
                            "size": "xl",
                            "margin": "sm",
                            "color": "#3E3100",
                        },
                        {
                            "type": "text",
                            "text": (
                                "📍 操作 3 步驟\n"
                                "同桌號：請確保程式選擇與平台一致的桌號資訊。\n"
                                "數據校正：先讓系統預測 3～5 顆，再開始下注。\n"
                                "跟隨訊號：依程式顯示的下一局方向評估操作。"
                            ),
                            "wrap": True,
                            "margin": "md",
                            "color": "#3E3100",
                        },
                        {
                            "type": "separator",
                            "margin": "md",
                            "color": "#F1B900",
                        },
                        {
                            "type": "text",
                            "text": (
                                "⚠️ 贏家 4 守則\n"
                                "專屬平台：僅限配合平台使用，非合作平台數據會產生誤差。\n"
                                "資金規劃：請將本金分成 20～30 等份，穩定下注。\n"
                                "紀律停利：每次獲利達標即離場，科學下注、不戀戰。\n"
                                "裝置綁定：程式已與您的 LINE 帳號綁定，無法轉借他人或跨裝置使用。"
                            ),
                            "wrap": True,
                            "margin": "md",
                            "color": "#3E3100",
                        },
                        {
                            "type": "box",
                            "layout": "vertical",
                            "margin": "lg",
                            "contents": [
                                _postback_button(
                                    "開始分析",
                                    "start_guide",
                                    color="#FFD400",
                                )
                            ],
                        },
                    ],
                },
            },
        }
    )


def selected_venue_panel(session: Mapping[str, Any]) -> Dict[str, Any]:
    venue_name = _venue_name(str(session.get("venue") or ""))
    return _clean_flex(
        {
            "type": "flex",
            "altText": f"已選擇：{venue_name}",
            "contents": {
                "type": "bubble",
                "size": "mega",
                "body": {
                    "type": "box",
                    "layout": "vertical",
                    "backgroundColor": "#FFF4B8",
                    "paddingAll": "18px",
                    "contents": [
                        {
                            "type": "text",
                            "text": "館別選擇完成",
                            "weight": "bold",
                            "size": "xl",
                            "color": "#7B5600",
                        },
                        {
                            "type": "text",
                            "text": f"目前選擇：{venue_name}\n桌號：{session.get('room') or '1'}",
                            "wrap": True,
                            "margin": "md",
                            "color": "#3E3100",
                        },
                        {
                            "type": "text",
                            "text": "下一步請輸入本次分析本金，系統會依正式訊號提供配置建議。",
                            "wrap": True,
                            "margin": "md",
                            "color": "#665000",
                        },
                    ],
                },
            },
        }
    )



def venue_panel(user_id: str) -> Dict[str, Any]:
    bubbles: List[Dict[str, Any]] = []
    for venue in VENUES:
        bubbles.append(
            {
                "type": "bubble",
                "size": "kilo",
                "hero": {
                    "type": "image",
                    "url": _public_asset(f"/static/venues/{venue['image']}"),
                    "size": "full",
                    "aspectRatio": "1:1",
                    "aspectMode": "cover",
                },
                "body": {
                    "type": "box",
                    "layout": "vertical",
                    "backgroundColor": "#FFF4B8",
                    "paddingAll": "12px",
                    "contents": [
                        {
                            "type": "text",
                            "text": venue["name"],
                            "weight": "bold",
                            "align": "center",
                            "color": "#4C3900",
                        },
                        _postback_button(
                            "選擇",
                            "venue",
                            venue=venue["code"],
                            room="1",
                        ),
                    ],
                },
            }
        )
    return _clean_flex(
        {
            "type": "flex",
            "altText": "BGS AI預測系統－請選擇遊戲館",
            "contents": {"type": "carousel", "contents": bubbles},
        }
    )




def manual_result_received_panel(outcome: str) -> Dict[str, Any]:
    value = str(outcome or "").upper()
    label, color = {
        "B": ("莊", "#D52B2B"),
        "P": ("閒", "#2667D8"),
        "T": ("和", "#159447"),
    }.get(value, ("未知", "#7B5600"))
    return _clean_flex(
        {
            "type": "flex",
            "altText": f"本局結果已更新：{label}",
            "contents": {
                "type": "bubble",
                "body": {
                    "type": "box", "layout": "vertical", "backgroundColor": "#FFF4B8", "paddingAll": "18px",
                    "contents": [
                        {"type": "text", "text": f"本局結果已更新：{label}", "weight": "bold", "size": "xl", "color": color},
                        {"type": "text", "text": "系統正在結算上一筆預測，並同步 B/P/T 完整歷史、牌路模型與下一局三方機率。", "wrap": True, "margin": "md", "color": "#4C3900"},
                    ],
                },
            },
        }
    )

def ready_panel(user_id: str, session: Mapping[str, Any]) -> Dict[str, Any]:
    venue = _venue_name(str(session.get("venue") or ""))
    bankroll = int(session.get("bankroll", 0) or 0)
    bankroll_text = f"{_format_money(bankroll)} 元" if bankroll > 0 else "尚未設定"
    body = {
        "type": "box",
        "layout": "vertical",
        "backgroundColor": "#FFF4B8",
        "paddingAll": "18px",
        "contents": [
            {
                "type": "text",
                "text": "BGS AI預測系統",
                "weight": "bold",
                "size": "xl",
                "color": "#7B5600",
            },
            {
                "type": "text",
                "text": (
                    f"分析館別：{venue}\n"
                    f"桌號：{session.get('room') or '1'}\n"
                    f"資金設定：{bankroll_text}\n\n"
                    "選擇館別後請設定本金，再上傳最新完整遊戲畫面。首次辨識完成後，每局只需回報莊／閒／和。"
                ),
                "wrap": True,
                "margin": "md",
                "color": "#4C3900",
            },
            {
                "type": "box",
                "layout": "vertical",
                "spacing": "sm",
                "margin": "lg",
                "contents": [
                    _postback_button("開始牌局分析", "start_screen"),
                    _postback_button("設定／調整本金", "change_bankroll", color="#E29B19"),
                    _postback_button("重新選擇館別", "venues", style="secondary"),
                ],
            },
        ],
    }
    return _clean_flex(
        {
            "type": "flex",
            "altText": "BGS AI預測系統",
            "contents": {"type": "bubble", "size": "mega", "body": body},
        }
    )



def bankroll_panel(user_id: str, session: Mapping[str, Any]) -> Dict[str, Any]:
    current = int(session.get("bankroll", 0) or 0)
    current_text = f"目前設定：{_format_money(current)} 元\n" if current > 0 else ""
    return _clean_flex(
        {
            "type": "flex",
            "altText": "設定分析本金",
            "contents": {
                "type": "bubble",
                "size": "mega",
                "body": {
                    "type": "box",
                    "layout": "vertical",
                    "backgroundColor": "#FFF4B8",
                    "paddingAll": "18px",
                    "contents": [
                        {
                            "type": "text",
                            "text": "設定分析本金",
                            "weight": "bold",
                            "size": "xl",
                            "color": "#7B5600",
                        },
                        {
                            "type": "text",
                            "text": (
                                f"{current_text}請直接輸入金額，例如：10000\n\n"
                                "系統會依方向訊號、模型一致度與風險區間計算建議配置。"
                            ),
                            "wrap": True,
                            "margin": "md",
                            "color": "#4C3900",
                        },
                        _postback_button("返回館別選單", "venues", style="secondary"),
                    ],
                },
            },
        }
    )



def upload_request_panel(user_id: str, session: Mapping[str, Any]) -> Dict[str, Any]:
    return _clean_flex(
        {
            "type": "flex",
            "altText": "上傳牌局畫面",
            "contents": {
                "type": "bubble",
                "size": "mega",
                "body": {
                    "type": "box",
                    "layout": "vertical",
                    "backgroundColor": "#FFF4B8",
                    "paddingAll": "18px",
                    "contents": [
                        {
                            "type": "text",
                            "text": "建立牌局資料",
                            "weight": "bold",
                            "size": "xl",
                            "color": "#7B5600",
                        },
                        {
                            "type": "text",
                            "text": (
                                f"館別：{_venue_name(str(session.get('venue') or ''))}\n"
                                f"桌號：{session.get('room') or '1'}\n"
                                f"本金：{_format_money(session.get('bankroll', 0))} 元\n\n"
                                "請開始上傳最新完整遊戲畫面進行分析。建議保留完整大路區域，避免裁掉左上起始格與六列格線。"
                            ),
                            "wrap": True,
                            "margin": "md",
                            "color": "#4C3900",
                        },
                        {
                            "type": "text",
                            "text": "首次畫面完成後，系統將建立初始牌路；後續每局只需回報實際開出莊或閒。",
                            "wrap": True,
                            "size": "sm",
                            "margin": "md",
                            "color": "#806A2A",
                        },
                        {
                            "type": "box",
                            "layout": "vertical",
                            "spacing": "sm",
                            "margin": "lg",
                            "contents": [
                                _postback_button("調整本金", "change_bankroll", color="#E29B19"),
                                _postback_button("重新選擇館別", "venues", style="secondary"),
                                _postback_button("結束本次分析", "end", style="secondary"),
                            ],
                        },
                    ],
                },
            },
        }
    )



def image_received_panel(session: Mapping[str, Any]) -> Dict[str, Any]:
    return _clean_flex(
        {
            "type": "flex",
            "altText": "牌局畫面已接收",
            "contents": {
                "type": "bubble",
                "body": {
                    "type": "box",
                    "layout": "vertical",
                    "backgroundColor": "#FFF4B8",
                    "paddingAll": "18px",
                    "contents": [
                        {
                            "type": "text",
                            "text": "牌局畫面已接收",
                            "weight": "bold",
                            "size": "xl",
                            "color": "#7B5600",
                        },
                        {
                            "type": "text",
                            "text": (
                                "系統將自動判斷完整遊戲畫面或牌路裁切圖，並同步處理：\n"
                                "• 館別與桌號快速辨識\n"
                                "• 大路莊閒序列建立\n"
                                "• 牌路先行與統一機率模型\n\n"
                                "分析完成後會自動推送下一局面板；後續每局只需點選莊或閒。"
                            ),
                            "wrap": True,
                            "margin": "md",
                            "color": "#4C3900",
                        },
                    ],
                },
            },
        }
    )


def result_panel(user_id: str, session: Mapping[str, Any]) -> Dict[str, Any]:
    prediction = dict(session.get("last_prediction") or {})
    stats = dict(session.get("stats") or {})
    recommend = str(prediction.get("recommend_text") or "-")
    action = str(prediction.get("action_text") or "觀望")
    result_text = str(prediction.get("virtual_outcome_text") or "-")
    verdict = str(prediction.get("verdict_text") or "-")
    round_number = int(session.get("hand_number", 0) or 0)

    body_contents: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": f"分析結果 #{round_number}",
            "weight": "bold",
            "size": "xl",
            "color": "#7B5600",
        },
        {
            "type": "text",
            "text": (
                f"館別：{_venue_name(str(session.get('venue') or ''))}｜桌號：{session.get('room') or '1'}\n"
                f"牌靴：{str(session.get('shoe_id') or '-')[:12]}｜剩餘：{len(session.get('virtual_shoe') or [])} 張"
            ),
            "wrap": True,
            "size": "sm",
            "margin": "sm",
            "color": "#665000",
        },
        {
            "type": "separator",
            "margin": "md",
            "color": "#E1BD43",
        },
        {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "margin": "md",
            "contents": [
                {
                    "type": "text",
                    "text": f"莊　{float(prediction.get('banker_rate', 0.0)):.2f}%",
                    "color": "#D52B2B",
                    "weight": "bold",
                },
                {
                    "type": "text",
                    "text": f"閒　{float(prediction.get('player_rate', 0.0)):.2f}%",
                    "color": "#2667D8",
                    "weight": "bold",
                },
                {
                    "type": "text",
                    "text": f"和　{float(prediction.get('tie_rate', 0.0)):.2f}%",
                    "color": "#259B55",
                    "weight": "bold",
                },
            ],
        },
        {
            "type": "text",
            "text": (
                f"分析方向：{recommend}\n"
                f"訊號：{action}｜品質：{prediction.get('confidence_label') or '偏低'}\n"
                f"核心：超幾何分布＋粒子/蒙地卡羅驗證\n"
                f"虛擬開獎：{result_text}｜{verdict}\n"
                f"累計：{stats.get('wins', 0)} 勝 / {stats.get('losses', 0)} 負 / "
                f"{stats.get('ties_skipped', 0)} 和不計 / {stats.get('observes', 0)} 觀望"
            ),
            "wrap": True,
            "margin": "md",
            "color": "#3E3100",
        },
        {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "margin": "lg",
            "contents": [
                _postback_button("繼續分析", "predict"),
                _postback_button("重新建立牌靴", "reset", color="#E29B19"),
                _postback_button("結束分析", "end", style="secondary"),
            ],
        },
        {
            "type": "text",
            "text": "僅分析程式內建虛擬牌靴，未連接外部真人桌。",
            "wrap": True,
            "size": "xs",
            "margin": "md",
            "color": "#806A2A",
        },
    ]
    return _clean_flex(
        {
            "type": "flex",
            "altText": f"分析結果：{recommend}",
            "contents": {
                "type": "bubble",
                "size": "mega",
                "body": {
                    "type": "box",
                    "layout": "vertical",
