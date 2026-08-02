"""BGS V10.3 LINE Bot：手機雙模式快速分析＋B/P/T完整歷史＋保守線上校準。

LINE 主流程：選館 -> 開始分析 -> 設定本金 -> 首次上傳截圖 -> 後續只按莊／閒／和。
收到圖片後先立即回覆「圖片已收到」，再於背景平行執行房間 OCR 與大路偵測，
首次辨識後先建立牌路 context，再交給超幾何＋粒子／蒙地卡羅核心統一判斷；後續按莊／閒／和持續更新同一 UID。
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
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import store
from predictor import run_virtual_round
from screen_pipeline import analyze_game_screen
from screenshot_predictor import predict_from_screenshot
from room_ocr import preload_ocr
from performance_tracker import resolve_latest_prediction


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TAIPEI_TZ = timezone(timedelta(hours=8))
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "").strip()
ADMIN_LINE_URL = os.getenv(
    "ADMIN_LINE_URL", "https://line.me/R/ti/p/%40jins888"
).strip()
ALLOW_UNSIGNED_WEBHOOK = os.getenv("ALLOW_UNSIGNED_WEBHOOK", "0").strip() == "1"
MAX_CONCURRENT_PREDICTIONS = max(
    1, min(4, int(os.getenv("APP_MAX_CONCURRENT_PREDICTIONS", "1") or "1"))
)
PREDICTION_QUEUE_TIMEOUT = max(
    5, min(55, int(os.getenv("APP_PREDICTION_QUEUE_TIMEOUT", "45") or "45"))
)
LINE_IMAGE_MAX_BYTES = max(
    1_000_000,
    min(20_000_000, int(os.getenv("LINE_IMAGE_MAX_BYTES", "10000000") or "10000000")),
)
_PREDICTION_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_PREDICTIONS)
_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()
_USER_LOCKS: Dict[str, asyncio.Lock] = {}
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
    title="BGS V10.4 1CPU Fast Screen Bot",
    version="9.7.2",
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


def _push(to: str, messages: List[Dict[str, Any]]) -> bool:
    """背景分析完成後，以 Push Message 主動傳送面板。"""
    target = str(to or "").strip()
    if not target:
        return False
    if not CHANNEL_ACCESS_TOKEN:
        print("LINE push preview", target, json.dumps(messages, ensure_ascii=False))
        return False
    response = _LINE_HTTP.post(
        "https://api.line.me/v2/bot/message/push",
        headers={
            "Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}",
            "Content-Type": "application/json",
        },
        json={"to": target, "messages": messages[:5]},
        timeout=10,
    )
    if response.status_code >= 300:
        print("LINE push failed", response.status_code, response.text)
        return False
    return True


def _schedule_background(coro: Any) -> None:
    """保存背景 Task 參照，避免工作尚未完成就被回收。"""
    task = asyncio.create_task(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


@app.on_event("startup")
async def _preload_ocr_on_startup() -> None:
    """背景預載 OCR；不阻塞 Web Service 啟動。"""
    if os.getenv("OCR_PRELOAD", "1").strip() == "1":
        _schedule_background(asyncio.to_thread(preload_ocr))


def _user_lock(user_id: str) -> asyncio.Lock:
    """同一 UID 的圖片與莊閒按鈕依序處理，不與其他 UID 互相阻塞。"""
    uid = str(user_id or "").strip()
    lock = _USER_LOCKS.get(uid)
    if lock is None:
        lock = asyncio.Lock()
        _USER_LOCKS[uid] = lock
    return lock


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
    """依正式訊號加入保守型資金配置；不改寫模型機率與方向。"""
    result = dict(prediction or {})
    bankroll = max(0, int(session.get("bankroll", 0) or 0))
    action_code = str(result.get("action") or "O").upper()
    edge = float(result.get("direction_edge", 0.0) or 0.0)
    quality = float(result.get("quality_score", result.get("confidence_score", 0.0)) or 0.0)
    confidence_label = str(result.get("confidence_label") or "偏低")
    signal_reason = str(result.get("signal_reason") or "模型正在等待更明確的方向差距")

    if bankroll <= 0:
        percentage = 0.0
        level = "尚未設定"
        reason = "請先設定本次分析本金"
    elif action_code not in {"B", "P"}:
        percentage = 0.0
        level = "暫緩配置"
        reason = signal_reason
    elif confidence_label == "較高" or (quality >= 0.72 and edge >= 0.04):
        percentage = 3.0
        level = "積極區間"
        reason = "方向差距、模型一致度與訊號品質均達較高區間"
    elif confidence_label == "中等" or quality >= 0.52:
        percentage = 2.0
        level = "標準區間"
        reason = "方向訊號已開放，採標準風險比例"
    else:
        percentage = 1.0
        level = "保守區間"
        reason = "方向訊號已開放，但品質仍屬保守區間"

    raw_amount = bankroll * percentage / 100.0
    suggested = int(raw_amount // 10 * 10) if percentage > 0 else 0
    if percentage > 0 and suggested <= 0:
        suggested = min(bankroll, 10)

    result.update(
        {
            "bankroll": bankroll,
            "suggested_bet_amount": suggested,
            "bet_percentage": percentage,
            "bet_level_text": level,
            "bet_reason": reason,
            "screen_edge": round(edge, 6),
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


def _download_line_image(message_id: str) -> Path:
    """從 LINE Content API 下載圖片到短期暫存檔。"""
    if not CHANNEL_ACCESS_TOKEN:
        raise RuntimeError("尚未設定 LINE_CHANNEL_ACCESS_TOKEN。")
    message_id = str(message_id or "").strip()
    if not message_id:
        raise ValueError("LINE 圖片 messageId 不存在。")

    response = _LINE_HTTP.get(
        f"https://api-data.line.me/v2/bot/message/{message_id}/content",
        headers={"Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}"},
        stream=True,
        timeout=(5, 25),
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
            "altText": "請選擇遊戲館",
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
                "text": "BGS 智慧牌局分析",
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
                    "開始後將建立此 UID 的獨立牌局工作階段，首次上傳畫面完成初始化，後續每局只需回報莊或閒。"
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
            "altText": "BGS 智慧牌局分析",
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
                                "請上傳目前最新完整遊戲畫面，建議包含館別／桌號、剩餘張數與完整大路區域。"
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
                    "backgroundColor": "#FFF4B8",
                    "paddingAll": "18px",
                    "contents": body_contents,
                },
            },
        }
    )




def screen_result_panel(user_id: str, session: Mapping[str, Any]) -> Dict[str, Any]:
    """首次圖片或後續 B/P/T 結果完成後的 UID 獨立專業分析面板。"""
    prediction = dict(session.get("screen_last_prediction") or {})
    ocr = dict(session.get("screen_last_ocr") or {})
    detection = dict(session.get("screen_last_detection") or {})
    road_support = dict(prediction.get("road_support") or {})
    fusion = dict(prediction.get("road_integration") or prediction.get("road_fusion") or {})
    calibration = dict(prediction.get("calibration") or {})
    adaptive = dict(prediction.get("adaptive_ensemble") or {})
    sequence = [str(item).upper() for item in list(session.get("road_sequence") or []) if str(item).upper() in {"B", "P"}]
    raw_outcomes = [str(item).upper() for item in list(session.get("raw_outcomes") or sequence) if str(item).upper() in {"B", "P", "T"}]
    tie_total = int(session.get("tie_total", 0) or sum(1 for item in raw_outcomes if item == "T"))

    venue_code = str(ocr.get("venue_code") or session.get("venue") or "")
    venue_name = str(ocr.get("venue_name") or _venue_name(venue_code))
    room = str(ocr.get("room") or session.get("room") or session.get("last_confirmed_room") or "1")
    remaining = int(prediction.get("screen_remaining_cards") or session.get("screen_remaining_cards") or ocr.get("remaining_cards") or 416)
    direction = str(prediction.get("recommend_text") or "尚未建立")
    action_code = str(prediction.get("action") or "O").upper()
    action = str(prediction.get("action_text") or "觀望")
    signal_status = str(prediction.get("signal_status_text") or ("方向訊號已開放" if action_code in {"B", "P", "T"} else "等待更明確訊號"))
    signal_reason = str(prediction.get("signal_reason") or "模型正在等待更明確的方向差距")
    quality_score = float(prediction.get("quality_score", 0.0) or 0.0)
    confidence_label = str(prediction.get("confidence_label") or "偏低")
    edge_percent = float(prediction.get("direction_edge_percent") or float(prediction.get("direction_edge", 0.0) or 0.0) * 100.0)
    consistency = float(prediction.get("model_consistency", 0.0) or 0.0) * 100.0
    analysis_number = int(session.get("screen_analysis_count", 0) or 0)
    bankroll = int(prediction.get("bankroll", session.get("bankroll", 0)) or 0)
    suggested = int(prediction.get("suggested_bet_amount", 0) or 0)
    percentage = float(prediction.get("bet_percentage", 0.0) or 0.0)
    bet_level = str(prediction.get("bet_level_text") or "暫緩配置")
    bet_reason = str(prediction.get("bet_reason") or signal_reason)
    manual_rounds = int(session.get("screen_manual_rounds", 0) or 0)
    recognized = int(detection.get("recognized_count", 0) or 0)
    road_direction = str(road_support.get("direction_text") or "資料建立中")
    road_samples = int(road_support.get("sample_count", len(sequence)) or len(sequence))
    fusion_text = "已納入統一核心" if bool(fusion.get("applied")) else "已檢查／核心為主"
    input_type = str(session.get("screen_input_type") or prediction.get("screen_input_type") or detection.get("input_type") or "unknown")
    input_type_text = "牌路裁切圖" if input_type == "road_crop" else "完整遊戲畫面" if input_type == "full_screen" else "遊戲畫面"
    room_source = str(session.get("screen_room_source") or prediction.get("room_source") or ocr.get("room_source") or "")
    room_source_text = "畫面辨識" if room_source == "image_ocr" else "沿用目前分析桌"
    data_quality = f"{input_type_text}已完成辨識" if recognized > 0 else "沿用本桌牌路資料"
    bet_text = f"{_format_money(suggested)} 元（{percentage:.1f}%｜{bet_level}）" if suggested > 0 else "0 元（暫緩配置）"
    direction_color = {"莊": "#D52B2B", "閒": "#2667D8", "和": "#159447"}.get(direction, "#7B5600")
    calibration_text = (
        f"已啟用｜{calibration.get('scope')}｜{int(calibration.get('sample_count', 0) or 0)} 筆"
        if calibration.get("active") else f"累積中｜{int(calibration.get('sample_count', 0) or 0)} 筆"
    )
    adaptive_text = (
        f"已啟用｜{float(adaptive.get('effective_share', 0.0) or 0.0) * 100.0:.1f}%"
        if adaptive.get("active") else "樣本累積中"
    )

    return _clean_flex({
        "type": "flex", "altText": f"BGS 下一局方向：{direction}", "quickReply": _road_quick_reply(),
        "contents": {
            "type": "bubble", "size": "mega",
            "body": {
                "type": "box", "layout": "vertical", "backgroundColor": "#FFF4B8", "paddingAll": "18px",
                "contents": [
                    {"type": "text", "text": f"BGS 下一局分析 #{analysis_number}", "weight": "bold", "size": "xl", "color": "#7B5600"},
                    {"type": "text", "text": (
                        f"館別：{venue_name or '-'}｜桌號：{room}（{room_source_text}）\n"
                        f"圖片確認：{int(session.get('initial_recognized_count', 0) or 0)} 局｜後續輸入：{len(session.get('manual_outcome_history') or [])} 局\n"
                        f"模型完整歷史：{len(raw_outcomes)} 局｜大路：{len(sequence)} 局｜和局：{tie_total} 局\n"
                        f"估計剩餘：{remaining} 張｜不確定格：{int(session.get('initial_uncertain_count', 0) or 0)}"
                    ), "wrap": True, "size": "sm", "margin": "sm", "color": "#665000"},
                    {"type": "separator", "margin": "md", "color": "#E1BD43"},
                    {"type": "text", "text": f"下一局方向評估：{direction}", "weight": "bold", "size": "xl", "margin": "md", "color": direction_color},
                    {"type": "box", "layout": "vertical", "spacing": "sm", "margin": "md", "contents": [
                        {"type": "text", "text": f"莊　{float(prediction.get('banker_rate', 0.0)):.2f}%", "color": "#D52B2B", "weight": "bold"},
                        {"type": "text", "text": f"閒　{float(prediction.get('player_rate', 0.0)):.2f}%", "color": "#2667D8", "weight": "bold"},
                        {"type": "text", "text": f"和　{float(prediction.get('tie_rate', 0.0)):.2f}%", "color": "#259B55", "weight": "bold"},
                    ]},
                    {"type": "text", "text": (
                        f"正式訊號：{action}\n訊號狀態：{signal_status}\n方向優勢：{edge_percent:.2f}%\n"
                        f"模型信心：{quality_score * 100.0:.1f}%（{confidence_label}）\n模型一致度：{consistency:.1f}%\n"
                        f"牌路先行：{road_direction}｜整合：{fusion_text}"
                    ), "wrap": True, "margin": "md", "color": "#3E3100"},
                    {"type": "text", "text": f"校準狀態：{calibration_text}\n自適應集成：{adaptive_text}", "wrap": True, "size": "sm", "margin": "md", "color": "#665000"},
                    {"type": "text", "text": f"訊號說明：{signal_reason}", "wrap": True, "size": "sm", "margin": "md", "color": "#665000"},
                    {"type": "separator", "margin": "md", "color": "#E1BD43"},
                    {"type": "text", "text": f"分析本金：{_format_money(bankroll)} 元\n建議配置：{bet_text}\n配置依據：{bet_reason}", "wrap": True, "margin": "md", "color": "#3E3100"},
                    {"type": "text", "text": f"資料狀態：{data_quality}\n牌路統計樣本：{road_samples} 局", "wrap": True, "size": "sm", "margin": "md", "color": "#806A2A"},
                    {"type": "box", "layout": "vertical", "spacing": "sm", "margin": "lg", "contents": [
                        _postback_button("🔴 本局結果：莊", "road_append", color="#D52B2B", result="B"),
                        _postback_button("🔵 本局結果：閒", "road_append", color="#2667D8", result="P"),
                        _postback_button("🟢 本局結果：和", "road_append", color="#159447", result="T"),
                        _postback_button("結束本次分析", "end", style="secondary"),
                    ]},
                    {"type": "text", "text": "首次圖片後，每局只需回報莊／閒／和。系統會用實際結果結算上一筆預測並更新校準資料。", "wrap": True, "size": "xs", "margin": "md", "color": "#806A2A"},
                ],
            },
        },
    })

def ended_panel() -> Dict[str, Any]:
    return _clean_flex(
        {
            "type": "flex",
            "altText": "分析已結束",
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
                            "text": "本次分析已結束",
                            "weight": "bold",
                            "size": "xl",
                            "color": "#7B5600",
                        },
                        {
                            "type": "text",
                            "text": "此 UID 的本次牌路、OCR 與預測已獨立清除；開通權限、館別與本金仍保留。",
                            "wrap": True,
                            "margin": "md",
                            "color": "#4C3900",
                        },
                        _postback_button("同館再次開始", "restart_screen"),
                        _postback_button("重新選館", "venues", style="secondary"),
                    ],
                },
            },
        }
    )


def _activate_code(user_id: str, code: str) -> str:
    code = _normalize_access_code(code)
    now = int(datetime.now().timestamp())
    if code in PERMANENT_CODES:
        store.upsert_session(
            user_id,
            {"permanent_access": True, "access_until": 0, "status": "分析中"},
        )
        return "永久版"
    if code in MONTHLY_CODES:
        store.upsert_session(
            user_id,
            {
                "permanent_access": False,
                "trial_started_at": now,
                "access_until": now + 30 * 24 * 60 * 60,
                "status": "分析中",
            },
        )
        return "30日版"
    if code in TEMP_CODES:
        store.upsert_session(
            user_id,
            {
                "permanent_access": False,
                "trial_started_at": now,
                "access_until": now + 30 * 60,
                "status": "分析中",
            },
        )
        return "30分鐘版"
    raise ValueError("開通碼錯誤")


def _ensure_access(user_id: str) -> Dict[str, Any]:
    status = store.access_status(user_id, start_trial=True)
    if not status.get("allowed"):
        raise AccessExpiredError("試用已到期")
    return status


def _run_prediction(user_id: str) -> Dict[str, Any]:
    _ensure_access(user_id)
    acquired = _PREDICTION_SLOTS.acquire(timeout=PREDICTION_QUEUE_TIMEOUT)
    if not acquired:
        raise RuntimeError("目前分析人數較多，請稍後再點一次。")
    try:
        return store.run_virtual_round(user_id, run_virtual_round)
    finally:
        _PREDICTION_SLOTS.release()



def _refresh_screen_prediction(
    user_id: str,
    outcome: str,
    expected_run_id: str,
) -> Dict[str, Any]:
    """加入本局 B/P/T，結算上一筆預測；不重跑 OCR，只更新模型。"""
    value = str(outcome or "").upper().strip()
    if value not in {"B", "P", "T"}:
        raise ValueError("本局結果只能是莊、閒或和。")

    session = store.get_session(user_id)
    if str(session.get("analysis_run_id") or "") != str(expected_run_id):
        raise RuntimeError("本次分析已結束或已重新開始，舊操作已忽略。")
    if not bool(session.get("analysis_active")):
        raise RuntimeError("本次分析已結束，請重新開始分析。")
    if not session.get("screen_last_prediction"):
        raise ValueError("請先上傳一次遊戲畫面，建立初始牌路。")

    expected_version = int(session.get("screen_data_version", 0) or 0)
    ocr = dict(session.get("screen_last_ocr") or {})
    detection = dict(session.get("screen_last_detection") or {})
    venue = str(ocr.get("venue_code") or session.get("venue") or "")
    room = str(ocr.get("room") or session.get("room") or session.get("last_confirmed_room") or "1")

    # 先用本局真實結果結算上一筆預測；校準器下一局只會看到過去資料。
    resolve_latest_prediction(user_id, value, venue=venue, room=room)

    initial_history = [
        str(item).upper() for item in list(session.get("initial_image_history") or [])
        if str(item).upper() in {"B", "P", "T"}
    ]
    manual_history = [
        str(item).upper() for item in list(session.get("manual_outcome_history") or [])
        if str(item).upper() in {"B", "P", "T"}
    ]
    manual_history.append(value)
    raw_history = initial_history + manual_history
    road_state = _derive_road_state(raw_history)
    sequence = list(road_state["road_sequence"])
    tie_markers = dict(road_state["tie_markers"])

    current_remaining = int(session.get("screen_remaining_cards") or ocr.get("remaining_cards") or 416)
    remaining = max(6, current_remaining - SCREEN_ESTIMATED_CARDS_PER_ROUND)
    screen_metadata = {
        "input_type": str(session.get("screen_input_type") or detection.get("input_type") or "full_screen"),
        "venue_source": str(session.get("screen_venue_source") or ocr.get("venue_source") or "session_selected"),
        "room_source": str(session.get("screen_room_source") or ocr.get("room_source") or "session_previous"),
        "room_confidence": float(session.get("screen_room_confidence", 0.0) or 0.0),
        "manual_update": True,
    }

    acquired = _PREDICTION_SLOTS.acquire(timeout=PREDICTION_QUEUE_TIMEOUT)
    if not acquired:
        raise RuntimeError("目前分析人數較多，請稍後再按一次。")
    try:
        prediction = predict_from_screenshot(
            sequence,
            raw_outcomes=raw_history,
            tie_markers=tie_markers,
            remaining_cards=remaining,
            prior_counts=None,
            venue=venue,
            room=room,
            user_id=user_id,
            screen_metadata=screen_metadata,
            initial_grid_cells=list(session.get("initial_grid_cells") or []),
            initial_image_history=initial_history,
            manual_outcome_history=manual_history,
        )
    finally:
        _PREDICTION_SLOTS.release()

    prediction["latest_actual_outcome"] = value
    prediction["latest_actual_outcome_text"] = {"B": "莊", "P": "閒", "T": "和"}[value]
    prediction["remaining_cards_estimated_after_manual"] = True
    prediction = _attach_bankroll_advice(prediction, session)
    resolved = {
        "venue_code": venue, "venue_name": str(ocr.get("venue_name") or ""),
        "room": room, "remaining_cards": remaining, **screen_metadata,
    }
    return store.update_screen_analysis(
        user_id,
        ocr=ocr,
        detection=detection,
        sequence=sequence,
        raw_outcomes=raw_history,
        tie_markers=tie_markers,
        initial_image_history=initial_history,
        manual_outcome_history=manual_history,
        initial_grid_cells=list(session.get("initial_grid_cells") or []),
        recognition_quality={
            "recognized_count": int(session.get("initial_recognized_count", len(initial_history)) or len(initial_history)),
            "uncertain_count": int(session.get("initial_uncertain_count", 0) or 0),
        },
        prediction=prediction,
        resolved=resolved,
        processing_ms=0.0,
        source=f"manual_result_{value}",
        expected_run_id=expected_run_id,
        expected_data_version=expected_version,
    )

def _start_screen_flow(
    user_id: str,
    *,
    new_session: bool = True,
) -> Dict[str, Any]:
    """開始分析時，只建立／清除目前 UID 的獨立分析 Session。"""
    _ensure_access(user_id)
    session = store.get_session(user_id)
    if not session.get("venue"):
        return {"panel": venue_panel(user_id), "state": "venue"}

    session = (
        _start_new_uid_analysis(user_id)
        if new_session
        else store.begin_screen_analysis(user_id, clear_existing=False)
    )
    if int(session.get("bankroll", 0) or 0) <= 0:
        session = _request_bankroll(user_id)
        return {"panel": bankroll_panel(user_id, session), "state": "bankroll"}

    return {"panel": upload_request_panel(user_id, session), "state": "image"}



async def _process_screen_image(
    user_id: str,
    message_id: str,
    expected_run_id: str,
) -> None:
    """背景完成雙模式辨識、牌路先行分析與統一主模型；只寫回同一 UID、同一代次。"""
    temporary_image: Optional[Path] = None
    started = time.perf_counter()
    async with _user_lock(user_id):
        try:
            current_session = store.get_session(user_id)
            if str(current_session.get("analysis_run_id") or "") != str(expected_run_id):
                return
            if not bool(current_session.get("analysis_active")):
                return

            download_started = time.perf_counter()
            temporary_image = await asyncio.to_thread(_download_line_image, message_id)
            download_ms = (time.perf_counter() - download_started) * 1000.0

            screen = await asyncio.to_thread(analyze_game_screen, temporary_image, current_session)
            sequence = list(screen.get("sequence") or [])
            raw_outcomes = list(screen.get("raw_outcomes") or sequence)
            tie_markers = dict(screen.get("tie_markers") or {})
            grid_cells = [dict(item) for item in list(screen.get("grid_cells") or []) if isinstance(item, Mapping)]
            recognized_count = int(screen.get("recognized_count", len(sequence)) or len(sequence))
            uncertain_count = int(screen.get("uncertain_count", 0) or 0)
            recognition_quality_ok = bool(screen.get("recognition_quality_ok", True))
            if not sequence:
                road_errors = list((screen.get("road") or {}).get("errors") or [])
                detail = road_errors[-1] if road_errors else "未偵測到大路圓圈"
                await asyncio.to_thread(
                    _push, user_id,
                    [_road_error_message(f"{detail}。請傳送包含完整大路的畫面或牌路裁切圖。")],
                )
                return
            if not recognition_quality_ok:
                await asyncio.to_thread(
                    _push, user_id,
                    [_road_error_message(
                        f"本次只確認 {recognized_count} 格，另有 {uncertain_count} 格無法可靠判定；為避免總局數錯誤，已停止送入模型。請裁切只保留完整大路後重新上傳。"
                    )],
                )
                return

            resolved = dict(screen.get("resolved") or {})
            remaining = int(resolved.get("remaining_cards") or 416)
            resolved["remaining_cards"] = remaining
            expected_version = int(current_session.get("screen_data_version", 0) or 0)
            screen_metadata = {
                "input_type": str(screen.get("input_type") or resolved.get("input_type") or "full_screen"),
                "venue_source": str(resolved.get("venue_source") or "session_selected"),
                "room_source": str(resolved.get("room_source") or "session_selected"),
                "room_confidence": float(resolved.get("room_confidence", 0.0) or 0.0),
                "ocr_timed_out": bool(resolved.get("ocr_timed_out")),
                "vision_timings": dict(screen.get("timings") or {}),
            }
            resolved.update(screen_metadata)

            acquired = await asyncio.to_thread(
                _PREDICTION_SLOTS.acquire, True, PREDICTION_QUEUE_TIMEOUT
            )
            if not acquired:
                raise RuntimeError("目前分析人數較多，請稍後重新上傳圖片。")
            model_started = time.perf_counter()
            try:
                prediction = await asyncio.to_thread(
                    predict_from_screenshot,
                    sequence,
                    raw_outcomes=raw_outcomes,
                    tie_markers=tie_markers,
                    remaining_cards=remaining,
                    prior_counts=None,
                    venue=str(resolved.get("venue_code") or current_session.get("venue") or ""),
                    room=str(resolved.get("room") or current_session.get("room") or current_session.get("last_confirmed_room") or "1"),
                    user_id=user_id,
                    screen_metadata=screen_metadata,
                    initial_grid_cells=grid_cells,
                    initial_image_history=raw_outcomes,
                    manual_outcome_history=[],
                )
            finally:
                _PREDICTION_SLOTS.release()
            model_ms = (time.perf_counter() - model_started) * 1000.0

            prediction = _attach_bankroll_advice(prediction, current_session)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            source = f"screen_image_{screen_metadata['input_type']}"
            session = store.update_screen_analysis(
                user_id,
                ocr=dict(screen.get("ocr") or {}),
                detection=dict(screen.get("road") or {}),
                sequence=sequence,
                raw_outcomes=raw_outcomes,
                tie_markers=tie_markers,
                initial_image_history=raw_outcomes,
                manual_outcome_history=[],
                initial_grid_cells=grid_cells,
                recognition_quality={
                    "recognized_count": recognized_count,
                    "uncertain_count": uncertain_count,
                    "quality_ok": recognition_quality_ok,
                },
                prediction=prediction,
                resolved=resolved,
                processing_ms=elapsed_ms,
                source=source,
                expected_run_id=expected_run_id,
                expected_data_version=expected_version,
            )
            print("screen_timing", json.dumps({
                "uid": user_id[-8:],
                "download_ms": round(download_ms, 2),
                **dict(screen.get("timings") or {}),
                "model_ms": round(model_ms, 2),
                "total_ms": round(elapsed_ms, 2),
                "input_type": screen_metadata["input_type"],
                "room_source": screen_metadata["room_source"],
                "road_count": len(sequence),
            }, ensure_ascii=False))
            await asyncio.to_thread(_push, user_id, [screen_result_panel(user_id, session)])
        except AccessExpiredError:
            await asyncio.to_thread(_push, user_id, [_text("試用已到期，請聯繫管理員開通。")])
        except Exception as exc:
            traceback.print_exc()
            message = str(exc)
            if "舊結果已忽略" in message or "已重新開始" in message or "已結束" in message:
                return
            await asyncio.to_thread(_push, user_id, [_road_error_message(f"圖片處理失敗：{exc}")])
        finally:
            if temporary_image is not None:
                temporary_image.unlink(missing_ok=True)


async def _process_manual_outcome(
    user_id: str,
    outcome: str,
    expected_run_id: str,
) -> None:
    """同一 UID 依序處理莊／閒／和按鈕，完成後 Push 新分析面板。"""
    async with _user_lock(user_id):
        try:
            _ensure_access(user_id)
            session = await asyncio.to_thread(
                _refresh_screen_prediction,
                user_id,
                outcome,
                expected_run_id,
            )
            await asyncio.to_thread(_push, user_id, [screen_result_panel(user_id, session)])
        except Exception as exc:
            traceback.print_exc()
            message = str(exc)
            if "舊操作已忽略" in message or "舊結果已忽略" in message:
                return
            await asyncio.to_thread(
                _push,
                user_id,
                [_text(f"本局結果更新失敗：{message}")],
            )


def _public_session(session: Mapping[str, Any]) -> Dict[str, Any]:
    data = copy_session = dict(session)
    copy_session.pop("virtual_shoe", None)
    copy_session["remaining_cards"] = len(session.get("virtual_shoe") or [])
    copy_session["venue_name"] = _venue_name(str(session.get("venue") or ""))
    copy_session["venues"] = [
        {
            **venue,
            "image_url": f"/static/venues/{venue['image']}",
        }
        for venue in VENUES
    ]
    copy_session["analysis_history"] = list(session.get("analysis_history") or [])[-40:]
    copy_session["round_history"] = list(session.get("round_history") or [])[-60:]
    return data


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "version": "10.3",
            "engine": "V10_3_FULL_HISTORY_GRID_QUALITY",
            "activation_code_fix": True,
            "activation_persistence_check": True,
            "storage_path": str(getattr(store, "SESSION_DATA_FILE", "")),
            "activation_code_counts": {
                "permanent": len(PERMANENT_CODES),
                "monthly": len(MONTHLY_CODES),
                "temporary": len(TEMP_CODES),
            },
            "default_permanent_code_loaded": "aaa1888007" in PERMANENT_CODES,
            "input_required": True,
            "virtual_only": False,
            "public_base_url_configured": bool(PUBLIC_BASE_URL),
            "venues": [venue["code"] for venue in VENUES],
            "max_concurrent_predictions": MAX_CONCURRENT_PREDICTIONS,
            "road_image_recognition": True,
            "room_info_ocr": True,
            "parallel_screen_pipeline": True,
            "road_manual_quick_reply": True,
            "uid_isolated_sessions": True,
            "first_image_then_bpt_only": True,
            "tie_result_supported": True,
            "online_calibration": True,
            "adaptive_ensemble": True,
            "stale_background_guard": True,
            "bankroll_flow": True,
            "immediate_image_ack": True,
            "background_push_result": True,
            "line_default_mode": "screen",
        }
    )


@app.get("/ping")
def ping() -> PlainTextResponse:
    return PlainTextResponse("OK")


@app.get("/api/session")
def api_session(user_id: str) -> JSONResponse:
    try:
        return JSONResponse({"ok": True, "session": _public_session(store.get_session(user_id))})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/venue")
def api_venue(payload: VenueRequest) -> JSONResponse:
    venue = payload.venue.upper()
    if venue not in VENUE_BY_CODE:
        raise HTTPException(status_code=400, detail="無效館別")
    session = store.select_venue(payload.user_id, venue, payload.room)
    return JSONResponse({"ok": True, "session": _public_session(session)})


@app.post("/api/room")
def api_room(payload: RoomRequest) -> JSONResponse:
    session = store.set_room(payload.user_id, payload.room)
    return JSONResponse({"ok": True, "session": _public_session(session)})


@app.post("/api/predict")
async def api_predict(payload: UserRequest) -> JSONResponse:
    try:
        result = await asyncio.to_thread(_start_screen_flow, payload.user_id)
        return JSONResponse(
            {
                "ok": True,
                "state": result["state"],
                "session": _public_session(store.get_session(payload.user_id)),
            }
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/api/reset")
def api_reset(payload: UserRequest) -> JSONResponse:
    session = store.clear_screen_analysis(payload.user_id, keep_mode=True)
    return JSONResponse({"ok": True, "session": _public_session(session)})


@app.post("/api/end")
def api_end(payload: UserRequest) -> JSONResponse:
    session = _end_uid_analysis(payload.user_id)
    return JSONResponse({"ok": True, "session": _public_session(session)})


@app.post("/webhook")
async def webhook(request: Request) -> JSONResponse:
    body = await request.body()
    if not _verify_signature(body, request.headers.get("X-Line-Signature")):
        return JSONResponse({"ok": False}, status_code=401)

    try:
        payload = json.loads(body.decode("utf-8") or "{}")
    except Exception:
        return JSONResponse({"ok": False}, status_code=400)

    for event in payload.get("events", []):
        token = str(event.get("replyToken") or "")
        source = event.get("source") or {}
        user_id = str(source.get("userId") or "").strip()
        if not user_id:
            _reply(token, [_text("無法取得 LINE UID，請改在與機器人的一對一聊天室操作。")])
            continue
        try:
            event_type = event.get("type")
            message = event.get("message") or {}
            message_type = str(message.get("type") or "")

            if event_type == "follow":
                _reply(token, [venue_panel(user_id)])
                continue

            # 圖片先立即回覆，再交由背景工作完成並 Push 結果。
            if event_type == "message" and message_type == "image":
                _ensure_access(user_id)
                session = store.get_session(user_id)
                if not session.get("venue"):
                    _reply(token, [_text("請先選擇遊戲館。"), venue_panel(user_id)])
                    continue
                if int(session.get("bankroll", 0) or 0) <= 0:
                    session = _request_bankroll(user_id)
                    _reply(token, [bankroll_panel(user_id, session)])
                    continue

                if not bool(session.get("analysis_active")):
                    _reply(token, [_text("請先點擊「開始分析」，再上傳圖片。"), ready_panel(user_id, session)])
                    continue
                session = store.begin_screen_analysis(user_id, clear_existing=False)
                message_id = str(message.get("id") or "").strip()
                if not message_id:
                    raise ValueError("LINE 圖片 messageId 不存在。")
                run_id = str(session.get("analysis_run_id") or "")
                _reply(token, [image_received_panel(session)])
                _schedule_background(_process_screen_image(user_id, message_id, run_id))
                continue

            if event_type == "message" and message_type == "text":
                raw_text = str(message.get("text") or "")
                text = unicodedata.normalize("NFKC", raw_text).strip()
                access_code = _normalize_access_code(raw_text)

                activation_match = access_code in ALL_CODES
                print(
                    "activation_debug",
                    json.dumps(
                        {
                            "uid": user_id[-8:],
                            "normalized": access_code,
                            "length": len(access_code),
                            "matched": activation_match,
                            "permanent": access_code in PERMANENT_CODES,
                        },
                        ensure_ascii=False,
                    ),
                )
                if activation_match:
                    plan = _activate_code(user_id, access_code)
                    saved = store.get_session(user_id)
                    if plan == "永久版" and not bool(saved.get("permanent_access")):
                        raise RuntimeError("開通資料未成功寫入，請檢查 SESSION_DATA_FILE 儲存路徑。")
                    _reply(token, [_text(f"✅ 已開通：{plan}"), venue_panel(user_id)])
                    continue

                if text in {"開通碼檢查", "檢查開通碼", "版本檢查"}:
                    _reply(
                        token,
                        [
                            _text(
                                "BGS 版本：9.7.2\n"
                                f"永久碼載入數：{len(PERMANENT_CODES)}\n"
                                f"aaa1888007 已載入：{'是' if 'aaa1888007' in PERMANENT_CODES else '否'}"
                            )
                        ],
                    )
                    continue

                session = store.get_session(user_id)

                # 本金輸入與本金指令。
                bankroll_command = bool(
                    re.fullmatch(
                        r"(?:本金|金額|資金)\s*[:：=]?\s*[0-9,，＄$ ]+",
                        text,
                        flags=re.IGNORECASE,
                    )
                )
                if bool(session.get("awaiting_bankroll")) or bankroll_command:
                    try:
                        bankroll = _parse_bankroll(text)
                        session = store.set_bankroll(user_id, bankroll, begin_screen=True)
                        _reply(
                            token,
                            [
                                _text(f"✅ 本金已設定為 {_format_money(bankroll)} 元"),
                                upload_request_panel(user_id, session),
                            ],
                        )
                    except ValueError as exc:
                        _reply(token, [_text(str(exc)), bankroll_panel(user_id, session)])
                    continue

                # 首次圖片完成後，文字入口同樣接受莊／閒／和。
                if text in {"🔴 本局：莊", "🔴 本局結果：莊", "補輸莊", "補莊", "莊"}:
                    current = store.get_session(user_id)
                    if not current.get("screen_last_prediction"):
                        _reply(token, [_text("請先點擊開始分析並上傳一次遊戲畫面。")])
                        continue
                    run_id = str(current.get("analysis_run_id") or "")
                    _reply(token, [manual_result_received_panel("B")])
                    _schedule_background(_process_manual_outcome(user_id, "B", run_id))
                    continue
                if text in {"🔵 本局：閒", "🔵 本局結果：閒", "補輸閒", "補閒", "閒"}:
                    current = store.get_session(user_id)
                    if not current.get("screen_last_prediction"):
                        _reply(token, [_text("請先點擊開始分析並上傳一次遊戲畫面。")])
                        continue
                    run_id = str(current.get("analysis_run_id") or "")
                    _reply(token, [manual_result_received_panel("P")])
                    _schedule_background(_process_manual_outcome(user_id, "P", run_id))
                    continue
                if text in {"🟢 本局：和", "🟢 本局結果：和", "補輸和", "補和", "和"}:
                    current = store.get_session(user_id)
                    if not current.get("screen_last_prediction"):
                        _reply(token, [_text("請先點擊開始分析並上傳一次遊戲畫面。")])
                        continue
                    run_id = str(current.get("analysis_run_id") or "")
                    _reply(token, [manual_result_received_panel("T")])
                    _schedule_background(_process_manual_outcome(user_id, "T", run_id))
                    continue
                if text in {"🔄 清除重來", "清除路紙", "清除畫面", "重來"}:
                    result = _start_screen_flow(user_id, new_session=True)
                    _reply(token, [_text("🔄 已清除這個 UID 的舊牌路。"), result["panel"]])
                    continue

                if text in {"開始", "選館", "重新選館", "館別"}:
                    _reply(token, [venue_panel(user_id)])
                    continue
                if text in {"開始分析"}:
                    result = _start_screen_flow(user_id, new_session=True)
                    _reply(token, [result["panel"]])
                    continue
                if text in {"上傳圖片", "圖片辨識", "路紙", "路單"}:
                    result = _start_screen_flow(user_id, new_session=False)
                    _reply(token, [result["panel"]])
                    continue
                if text in {"繼續分析"}:
                    _reply(token, [_text("首次圖片完成後，請直接按「本局結果：莊」或「本局結果：閒」。")])
                    continue
                if text in {"更改本金", "設定本金", "本金", "金額", "資金"}:
                    session = _request_bankroll(user_id)
                    _reply(token, [bankroll_panel(user_id, session)])
                    continue
                if text in {"結束", "結束分析"}:
                    _end_uid_analysis(user_id)
                    _reply(token, [ended_panel()])
                    continue

                if session.get("awaiting_screenshot"):
                    _reply(token, [_text("目前正在等待圖片，請直接上傳最新完整遊戲畫面。")])
                elif session.get("venue"):
                    _reply(token, [ready_panel(user_id, session)])
                else:
                    _reply(token, [venue_panel(user_id)])
                continue

            if event_type == "postback":
                query = {
                    key: values[0]
                    for key, values in urllib.parse.parse_qs(
                        str((event.get("postback") or {}).get("data") or "")
                    ).items()
                }
                action_name = str(query.get("action") or "")

                if action_name == "road_append":
                    value = str(query.get("result") or "").upper()
                    if value not in {"B", "P", "T"}:
                        raise ValueError("手動牌局結果不正確。")
                    current = store.get_session(user_id)
                    if not bool(current.get("analysis_active")):
                        _reply(token, [_text("本次分析已結束，請重新開始分析。")])
                        continue
                    if not current.get("screen_last_prediction"):
                        _reply(token, [_text("請先上傳一次遊戲畫面，建立初始牌路。")])
                        continue
                    run_id = str(current.get("analysis_run_id") or "")
                    _reply(token, [manual_result_received_panel(value)])
                    _schedule_background(_process_manual_outcome(user_id, value, run_id))
                elif action_name == "road_clear":
                    result = _start_screen_flow(user_id, new_session=True)
                    _reply(token, [_text("🔄 已清除這個 UID 的舊牌路。"), result["panel"]])
                elif action_name == "venue":
                    venue = str(query.get("venue") or "").upper()
                    if venue not in VENUE_BY_CODE:
                        raise ValueError("無效館別")
                    session = store.select_venue(
                        user_id,
                        venue,
                        str(query.get("room") or "1"),
                    )
                    _reply(token, [ready_panel(user_id, session)])
                elif action_name in {"start_screen", "restart_screen"}:
                    result = _start_screen_flow(user_id, new_session=True)
                    _reply(token, [result["panel"]])
                elif action_name in {"request_screen", "predict"}:
                    result = _start_screen_flow(user_id, new_session=False)
                    _reply(token, [result["panel"]])
                elif action_name == "change_bankroll":
                    session = _request_bankroll(user_id)
                    _reply(token, [bankroll_panel(user_id, session)])
                elif action_name in {"screen_clear", "reset"}:
                    result = _start_screen_flow(user_id, new_session=True)
                    _reply(token, [_text("🔄 已清除這個 UID 的舊牌路。"), result["panel"]])
                elif action_name == "end":
                    _end_uid_analysis(user_id)
                    _reply(token, [ended_panel()])
                elif action_name == "venues":
                    _reply(token, [venue_panel(user_id)])
                else:
                    _reply(token, [venue_panel(user_id)])
                continue
        except AccessExpiredError:
            _reply(
                token,
                [
                    _text("試用已到期，請聯繫管理員開通。"),
                    {
                        "type": "template",
                        "altText": "聯繫管理員",
                        "template": {
                            "type": "buttons",
                            "text": "試用已到期",
                            "actions": [
                                {
                                    "type": "uri",
                                    "label": "聯繫管理員",
                                    "uri": ADMIN_LINE_URL,
                                }
                            ],
                        },
                    },
                ],
            )
        except PermissionError as exc:
            traceback.print_exc()
            storage_path = str(getattr(store, "SESSION_DATA_FILE", "未設定"))
            _reply(token, [_text(f"資料儲存權限錯誤：{storage_path}\n請確認 Render Persistent Disk 掛載或移除錯誤的 SESSION_DATA_FILE。")])
        except Exception as exc:
            traceback.print_exc()
            message = str(exc)
            if "has no attribute" in message and "store" in message:
                message = "app.py 與 store.py 版本不一致，請同時覆蓋兩支檔案後重新部署。"
            _reply(token, [_text(f"系統忙碌：{message}")])

    return JSONResponse({"ok": True})
