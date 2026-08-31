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
import copy
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
from money_management import MIN_BET_RATIO, MAX_BET_RATIO
from shoe_composition import validate_remaining_counts
from shoe_constants import (
    AVERAGE_CARDS_PER_HAND,
    SHOE_DECKS,
    TOTAL_SHOE_CARDS,
    estimate_remaining_cards,
)


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
SCREEN_ESTIMATED_CARDS_PER_ROUND = AVERAGE_CARDS_PER_HAND  # compatibility alias


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
    text = text.translate(str.maketrans({
        "а": "a",
        "ɑ": "a",
        "Ａ": "A",
        "ａ": "a",
    }))
    text = re.sub(r"[^0-9A-Za-z]", "", text)
    return text.lower()


def _code_set(env_name: str, defaults: str) -> set[str]:
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


class ExactRemainingCountsRequest(UserRequest):
    """使用者明確輸入的點數 0..9 剩餘張數；不是 OCR 猜測值。"""
    remaining_counts: List[Any] = Field(default_factory=list)
    decks: int = Field(default=SHOE_DECKS, ge=1, le=16)


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
    target = str(to or "").strip()
    if not target:
        return False
    if not CHANNEL_ACCESS_TOKEN:
        print("LINE push preview", target, json.dumps(messages, ensure_ascii=False))
        return False
    retries = PUSH_MAX_RETRIES if max_retries is None else max(0, int(max_retries))
    delay = PUSH_RETRY_DELAY_SECONDS if retry_delay_seconds is None else max(0.0, float(retry_delay_seconds))
    payload = {"to": target, "messages": messages[:5]}
    headers = {"Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}", "Content-Type": "application/json"}
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
                    print("LINE push recovered", json.dumps({"uid": target[-8:], "attempt": attempt, "status": last_status}, ensure_ascii=False))
                return True
            print("LINE push failed", json.dumps({"uid": target[-8:], "attempt": attempt, "status": last_status, "body": last_body}, ensure_ascii=False))
        except Exception as exc:
            last_status = -1
            last_body = str(exc)[:500]
            print("LINE push exception", json.dumps({"uid": target[-8:], "attempt": attempt, "error": last_body}, ensure_ascii=False))
        if attempt < retries and delay > 0:
            time.sleep(delay)
    print("LINE push exhausted", json.dumps({"uid": target[-8:], "status": last_status, "body": last_body}, ensure_ascii=False))
    return False


def _schedule_background(coro: Any) -> None:
    task = asyncio.create_task(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


@app.on_event("startup")
async def _preload_ocr_on_startup() -> None:
    if os.getenv("OCR_PRELOAD", "0").strip() == "1":
        _schedule_background(asyncio.to_thread(preload_ocr))


def _user_lock(user_id: str) -> asyncio.Lock:
    uid = str(user_id or "").strip()
    lock = _USER_LOCKS.get(uid)
    if lock is None:
        lock = asyncio.Lock()
        _USER_LOCKS[uid] = lock
    return lock


def _user_image_lock(user_id: str) -> threading.Lock:
    uid = str(user_id or "").strip()
    with _USER_IMAGE_LOCKS_GUARD:
        lock = _USER_IMAGE_LOCKS.get(uid)
        if lock is None:
            lock = threading.Lock()
            _USER_IMAGE_LOCKS[uid] = lock
        return lock


def _remaining_image_time(deadline: float) -> float:
    return max(0.0, float(deadline) - time.perf_counter())


def _raise_if_image_timed_out(cancel_event: threading.Event, deadline: float) -> None:
    if cancel_event.is_set() or _remaining_image_time(deadline) <= 0.0:
        raise TimeoutError("LINE 圖片分析已超過 replyToken 時限。")


def _remaining_manual_time(deadline: float) -> float:
    return max(0.0, float(deadline) - time.perf_counter())


def _raise_if_manual_timed_out(cancel_event: threading.Event, deadline: float) -> None:
    if cancel_event.is_set() or _remaining_manual_time(deadline) <= 0.0:
        raise TimeoutError("LINE 手動結果分析已超過 replyToken 時限。")


def _request_bankroll(user_id: str) -> Dict[str, Any]:
    handler = getattr(store, "request_bankroll", None)
    if callable(handler):
        return handler(user_id)
    return store.upsert_session(user_id, {"analysis_mode": "screen", "analysis_active": True, "awaiting_bankroll": True, "awaiting_screenshot": False, "status": "等待輸入本金"})


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


def _attach_bankroll_advice(prediction: Mapping[str, Any], session: Mapping[str, Any]) -> Dict[str, Any]:
    """保留正式 Single-Brain Kelly；此 UI 層不得再建立 EV/observe gate。"""
    result = copy.deepcopy(dict(prediction or {}))
    bankroll = max(0, int(session.get("bankroll", 0) or 0))
    direction = str(result.get("action") or result.get("recommend") or result.get("direction") or "").upper().strip()
    if direction not in {"B", "P"}:
        banker = float(result.get("banker_rate", 50.0) or 50.0)
        player = float(result.get("player_rate", 50.0) or 50.0)
        direction = "B" if banker >= player else "P"
    fraction = float(result.get("final_bet_ratio", result.get("kelly_fraction", MIN_BET_RATIO)) or MIN_BET_RATIO)
    fraction = min(float(MAX_BET_RATIO), max(float(MIN_BET_RATIO), fraction))
    amount = bankroll * fraction
    result.update({
        "action": direction,
        "recommend": direction,
        "internal_action": direction,
        "internal_recommend": direction,
        "next_round_direction": direction,
        "bet_allowed": True,
        "mandatory_bet": True,
        "bankroll": bankroll,
        "kelly_fraction": fraction,
        "final_bet_ratio": fraction,
        "suggested_bet_amount": amount,
        "bet_amount": amount,
        "bet_percentage": fraction * 100.0,
        "kelly_percentage_applied": fraction * 100.0,
        "bet_level_text": "Single-Brain Kelly 5%～30%",
        "bet_reason": str(result.get("signal_reason") or "Contextual LinUCB 正式方向與 Kelly 配置"),
        "screen_edge": round(float(result.get("direction_edge", 0.0) or 0.0), 6),
    })
    return result


def _exact_shoe_context(session: Mapping[str, Any]) -> Dict[str, Any]:
    context: Dict[str, Any] = {"bankroll": max(0.0, float(session.get("bankroll", 0.0) or 0.0))}
    counts = session.get("exact_remaining_counts")
    if isinstance(counts, list) and len(counts) == 10:
        context.update({"remaining_counts": copy.deepcopy(list(counts)), "decks": int(session.get("exact_remaining_decks", SHOE_DECKS) or SHOE_DECKS), "source": "user_exact_remaining_counts"})
    return context


def _road_quick_reply() -> Dict[str, Any]:
    return {"items": [
        {"type": "action", "action": {"type": "postback", "label": "🔴 本局：莊", "data": "action=road_append&result=B", "displayText": "🔴 本局：莊"}},
        {"type": "action", "action": {"type": "postback", "label": "🔵 本局：閒", "data": "action=road_append&result=P", "displayText": "🔵 本局：閒"}},
        {"type": "action", "action": {"type": "postback", "label": "🟢 本局：和", "data": "action=road_append&result=T", "displayText": "🟢 本局：和"}},
    ]}


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
    return {"raw_outcomes": raw, "road_sequence": road[-500:], "tie_markers": markers, "tie_total": sum(1 for value in raw if value == "T"), "pending_opening_ties": pending_opening}


def _road_text_message(session: Mapping[str, Any], *, notice: str = "") -> Dict[str, Any]:
    sequence = [str(item).upper() for item in list(session.get("road_sequence") or []) if str(item).upper() in {"B", "P"}]
    raw_outcomes = [str(item).upper() for item in list(session.get("raw_outcomes") or sequence) if str(item).upper() in {"B", "P", "T"}]
    analysis = dict(session.get("road_last_analysis") or {})
    compact = "".join({"B": "莊", "P": "閒", "T": "和"}[item] for item in raw_outcomes[-24:])
    if len(sequence) > 24:
        compact = "…" + compact
    if not compact:
        compact = "尚未建立"
    lines: List[str] = []
    if notice:
        lines.append(str(notice))
    lines.extend([f"完整牌局：{len(raw_outcomes)} 局｜大路樣本：{len(sequence)} 局｜和局：{sum(1 for item in raw_outcomes if item == 'T')} 局", f"近期序列：{compact}"])
    if analysis:
        lines.extend([f"方向評估：{analysis.get('direction_text') or '-'}", f"訊號狀態：{analysis.get('signal_status_text') or analysis.get('action_text') or '暫緩'}", f"模型信心：{analysis.get('confidence_label') or '偏低'}"])
    lines.append("請使用下方按鈕回報本局莊／閒／和，系統將結算上一筆預測並更新下一局分析。")
    return {"type": "text", "text": "\n".join(lines)[:5000], "quickReply": _road_quick_reply()}


def _road_error_message(message: str) -> Dict[str, Any]:
    return {"type": "text", "text": (f"⚠️ 畫面資料尚未完整建立\n{message}\n\n" "請重新傳送包含完整大路區域的清晰截圖。")[:5000]}


def _download_line_image(message_id: str, *, timeout_seconds: Optional[float] = None) -> Path:
    if not CHANNEL_ACCESS_TOKEN:
        raise RuntimeError("尚未設定 LINE_CHANNEL_ACCESS_TOKEN。")
    message_id = str(message_id or "").strip()
    if not message_id:
        raise ValueError("LINE 圖片 messageId 不存在。")
    if timeout_seconds is None:
        request_timeout = (5.0, 25.0)
    else:
        available = max(0.25, float(timeout_seconds))
        request_timeout = (max(0.20, min(1.25, available)), max(0.20, min(2.50, available)))
    response = _LINE_HTTP.get(f"https://api-data.line.me/v2/bot/message/{message_id}/content", headers={"Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}"}, stream=True, timeout=request_timeout)
    if response.status_code >= 300:
        raise RuntimeError(f"LINE 圖片下載失敗（HTTP {response.status_code}）。")
    content_type = str(response.headers.get("Content-Type") or "").lower()
    suffix = ".png" if "png" in content_type else ".webp" if "webp" in content_type else ".jpg"
    expected_length = int(response.headers.get("Content-Length") or 0)
    if expected_length > LINE_IMAGE_MAX_BYTES:
        raise ValueError("圖片檔案過大，請裁切路紙區域後再傳送。")
    temporary = tempfile.NamedTemporaryFile(prefix="line_road_", suffix=suffix, delete=False)
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
    path = Path(source_path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return path
    height, width = image.shape[:2]
    long_side = max(height, width)
    short_side = min(height, width)
    target_long_side = 1800
    target_short_side = 900
    scale = max(1.0, target_long_side / max(1.0, float(long_side)), target_short_side / max(1.0, float(short_side)))
    scale = min(scale, 2.5)
    if scale <= 1.05:
        return path
    resized = cv2.resize(image, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))), interpolation=cv2.INTER_CUBIC)
    prepared = path.with_name(f"{path.stem}_analysis.png")
    ok, encoded = cv2.imencode(".png", resized)
    if not ok:
        return path
    encoded.tofile(str(prepared))
    return prepared


def _postback_button(label: str, action_name: str, *, style: str = "primary", color: str = "#F1C232", **kwargs: str) -> Dict[str, Any]:
    data = {"action": action_name, **{key: str(value) for key, value in kwargs.items()}}
    return {"type": "button", "style": style, "height": "sm", "color": color if style == "primary" else None, "action": {"type": "postback", "label": label[:20], "data": urllib.parse.urlencode(data), "displayText": label[:300]}}


def _clean_flex(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clean_flex(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_clean_flex(item) for item in value]
    return value


def guide_panel() -> Dict[str, Any]:
    return _clean_flex({"type": "flex", "altText": "BGS AI預測系統使用指南", "contents": {"type": "bubble", "size": "mega", "body": {"type": "box", "layout": "vertical", "backgroundColor": "#FFFFFF", "paddingAll": "18px", "contents": [
        {"type": "text", "text": "BGS AI預測系統", "weight": "bold", "size": "xl", "color": "#3E3100"},
        {"type": "text", "text": "BGS AI 預測系統使用指南", "weight": "bold", "size": "xl", "margin": "sm", "color": "#3E3100"},
        {"type": "text", "text": "📍 操作 3 步驟\n同桌號：請確保程式選擇與平台一致的桌號資訊。\n數據校正：先讓系統預測 3～5 顆，再開始下注。\n跟隨訊號：依程式顯示的下一局方向評估操作。", "wrap": True, "margin": "md", "color": "#3E3100"},
        {"type": "separator", "margin": "md", "color": "#F1B900"},
        {"type": "text", "text": "⚠️ 贏家 4 守則\n專屬平台：僅限配合平台使用，非合作平台數據會產生誤差。\n資金規劃：請將本金分成 20～30 等份，穩定下注。\n紀律停利：每次獲利達標即離場，科學下注、不戀戰。\n裝置綁定：程式已與您的 LINE 帳號綁定，無法轉借他人或跨裝置使用。", "wrap": True, "margin": "md", "color": "#3E3100"},
        {"type": "box", "layout": "vertical", "margin": "lg", "contents": [_postback_button("開始分析", "start_guide", color="#FFD400")]},
    ]}}})


def selected_venue_panel(session: Mapping[str, Any]) -> Dict[str, Any]:
    venue_name = _venue_name(str(session.get("venue") or ""))
    return _clean_flex({"type": "flex", "altText": f"已選擇：{venue_name}", "contents": {"type": "bubble", "size": "mega", "body": {"type": "box", "layout": "vertical", "backgroundColor": "#FFF4B8", "paddingAll": "18px", "contents": [
        {"type": "text", "text": "館別選擇完成", "weight": "bold", "size": "xl", "color": "#7B5600"},
        {"type": "text", "text": f"目前選擇：{venue_name}\n桌號：{session.get('room') or '1'}", "wrap": True, "margin": "md", "color": "#3E3100"},
        {"type": "text", "text": "下一步請輸入本次分析本金，系統會依正式訊號提供配置建議。", "wrap": True, "margin": "md", "color": "#665000"},
    ]}}})


def venue_panel(user_id: str) -> Dict[str, Any]:
    del user_id
    bubbles: List[Dict[str, Any]] = []
    for venue in VENUES:
        bubbles.append({"type": "bubble", "size": "kilo", "hero": {"type": "image", "url": _public_asset(f"/static/venues/{venue['image']}"), "size": "full", "aspectRatio": "1:1", "aspectMode": "cover"}, "body": {"type": "box", "layout": "vertical", "backgroundColor": "#FFF4B8", "paddingAll": "12px", "contents": [{"type": "text", "text": venue["name"], "weight": "bold", "align": "center", "color": "#4C3900"}, _postback_button("選擇", "venue", venue=venue["code"], room="1")]}})
    return _clean_flex({"type": "flex", "altText": "BGS AI預測系統－請選擇遊戲館", "contents": {"type": "carousel", "contents": bubbles}})


def manual_result_received_panel(outcome: str) -> Dict[str, Any]:
    value = str(outcome or "").upper()
    label, color = {"B": ("莊", "#D52B2B"), "P": ("閒", "#2667D8"), "T": ("和", "#159447")}.get(value, ("未知", "#7B5600"))
    return _clean_flex({"type": "flex", "altText": f"本局結果已更新：{label}", "contents": {"type": "bubble", "body": {"type": "box", "layout": "vertical", "backgroundColor": "#FFF4B8", "paddingAll": "18px", "contents": [{"type": "text", "text": f"本局結果已更新：{label}", "weight": "bold", "size": "xl", "color": color}, {"type": "text", "text": "系統正在結算上一筆預測，並同步 B/P/T 完整歷史、牌路模型與下一局三方機率。", "wrap": True, "margin": "md", "color": "#4C3900"}]}}})


def ready_panel(user_id: str, session: Mapping[str, Any]) -> Dict[str, Any]:
    del user_id
    venue = _venue_name(str(session.get("venue") or ""))
    bankroll = int(session.get("bankroll", 0) or 0)
    bankroll_text = f"{_format_money(bankroll)} 元" if bankroll > 0 else "尚未設定"
    body = {"type": "box", "layout": "vertical", "backgroundColor": "#FFF4B8", "paddingAll": "18px", "contents": [
        {"type": "text", "text": "BGS AI預測系統", "weight": "bold", "size": "xl", "color": "#7B5600"},
        {"type": "text", "text": f"分析館別：{venue}\n桌號：{session.get('room') or '1'}\n資金設定：{bankroll_text}\n\n選擇館別後請設定本金，再上傳最新完整遊戲畫面。首次辨識完成後，每局只需回報莊／閒／和。", "wrap": True, "margin": "md", "color": "#4C3900"},
        {"type": "box", "layout": "vertical", "spacing": "sm", "margin": "lg", "contents": [_postback_button("開始牌局分析", "start_screen"), _postback_button("設定／調整本金", "change_bankroll", color="#E29B19"), _postback_button("重新選擇館別", "venues", style="secondary")]},
    ]}
    return _clean_flex({"type": "flex", "altText": "BGS AI預測系統", "contents": {"type": "bubble", "size": "mega", "body": body}})


def bankroll_panel(user_id: str, session: Mapping[str, Any]) -> Dict[str, Any]:
    del user_id
    current = int(session.get("bankroll", 0) or 0)
    current_text = f"目前設定：{_format_money(current)} 元\n" if current > 0 else ""
    return _clean_flex({"type": "flex", "altText": "設定分析本金", "contents": {"type": "bubble", "size": "mega", "body": {"type": "box", "layout": "vertical", "backgroundColor": "#FFF4B8", "paddingAll": "18px", "contents": [
        {"type": "text", "text": "設定分析本金", "weight": "bold", "size": "xl", "color": "#7B5600"},
        {"type": "text", "text": f"{current_text}請直接輸入金額，例如：10000\n\n系統會依方向訊號、模型一致度與風險區間計算建議配置。", "wrap": True, "margin": "md", "color": "#4C3900"},
        _postback_button("返回館別選單", "venues", style="secondary"),
    ]}}})


def upload_request_panel(user_id: str, session: Mapping[str, Any]) -> Dict[str, Any]:
    del user_id
    return _clean_flex({"type": "flex", "altText": "上傳牌局畫面", "contents": {"type": "bubble", "size": "mega", "body": {"type": "box", "layout": "vertical", "backgroundColor": "#FFF4B8", "paddingAll": "18px", "contents": [
        {"type": "text", "text": "建立牌局資料", "weight": "bold", "size": "xl", "color": "#7B5600"},
        {"type": "text", "text": f"館別：{_venue_name(str(session.get('venue') or ''))}\n桌號：{session.get('room') or '1'}\n本金：{_format_money(session.get('bankroll', 0))} 元\n\n請開始上傳最新完整遊戲畫面進行分析。建議保留完整大路區域，避免裁掉左上起始格與六列格線。", "wrap": True, "margin": "md", "color": "#4C3900"},
        {"type": "text", "text": "首次畫面完成後，系統將建立初始牌路；後續每局只需回報實際開出莊或閒。", "wrap": True, "size": "sm", "margin": "md", "color": "#806A2A"},
        {"type": "box", "layout": "vertical", "spacing": "sm", "margin": "lg", "contents": [_postback_button("調整本金", "change_bankroll", color="#E29B19"), _postback_button("重新選擇館別", "venues", style="secondary"), _postback_button("結束本次分析", "end", style="secondary")]},
    ]}}})


def image_received_panel(session: Mapping[str, Any]) -> Dict[str, Any]:
    del session
    return _clean_flex({"type": "flex", "altText": "牌局畫面已接收", "contents": {"type": "bubble", "body": {"type": "box", "layout": "vertical", "backgroundColor": "#FFF4B8", "paddingAll": "18px", "contents": [
        {"type": "text", "text": "牌局畫面已接收", "weight": "bold", "size": "xl", "color": "#7B5600"},
        {"type": "text", "text": "系統將自動判斷完整遊戲畫面或牌路裁切圖，並同步處理：\n• 館別與桌號快速辨識\n• 大路莊閒序列建立\n• 牌路先行與統一機率模型\n\n分析完成後會自動推送下一局面板；後續每局只需點選莊或閒。", "wrap": True, "margin": "md", "color": "#4C3900"},
    ]}}})


def result_panel(user_id: str, session: Mapping[str, Any]) -> Dict[str, Any]:
    del user_id
    prediction = dict(session.get("last_prediction") or {})
    stats = dict(session.get("stats") or {})
    recommend = str(prediction.get("recommend_text") or "-")
    action = str(prediction.get("action_text") or "-")
    result_text = str(prediction.get("virtual_outcome_text") or "-")
    verdict = str(prediction.get("verdict_text") or "-")
    round_number = int(session.get("hand_number", 0) or 0)
    body_contents: List[Dict[str, Any]] = [
        {"type": "text", "text": f"分析結果 #{round_number}", "weight": "bold", "size": "xl", "color": "#7B5600"},
        {"type": "text", "text": f"館別：{_venue_name(str(session.get('venue') or ''))}｜桌號：{session.get('room') or '1'}\n牌靴：{str(session.get('shoe_id') or '-')[:12]}｜剩餘：{len(session.get('virtual_shoe') or [])} 張", "wrap": True, "size": "sm", "margin": "sm", "color": "#665000"},
        {"type": "separator", "margin": "md", "color": "#E1BD43"},
        {"type": "box", "layout": "vertical", "spacing": "sm", "margin": "md", "contents": [
            {"type": "text", "text": f"莊　{float(prediction.get('banker_rate', 0.0)):.2f}%", "color": "#D52B2B", "weight": "bold"},
            {"type": "text", "text": f"閒　{float(prediction.get('player_rate', 0.0)):.2f}%", "color": "#2667D8", "weight": "bold"},
            {"type": "text", "text": f"和　{float(prediction.get('tie_rate', 0.0)):.2f}%", "color": "#259B55", "weight": "bold"},
        ]},
        {"type": "text", "text": f"分析方向：{recommend}\n訊號：{action}｜品質：{prediction.get('confidence_label') or '偏低'}\n核心：超幾何分布＋粒子/蒙地卡羅驗證\n虛擬開獎：{result_text}｜{verdict}\n累計：{stats.get('wins', 0)} 勝 / {stats.get('losses', 0)} 負 / {stats.get('ties_skipped', 0)} 和不計 / {stats.get('observes', 0)} 觀望", "wrap": True, "margin": "md", "color": "#3E3100"},
        {"type": "box", "layout": "vertical", "spacing": "sm", "margin": "lg", "contents": [_postback_button("繼續分析", "predict"), _postback_button("重新建立牌靴", "reset", color="#E29B19"), _postback_button("結束分析", "end", style="secondary")]},
        {"type": "text", "text": "僅分析程式內建虛擬牌靴，未連接外部真人桌。", "wrap": True, "size": "xs", "margin": "md", "color": "#806A2A"},
    ]
    return _clean_flex({"type": "flex", "altText": f"分析結果：{recommend}", "contents": {"type": "bubble", "size": "mega", "body": {"type": "box", "layout": "vertical", "backgroundColor": "#FFF4B8", "paddingAll": "18px", "contents": body_contents}}})


def screen_result_panel(user_id: str, session: Mapping[str, Any]) -> Dict[str, Any]:
    del user_id
    prediction = dict(session.get("screen_last_prediction") or {})
    formal_code = str(prediction.get("internal_action") or prediction.get("internal_recommend") or prediction.get("action") or prediction.get("recommend") or "").upper().strip()
    if formal_code not in {"B", "P"}:
        banker = float(prediction.get("banker_rate", 0.0) or 0.0)
        player = float(prediction.get("player_rate", 0.0) or 0.0)
        formal_code = "B" if banker >= player else "P"
    formal_text = "莊" if formal_code == "B" else "閒"
    direction_color = "#D52B2B" if formal_code == "B" else "#2667D8"
    prediction["formal_direction"] = formal_code
    prediction["formal_direction_text"] = formal_text
    prediction["next_round_direction"] = formal_code
    prediction["next_round_direction_text"] = formal_text
    analysis_number = int(session.get("screen_analysis_count", 0) or 0)
    bankroll = int(prediction.get("bankroll", session.get("bankroll", 0)) or 0)
    suggested = int(prediction.get("suggested_bet_amount", 0) or 0)
    percentage = float(prediction.get("bet_percentage", 0.0) or 0.0)
    bet_level = str(prediction.get("bet_level_text") or "標準區間")
    bet_text = f"{_format_money(suggested)} 元（{percentage:.1f}%｜{bet_level}）" if suggested > 0 else f"0 元（{percentage:.1f}%）"
    return _clean_flex({"type": "flex", "altText": f"BGS AI 下一局方向：{formal_text}", "contents": {"type": "bubble", "size": "mega", "body": {"type": "box", "layout": "vertical", "backgroundColor": "#FFF4B8", "paddingAll": "18px", "contents": [
        {"type": "text", "text": f"BGS AI 下一局分析 #{analysis_number}", "weight": "bold", "size": "xl", "color": "#7B5600"},
        {"type": "separator", "margin": "md", "color": "#E1BD43"},
        {"type": "text", "text": f"下一局方向評估：{formal_text}", "weight": "bold", "size": "xl", "margin": "md", "color": direction_color},
        {"type": "box", "layout": "vertical", "spacing": "sm", "margin": "md", "contents": [
            {"type": "text", "text": f"莊　{float(prediction.get('banker_rate', 0.0)):.2f}%", "color": "#D52B2B", "weight": "bold"},
            {"type": "text", "text": f"閒　{float(prediction.get('player_rate', 0.0)):.2f}%", "color": "#2667D8", "weight": "bold"},
            {"type": "text", "text": f"和　{float(prediction.get('tie_rate', 0.0)):.2f}%", "color": "#259B55", "weight": "bold"},
        ]},
        {"type": "separator", "margin": "md", "color": "#E1BD43"},
        {"type": "text", "text": f"分析本金：{_format_money(bankroll)} 元\n建議配置：{bet_text}", "wrap": True, "margin": "md", "color": "#3E3100"},
        {"type": "box", "layout": "vertical", "spacing": "sm", "margin": "lg", "contents": [
            _postback_button("🔴 本局結果：莊", "road_append", color="#D52B2B", result="B"),
            _postback_button("🔵 本局結果：閒", "road_append", color="#2667D8", result="P"),
            _postback_button("🟢 本局結果：和", "road_append", color="#159447", result="T"),
            _postback_button("結束本次分析", "end", style="secondary"),
        ]},
    ]}}})


def ended_panel() -> Dict[str, Any]:
    return guide_panel()


def _activate_code(user_id: str, code: str) -> str:
    code = _normalize_access_code(code)
    now = int(datetime.now().timestamp())
    if code in PERMANENT_CODES:
        store.upsert_session(user_id, {"permanent_access": True, "access_until": 0, "status": "分析中"})
        return "永久版"
    if code in MONTHLY_CODES:
        store.upsert_session(user_id, {"permanent_access": False, "trial_started_at": now, "access_until": now + 30 * 24 * 60 * 60, "status": "分析中"})
        return "30日版"
    if code in TEMP_CODES:
        store.upsert_session(user_id, {"permanent_access": False, "trial_started_at": now, "access_until": now + 30 * 60, "status": "分析中"})
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
        return store.run_virtual_round(str(user_id or ""), run_virtual_round)
    finally:
        _PREDICTION_SLOTS.release()


def _refresh_screen_prediction(user_id: str, outcome: str, expected_run_id: str, cancel_event: threading.Event, deadline: float) -> Dict[str, Any]:
    value = str(outcome or "").upper().strip()
    if value not in {"B", "P", "T"}:
        raise ValueError("本局結果只能是莊、閒或和。")
    operation_lock = _user_image_lock(user_id)
    operation_lock_acquired = False
    prediction_slot_acquired = False
    try:
        _raise_if_manual_timed_out(cancel_event, deadline)
        lock_wait = _remaining_manual_time(deadline)
        if lock_wait <= 0.0 or not operation_lock.acquire(timeout=lock_wait):
            raise TimeoutError("同一使用者的上一個分析仍在處理中。")
        operation_lock_acquired = True
        _raise_if_manual_timed_out(cancel_event, deadline)
        _ensure_access(user_id)
        session = copy.deepcopy(store.get_session(user_id))
        if str(session.get("analysis_run_id") or "") != str(expected_run_id):
            raise RuntimeError("本次分析已結束或已重新開始，舊操作已忽略。")
        if not bool(session.get("analysis_active")):
            raise RuntimeError("本次分析已結束，請重新開始分析。")
        if not session.get("screen_last_prediction"):
            raise ValueError("請先上傳一次遊戲畫面，建立初始牌路。")
        expected_version = int(session.get("screen_data_version", 0) or 0)
        ocr = copy.deepcopy(dict(session.get("screen_last_ocr") or {}))
        detection = copy.deepcopy(dict(session.get("screen_last_detection") or {}))
        venue = str(ocr.get("venue_code") or session.get("venue") or "")
        room = str(ocr.get("room") or session.get("room") or session.get("last_confirmed_room") or "1")
        initial_history = [str(item).upper() for item in list(session.get("initial_image_history") or []) if str(item).upper() in {"B", "P", "T"}]
        manual_history = [str(item).upper() for item in list(session.get("manual_outcome_history") or []) if str(item).upper() in {"B", "P", "T"}]
        manual_history.append(value)
        raw_history = initial_history + manual_history
        road_state = _derive_road_state(raw_history)
        sequence = list(road_state["road_sequence"])
        tie_markers = dict(road_state["tie_markers"])
        previous_prediction_id = str(dict(session.get("screen_last_prediction") or {}).get("prediction_id", "") or "")
        remaining = int(round(estimate_remaining_cards(len(raw_history), decks=SHOE_DECKS)))
        screen_metadata = {"input_type": str(session.get("screen_input_type") or detection.get("input_type") or "full_screen"), "venue_source": str(session.get("screen_venue_source") or ocr.get("venue_source") or "session_selected"), "room_source": str(session.get("screen_room_source") or ocr.get("room_source") or "session_previous"), "room_confidence": float(session.get("screen_room_confidence", 0.0) or 0.0), "manual_update": True}
        _raise_if_manual_timed_out(cancel_event, deadline)
        semaphore_wait = min(float(PREDICTION_QUEUE_TIMEOUT), _remaining_manual_time(deadline))
        if semaphore_wait <= 0.0:
            raise TimeoutError("等待模型分析名額時已超時。")
        prediction_slot_acquired = _PREDICTION_SLOTS.acquire(timeout=semaphore_wait)
        if not prediction_slot_acquired:
            raise TimeoutError("目前分析人數較多，等待模型分析名額已超時。")
        try:
            prediction = predict_from_screenshot(
                copy.deepcopy(sequence),
                raw_outcomes=copy.deepcopy(raw_history),
                tie_markers=copy.deepcopy(tie_markers),
                remaining_cards=float(remaining),
                venue=str(venue or ""),
                room=str(room or ""),
                shoe_id=str(expected_run_id or ""),
                user_id=str(user_id or ""),
                screen_metadata=copy.deepcopy(screen_metadata),
                initial_grid_cells=copy.deepcopy(list(session.get("initial_grid_cells") or [])),
                initial_image_history=copy.deepcopy(initial_history),
                manual_outcome_history=copy.deepcopy(manual_history),
                previous_prediction_id=str(previous_prediction_id or ""),
                latest_actual_outcome=str(value or ""),
                shoe_context=copy.deepcopy(_exact_shoe_context(session)),
            )
        finally:
            if prediction_slot_acquired:
                _PREDICTION_SLOTS.release()
                prediction_slot_acquired = False
        _raise_if_manual_timed_out(cancel_event, deadline)
        prediction["latest_actual_outcome"] = value
        prediction["latest_actual_outcome_text"] = {"B": "莊", "P": "閒", "T": "和"}[value]
        prediction["remaining_cards_estimated_after_manual"] = True
        prediction = _attach_bankroll_advice(prediction, session)
        resolved = {"venue_code": venue, "venue_name": str(ocr.get("venue_name") or ""), "room": room, "remaining_cards": remaining, **screen_metadata}
        _raise_if_manual_timed_out(cancel_event, deadline)
        return store.update_screen_analysis(user_id, ocr=copy.deepcopy(ocr), detection=copy.deepcopy(detection), sequence=copy.deepcopy(sequence), raw_outcomes=copy.deepcopy(raw_history), tie_markers=copy.deepcopy(tie_markers), initial_image_history=copy.deepcopy(initial_history), manual_outcome_history=copy.deepcopy(manual_history), initial_grid_cells=copy.deepcopy(list(session.get("initial_grid_cells") or [])), recognition_quality={"recognized_count": int(session.get("initial_recognized_count", len(initial_history)) or len(initial_history)), "uncertain_count": int(session.get("initial_uncertain_count", 0) or 0)}, prediction=copy.deepcopy(prediction), resolved=copy.deepcopy(resolved), processing_ms=0.0, source=f"manual_result_{value}", expected_run_id=str(expected_run_id or ""), expected_data_version=expected_version)
    finally:
        if prediction_slot_acquired:
            try:
                _PREDICTION_SLOTS.release()
            except Exception:
                pass
        if operation_lock_acquired:
            operation_lock.release()


def _start_screen_flow(user_id: str, *, new_session: bool = True) -> Dict[str, Any]:
    _ensure_access(user_id)
    session = store.get_session(user_id)
    if not session.get("venue"):
        return {"panel": venue_panel(user_id), "state": "venue"}
    session = _start_new_uid_analysis(user_id) if new_session else store.begin_screen_analysis(user_id, clear_existing=False)
    if int(session.get("bankroll", 0) or 0) <= 0:
        session = _request_bankroll(user_id)
        return {"panel": bankroll_panel(user_id, session), "state": "bankroll"}
    return {"panel": upload_request_panel(user_id, session), "state": "image"}


def _process_screen_image_sync(user_id: str, message_id: str, expected_run_id: str, cancel_event: threading.Event, deadline: float) -> List[Dict[str, Any]]:
    temporary_image: Optional[Path] = None
    analysis_image: Optional[Path] = None
    started = time.perf_counter()
    image_lock = _user_image_lock(user_id)
    image_lock_acquired = False
    prediction_slot_acquired = False
    try:
        _raise_if_image_timed_out(cancel_event, deadline)
        lock_wait = _remaining_image_time(deadline)
        if lock_wait <= 0.0 or not image_lock.acquire(timeout=lock_wait):
            raise TimeoutError("同一使用者的上一張圖片仍在分析中。")
        image_lock_acquired = True
        _raise_if_image_timed_out(cancel_event, deadline)
        _ensure_access(user_id)
        current_session = copy.deepcopy(store.get_session(user_id))
        if str(current_session.get("analysis_run_id") or "") != str(expected_run_id):
            return [_text("本次分析已重新開始，請重新上傳最新圖片。")]
        if not bool(current_session.get("analysis_active")):
            return [_text("本次分析已結束，請重新點擊「開始分析」。")]
        semaphore_wait = min(float(PREDICTION_QUEUE_TIMEOUT), _remaining_image_time(deadline))
        if semaphore_wait <= 0.0:
            raise TimeoutError("等待圖片分析名額時已超時。")
        prediction_slot_acquired = _PREDICTION_SLOTS.acquire(timeout=semaphore_wait)
        if not prediction_slot_acquired:
            raise TimeoutError("目前分析人數較多，等待圖片分析名額已超時。")
        download_started = time.perf_counter()
        temporary_image = _download_line_image(message_id, timeout_seconds=_remaining_image_time(deadline))
        download_ms = (time.perf_counter() - download_started) * 1000.0
        _raise_if_image_timed_out(cancel_event, deadline)
        analysis_image = _prepare_analysis_image(temporary_image)
        screen = analyze_game_screen(analysis_image, copy.deepcopy(current_session))
        _raise_if_image_timed_out(cancel_event, deadline)
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
            return [_road_error_message(f"{detail}。請傳送包含完整大路的畫面或牌路裁切圖。")]
        if not recognition_quality_ok:
            return [_road_error_message(f"本次只確認 {recognized_count} 格，另有 {uncertain_count} 格無法可靠判定；為避免總局數錯誤，已停止送入模型。請裁切只保留完整大路後重新上傳。")]
        resolved = dict(screen.get("resolved") or {})
        remaining = int(resolved.get("remaining_cards") or TOTAL_SHOE_CARDS)
        resolved["remaining_cards"] = remaining
        expected_version = int(current_session.get("screen_data_version", 0) or 0)
        screen_metadata = {"input_type": str(screen.get("input_type") or resolved.get("input_type") or "full_screen"), "venue_source": str(resolved.get("venue_source") or "session_selected"), "room_source": str(resolved.get("room_source") or "session_selected"), "room_confidence": float(resolved.get("room_confidence", 0.0) or 0.0), "ocr_timed_out": bool(resolved.get("ocr_timed_out")), "vision_timings": dict(screen.get("timings") or {})}
        resolved.update(screen_metadata)
        _raise_if_image_timed_out(cancel_event, deadline)
        model_started = time.perf_counter()
        prediction = predict_from_screenshot(
            copy.deepcopy(sequence),
            raw_outcomes=copy.deepcopy(raw_outcomes),
            tie_markers=copy.deepcopy(tie_markers),
            remaining_cards=float(remaining),
            venue=str(resolved.get("venue_code") or current_session.get("venue") or ""),
            room=str(resolved.get("room") or current_session.get("room") or current_session.get("last_confirmed_room") or "1"),
            shoe_id=str(expected_run_id or ""),
            user_id=str(user_id or ""),
            screen_metadata=copy.deepcopy(screen_metadata),
            initial_grid_cells=copy.deepcopy(grid_cells),
            initial_image_history=copy.deepcopy(raw_outcomes),
            manual_outcome_history=[],
            shoe_context=copy.deepcopy(_exact_shoe_context(current_session)),
        )
        model_ms = (time.perf_counter() - model_started) * 1000.0
        _raise_if_image_timed_out(cancel_event, deadline)
        prediction = _attach_bankroll_advice(prediction, current_session)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        source = f"screen_image_{screen_metadata['input_type']}"
        session = store.update_screen_analysis(user_id, ocr=copy.deepcopy(dict(screen.get("ocr") or {})), detection=copy.deepcopy(dict(screen.get("road") or {})), sequence=copy.deepcopy(sequence), raw_outcomes=copy.deepcopy(raw_outcomes), tie_markers=copy.deepcopy(tie_markers), initial_image_history=copy.deepcopy(raw_outcomes), manual_outcome_history=[], initial_grid_cells=copy.deepcopy(grid_cells), recognition_quality={"recognized_count": recognized_count, "uncertain_count": uncertain_count, "quality_ok": recognition_quality_ok}, prediction=copy.deepcopy(prediction), resolved=copy.deepcopy(resolved), processing_ms=elapsed_ms, source=source, expected_run_id=str(expected_run_id or ""), expected_data_version=expected_version)
        print("screen_timing", json.dumps({"uid": user_id[-8:], "download_ms": round(download_ms, 2), **dict(screen.get("timings") or {}), "model_ms": round(model_ms, 2), "total_ms": round(elapsed_ms, 2), "input_type": screen_metadata["input_type"], "room_source": screen_metadata["room_source"], "road_count": len(sequence), "reply_mode": True}, ensure_ascii=False))
        return [screen_result_panel(user_id, session)]
    except AccessExpiredError:
        return [_text("試用已到期，請聯繫管理員開通。")]
    except TimeoutError:
        raise
    except Exception as exc:
        traceback.print_exc()
        message = str(exc)
        if "舊結果已忽略" in message or "已重新開始" in message or "已結束" in message:
            return [_text("本次圖片已失效，請重新上傳最新圖片。")]
        return [_road_error_message(f"圖片處理失敗：{message}")]
    finally:
        if prediction_slot_acquired:
            try:
                _PREDICTION_SLOTS.release()
            finally:
                prediction_slot_acquired = False
        if analysis_image is not None and analysis_image != temporary_image:
            analysis_image.unlink(missing_ok=True)
        if temporary_image is not None:
            temporary_image.unlink(missing_ok=True)
        if image_lock_acquired:
            image_lock.release()


async def _process_manual_outcome_via_reply(token: str, user_id: str, outcome: str, expected_run_id: str) -> bool:
    uid_tail = str(user_id or "")[-8:]
    started = time.perf_counter()
    cancel_event = threading.Event()
    work_timeout = max(1.0, MANUAL_OUTCOME_TIMEOUT - 2.0)
    deadline = started + work_timeout
    value = str(outcome or "").upper().strip()
    print("manual_start", json.dumps({"event": "manual_start", "uid": uid_tail, "outcome": value, "run_id": str(expected_run_id or "")[:12], "delivery": "replyToken", "timeout_s": MANUAL_OUTCOME_TIMEOUT, "work_deadline_s": work_timeout}, ensure_ascii=False))
    messages: List[Dict[str, Any]]
    result_event = "manual_reply_result"
    try:
        session = await asyncio.wait_for(asyncio.to_thread(_refresh_screen_prediction, user_id, value, expected_run_id, cancel_event, deadline), timeout=MANUAL_OUTCOME_TIMEOUT)
        predict_ms = (time.perf_counter() - started) * 1000.0
        print("manual_predict_ok", json.dumps({"event": "manual_predict_ok", "uid": uid_tail, "outcome": value, "ms": round(predict_ms, 2), "delivery": "replyToken"}, ensure_ascii=False))
        messages = [screen_result_panel(user_id, session)]
    except (asyncio.TimeoutError, TimeoutError):
        cancel_event.set()
        result_event = "manual_timeout"
        print("manual_timeout", json.dumps({"event": "manual_timeout", "uid": uid_tail, "outcome": value, "timeout_s": MANUAL_OUTCOME_TIMEOUT, "work_deadline_s": work_timeout, "delivery": "replyToken"}, ensure_ascii=False))
        messages = [_text("⚠️ 本局結果更新超時，已取消本次寫入。\n請稍後再按一次莊／閒／和；若持續發生，請重新上傳圖片。")]
    except AccessExpiredError:
        cancel_event.set()
        result_event = "manual_access_expired"
        messages = [_text("試用已到期，請聯繫管理員開通。")]
    except Exception as exc:
        cancel_event.set()
        traceback.print_exc()
        message = str(exc)
        result_event = "manual_error"
        print("manual_error", json.dumps({"event": "manual_error", "uid": uid_tail, "outcome": value, "error": message[:300], "delivery": "replyToken"}, ensure_ascii=False))
        if any(key in message for key in ("舊操作已忽略", "舊結果已忽略", "已重新開始", "已結束")):
            messages = [_text("本次操作已過期（分析已重新開始或結束），請重新開始分析。")]
        else:
            messages = [_text(f"本局結果更新失敗：{message}")]
    replied = await asyncio.to_thread(_reply, token, messages)
    total_ms = (time.perf_counter() - started) * 1000.0
    print("manual_reply_ok" if replied else "manual_reply_fail", json.dumps({"event": "manual_reply_ok" if replied else "manual_reply_fail", "uid": uid_tail, "outcome": value, "result_event": result_event, "total_ms": round(total_ms, 2), "delivery": "replyToken"}, ensure_ascii=False))
    return replied


def _public_session(session: Mapping[str, Any]) -> Dict[str, Any]:
    data = copy_session = copy.deepcopy(dict(session))
    copy_session.pop("virtual_shoe", None)
    copy_session["remaining_cards"] = len(session.get("virtual_shoe") or [])
    copy_session["venue_name"] = _venue_name(str(session.get("venue") or ""))
    copy_session["venues"] = [{**venue, "image_url": f"/static/venues/{venue['image']}"} for venue in VENUES]
    copy_session["analysis_history"] = list(session.get("analysis_history") or [])[-40:]
    copy_session["round_history"] = list(session.get("round_history") or [])[-60:]
    copy_session["history"] = list(session.get("raw_outcomes") or [])[-1000:]
    copy_session["round_no"] = len(copy_session["history"]) + 1
    copy_session["last_prediction"] = copy.deepcopy(dict(session.get("screen_last_prediction") or session.get("last_prediction") or {}))
    return data


def _liff_access_payload(user_id: str, *, start_trial: bool = False) -> Dict[str, Any]:
    status = dict(store.access_status(user_id, start_trial=start_trial))
    session = store.get_session(user_id)
    permanent = bool(session.get("permanent_access"))
    trial_started = int(session.get("trial_started_at", 0) or 0)
    active = bool(status.get("allowed"))
    can_start_trial = not permanent and trial_started <= 0
    seconds_left = status.get("seconds_left")
    return {"user_id": user_id, "active": active, "allowed": active, "can_start_trial": can_start_trial, "state": "active" if active else "trial_available" if can_start_trial else "expired", "plan": "permanent" if permanent else "trial", "plan_label": "永久版" if permanent else "試用中" if active else "可開始試用" if can_start_trial else "已到期", "remaining_seconds": seconds_left, "expires_at_taipei": "永久" if permanent else "", "redirect_after_seconds": 30, "message": "使用權限正常" if active else "首次分析將自動啟用試用" if can_start_trial else "試用已到期，請聯繫管理員"}


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "liff.html")


@app.get("/liff")
def liff_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "liff.html")


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({
        "ok": True,
        "version": "10.7.0-linucb-single-brain-v5",
        "engine": "CONTEXTUAL_LINUCB_SINGLE_BRAIN_BP",
        "activation_code_fix": True,
        "activation_persistence_check": True,
        "storage_path": str(getattr(store, "SESSION_DATA_FILE", "")),
        "activation_code_counts": {"permanent": len(PERMANENT_CODES), "monthly": len(MONTHLY_CODES), "temporary": len(TEMP_CODES)},
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
        "adaptive_ensemble": False,
        "single_brain_linucb": True,
        "exact_shoe_composition": True,
        "banker_commission_ev": True,
        "fractional_kelly": True,
        "mandatory_bet_min_percent": float(MIN_BET_RATIO * 100.0),
        "mandatory_bet_max_percent": float(MAX_BET_RATIO * 100.0),
        "ocr_preload_enabled": os.getenv("OCR_PRELOAD", "0").strip() == "1",
        "deepseek_active": False,
        "stale_background_guard": True,
        "bankroll_flow": True,
        "immediate_image_ack": False,
        "background_push_result": False,
        "manual_background_push": False,
        "manual_result_via_reply": True,
        "image_result_via_reply": True,
        "image_reply_timeout_seconds": LINE_IMAGE_ANALYSIS_TIMEOUT,
        "manual_outcome_timeout_seconds": MANUAL_OUTCOME_TIMEOUT,
        "push_max_retries": PUSH_MAX_RETRIES,
        "line_default_mode": "screen",
    })


@app.head("/health")
def health_head() -> Response:
    return Response(status_code=200)


@app.get("/ping")
def ping() -> PlainTextResponse:
    return PlainTextResponse("OK")


@app.head("/ping")
def ping_head() -> Response:
    return Response(status_code=200)


@app.get("/api/config")
def api_config() -> JSONResponse:
    return JSONResponse({"ok": True, "liffId": LIFF_ID, "adminLineUrl": ADMIN_LINE_URL, "accessRedirectSeconds": 30, "venues": VENUES, "rooms": [str(value) for value in range(1, 21)]})


@app.get("/api/access/status")
def api_access_status(user_id: str) -> JSONResponse:
    try:
        return JSONResponse({"ok": True, "access": _liff_access_payload(user_id)})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/access/activate")
def api_access_activate(payload: ActivationRequest) -> JSONResponse:
    try:
        plan = _activate_code(payload.user_id, payload.code)
        access = _liff_access_payload(payload.user_id)
        access["message"] = f"{plan}開通成功"
        return JSONResponse({"ok": True, "access": access})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/session/current")
def api_session_current(user_id: str) -> JSONResponse:
    return JSONResponse({"ok": True, "session": _public_session(store.get_session(user_id))})


@app.post("/api/session/start")
def api_session_start(payload: LiffSessionStartRequest) -> JSONResponse:
    venue = payload.venue.upper().strip()
    if venue not in VENUE_BY_CODE:
        raise HTTPException(status_code=400, detail="無效館別")
    try:
        store.select_venue(payload.user_id, venue, payload.room)
        session = store.clear_screen_analysis(payload.user_id, keep_mode=True)
        session = store.upsert_session(payload.user_id, {"shoe_id": str(payload.shoe_id or session.get("shoe_id") or ""), "status": "輸入中", "analysis_active": True, "awaiting_screenshot": False, "exact_remaining_counts": [], "exact_remaining_updated_at": 0})
        return JSONResponse({"ok": True, "session": _public_session(session)})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/round/add")
def api_round_add(payload: RoundResultRequest) -> JSONResponse:
    try:
        session = store.append_road_result(payload.user_id, payload.result)
        return JSONResponse({"ok": True, "session": _public_session(session)})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/round/undo")
def api_round_undo(payload: UserRequest) -> JSONResponse:
    try:
        session = store.get_session(payload.user_id)
        raw = list(session.get("raw_outcomes") or [])
        if raw:
            raw.pop()
        session = store.set_road_sequence(payload.user_id, [value for value in raw if value in {"B", "P"}], raw_outcomes=raw, source="manual")
        return JSONResponse({"ok": True, "session": _public_session(session)})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/session/reset")
def api_session_reset(payload: UserRequest) -> JSONResponse:
    session = store.clear_screen_analysis(payload.user_id, keep_mode=True)
    session = store.upsert_session(payload.user_id, {"exact_remaining_counts": [], "exact_remaining_updated_at": 0})
    return JSONResponse({"ok": True, "session": _public_session(session)})


@app.post("/api/session/end")
def api_session_end(payload: UserRequest) -> JSONResponse:
    session = _end_uid_analysis(payload.user_id)
    return JSONResponse({"ok": True, "session": _public_session(session)})


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


@app.post("/api/shoe/cards")
def api_shoe_cards(payload: ShoeCardsRequest) -> JSONResponse:
    try:
        session = store.set_observed_cards(payload.user_id, payload.cards, replace=bool(payload.replace))
        return JSONResponse({"ok": True, "session": _public_session(session)})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/shoe/exact-counts")
def api_exact_remaining_counts(payload: ExactRemainingCountsRequest) -> JSONResponse:
    try:
        counts = validate_remaining_counts(payload.remaining_counts, decks=int(payload.decks))
        session = store.upsert_session(payload.user_id, {"exact_remaining_counts": list(counts), "exact_remaining_decks": int(payload.decks), "exact_remaining_updated_at": int(time.time())})
        return JSONResponse({"ok": True, "session": _public_session(session)})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/predict")
async def api_predict(payload: UserRequest) -> JSONResponse:
    try:
        session = copy.deepcopy(store.get_session(payload.user_id))
        raw_outcomes = copy.deepcopy(list(session.get("raw_outcomes") or []))
        if raw_outcomes:
            _ensure_access(payload.user_id)
            previous_screen_prediction = copy.deepcopy(dict(session.get("screen_last_prediction") or {}))
            previous_timeline = dict(previous_screen_prediction.get("timeline_alignment") or {})
            previous_raw_rounds = int(previous_timeline.get("raw_round_index", -1) or -1)
            latest_actual_outcome = str(raw_outcomes[-1]).upper() if len(raw_outcomes) > previous_raw_rounds and str(raw_outcomes[-1]).upper() in {"B", "P", "T"} else ""
            prediction = await asyncio.to_thread(
                predict_from_screenshot,
                copy.deepcopy([value for value in raw_outcomes if value in {"B", "P"}]),
                raw_outcomes=copy.deepcopy(raw_outcomes),
                remaining_cards=0,
                venue=str(session.get("venue") or ""),
                room=str(session.get("room") or "1"),
                shoe_id=str(session.get("shoe_id") or session.get("analysis_run_id") or ""),
                user_id=str(payload.user_id or ""),
                initial_image_history=[],
                manual_outcome_history=copy.deepcopy(raw_outcomes),
                previous_prediction_id=str(previous_screen_prediction.get("prediction_id") or ""),
                latest_actual_outcome=str(latest_actual_outcome or ""),
                shoe_context=copy.deepcopy(_exact_shoe_context(session)),
            )
            prediction = _attach_bankroll_advice(prediction, session)
            updated = store.upsert_session(payload.user_id, {"screen_last_prediction": copy.deepcopy(prediction), "last_prediction": copy.deepcopy(prediction), "pending_prediction": copy.deepcopy(prediction), "screen_prediction_version": int(session.get("screen_data_version", 0) or 0), "screen_analysis_count": int(session.get("screen_analysis_count", 0) or 0) + 1, "status": "分析完成"})
            return JSONResponse({"ok": True, "state": "predicted", "prediction": prediction, "access": _liff_access_payload(payload.user_id), "session": _public_session(updated)})
        result = await asyncio.to_thread(_start_screen_flow, payload.user_id)
        return JSONResponse({"ok": True, "state": result["state"], "session": _public_session(store.get_session(payload.user_id))})
    except (PermissionError, AccessExpiredError) as exc:
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
                _reply(token, [guide_panel()])
                continue
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
                cancel_event = threading.Event()
                deadline = time.perf_counter() + LINE_IMAGE_ANALYSIS_TIMEOUT
                try:
                    messages = await asyncio.wait_for(asyncio.to_thread(_process_screen_image_sync, user_id, message_id, run_id, cancel_event, deadline), timeout=LINE_IMAGE_ANALYSIS_TIMEOUT)
                except TimeoutError:
                    cancel_event.set()
                    messages = [_text("⚠️ 伺服器分析超時，請重新上傳一次圖片試試看！")]
                except Exception as exc:
                    cancel_event.set()
                    traceback.print_exc()
                    messages = [_road_error_message(f"圖片處理失敗：{exc}")]
                _reply(token, messages or [_text("圖片分析未產生結果，請重新上傳一次。")])
                continue
            if event_type == "message" and message_type == "text":
                raw_text = str(message.get("text") or "")
                text = unicodedata.normalize("NFKC", raw_text).strip()
                access_code = _normalize_access_code(raw_text)
                activation_match = access_code in ALL_CODES
                print("activation_debug", json.dumps({"uid": user_id[-8:], "normalized": access_code, "length": len(access_code), "matched": activation_match, "permanent": access_code in PERMANENT_CODES}, ensure_ascii=False))
                if activation_match:
                    plan = _activate_code(user_id, access_code)
                    saved = store.get_session(user_id)
                    if plan == "永久版" and not bool(saved.get("permanent_access")):
                        raise RuntimeError("開通資料未成功寫入，請檢查 SESSION_DATA_FILE 儲存路徑。")
                    _reply(token, [_text(f"✅ 已開通：{plan}"), guide_panel()])
                    continue
                if text in {"開通碼檢查", "檢查開通碼", "版本檢查"}:
                    _reply(token, [_text("BGS AI預測系統版本：10.7.0\n" f"永久碼載入數：{len(PERMANENT_CODES)}\n" f"aaa1888007 已載入：{'是' if 'aaa1888007' in PERMANENT_CODES else '否'}")])
                    continue
                session = store.get_session(user_id)
                bankroll_command = bool(re.fullmatch(r"(?:本金|金額|資金)\s*[:：=]?\s*[0-9,，＄$ ]+", text, flags=re.IGNORECASE))
                if bool(session.get("awaiting_bankroll")) or bankroll_command:
                    try:
                        bankroll = _parse_bankroll(text)
                        session = store.set_bankroll(user_id, bankroll, begin_screen=True)
                        _reply(token, [_text(f"✅ 本金已設定為 {_format_money(bankroll)} 元"), upload_request_panel(user_id, session)])
                    except ValueError as exc:
                        _reply(token, [_text(str(exc)), bankroll_panel(user_id, session)])
                    continue
                if text in {"🔴 本局：莊", "🔴 本局結果：莊", "補輸莊", "補莊", "莊"}:
                    current = store.get_session(user_id)
                    if not current.get("screen_last_prediction"):
                        _reply(token, [_text("請先點擊開始分析並上傳一次遊戲畫面。")])
                        continue
                    await _process_manual_outcome_via_reply(token, user_id, "B", str(current.get("analysis_run_id") or ""))
                    continue
                if text in {"🔵 本局：閒", "🔵 本局結果：閒", "補輸閒", "補閒", "閒"}:
                    current = store.get_session(user_id)
                    if not current.get("screen_last_prediction"):
                        _reply(token, [_text("請先點擊開始分析並上傳一次遊戲畫面。")])
                        continue
                    await _process_manual_outcome_via_reply(token, user_id, "P", str(current.get("analysis_run_id") or ""))
                    continue
                if text in {"🟢 本局：和", "🟢 本局結果：和", "補輸和", "補和", "和"}:
                    current = store.get_session(user_id)
                    if not current.get("screen_last_prediction"):
                        _reply(token, [_text("請先點擊開始分析並上傳一次遊戲畫面。")])
                        continue
                    await _process_manual_outcome_via_reply(token, user_id, "T", str(current.get("analysis_run_id") or ""))
                    continue
                if text in {"🔄 清除重來", "清除路紙", "清除畫面", "重來"}:
                    result = _start_screen_flow(user_id, new_session=True)
                    _reply(token, [_text("🔄 已清除這個 UID 的舊牌路。"), result["panel"]])
                    continue
                if text in {"開始", "選館", "重新選館", "館別"}:
                    _reply(token, [venue_panel(user_id)])
                    continue
                if text in {"開始分析"}:
                    _reply(token, [venue_panel(user_id)])
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
                query = {key: values[0] for key, values in urllib.parse.parse_qs(str((event.get("postback") or {}).get("data") or "")).items()}
                action_name = str(query.get("action") or "")
                if action_name == "start_guide":
                    _reply(token, [venue_panel(user_id)])
                elif action_name == "road_append":
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
                    await _process_manual_outcome_via_reply(token, user_id, value, str(current.get("analysis_run_id") or ""))
                elif action_name == "road_clear":
                    result = _start_screen_flow(user_id, new_session=True)
                    _reply(token, [_text("🔄 已清除這個 UID 的舊牌路。"), result["panel"]])
                elif action_name == "venue":
                    venue = str(query.get("venue") or "").upper()
                    if venue not in VENUE_BY_CODE:
                        raise ValueError("無效館別")
                    session = store.select_venue(user_id, venue, str(query.get("room") or "1"))
                    session = _request_bankroll(user_id)
                    _reply(token, [selected_venue_panel(session), bankroll_panel(user_id, session)])
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
            _reply(token, [_text("試用已到期，請聯繫管理員開通。"), {"type": "template", "altText": "聯繫管理員", "template": {"type": "buttons", "text": "試用已到期", "actions": [{"type": "uri", "label": "聯繫管理員", "uri": ADMIN_LINE_URL}]}}])
        except PermissionError:
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
