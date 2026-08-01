"""BGS V7 hypergeometric click-only virtual-shoe baccarat application.

Features:
- Mobile web UI matching the supplied yellow card-style layout.
- LINE Flex venue selection with uploaded venue artwork.
- No point input: every click predicts and then deals one internal virtual hand.
- Eight-deck depletion, Monte Carlo replicas, hidden-order particle ensemble,
  calibrated sequence component, and honest virtual hit/miss statistics.

The model is connected only to its own virtual shoe, not an external live table.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import threading
import traceback
import urllib.parse
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
    1, min(4, int(os.getenv("APP_MAX_CONCURRENT_PREDICTIONS", "2") or "2"))
)
PREDICTION_QUEUE_TIMEOUT = max(
    5, min(55, int(os.getenv("APP_PREDICTION_QUEUE_TIMEOUT", "45") or "45"))
)
_PREDICTION_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_PREDICTIONS)


VENUES: List[Dict[str, str]] = [
    {"code": "DG", "name": "DG真人", "image": "dg.png"},
    {"code": "MT", "name": "MT真人", "image": "mt.png"},
    {"code": "DB", "name": "DB真人", "image": "db.png"},
    {"code": "SA", "name": "SA真人", "image": "sa.png"},
    {"code": "OB", "name": "歐博真人", "image": "ob.png"},
    {"code": "T9", "name": "T9真人", "image": "t9.png"},
]
VENUE_BY_CODE = {venue["code"]: venue for venue in VENUES}


def _code_set(env_name: str, defaults: str) -> set[str]:
    raw = os.getenv(env_name, defaults)
    return {item.strip() for item in raw.split(",") if item.strip()}


# Defaults preserve the user's existing deployment behavior. Move these to
# Render environment variables when convenient.
PERMANENT_CODES = _code_set(
    "PERMANENT_CODES", "aaa1688003,aaa1888007,aaa1000889"
)
MONTHLY_CODES = _code_set("MONTHLY_CODES", "aaa13002,aaa15001,aaa199801")
TEMP_CODES = _code_set("TEMP_CODES", "aaaa1999152,aaa345556,aaa987743")
ALL_CODES = PERMANENT_CODES | MONTHLY_CODES | TEMP_CODES


app = FastAPI(
    title="BGS V7 Hypergeometric Virtual Shoe Bot",
    version="7.0.0",
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


def _reply(token: str, messages: List[Dict[str, Any]]) -> None:
    if not token:
        return
    if not CHANNEL_ACCESS_TOKEN:
        print(json.dumps(messages, ensure_ascii=False))
        return
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


def _text(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": str(text)[:5000]}


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


def ready_panel(user_id: str, session: Mapping[str, Any]) -> Dict[str, Any]:
    venue = _venue_name(str(session.get("venue") or ""))
    body = {
        "type": "box",
        "layout": "vertical",
        "backgroundColor": "#FFF4B8",
        "paddingAll": "18px",
        "contents": [
            {
                "type": "text",
                "text": "虛擬牌靴已建立",
                "weight": "bold",
                "size": "xl",
                "color": "#7B5600",
            },
            {
                "type": "text",
                "text": (
                    f"館別：{venue}\n"
                    f"桌號：{session.get('room') or '1'}\n"
                    f"牌靴：{session.get('shoe_id') or '-'}\n"
                    f"剩餘牌數：{len(session.get('virtual_shoe') or [])}\n"
                    f"中途預跑：{int(session.get('warmup_rounds', 0) or 0)} 局\n\n"
                    "不需輸入點數，直接點擊開始分析。"
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
                    _postback_button("開始分析", "predict"),
                    _postback_button(
                        "重新選館",
                        "venues",
                        style="secondary",
                    ),
                ],
            },
        ],
    }
    return _clean_flex(
        {
            "type": "flex",
            "altText": "虛擬牌靴已建立",
            "contents": {"type": "bubble", "size": "mega", "body": body},
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
                        _postback_button("再次開始", "venues"),
                    ],
                },
            },
        }
    )


def _activate_code(user_id: str, code: str) -> str:
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
        raise PermissionError("試用已到期")
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
            "version": "7.0.0",
            "engine": "V7_HYPERGEOMETRIC_PARTICLE_MONTE_CARLO",
            "input_required": False,
            "virtual_only": True,
            "public_base_url_configured": bool(PUBLIC_BASE_URL),
            "venues": [venue["code"] for venue in VENUES],
            "max_concurrent_predictions": MAX_CONCURRENT_PREDICTIONS,
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
        session = await asyncio.to_thread(_run_prediction, payload.user_id)
        return JSONResponse({"ok": True, "session": _public_session(session)})
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/reset")
def api_reset(payload: UserRequest) -> JSONResponse:
    session = store.reset_shoe(payload.user_id, reset_stats=False)
    return JSONResponse({"ok": True, "session": _public_session(session)})


@app.post("/api/end")
def api_end(payload: UserRequest) -> JSONResponse:
    session = store.end_session(payload.user_id)
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
        user_id = str(
            source.get("userId") or source.get("groupId") or source.get("roomId") or "anonymous"
        )
        try:
            event_type = event.get("type")
            if event_type == "follow":
                _reply(token, [venue_panel(user_id)])
                continue

            if event_type == "message" and (event.get("message") or {}).get("type") == "text":
                text = str((event.get("message") or {}).get("text") or "").strip()
                if text in ALL_CODES:
                    plan = _activate_code(user_id, text)
                    _reply(token, [_text(f"✅ 已開通：{plan}"), venue_panel(user_id)])
                    continue

                if text in {"開始", "開始分析", "選館", "重新選館", "館別"}:
                    _reply(token, [venue_panel(user_id)])
                    continue

                session = store.get_session(user_id)
                if text in {"新牌靴", "換靴", "重置牌靴"}:
                    session = store.reset_shoe(user_id)
                    _reply(token, [ready_panel(user_id, session)])
                    continue

                if text in {"結束", "結束分析"}:
                    store.end_session(user_id)
                    _reply(token, [ended_panel()])
                    continue

                if session.get("venue"):
                    session = await asyncio.to_thread(_run_prediction, user_id)
                    _reply(token, [result_panel(user_id, session)])
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
                action_name = query.get("action")
                if action_name == "venue":
                    venue = str(query.get("venue") or "").upper()
                    if venue not in VENUE_BY_CODE:
                        raise ValueError("無效館別")
                    session = store.select_venue(
                        user_id,
                        venue,
                        str(query.get("room") or "1"),
                    )
                    _reply(token, [ready_panel(user_id, session)])
                elif action_name == "predict":
                    session = await asyncio.to_thread(_run_prediction, user_id)
                    _reply(token, [result_panel(user_id, session)])
                elif action_name == "reset":
                    session = store.reset_shoe(user_id)
                    _reply(token, [ready_panel(user_id, session)])
                elif action_name == "end":
                    store.end_session(user_id)
                    _reply(token, [ended_panel()])
                else:
                    _reply(token, [venue_panel(user_id)])
                continue
        except PermissionError:
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
                                {"type": "uri", "label": "聯繫管理員", "uri": ADMIN_LINE_URL}
                            ],
                        },
                    },
                ],
            )
        except Exception as exc:
            traceback.print_exc()
            _reply(token, [_text(f"系統忙碌：{exc}")])

    return JSONResponse({"ok": True})
