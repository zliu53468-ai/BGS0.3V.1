"""LINE chatbot for fast baccarat conditional Monte Carlo predictions.

Flow:
1. User starts a shoe.
2. Bot immediately returns the next-round panel.
3. User taps B / P / T after the actual result.
4. The result is stored and the next prediction panel is returned immediately.
5. Trial expiration produces a Flex panel with a direct administrator LINE link.
6. Existing activation codes can be typed directly in chat.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

import store
from predictor import predict

BASE_DIR = Path(__file__).resolve().parent
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "").strip()
TAIPEI_TZ = timezone(timedelta(hours=8))

ADMIN_LINE_ID = os.getenv("ADMIN_LINE_ID", "@jins888")
ADMIN_LINE_URL = os.getenv("ADMIN_LINE_URL", "https://line.me/R/ti/p/%40jins888")
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "30"))
TEMP_TRIAL_MINUTES = int(os.getenv("TEMP_TRIAL_MINUTES", "30"))
MONTHLY_DAYS = int(os.getenv("MONTHLY_DAYS", "30"))
ACCESS_DATA_FILE = Path(
    os.getenv("ACCESS_DATA_FILE", str(BASE_DIR / "data" / "access_control.json"))
)

PERMANENT_CODES = {"aaa1688003", "aaa1888007", "aaa1000889"}
MONTHLY_CODES = {"aaa13002", "aaa15001", "aaa199801"}
TEMP_TRIAL_CODES = {"aaaa1999152", "aaa345556", "aaa987743"}
ALL_ACCESS_CODES = PERMANENT_CODES | MONTHLY_CODES | TEMP_TRIAL_CODES

app = FastAPI(title="Baccarat LINE Monte Carlo Bot", version="3.0.0")


def now_taipei() -> datetime:
    return datetime.now(TAIPEI_TZ)


def _iso(dt: datetime) -> str:
    return dt.astimezone(TAIPEI_TZ).isoformat(timespec="seconds")


def _parse_dt(value: Any) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(value))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=TAIPEI_TZ)
        return dt.astimezone(TAIPEI_TZ)
    except Exception:
        return None


def _load_access() -> Dict[str, Any]:
    try:
        if not ACCESS_DATA_FILE.exists():
            return {}
        with ACCESS_DATA_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_access(data: Dict[str, Any]) -> None:
    ACCESS_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = ACCESS_DATA_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    tmp.replace(ACCESS_DATA_FILE)


def access_status(user_id: str) -> Dict[str, Any]:
    rec = _load_access().get(user_id) or {}
    now = now_taipei()
    if rec.get("permanent"):
        return {"active": True, "plan": "permanent", "label": "永久版"}
    access_exp = _parse_dt(rec.get("access_expires_at"))
    if access_exp and access_exp > now:
        return {"active": True, "plan": rec.get("plan", "monthly"), "label": "付費方案", "expires": _iso(access_exp)}
    trial_exp = _parse_dt(rec.get("trial_expires_at"))
    if trial_exp and trial_exp > now:
        return {"active": True, "plan": "trial", "label": "試用中", "expires": _iso(trial_exp)}
    if rec.get("used_trial"):
        return {"active": False, "expired": True, "label": "已到期"}
    return {"active": False, "trial_available": True, "label": "可開始試用"}


def ensure_access(user_id: str) -> Dict[str, Any]:
    status = access_status(user_id)
    if status.get("active"):
        return status
    if status.get("trial_available"):
        data = _load_access()
        now = now_taipei()
        data[user_id] = {
            **(data.get(user_id) or {}),
            "used_trial": True,
            "plan": "trial",
            "trial_started_at": _iso(now),
            "trial_expires_at": _iso(now + timedelta(minutes=TRIAL_MINUTES)),
        }
        _save_access(data)
        return access_status(user_id)
    raise PermissionError("expired")


def activate_user(user_id: str, code: str) -> Dict[str, Any]:
    code = str(code or "").strip()
    if code not in ALL_ACCESS_CODES:
        raise ValueError("開通碼錯誤")
    data = _load_access()
    rec = dict(data.get(user_id) or {})
    now = now_taipei()
    if code in PERMANENT_CODES:
        rec.update({"permanent": True, "plan": "permanent", "used_trial": True, "access_expires_at": ""})
    elif code in MONTHLY_CODES:
        rec.update({"permanent": False, "plan": "monthly", "used_trial": True, "access_expires_at": _iso(now + timedelta(days=MONTHLY_DAYS))})
    else:
        rec.update({"permanent": False, "plan": "temporary", "used_trial": True, "access_expires_at": _iso(now + timedelta(minutes=TEMP_TRIAL_MINUTES))})
    rec["updated_at"] = _iso(now)
    data[user_id] = rec
    _save_access(data)
    return access_status(user_id)


def verify_signature(body: bytes, signature: Optional[str]) -> bool:
    if not CHANNEL_SECRET:
        return True
    if not signature:
        return False
    expected = base64.b64encode(
        hmac.new(CHANNEL_SECRET.encode(), body, hashlib.sha256).digest()
    ).decode()
    return hmac.compare_digest(expected, signature)


def line_reply(reply_token: str, messages: List[Dict[str, Any]]) -> None:
    if not CHANNEL_ACCESS_TOKEN:
        print(json.dumps(messages, ensure_ascii=False))
        return
    response = requests.post(
        "https://api.line.me/v2/bot/message/reply",
        headers={"Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}", "Content-Type": "application/json"},
        json={"replyToken": reply_token, "messages": messages[:5]},
        timeout=6,
    )
    if response.status_code >= 300:
        print("LINE reply failed", response.status_code, response.text)


def text_message(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": str(text)[:5000]}


def postback(label: str, action: str, **kwargs: str) -> Dict[str, Any]:
    data = {"action": action, **{k: str(v) for k, v in kwargs.items()}}
    return {
        "type": "button",
        "style": "primary",
        "height": "sm",
        "action": {
            "type": "postback",
            "label": label[:20],
            "data": urllib.parse.urlencode(data),
        },
    }


def expired_flex() -> Dict[str, Any]:
    return {
        "type": "flex",
        "altText": "試用已到期",
        "contents": {
            "type": "bubble",
            "body": {
                "type": "box",
                "layout": "vertical",
                "backgroundColor": "#111111",
                "paddingAll": "18px",
                "contents": [
                    {"type": "text", "text": "試用已到期", "weight": "bold", "size": "xl", "color": "#FFD000"},
                    {"type": "text", "text": "請聯繫管理員取得開通碼，或在聊天室直接輸入開通碼。", "wrap": True, "margin": "md", "color": "#FFFFFF"},
                    {
                        "type": "button",
                        "style": "primary",
                        "color": "#06C755",
                        "margin": "lg",
                        "action": {"type": "uri", "label": "聯繫官方 LINE 管理員", "uri": ADMIN_LINE_URL},
                    },
                ],
            },
        },
    }


def prediction_flex(session: Dict[str, Any], notice: str = "") -> Dict[str, Any]:
    pred = session.get("last_prediction") or {}
    history = session.get("history") or []
    stats = session.get("stats") or {}
    contents: List[Dict[str, Any]] = [
        {"type": "text", "text": "下一局機率模擬", "weight": "bold", "size": "xl", "color": "#FFD000"},
        {"type": "text", "text": f"第 {len(history) + 1} 局｜資料庫條件 Monte Carlo", "size": "xs", "color": "#BBBBBB", "margin": "sm"},
    ]
    if notice:
        contents.append({"type": "text", "text": notice, "size": "sm", "color": "#FFFFFF", "wrap": True, "margin": "md"})
    contents += [
        {"type": "separator", "margin": "lg", "color": "#555555"},
        {"type": "text", "text": f"莊　{pred.get('banker_rate', 0):.1f}%", "size": "lg", "weight": "bold", "color": "#E74C3C", "margin": "lg"},
        {"type": "text", "text": f"閒　{pred.get('player_rate', 0):.1f}%", "size": "lg", "weight": "bold", "color": "#3498DB", "margin": "md"},
        {"type": "text", "text": f"和　{pred.get('tie_rate', 0):.1f}%", "size": "lg", "weight": "bold", "color": "#2ECC71", "margin": "md"},
        {"type": "text", "text": f"建議：{pred.get('recommend_text', '-')}", "size": "xl", "weight": "bold", "color": "#FFD000", "margin": "lg"},
        {"type": "text", "text": f"信號：{pred.get('signal_level', '-')}", "size": "sm", "color": "#FFFFFF", "margin": "sm"},
        {"type": "text", "text": f"目前紀錄：{' '.join(history[-20:]) or '尚無'}", "size": "xs", "color": "#BBBBBB", "wrap": True, "margin": "lg"},
        {"type": "text", "text": f"戰績：{stats.get('wins',0)}勝 {stats.get('losses',0)}敗｜最高連勝 {stats.get('max_win_streak',0)}", "size": "xs", "color": "#BBBBBB", "wrap": True, "margin": "sm"},
        {
            "type": "box",
            "layout": "horizontal",
            "spacing": "sm",
            "margin": "lg",
            "contents": [
                postback("開莊", "round", result="B"),
                postback("開閒", "round", result="P"),
                postback("開和", "round", result="T"),
            ],
        },
        {
            "type": "box",
            "layout": "horizontal",
            "spacing": "sm",
            "margin": "md",
            "contents": [
                postback("新靴", "new_shoe"),
                postback("上一步", "undo"),
            ],
        },
    ]
    return {
        "type": "flex",
        "altText": f"下一局建議：{pred.get('recommend_text', '-')}",
        "contents": {
            "type": "bubble",
            "size": "mega",
            "body": {"type": "box", "layout": "vertical", "backgroundColor": "#111111", "paddingAll": "18px", "contents": contents},
        },
    }


def _predict_session(user_id: str, session: Dict[str, Any]) -> Dict[str, Any]:
    pred = predict(
        session.get("history") or [],
        venue=session.get("venue", ""),
        room=session.get("room", ""),
        shoe_id=session.get("shoe_id", ""),
        user_id=user_id,
    )
    session["last_prediction"] = pred
    session["status"] = "可回報結果"
    return store.upsert_session(user_id, session)


def _start_or_predict(user_id: str) -> Dict[str, Any]:
    ensure_access(user_id)
    session = store.get_session(user_id) or store.new_session(user_id)
    return _predict_session(user_id, session)


@app.api_route("/", methods=["GET", "HEAD"])
def root() -> PlainTextResponse:
    return PlainTextResponse("OK")




@app.get("/liff")
async def liff_compatibility() -> PlainTextResponse:
    """Compatibility route for old LIFF/rich-menu links.

    The V3 bot no longer uses a web panel. Users should return to LINE and
    send「開始分析」to receive the Flex prediction panel in chat.
    """
    return PlainTextResponse(
        "此版本已改為直接在 LINE 聊天室顯示預測面板。請返回 LINE 並輸入「開始分析」。"
    )


@app.get("/favicon.ico")
async def favicon() -> PlainTextResponse:
    return PlainTextResponse("", status_code=204)


@app.api_route("/health", methods=["GET", "HEAD"])
def health() -> JSONResponse:
    return JSONResponse({"ok": True, "version": "3.0.0", "time": _iso(now_taipei())})


@app.post("/webhook")
async def webhook(request: Request) -> JSONResponse:
    body = await request.body()
    if not verify_signature(body, request.headers.get("X-Line-Signature")):
        return JSONResponse({"ok": False, "error": "bad signature"}, status_code=401)
    payload = json.loads(body.decode("utf-8") or "{}")
    for event in payload.get("events", []):
        reply_token = event.get("replyToken", "")
        source = event.get("source") or {}
        user_id = source.get("userId") or source.get("groupId") or source.get("roomId") or "anonymous"
        event_type = event.get("type")

        try:
            if event_type == "follow":
                line_reply(reply_token, [text_message("輸入「開始」即可使用。試用到期後可直接在聊天室輸入開通碼。")])
                continue

            if event_type == "message" and (event.get("message") or {}).get("type") == "text":
                text = str((event.get("message") or {}).get("text") or "").strip()
                if text in ALL_ACCESS_CODES:
                    status = activate_user(user_id, text)
                    session = _start_or_predict(user_id)
                    line_reply(reply_token, [text_message(f"✅ 開通成功：{status.get('label')}"), prediction_flex(session)])
                    continue
                if text.lower() in {"開始", "start", "ai", "預測", "開始分析"}:
                    try:
                        session = _start_or_predict(user_id)
                        line_reply(reply_token, [prediction_flex(session)])
                    except PermissionError:
                        line_reply(reply_token, [expired_flex()])
                    continue
                mapping = {"莊": "B", "庄": "B", "b": "B", "閒": "P", "闲": "P", "p": "P", "和": "T", "t": "T"}
                code = mapping.get(text) or mapping.get(text.lower())
                if code:
                    try:
                        ensure_access(user_id)
                        session = store.add_round(user_id, code)
                        session = _predict_session(user_id, session)
                        line_reply(reply_token, [prediction_flex(session, f"已回報：{code}")])
                    except PermissionError:
                        line_reply(reply_token, [expired_flex()])
                    continue
                line_reply(reply_token, [text_message("請輸入「開始」，或使用面板中的開莊／開閒／開和按鈕。")])
                continue

            if event_type == "postback":
                data = {
                    k: v[0]
                    for k, v in urllib.parse.parse_qs((event.get("postback") or {}).get("data", "")).items()
                }
                action = data.get("action")
                if action == "round":
                    try:
                        ensure_access(user_id)
                        result = data.get("result", "")
                        session = store.add_round(user_id, result)
                        session = _predict_session(user_id, session)
                        line_reply(reply_token, [prediction_flex(session, f"已回報：{result}")])
                    except PermissionError:
                        line_reply(reply_token, [expired_flex()])
                elif action == "new_shoe":
                    try:
                        ensure_access(user_id)
                        session = store.clear_history(user_id)
                        session = _predict_session(user_id, session)
                        line_reply(reply_token, [prediction_flex(session, "已建立新靴")])
                    except PermissionError:
                        line_reply(reply_token, [expired_flex()])
                elif action == "undo":
                    try:
                        ensure_access(user_id)
                        session = store.undo_round(user_id)
                        session = _predict_session(user_id, session)
                        line_reply(reply_token, [prediction_flex(session, "已刪除上一局")])
                    except PermissionError:
                        line_reply(reply_token, [expired_flex()])
                else:
                    session = _start_or_predict(user_id)
                    line_reply(reply_token, [prediction_flex(session)])
        except Exception as exc:
            line_reply(reply_token, [text_message(f"操作失敗：{exc}")])
    return JSONResponse({"ok": True})

@app.post("/callback")
async def callback_compatibility(request: Request) -> JSONResponse:
    """Compatibility alias for LINE Developers projects still using /callback."""
    return await webhook(request)
