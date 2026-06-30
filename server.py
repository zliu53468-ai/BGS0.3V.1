import base64
import hashlib
import hmac
import json
import os
import traceback
import urllib.parse
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import store
from predictor import predict

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "").strip()
LIFF_ID = os.getenv("LIFF_ID", "").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")


# ===== Access control / trial settings =====
TAIPEI_TZ = timezone(timedelta(hours=8))
ADMIN_LINE_ID = "@jins888"
ADMIN_LINE_URL = "https://line.me/R/ti/p/%40jins888"
ACCESS_DATA_FILE = Path(os.getenv("ACCESS_DATA_FILE", str(BASE_DIR / "data" / "access_control.json")))
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "30") or "30")
TEMP_TRIAL_MINUTES = int(os.getenv("TEMP_TRIAL_MINUTES", "30") or "30")
MONTHLY_DAYS = int(os.getenv("MONTHLY_DAYS", "30") or "30")
ACCESS_REDIRECT_SECONDS = int(os.getenv("ACCESS_REDIRECT_SECONDS", "30") or "30")

# Hard-coded activation codes requested by admin
PERMANENT_CODES = {"aaa1688003", "aaa1888007", "aaa1000889"}
MONTHLY_CODES = {"aaa13002", "aaa15001", "aaa199801"}
TEMP_TRIAL_CODES = {"aaaa1999152", "aaa345556", "aaa987743"}
ALL_ACCESS_CODES = PERMANENT_CODES | MONTHLY_CODES | TEMP_TRIAL_CODES

DEFAULT_VENUES = "OB:歐博真人,DG:DG真人,MT:MT真人,T9:T9真人,SA:SA真人,DB:DB真人"
VENUES_RAW = os.getenv("VENUES", DEFAULT_VENUES)
DEFAULT_ROOMS = os.getenv(
    "DEFAULT_ROOMS",
    "百家樂-中文廳,百家樂-亞洲廳,百家樂-極速廳,百家樂-保險廳,百家樂-VIP廳",
)

# panel  = 每次點莊/閒/和後，回覆新版面板，會立即看到「目前紀錄」
# silent = 只背景記錄，不洗版；需要按「查看紀錄」才更新面板
# compact = 每次只回覆一則很短的文字確認
ROUND_INPUT_REPLY_MODE = os.getenv("ROUND_INPUT_REPLY_MODE", "panel").strip().lower()
ACK_EVERY_N_ROUNDS = int(os.getenv("ACK_EVERY_N_ROUNDS", "0") or "0")

# 防止「開始AI判斷」因 DeepSeek 或外部 API 等太久而看起來卡住。
# 超時會回本地簡易備援預測，predictor.py 本身仍保留使用。
PREDICT_TIMEOUT_SEC = float(os.getenv("PREDICT_TIMEOUT_SEC", "14"))
_PREDICT_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("PREDICT_WORKERS", "2") or "2"))

# 開始 AI 判斷時，先用 replyToken 回「處理中」，再用 Push 補送結果。
# 這樣可避免 predictor / DeepSeek 太慢時，LINE replyToken 過期造成使用者完全沒反應。
PREDICT_ASYNC_PUSH = os.getenv("PREDICT_ASYNC_PUSH", "0") == "1"
_PUSH_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("PUSH_WORKERS", "2") or "2"))

# 按鈕回饋：點擊 Postback 按鈕時顯示 LINE 官方 Loading Animation。
# 只支援一對一聊天室；群組/多人聊天室會自動略過。
BUTTON_FEEDBACK_LOADING = os.getenv("BUTTON_FEEDBACK_LOADING", "1") == "1"
BUTTON_FEEDBACK_SECONDS = int(os.getenv("BUTTON_FEEDBACK_SECONDS", "5") or "5")

app = FastAPI(title="Baccarat LINE Postback AI Bot", version="2.3.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class StartSessionIn(BaseModel):
    user_id: str
    venue: str = ""
    room: str = ""
    shoe_id: str = ""


class UserIn(BaseModel):
    user_id: str


class AddRoundIn(BaseModel):
    user_id: str
    result: str


class PredictIn(BaseModel):
    user_id: str


class ActivationIn(BaseModel):
    user_id: str
    code: str


def parse_venues() -> List[Dict[str, str]]:
    venues: List[Dict[str, str]] = []
    for item in VENUES_RAW.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            code, name = item.split(":", 1)
        else:
            code, name = item, item
        venues.append({"code": code.strip(), "name": name.strip()})

    # 固定補上 DB真人，避免 Render 後台 VENUES 環境變數漏填時頁面沒有顯示
    if not any((v.get("code") or "").upper() == "DB" for v in venues):
        venues.append({"code": "DB", "name": "DB真人"})

    return venues


def parse_rooms() -> List[str]:
    return [x.strip() for x in DEFAULT_ROOMS.split(",") if x.strip()]


def venue_name(venue_code: str) -> str:
    for v in parse_venues():
        if v["code"] == venue_code:
            return v["name"]
    return venue_code or "-"


def build_liff_url(venue_code: str = "") -> str:
    """
    產生前端頁面網址。
    優先使用 PUBLIC_BASE_URL，讓遊戲館點擊後直接開 Render 網頁。
    若未設定 PUBLIC_BASE_URL，才退回 LIFF_ID。
    """
    query = urllib.parse.urlencode({"venue": venue_code}) if venue_code else ""
    if PUBLIC_BASE_URL:
        url = f"{PUBLIC_BASE_URL}/liff"
        return f"{url}?{query}" if query else url
    if LIFF_ID:
        url = f"https://liff.line.me/{LIFF_ID}"
        return f"{url}?{query}" if query else url
    return f"/liff?{query}" if query else "/liff"




class AccessDenied(Exception):
    pass


def now_taipei() -> datetime:
    return datetime.now(TAIPEI_TZ)


def dt_to_iso(dt: Optional[datetime]) -> str:
    if not dt:
        return ""
    return dt.astimezone(TAIPEI_TZ).isoformat(timespec="seconds")


def parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=TAIPEI_TZ)
        return dt.astimezone(TAIPEI_TZ)
    except Exception:
        return None


def load_access_db() -> Dict[str, Any]:
    try:
        if not ACCESS_DATA_FILE.exists():
            return {}
        with ACCESS_DATA_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print("load_access_db failed:", repr(exc))
        return {}


def save_access_db(data: Dict[str, Any]) -> None:
    try:
        ACCESS_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = ACCESS_DATA_FILE.with_suffix(ACCESS_DATA_FILE.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(ACCESS_DATA_FILE)
    except Exception as exc:
        print("save_access_db failed:", repr(exc))


def get_access_record(user_id: str) -> Dict[str, Any]:
    db = load_access_db()
    rec = db.get(user_id) or {}
    return rec if isinstance(rec, dict) else {}


def save_access_record(user_id: str, record: Dict[str, Any]) -> Dict[str, Any]:
    db = load_access_db()
    record = dict(record or {})
    record["user_id"] = user_id
    record["updated_at"] = dt_to_iso(now_taipei())
    db[user_id] = record
    save_access_db(db)
    return record


def code_kind(code: str) -> str:
    c = (code or "").strip()
    if c in PERMANENT_CODES:
        return "permanent"
    if c in MONTHLY_CODES:
        return "monthly"
    if c in TEMP_TRIAL_CODES:
        return "temporary"
    return ""


def is_access_code(text: str) -> bool:
    return (text or "").strip() in ALL_ACCESS_CODES


def activation_message(access: Dict[str, Any]) -> str:
    plan = access.get("plan_label") or access.get("plan") or "使用權限"
    expires = access.get("expires_at_taipei") or "永久"
    if access.get("plan") == "permanent":
        return f"✅ 已開通：{plan}\nLINE UID：{access.get('user_id', '-')}\n使用期限：永久"
    return f"✅ 已開通：{plan}\nLINE UID：{access.get('user_id', '-')}\n到期時間：{expires}（台北時間）"


def access_status(user_id: str) -> Dict[str, Any]:
    now = now_taipei()
    rec = get_access_record(user_id)

    base = {
        "user_id": user_id,
        "now_taipei": dt_to_iso(now),
        "admin_line_id": ADMIN_LINE_ID,
        "admin_line_url": ADMIN_LINE_URL,
        "redirect_after_seconds": ACCESS_REDIRECT_SECONDS,
    }

    if not user_id:
        return {
            **base,
            "state": "no_uid",
            "active": False,
            "can_predict": False,
            "can_start_trial": False,
            "plan": "none",
            "plan_label": "未取得 LINE UID",
            "message": f"無法取得 LINE UID，請從官方 LINE {ADMIN_LINE_ID} 重新開啟。",
        }

    if rec.get("permanent"):
        return {
            **base,
            "state": "active",
            "active": True,
            "can_predict": True,
            "can_start_trial": False,
            "plan": "permanent",
            "plan_label": "買斷永久版",
            "expires_at_taipei": "",
            "message": "買斷永久版已開通。",
        }

    access_exp = parse_dt(rec.get("access_expires_at"))
    if access_exp and access_exp > now:
        plan = rec.get("plan") or "monthly"
        label = "月租方案" if plan == "monthly" else "臨時開通"
        return {
            **base,
            "state": "active",
            "active": True,
            "can_predict": True,
            "can_start_trial": False,
            "plan": plan,
            "plan_label": label,
            "expires_at_taipei": dt_to_iso(access_exp),
            "remaining_seconds": max(0, int((access_exp - now).total_seconds())),
            "message": f"{label}使用中，到期時間：{dt_to_iso(access_exp)}（台北時間）。",
        }

    trial_exp = parse_dt(rec.get("trial_expires_at"))
    if trial_exp and trial_exp > now:
        return {
            **base,
            "state": "active",
            "active": True,
            "can_predict": True,
            "can_start_trial": False,
            "plan": "trial",
            "plan_label": "30分鐘試用",
            "trial_started_at_taipei": rec.get("trial_started_at", ""),
            "expires_at_taipei": dt_to_iso(trial_exp),
            "remaining_seconds": max(0, int((trial_exp - now).total_seconds())),
            "message": f"30分鐘試用中，到期時間：{dt_to_iso(trial_exp)}（台北時間）。",
        }

    if rec.get("used_trial"):
        expired_at = rec.get("trial_expires_at") or rec.get("access_expires_at") or ""
        return {
            **base,
            "state": "expired",
            "active": False,
            "can_predict": False,
            "can_start_trial": False,
            "plan": rec.get("plan") or "expired",
            "plan_label": "試用 / 權限已到期",
            "expires_at_taipei": expired_at,
            "message": f"試用或使用權限已到期，請聯繫管理員官方 LINE：{ADMIN_LINE_ID}",
        }

    return {
        **base,
        "state": "trial_available",
        "active": False,
        "can_predict": True,
        "can_start_trial": True,
        "plan": "trial_available",
        "plan_label": "尚未開始試用",
        "message": f"尚未開始試用。第一次點擊開始AI判斷後，會以台北時間開始計算 {TRIAL_MINUTES} 分鐘試用。",
    }


def activate_user(user_id: str, code: str, source: str = "web") -> Dict[str, Any]:
    if not user_id:
        raise ValueError("無法取得 LINE UID，請從 LINE 官方帳號內重新開啟。")
    c = (code or "").strip()
    kind = code_kind(c)
    if not kind:
        raise ValueError(f"開通碼錯誤，請聯繫管理員官方 LINE：{ADMIN_LINE_ID}")

    now = now_taipei()
    rec = get_access_record(user_id)
    rec["last_code"] = c
    rec["last_code_source"] = source
    rec["last_code_at"] = dt_to_iso(now)

    if kind == "permanent":
        rec.update({
            "plan": "permanent",
            "permanent": True,
            "used_trial": True,
            "access_expires_at": "",
        })
    elif kind == "monthly":
        exp = now + timedelta(days=MONTHLY_DAYS)
        rec.update({
            "plan": "monthly",
            "permanent": False,
            "used_trial": True,
            "access_expires_at": dt_to_iso(exp),
        })
    elif kind == "temporary":
        exp = now + timedelta(minutes=TEMP_TRIAL_MINUTES)
        rec.update({
            "plan": "temporary",
            "permanent": False,
            "used_trial": True,
            "access_expires_at": dt_to_iso(exp),
        })

    save_access_record(user_id, rec)
    return access_status(user_id)


def ensure_access_or_start_trial(user_id: str) -> Dict[str, Any]:
    status = access_status(user_id)
    if status.get("active"):
        return status
    if status.get("can_start_trial"):
        now = now_taipei()
        exp = now + timedelta(minutes=TRIAL_MINUTES)
        rec = get_access_record(user_id)
        rec.update({
            "plan": "trial",
            "permanent": False,
            "used_trial": True,
            "trial_started_at": dt_to_iso(now),
            "trial_expires_at": dt_to_iso(exp),
        })
        save_access_record(user_id, rec)
        return access_status(user_id)
    raise AccessDenied(status.get("message") or f"使用權限已到期，請聯繫管理員官方 LINE：{ADMIN_LINE_ID}")


def verify_line_signature(body: bytes, signature: Optional[str]) -> bool:
    if not CHANNEL_SECRET:
        return True
    if not signature:
        return False
    digest = hmac.new(CHANNEL_SECRET.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, signature)


def line_reply(reply_token: str, messages: List[Dict[str, Any]]) -> None:
    if not CHANNEL_ACCESS_TOKEN:
        print("LINE_CHANNEL_ACCESS_TOKEN is empty; reply skipped.")
        return
    r = requests.post(
        "https://api.line.me/v2/bot/message/reply",
        headers={
            "Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}",
            "Content-Type": "application/json",
        },
        json={"replyToken": reply_token, "messages": messages[:5]},
        timeout=8,
    )
    if r.status_code >= 300:
        print("LINE reply failed", r.status_code, r.text)


def line_push(to: str, messages: List[Dict[str, Any]]) -> bool:
    if not CHANNEL_ACCESS_TOKEN:
        print("LINE_CHANNEL_ACCESS_TOKEN is empty; push skipped.")
        return False
    if not to:
        print("LINE push target is empty; push skipped.")
        return False
    r = requests.post(
        "https://api.line.me/v2/bot/message/push",
        headers={
            "Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}",
            "Content-Type": "application/json",
        },
        json={"to": to, "messages": messages[:5]},
        timeout=8,
    )
    if r.status_code >= 300:
        print("LINE push failed", r.status_code, r.text)
        return False
    return True


def get_loading_chat_id(event: Dict[str, Any]) -> str:
    """
    LINE loading animation only supports 1-on-1 user chats.
    If the event comes from a group/room, return empty string to avoid API errors.
    """
    source = event.get("source") or {}
    if source.get("type") == "user":
        return source.get("userId", "")
    return ""


def line_loading(chat_id: str, seconds: int = 5) -> None:
    """
    Show LINE official loading animation as button-click feedback.
    This does not send a chat message and will not wash the conversation.
    """
    if not BUTTON_FEEDBACK_LOADING:
        return
    if not CHANNEL_ACCESS_TOKEN or not chat_id:
        return

    loading_seconds = max(5, min(60, int(seconds or 5)))
    try:
        r = requests.post(
            "https://api.line.me/v2/bot/chat/loading/start",
            headers={
                "Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}",
                "Content-Type": "application/json",
            },
            json={"chatId": chat_id, "loadingSeconds": loading_seconds},
            timeout=5,
        )
        if r.status_code >= 300:
            print("LINE loading failed", r.status_code, r.text)
    except Exception as exc:
        print("LINE loading exception", repr(exc))


def text_msg(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text[:5000]}


def postback_action(label: str, data: Dict[str, str]) -> Dict[str, Any]:
    # 不放 displayText，避免使用者每點一次按鈕，聊天室就多一則使用者文字。
    return {
        "type": "postback",
        "label": label[:20],
        "data": urllib.parse.urlencode(data),
    }


def button(
    label: str,
    data: Dict[str, str],
    color: str = "#FFD000",
    style: str = "primary",
) -> Dict[str, Any]:
    return {
        "type": "button",
        "style": style,
        "color": color,
        "height": "sm",
        "action": postback_action(label, data),
    }


def uri_button(
    label: str,
    uri: str,
    color: str = "#FFD000",
    style: str = "primary",
) -> Dict[str, Any]:
    return {
        "type": "button",
        "style": style,
        "color": color,
        "height": "sm",
        "action": {
            "type": "uri",
            "label": label[:20],
            "uri": uri[:1000],
        },
    }


def venue_web_button(v: Dict[str, str]) -> Dict[str, Any]:
    url = build_liff_url(v.get("code", ""))
    if url.startswith("http://") or url.startswith("https://"):
        return uri_button(v.get("name", ""), url, "#FFD000")
    return button(v.get("name", ""), {"action": "select_venue", "venue": v.get("code", "")}, "#FFD000")


def get_source_user_id(event: Dict[str, Any]) -> str:
    source = event.get("source") or {}
    return source.get("userId") or source.get("groupId") or source.get("roomId") or "anonymous"


def get_session_or_create(user_id: str) -> Dict[str, Any]:
    session = store.get_session(user_id)
    if not session:
        session = store.upsert_session(user_id, {})
    return session


def result_name(code: str) -> str:
    return {"B": "莊", "P": "閒", "T": "和"}.get(str(code).upper(), str(code))


def result_chip_text(history: List[str], limit: int = 32) -> str:
    if not history:
        return "尚未輸入"
    display = [result_name(x) for x in history[-limit:]]
    text = " ".join(display)
    if len(history) > limit:
        text = "… " + text
    return text


def compact_history(history: List[str], limit: int = 18) -> str:
    if not history:
        return "尚未輸入"
    display = [str(x).upper() for x in history[-limit:]]
    text = " ".join(display)
    if len(history) > limit:
        text = "… " + text
    return text


def percent_text(value: Any) -> str:
    try:
        v = float(value)
        if v <= 1:
            v *= 100
        return f"{v:.0f}%"
    except Exception:
        return f"{value}%" if value not in [None, ""] else "--"


def kv(label: str, value: Any) -> Dict[str, Any]:
    return {
        "type": "box",
        "layout": "horizontal",
        "margin": "md",
        "contents": [
            {"type": "text", "text": label, "size": "sm", "color": "#333333", "flex": 2},
            {"type": "text", "text": str(value), "size": "sm", "color": "#333333", "align": "end", "flex": 4, "wrap": True},
        ],
    }


def rate_line(label: str, value: str, color: str) -> Dict[str, Any]:
    return {
        "type": "box",
        "layout": "horizontal",
        "margin": "md",
        "contents": [
            {"type": "text", "text": label, "size": "md", "weight": "bold", "color": color, "flex": 1},
            {"type": "text", "text": value, "size": "sm", "color": "#333333", "align": "end", "flex": 3},
        ],
    }


def start_menu_flex(title: str = "AI 百家樂規律分析", subtitle: str = "點擊下方按鈕開始選擇遊戲館。") -> Dict[str, Any]:
    return {
        "type": "flex",
        "altText": "開始分析",
        "contents": {
            "type": "bubble",
            "size": "mega",
            "body": {
                "type": "box",
                "layout": "vertical",
                "backgroundColor": "#111111",
                "paddingAll": "18px",
                "contents": [
                    {"type": "text", "text": title, "weight": "bold", "size": "xl", "color": "#FFD000", "wrap": True},
                    {"type": "text", "text": subtitle, "size": "sm", "color": "#FFFFFF", "margin": "md", "wrap": True},
                    {"type": "separator", "margin": "lg", "color": "#FFD000"},
                    {
                        "type": "box",
                        "layout": "vertical",
                        "spacing": "md",
                        "margin": "lg",
                        "contents": [
                            button("開始分析", {"action": "open_venue"}, "#FFD000"),
                        ],
                    },
                    {"type": "text", "text": "點擊遊戲館後會開啟網頁操作面板。", "size": "xs", "color": "#AAAAAA", "margin": "lg", "wrap": True},
                ],
            },
        },
    }


def venue_flex() -> Dict[str, Any]:
    buttons = [venue_web_button(v) for v in parse_venues()]
    return {
        "type": "flex",
        "altText": "請選擇遊戲館",
        "contents": {
            "type": "bubble",
            "size": "mega",
            "body": {
                "type": "box",
                "layout": "vertical",
                "backgroundColor": "#111111",
                "paddingAll": "18px",
                "contents": [
                    {"type": "text", "text": "AI 規律模型", "weight": "bold", "size": "xl", "color": "#FFD000"},
                    {"type": "text", "text": "請選擇遊戲館，點擊後會開啟網頁操作面板。", "size": "sm", "color": "#FFFFFF", "margin": "md", "wrap": True},
                    {"type": "separator", "margin": "lg", "color": "#FFD000"},
                    {"type": "box", "layout": "vertical", "spacing": "md", "margin": "lg", "contents": buttons},
                ],
            },
        },
    }


def room_flex(venue_code: str) -> Dict[str, Any]:
    buttons = [button(room, {"action": "select_room", "venue": venue_code, "room": room}) for room in parse_rooms()]
    return {
        "type": "flex",
        "altText": "請選擇遊戲廳",
        "contents": {
            "type": "bubble",
            "size": "mega",
            "body": {
                "type": "box",
                "layout": "vertical",
                "backgroundColor": "#111111",
                "paddingAll": "18px",
                "contents": [
                    {"type": "text", "text": venue_name(venue_code), "weight": "bold", "size": "xl", "color": "#FFD000", "wrap": True},
                    {"type": "text", "text": "請選擇遊戲廳，選完後即可輸入莊 / 閒 / 和。", "size": "sm", "color": "#FFFFFF", "margin": "md", "wrap": True},
                    {"type": "separator", "margin": "lg", "color": "#FFD000"},
                    {"type": "box", "layout": "vertical", "spacing": "md", "margin": "lg", "contents": buttons},
                    {"type": "separator", "margin": "lg", "color": "#333333"},
                    {"type": "button", "style": "secondary", "height": "sm", "margin": "md", "action": postback_action("重新選館", {"action": "open_venue"})},
                ],
            },
        },
    }


def input_panel_flex(session: Dict[str, Any], notice: str = "") -> Dict[str, Any]:
    history = session.get("history", []) or []
    venue = session.get("venue", "")
    room = session.get("room", "")
    shoe_id = session.get("shoe_id", "") or "可直接輸入靴號"
    round_no = len(history) + 1

    contents: List[Dict[str, Any]] = [
        {"type": "text", "text": "AI 規律分析", "weight": "bold", "size": "xl", "color": "#111111"},
        {"type": "separator", "margin": "md", "color": "#FFD000"},
        kv("遊戲館", venue_name(venue)),
        kv("遊戲廳", room or "-"),
        kv("靴號", shoe_id),
        kv("目前局數", f"第 {round_no} 局"),
    ]

    if notice:
        contents.append({
            "type": "box",
            "layout": "vertical",
            "backgroundColor": "#FFF6CC",
            "cornerRadius": "md",
            "paddingAll": "8px",
            "margin": "md",
            "contents": [{"type": "text", "text": notice, "size": "xs", "color": "#333333", "wrap": True}],
        })

    contents.extend([
        {"type": "text", "text": f"目前紀錄｜已輸入 {len(history)} 局", "size": "sm", "color": "#111111", "weight": "bold", "margin": "lg"},
        {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": "#F7F7F7",
            "cornerRadius": "md",
            "paddingAll": "10px",
            "margin": "sm",
            "contents": [
                {"type": "text", "text": result_chip_text(history), "size": "sm", "wrap": True, "color": "#333333"},
                {"type": "text", "text": compact_history(history), "size": "xs", "wrap": True, "color": "#888888", "margin": "sm"},
            ],
        },
        {"type": "text", "text": "輸入莊 / 閒 / 和", "size": "sm", "color": "#111111", "weight": "bold", "margin": "lg"},
        {
            "type": "box",
            "layout": "horizontal",
            "spacing": "sm",
            "margin": "sm",
            "contents": [
                button("莊 B", {"action": "add_round", "result": "B"}, "#E60012"),
                button("閒 P", {"action": "add_round", "result": "P"}, "#0B46D9"),
                button("和 T", {"action": "add_round", "result": "T"}, "#00A040"),
            ],
        },
        {
            "type": "box",
            "layout": "horizontal",
            "spacing": "sm",
            "margin": "sm",
            "contents": [
                button("上一步", {"action": "undo_round"}, "#222222"),
                button("查看紀錄", {"action": "view_panel"}, "#222222"),
            ],
        },
        {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "margin": "md",
            "contents": [
                button("開始AI判斷", {"action": "predict"}, "#FFD000"),
                button("清除本靴", {"action": "reset_session"}, "#555555"),
                button("結束分析", {"action": "end_session"}, "#111111"),
            ],
        },
        {"type": "text", "text": "提示：LINE 不能修改已送出的舊 Flex 訊息。新版面板會以新訊息顯示最新紀錄。", "size": "xs", "color": "#888888", "margin": "lg", "wrap": True},
    ])

    return {
        "type": "flex",
        "altText": "莊閒和輸入面板",
        "contents": {
            "type": "bubble",
            "size": "mega",
            "body": {
                "type": "box",
                "layout": "vertical",
                "backgroundColor": "#FFFFFF",
                "paddingAll": "16px",
                "contents": contents,
            },
        },
    }


def prediction_section(pred: Dict[str, Any]) -> List[Dict[str, Any]]:
    recommend = pred.get("recommend_text") or pred.get("recommend") or "-"
    signal = pred.get("signal_level", "")
    reason = pred.get("reason", "")
    pattern = pred.get("pattern_label", "")
    ai_used = "DeepSeek 校準" if pred.get("ai_used") else "本地規律"

    return [
        {"type": "separator", "margin": "lg", "color": "#FFD000"},
        {"type": "text", "text": "分析數據", "size": "sm", "weight": "bold", "color": "#111111", "margin": "lg"},
        rate_line("莊", percent_text(pred.get("banker_rate", 0)), "#E60012"),
        rate_line("閒", percent_text(pred.get("player_rate", 0)), "#0000CC"),
        rate_line("和", percent_text(pred.get("tie_rate", 0)), "#00A000"),
        {
            "type": "box",
            "layout": "horizontal",
            "margin": "lg",
            "cornerRadius": "md",
            "backgroundColor": "#FFD000",
            "paddingAll": "10px",
            "contents": [
                {"type": "text", "text": "推薦", "weight": "bold", "color": "#111111", "flex": 1},
                {"type": "text", "text": str(recommend), "weight": "bold", "align": "end", "color": "#111111", "flex": 2},
            ],
        },
        {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": "#F7F7F7",
            "cornerRadius": "md",
            "paddingAll": "8px",
            "margin": "sm",
            "contents": [
                {"type": "text", "text": f"{signal}｜{ai_used}", "size": "xs", "color": "#333333", "wrap": True},
                {"type": "text", "text": pattern or reason or "規律模型已完成判斷", "size": "xs", "color": "#777777", "wrap": True, "margin": "xs"},
                {"type": "text", "text": reason, "size": "xxs", "color": "#999999", "wrap": True, "margin": "xs"} if reason else {"type": "text", "text": "", "size": "xxs", "color": "#999999"},
            ],
        },
    ]


def result_panel_flex(session: Dict[str, Any], notice: str = "AI 判斷完成") -> Dict[str, Any]:
    """
    結果直接整合到原本輸入面板的版型中。
    LINE 無法修改已送出的舊 Flex，只能送一張新版面板顯示最新紀錄與預測結果。
    """
    history = session.get("history", []) or []
    venue = session.get("venue", "")
    room = session.get("room", "")
    shoe_id = session.get("shoe_id", "") or "可直接輸入靴號"
    round_no = len(history) + 1
    pred = session.get("last_prediction") or {}

    contents: List[Dict[str, Any]] = [
        {"type": "text", "text": "AI 規律分析", "weight": "bold", "size": "xl", "color": "#111111"},
        {"type": "separator", "margin": "md", "color": "#FFD000"},
        kv("遊戲館", venue_name(venue)),
        kv("遊戲廳", room or "-"),
        kv("靴號", shoe_id),
        kv("目前局數", f"第 {round_no} 局"),
    ]

    if notice:
        contents.append({
            "type": "box",
            "layout": "vertical",
            "backgroundColor": "#FFF6CC",
            "cornerRadius": "md",
            "paddingAll": "8px",
            "margin": "md",
            "contents": [{"type": "text", "text": notice, "size": "xs", "color": "#333333", "wrap": True}],
        })

    contents.extend([
        {"type": "text", "text": f"目前紀錄｜已輸入 {len(history)} 局", "size": "sm", "color": "#111111", "weight": "bold", "margin": "lg"},
        {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": "#F7F7F7",
            "cornerRadius": "md",
            "paddingAll": "10px",
            "margin": "sm",
            "contents": [
                {"type": "text", "text": result_chip_text(history), "size": "sm", "wrap": True, "color": "#333333"},
                {"type": "text", "text": compact_history(history), "size": "xs", "wrap": True, "color": "#888888", "margin": "sm"},
            ],
        },
    ])

    if pred:
        contents.extend(prediction_section(pred))

    contents.extend([
        {"type": "text", "text": "輸入莊 / 閒 / 和", "size": "sm", "color": "#111111", "weight": "bold", "margin": "lg"},
        {
            "type": "box",
            "layout": "horizontal",
            "spacing": "sm",
            "margin": "sm",
            "contents": [
                button("莊 B", {"action": "add_round", "result": "B"}, "#E60012"),
                button("閒 P", {"action": "add_round", "result": "P"}, "#0B46D9"),
                button("和 T", {"action": "add_round", "result": "T"}, "#00A040"),
            ],
        },
        {
            "type": "box",
            "layout": "horizontal",
            "spacing": "sm",
            "margin": "sm",
            "contents": [
                button("上一步", {"action": "undo_round"}, "#222222"),
                button("查看紀錄", {"action": "view_panel"}, "#222222"),
            ],
        },
        {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "margin": "md",
            "contents": [
                button("繼續分析", {"action": "predict"}, "#FFD000"),
                button("清除本靴", {"action": "reset_session"}, "#555555"),
                button("結束分析", {"action": "end_session"}, "#111111"),
            ],
        },
        {"type": "text", "text": "提示：LINE 不能修改舊面板；系統會送出新版面板，直接包含最新預測結果。", "size": "xs", "color": "#888888", "margin": "lg", "wrap": True},
    ])

    return {
        "type": "flex",
        "altText": f"分析結果：推薦 {pred.get('recommend_text') or pred.get('recommend') or '-'}",
        "contents": {
            "type": "bubble",
            "size": "mega",
            "body": {
                "type": "box",
                "layout": "vertical",
                "backgroundColor": "#FFFFFF",
                "paddingAll": "16px",
                "contents": contents,
            },
        },
    }


def result_flex(session: Dict[str, Any]) -> Dict[str, Any]:
    return result_panel_flex(session, "AI 判斷完成，結果已更新在面板內")

def end_flex(session: Dict[str, Any]) -> Dict[str, Any]:
    total = len(session.get("history", []) or [])
    return start_menu_flex(
        title="本靴分析已結束",
        subtitle=f"總局數：{total} 局。需要下一靴時，點下方「開始分析」。",
    )


def _fallback_predict(history: List[str], venue: str = "", room: str = "", shoe_id: str = "", reason: str = "本地備援") -> Dict[str, Any]:
    h = [x for x in history if str(x).upper() in {"B", "P", "T"}]
    bp = [str(x).upper() for x in h if str(x).upper() in {"B", "P"}]
    if not bp:
        b, p, t = 45.9, 44.6, 9.5
        pick = "B"
        pattern = "冷啟動"
    else:
        last = bp[-1]
        streak = 1
        for x in reversed(bp[:-1]):
            if x == last:
                streak += 1
            else:
                break
        switches = sum(1 for a, b0 in zip(bp[-10:], bp[-9:]) if a != b0)
        switch_rate = switches / max(1, min(len(bp[-10:]) - 1, 9))
        if streak >= 3:
            pick = last
            pattern = f"{'莊' if last == 'B' else '閒'}{streak}連，備援偏續龍"
        elif switch_rate >= 0.65:
            pick = "P" if last == "B" else "B"
            pattern = "跳路備援"
        else:
            pick = "B" if bp.count("B") <= bp.count("P") else "P"
            pattern = "均衡備援"
        edge = 3.0 + min(4.0, streak * 0.7)
        if pick == "B":
            b, p = 45.9 + edge, 44.6 - edge
        else:
            b, p = 45.9 - edge, 44.6 + edge
        t = 9.5
        s = b + p + t
        b, p, t = b / s * 100, p / s * 100, t / s * 100

    return {
        "ok": True,
        "venue": venue,
        "room": room,
        "shoe_id": shoe_id,
        "round_no": len(h) + 1,
        "history_len": len(h),
        "banker_rate": round(b, 1),
        "player_rate": round(p, 1),
        "tie_rate": round(t, 1),
        "recommend": pick,
        "recommend_text": {"B": "莊", "P": "閒"}[pick],
        "confidence": 0.36,
        "signal_level": "備援訊號",
        "pattern_label": pattern,
        "reason": f"{pattern} / {reason}",
        "ai_used": False,
        "ai_result": None,
    }


def predict_and_save(user_id: str) -> Tuple[Dict[str, Any], bool, str]:
    session = store.get_session(user_id)
    if not session:
        raise ValueError("請先輸入「開始分析」並選擇遊戲館。")

    history = session.get("history", []) or []
    venue = session.get("venue", "")
    room = session.get("room", "")
    shoe_id = session.get("shoe_id", "")

    if len(history) < 1:
        raise ValueError("目前尚未輸入莊閒和紀錄，請至少輸入 1 局再開始判斷。")

    # 第一次真正開始 AI 判斷時，才啟動 30 分鐘試用；已到期則阻擋預測。
    ensure_access_or_start_trial(user_id)

    try:
        future = _PREDICT_EXECUTOR.submit(
            predict,
            history=history,
            venue=venue,
            room=room,
            shoe_id=shoe_id,
            user_id=user_id,
        )
        pred = future.result(timeout=PREDICT_TIMEOUT_SEC)
        used_fallback = False
        note = ""
    except TimeoutError:
        pred = _fallback_predict(history, venue, room, shoe_id, reason=f"AI判斷超過 {PREDICT_TIMEOUT_SEC:.0f} 秒，已先回本地備援")
        used_fallback = True
        note = "AI判斷逾時，已先回本地備援。"
    except Exception as exc:
        print("predict failed:", repr(exc))
        traceback.print_exc()
        pred = _fallback_predict(history, venue, room, shoe_id, reason=f"predictor錯誤：{exc}")
        used_fallback = True
        note = f"predictor 錯誤，已回本地備援：{exc}"

    saved = store.upsert_session(user_id, {**session, "last_prediction": pred, "status": "可押注"})
    return saved, used_fallback, note


def push_predict_result(user_id: str) -> None:
    try:
        session, used_fallback, note = predict_and_save(user_id)
        msgs = [result_flex(session)]
        if used_fallback and note:
            msgs.append(text_msg(note))
        ok = line_push(user_id, msgs)
        if not ok:
            # Flex 格式或推播失敗時，至少補一則文字版結果，方便在 Render log 對照。
            pred = session.get("last_prediction") or {}
            line_push(user_id, [text_msg(
                f"AI判斷完成\n"
                f"莊：{percent_text(pred.get('banker_rate', 0))}\n"
                f"閒：{percent_text(pred.get('player_rate', 0))}\n"
                f"和：{percent_text(pred.get('tie_rate', 0))}\n"
                f"推薦：{pred.get('recommend_text') or pred.get('recommend') or '-'}\n"
                f"{pred.get('signal_level', '')}｜{pred.get('reason', '')}"
            )])
    except Exception as exc:
        print("push_predict_result failed:", repr(exc))
        traceback.print_exc()
        line_push(user_id, [text_msg(f"AI判斷失敗：{exc}")])


@app.get("/")
def root() -> Dict[str, Any]:
    return {"ok": True, "service": "baccarat-line-postback-ai-bot", "version": "2.3.1"}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "baccarat-line-postback-ai-bot", "version": "2.3.1"}


@app.get("/ping")
def ping() -> PlainTextResponse:
    return PlainTextResponse("pong")


@app.get("/liff")
def liff_page() -> Any:
    # 保留舊 LIFF 頁面相容；正式主流程為 LINE 聊天室 Postback。
    html_path = STATIC_DIR / "liff.html"
    if not html_path.exists():
        return JSONResponse({"ok": False, "detail": "static/liff.html not found. 主流程可直接用 LINE Postback，不需 LIFF。"}, status_code=404)
    return FileResponse(html_path)


@app.get("/api/config")
def api_config() -> Dict[str, Any]:
    return {
        "liffId": LIFF_ID,
        "venues": parse_venues(),
        "rooms": parse_rooms(),
        "publicBaseUrl": PUBLIC_BASE_URL,
        "adminLineId": ADMIN_LINE_ID,
        "adminLineUrl": ADMIN_LINE_URL,
        "accessRedirectSeconds": ACCESS_REDIRECT_SECONDS,
    }


@app.get("/api/access/status")
def api_access_status(user_id: str) -> Dict[str, Any]:
    return {"ok": True, "access": access_status(user_id)}


@app.post("/api/access/activate")
def api_access_activate(body: ActivationIn) -> Dict[str, Any]:
    try:
        access = activate_user(body.user_id, body.code, source="web")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "access": access}


@app.get("/api/session/current")
def api_current(user_id: str) -> Dict[str, Any]:
    session = store.get_session(user_id)
    if not session:
        session = store.upsert_session(user_id, {})
    return {"ok": True, "session": session}


@app.post("/api/session/start")
def api_start(body: StartSessionIn) -> Dict[str, Any]:
    session = store.new_session(body.user_id, body.venue, body.room, body.shoe_id)
    return {"ok": True, "session": session}


@app.post("/api/round/add")
def api_add_round(body: AddRoundIn) -> Dict[str, Any]:
    try:
        session = store.add_round(body.user_id, body.result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "session": session}


@app.post("/api/round/undo")
def api_undo(body: UserIn) -> Dict[str, Any]:
    session = store.undo_round(body.user_id)
    return {"ok": True, "session": session}


@app.post("/api/session/reset")
def api_reset(body: UserIn) -> Dict[str, Any]:
    session = store.clear_history(body.user_id)
    return {"ok": True, "session": session}


@app.post("/api/session/end")
def api_end(body: UserIn) -> Dict[str, Any]:
    session = store.end_session(body.user_id)
    return {"ok": True, "session": session}


@app.post("/api/predict")
def api_predict(body: PredictIn) -> Dict[str, Any]:
    try:
        session, used_fallback, note = predict_and_save(body.user_id)
    except AccessDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "ok": True,
        "session": session,
        "prediction": session.get("last_prediction"),
        "used_fallback": used_fallback,
        "note": note,
        "access": access_status(body.user_id),
    }


@app.post("/callback")
async def callback(request: Request) -> JSONResponse:
    body = await request.body()
    signature = request.headers.get("x-line-signature")
    if not verify_line_signature(body, signature):
        raise HTTPException(status_code=403, detail="invalid signature")

    payload = json.loads(body.decode("utf-8") or "{}")
    for event in payload.get("events", []):
        reply_token = event.get("replyToken")
        if not reply_token:
            continue

        event_type = event.get("type")
        user_id = get_source_user_id(event)

        if event_type == "follow":
            access = access_status(user_id)
            if access.get("state") == "expired":
                line_reply(reply_token, [text_msg(f"使用權限已到期，請聯繫管理員官方 LINE：{ADMIN_LINE_ID}")])
            else:
                line_reply(reply_token, [start_menu_flex("歡迎使用 AI 規律分析", "點擊「開始分析」選擇遊戲館。")])
            continue

        if event_type == "message" and event.get("message", {}).get("type") == "text":
            text = event.get("message", {}).get("text", "").strip()
            lower_text = text.lower()

            if is_access_code(text):
                try:
                    access = activate_user(user_id, text, source="line")
                    line_reply(reply_token, [text_msg(activation_message(access))])
                except Exception as exc:
                    line_reply(reply_token, [text_msg(str(exc))])
                continue

            if any(k in text for k in ["開始", "選館", "遊戲館", "重新選館"]):
                line_reply(reply_token, [venue_flex()])
                continue

            if text in ["紀錄", "查看紀錄", "面板", "輸入面板"]:
                session = get_session_or_create(user_id)
                line_reply(reply_token, [input_panel_flex(session)])
                continue

            if text in ["AI", "開始AI判斷", "判斷", "預測"]:
                if PREDICT_ASYNC_PUSH:
                    session = get_session_or_create(user_id)
                    line_reply(reply_token, [input_panel_flex(session, "AI 規律模型判斷中，請稍候 3～15 秒；結果會以新版面板跳出。")])
                    _PUSH_EXECUTOR.submit(push_predict_result, user_id)
                else:
                    try:
                        session, used_fallback, note = predict_and_save(user_id)
                        msgs = [result_flex(session)]
                        if used_fallback and note:
                            msgs.append(text_msg(note))
                        line_reply(reply_token, msgs)
                    except Exception as exc:
                        line_reply(reply_token, [text_msg(str(exc))])
                continue

            if text in ["結束", "結束分析"]:
                session = store.end_session(user_id)
                line_reply(reply_token, [end_flex(session)])
                continue

            mapping = {"莊": "B", "庄": "B", "b": "B", "B": "B", "閒": "P", "闲": "P", "p": "P", "P": "P", "和": "T", "t": "T", "T": "T"}
            if text in mapping or lower_text in mapping:
                result = mapping.get(text) or mapping.get(lower_text)
                try:
                    session = store.add_round(user_id, result)
                    line_reply(reply_token, [input_panel_flex(session, f"已新增：{result_name(result)}")])
                except Exception as exc:
                    line_reply(reply_token, [text_msg(f"輸入失敗：{exc}")])
                continue

            # 已選廳後，使用者輸入一般文字時，當作靴號 / 桌號備註。
            session = get_session_or_create(user_id)
            if session.get("venue") or session.get("room"):
                session = store.upsert_session(user_id, {**session, "shoe_id": text})
                line_reply(reply_token, [input_panel_flex(session, "已更新靴號 / 桌號備註")])
            else:
                line_reply(reply_token, [start_menu_flex("尚未開始分析", "請點擊「開始分析」選擇遊戲館。")])

        elif event_type == "postback":
            loading_chat_id = get_loading_chat_id(event)
            if loading_chat_id:
                line_loading(loading_chat_id, BUTTON_FEEDBACK_SECONDS)

            raw_data = event.get("postback", {}).get("data", "")
            data = {k: v[0] for k, v in urllib.parse.parse_qs(raw_data).items()}
            action = data.get("action", "")

            try:
                if action == "open_venue":
                    line_reply(reply_token, [venue_flex()])

                elif action == "select_venue":
                    venue = data.get("venue", "")
                    session = get_session_or_create(user_id)
                    store.upsert_session(user_id, {**session, "venue": venue, "status": "選擇遊戲廳"})
                    line_reply(reply_token, [room_flex(venue)])

                elif action == "select_room":
                    venue = data.get("venue", "")
                    room = data.get("room", "")
                    session = store.new_session(user_id, venue, room, "")
                    line_reply(reply_token, [input_panel_flex(session, "已建立新靴，請開始輸入莊 / 閒 / 和")])

                elif action == "view_panel":
                    session = get_session_or_create(user_id)
                    line_reply(reply_token, [input_panel_flex(session)])

                elif action == "add_round":
                    result = data.get("result", "")
                    session = store.add_round(user_id, result)
                    history_len = len(session.get("history", []) or [])
                    notice = f"已新增：{result_name(result)}｜目前 {history_len} 局"

                    should_reply = (
                        ROUND_INPUT_REPLY_MODE == "panel"
                        or ROUND_INPUT_REPLY_MODE == "compact"
                        or (ACK_EVERY_N_ROUNDS > 0 and history_len % ACK_EVERY_N_ROUNDS == 0)
                    )

                    if ROUND_INPUT_REPLY_MODE == "compact":
                        line_reply(reply_token, [text_msg(f"{notice}\n紀錄：{compact_history(session.get('history', []), 24)}")])
                    elif should_reply:
                        line_reply(reply_token, [input_panel_flex(session, notice)])
                    # silent 模式不回覆；需要更新時可按「查看紀錄」。

                elif action == "undo_round":
                    session = store.undo_round(user_id)
                    line_reply(reply_token, [input_panel_flex(session, "已刪除上一局")])

                elif action == "reset_session":
                    session = store.clear_history(user_id)
                    line_reply(reply_token, [input_panel_flex(session, "已清除本靴紀錄")])

                elif action == "end_session":
                    session = store.end_session(user_id)
                    line_reply(reply_token, [end_flex(session)])

                elif action == "predict":
                    if PREDICT_ASYNC_PUSH:
                        session = get_session_or_create(user_id)
                        line_reply(reply_token, [input_panel_flex(session, "AI 規律模型判斷中，請稍候 3～15 秒；結果會以新版面板跳出。")])
                        _PUSH_EXECUTOR.submit(push_predict_result, user_id)
                    else:
                        session, used_fallback, note = predict_and_save(user_id)
                        msgs = [result_flex(session)]
                        if used_fallback and note:
                            msgs.append(text_msg(note))
                        line_reply(reply_token, msgs)

                else:
                    line_reply(reply_token, [start_menu_flex("尚未選擇動作", "請點擊「開始分析」重新開始。")])

            except Exception as exc:
                print("postback failed:", repr(exc))
                traceback.print_exc()
                line_reply(reply_token, [text_msg(f"操作失敗：{exc}")])

    return JSONResponse({"ok": True})
