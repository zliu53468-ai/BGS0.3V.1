from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import traceback
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
ADMIN_LINE_URL = os.getenv(
    "ADMIN_LINE_URL",
    "https://line.me/R/ti/p/%40jins888",
)
ACCESS_DATA_FILE = Path(
    os.getenv(
        "ACCESS_DATA_FILE",
        str(BASE_DIR / "data" / "access_control.json"),
    )
)

TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "30") or "30")
TEMP_TRIAL_MINUTES = int(os.getenv("TEMP_TRIAL_MINUTES", "30") or "30")
MONTHLY_DAYS = int(os.getenv("MONTHLY_DAYS", "30") or "30")

DEFAULT_VENUES = (
    "OB:歐博真人,DG:DG真人,MT:MT真人,"
    "T9:T9真人,SA:SA真人,DB:DB真人"
)
VENUES_RAW = os.getenv("VENUES", DEFAULT_VENUES)

# 房間可直接輸入；這個環境變數只提供提示範例，不會限制使用者輸入。
ROOM_INPUT_EXAMPLES = os.getenv(
    "ROOM_INPUT_EXAMPLES",
    "RB01、百家樂1、中文廳、VIP廳",
)

PERMANENT_CODES = {"aaa1688003", "aaa1888007", "aaa1000889"}
MONTHLY_CODES = {"aaa13002", "aaa15001", "aaa199801"}
TEMP_TRIAL_CODES = {"aaaa1999152", "aaa345556", "aaa987743"}
ALL_ACCESS_CODES = PERMANENT_CODES | MONTHLY_CODES | TEMP_TRIAL_CODES

app = FastAPI(
    title="Baccarat LINE Chat Panel Bot",
    version="3.1.0",
)


class AccessDenied(Exception):
    pass


# ---------------------------------------------------------------------------
# Time / access helpers
# ---------------------------------------------------------------------------


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
        with ACCESS_DATA_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print("load_access_db failed:", repr(exc))
        return {}


def save_access_db(data: Dict[str, Any]) -> None:
    ACCESS_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp = ACCESS_DATA_FILE.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    temp.replace(ACCESS_DATA_FILE)


def get_access_record(user_id: str) -> Dict[str, Any]:
    record = load_access_db().get(user_id) or {}
    return record if isinstance(record, dict) else {}


def save_access_record(
    user_id: str,
    record: Dict[str, Any],
) -> Dict[str, Any]:
    data = load_access_db()
    saved = dict(record or {})
    saved["user_id"] = user_id
    saved["updated_at"] = dt_to_iso(now_taipei())
    data[user_id] = saved
    save_access_db(data)
    return saved


def access_status(user_id: str) -> Dict[str, Any]:
    now = now_taipei()
    record = get_access_record(user_id)

    base = {
        "user_id": user_id,
        "now_taipei": dt_to_iso(now),
        "admin_line_id": ADMIN_LINE_ID,
        "admin_line_url": ADMIN_LINE_URL,
    }

    if not user_id:
        return {
            **base,
            "state": "no_uid",
            "active": False,
            "can_predict": False,
            "can_start_trial": False,
            "plan": "none",
            "plan_label": "未取得 UID",
            "remaining_seconds": 0,
            "message": "無法取得 LINE UID。",
        }

    if record.get("permanent"):
        return {
            **base,
            "state": "active",
            "active": True,
            "can_predict": True,
            "can_start_trial": False,
            "plan": "permanent",
            "plan_label": "永久版",
            "remaining_seconds": None,
            "expires_at_taipei": "",
            "message": "永久版已開通。",
        }

    access_exp = parse_dt(record.get("access_expires_at"))
    if access_exp and access_exp > now:
        plan = record.get("plan") or "monthly"
        label = "月租方案" if plan == "monthly" else "臨時開通"
        return {
            **base,
            "state": "active",
            "active": True,
            "can_predict": True,
            "can_start_trial": False,
            "plan": plan,
            "plan_label": label,
            "remaining_seconds": max(
                0,
                int((access_exp - now).total_seconds()),
            ),
            "expires_at_taipei": dt_to_iso(access_exp),
            "message": f"{label}使用中。",
        }

    trial_exp = parse_dt(record.get("trial_expires_at"))
    if trial_exp and trial_exp > now:
        return {
            **base,
            "state": "active",
            "active": True,
            "can_predict": True,
            "can_start_trial": False,
            "plan": "trial",
            "plan_label": f"{TRIAL_MINUTES}分鐘試用",
            "remaining_seconds": max(
                0,
                int((trial_exp - now).total_seconds()),
            ),
            "expires_at_taipei": dt_to_iso(trial_exp),
            "message": f"{TRIAL_MINUTES}分鐘試用中。",
        }

    if record.get("used_trial"):
        return {
            **base,
            "state": "expired",
            "active": False,
            "can_predict": False,
            "can_start_trial": False,
            "plan": record.get("plan") or "expired",
            "plan_label": "試用／權限已到期",
            "remaining_seconds": 0,
            "expires_at_taipei": (
                record.get("trial_expires_at")
                or record.get("access_expires_at")
                or ""
            ),
            "message": (
                "試用或使用權限已到期，"
                f"請聯繫管理員：{ADMIN_LINE_ID}"
            ),
        }

    return {
        **base,
        "state": "trial_available",
        "active": False,
        "can_predict": True,
        "can_start_trial": True,
        "plan": "trial_available",
        "plan_label": "尚未開始試用",
        "remaining_seconds": TRIAL_MINUTES * 60,
        "expires_at_taipei": "",
        "message": (
            f"第一次開始分析時，會啟動 {TRIAL_MINUTES} 分鐘試用。"
        ),
    }


def ensure_access_or_start_trial(user_id: str) -> Dict[str, Any]:
    status = access_status(user_id)
    if status.get("active"):
        return status

    if status.get("can_start_trial"):
        now = now_taipei()
        expiry = now + timedelta(minutes=TRIAL_MINUTES)
        record = get_access_record(user_id)
        record.update(
            {
                "plan": "trial",
                "permanent": False,
                "used_trial": True,
                "trial_started_at": dt_to_iso(now),
                "trial_expires_at": dt_to_iso(expiry),
            }
        )
        save_access_record(user_id, record)
        return access_status(user_id)

    raise AccessDenied(
        status.get("message")
        or f"使用權限已到期，請聯繫 {ADMIN_LINE_ID}"
    )


def activate_user(
    user_id: str,
    code: str,
) -> Dict[str, Any]:
    value = str(code or "").strip()
    if value not in ALL_ACCESS_CODES:
        raise ValueError("開通碼錯誤。")

    now = now_taipei()
    record = get_access_record(user_id)

    if value in PERMANENT_CODES:
        record.update(
            {
                "plan": "permanent",
                "permanent": True,
                "used_trial": True,
                "access_expires_at": "",
            }
        )
    elif value in MONTHLY_CODES:
        record.update(
            {
                "plan": "monthly",
                "permanent": False,
                "used_trial": True,
                "access_expires_at": dt_to_iso(
                    now + timedelta(days=MONTHLY_DAYS)
                ),
            }
        )
    else:
        record.update(
            {
                "plan": "temporary",
                "permanent": False,
                "used_trial": True,
                "access_expires_at": dt_to_iso(
                    now + timedelta(minutes=TEMP_TRIAL_MINUTES)
                ),
            }
        )

    save_access_record(user_id, record)
    return access_status(user_id)


def format_remaining(seconds: Optional[int]) -> str:
    if seconds is None:
        return "永久"
    value = max(0, int(seconds or 0))
    days, remainder = divmod(value, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)

    if days > 0:
        return f"{days}天 {hours}小時"
    if hours > 0:
        return f"{hours}小時 {minutes}分鐘"
    if minutes > 0:
        return f"{minutes}分 {secs}秒"
    return f"{secs}秒"


def access_summary_text(user_id: str) -> str:
    status = access_status(user_id)
    if status.get("state") == "trial_available":
        return f"UID 試用：尚未開始｜可用 {TRIAL_MINUTES} 分鐘"

    if status.get("state") == "expired":
        return "UID 試用／權限：已到期"

    return (
        f"UID 權限：{status.get('plan_label', '-')}"
        f"｜剩餘 {format_remaining(status.get('remaining_seconds'))}"
    )


# ---------------------------------------------------------------------------
# LINE helpers
# ---------------------------------------------------------------------------


def verify_line_signature(
    body: bytes,
    signature: Optional[str],
) -> bool:
    if not CHANNEL_SECRET:
        return True
    if not signature:
        return False
    digest = hmac.new(
        CHANNEL_SECRET.encode("utf-8"),
        body,
        hashlib.sha256,
    ).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, signature)


def line_reply(
    reply_token: str,
    messages: List[Dict[str, Any]],
) -> None:
    if not CHANNEL_ACCESS_TOKEN:
        print(json.dumps(messages, ensure_ascii=False))
        return

    response = requests.post(
        "https://api.line.me/v2/bot/message/reply",
        headers={
            "Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}",
            "Content-Type": "application/json",
        },
        json={
            "replyToken": reply_token,
            "messages": messages[:5],
        },
        timeout=8,
    )
    if response.status_code >= 300:
        print(
            "LINE reply failed",
            response.status_code,
            response.text,
        )


def text_msg(text: str) -> Dict[str, Any]:
    return {
        "type": "text",
        "text": str(text)[:5000],
    }


def postback_action(
    label: str,
    data: Dict[str, str],
) -> Dict[str, Any]:
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
    color: str = "#06C755",
) -> Dict[str, Any]:
    return {
        "type": "button",
        "style": "primary",
        "color": color,
        "height": "sm",
        "action": {
            "type": "uri",
            "label": label[:20],
            "uri": uri[:1000],
        },
    }


def parse_venues() -> List[Dict[str, str]]:
    venues: List[Dict[str, str]] = []
    for raw in VENUES_RAW.split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" in item:
            code, name = item.split(":", 1)
        else:
            code, name = item, item
        venues.append(
            {
                "code": code.strip().upper(),
                "name": name.strip(),
            }
        )
    return venues


def venue_name(code: str) -> str:
    value = str(code or "").upper()
    for venue in parse_venues():
        if venue["code"] == value:
            return venue["name"]
    return value or "-"


def get_source_user_id(event: Dict[str, Any]) -> str:
    source = event.get("source") or {}
    return (
        source.get("userId")
        or source.get("groupId")
        or source.get("roomId")
        or "anonymous"
    )


def get_session_or_create(user_id: str) -> Dict[str, Any]:
    session = store.get_session(user_id)
    if not session:
        session = store.upsert_session(user_id, {})
    return session


def result_name(code: str) -> str:
    return {
        "B": "莊",
        "P": "閒",
        "T": "和",
    }.get(str(code).upper(), str(code))


# ---------------------------------------------------------------------------
# Flex panels
# ---------------------------------------------------------------------------


def trial_info_box(user_id: str) -> Dict[str, Any]:
    status = access_status(user_id)
    expired = status.get("state") == "expired"
    return {
        "type": "box",
        "layout": "vertical",
        "backgroundColor": "#2A2A2A",
        "cornerRadius": "md",
        "paddingAll": "10px",
        "margin": "lg",
        "contents": [
            {
                "type": "text",
                "text": "LINE UID 使用權限",
                "size": "xs",
                "weight": "bold",
                "color": "#FFD000",
            },
            {
                "type": "text",
                "text": access_summary_text(user_id),
                "size": "xs",
                "color": "#FF7777" if expired else "#FFFFFF",
                "wrap": True,
                "margin": "sm",
            },
            {
                "type": "text",
                "text": f"UID：{user_id}",
                "size": "xxs",
                "color": "#999999",
                "wrap": True,
                "margin": "sm",
            },
        ],
    }


def start_menu_flex(user_id: str) -> Dict[str, Any]:
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
                    {
                        "type": "text",
                        "text": "AI 規律模型",
                        "weight": "bold",
                        "size": "xl",
                        "color": "#FFD000",
                    },
                    {
                        "type": "text",
                        "text": "點擊開始分析後，先選擇遊戲館。",
                        "size": "sm",
                        "color": "#FFFFFF",
                        "margin": "md",
                        "wrap": True,
                    },
                    trial_info_box(user_id),
                    {
                        "type": "separator",
                        "margin": "lg",
                        "color": "#FFD000",
                    },
                    {
                        "type": "box",
                        "layout": "vertical",
                        "spacing": "md",
                        "margin": "lg",
                        "contents": [
                            button(
                                "開始分析",
                                {"action": "open_venue"},
                            )
                        ],
                    },
                ],
            },
        },
    }


def venue_flex(user_id: str) -> Dict[str, Any]:
    buttons = [
        button(
            venue["name"],
            {
                "action": "select_venue",
                "venue": venue["code"],
            },
        )
        for venue in parse_venues()
    ]

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
                    {
                        "type": "text",
                        "text": "AI 規律模型",
                        "weight": "bold",
                        "size": "xl",
                        "color": "#FFD000",
                    },
                    {
                        "type": "text",
                        "text": (
                            "請選擇遊戲館，選擇後會提醒您"
                            "直接在聊天室輸入房間名稱或桌號。"
                        ),
                        "size": "sm",
                        "color": "#FFFFFF",
                        "margin": "md",
                        "wrap": True,
                    },
                    trial_info_box(user_id),
                    {
                        "type": "separator",
                        "margin": "lg",
                        "color": "#FFD000",
                    },
                    {
                        "type": "box",
                        "layout": "vertical",
                        "spacing": "md",
                        "margin": "lg",
                        "contents": buttons,
                    },
                ],
            },
        },
    }


def room_input_flex(
    user_id: str,
    venue_code: str,
) -> Dict[str, Any]:
    return {
        "type": "flex",
        "altText": "請輸入房間",
        "contents": {
            "type": "bubble",
            "size": "mega",
            "body": {
                "type": "box",
                "layout": "vertical",
                "backgroundColor": "#111111",
                "paddingAll": "18px",
                "contents": [
                    {
                        "type": "text",
                        "text": venue_name(venue_code),
                        "weight": "bold",
                        "size": "xl",
                        "color": "#FFD000",
                        "wrap": True,
                    },
                    {
                        "type": "text",
                        "text": "請直接在 LINE 聊天室輸入房間名稱、桌號或靴號。",
                        "size": "sm",
                        "color": "#FFFFFF",
                        "margin": "md",
                        "wrap": True,
                    },
                    {
                        "type": "box",
                        "layout": "vertical",
                        "backgroundColor": "#2A2A2A",
                        "cornerRadius": "md",
                        "paddingAll": "10px",
                        "margin": "lg",
                        "contents": [
                            {
                                "type": "text",
                                "text": "輸入範例",
                                "weight": "bold",
                                "size": "xs",
                                "color": "#FFD000",
                            },
                            {
                                "type": "text",
                                "text": ROOM_INPUT_EXAMPLES,
                                "size": "sm",
                                "color": "#FFFFFF",
                                "wrap": True,
                                "margin": "sm",
                            },
                        ],
                    },
                    trial_info_box(user_id),
                    {
                        "type": "separator",
                        "margin": "lg",
                        "color": "#FFD000",
                    },
                    {
                        "type": "box",
                        "layout": "vertical",
                        "spacing": "sm",
                        "margin": "lg",
                        "contents": [
                            button(
                                "重新選館",
                                {"action": "open_venue"},
                                "#555555",
                            )
                        ],
                    },
                ],
            },
        },
    }


def input_panel_flex(
    user_id: str,
    session: Dict[str, Any],
    notice: str = "",
) -> Dict[str, Any]:
    history = session.get("history") or []
    venue = session.get("venue") or ""
    room = session.get("room") or "-"
    next_round = len(history) + 1

    contents: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": "AI 規律分析",
            "weight": "bold",
            "size": "xl",
            "color": "#FFD000",
        },
        {
            "type": "text",
            "text": (
                f"{venue_name(venue)}｜{room}｜第 {next_round} 局"
            ),
            "size": "sm",
            "color": "#FFFFFF",
            "margin": "md",
            "wrap": True,
        },
        trial_info_box(user_id),
    ]

    if notice:
        contents.append(
            {
                "type": "box",
                "layout": "vertical",
                "backgroundColor": "#FFF2A8",
                "cornerRadius": "md",
                "paddingAll": "8px",
                "margin": "lg",
                "contents": [
                    {
                        "type": "text",
                        "text": notice,
                        "size": "xs",
                        "color": "#111111",
                        "wrap": True,
                    }
                ],
            }
        )

    contents.extend(
        [
            {
                "type": "separator",
                "margin": "lg",
                "color": "#FFD000",
            },
            {
                "type": "text",
                "text": (
                    "目前紀錄："
                    + (" ".join(history[-24:]) if history else "尚無")
                ),
                "size": "xs",
                "color": "#DDDDDD",
                "wrap": True,
                "margin": "lg",
            },
            {
                "type": "box",
                "layout": "horizontal",
                "spacing": "sm",
                "margin": "lg",
                "contents": [
                    button(
                        "莊 B",
                        {"action": "round", "result": "B"},
                        "#E60012",
                    ),
                    button(
                        "閒 P",
                        {"action": "round", "result": "P"},
                        "#0B46D9",
                    ),
                    button(
                        "和 T",
                        {"action": "round", "result": "T"},
                        "#00A040",
                    ),
                ],
            },
            {
                "type": "box",
                "layout": "horizontal",
                "spacing": "sm",
                "margin": "md",
                "contents": [
                    button(
                        "上一步",
                        {"action": "undo"},
                        "#555555",
                    ),
                    button(
                        "新靴",
                        {"action": "new_shoe"},
                        "#555555",
                    ),
                ],
            },
            {
                "type": "box",
                "layout": "vertical",
                "spacing": "sm",
                "margin": "md",
                "contents": [
                    button(
                        "開始 AI 判斷",
                        {"action": "predict"},
                        "#FFD000",
                    ),
                    button(
                        "結束分析",
                        {"action": "end"},
                        "#111111",
                    ),
                ],
            },
        ]
    )

    return {
        "type": "flex",
        "altText": "莊閒和輸入面板",
        "contents": {
            "type": "bubble",
            "size": "mega",
            "body": {
                "type": "box",
                "layout": "vertical",
                "backgroundColor": "#111111",
                "paddingAll": "18px",
                "contents": contents,
            },
        },
    }


def result_flex(
    user_id: str,
    session: Dict[str, Any],
) -> Dict[str, Any]:
    prediction = session.get("last_prediction") or {}
    history = session.get("history") or []
    venue = session.get("venue") or ""
    room = session.get("room") or "-"

    return {
        "type": "flex",
        "altText": (
            "下一局建議："
            f"{prediction.get('recommend_text', '-')}"
        ),
        "contents": {
            "type": "bubble",
            "size": "mega",
            "body": {
                "type": "box",
                "layout": "vertical",
                "backgroundColor": "#111111",
                "paddingAll": "18px",
                "contents": [
                    {
                        "type": "text",
                        "text": "下一局預測",
                        "weight": "bold",
                        "size": "xl",
                        "color": "#FFD000",
                    },
                    {
                        "type": "text",
                        "text": (
                            f"{venue_name(venue)}｜{room}"
                            f"｜第 {len(history) + 1} 局"
                        ),
                        "size": "xs",
                        "color": "#CCCCCC",
                        "margin": "md",
                        "wrap": True,
                    },
                    trial_info_box(user_id),
                    {
                        "type": "separator",
                        "margin": "lg",
                        "color": "#FFD000",
                    },
                    {
                        "type": "text",
                        "text": (
                            f"莊　{prediction.get('banker_rate', 0):.1f}%"
                        ),
                        "size": "lg",
                        "weight": "bold",
                        "color": "#FF4D4F",
                        "margin": "lg",
                    },
                    {
                        "type": "text",
                        "text": (
                            f"閒　{prediction.get('player_rate', 0):.1f}%"
                        ),
                        "size": "lg",
                        "weight": "bold",
                        "color": "#4D8DFF",
                        "margin": "md",
                    },
                    {
                        "type": "text",
                        "text": (
                            f"和　{prediction.get('tie_rate', 0):.1f}%"
                        ),
                        "size": "lg",
                        "weight": "bold",
                        "color": "#4DD36F",
                        "margin": "md",
                    },
                    {
                        "type": "box",
                        "layout": "horizontal",
                        "backgroundColor": "#FFD000",
                        "cornerRadius": "md",
                        "paddingAll": "10px",
                        "margin": "lg",
                        "contents": [
                            {
                                "type": "text",
                                "text": "推薦",
                                "weight": "bold",
                                "color": "#111111",
                            },
                            {
                                "type": "text",
                                "text": str(
                                    prediction.get(
                                        "recommend_text",
                                        "-",
                                    )
                                ),
                                "weight": "bold",
                                "color": "#111111",
                                "align": "end",
                            },
                        ],
                    },
                    {
                        "type": "text",
                        "text": (
                            f"{prediction.get('signal_level', '')}"
                            f"｜{prediction.get('reason', '')}"
                        ),
                        "size": "xs",
                        "color": "#AAAAAA",
                        "wrap": True,
                        "margin": "md",
                    },
                    {
                        "type": "box",
                        "layout": "horizontal",
                        "spacing": "sm",
                        "margin": "lg",
                        "contents": [
                            button(
                                "開莊",
                                {"action": "round", "result": "B"},
                                "#E60012",
                            ),
                            button(
                                "開閒",
                                {"action": "round", "result": "P"},
                                "#0B46D9",
                            ),
                            button(
                                "開和",
                                {"action": "round", "result": "T"},
                                "#00A040",
                            ),
                        ],
                    },
                    {
                        "type": "box",
                        "layout": "horizontal",
                        "spacing": "sm",
                        "margin": "md",
                        "contents": [
                            button(
                                "回輸入面板",
                                {"action": "view_panel"},
                                "#555555",
                            ),
                            button(
                                "結束分析",
                                {"action": "end"},
                                "#111111",
                            ),
                        ],
                    },
                ],
            },
        },
    }


def end_flex(
    user_id: str,
    session: Dict[str, Any],
) -> Dict[str, Any]:
    history = session.get("history") or []
    return {
        "type": "flex",
        "altText": "本靴分析已結束",
        "contents": {
            "type": "bubble",
            "size": "mega",
            "body": {
                "type": "box",
                "layout": "vertical",
                "backgroundColor": "#111111",
                "paddingAll": "18px",
                "contents": [
                    {
                        "type": "text",
                        "text": "本靴分析已結束",
                        "weight": "bold",
                        "size": "xl",
                        "color": "#FFD000",
                    },
                    {
                        "type": "text",
                        "text": (
                            f"總局數：{len(history)} 局。"
                            "需要下一靴時，點擊下方開始分析。"
                        ),
                        "size": "sm",
                        "color": "#FFFFFF",
                        "margin": "md",
                        "wrap": True,
                    },
                    trial_info_box(user_id),
                    {
                        "type": "separator",
                        "margin": "lg",
                        "color": "#FFD000",
                    },
                    {
                        "type": "box",
                        "layout": "vertical",
                        "spacing": "sm",
                        "margin": "lg",
                        "contents": [
                            button(
                                "開始分析",
                                {"action": "open_venue"},
                            )
                        ],
                    },
                ],
            },
        },
    }


def expired_flex(user_id: str) -> Dict[str, Any]:
    return {
        "type": "flex",
        "altText": "試用已到期",
        "contents": {
            "type": "bubble",
            "size": "mega",
            "body": {
                "type": "box",
                "layout": "vertical",
                "backgroundColor": "#111111",
                "paddingAll": "18px",
                "contents": [
                    {
                        "type": "text",
                        "text": "UID 試用已到期",
                        "weight": "bold",
                        "size": "xl",
                        "color": "#FFD000",
                    },
                    {
                        "type": "text",
                        "text": (
                            "請聯繫管理員取得開通碼，"
                            "或直接在聊天室輸入開通碼。"
                        ),
                        "size": "sm",
                        "color": "#FFFFFF",
                        "margin": "md",
                        "wrap": True,
                    },
                    trial_info_box(user_id),
                    {
                        "type": "box",
                        "layout": "vertical",
                        "spacing": "sm",
                        "margin": "lg",
                        "contents": [
                            uri_button(
                                "聯繫官方 LINE 管理員",
                                ADMIN_LINE_URL,
                            )
                        ],
                    },
                ],
            },
        },
    }


# ---------------------------------------------------------------------------
# Session / prediction
# ---------------------------------------------------------------------------


def save_selected_venue(
    user_id: str,
    venue_code: str,
) -> Dict[str, Any]:
    session = get_session_or_create(user_id)
    session.update(
        {
            "venue": venue_code,
            "room": "",
            "status": "等待輸入房間",
        }
    )
    return store.upsert_session(user_id, session)


def save_room(
    user_id: str,
    room_text: str,
) -> Dict[str, Any]:
    session = get_session_or_create(user_id)
    if not session.get("venue"):
        raise ValueError("請先選擇遊戲館。")

    session.update(
        {
            "room": room_text.strip(),
            "status": "分析中",
        }
    )
    return store.upsert_session(user_id, session)


def predict_and_save(user_id: str) -> Dict[str, Any]:
    session = get_session_or_create(user_id)

    if not session.get("venue"):
        raise ValueError("請先選擇遊戲館。")
    if not session.get("room"):
        raise ValueError("請先輸入房間名稱或桌號。")

    ensure_access_or_start_trial(user_id)

    prediction = predict(
        session.get("history") or [],
        venue=session.get("venue", ""),
        room=session.get("room", ""),
        shoe_id=session.get("shoe_id", ""),
        user_id=user_id,
    )
    session["last_prediction"] = prediction
    session["status"] = "可回報結果"
    return store.upsert_session(user_id, session)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.api_route("/", methods=["GET", "HEAD"])
def root() -> PlainTextResponse:
    return PlainTextResponse("OK")


@app.api_route("/health", methods=["GET", "HEAD"])
def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "version": "3.1.0",
            "time_taipei": dt_to_iso(now_taipei()),
        }
    )


@app.api_route("/ping", methods=["GET", "HEAD"])
def ping() -> PlainTextResponse:
    return PlainTextResponse("pong")


@app.get("/liff")
def liff_compatibility() -> PlainTextResponse:
    return PlainTextResponse(
        "此版本已改為直接在 LINE 聊天室操作，"
        "請返回 LINE 並輸入「開始分析」。"
    )


@app.get("/favicon.ico")
def favicon() -> PlainTextResponse:
    return PlainTextResponse("", status_code=204)


@app.post("/webhook")
async def webhook(request: Request) -> JSONResponse:
    body = await request.body()
    if not verify_line_signature(
        body,
        request.headers.get("X-Line-Signature"),
    ):
        return JSONResponse(
            {"ok": False, "error": "bad signature"},
            status_code=401,
        )

    payload = json.loads(body.decode("utf-8") or "{}")

    for event in payload.get("events", []):
        reply_token = event.get("replyToken", "")
        user_id = get_source_user_id(event)
        event_type = event.get("type")

        try:
            if event_type == "follow":
                line_reply(
                    reply_token,
                    [start_menu_flex(user_id)],
                )
                continue

            if (
                event_type == "message"
                and (event.get("message") or {}).get("type") == "text"
            ):
                text = str(
                    (event.get("message") or {}).get("text") or ""
                ).strip()
                lower = text.lower()

                if text in ALL_ACCESS_CODES:
                    status = activate_user(user_id, text)
                    line_reply(
                        reply_token,
                        [
                            text_msg(
                                "✅ 開通成功\n"
                                f"方案：{status.get('plan_label')}\n"
                                f"剩餘："
                                f"{format_remaining(status.get('remaining_seconds'))}"
                            ),
                            start_menu_flex(user_id),
                        ],
                    )
                    continue

                if text in {
                    "開始",
                    "開始分析",
                    "選館",
                    "重新選館",
                }:
                    status = access_status(user_id)
                    if status.get("state") == "expired":
                        line_reply(
                            reply_token,
                            [expired_flex(user_id)],
                        )
                    else:
                        line_reply(
                            reply_token,
                            [venue_flex(user_id)],
                        )
                    continue

                if text in {"面板", "輸入面板", "查看紀錄"}:
                    session = get_session_or_create(user_id)
                    line_reply(
                        reply_token,
                        [input_panel_flex(user_id, session)],
                    )
                    continue

                if text in {"AI", "預測", "開始AI判斷", "判斷"}:
                    try:
                        session = predict_and_save(user_id)
                        line_reply(
                            reply_token,
                            [result_flex(user_id, session)],
                        )
                    except AccessDenied:
                        line_reply(
                            reply_token,
                            [expired_flex(user_id)],
                        )
                    continue

                if text in {"結束", "結束分析"}:
                    session = store.end_session(user_id)
                    line_reply(
                        reply_token,
                        [end_flex(user_id, session)],
                    )
                    continue

                mapping = {
                    "莊": "B",
                    "庄": "B",
                    "b": "B",
                    "閒": "P",
                    "闲": "P",
                    "p": "P",
                    "和": "T",
                    "t": "T",
                }
                result = mapping.get(text) or mapping.get(lower)
                if result:
                    try:
                        ensure_access_or_start_trial(user_id)
                        session = store.add_round(user_id, result)
                        line_reply(
                            reply_token,
                            [
                                input_panel_flex(
                                    user_id,
                                    session,
                                    f"已新增：{result_name(result)}",
                                )
                            ],
                        )
                    except AccessDenied:
                        line_reply(
                            reply_token,
                            [expired_flex(user_id)],
                        )
                    continue

                # 已選館但尚未輸入房間時，任何一般文字都當房間名稱。
                session = get_session_or_create(user_id)
                if session.get("venue") and not session.get("room"):
                    saved = save_room(user_id, text)
                    line_reply(
                        reply_token,
                        [
                            input_panel_flex(
                                user_id,
                                saved,
                                f"已設定房間：{text}",
                            )
                        ],
                    )
                    continue

                line_reply(
                    reply_token,
                    [
                        text_msg(
                            "請輸入「開始分析」，"
                            "或使用面板按鈕操作。"
                        )
                    ],
                )
                continue

            if event_type == "postback":
                parsed = urllib.parse.parse_qs(
                    (event.get("postback") or {}).get("data", "")
                )
                data = {
                    key: values[0]
                    for key, values in parsed.items()
                }
                action = data.get("action", "")

                if action == "open_venue":
                    line_reply(
                        reply_token,
                        [venue_flex(user_id)],
                    )

                elif action == "select_venue":
                    venue = data.get("venue", "")
                    save_selected_venue(user_id, venue)
                    line_reply(
                        reply_token,
                        [room_input_flex(user_id, venue)],
                    )

                elif action == "view_panel":
                    session = get_session_or_create(user_id)
                    line_reply(
                        reply_token,
                        [input_panel_flex(user_id, session)],
                    )

                elif action == "round":
                    try:
                        ensure_access_or_start_trial(user_id)
                        result = data.get("result", "")
                        session = store.add_round(user_id, result)
                        line_reply(
                            reply_token,
                            [
                                input_panel_flex(
                                    user_id,
                                    session,
                                    f"已新增：{result_name(result)}",
                                )
                            ],
                        )
                    except AccessDenied:
                        line_reply(
                            reply_token,
                            [expired_flex(user_id)],
                        )

                elif action == "predict":
                    try:
                        session = predict_and_save(user_id)
                        line_reply(
                            reply_token,
                            [result_flex(user_id, session)],
                        )
                    except AccessDenied:
                        line_reply(
                            reply_token,
                            [expired_flex(user_id)],
                        )

                elif action == "undo":
                    session = store.undo_round(user_id)
                    line_reply(
                        reply_token,
                        [
                            input_panel_flex(
                                user_id,
                                session,
                                "已刪除上一局",
                            )
                        ],
                    )

                elif action == "new_shoe":
                    session = store.clear_history(user_id)
                    line_reply(
                        reply_token,
                        [
                            input_panel_flex(
                                user_id,
                                session,
                                "已建立新靴",
                            )
                        ],
                    )

                elif action == "end":
                    session = store.end_session(user_id)
                    line_reply(
                        reply_token,
                        [end_flex(user_id, session)],
                    )

                else:
                    line_reply(
                        reply_token,
                        [start_menu_flex(user_id)],
                    )

        except Exception as exc:
            print("event failed:", repr(exc))
            traceback.print_exc()
            line_reply(
                reply_token,
                [text_msg(f"操作失敗：{exc}")],
            )

    return JSONResponse({"ok": True})


@app.post("/callback")
async def callback(request: Request) -> JSONResponse:
    return await webhook(request)
