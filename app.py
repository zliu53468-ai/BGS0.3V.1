"""LINE point-input baccarat bot for V5 independent point prediction.

Operation flow
--------------
1. Start analysis.
2. Select venue.
3. Enter room.
4. Enter two digits directly in chat, e.g. 65:
      first digit  = Player point
      second digit = Banker point
5. The bot stores the observation for display/statistics, but V5 prediction
   uses only the newest point observation.
6. The result panel only keeps the End Analysis button.
7. Enter the next two-digit point result directly in chat to continue.

Older formats remain accepted:
    閒6莊5
    P6B5
    6,5
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import traceback
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

import store
from predictor import predict, parse_point_observation

BASE_DIR = Path(__file__).resolve().parent

CHANNEL_ACCESS_TOKEN = os.getenv(
    "LINE_CHANNEL_ACCESS_TOKEN",
    "",
).strip()
CHANNEL_SECRET = os.getenv(
    "LINE_CHANNEL_SECRET",
    "",
).strip()

TAIPEI_TZ = timezone(timedelta(hours=8))
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "30") or "30")
ADMIN_LINE_URL = os.getenv(
    "ADMIN_LINE_URL",
    "https://line.me/R/ti/p/%40jins888",
)

ACCESS_FILE = Path(
    os.getenv(
        "ACCESS_DATA_FILE",
        str(BASE_DIR / "data" / "access_control.json"),
    )
)

VENUES = [
    ("OB", "歐博真人"),
    ("DG", "DG真人"),
    ("MT", "MT真人"),
    ("T9", "T9真人"),
    ("SA", "SA真人"),
    ("DB", "DB真人"),
]

PERMANENT_CODES = {
    "aaa1688003",
    "aaa1888007",
    "aaa1000889",
}
MONTHLY_CODES = {
    "aaa13002",
    "aaa15001",
    "aaa199801",
}
TEMP_CODES = {
    "aaaa1999152",
    "aaa345556",
    "aaa987743",
}
ALL_CODES = PERMANENT_CODES | MONTHLY_CODES | TEMP_CODES

app = FastAPI(
    title="Baccarat V5 Independent Point Bot",
    version="5.0.0",
)


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------


def now() -> datetime:
    return datetime.now(TAIPEI_TZ)


def iso(value: datetime) -> str:
    return value.astimezone(TAIPEI_TZ).isoformat(timespec="seconds")


def parse_dt(value: Any) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=TAIPEI_TZ)
        return parsed.astimezone(TAIPEI_TZ)
    except Exception:
        return None


def load_access() -> Dict[str, Any]:
    try:
        if not ACCESS_FILE.exists():
            return {}
        with ACCESS_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_access(data: Dict[str, Any]) -> None:
    ACCESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = ACCESS_FILE.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temporary.replace(ACCESS_FILE)


def status(user_id: str) -> Dict[str, Any]:
    record = load_access().get(user_id, {})

    if record.get("permanent"):
        return {
            "active": True,
            "label": "永久版",
            "remaining": None,
        }

    expiry = parse_dt(
        record.get("access_expires_at")
        or record.get("trial_expires_at")
    )
    if expiry and expiry > now():
        return {
            "active": True,
            "label": record.get("plan", "試用"),
            "remaining": int((expiry - now()).total_seconds()),
        }

    if record.get("used_trial"):
        return {
            "active": False,
            "expired": True,
            "label": "已到期",
            "remaining": 0,
        }

    return {
        "active": False,
        "trial_available": True,
        "label": "尚未開始試用",
        "remaining": TRIAL_MINUTES * 60,
    }


def ensure(user_id: str) -> Dict[str, Any]:
    current = status(user_id)
    if current.get("active"):
        return current

    if current.get("trial_available"):
        data = load_access()
        record = data.get(user_id, {})
        record.update(
            {
                "used_trial": True,
                "plan": "trial",
                "trial_expires_at": iso(
                    now() + timedelta(minutes=TRIAL_MINUTES)
                ),
            }
        )
        data[user_id] = record
        save_access(data)
        return status(user_id)

    raise PermissionError("expired")


def activate(
    user_id: str,
    code: str,
) -> Dict[str, Any]:
    data = load_access()
    record = data.get(user_id, {})

    if code in PERMANENT_CODES:
        record.update(
            {
                "permanent": True,
                "used_trial": True,
                "plan": "permanent",
            }
        )
    elif code in MONTHLY_CODES:
        record.update(
            {
                "permanent": False,
                "used_trial": True,
                "plan": "monthly",
                "access_expires_at": iso(
                    now() + timedelta(days=30)
                ),
            }
        )
    elif code in TEMP_CODES:
        record.update(
            {
                "permanent": False,
                "used_trial": True,
                "plan": "temporary",
                "access_expires_at": iso(
                    now() + timedelta(minutes=30)
                ),
            }
        )
    else:
        raise ValueError("開通碼錯誤")

    data[user_id] = record
    save_access(data)
    return status(user_id)


def remaining_text(seconds: Optional[int]) -> str:
    if seconds is None:
        return "永久"
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    if minutes:
        return f"{minutes}分{secs}秒"
    return f"{secs}秒"


# ---------------------------------------------------------------------------
# LINE helpers
# ---------------------------------------------------------------------------


def verify(
    body: bytes,
    signature: Optional[str],
) -> bool:
    if not CHANNEL_SECRET:
        return True
    if not signature:
        return False

    expected = base64.b64encode(
        hmac.new(
            CHANNEL_SECRET.encode(),
            body,
            hashlib.sha256,
        ).digest()
    ).decode()

    return hmac.compare_digest(expected, signature)


def reply(
    token: str,
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
            "replyToken": token,
            "messages": messages[:5],
        },
        timeout=8,
    )
    if response.status_code >= 300:
        print(
            "LINE reply failed:",
            response.status_code,
            response.text,
        )


def text_message(text: str) -> Dict[str, Any]:
    return {
        "type": "text",
        "text": str(text)[:5000],
    }


def action(
    label: str,
    action_name: str,
    **kwargs: str,
) -> Dict[str, Any]:
    data = {
        "action": action_name,
        **{key: str(value) for key, value in kwargs.items()},
    }
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


def flex(
    title: str,
    body: str,
    buttons: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    contents: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": title,
            "weight": "bold",
            "size": "xl",
            "color": "#FFD000",
        },
        {
            "type": "text",
            "text": body,
            "wrap": True,
            "margin": "md",
            "color": "#FFFFFF",
        },
    ]

    if buttons:
        contents.append(
            {
                "type": "box",
                "layout": "vertical",
                "spacing": "sm",
                "margin": "lg",
                "contents": buttons,
            }
        )

    return {
        "type": "flex",
        "altText": title,
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


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------


def parse_chat_point_observation(
    text: str,
) -> Optional[Dict[str, int]]:
    """Parse chat point input.

    Compact format:
        65 -> Player 6, Banker 5
        00 -> Player 0, Banker 0

    Older formats remain supported through predictor.parse_point_observation.
    """
    value = str(text or "").strip()

    if re.fullmatch(r"\d{2}", value):
        return {
            "player": int(value[0]),
            "banker": int(value[1]),
        }

    return parse_point_observation(value)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def venue_panel(user_id: str) -> Dict[str, Any]:
    user_status = status(user_id)
    return flex(
        "AI 點數粒子模型 V5",
        (
            f"UID權限：{user_status['label']}"
            f"｜剩餘：{remaining_text(user_status.get('remaining'))}\n"
            "請選擇遊戲館。"
        ),
        [
            action(
                name,
                "venue",
                venue=code,
            )
            for code, name in VENUES
        ],
    )


def room_panel(venue_code: str) -> Dict[str, Any]:
    return flex(
        "請輸入房間",
        (
            f"已選擇 {dict(VENUES).get(venue_code, venue_code)}。\n"
            "請直接輸入房間名稱或桌號。"
        ),
    )


def ready_panel(
    user_id: str,
    session: Dict[str, Any],
) -> Dict[str, Any]:
    user_status = status(user_id)
    return flex(
        "可直接輸入點數",
        (
            f"館別：{session.get('venue') or '-'}\n"
            f"房間：{session.get('room') or '-'}\n"
            f"UID權限：{user_status['label']}"
            f"｜剩餘：{remaining_text(user_status.get('remaining'))}\n\n"
            "請直接在聊天室輸入兩位數：\n"
            "例如 65＝閒6點、莊5點。\n\n"
            "V5 每次只使用本次最新點數，"
            "不沿用上一筆點數或上一局粒子狀態。"
        ),
        [
            action(
                "結束分析",
                "end",
            )
        ],
    )


def _decision_source_text(value: Any) -> str:
    source = str(value or "")
    if source == "VALIDATED_MODEL":
        return "驗證模型"
    if source == "LOW_CONFIDENCE_BALANCED":
        return "低信心平衡"
    return source or "V5模型"


def result_panel(
    user_id: str,
    session: Dict[str, Any],
) -> Dict[str, Any]:
    prediction = session.get("last_prediction") or {}
    observations = session.get("observations") or []
    user_status = status(user_id)
    source_text = _decision_source_text(
        prediction.get("decision_source")
    )

    body = (
        f"本次第 {len(observations)} 次獨立分析\n"
        f"莊 {prediction.get('banker_rate', 0):.1f}%\n"
        f"閒 {prediction.get('player_rate', 0):.1f}%\n"
        f"和 {prediction.get('tie_rate', 0):.1f}%\n\n"
        f"推薦：{prediction.get('recommend_text', '-')}\n"
        f"訊號：{prediction.get('signal_level', '-')}\n"
        f"來源：{source_text}\n\n"
        f"UID權限：{user_status['label']}"
        f"｜剩餘：{remaining_text(user_status.get('remaining'))}\n\n"
        "請直接輸入下一組點數，例如：65\n"
        "每次分析完全獨立，不沿用上一筆點數。"
    )

    return flex(
        "V5 下一局點數模擬",
        body,
        [
            action(
                "結束分析",
                "end",
            )
        ],
    )


def ended_panel() -> Dict[str, Any]:
    return flex(
        "本次分析已結束",
        "需要再次分析時，請輸入「開始分析」。",
    )


def expired_panel() -> Dict[str, Any]:
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
                    {
                        "type": "text",
                        "text": "試用已到期",
                        "weight": "bold",
                        "size": "xl",
                        "color": "#FFD000",
                    },
                    {
                        "type": "button",
                        "style": "primary",
                        "color": "#06C755",
                        "margin": "lg",
                        "action": {
                            "type": "uri",
                            "label": "聯繫管理員",
                            "uri": ADMIN_LINE_URL,
                        },
                    },
                ],
            },
        },
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_session(
    user_id: str,
) -> Dict[str, Any]:
    """Return the last stored V5 prediction without running the model again."""
    session = (
        store.get_session(user_id)
        or store.new_session(user_id)
    )
    ensure(user_id)

    if not session.get("last_prediction"):
        raise ValueError("尚未輸入點數，請先輸入例如：65")

    return session


def add_points_and_predict(
    user_id: str,
    observation: Dict[str, int],
) -> Dict[str, Any]:
    """Store the point for UI/statistics, but pass only this point to V5."""
    ensure(user_id)

    session = store.add_point_observation(
        user_id,
        observation["player"],
        observation["banker"],
    )

    prediction = predict(
        [observation],
        venue=session.get("venue", ""),
        room=session.get("room", ""),
        shoe_id=session.get("shoe_id", ""),
        user_id=user_id,
    )

    if not prediction.get("ok"):
        raise ValueError(
            prediction.get("message")
            or prediction.get("error")
            or "V5預測失敗"
        )

    session["last_prediction"] = prediction
    return store.upsert_session(
        user_id,
        session,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.api_route(
    "/",
    methods=["GET", "HEAD"],
)
def root() -> PlainTextResponse:
    return PlainTextResponse("OK")


@app.api_route(
    "/health",
    methods=["GET", "HEAD"],
)
def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "version": "v5-independent-point-line",
            "engine": "V5_INDEPENDENT_POINT_PF_LINE",
        }
    )


@app.post("/webhook")
async def webhook(
    request: Request,
) -> JSONResponse:
    body = await request.body()

    if not verify(
        body,
        request.headers.get("X-Line-Signature"),
    ):
        return JSONResponse(
            {"ok": False},
            status_code=401,
        )

    payload = json.loads(
        body.decode() or "{}"
    )

    for event in payload.get("events", []):
        token = event.get("replyToken", "")
        source = event.get("source") or {}
        user_id = (
            source.get("userId")
            or source.get("groupId")
            or "anonymous"
        )

        try:
            if event.get("type") == "follow":
                reply(
                    token,
                    [venue_panel(user_id)],
                )
                continue

            if (
                event.get("type") == "message"
                and (event.get("message") or {}).get("type") == "text"
            ):
                text = str(
                    (event.get("message") or {}).get("text") or ""
                ).strip()

                if text in ALL_CODES:
                    activate(user_id, text)
                    reply(
                        token,
                        [
                            text_message("✅ 開通成功"),
                            venue_panel(user_id),
                        ],
                    )
                    continue

                if text in {
                    "開始",
                    "開始分析",
                    "選館",
                }:
                    reply(
                        token,
                        [venue_panel(user_id)],
                    )
                    continue

                session = (
                    store.get_session(user_id)
                    or store.new_session(user_id)
                )

                if (
                    session.get("venue")
                    and not session.get("room")
                ):
                    session["room"] = text
                    session = store.upsert_session(
                        user_id,
                        session,
                    )
                    reply(
                        token,
                        [ready_panel(user_id, session)],
                    )
                    continue

                observation = parse_chat_point_observation(text)
                if observation:
                    if not session.get("venue"):
                        reply(
                            token,
                            [
                                text_message(
                                    "請先輸入「開始分析」並選擇遊戲館。"
                                )
                            ],
                        )
                        continue

                    if not session.get("room"):
                        reply(
                            token,
                            [
                                text_message(
                                    "請先輸入房間名稱或桌號。"
                                )
                            ],
                        )
                        continue

                    try:
                        updated = add_points_and_predict(
                            user_id,
                            observation,
                        )
                        reply(
                            token,
                            [result_panel(user_id, updated)],
                        )
                    except PermissionError:
                        reply(
                            token,
                            [expired_panel()],
                        )
                    continue

                if text in {
                    "預測",
                    "AI",
                    "開始AI判斷",
                }:
                    try:
                        predicted = predict_session(user_id)
                        reply(
                            token,
                            [result_panel(user_id, predicted)],
                        )
                    except PermissionError:
                        reply(
                            token,
                            [expired_panel()],
                        )
                    except ValueError as exception:
                        reply(
                            token,
                            [text_message(str(exception))],
                        )
                    continue

                if text in {
                    "結束",
                    "結束分析",
                }:
                    store.end_session(user_id)
                    reply(
                        token,
                        [ended_panel()],
                    )
                    continue

                reply(
                    token,
                    [
                        text_message(
                            "請輸入「開始分析」，"
                            "或直接輸入兩位數，例如：65。"
                        )
                    ],
                )
                continue

            if event.get("type") == "postback":
                query = {
                    key: values[0]
                    for key, values in urllib.parse.parse_qs(
                        (event.get("postback") or {}).get(
                            "data",
                            "",
                        )
                    ).items()
                }
                action_name = query.get("action")

                if action_name == "venue":
                    session = (
                        store.get_session(user_id)
                        or store.new_session(user_id)
                    )
                    session.update(
                        {
                            "venue": query.get("venue", ""),
                            "room": "",
                            "shoe_id": "",
                            "observations": [],
                            "last_prediction": None,
                        }
                    )
                    store.upsert_session(
                        user_id,
                        session,
                    )
                    reply(
                        token,
                        [
                            room_panel(
                                query.get("venue", "")
                            )
                        ],
                    )

                elif action_name == "end":
                    store.end_session(user_id)
                    reply(
                        token,
                        [ended_panel()],
                    )

        except Exception as exception:
            traceback.print_exc()
            reply(
                token,
                [
                    text_message(
                        f"操作失敗：{exception}"
                    )
                ],
            )

    return JSONResponse({"ok": True})


@app.post("/callback")
async def callback(
    request: Request,
) -> JSONResponse:
    return await webhook(request)
