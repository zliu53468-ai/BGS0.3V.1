"""LINE point-input baccarat bot for V5.2.0 independent point prediction.

Operation flow
--------------
1. Start analysis, select venue, and enter the room.
2. Enter final points directly, e.g. 65 = Player 6 / Banker 5.
3. Optional draw-path suffix improves conditioning precision:
       N = neither side drew a third card
       P = Player only drew
       B = Banker only drew
       D = both sides drew
   Example: 65D.
4. The result is stored for settlement/statistics, while every prediction uses
   only the newest point observation and creates fresh 1000-2000 particles.

Older formats remain accepted: 閒6莊5, P6B5, 6,5.
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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(
    name: str,
    default: int,
    minimum: int = 0,
    maximum: Optional[int] = None,
) -> int:
    try:
        value = int(os.getenv(name, str(default)).strip())
    except Exception:
        value = default
    value = max(minimum, value)
    return min(maximum, value) if maximum is not None else value


APP_SHOW_MODEL_DIAGNOSTICS = _env_bool(
    "APP_SHOW_MODEL_DIAGNOSTICS",
    True,
)
APP_REQUIRE_DRAW_PATH = _env_bool(
    "APP_REQUIRE_DRAW_PATH",
    False,
)
APP_MAX_CONCURRENT_PREDICTIONS = _env_int(
    "APP_MAX_CONCURRENT_PREDICTIONS",
    1,
    1,
    4,
)
APP_PREDICTION_QUEUE_TIMEOUT = _env_int(
    "APP_PREDICTION_QUEUE_TIMEOUT",
    45,
    1,
    55,
)
MODEL_PARTICLES = _env_int("PF_PARTICLES", 1000, 64, 2000)
MODEL_REPLICAS = _env_int("PF_REPLICAS", 5, 3, 11)
_PREDICTION_SLOTS = threading.BoundedSemaphore(
    APP_MAX_CONCURRENT_PREDICTIONS
)

DRAW_PATH_SUFFIX_BY_INDEX = {0: "N", 1: "P", 2: "B", 3: "D"}
DRAW_PATH_TEXT = {
    "none": "雙方都沒補牌",
    "player_only": "只有閒家補牌",
    "banker_only": "只有莊家補牌",
    "both": "雙方都有補牌",
}

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
    title="Baccarat V5.2 Independent Point Bot",
    version="5.2.0",
)

# Reuse HTTPS connections to LINE instead of opening a new TLS connection for
# every reply. This changes transport only; all Flex panels and front-end text
# remain exactly as originally defined below.
_LINE_HTTP = requests.Session()
_LINE_ADAPTER = requests.adapters.HTTPAdapter(
    pool_connections=8,
    pool_maxsize=8,
    max_retries=0,
)
_LINE_HTTP.mount("https://", _LINE_ADAPTER)


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

    response = _LINE_HTTP.post(
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
) -> Optional[Dict[str, Any]]:
    """Parse final points and preserve the optional N/P/B/D draw path."""
    value = str(text or "").strip().upper()
    compact = re.fullmatch(r"([0-9])([0-9])([NPBD])?", value)
    if compact:
        suffix = compact.group(3) or ""
        return {
            "player": int(compact.group(1)),
            "banker": int(compact.group(2)),
            "suffix": suffix,
            "path_suffix": suffix,
            "path": {"N": 0, "P": 1, "B": 2, "D": 3}.get(suffix),
        }

    parsed = parse_point_observation(value)
    if not parsed:
        return None

    suffix = str(
        parsed.get("path_suffix")
        or parsed.get("suffix")
        or ""
    ).strip().upper()
    if suffix not in {"N", "P", "B", "D"}:
        try:
            suffix = DRAW_PATH_SUFFIX_BY_INDEX.get(
                int(parsed.get("path")),
                "",
            )
        except Exception:
            suffix = ""

    return {
        "player": int(parsed["player"]) % 10,
        "banker": int(parsed["banker"]) % 10,
        "suffix": suffix,
        "path_suffix": suffix,
        "path": {"N": 0, "P": 1, "B": 2, "D": 3}.get(suffix),
    }


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
            "請直接輸入最終點數：\n"
            "65＝閒6點、莊5點。\n"
            "知道補牌情況時可輸入 65N／65P／65B／65D。\n"
            "N雙方不補｜P僅閒補｜B僅莊補｜D雙方補。\n\n"
            f"目前模型：{MODEL_PARTICLES} 粒子 × {MODEL_REPLICAS} 副本。\n"
            "每次只使用最新點數，並重新建立獨立粒子。"
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
    names = {
        "VALIDATED_MODEL": "驗證模型",
        "LOW_CONFIDENCE_BALANCED": "低信心比較",
        "UNVALIDATED_COMPARISON": "未驗證比較",
    }
    return names.get(source, source or "V5.2模型")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _draw_path_text(prediction: Dict[str, Any]) -> str:
    known_path = prediction.get("known_draw_path")
    if known_path in DRAW_PATH_TEXT:
        return DRAW_PATH_TEXT[str(known_path)]
    return "未指定（模型分層估計）"


def _stored_prediction(session: Dict[str, Any]) -> Dict[str, Any]:
    """Read prediction data from the new store.py, with legacy fallback."""
    return (
        session.get("pending_prediction")
        or session.get("last_prediction")
        or {}
    )


def _stored_points(session: Dict[str, Any]) -> List[Any]:
    """Read point history from the new store.py, with legacy fallback."""
    return (
        session.get("point_history")
        or session.get("observations")
        or []
    )


def result_panel(
    user_id: str,
    session: Dict[str, Any],
) -> Dict[str, Any]:
    prediction = _stored_prediction(session)
    observations = _stored_points(session)
    user_status = status(user_id)
    source_text = _decision_source_text(
        prediction.get("decision_source")
    )
    particle_info = prediction.get("point_particle_filter") or {}
    path_info = prediction.get("draw_path_diagnostics") or {}
    particles = _safe_int(
        particle_info.get("particles_per_replica"),
        MODEL_PARTICLES,
    )
    replicas = _safe_int(
        particle_info.get("replicas", prediction.get("replica_count")),
        MODEL_REPLICAS,
    )
    validated = bool(prediction.get("validated_signal"))
    stability_names = {
        "STABLE": "穩定",
        "WATCH": "注意",
        "UNSTABLE": "不穩定",
    }
    stability = stability_names.get(
        str(prediction.get("stability") or ""),
        str(prediction.get("stability") or "-")
    )

    diagnostics = ""
    if APP_SHOW_MODEL_DIAGNOSTICS:
        diagnostics = (
            f"\n模型：{particles}粒子 × {replicas}副本"
            f"\n驗證：{'通過' if validated else '未通過'}｜穩定：{stability}"
            f"\nESS：{_safe_float(particle_info.get('average_effective_sample_size')):.0f}"
            f"｜路徑覆蓋：{_safe_float(path_info.get('coverage')) * 100.0:.0f}%"
        )

    body = (
        f"本次第 {len(observations)} 次獨立分析\n"
        f"輸入：{prediction.get('conditioning_point', '-')}\n"
        f"補牌：{_draw_path_text(prediction)}\n\n"
        f"莊 {_safe_float(prediction.get('banker_rate')):.1f}%\n"
        f"閒 {_safe_float(prediction.get('player_rate')):.1f}%\n"
        f"和 {_safe_float(prediction.get('tie_rate')):.1f}%\n\n"
        f"推薦：{prediction.get('recommend_text', '-')}\n"
        f"訊號：{prediction.get('signal_level', '-')}\n"
        f"來源：{source_text}"
        f"{diagnostics}\n\n"
        f"UID權限：{user_status['label']}"
        f"｜剩餘：{remaining_text(user_status.get('remaining'))}\n\n"
        "請輸入下一組點數，例如 65 或 65D。\n"
        "每次分析完全獨立，不沿用上一局粒子。"
    )

    return flex(
        "V5.2 下一局粒子模擬",
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
        "需要再次分析時，請點擊下方「開始分析」。",
        [
            action(
                "開始分析",
                "start",
            )
        ],
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
    session = store.get_session(user_id)
    ensure(user_id)

    if not _stored_prediction(session):
        raise ValueError("尚未輸入點數，請先輸入例如：65")

    return session


def add_points_and_predict(
    user_id: str,
    observation: Dict[str, Any],
) -> Dict[str, Any]:
    """Settle the prior result and run one fresh V5.2 particle analysis."""
    ensure(user_id)

    acquired = _PREDICTION_SLOTS.acquire(
        timeout=APP_PREDICTION_QUEUE_TIMEOUT
    )
    if not acquired:
        raise RuntimeError(
            "目前高粒子模型正在分析其他請求，請稍後重新輸入點數。"
        )

    try:
        player = int(observation["player"]) % 10
        banker = int(observation["banker"]) % 10
        suffix = str(
            observation.get("path_suffix")
            or observation.get("suffix")
            or ""
        ).strip().upper()
        if suffix not in {"N", "P", "B", "D"}:
            try:
                suffix = DRAW_PATH_SUFFIX_BY_INDEX.get(
                    int(observation.get("path")),
                    "",
                )
            except Exception:
                suffix = ""

        point = f"{player}{banker}"
        session = store.record_point_and_settle(
            user_id,
            point,
        )

        # predictor.parse_point_observation expects path_suffix for mapping input.
        # Passing it explicitly prevents 65N/65P/65B/65D from losing the path.
        model_observation = {
            "player": player,
            "banker": banker,
            "path_suffix": suffix,
        }
        prediction = predict(
            [model_observation],
            venue=session.get("venue", ""),
            room=session.get("room", ""),
            shoe_id=session.get("shoe_id", ""),
            user_id=user_id,
        )

        if not prediction.get("ok"):
            raise ValueError(
                prediction.get("message")
                or prediction.get("error")
                or "V5.2預測失敗"
            )

        return store.save_prediction(
            user_id,
            prediction,
        )
    finally:
        _PREDICTION_SLOTS.release()


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
            "version": "v5.2.0-independent-point-line",
            "engine": "V5_2_INDEPENDENT_POINT_PF_LINE",
            "particles": MODEL_PARTICLES,
            "particle_limit": 2000,
            "replicas": MODEL_REPLICAS,
            "draw_path_input": "N/P/B/D",
            "require_draw_path": APP_REQUIRE_DRAW_PATH,
            "max_concurrent_predictions": APP_MAX_CONCURRENT_PREDICTIONS,
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

                session = store.get_session(user_id)

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
                    if (
                        APP_REQUIRE_DRAW_PATH
                        and not observation.get("path_suffix")
                    ):
                        reply(
                            token,
                            [
                                text_message(
                                    "目前設定必須輸入補牌路徑：例如 65N、65P、65B 或 65D。"
                                )
                            ],
                        )
                        continue

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
                        updated = await asyncio.to_thread(
                            add_points_and_predict,
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
                            "或直接輸入點數，例如 65 或 65D。"
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
                    session = store.start_session(user_id)
                    session.update(
                        {
                            "venue": query.get("venue", ""),
                            "room": "",
                            "shoe_id": "",
                            "point_history": [],
                            "pending_prediction": {},
                            "last_settlement": {},
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

                elif action_name == "start":
                    reply(
                        token,
                        [venue_panel(user_id)],
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
