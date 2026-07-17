"""LINE baccarat bot for V5.3 factual-shoe-context HYBRID prediction.

Supported inputs
----------------
65          = Player 6 / Banker 5
65D         = both sides drew
65D@38      = both sides drew, physical shoe hand 38
65D@38 P:2,1,3 B:4,0,1
            = exact cards for this hand

Every request creates fresh particles. Only factual shoe context is persisted:
hand number and exact cards explicitly entered by the user. Road, streak,
previous recommendation and settlement results are never model features.
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
APP_REQUIRE_HAND_NUMBER = _env_bool(
    "APP_REQUIRE_HAND_NUMBER",
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
MODEL_PARTICLES = _env_int("PF_PARTICLES", 384, 64, 2000)
MODEL_REPLICAS = _env_int("PF_REPLICAS", 5, 3, 11)
MODEL_HYBRID_MODE = os.getenv(
    "PF_HYBRID_MODE",
    "hybrid",
).strip().lower()
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
    title="Baccarat V5.3 Factual Shoe HYBRID Bot",
    version="5.3.0",
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


def _parse_card_values(text: str) -> List[int]:
    values = [
        item
        for item in re.split(r"\s*[,/]\s*", str(text or "").strip())
        if item != ""
    ]
    if len(values) not in {2, 3} or any(
        not re.fullmatch(r"[0-9]", item)
        for item in values
    ):
        raise ValueError("牌值必須是2或3張，例如 P:2,1,3。")
    return [int(item) for item in values]


def _extract_known_cards(text: str) -> Dict[str, List[int]]:
    tail = str(text or "").strip().upper()
    if not tail:
        return {}

    patterns = {
        "player": r"(?:P|PLAYER|閒|闲)\s*[:=]\s*([0-9](?:\s*[,/]\s*[0-9]){1,2})",
        "banker": r"(?:B|BANKER|莊|庄)\s*[:=]\s*([0-9](?:\s*[,/]\s*[0-9]){1,2})",
    }
    cards: Dict[str, List[int]] = {}
    for side, pattern in patterns.items():
        match = re.search(pattern, tail, flags=re.IGNORECASE)
        if match:
            cards[side] = _parse_card_values(match.group(1))

    if cards and set(cards) != {"player", "banker"}:
        raise ValueError("輸入實際牌值時，閒家與莊家牌值都必須提供。")
    if not cards:
        raise ValueError(
            "牌值格式錯誤，請使用：65D@38 P:2,1,3 B:4,0,1"
        )
    return cards


def _path_from_cards(cards: Mapping[str, Sequence[int]]) -> str:
    player_len = len(cards.get("player", []))
    banker_len = len(cards.get("banker", []))
    return {
        (2, 2): "N",
        (3, 2): "P",
        (2, 3): "B",
        (3, 3): "D",
    }.get((player_len, banker_len), "")


def parse_chat_point_observation(
    text: str,
) -> Optional[Dict[str, Any]]:
    """Parse points, optional draw path, hand number and exact cards."""
    value = str(text or "").strip().upper()
    compact = re.fullmatch(
        r"([0-9])([0-9])([NPBD])?(?:@([0-9]{1,3}))?(?:\s+(.+))?",
        value,
    )
    if compact:
        player = int(compact.group(1))
        banker = int(compact.group(2))
        suffix = compact.group(3) or ""
        hand_number = int(compact.group(4) or 0)
        if hand_number > 120:
            raise ValueError("牌靴局數請輸入1到120。")

        cards = _extract_known_cards(compact.group(5) or "")
        if cards:
            if sum(cards["player"]) % 10 != player:
                raise ValueError("閒家實際牌值加總與最終點數不一致。")
            if sum(cards["banker"]) % 10 != banker:
                raise ValueError("莊家實際牌值加總與最終點數不一致。")
            inferred_suffix = _path_from_cards(cards)
            if suffix and inferred_suffix and suffix != inferred_suffix:
                raise ValueError("N/P/B/D與實際牌張數不一致。")
            suffix = suffix or inferred_suffix

        return {
            "player": player,
            "banker": banker,
            "suffix": suffix,
            "path_suffix": suffix,
            "path": {"N": 0, "P": 1, "B": 2, "D": 3}.get(suffix),
            "hand_number": hand_number,
            "known_cards": cards,
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
        "hand_number": int(parsed.get("hand_number") or 0),
        "known_cards": dict(parsed.get("known_cards") or {}),
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
        "V5.3 HYBRID 可開始分析",
        (
            f"館別：{session.get('venue') or '-'}\n"
            f"房間：{session.get('room') or '-'}\n"
            f"UID權限：{user_status['label']}"
            f"｜剩餘：{remaining_text(user_status.get('remaining'))}\n\n"
            "基本輸入：65\n"
            "補牌＋局數：65D@38\n"
            "完整牌值：65D@38 P:2,1,3 B:4,0,1\n\n"
            "N雙方不補｜P僅閒補｜B僅莊補｜D雙方補\n"
            "若從牌靴中途開始，請務必加上 @目前局數。\n"
            "只有每局都輸入完整牌值，才會啟用精確剩餘牌組。\n\n"
            f"模型：HYBRID｜{MODEL_PARTICLES}粒子 × "
            f"{MODEL_REPLICAS}副本。"
        ),
        [
            action("新牌靴", "new_shoe"),
            action("結束分析", "end"),
        ],
    )


def _decision_source_text(value: Any) -> str:
    source = str(value or "")
    names = {
        "VALIDATED_MODEL": "驗證模型",
        "LOW_CONFIDENCE_BALANCED": "低信心比較",
        "UNVALIDATED_COMPARISON": "未驗證比較",
    }
    return names.get(source, source or "V5.3 HYBRID模型")


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
    hybrid_info = prediction.get("hybrid") or {}
    hybrid_weights = hybrid_info.get("weights") or {}
    shoe_context = prediction.get("shoe_context") or {}

    particles = _safe_int(
        particle_info.get("particles_per_replica"),
        MODEL_PARTICLES,
    )
    replicas = _safe_int(
        particle_info.get(
            "replicas",
            prediction.get("replica_count"),
        ),
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
        str(prediction.get("stability") or "-"),
    )

    hand_number = _safe_int(
        shoe_context.get("hand_number"),
        _safe_int(session.get("hand_number")),
    )
    exact_state = bool(
        shoe_context.get("exact_state_enabled")
        or hybrid_info.get("exact_state_enabled")
    )
    known_cards = prediction.get("known_cards") or {}
    if exact_state:
        shoe_mode = "完整牌值追蹤"
    elif known_cards:
        shoe_mode = "本局牌值＋局數"
    elif hand_number > 0:
        shoe_mode = "牌靴局數先驗"
    else:
        shoe_mode = "早中晚混合先驗"

    diagnostics = ""
    if APP_SHOW_MODEL_DIAGNOSTICS:
        diagnostics = (
            f"\n模型：{particles}粒子 × {replicas}副本"
            f"\nHYBRID閘門："
            f"{_safe_float(hybrid_info.get('gate')) * 100.0:.0f}%"
            f"｜PF權重："
            f"{_safe_float(hybrid_weights.get('particle')) * 100.0:.0f}%"
            f"\n精確牌組權重："
            f"{_safe_float(hybrid_weights.get('exact_shoe_state')) * 100.0:.0f}%"
            f"｜基準收縮："
            f"{_safe_float(hybrid_weights.get('baseline')) * 100.0:.0f}%"
            f"\n驗證：{'通過' if validated else '未通過'}"
            f"｜穩定：{stability}"
            f"\nESS："
            f"{_safe_float(particle_info.get('average_effective_sample_size')):.0f}"
            f"｜路徑覆蓋："
            f"{_safe_float(path_info.get('coverage')) * 100.0:.0f}%"
        )

    body = (
        f"本牌靴第 {hand_number or '-'} 局"
        f"｜第 {len(observations)} 次分析\n"
        f"輸入：{prediction.get('conditioning_point', '-')}\n"
        f"補牌：{_draw_path_text(prediction)}\n"
        f"牌靴資訊：{shoe_mode}\n\n"
        f"莊 {_safe_float(prediction.get('banker_rate')):.1f}%\n"
        f"閒 {_safe_float(prediction.get('player_rate')):.1f}%\n"
        f"和 {_safe_float(prediction.get('tie_rate')):.1f}%\n\n"
        f"推薦：{prediction.get('recommend_text', '-')}\n"
        f"訊號：{prediction.get('signal_level', '-')}\n"
        f"來源：{source_text}"
        f"{diagnostics}\n\n"
        f"UID權限：{user_status['label']}"
        f"｜剩餘：{remaining_text(user_status.get('remaining'))}\n\n"
        "下一局可輸入 65、65D@39，"
        "或完整牌值格式。\n"
        "換靴時請輸入「新牌靴」。"
    )

    return flex(
        "V5.3 HYBRID 下一局分析",
        body,
        [
            action("新牌靴", "new_shoe"),
            action("結束分析", "end"),
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
    """Settle the prior result and run one fresh V5.3 HYBRID analysis."""
    ensure(user_id)

    acquired = _PREDICTION_SLOTS.acquire(
        timeout=APP_PREDICTION_QUEUE_TIMEOUT
    )
    if not acquired:
        raise RuntimeError(
            "目前HYBRID模型正在分析其他請求，請稍後重新輸入點數。"
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

        normalized_observation = {
            "player": player,
            "banker": banker,
            "path_suffix": suffix,
            "path": {
                "N": 0,
                "P": 1,
                "B": 2,
                "D": 3,
            }.get(suffix),
            "hand_number": int(
                observation.get("hand_number") or 0
            ),
            "known_cards": dict(
                observation.get("known_cards") or {}
            ),
        }

        point = f"{player}{banker}"
        session = store.record_point_and_settle(
            user_id,
            point,
            normalized_observation,
        )
        context = store.shoe_context(session)
        last_observation = dict(
            session.get("last_observation")
            or normalized_observation
        )
        last_observation.update(context)

        prediction = predict(
            [last_observation],
            venue=session.get("venue", ""),
            room=session.get("room", ""),
            shoe_id=session.get("shoe_id", ""),
            user_id=user_id,
            shoe_context=context,
        )

        if not prediction.get("ok"):
            raise ValueError(
                prediction.get("message")
                or prediction.get("error")
                or "V5.3 HYBRID預測失敗"
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
            "version": "v5.3.0-factual-shoe-hybrid-line",
            "engine": "V5_3_FACTUAL_SHOE_HYBRID_LINE",
            "hybrid_mode": MODEL_HYBRID_MODE,
            "particles": MODEL_PARTICLES,
            "particle_limit": 2000,
            "replicas": MODEL_REPLICAS,
            "input_formats": [
                "65",
                "65D",
                "65D@38",
                "65D@38 P:2,1,3 B:4,0,1",
            ],
            "require_draw_path": APP_REQUIRE_DRAW_PATH,
            "require_hand_number": APP_REQUIRE_HAND_NUMBER,
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
                    text in {"新牌靴", "換靴", "重置牌靴"}
                    and session.get("room")
                ):
                    session = store.reset_shoe(user_id)
                    reply(
                        token,
                        [
                            text_message("✅ 已建立新牌靴，局數與牌值追蹤已歸零。"),
                            ready_panel(user_id, session),
                        ],
                    )
                    continue

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

                    if (
                        APP_REQUIRE_HAND_NUMBER
                        and not observation.get("hand_number")
                    ):
                        reply(
                            token,
                            [
                                text_message(
                                    "目前設定必須輸入牌靴局數，例如：65D@38。"
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
                            "或輸入 65、65D@38；換靴請輸入「新牌靴」。"
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
                    session = store.reset_shoe(
                        user_id,
                        venue=query.get("venue", ""),
                        room="",
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

                elif action_name == "new_shoe":
                    session = store.reset_shoe(user_id)
                    reply(
                        token,
                        [
                            text_message("✅ 已建立新牌靴，局數與牌值追蹤已歸零。"),
                            ready_panel(user_id, session),
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
