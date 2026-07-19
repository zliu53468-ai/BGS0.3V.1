"""LINE baccarat bot for independent point-and-draw-path prediction.

Recommended inputs
------------------
652 = Player 6 / Banker 5 / banker only drew
571 = Player 5 / Banker 7 / player only drew

The first two digits are the final Player/Banker points. The third digit is
the current-hand draw code: 1=P, 2=B, 3=D, 4=N. Letter forms such as 65B and
57P remain supported. Every request is predicted independently and does not
use a hand number, prior shoe state, previous points or previous predictions.
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


def _env_csv_ints(
    name: str,
    default: Sequence[int],
) -> Tuple[int, ...]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return tuple(int(item) for item in default)
    values: List[int] = []
    for part in raw.split(","):
        try:
            value = int(part.strip())
        except Exception:
            continue
        if value > 0 and value not in values:
            values.append(value)
    return tuple(values) if values else tuple(int(item) for item in default)


APP_SHOW_MODEL_DIAGNOSTICS = _env_bool(
    "APP_SHOW_MODEL_DIAGNOSTICS",
    True,
)
APP_REQUIRE_DRAW_PATH = _env_bool(
    "APP_REQUIRE_DRAW_PATH",
    False,
)
# Independent mode never requires or consumes a shoe hand number.
APP_REQUIRE_HAND_NUMBER = False
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
APP_BANKROLL_MIN = _env_int("APP_BANKROLL_MIN", 1000, 100, 100_000_000)
APP_BANKROLL_MAX = _env_int(
    "APP_BANKROLL_MAX",
    10_000_000,
    APP_BANKROLL_MIN,
    100_000_000,
)
APP_BANKROLL_PRESETS = tuple(
    value
    for value in _env_csv_ints(
        "APP_BANKROLL_PRESETS",
        (1000, 3000, 5000, 10000),
    )
    if APP_BANKROLL_MIN <= value <= APP_BANKROLL_MAX
)[:4]
BET_ROUND_UNIT = _env_int("BET_ROUND_UNIT", 100, 100, 100_000)
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
DRAW_CODE_TO_SUFFIX = {"1": "P", "2": "B", "3": "D", "4": "N"}
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
            "牌值格式錯誤，請使用：65D@38 P:2,1,3 B:4,0,1，"
            "或 653@38 P:2,1,3 B:4,0,1"
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
    """Parse current points and draw path; hand number is not required."""
    value = str(text or "").strip().upper()
    compact = re.fullmatch(
        r"([0-9])([0-9])([NPBD]|[1-4])?(?:@([0-9]{1,3}))?(?:\s+(.+))?",
        value,
    )
    if compact:
        player = int(compact.group(1))
        banker = int(compact.group(2))
        suffix = compact.group(3) or ""
        suffix = DRAW_CODE_TO_SUFFIX.get(suffix, suffix)
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

        raw_suffix = compact.group(3) or ""
        return {
            "player": player,
            "banker": banker,
            "suffix": suffix,
            "path_suffix": suffix,
            "path": {"N": 0, "P": 1, "B": 2, "D": 3}.get(suffix),
            "hand_number": 0,
            "known_cards": {},
            "input_text": f"{player}{banker}{raw_suffix}",
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

    player = int(parsed["player"]) % 10
    banker = int(parsed["banker"]) % 10
    return {
        "player": player,
        "banker": banker,
        "suffix": suffix,
        "path_suffix": suffix,
        "path": {"N": 0, "P": 1, "B": 2, "D": 3}.get(suffix),
        "hand_number": 0,
        "known_cards": {},
        "input_text": f"{player}{banker}{suffix}",
    }


def _parse_bankroll(text: Any) -> int:
    cleaned = str(text or "").strip().upper()
    cleaned = re.sub(r"^(?:本金|金額|資金)\s*[:：=]?\s*", "", cleaned)
    cleaned = cleaned.replace(",", "").replace("，", "")
    cleaned = cleaned.replace("NT$", "").replace("$", "").replace("元", "")
    cleaned = cleaned.strip()
    if not re.fullmatch(r"[0-9]+", cleaned):
        raise ValueError("請輸入純數字本金，例如 10000。")
    bankroll = int(cleaned)
    if bankroll < APP_BANKROLL_MIN or bankroll > APP_BANKROLL_MAX:
        raise ValueError(
            f"本金請輸入 {APP_BANKROLL_MIN:,}～{APP_BANKROLL_MAX:,} 元。"
        )
    return bankroll


def _format_money(value: Any) -> str:
    return f"{max(0, _safe_int(value)):,}"


def _round_bet_amount(
    bankroll: int,
    fraction: float,
    unit: int = BET_ROUND_UNIT,
) -> int:
    bankroll = max(0, int(bankroll))
    unit = max(100, int(unit))
    if bankroll <= 0 or fraction <= 0:
        return 0
    raw = bankroll * max(0.0, min(1.0, float(fraction)))
    # Conventional half-up rounding; unlike Python round(), exact halves do not
    # use bankers rounding.  The result therefore has no tens or ones digits.
    rounded = int((raw + unit / 2.0) // unit) * unit
    affordable = (bankroll // unit) * unit
    if affordable <= 0:
        return 0
    return max(unit, min(rounded, affordable))


def _save_bankroll(
    user_id: str,
    session: Dict[str, Any],
    bankroll: int,
) -> Dict[str, Any]:
    session = dict(session or {})
    session["bankroll"] = int(bankroll)
    session["initial_bankroll"] = int(bankroll)
    session["awaiting_bankroll"] = False
    return store.upsert_session(user_id, session)


def _restore_bankroll_after_reset(
    user_id: str,
    old_session: Mapping[str, Any],
    reset_session: Dict[str, Any],
) -> Dict[str, Any]:
    bankroll = _safe_int(old_session.get("bankroll"), 0)
    if bankroll <= 0:
        return reset_session
    reset_session = dict(reset_session or {})
    reset_session["bankroll"] = bankroll
    reset_session["initial_bankroll"] = _safe_int(
        old_session.get("initial_bankroll"),
        bankroll,
    )
    reset_session["awaiting_bankroll"] = False
    return store.upsert_session(user_id, reset_session)


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


def bankroll_panel(
    user_id: str,
    session: Dict[str, Any],
) -> Dict[str, Any]:
    buttons = [
        action(
            f"{value:,} 元",
            "bankroll",
            amount=str(value),
        )
        for value in APP_BANKROLL_PRESETS
    ]
    return flex(
        "請選擇本金",
        (
            f"館別：{session.get('venue') or '-'}\n"
            f"房間：{session.get('room') or '-'}\n\n"
            "請點選常用本金，或直接輸入自訂金額。\n"
            f"可輸入範圍：{APP_BANKROLL_MIN:,}～{APP_BANKROLL_MAX:,} 元。\n"
            "例如：10000 或 本金10000。"
        ),
        buttons or None,
    )


def ready_panel(
    user_id: str,
    session: Dict[str, Any],
) -> Dict[str, Any]:
    user_status = status(user_id)
    return flex(
        "AI預測 可開始分析",
        (
            f"館別：{session.get('venue') or '-'}\n"
            f"房間：{session.get('room') or '-'}\n"
            f"本金：{_format_money(session.get('bankroll'))} 元\n"
            f"UID權限：{user_status['label']}"
            f"｜剩餘：{remaining_text(user_status.get('remaining'))}\n\n"
            "直接輸入三位數即可開始預測：\n"
            "652＝閒6、莊5、只有莊家補牌\n"
            "571＝閒5、莊7、只有閒家補牌\n\n"
            "第三碼補牌代號：\n"
            "1＝只有閒家補牌\n"
            "2＝只有莊家補牌\n"
            "3＝雙方都補牌\n"
            "4＝雙方都不補牌\n\n"
            "字母格式也可使用：65B、57P、65D、67N。\n"
            "每一組點數都會完全獨立預測，不需要輸入局數，"
            "也不會沿用上一局資料。\n"
            "30分鐘試用會在首次成功輸入點數後開始計時。"
        ),
        [
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
    user_status = status(user_id)
    bankroll = _safe_int(
        prediction.get("bankroll", session.get("bankroll", 0)),
        0,
    )
    suggested_bet = _safe_int(
        prediction.get("suggested_bet_amount", 0),
        0,
    )
    bet_percentage = _safe_float(prediction.get("bet_percentage", 0.0))
    bet_level_text = str(prediction.get("bet_level_text") or "保守")
    bet_line = (
        f"建議金額：{_format_money(suggested_bet)} 元"
        f"（{bet_percentage:.1f}%｜{bet_level_text}）"
        if suggested_bet > 0
        else "建議金額：本局觀望"
    )

    body = (
        f"輸入：{prediction.get('conditioning_point', '-')}\n"
        f"補牌：{_draw_path_text(prediction)}\n\n"
        f"莊 {_safe_float(prediction.get('banker_rate')):.1f}%\n"
        f"閒 {_safe_float(prediction.get('player_rate')):.1f}%\n"
        f"和 {_safe_float(prediction.get('tie_rate')):.1f}%\n\n"
        f"推薦：{prediction.get('recommend_text', '-')}\n"
        f"本金：{_format_money(bankroll)} 元\n"
        f"{bet_line}\n\n"
        f"UID權限：{user_status['label']}"
        f"｜剩餘：{remaining_text(user_status.get('remaining'))}\n\n"
        "下一組直接輸入三位數，例如 652 或 571。\n"
        "1=僅閒補｜2=僅莊補｜3=雙方補｜4=雙方不補。\n"
        "每次分析完全獨立，不沿用上一局。"
    )

    return flex(
        "AI預測",
        body,
        [
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

    if not _stored_prediction(session):
        raise ValueError("尚未輸入點數，請先輸入例如：65")

    ensure(user_id)

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
            "hand_number": 0,
            "known_cards": {},
            "input_text": str(
                observation.get("input_text")
                or f"{player}{banker}{suffix}"
            ),
        }

        point = f"{player}{banker}"
        session = store.record_point_and_settle(
            user_id,
            point,
            normalized_observation,
        )
        # Do not load or merge shoe context.  Only the current point and
        # current draw path are sent to the predictor for this request.
        prediction = predict(
            [normalized_observation],
            venue=session.get("venue", ""),
            room=session.get("room", ""),
            shoe_id=session.get("shoe_id", ""),
            user_id=user_id,
            shoe_context=None,
        )
        prediction["conditioning_point"] = normalized_observation[
            "input_text"
        ]

        if not prediction.get("ok"):
            raise ValueError(
                prediction.get("message")
                or prediction.get("error")
                or "V5.3 HYBRID預測失敗"
            )

        bankroll = _safe_int(session.get("bankroll"), 0)
        if bankroll <= 0:
            raise ValueError("尚未設定本金，請先輸入本金金額。")
        bet_fraction = _safe_float(prediction.get("bet_fraction"), 0.0)
        bet_allowed = bool(prediction.get("bet_allowed", False))
        suggested_bet = (
            _round_bet_amount(bankroll, bet_fraction)
            if bet_allowed and prediction.get("recommend") != "O"
            else 0
        )
        prediction["bankroll"] = bankroll
        prediction["suggested_bet_amount"] = suggested_bet
        prediction["suggested_bet_round_unit"] = BET_ROUND_UNIT
        prediction["suggested_bet_display"] = (
            f"{suggested_bet:,} 元" if suggested_bet > 0 else "本局觀望"
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
                "652",
                "571",
                "653",
                "674",
                "65B",
                "57P",
                "65D",
                "67N",
            ],
            "require_draw_path": APP_REQUIRE_DRAW_PATH,
            "require_hand_number": APP_REQUIRE_HAND_NUMBER,
            "max_concurrent_predictions": APP_MAX_CONCURRENT_PREDICTIONS,
            "bankroll_min": APP_BANKROLL_MIN,
            "bankroll_max": APP_BANKROLL_MAX,
            "bankroll_presets": list(APP_BANKROLL_PRESETS),
            "bet_round_unit": BET_ROUND_UNIT,
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
                    old_session = dict(session)
                    session = store.reset_shoe(user_id)
                    session = _restore_bankroll_after_reset(
                        user_id,
                        old_session,
                        session,
                    )
                    reply(
                        token,
                        [
                            text_message("✅ 已清除本次分析紀錄。"),
                            ready_panel(user_id, session),
                        ],
                    )
                    continue

                if (
                    session.get("venue")
                    and not session.get("room")
                ):
                    session["room"] = text
                    session["bankroll"] = 0
                    session["initial_bankroll"] = 0
                    session["awaiting_bankroll"] = True
                    session = store.upsert_session(
                        user_id,
                        session,
                    )
                    reply(
                        token,
                        [bankroll_panel(user_id, session)],
                    )
                    continue

                if text in {"更改本金", "設定本金", "本金"} and session.get("room"):
                    session["awaiting_bankroll"] = True
                    session = store.upsert_session(user_id, session)
                    reply(token, [bankroll_panel(user_id, session)])
                    continue

                if session.get("room") and (
                    session.get("awaiting_bankroll")
                    or _safe_int(session.get("bankroll"), 0) <= 0
                ):
                    try:
                        bankroll = _parse_bankroll(text)
                        session = _save_bankroll(user_id, session, bankroll)
                        reply(
                            token,
                            [
                                text_message(
                                    f"✅ 本金已設定為 {bankroll:,} 元"
                                ),
                                ready_panel(user_id, session),
                            ],
                        )
                    except ValueError as exception:
                        reply(
                            token,
                            [
                                text_message(str(exception)),
                                bankroll_panel(user_id, session),
                            ],
                        )
                    continue

                bankroll_command = re.fullmatch(
                    r"(?:本金|金額|資金)\s*[:：=]?\s*(.+)",
                    text,
                    flags=re.IGNORECASE,
                )
                if bankroll_command and session.get("room"):
                    try:
                        bankroll = _parse_bankroll(bankroll_command.group(1))
                        session = _save_bankroll(user_id, session, bankroll)
                        reply(
                            token,
                            [
                                text_message(
                                    f"✅ 本金已更新為 {bankroll:,} 元"
                                ),
                                ready_panel(user_id, session),
                            ],
                        )
                    except ValueError as exception:
                        reply(token, [text_message(str(exception))])
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
                                    "請輸入三位數，例如652或571。"
                                    "前兩碼是閒、莊點數，第三碼為補牌代號："
                                    "1=僅閒補、2=僅莊補、3=雙方補、4=雙方不補。"
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

                    if _safe_int(session.get("bankroll"), 0) <= 0:
                        session["awaiting_bankroll"] = True
                        session = store.upsert_session(user_id, session)
                        reply(
                            token,
                            [bankroll_panel(user_id, session)],
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
                            "格式錯誤。請輸入「開始分析」，"
                            "或直接輸入三位數，例如652、571。"
                            "需要調整金額可輸入「更改本金」。"
                            "前兩碼是閒、莊點數；第三碼："
                            "1=僅閒補、2=僅莊補、3=雙方補、4=雙方不補。"
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

                elif action_name == "bankroll":
                    session = store.get_session(user_id)
                    bankroll = _parse_bankroll(query.get("amount", ""))
                    session = _save_bankroll(user_id, session, bankroll)
                    reply(
                        token,
                        [
                            text_message(
                                f"✅ 本金已設定為 {bankroll:,} 元"
                            ),
                            ready_panel(user_id, session),
                        ],
                    )

                elif action_name == "start":
                    reply(
                        token,
                        [venue_panel(user_id)],
                    )

                elif action_name == "new_shoe":
                    old_session = store.get_session(user_id)
                    session = store.reset_shoe(user_id)
                    session = _restore_bankroll_after_reset(
                        user_id,
                        old_session,
                        session,
                    )
                    reply(
                        token,
                        [
                            text_message("✅ 已清除本次分析紀錄。"),
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
