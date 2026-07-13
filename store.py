"""Persistent per-LINE-user session storage.

The API intentionally keeps the names used by older app.py versions:
get_session, upsert_session, new_session, add_round, undo_round,
clear_history, end_session, list_sessions.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
SESSION_DATA_FILE = Path(
    os.getenv("SESSION_DATA_FILE", str(BASE_DIR / "data" / "sessions.json"))
)
_LOCK = threading.RLock()
_ALLOWED = {"B", "P", "T"}


def _now() -> int:
    return int(time.time())


def _default_session(user_id: str) -> Dict[str, Any]:
    return {
        "user_id": user_id,
        "venue": "",
        "room": "",
        "shoe_id": uuid.uuid4().hex[:12],
        "history": [],
        "status": "尚未開始",
        "last_prediction": {},
        "stats": {
            "predictions": 0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "current_win_streak": 0,
            "current_loss_streak": 0,
            "max_win_streak": 0,
            "max_loss_streak": 0,
        },
        "created_at": _now(),
        "updated_at": _now(),
    }


def _load_all() -> Dict[str, Dict[str, Any]]:
    try:
        if not SESSION_DATA_FILE.exists():
            return {}
        with SESSION_DATA_FILE.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _save_all(data: Dict[str, Dict[str, Any]]) -> None:
    SESSION_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SESSION_DATA_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    tmp.replace(SESSION_DATA_FILE)


def get_session(user_id: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        session = _load_all().get(str(user_id))
        return dict(session) if isinstance(session, dict) else None


def upsert_session(user_id: str, session: Dict[str, Any]) -> Dict[str, Any]:
    uid = str(user_id or "").strip()
    if not uid:
        raise ValueError("user_id is required")
    with _LOCK:
        data = _load_all()
        current = data.get(uid) or _default_session(uid)
        merged = {**current, **dict(session or {})}
        merged["user_id"] = uid
        merged["history"] = [
            str(x).upper() for x in merged.get("history", []) if str(x).upper() in _ALLOWED
        ]
        merged["updated_at"] = _now()
        data[uid] = merged
        _save_all(data)
        return dict(merged)


def new_session(
    user_id: str,
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
) -> Dict[str, Any]:
    session = _default_session(str(user_id))
    session.update(
        {
            "venue": str(venue or ""),
            "room": str(room or ""),
            "shoe_id": str(shoe_id or uuid.uuid4().hex[:12]),
            "status": "分析中",
        }
    )
    return upsert_session(user_id, session)


def _settle_previous_prediction(session: Dict[str, Any], result: str) -> None:
    pred = session.get("last_prediction") or {}
    recommend = str(pred.get("recommend") or "").upper()
    if recommend not in {"B", "P", "T"}:
        return
    stats = dict(session.get("stats") or {})
    stats.setdefault("predictions", 0)
    stats.setdefault("wins", 0)
    stats.setdefault("losses", 0)
    stats.setdefault("ties", 0)
    stats.setdefault("current_win_streak", 0)
    stats.setdefault("current_loss_streak", 0)
    stats.setdefault("max_win_streak", 0)
    stats.setdefault("max_loss_streak", 0)
    stats["predictions"] += 1
    if result == "T" and recommend != "T":
        stats["ties"] += 1
    elif result == recommend:
        stats["wins"] += 1
        stats["current_win_streak"] += 1
        stats["current_loss_streak"] = 0
        stats["max_win_streak"] = max(
            stats["max_win_streak"], stats["current_win_streak"]
        )
    else:
        stats["losses"] += 1
        stats["current_loss_streak"] += 1
        stats["current_win_streak"] = 0
        stats["max_loss_streak"] = max(
            stats["max_loss_streak"], stats["current_loss_streak"]
        )
    session["stats"] = stats


def add_round(user_id: str, result: str) -> Dict[str, Any]:
    code = str(result or "").strip().upper()
    if code not in _ALLOWED:
        raise ValueError("result must be B, P or T")
    with _LOCK:
        session = get_session(user_id) or new_session(user_id)
        _settle_previous_prediction(session, code)
        history = list(session.get("history") or [])
        history.append(code)
        session["history"] = history
        session["status"] = "已回報結果，準備下一局"
        return upsert_session(user_id, session)


def undo_round(user_id: str) -> Dict[str, Any]:
    session = get_session(user_id) or new_session(user_id)
    history = list(session.get("history") or [])
    if history:
        history.pop()
    session["history"] = history
    session["status"] = "已刪除上一局"
    return upsert_session(user_id, session)


def clear_history(user_id: str) -> Dict[str, Any]:
    session = get_session(user_id) or new_session(user_id)
    session.update(
        {
            "shoe_id": uuid.uuid4().hex[:12],
            "history": [],
            "last_prediction": {},
            "status": "新靴",
        }
    )
    return upsert_session(user_id, session)


def end_session(user_id: str) -> Dict[str, Any]:
    session = get_session(user_id) or new_session(user_id)
    session["status"] = "已結束"
    return upsert_session(user_id, session)


def list_sessions() -> List[Dict[str, Any]]:
    with _LOCK:
        return [dict(v) for v in _load_all().values() if isinstance(v, dict)]
