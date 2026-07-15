"""Atomic JSON storage for per-LINE-user state and optional access control."""
from __future__ import annotations

import copy
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
SESSION_DATA_FILE = Path(os.getenv("SESSION_DATA_FILE", str(BASE_DIR / "data" / "sessions.json")))
TRIAL_MINUTES = max(0, int(os.getenv("TRIAL_MINUTES", "30") or "30"))
ACTIVATION_CODES = {
    item.strip() for item in os.getenv("ACTIVATION_CODES", "").split(",") if item.strip()
}
_LOCK = threading.RLock()


def _now() -> int:
    return int(time.time())


def _stats() -> Dict[str, int]:
    return {
        "wins": 0,
        "losses": 0,
        "ties_skipped": 0,
        "current_win_streak": 0,
        "current_loss_streak": 0,
        "max_win_streak": 0,
        "max_loss_streak": 0,
    }


def _default_session(user_id: str) -> Dict[str, Any]:
    return {
        "user_id": user_id,
        "status": "尚未開始",
        "venue": "",
        "room": "",
        "shoe_id": uuid.uuid4().hex[:12],
        "point_history": [],
        "pending_prediction": {},
        "last_settlement": {},
        "stats": _stats(),
        "trial_started_at": 0,
        "access_until": 0,
        "permanent_access": False,
        "created_at": _now(),
        "updated_at": _now(),
    }


def _load_all() -> Dict[str, Dict[str, Any]]:
    try:
        if not SESSION_DATA_FILE.exists():
            return {}
        with SESSION_DATA_FILE.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _save_all(data: Dict[str, Dict[str, Any]]) -> None:
    SESSION_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SESSION_DATA_FILE.with_suffix(SESSION_DATA_FILE.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    tmp.replace(SESSION_DATA_FILE)


def get_session(user_id: str) -> Dict[str, Any]:
    uid = str(user_id or "").strip()
    if not uid:
        raise ValueError("user_id is required")
    with _LOCK:
        data = _load_all()
        session = data.get(uid)
        if not isinstance(session, dict):
            session = _default_session(uid)
            data[uid] = session
            _save_all(data)
        return copy.deepcopy(session)


def upsert_session(user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    uid = str(user_id or "").strip()
    if not uid:
        raise ValueError("user_id is required")
    with _LOCK:
        data = _load_all()
        session = data.get(uid) if isinstance(data.get(uid), dict) else _default_session(uid)
        session.update(copy.deepcopy(updates or {}))
        session["user_id"] = uid
        session["updated_at"] = _now()
        data[uid] = session
        _save_all(data)
        return copy.deepcopy(session)


def start_session(user_id: str, venue: str = "", room: str = "") -> Dict[str, Any]:
    session = get_session(user_id)
    session.update({
        "status": "分析中",
        "venue": str(venue or session.get("venue") or ""),
        "room": str(room or session.get("room") or ""),
    })
    return upsert_session(user_id, session)


def end_session(user_id: str) -> Dict[str, Any]:
    return upsert_session(user_id, {"status": "已結束", "pending_prediction": {}})


def reset_session(user_id: str) -> Dict[str, Any]:
    old = get_session(user_id)
    fresh = _default_session(user_id)
    fresh.update({
        "status": "分析中",
        "venue": old.get("venue", ""),
        "room": old.get("room", ""),
        "permanent_access": bool(old.get("permanent_access")),
        "trial_started_at": int(old.get("trial_started_at", 0) or 0),
        "access_until": int(old.get("access_until", 0) or 0),
    })
    return upsert_session(user_id, fresh)


def access_status(user_id: str, start_trial: bool = False) -> Dict[str, Any]:
    session = get_session(user_id)
    now = _now()
    if bool(session.get("permanent_access")):
        return {"allowed": True, "type": "permanent", "seconds_left": None}
    trial_started = int(session.get("trial_started_at", 0) or 0)
    access_until = int(session.get("access_until", 0) or 0)
    if start_trial and trial_started <= 0 and TRIAL_MINUTES > 0:
        trial_started = now
        access_until = now + TRIAL_MINUTES * 60
        session = upsert_session(user_id, {
            "trial_started_at": trial_started,
            "access_until": access_until,
        })
    if TRIAL_MINUTES <= 0:
        return {"allowed": True, "type": "open", "seconds_left": None}
    allowed = access_until > now
    return {
        "allowed": allowed,
        "type": "trial" if allowed else "expired",
        "seconds_left": max(0, access_until - now),
        "trial_started_at": trial_started,
        "access_until": access_until,
    }


def activate(user_id: str, code: str) -> bool:
    value = str(code or "").strip()
    if not value or value not in ACTIVATION_CODES:
        return False
    upsert_session(user_id, {"permanent_access": True, "status": "分析中"})
    return True


def _outcome_from_point(point: str) -> str:
    text = str(point or "").strip().upper()
    if len(text) < 2 or not text[:2].isdigit():
        raise ValueError("invalid point")
    player, banker = int(text[0]), int(text[1])
    return "B" if banker > player else "P" if player > banker else "T"


def record_point_and_settle(user_id: str, point: str) -> Dict[str, Any]:
    """Settle the previous recommendation with this hand, then store this point.

    The history is for user-facing reporting only. predictor.py still receives only
    the newest point, so no historical signal enters the model.
    """
    session = get_session(user_id)
    actual = _outcome_from_point(point)
    pending = dict(session.get("pending_prediction") or {})
    stats = dict(session.get("stats") or _stats())
    settlement: Dict[str, Any] = {}
    recommend = str(pending.get("recommend") or "").upper()
    if recommend in {"B", "P"}:
        if actual == "T":
            stats["ties_skipped"] = int(stats.get("ties_skipped", 0)) + 1
            settlement = {"actual": actual, "recommend": recommend, "verdict": "TIE_SKIPPED"}
        elif actual == recommend:
            stats["wins"] = int(stats.get("wins", 0)) + 1
            stats["current_win_streak"] = int(stats.get("current_win_streak", 0)) + 1
            stats["current_loss_streak"] = 0
            stats["max_win_streak"] = max(int(stats.get("max_win_streak", 0)), stats["current_win_streak"])
            settlement = {"actual": actual, "recommend": recommend, "verdict": "HIT"}
        else:
            stats["losses"] = int(stats.get("losses", 0)) + 1
            stats["current_loss_streak"] = int(stats.get("current_loss_streak", 0)) + 1
            stats["current_win_streak"] = 0
            stats["max_loss_streak"] = max(int(stats.get("max_loss_streak", 0)), stats["current_loss_streak"])
            settlement = {"actual": actual, "recommend": recommend, "verdict": "LOSS"}
    history = list(session.get("point_history") or [])
    history.append(str(point).strip().upper())
    history = history[-300:]
    return upsert_session(user_id, {
        "status": "分析中",
        "point_history": history,
        "last_settlement": settlement,
        "pending_prediction": {},
        "stats": stats,
    })


def save_prediction(user_id: str, prediction: Dict[str, Any]) -> Dict[str, Any]:
    return upsert_session(user_id, {
        "pending_prediction": copy.deepcopy(prediction),
        "status": "等待下一局點數",
    })


def list_sessions() -> List[Dict[str, Any]]:
    with _LOCK:
        return [copy.deepcopy(x) for x in _load_all().values() if isinstance(x, dict)]
