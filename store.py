import json
import os
import threading
import time
from copy import deepcopy
from typing import Any, Dict, Optional

DATA_FILE = os.getenv("SESSION_DATA_FILE", "sessions.json")
_LOCK = threading.Lock()

DEFAULT_SESSION: Dict[str, Any] = {
    "user_id": "",
    "venue": "",
    "room": "",
    "shoe_id": "",
    "round_no": 1,
    "history": [],
    "last_prediction": None,
    "status": "待輸入",
    "created_at": 0,
    "updated_at": 0,
}


def _now() -> int:
    return int(time.time())


def _load_all() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(DATA_FILE):
        return {}
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def _save_all(data: Dict[str, Dict[str, Any]]) -> None:
    folder = os.path.dirname(DATA_FILE)
    if folder:
        os.makedirs(folder, exist_ok=True)
    tmp = f"{DATA_FILE}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, DATA_FILE)


def get_session(user_id: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        data = _load_all()
        session = data.get(user_id)
        return deepcopy(session) if session else None


def upsert_session(user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    with _LOCK:
        data = _load_all()
        session = deepcopy(DEFAULT_SESSION)
        session["user_id"] = user_id
        session["created_at"] = _now()
        if user_id in data and isinstance(data[user_id], dict):
            session.update(data[user_id])
        session.update(updates)
        session["user_id"] = user_id
        session["round_no"] = len(session.get("history", [])) + 1
        session["updated_at"] = _now()
        data[user_id] = session
        _save_all(data)
        return deepcopy(session)


def new_session(user_id: str, venue: str, room: str, shoe_id: str) -> Dict[str, Any]:
    return upsert_session(
        user_id,
        {
            "venue": venue,
            "room": room,
            "shoe_id": shoe_id,
            "round_no": 1,
            "history": [],
            "last_prediction": None,
            "status": "輸入中",
            "created_at": _now(),
        },
    )


def add_round(user_id: str, result: str) -> Dict[str, Any]:
    result = result.upper()
    if result not in {"B", "P", "T"}:
        raise ValueError("result must be B/P/T")
    session = get_session(user_id)
    if not session:
        session = deepcopy(DEFAULT_SESSION)
        session["user_id"] = user_id
        session["created_at"] = _now()
    history = list(session.get("history", []))
    history.append(result)
    session["history"] = history
    session["last_prediction"] = None
    session["status"] = "輸入中"
    return upsert_session(user_id, session)


def undo_round(user_id: str) -> Dict[str, Any]:
    session = get_session(user_id)
    if not session:
        return upsert_session(user_id, {})
    history = list(session.get("history", []))
    if history:
        history.pop()
    session["history"] = history
    session["last_prediction"] = None
    session["status"] = "輸入中"
    return upsert_session(user_id, session)


def clear_history(user_id: str) -> Dict[str, Any]:
    session = get_session(user_id)
    if not session:
        return upsert_session(user_id, {})
    session["history"] = []
    session["last_prediction"] = None
    session["status"] = "輸入中"
    return upsert_session(user_id, session)


def end_session(user_id: str) -> Dict[str, Any]:
    session = get_session(user_id)
    if not session:
        return upsert_session(user_id, {"status": "已結束"})
    session["status"] = "已結束"
    return upsert_session(user_id, session)
