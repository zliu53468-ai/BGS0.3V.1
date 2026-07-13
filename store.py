"""Persistent sessions for point-input baccarat particle filtering."""
from __future__ import annotations
import json, os, threading, time, uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
SESSION_DATA_FILE = Path(os.getenv('SESSION_DATA_FILE', str(BASE_DIR / 'data' / 'sessions.json')))
_LOCK = threading.RLock()


def _default(user_id: str) -> Dict[str, Any]:
    return {'user_id': user_id, 'venue': '', 'room': '', 'shoe_id': uuid.uuid4().hex[:12], 'observations': [], 'status': '尚未開始', 'last_prediction': {}, 'updated_at': int(time.time())}


def _load() -> Dict[str, Any]:
    try:
        if not SESSION_DATA_FILE.exists(): return {}
        with SESSION_DATA_FILE.open('r', encoding='utf-8') as f: data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception: return {}


def _save(data: Dict[str, Any]) -> None:
    SESSION_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SESSION_DATA_FILE.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(SESSION_DATA_FILE)


def get_session(user_id: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        value = _load().get(user_id)
        return dict(value) if isinstance(value, dict) else None


def upsert_session(user_id: str, session: Dict[str, Any]) -> Dict[str, Any]:
    with _LOCK:
        data = _load(); current = data.get(user_id) or _default(user_id)
        current.update(dict(session or {})); current['updated_at'] = int(time.time()); data[user_id] = current; _save(data); return dict(current)


def new_session(user_id: str, venue: str = '', room: str = '', shoe_id: str = '') -> Dict[str, Any]:
    value = _default(user_id); value.update({'venue': venue, 'room': room, 'shoe_id': shoe_id or uuid.uuid4().hex[:12], 'status': '分析中'}); return upsert_session(user_id, value)


def add_point_observation(user_id: str, player: int, banker: int) -> Dict[str, Any]:
    session = get_session(user_id) or new_session(user_id)
    observations = list(session.get('observations') or [])
    p, b = int(player) % 10, int(banker) % 10
    observations.append({'player': p, 'banker': b, 'outcome': 'B' if b > p else 'P' if p > b else 'T'})
    session['observations'] = observations; session['status'] = '已輸入點數'; return upsert_session(user_id, session)


def undo_round(user_id: str) -> Dict[str, Any]:
    session = get_session(user_id) or new_session(user_id); obs = list(session.get('observations') or [])
    if obs: obs.pop()
    session['observations'] = obs; return upsert_session(user_id, session)


def clear_history(user_id: str) -> Dict[str, Any]:
    session = get_session(user_id) or new_session(user_id); session.update({'shoe_id': uuid.uuid4().hex[:12], 'observations': [], 'last_prediction': {}, 'status': '新靴'}); return upsert_session(user_id, session)


def end_session(user_id: str) -> Dict[str, Any]:
    session = get_session(user_id) or new_session(user_id); session['status'] = '已結束'; return upsert_session(user_id, session)


def list_sessions() -> List[Dict[str, Any]]:
    return [dict(v) for v in _load().values() if isinstance(v, dict)]
