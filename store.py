"""Atomic JSON storage for click-only virtual-shoe baccarat sessions."""
from __future__ import annotations

import copy
import json
import os
import secrets
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from particle_filter_points import counts_from_shoe, create_virtual_shoe


BASE_DIR = Path(__file__).resolve().parent
SESSION_DATA_FILE = Path(
    os.getenv("SESSION_DATA_FILE", str(BASE_DIR / "data" / "sessions.json"))
)
PF_DECKS = max(1, min(16, int(os.getenv("PF_DECKS", "8") or "8")))
TRIAL_MINUTES = max(0, int(os.getenv("TRIAL_MINUTES", "30") or "30"))
ACTIVATION_CODES = {
    item.strip()
    for item in os.getenv("ACTIVATION_CODES", "").split(",")
    if item.strip()
}
HISTORY_LIMIT = max(10, min(300, int(os.getenv("SESSION_HISTORY_LIMIT", "80") or "80")))
_LOCK = threading.RLock()


def _now() -> int:
    return int(time.time())


def _fresh_stats() -> Dict[str, int]:
    return {
        "wins": 0,
        "losses": 0,
        "ties_skipped": 0,
        "total_rounds": 0,
        "current_win_streak": 0,
        "current_loss_streak": 0,
        "max_win_streak": 0,
        "max_loss_streak": 0,
    }


def _new_cut_card(decks: int = PF_DECKS) -> int:
    # Leave roughly 60-85 cards behind in an eight-deck shoe. Scale for other
    # deck counts while keeping at least one full hand safely available.
    total_cards = 52 * max(1, decks)
    low = max(18, int(total_cards * 0.14))
    high = max(low + 1, int(total_cards * 0.21))
    return low + secrets.randbelow(max(1, high - low + 1))


def _new_shoe_state(decks: int = PF_DECKS) -> Dict[str, Any]:
    shoe = create_virtual_shoe(decks=decks)
    return {
        "shoe_id": uuid.uuid4().hex[:12],
        "virtual_shoe": shoe,
        "remaining_counts": counts_from_shoe(shoe),
        "cut_card_remaining": _new_cut_card(decks),
        "hand_number": 0,
        "shoe_started_at": _now(),
        "shoe_reset_count": 0,
    }


def _default_session(user_id: str) -> Dict[str, Any]:
    timestamp = _now()
    return {
        "user_id": user_id,
        "status": "尚未開始",
        "venue": "",
        "room": "1",
        "round_history": [],
        "analysis_history": [],
        "last_prediction": {},
        "pending_prediction": {},
        "last_virtual_hand": {},
        "stats": _fresh_stats(),
        "trial_started_at": 0,
        "access_until": 0,
        "permanent_access": False,
        "created_at": timestamp,
        "updated_at": timestamp,
        **_new_shoe_state(),
    }


def _load_all_unlocked() -> Dict[str, Dict[str, Any]]:
    try:
        if not SESSION_DATA_FILE.exists():
            return {}
        with SESSION_DATA_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_all_unlocked(data: Dict[str, Dict[str, Any]]) -> None:
    SESSION_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = SESSION_DATA_FILE.with_suffix(SESSION_DATA_FILE.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    temporary.replace(SESSION_DATA_FILE)


def _valid_shoe(value: Any) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(card, int) and 0 <= card <= 9 for card in value)
    )


def _migrate_session(session: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    defaults = _default_session(user_id)
    for key, value in defaults.items():
        if key not in session:
            session[key] = copy.deepcopy(value)

    if not _valid_shoe(session.get("virtual_shoe")) or len(session["virtual_shoe"]) < 6:
        preserved_reset_count = int(session.get("shoe_reset_count", 0) or 0)
        session.update(_new_shoe_state())
        session["shoe_reset_count"] = preserved_reset_count + 1

    session["remaining_counts"] = counts_from_shoe(session["virtual_shoe"])
    if not isinstance(session.get("round_history"), list):
        session["round_history"] = []
    if not isinstance(session.get("analysis_history"), list):
        session["analysis_history"] = []
    if not isinstance(session.get("stats"), dict):
        session["stats"] = _fresh_stats()
    else:
        merged_stats = _fresh_stats()
        merged_stats.update({
            key: int(value or 0)
            for key, value in session["stats"].items()
            if key in merged_stats
        })
        session["stats"] = merged_stats

    session["user_id"] = user_id
    session["room"] = str(session.get("room") or "1")
    session["hand_number"] = max(0, int(session.get("hand_number", 0) or 0))
    session["cut_card_remaining"] = max(
        6,
        int(session.get("cut_card_remaining", _new_cut_card()) or _new_cut_card()),
    )
    return session


def get_session(user_id: str) -> Dict[str, Any]:
    uid = str(user_id or "").strip()
    if not uid:
        raise ValueError("user_id is required")
    with _LOCK:
        data = _load_all_unlocked()
        raw = data.get(uid)
        session = _migrate_session(raw, uid) if isinstance(raw, dict) else _default_session(uid)
        data[uid] = session
        _save_all_unlocked(data)
        return copy.deepcopy(session)


def upsert_session(user_id: str, updates: Mapping[str, Any]) -> Dict[str, Any]:
    uid = str(user_id or "").strip()
    if not uid:
        raise ValueError("user_id is required")
    with _LOCK:
        data = _load_all_unlocked()
        raw = data.get(uid)
        session = _migrate_session(raw, uid) if isinstance(raw, dict) else _default_session(uid)
        session.update(copy.deepcopy(dict(updates or {})))
        session = _migrate_session(session, uid)
        session["updated_at"] = _now()
        data[uid] = session
        _save_all_unlocked(data)
        return copy.deepcopy(session)


def select_venue(user_id: str, venue: str, room: str = "1") -> Dict[str, Any]:
    uid = str(user_id or "").strip()
    if not uid:
        raise ValueError("user_id is required")
    with _LOCK:
        data = _load_all_unlocked()
        raw = data.get(uid)
        session = _migrate_session(raw, uid) if isinstance(raw, dict) else _default_session(uid)
        changed = str(session.get("venue") or "") != str(venue or "")
        session["venue"] = str(venue or "")
        session["room"] = str(room or "1")
        session["status"] = "分析中"
        if changed:
            preserved_reset_count = int(session.get("shoe_reset_count", 0) or 0)
            session.update(_new_shoe_state())
            session["shoe_reset_count"] = preserved_reset_count + 1
            session["round_history"] = []
            session["analysis_history"] = []
            session["last_prediction"] = {}
            session["last_virtual_hand"] = {}
            session["stats"] = _fresh_stats()
        session["updated_at"] = _now()
        data[uid] = session
        _save_all_unlocked(data)
        return copy.deepcopy(session)


def set_room(user_id: str, room: str) -> Dict[str, Any]:
    return upsert_session(user_id, {"room": str(room or "1")})


def start_session(user_id: str, venue: str = "", room: str = "1") -> Dict[str, Any]:
    if venue:
        return select_venue(user_id, venue, room)
    return upsert_session(user_id, {"status": "分析中", "room": str(room or "1")})


def end_session(user_id: str) -> Dict[str, Any]:
    return upsert_session(user_id, {"status": "已結束", "pending_prediction": {}})


def reset_shoe(
    user_id: str,
    *,
    venue: Optional[str] = None,
    room: Optional[str] = None,
    reset_stats: bool = False,
) -> Dict[str, Any]:
    uid = str(user_id or "").strip()
    if not uid:
        raise ValueError("user_id is required")
    with _LOCK:
        data = _load_all_unlocked()
        raw = data.get(uid)
        session = _migrate_session(raw, uid) if isinstance(raw, dict) else _default_session(uid)
        preserved_reset_count = int(session.get("shoe_reset_count", 0) or 0)
        session.update(_new_shoe_state())
        session["shoe_reset_count"] = preserved_reset_count + 1
        session["status"] = "分析中"
        if venue is not None:
            session["venue"] = str(venue or "")
        if room is not None:
            session["room"] = str(room or "1")
        session["round_history"] = []
        session["analysis_history"] = []
        session["last_prediction"] = {}
        session["pending_prediction"] = {}
        session["last_virtual_hand"] = {}
        if reset_stats:
            session["stats"] = _fresh_stats()
        session["updated_at"] = _now()
        data[uid] = session
        _save_all_unlocked(data)
        return copy.deepcopy(session)


def reset_session(user_id: str) -> Dict[str, Any]:
    old = get_session(user_id)
    fresh = _default_session(str(user_id))
    fresh.update(
        {
            "venue": old.get("venue", ""),
            "room": old.get("room", "1"),
            "permanent_access": bool(old.get("permanent_access")),
            "trial_started_at": int(old.get("trial_started_at", 0) or 0),
            "access_until": int(old.get("access_until", 0) or 0),
            "status": "分析中",
        }
    )
    return upsert_session(user_id, fresh)


def _update_stats(stats: Dict[str, int], verdict: str) -> Dict[str, int]:
    result = dict(_fresh_stats())
    result.update({key: int(value or 0) for key, value in stats.items() if key in result})
    result["total_rounds"] += 1
    if verdict == "HIT":
        result["wins"] += 1
        result["current_win_streak"] += 1
        result["current_loss_streak"] = 0
        result["max_win_streak"] = max(result["max_win_streak"], result["current_win_streak"])
    elif verdict == "MISS":
        result["losses"] += 1
        result["current_loss_streak"] += 1
        result["current_win_streak"] = 0
        result["max_loss_streak"] = max(result["max_loss_streak"], result["current_loss_streak"])
    else:
        result["ties_skipped"] += 1
    return result


def run_virtual_round(
    user_id: str,
    runner: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> Dict[str, Any]:
    """Run one prediction/deal/update cycle under an atomic lock."""
    uid = str(user_id or "").strip()
    if not uid:
        raise ValueError("user_id is required")

    with _LOCK:
        data = _load_all_unlocked()
        raw = data.get(uid)
        session = _migrate_session(raw, uid) if isinstance(raw, dict) else _default_session(uid)
        if not session.get("venue"):
            raise ValueError("請先選擇館別。")

        shoe_reset = False
        minimum_remaining = int(session.get("cut_card_remaining", 70) or 70)
        if len(session["virtual_shoe"]) <= minimum_remaining + 6:
            preserved_reset_count = int(session.get("shoe_reset_count", 0) or 0)
            session.update(_new_shoe_state())
            session["shoe_reset_count"] = preserved_reset_count + 1
            # A new shoe must clear sequence conditioning, but the visible
            # analysis log and cumulative statistics remain available.
            session["round_history"] = []
            session["last_prediction"] = {}
            session["last_virtual_hand"] = {}
            shoe_reset = True

        output = dict(runner(copy.deepcopy(session)))
        prediction = dict(output.get("prediction") or {})
        hand = dict(output.get("hand") or {})
        remaining_shoe = [int(card) for card in list(output.get("remaining_shoe") or [])]
        if not prediction.get("ok") or not hand or not _valid_shoe(remaining_shoe):
            raise ValueError("虛擬牌靴分析回傳格式錯誤。")

        session["virtual_shoe"] = remaining_shoe
        session["remaining_counts"] = counts_from_shoe(remaining_shoe)
        session["hand_number"] = int(session.get("hand_number", 0) or 0) + 1
        session["status"] = "分析中"
        session["last_prediction"] = prediction
        session["pending_prediction"] = prediction
        session["last_virtual_hand"] = hand
        session["stats"] = _update_stats(
            dict(session.get("stats") or {}),
            str(prediction.get("verdict") or "TIE_SKIPPED"),
        )

        round_record = {
            "round_number": session["hand_number"],
            "outcome": str(hand.get("outcome") or ""),
            "draw_path": str(hand.get("draw_path") or ""),
            "player_total": int(hand.get("player_total", 0) or 0),
            "banker_total": int(hand.get("banker_total", 0) or 0),
            "cards_used": int(hand.get("cards_used", 0) or 0),
            "created_at": _now(),
        }
        session["round_history"] = (
            list(session.get("round_history") or []) + [round_record]
        )[-HISTORY_LIMIT:]

        analysis_record = {
            "round_number": session["hand_number"],
            "created_at": _now(),
            "shoe_id": session.get("shoe_id"),
            "venue": session.get("venue"),
            "room": session.get("room"),
            "banker_rate": prediction.get("banker_rate"),
            "player_rate": prediction.get("player_rate"),
            "tie_rate": prediction.get("tie_rate"),
            "recommend": prediction.get("recommend"),
            "recommend_text": prediction.get("recommend_text"),
            "action": prediction.get("action"),
            "action_text": prediction.get("action_text"),
            "confidence_label": prediction.get("confidence_label"),
            "quality_score": prediction.get("quality_score"),
            "uncertainty": prediction.get("uncertainty"),
            "virtual_outcome": prediction.get("virtual_outcome"),
            "virtual_outcome_text": prediction.get("virtual_outcome_text"),
            "verdict": prediction.get("verdict"),
            "verdict_text": prediction.get("verdict_text"),
            "player_cards": hand.get("player_cards", []),
            "banker_cards": hand.get("banker_cards", []),
            "player_total": hand.get("player_total"),
            "banker_total": hand.get("banker_total"),
            "draw_path_text": hand.get("draw_path_text"),
            "remaining_cards_after": len(remaining_shoe),
            "shoe_reset": shoe_reset,
        }
        session["analysis_history"] = (
            list(session.get("analysis_history") or []) + [analysis_record]
        )[-HISTORY_LIMIT:]
        session["updated_at"] = _now()

        data[uid] = session
        _save_all_unlocked(data)
        return copy.deepcopy(session)


def save_prediction(user_id: str, prediction: Mapping[str, Any]) -> Dict[str, Any]:
    return upsert_session(
        user_id,
        {
            "last_prediction": dict(prediction or {}),
            "pending_prediction": dict(prediction or {}),
        },
    )


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
        upsert_session(
            user_id,
            {"trial_started_at": trial_started, "access_until": access_until},
        )

    if TRIAL_MINUTES <= 0:
        return {"allowed": True, "type": "open", "seconds_left": None}
    return {
        "allowed": access_until > now,
        "type": "trial" if access_until > now else "expired",
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


__all__ = [
    "access_status",
    "activate",
    "end_session",
    "get_session",
    "reset_session",
    "reset_shoe",
    "run_virtual_round",
    "save_prediction",
    "select_venue",
    "set_room",
    "start_session",
    "upsert_session",
]
