"""Atomic JSON storage for per-LINE-user baccarat shoe state and access data.

V5.3 keeps only factual shoe context:
- hand number
- optional N/P/B/D draw path
- exact cards explicitly entered by the user
- remaining card counts only while full tracking is continuous

It does not persist road, streak, Markov state, or previous model direction as an
input feature. Previous recommendations are stored only for settlement statistics.
"""
from __future__ import annotations

import copy
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

BASE_DIR = Path(__file__).resolve().parent
SESSION_DATA_FILE = Path(
    os.getenv("SESSION_DATA_FILE", str(BASE_DIR / "data" / "sessions.json"))
)
TRIAL_MINUTES = max(0, int(os.getenv("TRIAL_MINUTES", "30") or "30"))
PF_DECKS = max(1, min(16, int(os.getenv("PF_DECKS", "8") or "8")))
ACTIVATION_CODES = {
    item.strip()
    for item in os.getenv("ACTIVATION_CODES", "").split(",")
    if item.strip()
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


def _fresh_counts(decks: int = PF_DECKS) -> List[int]:
    # Baccarat card-value counts: 10/J/Q/K are all value 0.
    return [16 * decks] + [4 * decks] * 9


def _empty_shoe_state() -> Dict[str, Any]:
    return {
        "hand_number": 0,
        "known_card_counts": [0] * 10,
        "remaining_counts": _fresh_counts(),
        "tracked_card_hands": 0,
        "state_complete": True,
        "state_source": "NEW_SHOE",
        "shoe_observations": [],
        "last_observation": {},
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
        "last_prediction": {},
        "last_settlement": {},
        "stats": _stats(),
        "trial_started_at": 0,
        "access_until": 0,
        "permanent_access": False,
        **_empty_shoe_state(),
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


def _migrate_session(session: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Add V5.3 fields to an existing session without deleting user data."""
    defaults = _default_session(user_id)
    for key, value in defaults.items():
        if key not in session:
            session[key] = copy.deepcopy(value)

    counts = session.get("known_card_counts")
    if not isinstance(counts, list) or len(counts) != 10:
        session["known_card_counts"] = [0] * 10

    remaining = session.get("remaining_counts")
    if not isinstance(remaining, list) or len(remaining) != 10:
        session["remaining_counts"] = _fresh_counts()

    if not isinstance(session.get("shoe_observations"), list):
        session["shoe_observations"] = []
    if not isinstance(session.get("last_observation"), dict):
        session["last_observation"] = {}
    if not isinstance(session.get("stats"), dict):
        session["stats"] = _stats()
    return session


def get_session(user_id: str) -> Dict[str, Any]:
    uid = str(user_id or "").strip()
    if not uid:
        raise ValueError("user_id is required")
    with _LOCK:
        data = _load_all()
        session = data.get(uid)
        if not isinstance(session, dict):
            session = _default_session(uid)
        else:
            session = _migrate_session(session, uid)
        data[uid] = session
        _save_all(data)
        return copy.deepcopy(session)


def upsert_session(user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    uid = str(user_id or "").strip()
    if not uid:
        raise ValueError("user_id is required")
    with _LOCK:
        data = _load_all()
        current = data.get(uid)
        session = (
            _migrate_session(current, uid)
            if isinstance(current, dict)
            else _default_session(uid)
        )
        session.update(copy.deepcopy(updates or {}))
        session = _migrate_session(session, uid)
        session["user_id"] = uid
        session["updated_at"] = _now()
        data[uid] = session
        _save_all(data)
        return copy.deepcopy(session)


def start_session(user_id: str, venue: str = "", room: str = "") -> Dict[str, Any]:
    session = get_session(user_id)
    session.update(
        {
            "status": "分析中",
            "venue": str(venue or session.get("venue") or ""),
            "room": str(room or session.get("room") or ""),
        }
    )
    return upsert_session(user_id, session)


def end_session(user_id: str) -> Dict[str, Any]:
    return upsert_session(
        user_id,
        {"status": "已結束", "pending_prediction": {}},
    )


def reset_shoe(
    user_id: str,
    *,
    venue: Optional[str] = None,
    room: Optional[str] = None,
    reset_stats: bool = False,
) -> Dict[str, Any]:
    """Start a new physical shoe while preserving access information."""
    old = get_session(user_id)
    updates: Dict[str, Any] = {
        **_empty_shoe_state(),
        "shoe_id": uuid.uuid4().hex[:12],
        "point_history": [],
        "pending_prediction": {},
        "last_prediction": {},
        "last_settlement": {},
        "status": "分析中",
        "venue": old.get("venue", "") if venue is None else str(venue or ""),
        "room": old.get("room", "") if room is None else str(room or ""),
    }
    if reset_stats:
        updates["stats"] = _stats()
    return upsert_session(user_id, updates)


def reset_session(user_id: str) -> Dict[str, Any]:
    old = get_session(user_id)
    fresh = _default_session(user_id)
    fresh.update(
        {
            "status": "分析中",
            "venue": old.get("venue", ""),
            "room": old.get("room", ""),
            "permanent_access": bool(old.get("permanent_access")),
            "trial_started_at": int(old.get("trial_started_at", 0) or 0),
            "access_until": int(old.get("access_until", 0) or 0),
        }
    )
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
        upsert_session(
            user_id,
            {
                "trial_started_at": trial_started,
                "access_until": access_until,
            },
        )

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
    upsert_session(
        user_id,
        {"permanent_access": True, "status": "分析中"},
    )
    return True


def _outcome_from_point(point: str) -> str:
    text = str(point or "").strip().upper()
    if len(text) < 2 or not text[:2].isdigit():
        raise ValueError("invalid point")
    player, banker = int(text[0]), int(text[1])
    return "B" if banker > player else "P" if player > banker else "T"


def _normalize_cards(value: Any) -> Dict[str, List[int]]:
    if not isinstance(value, Mapping):
        return {}
    result: Dict[str, List[int]] = {}
    for side, aliases in {
        "player": ("player", "P", "閒", "闲"),
        "banker": ("banker", "B", "莊", "庄"),
    }.items():
        raw = None
        for alias in aliases:
            if alias in value:
                raw = value.get(alias)
                break
        if raw is None:
            continue
        try:
            cards = [int(card) % 10 for card in list(raw)]
        except Exception:
            continue
        if len(cards) in {2, 3}:
            result[side] = cards
    return result


def _card_counts(cards: Mapping[str, Sequence[int]]) -> List[int]:
    counts = [0] * 10
    for side in ("player", "banker"):
        for value in cards.get(side, ()):
            counts[int(value) % 10] += 1
    return counts


def _display_point(observation: Mapping[str, Any], point: str) -> str:
    suffix = str(
        observation.get("path_suffix")
        or observation.get("suffix")
        or ""
    ).strip().upper()
    hand_number = observation.get("hand_number")
    text = f"{point[:2]}{suffix if suffix in {'N', 'P', 'B', 'D'} else ''}"
    try:
        number = int(hand_number)
    except Exception:
        number = 0
    if number > 0:
        text += f"@{number}"
    return text


def record_point_and_settle(
    user_id: str,
    point: str,
    observation: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Settle the prior recommendation and persist factual shoe context.

    The model may use hand number and exact cards. It never receives settlement
    outcome, win/loss streak, or previous recommendations as predictive features.
    """
    session = get_session(user_id)
    obs = dict(observation or {})
    actual = _outcome_from_point(point)
    pending = dict(session.get("pending_prediction") or {})
    stats = dict(session.get("stats") or _stats())
    settlement: Dict[str, Any] = {}

    recommend = str(pending.get("recommend") or "").upper()
    if recommend in {"B", "P"}:
        if actual == "T":
            stats["ties_skipped"] = int(stats.get("ties_skipped", 0)) + 1
            settlement = {
                "actual": actual,
                "recommend": recommend,
                "verdict": "TIE_SKIPPED",
            }
        elif actual == recommend:
            stats["wins"] = int(stats.get("wins", 0)) + 1
            stats["current_win_streak"] = int(
                stats.get("current_win_streak", 0)
            ) + 1
            stats["current_loss_streak"] = 0
            stats["max_win_streak"] = max(
                int(stats.get("max_win_streak", 0)),
                stats["current_win_streak"],
            )
            settlement = {
                "actual": actual,
                "recommend": recommend,
                "verdict": "HIT",
            }
        else:
            stats["losses"] = int(stats.get("losses", 0)) + 1
            stats["current_loss_streak"] = int(
                stats.get("current_loss_streak", 0)
            ) + 1
            stats["current_win_streak"] = 0
            stats["max_loss_streak"] = max(
                int(stats.get("max_loss_streak", 0)),
                stats["current_loss_streak"],
            )
            settlement = {
                "actual": actual,
                "recommend": recommend,
                "verdict": "LOSS",
            }

    previous_hand = max(0, int(session.get("hand_number", 0) or 0))
    try:
        explicit_hand = int(obs.get("hand_number") or 0)
    except Exception:
        explicit_hand = 0
    if explicit_hand > 0 and explicit_hand <= previous_hand:
        raise ValueError(
            "輸入的牌靴局數沒有往前增加；若已換靴，請先輸入「新牌靴」。"
        )
    hand_number = explicit_hand if explicit_hand > 0 else previous_hand + 1
    hand_number = max(1, min(120, hand_number))
    continuous = hand_number == previous_hand + 1

    cards = _normalize_cards(obs.get("known_cards") or obs.get("cards"))
    has_complete_hand_cards = (
        len(cards.get("player", [])) in {2, 3}
        and len(cards.get("banker", [])) in {2, 3}
    )
    additions = _card_counts(cards) if has_complete_hand_cards else [0] * 10

    known_counts = list(session.get("known_card_counts") or [0] * 10)
    if len(known_counts) != 10:
        known_counts = [0] * 10

    deck_counts = _fresh_counts()
    counts_fit = all(
        int(known_counts[i]) + int(additions[i]) <= int(deck_counts[i])
        for i in range(10)
    )
    if not counts_fit:
        raise ValueError("輸入的已知牌超過八副牌可用數量，請先輸入「新牌靴」重置。")

    known_counts = [
        int(known_counts[i]) + int(additions[i])
        for i in range(10)
    ]
    remaining = [
        max(0, int(deck_counts[i]) - int(known_counts[i]))
        for i in range(10)
    ]

    previous_complete = bool(session.get("state_complete", True))
    # Full exact tracking is valid only when every hand from a new shoe is
    # continuous and includes all dealt cards.
    state_complete = bool(
        previous_complete
        and continuous
        and has_complete_hand_cards
        and (previous_hand > 0 or hand_number == 1)
    )
    tracked_hands = int(session.get("tracked_card_hands", 0) or 0)
    if has_complete_hand_cards:
        tracked_hands += 1

    suffix = str(
        obs.get("path_suffix")
        or obs.get("suffix")
        or ""
    ).strip().upper()
    normalized_observation = {
        "player": int(point[0]),
        "banker": int(point[1]),
        "path_suffix": suffix if suffix in {"N", "P", "B", "D"} else "",
        "path": obs.get("path"),
        "hand_number": hand_number,
        "known_cards": cards,
        "has_exact_cards": has_complete_hand_cards,
        "recorded_at": _now(),
    }

    shoe_observations = list(session.get("shoe_observations") or [])
    shoe_observations.append(normalized_observation)
    shoe_observations = shoe_observations[-300:]

    history = list(session.get("point_history") or [])
    history.append(_display_point(normalized_observation, point))
    history = history[-300:]

    if state_complete:
        state_source = "EXACT_CONTINUOUS_CARDS"
    elif has_complete_hand_cards:
        state_source = "CURRENT_HAND_CARDS"
    elif explicit_hand > 0:
        state_source = "HAND_NUMBER_ONLY"
    else:
        state_source = "AUTO_HAND_NUMBER"

    return upsert_session(
        user_id,
        {
            "status": "分析中",
            "hand_number": hand_number,
            "known_card_counts": known_counts,
            "remaining_counts": remaining,
            "tracked_card_hands": tracked_hands,
            "state_complete": state_complete,
            "state_source": state_source,
            "shoe_observations": shoe_observations,
            "last_observation": normalized_observation,
            "point_history": history,
            "last_settlement": settlement,
            "pending_prediction": {},
            "stats": stats,
        },
    )


def shoe_context(session: Mapping[str, Any]) -> Dict[str, Any]:
    """Return only context that is safe for the prediction engine."""
    last = dict(session.get("last_observation") or {})
    return {
        "hand_number": int(session.get("hand_number", 0) or 0),
        "known_cards": copy.deepcopy(last.get("known_cards") or {}),
        "known_card_counts": list(
            session.get("known_card_counts") or [0] * 10
        ),
        "remaining_counts": list(
            session.get("remaining_counts") or _fresh_counts()
        ),
        "state_complete": bool(session.get("state_complete", False)),
        "state_source": str(session.get("state_source") or "UNKNOWN"),
        "tracked_card_hands": int(
            session.get("tracked_card_hands", 0) or 0
        ),
    }


def save_prediction(
    user_id: str,
    prediction: Dict[str, Any],
) -> Dict[str, Any]:
    return upsert_session(
        user_id,
        {
            "pending_prediction": copy.deepcopy(prediction),
            "last_prediction": copy.deepcopy(prediction),
            "status": "等待下一局點數",
        },
    )


def list_sessions() -> List[Dict[str, Any]]:
    with _LOCK:
        return [
            copy.deepcopy(x)
            for x in _load_all().values()
            if isinstance(x, dict)
        ]
