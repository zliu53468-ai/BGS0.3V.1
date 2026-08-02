"""Atomic JSON storage for BGS V9.2 screen-analysis and virtual compatibility sessions."""
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

from particle_filter_points import (
    counts_from_shoe,
    create_virtual_shoe,
    deal_ordered_hand,
)


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
HISTORY_LIMIT = max(
    10,
    min(300, int(os.getenv("SESSION_HISTORY_LIMIT", "80") or "80")),
)
VIRTUAL_WARMUP_ENABLED = os.getenv("VIRTUAL_WARMUP_ENABLED", "1").strip() == "1"
VIRTUAL_WARMUP_MIN = max(
    0,
    min(80, int(os.getenv("VIRTUAL_WARMUP_MIN", "8") or "8")),
)
VIRTUAL_WARMUP_MAX = max(
    VIRTUAL_WARMUP_MIN,
    min(100, int(os.getenv("VIRTUAL_WARMUP_MAX", "25") or "25")),
)
DEFAULT_ANALYSIS_MODE = os.getenv("DEFAULT_ANALYSIS_MODE", "screen").strip().lower()
if DEFAULT_ANALYSIS_MODE not in {"screen", "virtual"}:
    DEFAULT_ANALYSIS_MODE = "screen"
_LOCK = threading.RLock()


def _now() -> int:
    return int(time.time())


def _fresh_stats() -> Dict[str, int]:
    return {
        "wins": 0,
        "losses": 0,
        "ties_skipped": 0,
        "observes": 0,
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
    """Create a fresh hidden shoe and optionally simulate middle entry.

    Warm-up hands are real internal virtual hands: cards are removed, the
    remaining counts are updated, and their outcomes seed only the virtual
    history model. They do not count toward the user's visible win/loss stats.
    """
    shoe = create_virtual_shoe(decks=decks)
    cut_card_remaining = _new_cut_card(decks)
    warmup_target = 0
    if VIRTUAL_WARMUP_ENABLED and VIRTUAL_WARMUP_MAX > 0:
        span = VIRTUAL_WARMUP_MAX - VIRTUAL_WARMUP_MIN + 1
        warmup_target = VIRTUAL_WARMUP_MIN + secrets.randbelow(max(1, span))

    warmup_history: List[Dict[str, Any]] = []
    for index in range(warmup_target):
        if len(shoe) <= cut_card_remaining + 6:
            break
        hand, shoe = deal_ordered_hand(shoe)
        hand_data = hand.as_dict()
        warmup_history.append(
            {
                "round_number": index + 1,
                "outcome": hand.outcome,
                "draw_path": hand.draw_path,
                "player_total": hand.player_total,
                "banker_total": hand.banker_total,
                "cards_used": hand.cards_used,
                "warmup": True,
                "created_at": _now(),
            }
        )

    warmup_rounds = len(warmup_history)
    return {
        "shoe_id": uuid.uuid4().hex[:12],
        "virtual_shoe": shoe,
        "remaining_counts": counts_from_shoe(shoe),
        "cut_card_remaining": cut_card_remaining,
        "hand_number": warmup_rounds,
        "warmup_rounds": warmup_rounds,
        "round_history": warmup_history[-HISTORY_LIMIT:],
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
        # 分析模式分流。screen=真人畫面截圖；virtual=程式內建虛擬牌靴。
        "analysis_mode": DEFAULT_ANALYSIS_MODE,
        "awaiting_screenshot": False,
        "screen_data_version": 0,
        "screen_prediction_version": 0,
        "screen_last_input_source": "",
        "screen_last_data_updated_at": 0,
        # 路紙影像辨識與手動校正狀態，與虛擬牌靴狀態分開保存。
        "road_sequence": [],
        "road_last_analysis": {},
        "road_last_vision": {},
        "road_source": "",
        "road_last_image_at": 0,
        "road_corrections": 0,
        # 全畫面 OCR＋大路辨識狀態。
        "screen_last_ocr": {},
        "screen_last_detection": {},
        "screen_last_prediction": {},
        "screen_remaining_cards": 0,
        "screen_analysis_count": 0,
        "screen_last_analyzed_at": 0,
        "screen_processing_ms": 0.0,
        # 本金與建議金額狀態。
        "bankroll": 0,
        "initial_bankroll": 0,
        "awaiting_bankroll": False,
        "last_suggested_bet": 0,
        "last_bet_percentage": 0.0,
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

    # 舊版 Session 自動補上路紙欄位，並清除不合法字元。
    if not isinstance(session.get("road_sequence"), list):
        session["road_sequence"] = []
    session["road_sequence"] = [
        str(item).upper().strip()
        for item in session.get("road_sequence", [])
        if str(item).upper().strip() in {"B", "P"}
    ][-500:]
    if not isinstance(session.get("road_last_analysis"), dict):
        session["road_last_analysis"] = {}
    if not isinstance(session.get("road_last_vision"), dict):
        session["road_last_vision"] = {}
    session["road_source"] = str(session.get("road_source") or "")
    session["road_last_image_at"] = max(
        0, int(session.get("road_last_image_at", 0) or 0)
    )
    session["road_corrections"] = max(
        0, int(session.get("road_corrections", 0) or 0)
    )

    # 舊版 Session 自動補上模式與畫面資料版本欄位。
    mode = str(session.get("analysis_mode") or DEFAULT_ANALYSIS_MODE).strip().lower()
    session["analysis_mode"] = mode if mode in {"screen", "virtual"} else DEFAULT_ANALYSIS_MODE
    session["awaiting_screenshot"] = bool(session.get("awaiting_screenshot", False))
    session["screen_data_version"] = max(
        0, int(session.get("screen_data_version", 0) or 0)
    )
    session["screen_prediction_version"] = max(
        0, int(session.get("screen_prediction_version", 0) or 0)
    )
    session["screen_prediction_version"] = min(
        session["screen_prediction_version"],
        session["screen_data_version"],
    )
    session["screen_last_input_source"] = str(
        session.get("screen_last_input_source") or ""
    )
    session["screen_last_data_updated_at"] = max(
        0, int(session.get("screen_last_data_updated_at", 0) or 0)
    )

    # 舊版 Session 自動補上全畫面辨識欄位。
    for key in ("screen_last_ocr", "screen_last_detection", "screen_last_prediction"):
        if not isinstance(session.get(key), dict):
            session[key] = {}
    session["screen_remaining_cards"] = max(
        0, min(416, int(session.get("screen_remaining_cards", 0) or 0))
    )
    session["screen_analysis_count"] = max(
        0, int(session.get("screen_analysis_count", 0) or 0)
    )
    session["screen_last_analyzed_at"] = max(
        0, int(session.get("screen_last_analyzed_at", 0) or 0)
    )
    try:
        session["screen_processing_ms"] = max(
            0.0, float(session.get("screen_processing_ms", 0.0) or 0.0)
        )
    except Exception:
        session["screen_processing_ms"] = 0.0

    # 舊版 Session 自動補上本金欄位。
    session["bankroll"] = max(
        0, min(100_000_000, int(session.get("bankroll", 0) or 0))
    )
    session["initial_bankroll"] = max(
        0, min(100_000_000, int(session.get("initial_bankroll", 0) or 0))
    )
    if session["initial_bankroll"] <= 0 and session["bankroll"] > 0:
        session["initial_bankroll"] = session["bankroll"]
    session["awaiting_bankroll"] = bool(session.get("awaiting_bankroll", False))
    session["last_suggested_bet"] = max(
        0, min(session["bankroll"], int(session.get("last_suggested_bet", 0) or 0))
    )
    try:
        session["last_bet_percentage"] = max(
            0.0, min(100.0, float(session.get("last_bet_percentage", 0.0) or 0.0))
        )
    except Exception:
        session["last_bet_percentage"] = 0.0

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
        session["analysis_mode"] = "screen"
        session["awaiting_screenshot"] = False
        session["awaiting_bankroll"] = False
        session["status"] = "待開始分析"
        if changed:
            preserved_reset_count = int(session.get("shoe_reset_count", 0) or 0)
            session.update(_new_shoe_state())
            session["shoe_reset_count"] = preserved_reset_count + 1
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
    return upsert_session(
        user_id,
        {
            "status": "已結束",
            "pending_prediction": {},
            "awaiting_screenshot": False,
            "awaiting_bankroll": False,
        },
    )


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
        session["analysis_mode"] = "virtual"
        session["awaiting_screenshot"] = False
        session["status"] = "分析中"
        if venue is not None:
            session["venue"] = str(venue or "")
        if room is not None:
            session["room"] = str(room or "1")
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
    result.update(
        {
            key: int(value or 0)
            for key, value in stats.items()
            if key in result
        }
    )
    result["total_rounds"] += 1
    if verdict == "HIT":
        result["wins"] += 1
        result["current_win_streak"] += 1
        result["current_loss_streak"] = 0
        result["max_win_streak"] = max(
            result["max_win_streak"],
            result["current_win_streak"],
        )
    elif verdict == "MISS":
        result["losses"] += 1
        result["current_loss_streak"] += 1
        result["current_win_streak"] = 0
        result["max_loss_streak"] = max(
            result["max_loss_streak"],
            result["current_loss_streak"],
        )
    elif verdict == "OBSERVE":
        result["observes"] += 1
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
            # The replacement shoe already contains its own hidden warm-up
            # history. Visible analysis logs and cumulative statistics remain.
            session["last_prediction"] = {}
            session["last_virtual_hand"] = {}
            shoe_reset = True

        output = dict(runner(copy.deepcopy(session)))
        prediction = dict(output.get("prediction") or {})
        hand = dict(output.get("hand") or {})
        remaining_shoe = [int(card) for card in list(output.get("remaining_shoe") or [])]
        if not prediction.get("ok") or not hand or not _valid_shoe(remaining_shoe):
            raise ValueError("虛擬牌靴分析回傳格式錯誤。")

        session["analysis_mode"] = "virtual"
        session["awaiting_screenshot"] = False
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
            "validation_gap": prediction.get("validation_gap"),
            "model_core": prediction.get("model_core"),
            "hypergeometric_probabilities": prediction.get(
                "hypergeometric_probabilities"
            ),
            "weights": prediction.get("weights"),
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


def request_bankroll(user_id: str) -> Dict[str, Any]:
    """標記下一則文字輸入為本金。"""
    session = get_session(user_id)
    if not session.get("venue"):
        raise ValueError("請先選擇館別。")
    return upsert_session(
        user_id,
        {
            "analysis_mode": "screen",
            "awaiting_bankroll": True,
            "awaiting_screenshot": False,
            "status": "等待輸入本金",
        },
    )


def set_bankroll(
    user_id: str,
    bankroll: int,
    *,
    begin_screen: bool = True,
) -> Dict[str, Any]:
    """儲存本金；預設完成後直接進入等待遊戲截圖。"""
    value = int(bankroll)
    if value < 100:
        raise ValueError("本金至少需輸入 100 元。")
    if value > 100_000_000:
        raise ValueError("本金數字過大，請重新輸入。")
    session = get_session(user_id)
    if not session.get("venue"):
        raise ValueError("請先選擇館別。")
    return upsert_session(
        user_id,
        {
            "bankroll": value,
            "initial_bankroll": value,
            "awaiting_bankroll": False,
            "analysis_mode": "screen",
            "awaiting_screenshot": bool(begin_screen),
            "status": "等待遊戲截圖" if begin_screen else "待開始分析",
        },
    )


def begin_screen_analysis(
    user_id: str,
    *,
    clear_existing: bool = False,
) -> Dict[str, Any]:
    """切換到截圖模式並標記為等待最新遊戲畫面。"""
    uid = str(user_id or "").strip()
    if not uid:
        raise ValueError("user_id is required")
    with _LOCK:
        data = _load_all_unlocked()
        raw = data.get(uid)
        session = _migrate_session(raw, uid) if isinstance(raw, dict) else _default_session(uid)
        if not session.get("venue"):
            raise ValueError("請先選擇館別。")
        if int(session.get("bankroll", 0) or 0) <= 0:
            session["analysis_mode"] = "screen"
            session["awaiting_bankroll"] = True
            session["awaiting_screenshot"] = False
            session["status"] = "等待輸入本金"
        else:
            session["analysis_mode"] = "screen"
            session["awaiting_bankroll"] = False
            session["awaiting_screenshot"] = True
            session["status"] = "等待遊戲截圖"
        if clear_existing:
            session["road_sequence"] = []
            session["road_last_analysis"] = {}
            session["road_last_vision"] = {}
            session["screen_last_ocr"] = {}
            session["screen_last_detection"] = {}
            session["screen_last_prediction"] = {}
            session["screen_remaining_cards"] = 0
            session["screen_processing_ms"] = 0.0
            session["screen_data_version"] = 0
            session["screen_prediction_version"] = 0
            session["screen_last_input_source"] = ""
            session["screen_last_data_updated_at"] = 0
            session["last_suggested_bet"] = 0
            session["last_bet_percentage"] = 0.0
        session["updated_at"] = _now()
        data[uid] = session
        _save_all_unlocked(data)
        return copy.deepcopy(session)

def begin_virtual_analysis(user_id: str) -> Dict[str, Any]:
    """明確切換回內建虛擬牌靴模式。"""
    return upsert_session(
        user_id,
        {
            "analysis_mode": "virtual",
            "awaiting_screenshot": False,
            "status": "分析中",
        },
    )


def clear_screen_analysis(
    user_id: str,
    *,
    keep_mode: bool = True,
) -> Dict[str, Any]:
    """清除 OCR、路紙與截圖預測，不破壞試用、開通與虛擬牌靴資料。"""
    updates: Dict[str, Any] = {
        "road_sequence": [],
        "road_last_analysis": {},
        "road_last_vision": {},
        "road_source": "",
        "road_last_image_at": 0,
        "road_corrections": 0,
        "screen_last_ocr": {},
        "screen_last_detection": {},
        "screen_last_prediction": {},
        "screen_remaining_cards": 0,
        "screen_processing_ms": 0.0,
        "screen_data_version": 0,
        "screen_prediction_version": 0,
        "screen_last_input_source": "",
        "screen_last_data_updated_at": 0,
        "awaiting_screenshot": bool(keep_mode),
        "awaiting_bankroll": False,
        "last_suggested_bet": 0,
        "last_bet_percentage": 0.0,
        "status": "等待遊戲截圖" if keep_mode else "待開始分析",
    }
    if keep_mode:
        updates["analysis_mode"] = "screen"
    return upsert_session(user_id, updates)


def screen_has_fresh_data(session: Mapping[str, Any]) -> bool:
    """判斷截圖/手動結果是否比最後一次預測更新。"""
    return int(session.get("screen_data_version", 0) or 0) > int(
        session.get("screen_prediction_version", 0) or 0
    )


def set_road_sequence(
    user_id: str,
    sequence: List[str],
    *,
    analysis: Optional[Mapping[str, Any]] = None,
    vision: Optional[Mapping[str, Any]] = None,
    source: str = "image",
) -> Dict[str, Any]:
    """覆蓋使用者的路紙序列，通常由影像辨識成功後呼叫。"""
    cleaned = [
        str(item).upper().strip()
        for item in list(sequence or [])
        if str(item).upper().strip() in {"B", "P"}
    ][-500:]
    return upsert_session(
        user_id,
        {
            "road_sequence": cleaned,
            "road_last_analysis": dict(analysis or {}),
            "road_last_vision": dict(vision or {}),
            "road_source": str(source or "image"),
            "road_last_image_at": _now() if str(source or "") == "image" else 0,
        },
    )


def append_road_result(
    user_id: str,
    outcome: str,
    *,
    analysis: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """在路紙尾端手動補上一局 B 或 P。"""
    value = str(outcome or "").upper().strip()
    if value not in {"B", "P"}:
        raise ValueError("路紙結果只能是 B 或 P。")

    with _LOCK:
        data = _load_all_unlocked()
        uid = str(user_id or "").strip()
        if not uid:
            raise ValueError("user_id is required")
        raw = data.get(uid)
        session = _migrate_session(raw, uid) if isinstance(raw, dict) else _default_session(uid)
        sequence = list(session.get("road_sequence") or [])
        sequence.append(value)
        session["road_sequence"] = sequence[-500:]
        session["road_source"] = "manual"
        session["road_corrections"] = int(session.get("road_corrections", 0) or 0) + 1
        if str(session.get("analysis_mode") or "") == "screen":
            session["screen_data_version"] = int(
                session.get("screen_data_version", 0) or 0
            ) + 1
            session["screen_last_input_source"] = "manual"
            session["screen_last_data_updated_at"] = _now()
            session["awaiting_screenshot"] = False
        if analysis is not None:
            session["road_last_analysis"] = dict(analysis)
        session["updated_at"] = _now()
        data[uid] = session
        _save_all_unlocked(data)
        return copy.deepcopy(session)


def update_road_analysis(
    user_id: str,
    analysis: Mapping[str, Any],
) -> Dict[str, Any]:
    """只更新路紙模型結果，不改動序列。"""
    return upsert_session(user_id, {"road_last_analysis": dict(analysis or {})})


def clear_road_sequence(user_id: str) -> Dict[str, Any]:
    """清除路紙與截圖辨識結果；保留目前分析模式。"""
    current = get_session(user_id)
    return clear_screen_analysis(
        user_id,
        keep_mode=str(current.get("analysis_mode") or "") == "screen",
    )


def update_screen_analysis(
    user_id: str,
    *,
    ocr: Mapping[str, Any],
    detection: Mapping[str, Any],
    sequence: List[str],
    prediction: Mapping[str, Any],
    resolved: Optional[Mapping[str, Any]] = None,
    processing_ms: float = 0.0,
    source: str = "screen_image",
) -> Dict[str, Any]:
    """原子更新一張遊戲截圖的 OCR、路紙序列與模型結果。"""
    uid = str(user_id or "").strip()
    if not uid:
        raise ValueError("user_id is required")
    cleaned = [
        str(item).upper().strip()
        for item in list(sequence or [])
        if str(item).upper().strip() in {"B", "P"}
    ][-500:]
    resolved_data = dict(resolved or {})

    with _LOCK:
        data = _load_all_unlocked()
        raw = data.get(uid)
        session = _migrate_session(raw, uid) if isinstance(raw, dict) else _default_session(uid)

        venue = str(resolved_data.get("venue_code") or "").upper().strip()
        room = str(resolved_data.get("room") or "").strip()
        remaining = resolved_data.get("remaining_cards")
        if venue:
            session["venue"] = venue
        if room:
            session["room"] = room
        try:
            remaining_value = max(0, min(416, int(remaining or 0)))
        except Exception:
            remaining_value = 0

        next_data_version = int(session.get("screen_data_version", 0) or 0) + 1
        session["analysis_mode"] = "screen"
        session["awaiting_screenshot"] = False
        session["screen_data_version"] = next_data_version
        session["screen_prediction_version"] = next_data_version
        session["screen_last_input_source"] = str(source or "screen_image")
        session["screen_last_data_updated_at"] = _now()
        session["road_sequence"] = cleaned
        session["road_last_analysis"] = dict(prediction or {})
        session["road_last_vision"] = dict(detection or {})
        session["road_source"] = str(source or "screen_image")
        session["road_last_image_at"] = _now()
        session["screen_last_ocr"] = dict(ocr or {})
        session["screen_last_detection"] = dict(detection or {})
        session["screen_last_prediction"] = dict(prediction or {})
        session["screen_remaining_cards"] = remaining_value
        session["screen_analysis_count"] = int(session.get("screen_analysis_count", 0) or 0) + 1
        session["screen_last_analyzed_at"] = _now()
        session["screen_processing_ms"] = max(0.0, float(processing_ms or 0.0))
        session["last_suggested_bet"] = max(
            0, int(dict(prediction or {}).get("suggested_bet_amount", 0) or 0)
        )
        session["last_bet_percentage"] = max(
            0.0, float(dict(prediction or {}).get("bet_percentage", 0.0) or 0.0)
        )
        session["awaiting_bankroll"] = False
        session["status"] = "分析完成"
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
    "append_road_result",
    "begin_screen_analysis",
    "begin_virtual_analysis",
    "clear_screen_analysis",
    "clear_road_sequence",
    "end_session",
    "get_session",
    "request_bankroll",
    "reset_session",
    "reset_shoe",
    "run_virtual_round",
    "save_prediction",
    "screen_has_fresh_data",
    "select_venue",
    "set_road_sequence",
    "update_road_analysis",
    "update_screen_analysis",
    "set_bankroll",
    "set_room",
    "start_session",
    "upsert_session",
]
