"""UID-scoped final-total input/commit for LINE's BBB V2.6 point-only mode.

The session audit is not passed as model history: BBB V2.6 always starts a new
PF256 from exactly one (Player, Banker) resolved score pair.
"""
from __future__ import annotations

import copy
import re
import unicodedata
import uuid
from typing import Any, Mapping

import store
from bbb_v26_points_runtime import predict_final_score

_PAIR_PB = re.compile(r"^閒\s*[:=：]?\s*([0-9])\s*[,，/、\- ]*莊\s*[:=：]?\s*([0-9])$")
_PAIR_BP = re.compile(r"^莊\s*[:=：]?\s*([0-9])\s*[,，/、\- ]*閒\s*[:=：]?\s*([0-9])$")
_PAIR_BARE = re.compile(r"^([0-9])\s*[,，/、\- ]+\s*([0-9])$")
_MAX_BATCH = 20


def parse_score_pairs(value: str) -> list[dict[str, int]] | None:
    """Left Player then Banker; accept one per line or a short historical batch.

    Return None for non-score commands (including unlock codes).
    """
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    if not text:
        return None
    chunks = [part.strip() for part in re.split(r"[\n;；]+", text) if part.strip()]
    if not chunks or len(chunks) > _MAX_BATCH:
        return None
    pairs = []
    for chunk in chunks:
        match = _PAIR_PB.fullmatch(chunk)
        if match:
            player, banker = map(int, match.groups())
        else:
            match = _PAIR_BP.fullmatch(chunk)
            if match:
                banker, player = map(int, match.groups())
            else:
                match = _PAIR_BARE.fullmatch(chunk)
                if not match:
                    return None
                player, banker = map(int, match.groups())
        pairs.append({"player": player, "banker": banker})
    return pairs


def begin(uid: str) -> dict[str, Any]:
    """Atomic UID-only start; keep existing access, venue, room and bankroll."""
    uid = str(uid or "").strip()
    if not uid:
        raise ValueError("LINE UID required")
    with store._LOCK:
        data = store._load_all_unlocked()
        raw = data.get(uid)
        session = store._migrate_session(raw, uid) if isinstance(raw, dict) else store._default_session(uid)
        if not session.get("venue"):
            raise ValueError("請先選擇館別。")
        store._clear_screen_fields(session)
        session.update({
            "analysis_mode": "screen", "analysis_active": True,
            "analysis_run_id": uuid.uuid4().hex[:16],
            "analysis_started_at": store._now(), "analysis_ended_at": 0,
            "point_score_mode": True, "point_score_audit": [],
            "last_point_score": {}, "last_point_message_id": "",
            "awaiting_screenshot": False, "awaiting_bankroll": False,
            "status": "等待輸入閒莊點數", "updated_at": store._now(),
        })
        data[uid] = session
        store._save_all_unlocked(data)
        return copy.deepcopy(session)


def _check_active(session: Mapping[str, Any], expected_run_id: str) -> None:
    if (not bool(session.get("analysis_active")) or
            not bool(session.get("point_score_mode")) or
            str(session.get("analysis_run_id") or "") != str(expected_run_id)):
        raise RuntimeError("本次分析已結束或已重新開始，舊點數不會寫入。")


def evaluate_and_save(uid: str, expected_run_id: str,
                      pairs: list[dict[str, int]], message_id: str = "") -> dict[str, Any]:
    """Calculate one independent model on the last submitted score pair only.

    An explicitly supplied historical batch is audit-only and never used for
    PF conditioning. The actual post-calculation commit is run-id CAS protected.
    """
    uid = str(uid or "").strip()
    if not uid or not pairs or len(pairs) > _MAX_BATCH:
        raise ValueError("請輸入正確的閒莊點數")
    with store._LOCK:
        current = store._load_all_unlocked().get(uid)
        session = store._migrate_session(current, uid) if isinstance(current, dict) else store._default_session(uid)
        _check_active(session, expected_run_id)
        if message_id and session.get("last_point_message_id") == message_id:
            return copy.deepcopy(session)

    last = pairs[-1]
    report = predict_final_score(last["player"], last["banker"])
    direction = report["direction"]
    if direction not in ("B", "P", "SKIP") or report["action"] != "SKIP":
        raise RuntimeError("BBB V2.6 unverified point display contract changed")
    probs = report["physical_ev"]["simulated_probability"]
    prediction = {
        "engine": "BBB_V26_INDEPENDENT_FINAL_SCORE",
        "bbb_v26_score": report,
        "action": direction, "recommend": direction, "direction": direction,
        "action_text": "觀望" if direction == "SKIP" else "莊" if direction == "B" else "閒",
        "recommend_text": "觀望" if direction == "SKIP" else "莊" if direction == "B" else "閒",
        "banker_rate": float(probs["B"]) * 100.0,
        "player_rate": float(probs["P"]) * 100.0,
        "tie_rate": float(probs["T"]) * 100.0,
        "ev_banker": report["ev_banker"], "ev_player": report["ev_player"],
        "bet_allowed": False, "formal_action": "Skip",
        "suggested_bet_amount": 0, "bet_amount": 0, "bet_percentage": 0,
        "confidence_label": "模擬 EV，非命中率",
        "point_score_observation_count": 1,
    }
    with store._LOCK:
        data = store._load_all_unlocked()
        current = data.get(uid)
        session = store._migrate_session(current, uid) if isinstance(current, dict) else store._default_session(uid)
        _check_active(session, expected_run_id)
        if message_id and session.get("last_point_message_id") == message_id:
            return copy.deepcopy(session)
        before = session.get("point_score_audit")
        if not isinstance(before, list):
            before = []
        session["point_score_audit"] = (before + [
            {"player": int(p["player"]), "banker": int(p["banker"])}
            for p in pairs
        ])[-500:]
        session["last_point_score"] = dict(last)
        session["last_point_message_id"] = str(message_id or "")[:128]
        session["screen_last_prediction"] = copy.deepcopy(prediction)
        session["screen_last_analyzed_at"] = store._now()
        session["screen_data_version"] = int(session.get("screen_data_version", 0) or 0) + 1
        session["screen_last_input_source"] = "manual_final_score"
        session["status"] = "點數分析中"
        session["updated_at"] = store._now()
        data[uid] = session
        store._save_all_unlocked(data)
        return copy.deepcopy(session)
