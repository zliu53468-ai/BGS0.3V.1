import json
import os
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple

try:
    from config import COMBO_DB_PATH
except Exception:
    COMBO_DB_PATH = os.getenv("COMBO_DB_PATH", "data/combo_db_3m.json")

BASE_BANKER_NO_TIE = 0.5068


def _resolve_path(path: str) -> str:
    if os.path.exists(path):
        return path
    cwd_path = os.path.join(os.getcwd(), path)
    if os.path.exists(cwd_path):
        return cwd_path
    return path


@lru_cache(maxsize=1)
def load_combo_db() -> Dict[str, Any]:
    path = _resolve_path(COMBO_DB_PATH)
    if not os.path.exists(path):
        return {
            "meta": {
                "source": "COMBO_DB_FILE_NOT_FOUND",
                "path": COMBO_DB_PATH,
                "total_simulated_samples": 0,
                "record_count": 0,
            },
            "records": {},
        }

    with open(path, "r", encoding="utf-8") as f:
        db = json.load(f)

    if not isinstance(db, dict):
        return {
            "meta": {
                "source": "COMBO_DB_FORMAT_ERROR",
                "path": COMBO_DB_PATH,
                "total_simulated_samples": 0,
                "record_count": 0,
            },
            "records": {},
        }

    if "records" not in db:
        db = {
            "meta": db.get("meta", {}) if isinstance(db.get("meta", {}), dict) else {},
            "records": db,
        }

    if not isinstance(db.get("records"), dict):
        db["records"] = {}

    return db


def combo_db_meta() -> Dict[str, Any]:
    db = load_combo_db()
    meta = db.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    records = db.get("records", {})
    meta.setdefault("record_count", len(records) if isinstance(records, dict) else 0)
    meta.setdefault("total_simulated_samples", 0)
    meta.setdefault("source", "COMBO_DB")
    return meta


def normalize_prob_pair(banker: float, player: float) -> Tuple[float, float]:
    banker = float(banker)
    player = float(player)
    if banker > 1:
        banker /= 100.0
    if player > 1:
        player /= 100.0
    total = banker + player
    if total <= 0:
        banker = BASE_BANKER_NO_TIE
        player = 1.0 - banker
    else:
        banker /= total
        player /= total
    return banker, player


def normalize_round_result(value: Any) -> Optional[str]:
    if value is None:
        return None

    if isinstance(value, dict):
        raw = (
            value.get("result")
            or value.get("winner")
            or value.get("last_result")
            or value.get("outcome")
            or value.get("side")
        )
        if raw is not None:
            return normalize_round_result(raw)

        pp = value.get("player_point", value.get("player", value.get("p")))
        bp = value.get("banker_point", value.get("banker", value.get("b")))
        try:
            if pp is None or bp is None:
                return None
            pp = int(pp)
            bp = int(bp)
            if pp > bp:
                return "P"
            if bp > pp:
                return "B"
            return "T"
        except Exception:
            return None

    s = str(value).strip().upper()
    if s in {"B", "BANKER", "莊", "庄"}:
        return "B"
    if s in {"P", "PLAYER", "閒", "閑", "闲"}:
        return "P"
    if s in {"T", "TIE", "和", "和局"}:
        return "T"
    return None


def rounds_to_results(rounds: Optional[List[Any]]) -> List[str]:
    if not rounds:
        return []
    out: List[str] = []
    for r in rounds:
        ch = normalize_round_result(r)
        if ch in {"B", "P", "T"}:
            out.append(ch)
    return out


def normalize_w7_pattern(seq: List[str]) -> Optional[str]:
    fixed: List[str] = []
    last: Optional[str] = None
    for s in seq:
        if s in {"B", "P"}:
            fixed.append(s)
            last = s
        else:
            if last is None:
                return None
            fixed.append(last)
    return "".join(fixed)


def streak_info(results: List[str]) -> Tuple[str, int]:
    count = 0
    side: Optional[str] = None
    for r in reversed(results):
        if r == "T":
            continue
        if side is None:
            side = r
            count = 1
        elif r == side:
            count += 1
        else:
            break
    if side is None:
        return "N", 0
    return side, count


def alt_bucket(results: List[str], window: int = 6) -> str:
    seq = [x for x in results[-window:] if x in {"B", "P"}]
    if len(seq) < 3:
        return "NA"
    alt = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    if alt >= 5:
        return "ALT5"
    if alt >= 3:
        return "ALT3_4"
    return "ALT0_2"


def balance10(results: List[str]) -> str:
    seq = [x for x in results[-10:] if x in {"B", "P"}]
    if not seq:
        return "NA"
    b = seq.count("B")
    p = seq.count("P")
    if b - p >= 4:
        return "B_HEAVY"
    if p - b >= 4:
        return "P_HEAVY"
    if abs(b - p) <= 1:
        return "BALANCED"
    return "MID"


def tie_age(results: List[str]) -> str:
    age = 0
    for r in reversed(results):
        if r == "T":
            break
        age += 1
    if age == 0:
        return "T0"
    if age == 1:
        return "T1"
    if age == 2:
        return "T2"
    if age <= 4:
        return "T3_4"
    return "T5P"


def point_key(player_point: int, banker_point: int) -> str:
    return f"P{int(player_point)}_B{int(banker_point)}"


def build_combo_candidate_keys(player_point: int, banker_point: int, rounds: Optional[List[Any]]) -> List[str]:
    pkey = point_key(player_point, banker_point)
    results = rounds_to_results(rounds)
    keys: List[str] = []

    # 查詢順序：先最具體，再較粗略。
    if len(results) >= 5:
        pat5 = "".join(results[-5:])
        side, streak = streak_info(results)
        streak_bucket = min(streak, 6)
        keys.append(f"{pkey}|W5:{pat5}|ALT:{alt_bucket(results)}")
        keys.append(f"{pkey}|W5:{pat5}|STREAK:{side}{streak_bucket}")

    for w in [7, 5, 3]:
        if len(results) >= w:
            raw_seq = results[-w:]
            if w == 7:
                pat = normalize_w7_pattern(raw_seq)
                if not pat:
                    continue
            else:
                pat = "".join(raw_seq)
            keys.append(f"{pkey}|W{w}:{pat}")

    side, streak = streak_info(results)
    streak_bucket = min(streak, 6)
    keys.append(f"{pkey}|STREAK:{side}{streak_bucket}")
    keys.append(f"{pkey}|ALT:{alt_bucket(results)}")
    keys.append(f"{pkey}|BAL10:{balance10(results)}")
    keys.append(f"{pkey}|TIEAGE:{tie_age(results)}")

    seen = set()
    out: List[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def extract_combo_record(rec: Dict[str, Any], key: str) -> Dict[str, Any]:
    rec = dict(rec)

    banker = (
        rec.get("next_banker_rate")
        if rec.get("next_banker_rate") is not None
        else rec.get("banker_prob")
        if rec.get("banker_prob") is not None
        else rec.get("banker_rate")
    )
    player = (
        rec.get("next_player_rate")
        if rec.get("next_player_rate") is not None
        else rec.get("player_prob")
        if rec.get("player_prob") is not None
        else rec.get("player_rate")
    )

    if banker is None or player is None:
        raise ValueError(f"Combo DB record missing banker/player probability: {key}")

    banker_prob, player_prob = normalize_prob_pair(float(banker), float(player))
    sample = rec.get("sample", rec.get("sample_size", rec.get("no_tie_sample", 0)))
    try:
        sample = int(sample)
    except Exception:
        sample = 0

    rec["feature_key"] = key
    rec["next_banker_rate"] = banker_prob
    rec["next_player_rate"] = player_prob
    rec["banker_prob"] = banker_prob
    rec["player_prob"] = player_prob
    rec["sample"] = sample
    rec.setdefault("source", "COMBO_DB")
    return rec


def find_combo_record(player_point: int, banker_point: int, rounds: Optional[List[Any]], min_sample: int = 80) -> Optional[Dict[str, Any]]:
    db = load_combo_db()
    records = db.get("records", {})
    if not isinstance(records, dict) or not records:
        return None

    for key in build_combo_candidate_keys(player_point, banker_point, rounds):
        rec = records.get(key)
        if not isinstance(rec, dict):
            continue
        try:
            sample = int(rec.get("sample", rec.get("no_tie_sample", 0)) or 0)
        except Exception:
            sample = 0
        if sample < int(min_sample):
            continue
        return extract_combo_record(rec, key)

    return None


def combo_lookup(player_point: int, banker_point: int, rounds: Optional[List[Any]], min_sample: int = 80) -> Dict[str, Any]:
    meta = combo_db_meta()
    rec = find_combo_record(player_point, banker_point, rounds, min_sample=min_sample)
    if not rec:
        return {
            "available": False,
            "feature_key": point_key(player_point, banker_point),
            "banker_prob": BASE_BANKER_NO_TIE,
            "player_prob": 1.0 - BASE_BANKER_NO_TIE,
            "source": "COMBO_DB_NOT_MATCHED",
            "sample_size": 0,
            "total_simulated_samples": int(meta.get("total_simulated_samples", 0) or 0),
            "candidate_keys": build_combo_candidate_keys(player_point, banker_point, rounds)[:8],
        }

    return {
        "available": True,
        "feature_key": rec.get("feature_key", point_key(player_point, banker_point)),
        "banker_prob": float(rec["banker_prob"]),
        "player_prob": float(rec["player_prob"]),
        "source": rec.get("source", "COMBO_DB"),
        "sample_size": int(rec.get("sample", 0) or 0),
        "total_simulated_samples": int(meta.get("total_simulated_samples", 0) or 0),
        "candidate_keys": build_combo_candidate_keys(player_point, banker_point, rounds)[:8],
    }


def clear_combo_db_cache():
    load_combo_db.cache_clear()
