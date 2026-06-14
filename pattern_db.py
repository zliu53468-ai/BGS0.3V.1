import json
import os
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple

from config import RESULT_PATTERN_DB_PATH


BASE_BANKER_NO_TIE = 0.5068


@lru_cache(maxsize=1)
def load_pattern_db() -> Dict[str, Any]:
    path = RESULT_PATTERN_DB_PATH

    if not os.path.exists(path):
        path = os.path.join(os.getcwd(), path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Pattern DB file not found: {RESULT_PATTERN_DB_PATH}")

    with open(path, "r", encoding="utf-8") as f:
        db = json.load(f)

    if not isinstance(db, dict):
        raise ValueError("Pattern DB format error: root must be dict")

    if "records" not in db:
        db = {
            "meta": db.get("meta", {}) if isinstance(db.get("meta", {}), dict) else {},
            "records": db,
        }

    if not isinstance(db.get("records"), (dict, list)):
        raise ValueError("Pattern DB format error: records must be dict or list")

    return db


def pattern_db_meta() -> Dict[str, Any]:
    """
    給 server.py 使用。
    server.py 目前有：
    from pattern_db import pattern_db_meta

    所以這個函式一定要存在，否則 Render 啟動會失敗。
    """
    try:
        db = load_pattern_db()
        meta = db.get("meta", {})

        if not isinstance(meta, dict):
            meta = {}

        records = db.get("records", {})

        if isinstance(records, dict):
            meta.setdefault("record_count", len(records))
        elif isinstance(records, list):
            meta.setdefault("record_count", len(records))
        else:
            meta.setdefault("record_count", 0)

        meta.setdefault("total_simulated_samples", 0)
        meta.setdefault("source", "RESULT_PATTERN_DB")

        return meta

    except Exception as e:
        return {
            "source": f"PATTERN_DB_META_ERROR:{e}",
            "total_simulated_samples": 0,
            "record_count": 0,
        }


def pattern_db_meta_safe() -> Dict[str, Any]:
    """
    給新版 predictor.py 相容使用。
    不影響舊版 server.py。
    """
    return pattern_db_meta()


def normalize_prob_pair(banker: float, player: float) -> Tuple[float, float]:
    banker = float(banker)
    player = float(player)

    if banker > 1:
        banker = banker / 100.0

    if player > 1:
        player = player / 100.0

    total = banker + player

    if total <= 0:
        banker = BASE_BANKER_NO_TIE
        player = 1.0 - banker
    else:
        banker = banker / total
        player = player / total

    return banker, player


def extract_pattern_record(rec: Dict[str, Any], key: str, window: int, pattern: str) -> Dict[str, Any]:
    rec = dict(rec)

    banker = (
        rec.get("next_banker_rate")
        if rec.get("next_banker_rate") is not None
        else rec.get("banker_prob")
        if rec.get("banker_prob") is not None
        else rec.get("banker_rate")
        if rec.get("banker_rate") is not None
        else rec.get("next_banker_prob")
        if rec.get("next_banker_prob") is not None
        else rec.get("banker")
    )

    player = (
        rec.get("next_player_rate")
        if rec.get("next_player_rate") is not None
        else rec.get("player_prob")
        if rec.get("player_prob") is not None
        else rec.get("player_rate")
        if rec.get("player_rate") is not None
        else rec.get("next_player_prob")
        if rec.get("next_player_prob") is not None
        else rec.get("player")
    )

    if banker is None or player is None:
        raise ValueError(f"Pattern DB record missing banker/player probability: {key}")

    banker_prob, player_prob = normalize_prob_pair(float(banker), float(player))

    sample = (
        rec.get("sample")
        if rec.get("sample") is not None
        else rec.get("sample_count")
        if rec.get("sample_count") is not None
        else rec.get("sample_size")
        if rec.get("sample_size") is not None
        else rec.get("count", 0)
    )

    try:
        sample = int(sample)
    except Exception:
        sample = 0

    rec["feature_key"] = key
    rec["pattern"] = pattern
    rec["window"] = window
    rec["next_banker_rate"] = banker_prob
    rec["next_player_rate"] = player_prob
    rec["banker_prob"] = banker_prob
    rec["player_prob"] = player_prob
    rec["sample"] = sample
    rec.setdefault("source", "RESULT_PATTERN_DB")

    return rec


def get_pattern_record(pattern: str, window: int) -> Optional[Dict[str, Any]]:
    db = load_pattern_db()
    records = db.get("records", {})
    pattern = str(pattern or "").strip().upper()
    window = int(window)

    keys_to_try = [
        f"W{window}:{pattern}",
        f"W{window}_{pattern}",
        f"{window}:{pattern}",
        f"{pattern}:{window}",
        pattern,
    ]

    if isinstance(records, dict):
        for key in keys_to_try:
            rec = records.get(key)

            if isinstance(rec, dict):
                return extract_pattern_record(rec, key, window, pattern)

    if isinstance(records, list):
        for rec in records:
            if not isinstance(rec, dict):
                continue

            rec_pattern = str(
                rec.get("pattern")
                or rec.get("feature_key")
                or rec.get("key")
                or rec.get("pattern_key")
                or ""
            ).strip().upper()

            rec_window = rec.get("window", rec.get("w", window))

            try:
                rec_window = int(rec_window)
            except Exception:
                rec_window = window

            if rec_window == window and rec_pattern in {pattern, *keys_to_try}:
                return extract_pattern_record(rec, rec_pattern or f"W{window}:{pattern}", window, pattern)

    return None


def normalize_result_symbol(result) -> str:
    """
    支援：
    - "莊" / "閒" / "和"
    - "B" / "P" / "T"
    - {"last_result": "..."} / {"result": "..."} / {"winner": "..."}
    - {"player_point": 6, "banker_point": 5}
    """
    if isinstance(result, dict):
        raw = (
            result.get("last_result")
            or result.get("result")
            or result.get("winner")
            or result.get("outcome")
            or result.get("side")
        )

        if raw is not None:
            return normalize_result_symbol(raw)

        pp = (
            result.get("player_point")
            if result.get("player_point") is not None
            else result.get("player")
            if result.get("player") is not None
            else result.get("p")
        )
        bp = (
            result.get("banker_point")
            if result.get("banker_point") is not None
            else result.get("banker")
            if result.get("banker") is not None
            else result.get("b")
        )

        try:
            if pp is not None and bp is not None:
                pp = int(pp)
                bp = int(bp)

                if pp > bp:
                    return "P"
                if bp > pp:
                    return "B"
                return "T"
        except Exception:
            return "T"

        return "T"

    text = str(result or "").strip()

    if text in {"莊", "庄"}:
        return "B"

    if text in {"閒", "闲", "閑"}:
        return "P"

    if text in {"和", "和局"}:
        return "T"

    upper = text.upper()

    if upper in {"B", "BANKER"}:
        return "B"

    if upper in {"P", "PLAYER"}:
        return "P"

    if upper in {"T", "TIE"}:
        return "T"

    return "T"


def build_patterns_from_rounds(rounds: List[Dict[str, Any]]) -> Dict[int, str]:
    symbols = [normalize_result_symbol(r) for r in rounds or []]
    out = {}

    for w in [3, 5, 7]:
        if len(symbols) >= w:
            pat = "".join(symbols[-w:])

            if w == 7:
                # W7資料庫為B/P主型態；T用上一個非T修正，若沒有則略過。
                fixed = []
                last = None

                for s in pat:
                    if s in {"B", "P"}:
                        fixed.append(s)
                        last = s
                    else:
                        fixed.append(last or "B")

                pat = "".join(fixed)

            out[w] = pat

    return out


def pattern_lookup(rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
    patterns = build_patterns_from_rounds(rounds)

    if not patterns:
        return {
            "available": False,
            "banker_prob": BASE_BANKER_NO_TIE,
            "player_prob": 1.0 - BASE_BANKER_NO_TIE,
            "matched": [],
            "sample_size": 0,
            "source": "NO_PATTERN_YET",
        }

    weighted_b = 0.0
    total_weight = 0.0
    matched = []

    # 越長權重越高
    window_weights = {
        3: 0.75,
        5: 1.15,
        7: 1.45,
    }

    for w, pat in patterns.items():
        rec = get_pattern_record(pat, w)

        if not rec:
            continue

        sample = int(rec.get("sample", 0))
        sample_weight = min(max(sample / 10000, 0.45), 2.2)
        weight = window_weights.get(w, 1.0) * sample_weight

        banker = float(rec.get("next_banker_rate", BASE_BANKER_NO_TIE))
        player = float(rec.get("next_player_rate", 1.0 - BASE_BANKER_NO_TIE))
        banker, player = normalize_prob_pair(banker, player)

        weighted_b += banker * weight
        total_weight += weight

        matched.append({
            "window": w,
            "pattern": pat,
            "sample": sample,
            "banker_rate": banker,
            "player_rate": player,
            "source": rec.get("source", "RESULT_PATTERN_DB"),
        })

    if total_weight <= 0:
        return {
            "available": False,
            "banker_prob": BASE_BANKER_NO_TIE,
            "player_prob": 1.0 - BASE_BANKER_NO_TIE,
            "matched": matched,
            "sample_size": 0,
            "source": "PATTERN_NOT_MATCHED",
        }

    banker_prob = weighted_b / total_weight

    return {
        "available": True,
        "banker_prob": banker_prob,
        "player_prob": 1.0 - banker_prob,
        "matched": matched,
        "sample_size": sum(m["sample"] for m in matched),
        "source": "RESULT_PATTERN_DB_3M",
        "pattern": matched[-1]["pattern"] if matched else "",
        "feature_key": matched[-1]["pattern"] if matched else "",
        "window": matched[-1]["window"] if matched else 0,
    }


def clear_pattern_db_cache():
    load_pattern_db.cache_clear()
