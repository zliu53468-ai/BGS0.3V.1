import json
import os
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple

try:
    from config import RESULT_PATTERN_DB_PATH
except Exception:
    RESULT_PATTERN_DB_PATH = os.getenv("RESULT_PATTERN_DB_PATH", "data/result_pattern_db_3m.json")

BASE_BANKER_NO_TIE = 0.5068


def _resolve_path(path: str) -> str:
    if os.path.exists(path):
        return path
    cwd_path = os.path.join(os.getcwd(), path)
    if os.path.exists(cwd_path):
        return cwd_path
    return path


@lru_cache(maxsize=1)
def load_pattern_db() -> Dict[str, Any]:
    path = _resolve_path(RESULT_PATTERN_DB_PATH)

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

    if not isinstance(db.get("records"), dict):
        raise ValueError("Pattern DB format error: records must be dict")

    return db


def pattern_db_meta_safe() -> Dict[str, Any]:
    """
    Safe meta accessor used by predictor and server health checks.
    Always returns a dict, even when DB file is missing or malformed.
    """
    try:
        db = load_pattern_db()
        meta = db.get("meta", {})
        if not isinstance(meta, dict):
            meta = {}

        records = db.get("records", {})
        meta.setdefault("record_count", len(records) if isinstance(records, dict) else 0)
        meta.setdefault("total_simulated_samples", 0)
        meta.setdefault("source", "RESULT_PATTERN_DB")
        meta.setdefault("path", RESULT_PATTERN_DB_PATH)
        return meta

    except Exception as e:
        return {
            "source": f"PATTERN_DB_META_ERROR:{e}",
            "total_simulated_samples": 0,
            "record_count": 0,
            "path": RESULT_PATTERN_DB_PATH,
        }


def pattern_db_meta() -> Dict[str, Any]:
    """
    Compatibility wrapper for server.py:
        from pattern_db import pattern_db_meta

    Your Render crash happened because server.py imported this name but
    the deployed pattern_db.py only exposed pattern_db_meta_safe().
    """
    return pattern_db_meta_safe()


def get_pattern_record(pattern: str, window: int) -> Optional[Dict[str, Any]]:
    try:
        db = load_pattern_db()
        records = db.get("records", {})
        rec = records.get(f"W{int(window)}:{pattern}")
        return dict(rec) if isinstance(rec, dict) else None
    except Exception:
        return None


def normalize_result_symbol(result: Any) -> str:
    if isinstance(result, dict):
        raw = result.get("last_result") or result.get("result") or result.get("winner") or result.get("side")
        if raw is not None:
            return normalize_result_symbol(raw)

        pp = result.get("player_point", result.get("player", result.get("p")))
        bp = result.get("banker_point", result.get("banker", result.get("b")))

        try:
            if pp is None or bp is None:
                return "T"

            pp = int(pp)
            bp = int(bp)

            if pp > bp:
                return "P"
            if bp > pp:
                return "B"
            return "T"
        except Exception:
            return "T"

    text = str(result or "").strip()
    up = text.upper()

    if text in {"莊", "庄"} or up in {"B", "BANKER"}:
        return "B"

    if text in {"閒", "閑", "闲"} or up in {"P", "PLAYER"}:
        return "P"

    if text in {"和", "和局"} or up in {"T", "TIE"}:
        return "T"

    return "T"


def normalize_w7_pattern(pat: str) -> Optional[str]:
    fixed: List[str] = []
    last: Optional[str] = None

    for s in pat:
        if s in {"B", "P"}:
            fixed.append(s)
            last = s
        else:
            if last is None:
                return None
            fixed.append(last)

    return "".join(fixed)


def build_patterns_from_rounds(rounds: List[Dict[str, Any]]) -> Dict[int, str]:
    symbols = [normalize_result_symbol(r) for r in rounds]
    out: Dict[int, str] = {}

    for w in [3, 5, 7]:
        if len(symbols) >= w:
            pat = "".join(symbols[-w:])

            if w == 7:
                fixed = normalize_w7_pattern(pat)
                if not fixed:
                    continue
                pat = fixed

            out[w] = pat

    return out


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


def _extract_sample_size(rec: Dict[str, Any]) -> int:
    if not isinstance(rec, dict):
        return 0

    for key in ("sample_size", "sample", "no_tie_sample", "count", "total", "n"):
        try:
            value = int(rec.get(key, 0) or 0)
            if value > 0:
                return value
        except Exception:
            continue

    return 0


def pattern_lookup(rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
    patterns = build_patterns_from_rounds(rounds or [])

    if not patterns:
        return {
            "available": False,
            "banker_prob": BASE_BANKER_NO_TIE,
            "player_prob": 1.0 - BASE_BANKER_NO_TIE,
            "matched": [],
            "sample_size": 0,
            "source": "NO_PATTERN_YET",
            "feature_key": "NO_PATTERN",
            "pattern": "",
            "window": 0,
        }

    weighted_b = 0.0
    total_weight = 0.0
    matched = []

    # 越長權重越高，但樣本數仍然會約束。
    window_weights = {3: 0.75, 5: 1.15, 7: 1.45}

    for w, pat in patterns.items():
        rec = get_pattern_record(pat, w)

        if not rec:
            continue

        sample = _extract_sample_size(rec)
        sample_weight = min(max(sample / 10000, 0.45), 2.2)
        weight = window_weights.get(w, 1.0) * sample_weight

        banker_raw = rec.get("next_banker_rate", rec.get("banker_prob", BASE_BANKER_NO_TIE))
        player_raw = rec.get("next_player_rate", rec.get("player_prob", 1.0 - BASE_BANKER_NO_TIE))
        banker, player = normalize_prob_pair(float(banker_raw), float(player_raw))

        weighted_b += banker * weight
        total_weight += weight

        matched.append({
            "window": w,
            "pattern": pat,
            "sample": sample,
            "sample_size": sample,
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
            "feature_key": "PATTERN_NOT_MATCHED",
            "pattern": "".join([patterns[k] for k in sorted(patterns.keys())][-1:]),
            "window": 0,
        }

    banker_prob = weighted_b / total_weight
    player_prob = 1.0 - banker_prob
    best = matched[-1] if matched else {}
    sample_size = sum(int(m.get("sample", 0) or 0) for m in matched)

    return {
        "available": sample_size > 0,
        "banker_prob": banker_prob,
        "player_prob": player_prob,
        "matched": matched,
        "sample_size": sample_size,
        "source": "REAL_SHOE_RESULT_PATTERN_DB",
        "feature_key": f"W{best.get('window', 0)}:{best.get('pattern', '')}",
        "pattern": best.get("pattern", ""),
        "window": int(best.get("window", 0) or 0),
    }


def clear_pattern_db_cache():
    load_pattern_db.cache_clear()
