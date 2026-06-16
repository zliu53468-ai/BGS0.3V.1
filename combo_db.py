import json
import os
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple

try:
    from config import COMBO_DB_PATH
except Exception:
    COMBO_DB_PATH = os.getenv("COMBO_DB_PATH", "data/combo_db_3m.json")

BASE_BANKER_NO_TIE = 0.5068

SCENARIO_ALIASES = {
    "NONE_DRAW": ["NONE_DRAW", "NO_DRAW", "none_draw", "no_draw"],
    "PLAYER_DRAW": ["PLAYER_DRAW", "P_DRAW", "player_draw"],
    "BANKER_DRAW": ["BANKER_DRAW", "B_DRAW", "banker_draw"],
    "BOTH_DRAW": ["BOTH_DRAW", "BOTH", "both_draw"],
}


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

    try:
        with open(path, "r", encoding="utf-8") as f:
            db = json.load(f)
    except Exception as e:
        return {
            "meta": {
                "source": f"COMBO_DB_LOAD_ERROR:{e}",
                "path": COMBO_DB_PATH,
                "total_simulated_samples": 0,
                "record_count": 0,
            },
            "records": {},
        }

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
    meta.setdefault("path", COMBO_DB_PATH)
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


def point_key(player_point: int, banker_point: int) -> str:
    return f"P{int(player_point)}_B{int(banker_point)}"


def _scenario_norm(value: Any) -> str:
    s = str(value or "").strip().upper()
    if s in {"NO_DRAW", "NONE", "NONE_DRAW"}:
        return "NONE_DRAW"
    if s in {"PLAYER_DRAW", "P_DRAW", "PLAYER", "P"}:
        return "PLAYER_DRAW"
    if s in {"BANKER_DRAW", "B_DRAW", "BANKER", "B"}:
        return "BANKER_DRAW"
    if s in {"BOTH_DRAW", "BOTH", "BOTH_DRAWN"}:
        return "BOTH_DRAW"
    return s or "UNKNOWN"


def extract_scenarios(composition: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(composition, dict):
        return []
    items = composition.get("scenario_debug") or []
    out: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return out

    for raw in items:
        if not isinstance(raw, dict):
            continue
        scenario = _scenario_norm(raw.get("scenario"))
        if scenario == "UNKNOWN":
            continue
        try:
            prob = float(raw.get("scenario_probability", raw.get("scenario_weight", 0.0)) or 0.0)
        except Exception:
            prob = 0.0
        if prob <= 0:
            prob = 0.01
        out.append({
            "scenario": scenario,
            "weight": prob,
            "player_count": int(raw.get("player_count", 0) or 0),
            "banker_count": int(raw.get("banker_count", 0) or 0),
            "raw": raw,
        })

    total = sum(x["weight"] for x in out)
    if total > 0:
        for x in out:
            x["weight"] = x["weight"] / total

    out.sort(key=lambda x: x.get("weight", 0.0), reverse=True)
    return out


def candidate_keys_for_scenario(player_point: int, banker_point: int, scenario: str, player_count: int = 0, banker_count: int = 0) -> List[str]:
    pkey = point_key(player_point, banker_point)
    scenario = _scenario_norm(scenario)
    keys: List[str] = []

    aliases = SCENARIO_ALIASES.get(scenario, [scenario])

    for sc in aliases:
        keys.extend([
            f"{pkey}|SC:{sc}",
            f"{pkey}|SC_{sc}",
            f"{pkey}_SC_{sc}",
            f"{pkey}:{sc}",
            f"{pkey}|{sc}",
        ])
        if player_count and banker_count:
            keys.extend([
                f"{pkey}|SC:{sc}|PC:{player_count}|BC:{banker_count}",
                f"{pkey}|PC:{player_count}|BC:{banker_count}|SC:{sc}",
                f"{pkey}_PC{player_count}_BC{banker_count}_SC_{sc}",
            ])

    # 最粗 fallback：只用點數。
    keys.extend([pkey, f"{pkey}|BASE", f"{pkey}_BASE"])

    seen = set()
    out = []
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
        if rec.get("banker_rate") is not None
        else rec.get("B")
        if rec.get("B") is not None
        else rec.get("b")
    )
    player = (
        rec.get("next_player_rate")
        if rec.get("next_player_rate") is not None
        else rec.get("player_prob")
        if rec.get("player_prob") is not None
        else rec.get("player_rate")
        if rec.get("player_rate") is not None
        else rec.get("P")
        if rec.get("P") is not None
        else rec.get("p")
    )

    if banker is None or player is None:
        raise ValueError(f"Combo DB record missing banker/player probability: {key}")

    banker_prob, player_prob = normalize_prob_pair(float(banker), float(player))
    sample = rec.get("sample", rec.get("sample_size", rec.get("no_tie_sample", rec.get("count", 0))))
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
    rec.setdefault("source", "POINT_CONDITION_COMBO_DB")
    return rec


def _find_record_by_keys(records: Dict[str, Any], keys: List[str], min_sample: int) -> Optional[Dict[str, Any]]:
    for key in keys:
        rec = records.get(key)
        if not isinstance(rec, dict):
            continue
        try:
            sample = int(rec.get("sample", rec.get("sample_size", rec.get("no_tie_sample", rec.get("count", 0)))) or 0)
        except Exception:
            sample = 0
        if sample < int(min_sample):
            continue
        try:
            return extract_combo_record(rec, key)
        except Exception:
            continue
    return None


def combo_lookup(
    player_point: int,
    banker_point: int,
    rounds: Optional[List[Any]] = None,
    composition: Optional[Dict[str, Any]] = None,
    min_sample: int = 80,
) -> Dict[str, Any]:
    """
    V9 主查詢：使用「點數 + 補牌情境」去查 300 萬組條件資料庫。
    rounds 只保留參數相容，主邏輯不依賴前面路單。
    """
    meta = combo_db_meta()
    db = load_combo_db()
    records = db.get("records", {})
    if not isinstance(records, dict) or not records:
        return {
            "available": False,
            "feature_key": point_key(player_point, banker_point),
            "banker_prob": BASE_BANKER_NO_TIE,
            "player_prob": 1.0 - BASE_BANKER_NO_TIE,
            "source": "COMBO_DB_NOT_MATCHED_OR_EMPTY",
            "sample_size": 0,
            "total_simulated_samples": int(meta.get("total_simulated_samples", 0) or 0),
            "candidate_keys": [point_key(player_point, banker_point)],
            "matched_records": [],
            "top_scenario": "UNKNOWN",
        }

    scenarios = extract_scenarios(composition)
    if not scenarios:
        scenarios = [{"scenario": "NONE_DRAW", "weight": 1.0, "player_count": 0, "banker_count": 0, "raw": {}}]

    weighted_b = 0.0
    total_weight = 0.0
    sample_total = 0
    matched_records: List[Dict[str, Any]] = []
    candidate_keys: List[str] = []

    for sc in scenarios:
        keys = candidate_keys_for_scenario(
            player_point,
            banker_point,
            sc.get("scenario", "UNKNOWN"),
            player_count=int(sc.get("player_count", 0) or 0),
            banker_count=int(sc.get("banker_count", 0) or 0),
        )
        candidate_keys.extend(keys[:6])
        rec = _find_record_by_keys(records, keys, min_sample=min_sample)
        if not rec:
            continue

        sample = int(rec.get("sample", 0) or 0)
        # 情境權重 * 樣本信任權重。樣本越大權重越高，但避免過度極端。
        scenario_weight = float(sc.get("weight", 0.0) or 0.0)
        sample_weight = min(max(sample / 10000.0, 0.45), 2.2)
        w = max(0.0001, scenario_weight * sample_weight)

        weighted_b += float(rec["banker_prob"]) * w
        total_weight += w
        sample_total += sample
        matched_records.append({
            "scenario": sc.get("scenario"),
            "weight": w,
            "scenario_weight": scenario_weight,
            "feature_key": rec.get("feature_key"),
            "banker_prob": rec.get("banker_prob"),
            "player_prob": rec.get("player_prob"),
            "sample": sample,
            "source": rec.get("source", "POINT_CONDITION_COMBO_DB"),
        })

    seen = set()
    candidate_keys = [x for x in candidate_keys if not (x in seen or seen.add(x))]

    if total_weight <= 0:
        return {
            "available": False,
            "feature_key": point_key(player_point, banker_point),
            "banker_prob": BASE_BANKER_NO_TIE,
            "player_prob": 1.0 - BASE_BANKER_NO_TIE,
            "source": "POINT_CONDITION_COMBO_DB_NOT_MATCHED",
            "sample_size": 0,
            "total_simulated_samples": int(meta.get("total_simulated_samples", 0) or 0),
            "candidate_keys": candidate_keys[:12],
            "matched_records": [],
            "top_scenario": scenarios[0].get("scenario", "UNKNOWN") if scenarios else "UNKNOWN",
        }

    banker_prob = weighted_b / total_weight
    player_prob = 1.0 - banker_prob
    banker_prob, player_prob = normalize_prob_pair(banker_prob, player_prob)

    best = matched_records[0] if matched_records else {}
    return {
        "available": True,
        "feature_key": best.get("feature_key", point_key(player_point, banker_point)),
        "banker_prob": banker_prob,
        "player_prob": player_prob,
        "source": "POINT_CONDITION_COMBO_DB_V9",
        "sample_size": sample_total,
        "total_simulated_samples": int(meta.get("total_simulated_samples", 0) or 0),
        "candidate_keys": candidate_keys[:12],
        "matched_records": matched_records,
        "top_scenario": scenarios[0].get("scenario", "UNKNOWN") if scenarios else "UNKNOWN",
    }


def clear_combo_db_cache():
    load_combo_db.cache_clear()
