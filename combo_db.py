import json
import os
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple

try:
    import config
except Exception:
    config = None

# 重要：優先吃 Render Environment 的 COMBO_DB_PATH；沒有才吃 config；再沒有才用 data/combo_db_3m.json
COMBO_DB_PATH = os.getenv(
    "COMBO_DB_PATH",
    getattr(config, "COMBO_DB_PATH", "data/combo_db_3m.json") if config is not None else "data/combo_db_3m.json",
).strip()

BASE_BANKER_NO_TIE = 0.5000

# 支援英文 key + 中文補牌情境顯示文字，避免「莊閒皆補」無法對到 BOTH_DRAW
SCENARIO_ALIASES = {
    "NONE_DRAW": [
        "NONE_DRAW", "NO_DRAW", "NONE", "NO", "N", "none_draw", "no_draw",
        "雙方不補", "雙方未補", "皆不補", "不補牌", "無補牌", "都不補", "沒補牌",
    ],
    "PLAYER_DRAW": [
        "PLAYER_DRAW", "P_DRAW", "PLAYER", "P", "player_draw",
        "閒補", "閒家補", "閒補牌", "閒補莊不補", "閒補莊未補", "閒家補牌", "閒補莊沒補",
    ],
    "BANKER_DRAW": [
        "BANKER_DRAW", "B_DRAW", "BANKER", "B", "banker_draw",
        "莊補", "莊家補", "莊補牌", "莊補閒不補", "莊補閒未補", "莊家補牌", "莊補閒沒補",
    ],
    "BOTH_DRAW": [
        "BOTH_DRAW", "BOTH", "BOTH_DRAWN", "both_draw",
        "莊閒皆補", "雙方補", "雙方皆補", "兩邊補", "閒莊皆補", "莊閒都補", "雙方補牌", "皆補牌",
    ],
}

_ALIAS_TO_CANONICAL: Dict[str, str] = {}
for canonical, aliases in SCENARIO_ALIASES.items():
    for alias in aliases:
        _ALIAS_TO_CANONICAL[str(alias).strip().upper().replace(" ", "")] = canonical


def _resolve_path(path: str) -> str:
    if os.path.exists(path):
        return path
    cwd_path = os.path.join(os.getcwd(), path)
    if os.path.exists(cwd_path):
        return cwd_path
    # Render 有時工作目錄在 /opt/render/project/src，可再試 src/data
    src_path = os.path.join("/opt/render/project/src", path)
    if os.path.exists(src_path):
        return src_path
    return path


@lru_cache(maxsize=1)
def load_combo_db() -> Dict[str, Any]:
    path = _resolve_path(COMBO_DB_PATH)
    if not os.path.exists(path):
        return {
            "meta": {
                "source": "COMBO_DB_FILE_NOT_FOUND",
                "path": COMBO_DB_PATH,
                "resolved_path": path,
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
                "resolved_path": path,
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
                "resolved_path": path,
                "total_simulated_samples": 0,
                "record_count": 0,
            },
            "records": {},
        }

    # 支援兩種格式：{"meta":..., "records": {...}} 或直接 {...records...}
    if "records" not in db:
        meta = db.get("meta", {}) if isinstance(db.get("meta", {}), dict) else {}
        records = {k: v for k, v in db.items() if k != "meta"}
        db = {"meta": meta, "records": records}

    if not isinstance(db.get("records"), dict):
        db["records"] = {}

    db.setdefault("meta", {})
    if isinstance(db["meta"], dict):
        db["meta"].setdefault("path", COMBO_DB_PATH)
        db["meta"].setdefault("resolved_path", path)
        db["meta"].setdefault("record_count", len(db["records"]))

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
    raw = str(value or "").strip()
    if not raw:
        return "UNKNOWN"
    key = raw.upper().replace(" ", "")
    return _ALIAS_TO_CANONICAL.get(key, key)


def _first_present(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if d.get(k) is not None:
            return d.get(k)
    return default


def extract_scenarios(composition: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    從 point_composition_mc.py 回傳的 composition 裡取補牌情境。
    修正版重點：
    1. 支援 scenario_debug 裡不同欄位名稱。
    2. 支援 top_scenario 當 fallback。
    3. 支援中文補牌情境，例如「莊閒皆補」。
    """
    if not isinstance(composition, dict):
        return []

    items = composition.get("scenario_debug") or []
    out: List[Dict[str, Any]] = []

    if isinstance(items, list):
        for raw in items:
            if not isinstance(raw, dict):
                continue
            scenario_raw = _first_present(
                raw,
                [
                    "scenario", "scenario_key", "scenario_name", "scenario_label",
                    "top_scenario", "draw_scenario", "name", "label", "type",
                ],
            )
            scenario = _scenario_norm(scenario_raw)
            if scenario == "UNKNOWN":
                continue
            try:
                prob = float(
                    _first_present(
                        raw,
                        ["scenario_probability", "scenario_weight", "probability", "prob", "weight", "rate", "pct"],
                        0.0,
                    ) or 0.0
                )
            except Exception:
                prob = 0.0
            # 若 pct 是 0~100，轉 0~1
            if prob > 1:
                prob /= 100.0
            if prob <= 0:
                prob = 0.01
            out.append({
                "scenario": scenario,
                "weight": prob,
                "player_count": int(raw.get("player_count", raw.get("p_count", raw.get("player_cards", 0))) or 0),
                "banker_count": int(raw.get("banker_count", raw.get("b_count", raw.get("banker_cards", 0))) or 0),
                "raw": raw,
            })

    # 若 scenario_debug 沒東西，用 top_scenario fallback。
    if not out:
        top_raw = _first_present(composition, ["top_scenario", "scenario", "scenario_key", "scenario_name", "draw_scenario"])
        top_scenario = _scenario_norm(top_raw)
        if top_scenario != "UNKNOWN":
            out.append({
                "scenario": top_scenario,
                "weight": 1.0,
                "player_count": int(composition.get("player_count", 0) or 0),
                "banker_count": int(composition.get("banker_count", 0) or 0),
                "raw": {"fallback_top_scenario": top_raw},
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
        sc_norm = str(sc).strip()
        sc_upper = sc_norm.upper().replace(" ", "")
        keys.extend([
            f"{pkey}|SC:{sc_upper}",
            f"{pkey}|SC_{sc_upper}",
            f"{pkey}_SC_{sc_upper}",
            f"{pkey}:SC:{sc_upper}",
            f"{pkey}:{sc_upper}",
            f"{pkey}|{sc_upper}",
            f"{pkey}_{sc_upper}",
            f"{pkey}|DRAW:{sc_upper}",
            f"{pkey}|SCENARIO:{sc_upper}",
        ])
        if player_count and banker_count:
            keys.extend([
                f"{pkey}|SC:{sc_upper}|PC:{player_count}|BC:{banker_count}",
                f"{pkey}|PC:{player_count}|BC:{banker_count}|SC:{sc_upper}",
                f"{pkey}_PC{player_count}_BC{banker_count}_SC_{sc_upper}",
            ])

    # 最粗 fallback：只用點數。這很重要，避免條件 key 對不到就永遠樣本 0。
    keys.extend([pkey, f"{pkey}|BASE", f"{pkey}_BASE", f"{pkey}:BASE"])

    seen = set()
    out = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def extract_combo_record(rec: Dict[str, Any], key: str) -> Dict[str, Any]:
    rec = dict(rec)
    banker = _first_present(
        rec,
        ["next_banker_rate", "banker_prob", "banker_rate", "b_rate", "B", "b", "banker", "莊", "庄"],
    )
    player = _first_present(
        rec,
        ["next_player_rate", "player_prob", "player_rate", "p_rate", "P", "p", "player", "閒", "闲"],
    )
    if banker is None or player is None:
        raise ValueError(f"Combo DB record missing banker/player probability: {key}")
    banker_prob, player_prob = normalize_prob_pair(float(banker), float(player))
    sample = _first_present(rec, ["sample", "sample_size", "no_tie_sample", "count", "n", "total", "samples"], 0)
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
            sample = int(_first_present(rec, ["sample", "sample_size", "no_tie_sample", "count", "n", "total", "samples"], 0) or 0)
        except Exception:
            sample = 0
        if sample < int(min_sample):
            continue
        try:
            return extract_combo_record(rec, key)
        except Exception:
            continue
    return None


def combo_lookup(player_point: int, banker_point: int, rounds: Optional[List[Any]] = None, composition: Optional[Dict[str, Any]] = None, min_sample: int = 80) -> Dict[str, Any]:
    """
    V9.3 主查詢：用「點數 + 補牌情境」查 combo_db_3m.json。
    rounds 只保留相容，不依賴前面路單。
    """
    meta = combo_db_meta()
    db = load_combo_db()
    records = db.get("records", {})
    pkey = point_key(player_point, banker_point)

    if not isinstance(records, dict) or not records:
        return {
            "available": False,
            "feature_key": pkey,
            "banker_prob": BASE_BANKER_NO_TIE,
            "player_prob": 1.0 - BASE_BANKER_NO_TIE,
            "source": "COMBO_DB_NOT_MATCHED_OR_EMPTY",
            "sample_size": 0,
            "total_simulated_samples": int(meta.get("total_simulated_samples", 0) or 0),
            "candidate_keys": [pkey],
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
        candidate_keys.extend(keys[:10])
        rec = _find_record_by_keys(records, keys, min_sample=min_sample)
        if not rec:
            continue

        sample = int(rec.get("sample", 0) or 0)
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
            "feature_key": pkey,
            "banker_prob": BASE_BANKER_NO_TIE,
            "player_prob": 1.0 - BASE_BANKER_NO_TIE,
            "source": "POINT_CONDITION_COMBO_DB_NOT_MATCHED",
            "sample_size": 0,
            "total_simulated_samples": int(meta.get("total_simulated_samples", 0) or 0),
            "candidate_keys": candidate_keys[:20],
            "matched_records": [],
            "top_scenario": scenarios[0].get("scenario", "UNKNOWN") if scenarios else "UNKNOWN",
        }

    banker_prob = weighted_b / total_weight
    player_prob = 1.0 - banker_prob
    banker_prob, player_prob = normalize_prob_pair(banker_prob, player_prob)
    best = matched_records[0] if matched_records else {}

    return {
        "available": True,
        "feature_key": best.get("feature_key", pkey),
        "banker_prob": banker_prob,
        "player_prob": player_prob,
        "source": "POINT_CONDITION_COMBO_DB_V9_3_KEYFIX",
        "sample_size": sample_total,
        "total_simulated_samples": int(meta.get("total_simulated_samples", 0) or 0),
        "candidate_keys": candidate_keys[:20],
        "matched_records": matched_records,
        "top_scenario": scenarios[0].get("scenario", "UNKNOWN") if scenarios else "UNKNOWN",
    }


def clear_combo_db_cache():
    load_combo_db.cache_clear()
