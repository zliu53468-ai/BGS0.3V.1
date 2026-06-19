# -*- coding: utf-8 -*-
"""
combo_db.py - V9.4 compatible connector for data/combo_db_3m.json

Purpose:
- Keep the original database/prediction logic intact.
- Only fix the connector layer so combo_db_3m.json can be read reliably.
- Support both JSON formats:
    1) {"meta": {...}, "records": {"P2_B7|SC:BOTH_DRAW": {...}}}
    2) {"__meta__": {...}, "P2_B7|SC:BOTH_DRAW": {...}}
- Support Chinese scenario names such as 「莊閒皆補」 -> BOTH_DRAW.
- Support multiple key styles and fallback to BASE / point-only records.
"""

import json
import os
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple

try:
    import config
except Exception:
    config = None

COMBO_DB_PATH = os.getenv(
    "COMBO_DB_PATH",
    getattr(config, "COMBO_DB_PATH", "data/combo_db_3m.json") if config is not None else "data/combo_db_3m.json",
).strip()

BASE_BANKER_NO_TIE = 0.5000

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

META_KEYS = {"meta", "__meta__", "metadata", "_meta"}


def _resolve_path(path: str) -> str:
    candidates = [
        path,
        os.path.join(os.getcwd(), path),
        os.path.join("/opt/render/project/src", path),
        os.path.join(os.path.dirname(__file__), path),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
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
                "record_count": 0,
                "total_simulated_samples": 0,
            },
            "records": {},
        }

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_db = json.load(f)
    except Exception as e:
        return {
            "meta": {
                "source": f"COMBO_DB_LOAD_ERROR:{e}",
                "path": COMBO_DB_PATH,
                "resolved_path": path,
                "record_count": 0,
                "total_simulated_samples": 0,
            },
            "records": {},
        }

    if not isinstance(raw_db, dict):
        return {
            "meta": {
                "source": "COMBO_DB_FORMAT_ERROR_NOT_DICT",
                "path": COMBO_DB_PATH,
                "resolved_path": path,
                "record_count": 0,
                "total_simulated_samples": 0,
            },
            "records": {},
        }

    # 支援 {"records": {...}} 與直接 key-value 兩種資料庫格式。
    if isinstance(raw_db.get("records"), dict):
        records = raw_db.get("records", {})
        meta = raw_db.get("meta") if isinstance(raw_db.get("meta"), dict) else {}
    else:
        meta = {}
        for mk in META_KEYS:
            if isinstance(raw_db.get(mk), dict):
                meta.update(raw_db.get(mk, {}))
        records = {k: v for k, v in raw_db.items() if str(k) not in META_KEYS}

    if not isinstance(records, dict):
        records = {}
    if not isinstance(meta, dict):
        meta = {}

    # 建立大小寫/空白不敏感索引，避免 key 格式小差異導致找不到。
    normalized_index = {}
    for k in records.keys():
        nk = _key_norm(k)
        if nk not in normalized_index:
            normalized_index[nk] = k

    meta.setdefault("source", "COMBO_DB")
    meta.setdefault("path", COMBO_DB_PATH)
    meta.setdefault("resolved_path", path)
    meta.setdefault("record_count", len(records))
    meta.setdefault("total_simulated_samples", 0)

    return {
        "meta": meta,
        "records": records,
        "normalized_index": normalized_index,
    }


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


def _key_norm(value: Any) -> str:
    return str(value or "").strip().upper().replace(" ", "")


def _scenario_norm(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "UNKNOWN"
    key = _key_norm(raw)
    return _ALIAS_TO_CANONICAL.get(key, key)


def _first_present(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    if not isinstance(d, dict):
        return default
    for k in keys:
        if d.get(k) is not None:
            return d.get(k)
    return default


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


def _get_record(records: Dict[str, Any], normalized_index: Dict[str, str], key: str) -> Optional[Dict[str, Any]]:
    rec = records.get(key)
    if isinstance(rec, dict):
        return rec
    real_key = normalized_index.get(_key_norm(key))
    if real_key:
        rec = records.get(real_key)
        if isinstance(rec, dict):
            return rec
    return None


def _sample_of(rec: Dict[str, Any]) -> int:
    try:
        return int(_first_present(rec, ["sample", "sample_size", "no_tie_sample", "no_tie_sample_size", "count", "n", "total", "samples"], 0) or 0)
    except Exception:
        return 0


def extract_scenarios(composition: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

    # 優先 canonical，再補 aliases。資料庫 generator 使用 P2_B7|SC:BOTH_DRAW。
    candidates = [scenario] + [str(a).strip().upper().replace(" ", "") for a in aliases]
    seen_sc = set()
    scenario_names = []
    for sc in candidates:
        if sc and sc not in seen_sc:
            seen_sc.add(sc)
            scenario_names.append(sc)

    for sc in scenario_names:
        keys.extend([
            f"{pkey}|SC:{sc}",
            f"{pkey}|SC_{sc}",
            f"{pkey}_SC_{sc}",
            f"{pkey}:SC:{sc}",
            f"{pkey}:{sc}",
            f"{pkey}|{sc}",
            f"{pkey}_{sc}",
            f"{pkey}|DRAW:{sc}",
            f"{pkey}|SCENARIO:{sc}",
        ])
        if player_count and banker_count:
            keys.extend([
                f"{pkey}|SC:{sc}|PC:{player_count}|BC:{banker_count}",
                f"{pkey}|PC:{player_count}|BC:{banker_count}|SC:{sc}",
                f"{pkey}_PC{player_count}_BC{banker_count}_SC_{sc}",
            ])

    # Fallback：BASE 與點數本身。若你的 DB 只有 point-only，也能接上。
    keys.extend([f"{pkey}|BASE", f"{pkey}_BASE", f"{pkey}:BASE", pkey])

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
    sample = _sample_of(rec)

    rec["feature_key"] = key
    rec["next_banker_rate"] = banker_prob
    rec["next_player_rate"] = player_prob
    rec["banker_prob"] = banker_prob
    rec["player_prob"] = player_prob
    rec["sample"] = sample
    rec.setdefault("source", "POINT_CONDITION_COMBO_DB")
    return rec


def _find_record_by_keys(records: Dict[str, Any], normalized_index: Dict[str, str], keys: List[str], min_sample: int) -> Optional[Dict[str, Any]]:
    for key in keys:
        raw = _get_record(records, normalized_index, key)
        if not isinstance(raw, dict):
            continue
        sample = _sample_of(raw)
        if sample < int(min_sample):
            continue
        try:
            return extract_combo_record(raw, key)
        except Exception:
            continue
    return None


def _aggregate_point_records(records: Dict[str, Any], normalized_index: Dict[str, str], player_point: int, banker_point: int, min_sample: int) -> Optional[Dict[str, Any]]:
    """最後保險：若精準情境 key 找不到，聚合該點數所有 scenario/base records。"""
    pkey = point_key(player_point, banker_point)
    prefixes = [f"{pkey}|SC:", f"{pkey}|SC_", f"{pkey}_SC_", f"{pkey}|", f"{pkey}_"]
    weighted_b = 0.0
    total_w = 0.0
    sample_total = 0
    used = []

    for key, raw in records.items():
        if not isinstance(raw, dict):
            continue
        skey = str(key)
        if skey == pkey or any(skey.startswith(px) for px in prefixes):
            # 排除非本點數的近似 key，例如 P2_B70
            if not (skey == pkey or skey.startswith(f"{pkey}|") or skey.startswith(f"{pkey}_")):
                continue
            sample = _sample_of(raw)
            if sample < int(min_sample):
                continue
            try:
                rec = extract_combo_record(raw, skey)
            except Exception:
                continue
            w = max(1.0, min(sample / 10000.0, 3.0))
            weighted_b += float(rec["banker_prob"]) * w
            total_w += w
            sample_total += sample
            used.append({
                "feature_key": skey,
                "sample": sample,
                "banker_prob": rec.get("banker_prob"),
                "player_prob": rec.get("player_prob"),
                "source": rec.get("source", "POINT_CONDITION_COMBO_DB"),
            })

    if total_w <= 0:
        return None

    banker_prob = weighted_b / total_w
    player_prob = 1.0 - banker_prob
    banker_prob, player_prob = normalize_prob_pair(banker_prob, player_prob)
    return {
        "available": True,
        "feature_key": pkey,
        "banker_prob": banker_prob,
        "player_prob": player_prob,
        "source": "POINT_CONDITION_COMBO_DB_POINT_AGGREGATE_FALLBACK",
        "sample": sample_total,
        "sample_size": sample_total,
        "matched_records": used[:20],
    }


def combo_lookup(
    player_point: int,
    banker_point: int,
    rounds: Optional[List[Any]] = None,
    composition: Optional[Dict[str, Any]] = None,
    min_sample: int = 80,
) -> Dict[str, Any]:
    """
    V9.4 主查詢：使用「點數 + 補牌情境」查 combo_db_3m.json。
    rounds 只保留參數相容，主邏輯不依賴前面路單。
    """
    meta = combo_db_meta()
    db = load_combo_db()
    records = db.get("records", {})
    normalized_index = db.get("normalized_index", {})
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
            "meta": meta,
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
        candidate_keys.extend(keys[:12])
        rec = _find_record_by_keys(records, normalized_index, keys, min_sample=min_sample)
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

    # 若情境 key 都找不到，最後用該點數全部可用條件聚合，不改原資料，只做讀取 fallback。
    if total_weight <= 0:
        agg = _aggregate_point_records(records, normalized_index, player_point, banker_point, min_sample=min_sample)
        if agg:
            return {
                "available": True,
                "feature_key": agg.get("feature_key", pkey),
                "banker_prob": agg.get("banker_prob", BASE_BANKER_NO_TIE),
                "player_prob": agg.get("player_prob", 1.0 - BASE_BANKER_NO_TIE),
                "source": agg.get("source", "POINT_CONDITION_COMBO_DB_POINT_AGGREGATE_FALLBACK"),
                "sample_size": int(agg.get("sample_size", 0) or 0),
                "total_simulated_samples": int(meta.get("total_simulated_samples", 0) or 0),
                "candidate_keys": candidate_keys[:24],
                "matched_records": agg.get("matched_records", []),
                "top_scenario": scenarios[0].get("scenario", "UNKNOWN") if scenarios else "UNKNOWN",
                "meta": meta,
            }

        return {
            "available": False,
            "feature_key": pkey,
            "banker_prob": BASE_BANKER_NO_TIE,
            "player_prob": 1.0 - BASE_BANKER_NO_TIE,
            "source": "POINT_CONDITION_COMBO_DB_NOT_MATCHED",
            "sample_size": 0,
            "total_simulated_samples": int(meta.get("total_simulated_samples", 0) or 0),
            "candidate_keys": candidate_keys[:24],
            "matched_records": [],
            "top_scenario": scenarios[0].get("scenario", "UNKNOWN") if scenarios else "UNKNOWN",
            "meta": meta,
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
        "source": "POINT_CONDITION_COMBO_DB_V9_4_CONNECTED",
        "sample_size": sample_total,
        "total_simulated_samples": int(meta.get("total_simulated_samples", 0) or 0),
        "candidate_keys": candidate_keys[:24],
        "matched_records": matched_records,
        "top_scenario": scenarios[0].get("scenario", "UNKNOWN") if scenarios else "UNKNOWN",
        "meta": meta,
    }


def clear_combo_db_cache():
    load_combo_db.cache_clear()


if __name__ == "__main__":
    print("META =", combo_db_meta())
    for pp, bp, sc in [(2, 7, "莊閒皆補"), (2, 7, "BOTH_DRAW"), (6, 5, "PLAYER_DRAW")]:
        print("LOOKUP", pp, bp, sc, "=>")
        print(combo_lookup(pp, bp, composition={"top_scenario": sc}, min_sample=1))
