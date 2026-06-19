# -*- coding: utf-8 -*-
"""
road_profile_db.py - V9.4 無記憶牌路資料庫比對層

設計重點：
- 不保存、不延續用戶每次在 LINE / 前端輸入的歷史紀錄。
- 只用「當前這一局點數 + 補牌情境」去查 data/road_profile_db_3m.json。
- 牌路名稱（單跳、雙跳、一房兩廳、長龍、同點重複）是資料庫裡的相似路段統計，不是用戶目前真實路單。
- 可作為 predictor.py 的額外參考層，不取代 point_db / combo_db / 補牌 MC。
"""

import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

try:
    import config
except Exception:
    config = None

ROAD_PROFILE_DB_PATH = os.getenv(
    "ROAD_PROFILE_DB_PATH",
    getattr(config, "ROAD_PROFILE_DB_PATH", "data/road_profile_db_3m.json") if config is not None else "data/road_profile_db_3m.json",
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

ROAD_PROFILE_ZH = {
    "SINGLE_JUMP": "單跳",
    "DOUBLE_JUMP": "雙跳",
    "ONE_ROOM_TWO_HALLS": "一房兩廳",
    "LONG_BANKER": "莊長龍",
    "LONG_PLAYER": "閒長龍",
    "LONG_DRAGON": "長龍",
    "SAME_POINT_REPEAT": "同點重複",
    "NEUTRAL": "中性路段",
}

ROAD_PROFILE_ALIASES = {
    "SINGLE_JUMP": ["SINGLE_JUMP", "SINGLE", "JUMP", "單跳", "單跳路"],
    "DOUBLE_JUMP": ["DOUBLE_JUMP", "DOUBLE", "雙跳", "雙跳路"],
    "ONE_ROOM_TWO_HALLS": ["ONE_ROOM_TWO_HALLS", "ONE_TWO", "1R2H", "一房兩廳"],
    "LONG_BANKER": ["LONG_BANKER", "BANKER_LONG", "LONG_B", "莊長龍", "莊龍"],
    "LONG_PLAYER": ["LONG_PLAYER", "PLAYER_LONG", "LONG_P", "閒長龍", "閒龍"],
    "LONG_DRAGON": ["LONG_DRAGON", "LONG", "長龍"],
    "SAME_POINT_REPEAT": ["SAME_POINT_REPEAT", "SAME_POINT", "REPEAT_POINT", "同點重複", "同一點數重複"],
    "NEUTRAL": ["NEUTRAL", "BASE", "中性", "一般"],
}

_ALIAS_TO_SCENARIO: Dict[str, str] = {}
for canonical, aliases in SCENARIO_ALIASES.items():
    for alias in aliases:
        _ALIAS_TO_SCENARIO[str(alias).strip().upper().replace(" ", "")] = canonical

_ALIAS_TO_PROFILE: Dict[str, str] = {}
for canonical, aliases in ROAD_PROFILE_ALIASES.items():
    for alias in aliases:
        _ALIAS_TO_PROFILE[str(alias).strip().upper().replace(" ", "")] = canonical


def _norm_scenario(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "UNKNOWN"
    key = raw.upper().replace(" ", "")
    return _ALIAS_TO_SCENARIO.get(key, key)


def _norm_profile(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "NEUTRAL"
    key = raw.upper().replace(" ", "")
    return _ALIAS_TO_PROFILE.get(key, key)


def _resolve_path(path: str) -> str:
    if os.path.exists(path):
        return path
    cwd_path = os.path.join(os.getcwd(), path)
    if os.path.exists(cwd_path):
        return cwd_path
    render_path = os.path.join("/opt/render/project/src", path)
    if os.path.exists(render_path):
        return render_path
    return path


@lru_cache(maxsize=1)
def load_road_profile_db() -> Dict[str, Any]:
    path = _resolve_path(ROAD_PROFILE_DB_PATH)
    if not os.path.exists(path):
        return {
            "meta": {
                "source": "ROAD_PROFILE_DB_FILE_NOT_FOUND",
                "path": ROAD_PROFILE_DB_PATH,
                "resolved_path": path,
                "record_count": 0,
                "total_simulated_samples": 0,
            },
            "records": {},
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            db = json.load(f)
    except Exception as e:
        return {
            "meta": {
                "source": f"ROAD_PROFILE_DB_LOAD_ERROR:{e}",
                "path": ROAD_PROFILE_DB_PATH,
                "resolved_path": path,
                "record_count": 0,
                "total_simulated_samples": 0,
            },
            "records": {},
        }

    if not isinstance(db, dict):
        return {
            "meta": {"source": "ROAD_PROFILE_DB_FORMAT_ERROR", "path": ROAD_PROFILE_DB_PATH, "resolved_path": path, "record_count": 0},
            "records": {},
        }

    if "records" not in db:
        meta = db.get("meta", db.get("__meta__", {}))
        if not isinstance(meta, dict):
            meta = {}
        records = {k: v for k, v in db.items() if k not in {"meta", "__meta__"}}
        db = {"meta": meta, "records": records}

    if not isinstance(db.get("records"), dict):
        db["records"] = {}

    db.setdefault("meta", {})
    if isinstance(db["meta"], dict):
        db["meta"].setdefault("path", ROAD_PROFILE_DB_PATH)
        db["meta"].setdefault("resolved_path", path)
        db["meta"].setdefault("record_count", len(db["records"]))
        db["meta"].setdefault("source", "ROAD_PROFILE_DB")
    return db


def road_profile_db_meta() -> Dict[str, Any]:
    db = load_road_profile_db()
    meta = db.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    records = db.get("records", {})
    meta.setdefault("record_count", len(records) if isinstance(records, dict) else 0)
    meta.setdefault("total_simulated_samples", 0)
    meta.setdefault("source", "ROAD_PROFILE_DB")
    meta.setdefault("path", ROAD_PROFILE_DB_PATH)
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


def _first_present(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if d.get(k) is not None:
            return d.get(k)
    return default


def _extract_scenarios(composition: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(composition, dict):
        return []
    out: List[Dict[str, Any]] = []
    items = composition.get("scenario_debug") or []
    if isinstance(items, list):
        for raw in items:
            if not isinstance(raw, dict):
                continue
            scenario_raw = _first_present(raw, ["scenario", "scenario_key", "scenario_name", "scenario_label", "top_scenario", "draw_scenario", "name", "label", "type"])
            scenario = _norm_scenario(scenario_raw)
            if scenario == "UNKNOWN":
                continue
            try:
                weight = float(_first_present(raw, ["scenario_probability", "scenario_weight", "probability", "prob", "weight", "rate", "pct"], 0) or 0)
            except Exception:
                weight = 0.0
            if weight > 1:
                weight /= 100.0
            if weight <= 0:
                weight = 0.01
            out.append({"scenario": scenario, "weight": weight, "raw": raw})

    if not out:
        top_raw = _first_present(composition, ["top_scenario", "scenario", "scenario_key", "scenario_name", "draw_scenario"])
        top = _norm_scenario(top_raw)
        if top != "UNKNOWN":
            out.append({"scenario": top, "weight": 1.0, "raw": {"fallback_top_scenario": top_raw}})

    total = sum(x["weight"] for x in out)
    if total > 0:
        for x in out:
            x["weight"] = x["weight"] / total
    out.sort(key=lambda x: x.get("weight", 0), reverse=True)
    return out


def _candidate_keys(player_point: int, banker_point: int, scenario: str) -> List[str]:
    pkey = point_key(player_point, banker_point)
    scenario = _norm_scenario(scenario)
    profiles = list(ROAD_PROFILE_ZH.keys())
    keys: List[str] = []
    scenario_aliases = SCENARIO_ALIASES.get(scenario, [scenario])
    for sc in scenario_aliases:
        sc = str(sc).strip().upper().replace(" ", "")
        for profile in profiles:
            keys.extend([
                f"{pkey}|SC:{sc}|ROAD:{profile}",
                f"{pkey}|ROAD:{profile}|SC:{sc}",
                f"{pkey}|SC_{sc}|ROAD_{profile}",
                f"{pkey}_{sc}_{profile}",
            ])
    # fallback：不含補牌情境，只含點數 + 牌路類型。
    for profile in profiles:
        keys.extend([
            f"{pkey}|ROAD:{profile}",
            f"{pkey}|ROAD_{profile}",
            f"{pkey}_{profile}",
        ])
    # 最粗 fallback：點數層。
    keys.extend([f"{pkey}|ROAD_PROFILE", f"{pkey}|BASE", f"{pkey}_BASE", pkey])
    seen = set()
    out = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _extract_record(rec: Dict[str, Any], key: str) -> Dict[str, Any]:
    rec = dict(rec)
    banker = _first_present(rec, ["next_banker_rate", "banker_prob", "banker_rate", "b_rate", "B", "b", "banker", "莊", "庄"])
    player = _first_present(rec, ["next_player_rate", "player_prob", "player_rate", "p_rate", "P", "p", "player", "閒", "闲"])
    if banker is None or player is None:
        raise ValueError(f"road profile record missing banker/player probability: {key}")
    banker_prob, player_prob = normalize_prob_pair(float(banker), float(player))
    sample = _first_present(rec, ["sample", "sample_size", "no_tie_sample", "count", "n", "total", "samples"], 0)
    try:
        sample = int(sample)
    except Exception:
        sample = 0
    profile = _norm_profile(_first_present(rec, ["road_profile", "profile", "road", "type"], "NEUTRAL"))
    rec["feature_key"] = key
    rec["banker_prob"] = banker_prob
    rec["player_prob"] = player_prob
    rec["sample"] = sample
    rec["road_profile"] = profile
    rec["road_profile_zh"] = rec.get("road_profile_zh", ROAD_PROFILE_ZH.get(profile, profile))
    rec.setdefault("source", "ROAD_PROFILE_DB")
    return rec


def _records_by_keys(records: Dict[str, Any], keys: List[str], min_sample: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen_keys = set()
    for key in keys:
        if key in seen_keys:
            continue
        seen_keys.add(key)
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
            out.append(_extract_record(rec, key))
        except Exception:
            continue
    return out


def road_profile_lookup(
    player_point: int,
    banker_point: int,
    composition: Optional[Dict[str, Any]] = None,
    min_sample: int = 50,
) -> Dict[str, Any]:
    """
    無記憶查詢：只用當前點數 + 補牌情境，回傳資料庫相似路段統計。
    不讀 rounds，不保存用戶歷史。
    """
    meta = road_profile_db_meta()
    db = load_road_profile_db()
    records = db.get("records", {})
    pkey = point_key(player_point, banker_point)

    if not isinstance(records, dict) or not records:
        return {
            "available": False,
            "feature_key": pkey,
            "banker_prob": BASE_BANKER_NO_TIE,
            "player_prob": 1.0 - BASE_BANKER_NO_TIE,
            "sample_size": 0,
            "total_simulated_samples": int(meta.get("total_simulated_samples", 0) or 0),
            "source": "ROAD_PROFILE_DB_EMPTY_OR_NOT_FOUND",
            "top_road_profile": "NEUTRAL",
            "top_road_profile_zh": ROAD_PROFILE_ZH["NEUTRAL"],
            "profile_distribution": [],
            "candidate_keys": [pkey],
            "use_user_history": False,
        }

    scenarios = _extract_scenarios(composition)
    if not scenarios:
        scenarios = [{"scenario": "UNKNOWN", "weight": 1.0, "raw": {}}]

    weighted_b = 0.0
    total_weight = 0.0
    sample_total = 0
    matched: List[Dict[str, Any]] = []
    candidate_keys: List[str] = []

    for sc in scenarios:
        keys = _candidate_keys(player_point, banker_point, sc.get("scenario", "UNKNOWN"))
        candidate_keys.extend(keys[:20])
        recs = _records_by_keys(records, keys, min_sample=min_sample)
        if not recs:
            continue
        scenario_weight = float(sc.get("weight", 1.0) or 1.0)
        for rec in recs:
            sample = int(rec.get("sample", 0) or 0)
            sample_weight = min(max(sample / 5000.0, 0.35), 2.5)
            w = max(0.0001, scenario_weight * sample_weight)
            weighted_b += float(rec["banker_prob"]) * w
            total_weight += w
            sample_total += sample
            matched.append({
                "feature_key": rec.get("feature_key"),
                "road_profile": rec.get("road_profile", "NEUTRAL"),
                "road_profile_zh": rec.get("road_profile_zh", ROAD_PROFILE_ZH.get(rec.get("road_profile", "NEUTRAL"), rec.get("road_profile", "NEUTRAL"))),
                "banker_prob": rec.get("banker_prob"),
                "player_prob": rec.get("player_prob"),
                "sample": sample,
                "weight": w,
                "source": rec.get("source", "ROAD_PROFILE_DB"),
                "same_point_repeat_avg": rec.get("same_point_repeat_avg", 0),
            })

    seen = set()
    candidate_keys = [x for x in candidate_keys if not (x in seen or seen.add(x))]

    if total_weight <= 0:
        return {
            "available": False,
            "feature_key": pkey,
            "banker_prob": BASE_BANKER_NO_TIE,
            "player_prob": 1.0 - BASE_BANKER_NO_TIE,
            "sample_size": 0,
            "total_simulated_samples": int(meta.get("total_simulated_samples", 0) or 0),
            "source": "ROAD_PROFILE_DB_NOT_MATCHED",
            "top_road_profile": "NEUTRAL",
            "top_road_profile_zh": ROAD_PROFILE_ZH["NEUTRAL"],
            "profile_distribution": [],
            "candidate_keys": candidate_keys[:30],
            "use_user_history": False,
        }

    banker_prob = weighted_b / total_weight
    player_prob = 1.0 - banker_prob
    banker_prob, player_prob = normalize_prob_pair(banker_prob, player_prob)

    # 取樣本最高的 profile 作為顯示用「資料庫最常見路段」。
    profile_samples: Dict[str, Dict[str, Any]] = {}
    for m in matched:
        p = m.get("road_profile", "NEUTRAL")
        z = m.get("road_profile_zh", ROAD_PROFILE_ZH.get(p, p))
        bucket = profile_samples.setdefault(p, {"road_profile": p, "road_profile_zh": z, "sample": 0, "banker_weighted": 0.0})
        sample = int(m.get("sample", 0) or 0)
        bucket["sample"] += sample
        bucket["banker_weighted"] += float(m.get("banker_prob", BASE_BANKER_NO_TIE)) * sample

    distribution = []
    for p, bucket in profile_samples.items():
        sample = int(bucket["sample"] or 0)
        b = bucket["banker_weighted"] / sample if sample > 0 else BASE_BANKER_NO_TIE
        distribution.append({
            "road_profile": p,
            "road_profile_zh": bucket["road_profile_zh"],
            "sample": sample,
            "banker_prob": b,
            "player_prob": 1.0 - b,
        })
    distribution.sort(key=lambda x: x.get("sample", 0), reverse=True)
    top = distribution[0] if distribution else {"road_profile": "NEUTRAL", "road_profile_zh": ROAD_PROFILE_ZH["NEUTRAL"]}

    same_point_repeat_avg = 0.0
    repeat_weight = 0.0
    for m in matched:
        try:
            sp = float(m.get("same_point_repeat_avg", 0) or 0)
            w = float(m.get("sample", 0) or 0)
            same_point_repeat_avg += sp * w
            repeat_weight += w
        except Exception:
            pass
    if repeat_weight > 0:
        same_point_repeat_avg /= repeat_weight

    return {
        "available": True,
        "feature_key": top.get("feature_key", pkey),
        "banker_prob": banker_prob,
        "player_prob": player_prob,
        "sample_size": sample_total,
        "total_simulated_samples": int(meta.get("total_simulated_samples", 0) or 0),
        "source": "ROAD_PROFILE_DB_V9_4_MEMORYLESS",
        "top_road_profile": top.get("road_profile", "NEUTRAL"),
        "top_road_profile_zh": top.get("road_profile_zh", ROAD_PROFILE_ZH["NEUTRAL"]),
        "profile_distribution": distribution[:8],
        "matched_records": matched[:20],
        "candidate_keys": candidate_keys[:30],
        "same_point_repeat_avg": same_point_repeat_avg,
        "use_user_history": False,
    }


def clear_road_profile_db_cache():
    load_road_profile_db.cache_clear()
