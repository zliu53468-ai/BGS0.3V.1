# -*- coding: utf-8 -*-
"""
road_profile_db.py - V9.6 無記憶牌路規律 Profile 強化版

設計重點：
- 不保存、不延續用戶每次在 LINE / 前端輸入的歷史紀錄。
- 不吃真實路單歷史，不需要暖機，第一局也可以直接用當前點數判斷。
- 只用「當前這一局點數 + 補牌情境」去查 data/road_profile_db_3m.json。
- 新增支援更多牌路 profile：
  單跳、雙跳、一房兩廳、兩房一廳、短龍、斬龍、跟龍、轉跳、長龍、同點重複。
- 若資料庫內有對應 profile key，會優先採用；若沒有，仍會 fallback 到原本點數 / BASE 邏輯。
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

# -----------------------------
# Environment helpers
# -----------------------------

def _env_bool(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}

def _env_float(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return float(default)

def _env_int(name: str, default: str) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return int(default)

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# 啟用 profile 強化
ROAD_PATTERN_ENHANCE = _env_bool("ROAD_PATTERN_ENHANCE", "1")

# 整體牌路敏感度。建議 1.00 ~ 1.18，不要太高避免蓋過點數。
ROAD_PATTERN_SENSITIVITY = _env_float("ROAD_PATTERN_SENSITIVITY", "1.08")

# 短龍不要過度追，避免 1~2 顆就被誤認成會繼續龍。
ROAD_SHORT_DRAGON_SHRINK = _env_float("ROAD_SHORT_DRAGON_SHRINK", "0.88")

# 規律跳 / 房廳 / 轉跳 profile 的額外權重。
ROAD_SINGLE_JUMP_BOOST = _env_float("ROAD_SINGLE_JUMP_BOOST", "1.10")
ROAD_DOUBLE_JUMP_BOOST = _env_float("ROAD_DOUBLE_JUMP_BOOST", "1.10")
ROAD_ROOM_PATTERN_BOOST = _env_float("ROAD_ROOM_PATTERN_BOOST", "1.12")
ROAD_CUT_DRAGON_BOOST = _env_float("ROAD_CUT_DRAGON_BOOST", "1.12")
ROAD_FOLLOW_DRAGON_BOOST = _env_float("ROAD_FOLLOW_DRAGON_BOOST", "1.05")
ROAD_TURN_JUMP_BOOST = _env_float("ROAD_TURN_JUMP_BOOST", "1.12")
ROAD_LONG_DRAGON_BOOST = _env_float("ROAD_LONG_DRAGON_BOOST", "1.04")

# profile 差距放大上限，避免牌路層過度影響 point_db。
ROAD_PROFILE_GAP_AMP_MAX = _env_float("ROAD_PROFILE_GAP_AMP_MAX", "1.22")

# 若找不到精準 profile key，可否聚合同一點數下所有 ROAD 記錄。
ROAD_PROFILE_POINT_AGGREGATE_FALLBACK = _env_bool("ROAD_PROFILE_POINT_AGGREGATE_FALLBACK", "1")


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
    "TWO_ROOM_ONE_HALL": "兩房一廳",
    "SHORT_DRAGON": "短龍",
    "CUT_DRAGON": "斬龍",
    "FOLLOW_DRAGON": "跟龍",
    "TURN_JUMP": "轉跳",
    "LONG_BANKER": "莊長龍",
    "LONG_PLAYER": "閒長龍",
    "LONG_DRAGON": "長龍",
    "SAME_POINT_REPEAT": "同點重複",
    "NEUTRAL": "中性路段",
}

ROAD_PROFILE_ALIASES = {
    "SINGLE_JUMP": [
        "SINGLE_JUMP", "SINGLE", "JUMP", "SINGLE_JUMP_ROAD",
        "單跳", "單跳路", "單跳規律", "跳路",
    ],
    "DOUBLE_JUMP": [
        "DOUBLE_JUMP", "DOUBLE", "DOUBLE_JUMP_ROAD", "TWO_JUMP",
        "雙跳", "雙跳路", "雙跳規律",
    ],
    "ONE_ROOM_TWO_HALLS": [
        "ONE_ROOM_TWO_HALLS", "ONE_ROOM_TWO_HALL", "ONE_TWO", "1R2H", "ONE_R_TWO_H",
        "一房兩廳", "一房二廳", "一房兩庭", "一房二庭",
    ],
    "TWO_ROOM_ONE_HALL": [
        "TWO_ROOM_ONE_HALL", "TWO_ROOM_ONE_HALLS", "TWO_ONE", "2R1H", "TWO_R_ONE_H",
        "兩房一廳", "二房一廳", "兩房一庭", "二房一庭",
    ],
    "SHORT_DRAGON": [
        "SHORT_DRAGON", "SHORT_LONG", "SHORT", "MINI_DRAGON",
        "短龍", "短連", "短連莊", "短連閒",
    ],
    "CUT_DRAGON": [
        "CUT_DRAGON", "DRAGON_CUT", "CUT", "BREAK_DRAGON", "DRAGON_BREAK",
        "斬龍", "斷龍", "切龍", "破龍",
    ],
    "FOLLOW_DRAGON": [
        "FOLLOW_DRAGON", "DRAGON_FOLLOW", "FOLLOW", "FOLLOW_LONG",
        "跟龍", "順龍", "跟長龍", "追龍",
    ],
    "TURN_JUMP": [
        "TURN_JUMP", "TURN", "REVERSAL_JUMP", "DRAGON_TURN", "SWITCH_JUMP", "TRANSITION",
        "轉跳", "轉向", "轉龍", "反跳", "跳轉",
    ],
    "LONG_BANKER": [
        "LONG_BANKER", "BANKER_LONG", "LONG_B", "B_LONG",
        "莊長龍", "莊龍", "莊連", "連莊",
    ],
    "LONG_PLAYER": [
        "LONG_PLAYER", "PLAYER_LONG", "LONG_P", "P_LONG",
        "閒長龍", "閒龍", "閒連", "連閒",
    ],
    "LONG_DRAGON": [
        "LONG_DRAGON", "LONG", "DRAGON", "LONG_ROAD",
        "長龍", "長連", "連龍",
    ],
    "SAME_POINT_REPEAT": [
        "SAME_POINT_REPEAT", "SAME_POINT", "REPEAT_POINT", "POINT_REPEAT",
        "同點重複", "同一點數重複", "點數重複",
    ],
    "NEUTRAL": [
        "NEUTRAL", "BASE", "NORMAL", "MIXED",
        "中性", "中性路段", "一般", "無規律",
    ],
}

_ALIAS_TO_SCENARIO: Dict[str, str] = {}
for canonical, aliases in SCENARIO_ALIASES.items():
    for alias in aliases:
        _ALIAS_TO_SCENARIO[str(alias).strip().upper().replace(" ", "")] = canonical

_ALIAS_TO_PROFILE: Dict[str, str] = {}
for canonical, aliases in ROAD_PROFILE_ALIASES.items():
    for alias in aliases:
        _ALIAS_TO_PROFILE[str(alias).strip().upper().replace(" ", "")] = canonical


def _key_norm(value: Any) -> str:
    return str(value or "").strip().upper().replace(" ", "")


def _norm_scenario(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "UNKNOWN"
    key = _key_norm(raw)
    return _ALIAS_TO_SCENARIO.get(key, key)


def _norm_profile(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "NEUTRAL"
    key = _key_norm(raw)
    return _ALIAS_TO_PROFILE.get(key, key)


def _resolve_path(path: str) -> str:
    """
    Render 常見工作目錄是 /opt/render/project/src。
    這裡多加幾個候選路徑，避免 road_profile_db_3m.json 明明在 data/ 裡卻讀不到。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    base = os.path.basename(path)

    candidates = [
        path,
        os.path.join(cwd, path),
        os.path.join(here, path),
        os.path.join("/opt/render/project/src", path),
        os.path.join(cwd, "data", base),
        os.path.join(here, "data", base),
        os.path.join("/opt/render/project/src/data", base),
    ]

    seen = set()
    for p in candidates:
        if not p:
            continue
        p = os.path.normpath(p)
        if p in seen:
            continue
        seen.add(p)
        if os.path.exists(p):
            return p

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
            "normalized_index": {},
        }

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_db = json.load(f)
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
            "normalized_index": {},
        }

    if not isinstance(raw_db, dict):
        return {
            "meta": {
                "source": "ROAD_PROFILE_DB_FORMAT_ERROR",
                "path": ROAD_PROFILE_DB_PATH,
                "resolved_path": path,
                "record_count": 0,
                "total_simulated_samples": 0,
            },
            "records": {},
            "normalized_index": {},
        }

    meta = {}
    for mk in ("meta", "__meta__", "metadata", "_meta"):
        if isinstance(raw_db.get(mk), dict):
            meta.update(raw_db.get(mk, {}))

    if isinstance(raw_db.get("records"), dict):
        records = raw_db.get("records", {})
        if isinstance(raw_db.get("meta"), dict):
            meta.update(raw_db.get("meta", {}))
    elif isinstance(raw_db.get("data"), dict):
        records = raw_db.get("data", {})
    elif isinstance(raw_db.get("items"), dict):
        records = raw_db.get("items", {})
    else:
        records = {k: v for k, v in raw_db.items() if k not in {"meta", "__meta__", "metadata", "_meta", "records", "data", "items"}}

    if not isinstance(records, dict):
        records = {}

    normalized_index: Dict[str, str] = {}
    for k in records.keys():
        nk = _key_norm(k)
        if nk and nk not in normalized_index:
            normalized_index[nk] = k

    total_samples = (
        meta.get("total_simulated_samples")
        or meta.get("total_samples")
        or meta.get("sample_total")
        or meta.get("samples")
        or 0
    )

    meta.setdefault("path", ROAD_PROFILE_DB_PATH)
    meta.setdefault("resolved_path", path)
    meta.setdefault("record_count", len(records))
    meta.setdefault("total_simulated_samples", int(total_samples or 0))
    meta.setdefault("source", "ROAD_PROFILE_DB_V9_6_MEMORYLESS_PATTERN")

    return {
        "meta": meta,
        "records": records,
        "normalized_index": normalized_index,
    }


def road_profile_db_meta() -> Dict[str, Any]:
    db = load_road_profile_db()
    meta = db.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    records = db.get("records", {})
    meta.setdefault("record_count", len(records) if isinstance(records, dict) else 0)
    meta.setdefault("total_simulated_samples", 0)
    meta.setdefault("source", "ROAD_PROFILE_DB_V9_6_MEMORYLESS_PATTERN")
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
    if not isinstance(d, dict):
        return default
    for k in keys:
        if d.get(k) is not None:
            return d.get(k)
    return default


def _sample_of(rec: Dict[str, Any]) -> int:
    try:
        return int(_first_present(rec, ["sample", "sample_size", "no_tie_sample", "count", "n", "total", "samples"], 0) or 0)
    except Exception:
        return 0


def _extract_scenarios(composition: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(composition, dict):
        return []

    out: List[Dict[str, Any]] = []
    items = composition.get("scenario_debug") or []

    if isinstance(items, list):
        for raw in items:
            if not isinstance(raw, dict):
                continue
            scenario_raw = _first_present(
                raw,
                ["scenario", "scenario_key", "scenario_name", "scenario_label", "top_scenario", "draw_scenario", "name", "label", "type", "scenario_zh"],
            )
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
        top_raw = _first_present(composition, ["top_scenario", "scenario", "scenario_key", "scenario_name", "draw_scenario", "scenario_zh"])
        top = _norm_scenario(top_raw)
        if top != "UNKNOWN":
            out.append({"scenario": top, "weight": 1.0, "raw": {"fallback_top_scenario": top_raw}})

    total = sum(x["weight"] for x in out)
    if total > 0:
        for x in out:
            x["weight"] = x["weight"] / total

    out.sort(key=lambda x: x.get("weight", 0), reverse=True)
    return out


def _profile_order() -> List[str]:
    """
    查詢順序很重要：
    規律 profile 優先於長龍與中性，避免全部被 NEUTRAL / BASE 吃掉。
    """
    return [
        "SINGLE_JUMP",
        "DOUBLE_JUMP",
        "ONE_ROOM_TWO_HALLS",
        "TWO_ROOM_ONE_HALL",
        "TURN_JUMP",
        "CUT_DRAGON",
        "FOLLOW_DRAGON",
        "SHORT_DRAGON",
        "LONG_BANKER",
        "LONG_PLAYER",
        "LONG_DRAGON",
        "SAME_POINT_REPEAT",
        "NEUTRAL",
    ]


def _candidate_keys(player_point: int, banker_point: int, scenario: str) -> List[str]:
    pkey = point_key(player_point, banker_point)
    scenario = _norm_scenario(scenario)
    keys: List[str] = []

    scenario_aliases = SCENARIO_ALIASES.get(scenario, [scenario])
    profiles = _profile_order()

    # 精準：點數 + 補牌情境 + profile
    for sc in scenario_aliases:
        sc = _key_norm(sc)
        for profile in profiles:
            aliases = ROAD_PROFILE_ALIASES.get(profile, [profile])
            profile_names = [profile] + [_key_norm(a) for a in aliases]
            seen_profiles = set()
            for pf in profile_names:
                if not pf or pf in seen_profiles:
                    continue
                seen_profiles.add(pf)
                keys.extend([
                    f"{pkey}|SC:{sc}|ROAD:{pf}",
                    f"{pkey}|ROAD:{pf}|SC:{sc}",
                    f"{pkey}|SC_{sc}|ROAD_{pf}",
                    f"{pkey}|SCENARIO:{sc}|ROAD:{pf}",
                    f"{pkey}_{sc}_{pf}",
                    f"{pkey}|{sc}|{pf}",
                ])

    # fallback：點數 + profile
    for profile in profiles:
        aliases = ROAD_PROFILE_ALIASES.get(profile, [profile])
        profile_names = [profile] + [_key_norm(a) for a in aliases]
        seen_profiles = set()
        for pf in profile_names:
            if not pf or pf in seen_profiles:
                continue
            seen_profiles.add(pf)
            keys.extend([
                f"{pkey}|ROAD:{pf}",
                f"{pkey}|ROAD_{pf}",
                f"{pkey}|PROFILE:{pf}",
                f"{pkey}_{pf}",
                f"{pkey}|{pf}",
            ])

    # 最粗 fallback：點數層
    keys.extend([f"{pkey}|ROAD_PROFILE", f"{pkey}|BASE", f"{pkey}_BASE", pkey])

    seen = set()
    out = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


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


def _extract_record(rec: Dict[str, Any], key: str) -> Dict[str, Any]:
    rec = dict(rec)
    banker = _first_present(rec, ["next_banker_rate", "banker_prob", "banker_rate", "b_rate", "B", "b", "banker", "莊", "庄"])
    player = _first_present(rec, ["next_player_rate", "player_prob", "player_rate", "p_rate", "P", "p", "player", "閒", "闲"])

    if banker is None or player is None:
        raise ValueError(f"road profile record missing banker/player probability: {key}")

    banker_prob, player_prob = normalize_prob_pair(float(banker), float(player))
    sample = _sample_of(rec)

    profile_raw = _first_present(rec, ["road_profile", "profile", "road", "type", "pattern", "road_pattern"], None)
    if profile_raw is None:
        # 從 key 嘗試反推 profile
        profile_raw = _profile_from_key(key)

    profile = _norm_profile(profile_raw)

    rec["feature_key"] = key
    rec["banker_prob"] = banker_prob
    rec["player_prob"] = player_prob
    rec["sample"] = sample
    rec["road_profile"] = profile
    rec["road_profile_zh"] = rec.get("road_profile_zh", ROAD_PROFILE_ZH.get(profile, profile))
    rec.setdefault("source", "ROAD_PROFILE_DB")
    return rec


def _profile_from_key(key: str) -> str:
    nk = _key_norm(key)
    for profile, aliases in ROAD_PROFILE_ALIASES.items():
        if _key_norm(profile) in nk:
            return profile
        for a in aliases:
            if _key_norm(a) in nk:
                return profile
    return "NEUTRAL"


def _profile_weight_multiplier(profile: str) -> float:
    profile = _norm_profile(profile)
    if profile == "SINGLE_JUMP":
        return ROAD_SINGLE_JUMP_BOOST
    if profile == "DOUBLE_JUMP":
        return ROAD_DOUBLE_JUMP_BOOST
    if profile in {"ONE_ROOM_TWO_HALLS", "TWO_ROOM_ONE_HALL"}:
        return ROAD_ROOM_PATTERN_BOOST
    if profile == "CUT_DRAGON":
        return ROAD_CUT_DRAGON_BOOST
    if profile == "FOLLOW_DRAGON":
        return ROAD_FOLLOW_DRAGON_BOOST
    if profile == "TURN_JUMP":
        return ROAD_TURN_JUMP_BOOST
    if profile in {"LONG_BANKER", "LONG_PLAYER", "LONG_DRAGON"}:
        return ROAD_LONG_DRAGON_BOOST
    if profile == "SHORT_DRAGON":
        return ROAD_SHORT_DRAGON_SHRINK
    return 1.0


def _enhance_profile_probability(banker_prob: float, profile: str) -> float:
    """
    對 profile 層的方向差距做小幅校正。
    - 單跳 / 雙跳 / 房廳 / 斬龍 / 轉跳：略提高敏感度。
    - 短龍：略收斂，避免 1~2 顆短龍過度跟龍。
    """
    if not ROAD_PATTERN_ENHANCE:
        return banker_prob

    profile = _norm_profile(profile)
    mult = _profile_weight_multiplier(profile)
    amp = _clamp(mult * ROAD_PATTERN_SENSITIVITY, 0.70, ROAD_PROFILE_GAP_AMP_MAX)

    # 短龍特別處理：收斂，不追太快。
    if profile == "SHORT_DRAGON":
        amp = _clamp(ROAD_SHORT_DRAGON_SHRINK, 0.50, 1.00)

    gap = float(banker_prob) - BASE_BANKER_NO_TIE
    enhanced = BASE_BANKER_NO_TIE + gap * amp
    return _clamp(enhanced, 0.38, 0.62)


def _records_by_keys(
    records: Dict[str, Any],
    normalized_index: Dict[str, str],
    keys: List[str],
    min_sample: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen_keys = set()

    for key in keys:
        if key in seen_keys:
            continue
        seen_keys.add(key)

        raw = _get_record(records, normalized_index, key)
        if not isinstance(raw, dict):
            continue

        sample = _sample_of(raw)
        if sample < int(min_sample):
            continue

        try:
            out.append(_extract_record(raw, key))
        except Exception:
            continue

    return out


def _aggregate_point_road_records(
    records: Dict[str, Any],
    player_point: int,
    banker_point: int,
    min_sample: int,
) -> List[Dict[str, Any]]:
    """
    若精準 key 找不到，用同一點數底下所有 ROAD / PROFILE 記錄聚合。
    這不是用戶歷史，只是資料庫同點數 profile fallback。
    """
    if not ROAD_PROFILE_POINT_AGGREGATE_FALLBACK:
        return []

    pkey = point_key(player_point, banker_point)
    out: List[Dict[str, Any]] = []

    for key, raw in records.items():
        if not isinstance(raw, dict):
            continue

        skey = str(key)
        if not (skey == pkey or skey.startswith(f"{pkey}|") or skey.startswith(f"{pkey}_")):
            continue

        # 只收 ROAD / PROFILE / BASE 類資料，避免誤吃其他 DB。
        nk = _key_norm(skey)
        if "ROAD" not in nk and "PROFILE" not in nk and "BASE" not in nk and nk != _key_norm(pkey):
            continue

        sample = _sample_of(raw)
        if sample < int(min_sample):
            continue

        try:
            out.append(_extract_record(raw, skey))
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
    無記憶查詢：
    只用當前點數 + 補牌情境，回傳資料庫相似路段統計。
    不讀 rounds，不保存用戶歷史。
    """
    meta = road_profile_db_meta()
    db = load_road_profile_db()
    records = db.get("records", {})
    normalized_index = db.get("normalized_index", {})
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
            "pattern_enhance": ROAD_PATTERN_ENHANCE,
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
        candidate_keys.extend(keys[:30])
        recs = _records_by_keys(records, normalized_index, keys, min_sample=min_sample)

        # 若精準 key 沒有，再同點數聚合。
        if not recs:
            recs = _aggregate_point_road_records(records, player_point, banker_point, min_sample=min_sample)

        if not recs:
            continue

        scenario_weight = float(sc.get("weight", 1.0) or 1.0)

        for rec in recs:
            sample = int(rec.get("sample", 0) or 0)
            profile = _norm_profile(rec.get("road_profile", "NEUTRAL"))

            raw_banker = float(rec.get("banker_prob", BASE_BANKER_NO_TIE))
            enhanced_banker = _enhance_profile_probability(raw_banker, profile)
            enhanced_player = 1.0 - enhanced_banker

            sample_weight = min(max(sample / 5000.0, 0.35), 2.5)
            profile_mult = _profile_weight_multiplier(profile) if ROAD_PATTERN_ENHANCE else 1.0
            profile_mult = _clamp(profile_mult, 0.55, 1.35)

            w = max(0.0001, scenario_weight * sample_weight * profile_mult)

            weighted_b += enhanced_banker * w
            total_weight += w
            sample_total += sample

            matched.append({
                "feature_key": rec.get("feature_key"),
                "road_profile": profile,
                "road_profile_zh": rec.get("road_profile_zh", ROAD_PROFILE_ZH.get(profile, profile)),
                "banker_prob": enhanced_banker,
                "player_prob": enhanced_player,
                "raw_banker_prob": raw_banker,
                "raw_player_prob": float(rec.get("player_prob", 1.0 - raw_banker)),
                "sample": sample,
                "weight": w,
                "profile_multiplier": profile_mult,
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
            "candidate_keys": candidate_keys[:40],
            "use_user_history": False,
            "pattern_enhance": ROAD_PATTERN_ENHANCE,
        }

    banker_prob = weighted_b / total_weight
    player_prob = 1.0 - banker_prob
    banker_prob, player_prob = normalize_prob_pair(banker_prob, player_prob)

    # 取樣本最高的 profile 作為顯示用「資料庫最常見路段」。
    profile_samples: Dict[str, Dict[str, Any]] = {}
    for m in matched:
        p = _norm_profile(m.get("road_profile", "NEUTRAL"))
        z = m.get("road_profile_zh", ROAD_PROFILE_ZH.get(p, p))
        bucket = profile_samples.setdefault(
            p,
            {"road_profile": p, "road_profile_zh": z, "sample": 0, "banker_weighted": 0.0, "weighted_sum": 0.0},
        )
        sample = int(m.get("sample", 0) or 0)
        w = float(m.get("weight", 0.0) or 0.0)
        bucket["sample"] += sample
        bucket["banker_weighted"] += float(m.get("banker_prob", BASE_BANKER_NO_TIE)) * max(sample, 1)
        bucket["weighted_sum"] += w

    distribution = []
    for p, bucket in profile_samples.items():
        sample = int(bucket["sample"] or 0)
        b = bucket["banker_weighted"] / sample if sample > 0 else BASE_BANKER_NO_TIE
        b, pl = normalize_prob_pair(b, 1.0 - b)
        distribution.append({
            "road_profile": p,
            "road_profile_zh": bucket["road_profile_zh"],
            "sample": sample,
            "banker_prob": b,
            "player_prob": pl,
            "weighted_sum": bucket.get("weighted_sum", 0.0),
        })

    # 顯示排序：先看樣本，再看規律 profile 權重。
    distribution.sort(
        key=lambda x: (
            x.get("sample", 0),
            _profile_weight_multiplier(x.get("road_profile", "NEUTRAL")),
        ),
        reverse=True,
    )

    top = distribution[0] if distribution else {
        "road_profile": "NEUTRAL",
        "road_profile_zh": ROAD_PROFILE_ZH["NEUTRAL"],
    }

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
        "source": "ROAD_PROFILE_DB_V9_6_MEMORYLESS_PATTERN",
        "top_road_profile": top.get("road_profile", "NEUTRAL"),
        "top_road_profile_zh": top.get("road_profile_zh", ROAD_PROFILE_ZH["NEUTRAL"]),
        "profile_distribution": distribution[:12],
        "matched_records": matched[:30],
        "candidate_keys": candidate_keys[:40],
        "same_point_repeat_avg": same_point_repeat_avg,
        "use_user_history": False,
        "pattern_enhance": ROAD_PATTERN_ENHANCE,
        "pattern_sensitivity": ROAD_PATTERN_SENSITIVITY,
    }


def clear_road_profile_db_cache():
    load_road_profile_db.cache_clear()


if __name__ == "__main__":
    print("META =", road_profile_db_meta())
    for pp, bp, sc in [
        (2, 7, "莊閒皆補"),
        (6, 5, "PLAYER_DRAW"),
        (0, 0, "雙方不補"),
    ]:
        print("=" * 60)
        print("LOOKUP", pp, bp, sc)
        print(road_profile_lookup(pp, bp, composition={"top_scenario": sc}, min_sample=1))
