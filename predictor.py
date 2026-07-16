"""LINE-compatible V5.2 three-tier independent point predictor."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
import os
import re
import secrets

from particle_filter_points import (
    DB_HOLDOUT,
    V5IndependentBaccaratEngine,
    clear_runtime_caches,
)
from shoe_state_db import get_shoe_state_database

PATH_SUFFIX = {"N": 0, "P": 1, "B": 2, "D": 3}
PATH_NAMES = ("none", "player_only", "banker_only", "both")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default)).strip()))
    except Exception:
        return default


RANDOMIZE_EACH_CALL = _env_bool("PF_RANDOMIZE_EACH_CALL", True)
FIXED_RUN_SEED = _env_int("PF_FIXED_RUN_SEED", 0, 0)
DEBUG_V5_RESULT = _env_bool("PF_DEBUG_V5_RESULT", False)

# The engine is stateless. Reusing one configured instance avoids reconstructing
# the settings dictionary for every LINE message; request-specific randomness is
# still supplied to analyze() through the run seed.
_ENGINE = V5IndependentBaccaratEngine()


def parse_point_observation(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, Mapping):
        player = value.get("player", value.get("P", value.get("閒")))
        banker = value.get("banker", value.get("B", value.get("莊")))
        suffix = str(value.get("path_suffix", "") or "").strip().upper()
        try:
            return {
                "player": int(player) % 10,
                "banker": int(banker) % 10,
                "path": PATH_SUFFIX.get(suffix),
                "suffix": suffix if suffix in PATH_SUFFIX else "",
            }
        except Exception:
            return None
    text = str(value or "").strip().upper()
    compact = re.fullmatch(r"([0-9])([0-9])([NPBD])?", text)
    if compact:
        suffix = compact.group(3) or ""
        return {
            "player": int(compact.group(1)),
            "banker": int(compact.group(2)),
            "path": PATH_SUFFIX.get(suffix),
            "suffix": suffix,
        }
    patterns = [
        r"(?:P|PLAYER|閒|闲)\s*([0-9])\D*(?:B|BANKER|莊|庄)\s*([0-9])",
        r"(?:B|BANKER|莊|庄)\s*([0-9])\D*(?:P|PLAYER|閒|闲)\s*([0-9])",
        r"^\s*([0-9])\s*[,/\- ]\s*([0-9])\s*$",
    ]
    match = re.search(patterns[0], text)
    if match:
        return {"player": int(match.group(1)), "banker": int(match.group(2)), "path": None, "suffix": ""}
    match = re.search(patterns[1], text)
    if match:
        return {"player": int(match.group(2)), "banker": int(match.group(1)), "path": None, "suffix": ""}
    match = re.search(patterns[2], text)
    if match:
        return {"player": int(match.group(1)), "banker": int(match.group(2)), "path": None, "suffix": ""}
    return None


def _clean_observations(values: Union[str, Iterable[Any], None]) -> List[Dict[str, Any]]:
    if values is None:
        return []
    chunks = [x for x in re.split(r"[;|\n]+", values) if x.strip()] if isinstance(values, str) else list(values)
    out: List[Dict[str, Any]] = []
    for item in chunks:
        parsed = parse_point_observation(item)
        if parsed is not None:
            out.append(parsed)
    return out


def _outcome(observation: Mapping[str, Any]) -> str:
    player, banker = int(observation["player"]), int(observation["banker"])
    return "B" if banker > player else "P" if player > banker else "T"


def _new_seed(explicit: Optional[int] = None) -> int:
    if explicit is not None:
        return int(explicit) & 0xFFFFFFFF
    if FIXED_RUN_SEED > 0:
        return FIXED_RUN_SEED & 0xFFFFFFFF
    return secrets.randbits(32) if RANDOMIZE_EACH_CALL else 20260714


def _probability_dict(values: Any) -> Dict[str, float]:
    return {"B": float(values[0]), "P": float(values[1]), "T": float(values[2])}


def _draw_path_dict(values: Any) -> Dict[str, float]:
    return {name: float(values[i]) for i, name in enumerate(PATH_NAMES)}


def predict(
    history: Union[str, Iterable[Any]],
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
    user_id: str = "",
    run_seed: Optional[int] = None,
) -> Dict[str, Any]:
    observations = _clean_observations(history)
    latest = observations[-1] if observations else None
    if latest is None:
        return {"ok": False, "error": "missing_point_observation", "message": "請輸入兩位數點數，例如65代表閒6莊5。"}

    seed = _new_seed(run_seed)
    result = _ENGINE.analyze(
        latest["player"],
        latest["banker"],
        seed,
        latest.get("path"),
    )
    probabilities = _probability_dict(result["fused"])
    pf_probabilities = _probability_dict(result["pf"])
    control_probabilities = _probability_dict(result["control"])
    db_probabilities = _probability_dict(result["database"])
    recommend = str(result["recommend"]).upper()
    if recommend == "B":
        recommend_text = "莊"
    elif recommend == "P":
        recommend_text = "閒"
    else:
        recommend = "NONE"
        recommend_text = "觀望"

    confidence = max(probabilities["B"], probabilities["P"]) / max(
        1e-12,
        probabilities["B"] + probabilities["P"],
    )
    point_text = f"{latest['player']}{latest['banker']}{latest.get('suffix', '')}"

    response: Dict[str, Any] = {
        "ok": True,
        "engine": "V5_2_THREE_TIER_POINT_PF_LINE",
        "model_version": "V5.2-THREE-TIER-FAST-20260716",
        "user_id": user_id,
        "venue": venue,
        "room": room,
        "shoe_id": shoe_id,
        "run_seed": seed,
        "independent_mode": True,
        "history_continuation": False,
        "persistent_particle_state": False,
        "used_observation_count": 1,
        "ignored_prior_observations": max(0, len(observations) - 1),
        "conditioning_point": point_text,
        "conditioning_observation": {"player": latest["player"], "banker": latest["banker"]},
        "conditioning_outcome": _outcome(latest),
        "known_draw_path": PATH_NAMES[latest["path"]] if latest.get("path") is not None else None,
        "banker_rate": round(probabilities["B"] * 100.0, 1),
        "player_rate": round(probabilities["P"] * 100.0, 1),
        "tie_rate": round(probabilities["T"] * 100.0, 1),
        "probabilities": probabilities,
        "particle_probabilities": pf_probabilities,
        "control_probabilities": control_probabilities,
        "shoe_database_probabilities": db_probabilities,
        "recommend": recommend,
        "recommend_text": recommend_text,
        "is_observe": recommend == "NONE",
        "confidence": round(confidence, 6),
        "confidence_pct": round(confidence * 100.0, 1),
        "decision_edge": round(float(result["edge"]), 8),
        "signal_level": str(result["signal_level"]),
        "decision_source": str(result["decision_source"]),
        "validated_signal": bool(result["validated_signal"]),
        "quality_pass": bool(result.get("quality_pass", False)),
        "general_quality_pass": bool(result.get("general_quality_pass", False)),
        "decision_tier": str(result.get("decision_tier", "OBSERVE")),
        "lower_bound": round(float(result["lower_bound"]), 8),
        "centered_edge": round(float(result["center"]), 8),
        "center_se": round(float(result["center_se"]), 8),
        "replica_count": int(result["replicas"]),
        "replica_directions": list(result["replica_directions"]),
        "replica_agreement": round(float(result["replica_agreement"]), 6),
        "split_agreement": round(float(result["split_agreement"]), 6),
        "effective_replicas": round(float(result["effective_replicas"]), 4),
        "stability": str(result["stability"]),
        "weakness_reason": str(result["weakness_reason"]),
        "current_point_draw_paths": _draw_path_dict(result["draw_paths"]),
        "next_hand_draw_paths": _draw_path_dict(result["next_draw_paths"]),
        "top_points": list(result["top_points"]),
        "draw_path_diagnostics": {
            "coverage": round(float(result["average_path_coverage"]), 6),
            "legacy_coverage": round(float(result["average_legacy_path_coverage"]), 6),
            "ess_quality": round(float(result["average_path_ess_quality"]), 6),
            "candidates": [round(float(x), 2) for x in result["average_path_candidates"]],
            "ess": [round(float(x), 2) for x in result["average_path_ess"]],
            "allocated": [round(float(x), 2) for x in result["average_path_allocated"]],
            "current_path_agreement": round(float(result["average_current_path_agreement"]), 6),
            "next_draw_agreement": round(float(result["average_draw_agreement"]), 6),
        },
        "point_particle_filter": {
            "particles_per_replica": int(result["settings"]["particles"]),
            "replicas": int(result["replicas"]),
            "average_matches": round(float(result["average_matches"]), 3),
            "average_effective_sample_size": round(float(result["average_ess"]), 3),
            "average_acceptance": round(float(result["average_acceptance"]), 8),
            "average_attempts": round(float(result["average_attempts"]), 3),
            "average_diversity": round(float(result["average_diversity"]), 6),
            "total_forecast_simulations": int(result["total_forecast_simulations"]),
            "total_condition_attempts": int(result["total_condition_attempts"]),
            "state_digest": str(result["state_digest"]),
            "conditional_generator": str(result["conditional_generator"]),
            "variance_reduction": str(result["variance_reduction"]),
        },
        "shoe_state_database": {
            **get_shoe_state_database().database_info(),
            "probabilities": db_probabilities,
            "average_samples": round(float(result["database_samples"]), 3),
            "effective_weight": round(float(result["average_database_weight"]), 8),
            "holdout": dict(DB_HOLDOUT),
        },
        "reason": (
            "V5.2單局獨立粒子模型；只使用本次最新點數；四種合法補牌路徑分層；"
            f"每副本實際模擬總數約"
            f"{int(result['total_forecast_simulations']) // max(1, int(result['replicas']))}；"
            "不使用歷史、牌路或連勝連敗；採正式、一般、觀望三層品質判定。"
            f"決策來源={result['decision_source']}；{result['reason']}。"
        ),
        "debug": None,
    }
    if DEBUG_V5_RESULT:
        response["debug"] = {
            "settings": result["settings"],
            "votes": result["votes"],
            "outlier_count": result["outlier_count"],
            "robust_mad": result["robust_mad"],
        }
    return response


def fit_history(history: Union[str, Iterable[Any]], venue: str = "", room: str = "", shoe_id: str = "", user_id: str = "", force: bool = True) -> Dict[str, Any]:
    result = predict(history, venue, room, shoe_id, user_id)
    return {"ok": bool(result.get("ok")), "model": result.get("engine"), "independent_samples": int(result.get("used_observation_count", 0))}


def reset_uid_model(user_id: str, venue: str = "", room: str = "", shoe_id: str = "") -> Dict[str, Any]:
    return {"ok": True, "removed": 0, "independent_mode": True, "message": "V5.1.6為無狀態模式，沒有UID粒子可清除。"}


def clear_model_cache(user_id: Optional[str] = None) -> Dict[str, Any]:
    removed = clear_runtime_caches()
    return {
        "ok": True,
        "removed": removed,
        "independent_mode": True,
        "message": "已清除粒子先驗快取；不影響任何UID歷史資料。",
    }
