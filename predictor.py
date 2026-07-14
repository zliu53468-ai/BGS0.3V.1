"""Official LINE-compatible V5 independent point predictor.

Drop-in public API:
    predict(history, venue='', room='', shoe_id='', user_id='')

Even if app.py passes an entire point history, V5 deliberately uses only the
newest valid observation. No prior point, previous forecast, or UID particle
state is retained.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
import os
import re
import secrets

from particle_filter_points import (
    BASELINE,
    DB_HOLDOUT,
    V5IndependentBaccaratEngine,
)
from shoe_state_db import get_shoe_state_database


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default)).strip()))
    except Exception:
        return default


RANDOMIZE_EACH_CALL = _env_bool("PF_RANDOMIZE_EACH_CALL", True)
FIXED_RUN_SEED = _env_int("PF_FIXED_RUN_SEED", 0, 0)
DEBUG_V5_RESULT = _env_bool("PF_DEBUG_V5_RESULT", False)


def parse_point_observation(value: Any) -> Optional[Dict[str, int]]:
    if isinstance(value, Mapping):
        player = value.get("player", value.get("P", value.get("閒")))
        banker = value.get("banker", value.get("B", value.get("莊")))
        try:
            return {"player": int(player) % 10, "banker": int(banker) % 10}
        except Exception:
            return None
    text = str(value or "").strip().upper()
    patterns = [
        r"(?:P|PLAYER|閒|闲)\s*([0-9])\D*(?:B|BANKER|莊|庄)\s*([0-9])",
        r"(?:B|BANKER|莊|庄)\s*([0-9])\D*(?:P|PLAYER|閒|闲)\s*([0-9])",
        r"^\s*([0-9])\s*[,/\- ]\s*([0-9])\s*$",
        r"^\s*([0-9])([0-9])\s*$",
    ]
    match = re.search(patterns[0], text)
    if match:
        return {"player": int(match.group(1)), "banker": int(match.group(2))}
    match = re.search(patterns[1], text)
    if match:
        return {"player": int(match.group(2)), "banker": int(match.group(1))}
    for pattern in patterns[2:]:
        match = re.search(pattern, text)
        if match:
            return {"player": int(match.group(1)), "banker": int(match.group(2))}
    return None


def _clean_observations(values: Union[str, Iterable[Any], None]) -> List[Dict[str, int]]:
    if values is None:
        return []
    if isinstance(values, str):
        chunks = [item for item in re.split(r"[;|\n]+", values) if item.strip()]
    else:
        chunks = list(values)
    observations: List[Dict[str, int]] = []
    for item in chunks:
        parsed = parse_point_observation(item)
        if parsed is not None:
            observations.append(parsed)
    return observations


def _outcome(observation: Mapping[str, int]) -> str:
    player = int(observation["player"])
    banker = int(observation["banker"])
    return "B" if banker > player else "P" if player > banker else "T"


def _new_seed(explicit: Optional[int] = None) -> int:
    if explicit is not None:
        return int(explicit) & 0xFFFFFFFF
    if FIXED_RUN_SEED > 0:
        return FIXED_RUN_SEED & 0xFFFFFFFF
    if RANDOMIZE_EACH_CALL:
        return secrets.randbits(32)
    return 20260714


def _probability_dict(values: Any) -> Dict[str, float]:
    return {"B": float(values[0]), "P": float(values[1]), "T": float(values[2])}


def _draw_path_dict(values: Any) -> Dict[str, float]:
    return {
        "none": float(values[0]),
        "player_only": float(values[1]),
        "banker_only": float(values[2]),
        "both": float(values[3]),
    }


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
        return {
            "ok": False,
            "error": "missing_point_observation",
            "message": "請輸入兩位數點數，例如65代表閒6莊5。",
        }

    seed = _new_seed(run_seed)
    engine = V5IndependentBaccaratEngine()
    result = engine.analyze(latest["player"], latest["banker"], seed)
    probabilities = _probability_dict(result["fused"])
    pf_probabilities = _probability_dict(result["pf"])
    control_probabilities = _probability_dict(result["control"])
    db_probabilities = _probability_dict(result["database"])
    draw_paths = _draw_path_dict(result["draw_paths"])
    recommend = str(result["recommend"])
    recommend_text = "莊" if recommend == "B" else "閒"
    history_length = len(observations)
    confidence = max(probabilities["B"], probabilities["P"]) / max(
        1e-12, probabilities["B"] + probabilities["P"]
    )

    response: Dict[str, Any] = {
        "ok": True,
        "engine": "V5_INDEPENDENT_POINT_PF_LINE",
        "model_version": "V5-LINE-20260714",
        "user_id": user_id,
        "venue": venue,
        "room": room,
        "shoe_id": shoe_id,
        "run_seed": seed,
        "round_no": history_length + 1,
        "history_len": history_length,
        "independent_mode": True,
        "history_continuation": False,
        "accumulated_state": False,
        "persistent_particle_state": False,
        "used_observation_count": 1,
        "ignored_prior_observations": max(0, history_length - 1),
        "conditioning_observation": dict(latest),
        "conditioning_outcome": _outcome(latest),
        "banker_rate": round(probabilities["B"] * 100.0, 1),
        "player_rate": round(probabilities["P"] * 100.0, 1),
        "tie_rate": round(probabilities["T"] * 100.0, 1),
        "probabilities": probabilities,
        "particle_probabilities": pf_probabilities,
        "direct_simulation_probabilities": pf_probabilities,
        "control_probabilities": control_probabilities,
        "shoe_database_probabilities": db_probabilities,
        "recommend": recommend,
        "recommend_text": recommend_text,
        "fallback_recommend": recommend,
        "is_observe": False,
        "observe_reason": "",
        "observe_gate_removed": True,
        "confidence": round(confidence, 6),
        "confidence_pct": round(confidence * 100.0, 1),
        "decision_edge": round(float(result["edge"]), 8),
        "signal_edge": round(float(result["edge"]), 8),
        "signal_level": str(result["signal_level"]),
        "decision_source": str(result["decision_source"]),
        "validated_signal": bool(result["validated_signal"]),
        "model_side": result.get("model_side"),
        "quality_pass": bool(result.get("quality_pass", False)),
        "lower_bound": round(float(result["lower_bound"]), 8),
        "centered_edge": round(float(result["center"]), 8),
        "raw_center": round(float(result["raw_center"]), 8),
        "median_center": round(float(result["median_center"]), 8),
        "center_std": round(float(result["center_std"]), 8),
        "center_se": round(float(result["center_se"]), 8),
        "fallback_score": round(float(result["fallback_score"]), 8),
        "banker_ev": round(float(result["banker_ev"]), 8),
        "player_ev": round(float(result["player_ev"]), 8),
        "replica_count": int(result["replicas"]),
        "replica_directions": list(result["replica_directions"]),
        "replica_agreement": round(float(result["replica_agreement"]), 6),
        "replica_votes": dict(result["votes"]),
        "stability": str(result["stability"]),
        "weakness_reason": str(result["weakness_reason"]),
        "current_point_draw_paths": draw_paths,
        "point_particle_filter": {
            "particles_per_replica": int(result["settings"]["particles"]),
            "replicas": int(result["replicas"]),
            "average_matches": round(float(result["average_matches"]), 3),
            "average_effective_sample_size": round(float(result["average_ess"]), 3),
            "average_acceptance": round(float(result["average_acceptance"]), 8),
            "average_attempts": round(float(result["average_attempts"]), 3),
            "average_diversity": round(float(result["average_diversity"]), 6),
            "mean_unknown_shoe_depth": round(float(result["mean_depth"]), 3),
            "min_unknown_shoe_depth": int(result["min_depth"]),
            "max_unknown_shoe_depth": int(result["max_depth"]),
            "cards_remaining": round(float(result["cards_remaining"]), 3),
            "shoe_depth_ratio": round(float(result["shoe_depth"]), 8),
            "state_digest": str(result["state_digest"]),
            "total_forecast_simulations": int(result["total_forecast_simulations"]),
            "total_condition_attempts": int(result["total_condition_attempts"]),
            "forecast_mutated_state": False,
            "ancestry_pairing": bool(result["all_ancestry_paired"]),
            "all_replicas_updated": bool(result["all_replicas_updated"]),
            "conditional_generator": str(result["conditional_generator"]),
            "variance_reduction": str(result["variance_reduction"]),
            "depth_profile": str(result["depth_profile"]),
            "deduplicated": False,
            "fallback_to_unconditioned": bool(result["fallback_to_unconditioned"]),
        },
        "shoe_state_database": {
            **get_shoe_state_database().database_info(),
            "probabilities": db_probabilities,
            "average_samples": round(float(result["database_samples"]), 3),
            "effective_weight": round(float(result["database_effective_weight"]), 8),
            "validation_mode": str(result["settings"]["database_validation_mode"]),
            "holdout": dict(DB_HOLDOUT),
        },
        "pattern_features": {
            "enabled": False,
            "streak": False,
            "road": False,
            "markov": False,
            "point_sequence": False,
            "outcome_momentum": False,
            "previous_prediction": False,
        },
        # Compatibility fields used by older app.py/front ends.
        "ml_trained": True,
        "ml_samples": 1,
        "tf_available": False,
        "lstm_status": "disabled_v5_independent_point_pf",
        "global_lstm_status": "disabled_v5_independent_point_pf",
        "ai_used": False,
        "ai_status": "disabled_v5",
        "deepseek_probabilities": None,
        "deepseek": {
            "enabled": False,
            "status": "disabled_v5",
            "effective_weight": 0.0,
        },
        "configured_weights": {
            "particle_filter": 1.0,
            "shoe_database": float(result["settings"]["database_weight"]),
            "deepseek": 0.0,
        },
        "effective_weights": {
            "particle_filter": 1.0,
            "shoe_database": round(float(result["database_effective_weight"]), 8),
            "deepseek": 0.0,
        },
        "reason": (
            "V5單局獨立粒子濾波；只使用本次最新點數；固定10/30/40/20未知牌靴深度先驗；"
            "精確合法第三張牌完成＋重要性權重；每顆條件粒子與真正祖先牌靴一對一配對；"
            "共同亂數＋對偶抽樣；不讀取前手點數、不累積UID狀態、不使用牌路規律；"
            f"決策來源={result['decision_source']}；{result['reason']}。"
        ),
        "debug": result["replica_rows"] if DEBUG_V5_RESULT else None,
    }
    return response


def fit_history(
    history: Union[str, Iterable[Any]],
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
    user_id: str = "",
    force: bool = True,
) -> Dict[str, Any]:
    result = predict(history, venue, room, shoe_id, user_id)
    return {
        "ok": bool(result.get("ok")),
        "history_len": int(result.get("history_len", 0)),
        "independent_samples": int(result.get("used_observation_count", 0)),
        "model": result.get("engine"),
        "decision_source": result.get("decision_source"),
    }


def reset_uid_model(
    user_id: str,
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
) -> Dict[str, Any]:
    return {
        "ok": True,
        "removed": 0,
        "training_key": "stateless",
        "independent_mode": True,
        "message": "V5為無狀態模式，沒有UID粒子可清除。",
    }


def clear_model_cache(user_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "ok": True,
        "removed": 0,
        "independent_mode": True,
        "message": "V5沒有粒子或歷史快取。",
    }


def get_model_cache_info() -> Dict[str, Any]:
    return {
        "size": 0,
        "keys": [],
        "persistent_particle_filters": 0,
        "engine": "V5_INDEPENDENT_POINT_PF_LINE",
        "independent_mode": True,
        "database": get_shoe_state_database().database_info(),
        "database_holdout": dict(DB_HOLDOUT),
    }
