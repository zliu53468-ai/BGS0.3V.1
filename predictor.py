"""LINE-compatible V5.5 1000-particle draw-path predictor."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
import os
import re
import secrets

import numpy as np

from particle_filter_points import (
    DB_HOLDOUT,
    V5IndependentBaccaratEngine,
    clear_runtime_caches,
    mix_seed,
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


def _env_float(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(os.getenv(name, str(default)).strip())
    except Exception:
        value = default
    return max(low, min(high, value))


RANDOMIZE_EACH_CALL = _env_bool("PF_RANDOMIZE_EACH_CALL", True)
FIXED_RUN_SEED = _env_int("PF_FIXED_RUN_SEED", 0, 0)
DEBUG_V5_RESULT = _env_bool("PF_DEBUG_V5_RESULT", False)

# V5.3 uses one full primary run and two lightweight independent master-seed
# validators. The validators do not create a direction; they can only confirm or
# reject the primary direction. This prevents one fixed seed from permanently
# mapping a point to an unstable side while keeping LINE latency practical.
CONSENSUS_RUNS = min(5, _env_int("PF_CONSENSUS_RUNS", 3, 1))
CONSENSUS_MIN_AGREEMENT = _env_float("PF_CONSENSUS_MIN_AGREEMENT", 1.0, 0.50, 1.0)
CONSENSUS_MIN_DECISIVE_RUNS = min(
    CONSENSUS_RUNS,
    _env_int("PF_CONSENSUS_MIN_DECISIVE_RUNS", CONSENSUS_RUNS, 1),
)
CONSENSUS_MIN_EDGE = _env_float("PF_CONSENSUS_MIN_EDGE", 0.0005, 0.0, 0.05)
CONSENSUS_MIN_VALIDATOR_AGREEMENT = _env_float(
    "PF_CONSENSUS_MIN_VALIDATOR_AGREEMENT", 0.60, 0.50, 1.0
)
CONSENSUS_MIN_VALIDATOR_SPLIT = _env_float(
    "PF_CONSENSUS_MIN_VALIDATOR_SPLIT", 0.0, 0.0, 1.0
)
CONSENSUS_MIN_VALIDATOR_ESS = _env_float(
    "PF_CONSENSUS_MIN_VALIDATOR_ESS", 40.0, 1.0, 4000.0
)
CONSENSUS_MIN_VALIDATOR_DIVERSITY = _env_float(
    "PF_CONSENSUS_MIN_VALIDATOR_DIVERSITY", 0.20, 0.0, 1.0
)
CONSENSUS_MIN_VALIDATOR_PATH = _env_float(
    "PF_CONSENSUS_MIN_VALIDATOR_PATH", 0.50, 0.0, 1.0
)
VALIDATOR_PARTICLES = min(500, _env_int("PF_CONSENSUS_VALIDATOR_PARTICLES", 128, 64))
VALIDATOR_REPLICAS = min(5, _env_int("PF_CONSENSUS_VALIDATOR_REPLICAS", 3, 3))
VALIDATOR_SAMPLE_CAP = _env_int("PF_CONSENSUS_VALIDATOR_SAMPLE_CAP", 300, 200)
VALIDATOR_MAX_PROPOSALS = _env_int("PF_CONSENSUS_VALIDATOR_MAX_PROPOSALS", 4500, 500)

# Engines are stateless and safe to reuse. Immutable prior caching is handled by
# particle_filter_points.py.
_ENGINE = V5IndependentBaccaratEngine()
_VALIDATOR_ENGINE = V5IndependentBaccaratEngine(
    {
        "particles": VALIDATOR_PARTICLES,
        "replicas": VALIDATOR_REPLICAS,
        "target_matches": 80,
        "target_ess": 50.0,
        "max_update_proposals": VALIDATOR_MAX_PROPOSALS,
        "path_target_matches": 8,
        "predict_simulations_per_replica": VALIDATOR_SAMPLE_CAP // 2,
        "point_joint_simulations_per_replica": VALIDATOR_SAMPLE_CAP // 2,
        "forecast_sample_cap": VALIDATOR_SAMPLE_CAP,
        "fast_particle_cap": VALIDATOR_PARTICLES,
        "fast_target_matches_cap": 80,
        "fast_target_ess_cap": 50.0,
        "fast_max_update_proposals": VALIDATOR_MAX_PROPOSALS,
        "fast_path_target_matches_cap": 8,
    }
)


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


def _validator_quality(result: Mapping[str, Any]) -> bool:
    return (
        bool(result.get("all_ancestry_paired", False))
        and float(result.get("replica_agreement", 0.0))
        >= CONSENSUS_MIN_VALIDATOR_AGREEMENT
        and float(result.get("split_agreement", 0.0))
        >= CONSENSUS_MIN_VALIDATOR_SPLIT
        and float(result.get("average_ess", 0.0))
        >= CONSENSUS_MIN_VALIDATOR_ESS
        and float(result.get("average_diversity", 0.0))
        >= CONSENSUS_MIN_VALIDATOR_DIVERSITY
        and float(result.get("average_path_coverage", 0.0))
        >= CONSENSUS_MIN_VALIDATOR_PATH
        and abs(float(result.get("center", 0.0))) >= CONSENSUS_MIN_EDGE
    )


def _average_arrays(results: List[Mapping[str, Any]], key: str) -> Any:
    return np.mean(np.stack([np.asarray(item[key], dtype=float) for item in results]), axis=0)


def _combine_master_seed_results(
    primary: Dict[str, Any],
    validators: List[Dict[str, Any]],
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = [primary, *validators]
    combined = dict(primary)

    for key in ("pf", "control", "database", "fused", "draw_paths", "next_draw_paths", "point_matrix"):
        combined[key] = _average_arrays(results, key)

    point_matrix = np.asarray(combined["point_matrix"], dtype=float)
    top_idx = np.argsort(point_matrix)[::-1][:10]
    combined["top_points"] = [
        {
            "point": f"{int(index // 10)}{int(index % 10)}",
            "probability": float(point_matrix[index]),
            "outcome": (
                "B" if int(index % 10) > int(index // 10)
                else "P" if int(index // 10) > int(index % 10)
                else "T"
            ),
        }
        for index in top_idx
    ]

    primary_side = str(primary.get("recommend", "NONE")).upper()
    model_sides = [str(item.get("model_side", "NONE")).upper() for item in results]
    validator_quality = [_validator_quality(item) for item in validators]
    decisive = [side for side in model_sides if side in {"B", "P"}]
    counts = {"B": decisive.count("B"), "P": decisive.count("P")}
    dominant = "B" if counts["B"] >= counts["P"] else "P"
    agreement = max(counts.values()) / max(1, len(results))
    centers = [float(item.get("center", 0.0)) for item in results]
    signed_center = float(np.mean(centers))
    minimum_edge = min(abs(value) for value in centers)
    direction_consistent = (signed_center >= 0 and dominant == "B") or (
        signed_center < 0 and dominant == "P"
    )

    consensus_pass = (
        primary_side in {"B", "P"}
        and primary_side == dominant
        and len(decisive) >= CONSENSUS_MIN_DECISIVE_RUNS
        and agreement >= CONSENSUS_MIN_AGREEMENT
        and all(validator_quality)
        and minimum_edge >= CONSENSUS_MIN_EDGE
        and direction_consistent
    )

    combined["master_seed_count"] = len(results)
    combined["master_seed_directions"] = model_sides
    combined["master_seed_agreement"] = agreement
    combined["master_seed_validator_quality"] = validator_quality
    combined["master_seed_consensus_pass"] = consensus_pass
    combined["master_seed_minimum_edge"] = minimum_edge
    combined["master_seed_centers"] = centers
    combined["center"] = signed_center
    combined["raw_center"] = signed_center
    combined["edge"] = min(float(primary.get("edge", 0.0)), minimum_edge) if consensus_pass else 0.0
    combined["lower_bound"] = min(float(item.get("lower_bound", 0.0)) for item in results)
    combined["replicas"] = sum(int(item.get("replicas", 0)) for item in results)
    combined["replica_directions"] = [
        direction
        for item in results
        for direction in list(item.get("replica_directions", []))
    ]
    combined["replica_agreement"] = float(
        np.mean([float(item.get("replica_agreement", 0.0)) for item in results])
    )
    combined["split_agreement"] = float(
        np.mean([float(item.get("split_agreement", 0.0)) for item in results])
    )
    combined["effective_replicas"] = float(
        np.mean([float(item.get("effective_replicas", 0.0)) for item in results])
    )
    for key in (
        "average_matches", "average_ess", "average_acceptance", "average_attempts",
        "average_diversity", "average_path_coverage", "average_legacy_path_coverage",
        "average_path_ess_quality", "average_current_path_agreement",
        "average_draw_agreement", "average_point_concentration", "database_samples",
        "average_database_weight", "mean_depth", "cards_remaining", "shoe_depth",
    ):
        combined[key] = float(np.mean([float(item.get(key, 0.0)) for item in results]))
    combined["min_depth"] = min(int(item.get("min_depth", 0)) for item in results)
    combined["max_depth"] = max(int(item.get("max_depth", 0)) for item in results)
    combined["total_forecast_simulations"] = sum(
        int(item.get("total_forecast_simulations", 0)) for item in results
    )
    combined["total_condition_attempts"] = sum(
        int(item.get("total_condition_attempts", 0)) for item in results
    )
    combined["all_ancestry_paired"] = all(
        bool(item.get("all_ancestry_paired", False)) for item in results
    )
    combined["all_replicas_updated"] = all(
        bool(item.get("all_replicas_updated", False)) for item in results
    )
    combined["fallback_to_unconditioned"] = any(
        bool(item.get("fallback_to_unconditioned", False)) for item in results
    )

    if not consensus_pass:
        combined.update(
            {
                "recommend": "NONE",
                "decision_tier": "OBSERVE",
                "decision_source": "OBSERVE",
                "signal_level": "OBSERVE",
                "validated_signal": False,
                "quality_pass": False,
                "general_quality_pass": False,
                "is_observe": True,
                "stability": "UNSTABLE",
                "reason": "主模型未取得多主種子一致驗證，本局觀望且不建立方向戰績",
            }
        )
        combined["weakness_reason"] = (
            str(primary.get("weakness_reason", ""))
            + "；多主種子方向或驗證品質未形成完整共識"
        ).strip("；")
    else:
        combined["recommend"] = dominant
        combined["is_observe"] = False
        combined["weakness_reason"] = (
            str(primary.get("weakness_reason", ""))
            + "；已通過三組獨立主種子方向共識"
        ).strip("；")
        combined["reason"] = (
            str(primary.get("reason", ""))
            + "；並通過三組獨立主種子方向與最低品質驗證"
        ).strip("；")

    return combined


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
    primary = _ENGINE.analyze(
        latest["player"],
        latest["banker"],
        seed,
        latest.get("path"),
    )
    # V5.5 uses one complete 1000-particle primary run. The 2,000-sample paired
    # forecast pool lets all 1,000 particle indices participate once per replica
    # while avoiding the latency of extra consensus engines.
    result = primary
    result["master_seed_count"] = 1
    result["master_seed_directions"] = [str(result.get("recommend", "B"))]
    result["master_seed_agreement"] = 1.0
    result["master_seed_consensus_pass"] = True
    result["master_seed_minimum_edge"] = abs(float(result.get("center", 0.0)))
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
        recommend = "B" if float(result.get("center", 0.0)) >= 0 else "P"
        recommend_text = "莊" if recommend == "B" else "閒"

    confidence = max(probabilities["B"], probabilities["P"]) / max(
        1e-12,
        probabilities["B"] + probabilities["P"],
    )
    point_text = f"{latest['player']}{latest['banker']}{latest.get('suffix', '')}"

    response: Dict[str, Any] = {
        "ok": True,
        "engine": "V5_5_DRAW_PATH_FUSION_1000P_LINE",
        "model_version": "V5.5-DRAW-PATH-FUSION-1000P-20260717",
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
        "is_observe": False,
        "confidence": round(confidence, 6),
        "confidence_pct": round(confidence * 100.0, 1),
        "decision_edge": round(float(result["edge"]), 8),
        "signal_level": str(result["signal_level"]),
        "decision_source": str(result["decision_source"]),
        "validated_signal": bool(result["validated_signal"]),
        "quality_pass": bool(result.get("quality_pass", False)),
        "general_quality_pass": bool(result.get("general_quality_pass", False)),
        "decision_tier": str(result.get("decision_tier", "OBSERVE")),
        "master_seed_consensus_pass": bool(result.get("master_seed_consensus_pass", False)),
        "master_seed_count": int(result.get("master_seed_count", 1)),
        "master_seed_directions": list(result.get("master_seed_directions", [])),
        "master_seed_agreement": round(float(result.get("master_seed_agreement", 0.0)), 6),
        "master_seed_minimum_edge": round(float(result.get("master_seed_minimum_edge", 0.0)), 8),
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
            "V5.4單局獨立500粒子模型；只使用本次最新點數；當局與下一局四種合法補牌路徑直接融合；"
            f"每副本實際模擬總數約"
            f"{int(result['total_forecast_simulations']) // max(1, int(result['replicas']))}；"
            "不使用歷史或牌路；品質層級只作標示，不再以觀望阻斷方向。"
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
            "master_seed_directions": result.get("master_seed_directions", []),
            "master_seed_validator_quality": result.get("master_seed_validator_quality", []),
            "master_seed_centers": result.get("master_seed_centers", []),
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
