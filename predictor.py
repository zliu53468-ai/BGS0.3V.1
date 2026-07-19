"""LINE-compatible V5.4.1 aggressive quality-gated draw-path predictor."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
import os
import re
import secrets

from particle_filter_points import (
    DB_HOLDOUT,
    V5IndependentBaccaratEngine,
)
from shoe_state_db import get_shoe_state_database

PATH_SUFFIX = {"N": 0, "P": 1, "B": 2, "D": 3}
DRAW_CODE_TO_SUFFIX = {"1": "P", "2": "B", "3": "D", "4": "N"}
PATH_NAMES = ("none", "player_only", "banker_only", "both")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return (
        default
        if raw is None
        else raw.strip().lower() in {"1", "true", "yes", "on"}
    )


def _env_int(
    name: str,
    default: int,
    minimum: int = 0,
) -> int:
    try:
        return max(
            minimum,
            int(os.getenv(name, str(default)).strip()),
        )
    except Exception:
        return default


def _env_float(
    name: str,
    default: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    try:
        value = float(os.getenv(name, str(default)).strip())
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


RANDOMIZE_EACH_CALL = _env_bool("PF_RANDOMIZE_EACH_CALL", True)
FIXED_RUN_SEED = _env_int("PF_FIXED_RUN_SEED", 0, 0)
DEBUG_V5_RESULT = _env_bool("PF_DEBUG_V5_RESULT", False)
OBSERVE_ON_UNVALIDATED = _env_bool(
    "PF_OBSERVE_ON_UNVALIDATED",
    False,
)

BET_SIZING_ENABLED = _env_bool("BET_SIZING_ENABLED", True)
BET_MIN_FRACTION = _env_float("BET_MIN_FRACTION", 0.10, 0.0, 1.0)
BET_MAX_FRACTION = _env_float("BET_MAX_FRACTION", 0.40, 0.0, 1.0)
BET_REQUIRE_VALIDATED = _env_bool("BET_REQUIRE_VALIDATED", False)
BET_UNVALIDATED_MAX_FRACTION = _env_float(
    "BET_UNVALIDATED_MAX_FRACTION",
    0.40,
    0.0,
    1.0,
)


def _normalize_known_cards(value: Any) -> Dict[str, List[int]]:
    if not isinstance(value, Mapping):
        return {}
    out: Dict[str, List[int]] = {}
    aliases = {
        "player": ("player", "P", "閒", "闲"),
        "banker": ("banker", "B", "莊", "庄"),
    }
    for side, names in aliases.items():
        raw = None
        for name in names:
            if name in value:
                raw = value.get(name)
                break
        if raw is None:
            continue
        try:
            cards = [int(card) % 10 for card in list(raw)]
        except Exception:
            continue
        if len(cards) in {2, 3}:
            out[side] = cards
    return out


def _normalize_counts(value: Any) -> Optional[List[int]]:
    try:
        result = [int(item) for item in list(value)]
    except Exception:
        return None
    if len(result) != 10 or any(item < 0 for item in result):
        return None
    return result


def parse_point_observation(
    value: Any,
) -> Optional[Dict[str, Any]]:
    """Parse current-hand points and draw path, e.g. 382 or 38B."""
    if isinstance(value, Mapping):
        player = value.get(
            "player",
            value.get("P", value.get("閒")),
        )
        banker = value.get(
            "banker",
            value.get("B", value.get("莊")),
        )
        suffix = str(
            value.get("path_suffix")
            or value.get("suffix")
            or ""
        ).strip().upper()
        suffix = DRAW_CODE_TO_SUFFIX.get(suffix, suffix)
        try:
            explicit_path = value.get("path")
            path = (
                PATH_SUFFIX.get(suffix)
                if suffix in PATH_SUFFIX
                else int(explicit_path)
                if explicit_path is not None
                else None
            )
            if path not in {0, 1, 2, 3}:
                path = None
            hand_number = int(value.get("hand_number") or 0)
            return {
                "player": int(player) % 10,
                "banker": int(banker) % 10,
                "path": path,
                "suffix": (
                    suffix
                    if suffix in PATH_SUFFIX
                    else next(
                        (
                            key
                            for key, index in PATH_SUFFIX.items()
                            if index == path
                        ),
                        "",
                    )
                ),
                "hand_number": max(0, min(120, hand_number)),
                "known_cards": _normalize_known_cards(
                    value.get("known_cards")
                    or value.get("cards")
                    or {}
                ),
                "remaining_counts": _normalize_counts(
                    value.get("remaining_counts")
                ),
                "known_card_counts": _normalize_counts(
                    value.get("known_card_counts")
                ),
                "state_complete": bool(
                    value.get("state_complete", False)
                ),
                "state_source": str(
                    value.get("state_source") or ""
                ),
                "tracked_card_hands": int(
                    value.get("tracked_card_hands") or 0
                ),
            }
        except Exception:
            return None

    text = str(value or "").strip().upper()
    compact = re.fullmatch(
        r"([0-9])([0-9])([NPBD]|[1-4])?(?:@([0-9]{1,3}))?",
        text,
    )
    if compact:
        suffix = compact.group(3) or ""
        suffix = DRAW_CODE_TO_SUFFIX.get(suffix, suffix)
        return {
            "player": int(compact.group(1)),
            "banker": int(compact.group(2)),
            "path": PATH_SUFFIX.get(suffix),
            "suffix": suffix,
            "hand_number": max(
                0,
                min(120, int(compact.group(4) or 0)),
            ),
            "known_cards": {},
            "remaining_counts": None,
            "known_card_counts": None,
            "state_complete": False,
            "state_source": "",
            "tracked_card_hands": 0,
        }

    patterns = [
        (
            r"(?:P|PLAYER|閒|闲)\s*([0-9])"
            r"\D*(?:B|BANKER|莊|庄)\s*([0-9])"
        ),
        (
            r"(?:B|BANKER|莊|庄)\s*([0-9])"
            r"\D*(?:P|PLAYER|閒|闲)\s*([0-9])"
        ),
        r"^\s*([0-9])\s*[,/\- ]\s*([0-9])\s*$",
    ]
    match = re.search(patterns[0], text)
    if match:
        return {
            "player": int(match.group(1)),
            "banker": int(match.group(2)),
            "path": None,
            "suffix": "",
            "hand_number": 0,
            "known_cards": {},
        }
    match = re.search(patterns[1], text)
    if match:
        return {
            "player": int(match.group(2)),
            "banker": int(match.group(1)),
            "path": None,
            "suffix": "",
            "hand_number": 0,
            "known_cards": {},
        }
    match = re.search(patterns[2], text)
    if match:
        return {
            "player": int(match.group(1)),
            "banker": int(match.group(2)),
            "path": None,
            "suffix": "",
            "hand_number": 0,
            "known_cards": {},
        }
    return None


def _clean_observations(
    values: Union[str, Iterable[Any], None],
) -> List[Dict[str, Any]]:
    if values is None:
        return []
    chunks = (
        [
            item
            for item in re.split(r"[;|\n]+", values)
            if item.strip()
        ]
        if isinstance(values, str)
        else list(values)
    )
    out: List[Dict[str, Any]] = []
    for item in chunks:
        parsed = parse_point_observation(item)
        if parsed is not None:
            out.append(parsed)
    return out


def _outcome(observation: Mapping[str, Any]) -> str:
    player = int(observation["player"])
    banker = int(observation["banker"])
    return (
        "B"
        if banker > player
        else "P"
        if player > banker
        else "T"
    )


def _new_seed(explicit: Optional[int] = None) -> int:
    if explicit is not None:
        return int(explicit) & 0xFFFFFFFF
    if FIXED_RUN_SEED > 0:
        return FIXED_RUN_SEED & 0xFFFFFFFF
    return (
        secrets.randbits(32)
        if RANDOMIZE_EACH_CALL
        else 20260717
    )


def _probability_dict(values: Any) -> Dict[str, float]:
    return {
        "B": float(values[0]),
        "P": float(values[1]),
        "T": float(values[2]),
    }


def _draw_path_dict(values: Any) -> Dict[str, float]:
    return {
        name: float(values[index])
        for index, name in enumerate(PATH_NAMES)
    }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _calculate_bet_sizing(
    *,
    recommend: str,
    confidence: float,
    validated_signal: bool,
    quality_pass: bool,
    decision_edge: float,
    replica_agreement: float,
    split_agreement: float,
    path_quality_score: float,
    current_path_agreement: float,
    advantage_continuity_score: float,
    decision_aggression_score: float,
) -> Dict[str, Any]:
    """Return one normalized bet-sizing payload for app.py.

    Predictor is the single source of truth for the fraction. app.py only
    converts the fraction into a rounded currency amount.
    """
    side = str(recommend or "").upper()
    minimum = _clamp(BET_MIN_FRACTION, 0.0, 1.0)
    maximum = _clamp(BET_MAX_FRACTION, minimum, 1.0)
    unvalidated_maximum = _clamp(
        BET_UNVALIDATED_MAX_FRACTION,
        minimum,
        maximum,
    )

    allowed = False
    fraction = 0.0
    level = "observe"
    level_text = "觀望"
    reason = "recommend_observe"

    if not BET_SIZING_ENABLED:
        reason = "bet_sizing_disabled"
        level_text = "未啟用"
    elif side not in {"B", "P"}:
        reason = "recommend_observe"
    elif BET_REQUIRE_VALIDATED and not validated_signal:
        reason = "validation_required"
        level_text = "等待驗證"
    else:
        allowed = True
        cap = maximum if validated_signal else unvalidated_maximum

        confidence_score = _clamp(
            (float(confidence) - 0.50) / 0.06,
            0.0,
            1.0,
        )
        edge_score = _clamp(
            abs(float(decision_edge)) / 0.012,
            0.0,
            1.0,
        )
        replica_score = _clamp(
            (float(replica_agreement) - 0.50) / 0.30,
            0.0,
            1.0,
        )
        split_score = _clamp(
            (float(split_agreement) - 0.50) / 0.30,
            0.0,
            1.0,
        )
        path_score = _clamp(float(path_quality_score), 0.0, 1.0)
        current_path_score = _clamp(
            (float(current_path_agreement) - 0.50) / 0.35,
            0.0,
            1.0,
        )
        continuity_score = _clamp(
            float(advantage_continuity_score),
            0.0,
            1.0,
        )
        aggression_score = _clamp(
            float(decision_aggression_score),
            0.0,
            1.0,
        )
        quality_bonus = 0.08 if quality_pass else 0.0

        strength = _clamp(
            0.28 * confidence_score
            + 0.17 * edge_score
            + 0.13 * replica_score
            + 0.10 * split_score
            + 0.13 * path_score
            + 0.07 * current_path_score
            + 0.07 * continuity_score
            + 0.05 * aggression_score
            + quality_bonus,
            0.0,
            1.0,
        )

        if (
            validated_signal
            and path_score >= 0.75
            and current_path_score >= 0.55
            and aggression_score >= 0.60
        ):
            strength = max(
                strength,
                min(
                    0.92,
                    0.54
                    + 0.22 * aggression_score
                    + 0.12 * continuity_score,
                ),
            )

        strength = strength ** 0.72
        fraction = minimum + (cap - minimum) * strength
        fraction = _clamp(fraction, minimum, cap)

        relative = (
            (fraction - minimum) / max(1e-12, maximum - minimum)
            if maximum > minimum
            else 0.0
        )
        if relative >= 0.66:
            level = "aggressive"
            level_text = "積極"
        elif relative >= 0.33:
            level = "balanced"
            level_text = "穩健"
        else:
            level = "conservative"
            level_text = "保守"

        reason = (
            "validated_signal"
            if validated_signal
            else "unvalidated_signal_allowed"
        )

    percentage = round(fraction * 100.0, 1)
    payload = {
        "enabled": bool(BET_SIZING_ENABLED),
        "allowed": bool(allowed),
        "fraction": round(float(fraction), 6),
        "percentage": percentage,
        "level": level,
        "level_text": level_text,
        "reason": reason,
        "minimum_fraction": round(minimum, 6),
        "maximum_fraction": round(maximum, 6),
        "unvalidated_maximum_fraction": round(unvalidated_maximum, 6),
        "requires_validated_signal": bool(BET_REQUIRE_VALIDATED),
        "advantage_continuity_score": round(
            _clamp(float(advantage_continuity_score), 0.0, 1.0),
            6,
        ),
        "decision_aggression_score": round(
            _clamp(float(decision_aggression_score), 0.0, 1.0),
            6,
        ),
    }
    return {
        "bet_sizing": payload,
        "bet_allowed": payload["allowed"],
        "bet_fraction": payload["fraction"],
        "bet_percentage": payload["percentage"],
        "bet_level": payload["level"],
        "bet_level_text": payload["level_text"],
        "bet_reason": payload["reason"],
    }


def predict(
    history: Union[str, Iterable[Any]],
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
    user_id: str = "",
    run_seed: Optional[int] = None,
    shoe_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    observations = _clean_observations(history)
    latest = observations[-1] if observations else None
    if latest is None:
        return {
            "ok": False,
            "error": "missing_point_observation",
            "message": (
                "請輸入本局點數與補牌代號，例如382或571；"
                "前兩碼是閒家、莊家點數，第三碼為補牌代號："
                "1=僅閒補、2=僅莊補、3=雙方補、4=雙方不補。"
            ),
        }

    # Independent mode still rebuilds all particles on every request.  It may
    # use factual shoe context explicitly supplied with this request, but never
    # carries road history, previous recommendations or particle direction.
    context = dict(shoe_context) if isinstance(shoe_context, Mapping) else {}
    raw_hand_number = context.get(
        "hand_number",
        latest.get("hand_number", 0),
    )
    try:
        hand_number = max(0, min(120, int(raw_hand_number or 0)))
    except Exception:
        hand_number = 0

    known_cards = _normalize_known_cards(
        context.get("known_cards")
        or context.get("cards")
        or latest.get("known_cards")
        or {}
    )
    remaining_counts = _normalize_counts(
        context.get("remaining_counts")
        if context.get("remaining_counts") is not None
        else latest.get("remaining_counts")
    )
    state_complete = bool(
        context.get(
            "state_complete",
            latest.get("state_complete", False),
        )
        and remaining_counts is not None
    )
    state_source = str(
        context.get("state_source")
        or latest.get("state_source")
        or (
            "REQUEST_EXACT_SHOE_STATE"
            if state_complete
            else "REQUEST_FACTUAL_DEPTH"
            if hand_number > 0
            else "INDEPENDENT_CURRENT_HAND"
        )
    )
    try:
        tracked_card_hands = max(
            0,
            int(
                context.get(
                    "tracked_card_hands",
                    latest.get("tracked_card_hands", 0),
                )
                or 0
            ),
        )
    except Exception:
        tracked_card_hands = 0

    seed = _new_seed(run_seed)
    engine = V5IndependentBaccaratEngine()
    try:
        result = engine.analyze(
            latest["player"],
            latest["banker"],
            seed,
            latest.get("path"),
            hand_number=hand_number or None,
            known_cards=known_cards or None,
            remaining_counts=remaining_counts,
            state_complete=state_complete,
        )
    except ValueError as exception:
        return {
            "ok": False,
            "error": "invalid_shoe_context",
            "message": str(exception),
        }

    probabilities = _probability_dict(result["fused"])
    pf_probabilities = _probability_dict(result["pf"])
    control_probabilities = _probability_dict(result["control"])
    db_probabilities = _probability_dict(result["database"])
    shoe_state_probabilities = _probability_dict(
        result["shoe_state"]
    )
    particle_db_probabilities = _probability_dict(
        result["particle_database_fused"]
    )
    independent_path_model = dict(
        result.get("independent_draw_path_model") or {}
    )
    independent_path_probabilities = _probability_dict(
        independent_path_model.get("probabilities", result["pf"])
    )
    independent_path_control_probabilities = _probability_dict(
        independent_path_model.get("control_probabilities", result["control"])
    )

    raw_recommend = str(result["recommend"])
    is_observe = bool(
        OBSERVE_ON_UNVALIDATED
        and not result.get("validated_signal", False)
    )
    recommend = "O" if is_observe else raw_recommend
    confidence = max(
        probabilities["B"],
        probabilities["P"],
    ) / max(
        1e-12,
        probabilities["B"] + probabilities["P"],
    )

    point_text = (
        f"{latest['player']}{latest['banker']}"
        f"{latest.get('suffix', '')}"
    )

    particle_count = int(result["settings"]["particles"])
    hybrid = dict(result.get("hybrid") or {})
    hybrid_weights = dict(hybrid.get("weights") or {})

    bet_sizing = _calculate_bet_sizing(
        recommend=recommend,
        confidence=confidence,
        validated_signal=bool(result.get("validated_signal", False)),
        quality_pass=bool(result.get("quality_pass", False)),
        decision_edge=float(result.get("edge", 0.0)),
        replica_agreement=float(result.get("replica_agreement", 0.5)),
        split_agreement=float(result.get("split_agreement", 0.5)),
        path_quality_score=float(result.get("path_quality_score", 0.0)),
        current_path_agreement=float(
            result.get("average_current_path_agreement", 0.5)
        ),
        advantage_continuity_score=float(
            result.get("advantage_continuity_score", 0.0)
        ),
        decision_aggression_score=float(
            result.get("decision_aggression_score", 0.0)
        ),
    )

    response: Dict[str, Any] = {
        "ok": True,
        "engine": (
            f"V5_4_1_AGGRESSIVE_DRAW_PATH_{particle_count}_PARTICLE_LINE"
        ),
        "model_version": (
            f"V5.4.1-AGGRESSIVE-PATH-QUALITY-DYNAMIC-GATE-{particle_count}P-LINE-20260719"
        ),
        "user_id": user_id,
        "venue": venue,
        "room": room,
        "shoe_id": shoe_id,
        "run_seed": seed,
        "fresh_particle_mode": True,
        "persistent_particle_state": False,
        "persistent_shoe_context": False,
        "road_history_used": False,
        "used_observation_count": 1,
        "ignored_prior_observations": max(
            0,
            len(observations) - 1,
        ),
        "conditioning_point": point_text,
        "conditioning_observation": {
            "player": latest["player"],
            "banker": latest["banker"],
            "hand_number": hand_number,
        },
        "conditioning_outcome": _outcome(latest),
        "known_draw_path": (
            PATH_NAMES[latest["path"]]
            if latest.get("path") is not None
            else None
        ),
        "known_cards": known_cards,
        "banker_rate": round(
            probabilities["B"] * 100.0,
            1,
        ),
        "player_rate": round(
            probabilities["P"] * 100.0,
            1,
        ),
        "tie_rate": round(
            probabilities["T"] * 100.0,
            1,
        ),
        "probabilities": probabilities,
        "hybrid_probabilities": probabilities,
        "particle_probabilities": pf_probabilities,
        "particle_database_probabilities": (
            particle_db_probabilities
        ),
        "control_probabilities": control_probabilities,
        "shoe_database_probabilities": db_probabilities,
        "exact_shoe_state_probabilities": (
            shoe_state_probabilities
        ),
        "hybrid": {
            "mode": hybrid.get("mode", "hybrid"),
            "gate": round(float(hybrid.get("gate", 0.0)), 6),
            "weights": {
                key: round(float(value), 6)
                for key, value in hybrid_weights.items()
            },
            "exact_state_enabled": bool(
                hybrid.get("exact_state_enabled", False)
            ),
            "state_reliability": round(
                float(
                    hybrid.get("state_reliability", 0.0)
                ),
                6,
            ),
            "card_validation": str(
                hybrid.get("card_validation") or ""
            ),
            "independent_path_reliability": round(
                float(hybrid.get("independent_path_reliability", 0.0)),
                6,
            ),
            "independent_path_effective_weight": round(
                float(hybrid.get("independent_path_effective_weight", 0.0)),
                6,
            ),
            "path_gate": round(
                float(hybrid.get("path_gate", 0.0)),
                6,
            ),
            "path_coverage_quality": round(
                float(hybrid.get("path_coverage_quality", 0.0)),
                6,
            ),
            "path_ess_quality": round(
                float(hybrid.get("path_ess_quality", 0.0)),
                6,
            ),
            "current_path_agreement_quality": round(
                float(hybrid.get("current_path_agreement_quality", 0.0)),
                6,
            ),
            "configured_particle_max_weight": round(
                float(hybrid.get("configured_particle_max_weight", 0.0)),
                6,
            ),
            "dynamic_particle_max_weight": round(
                float(hybrid.get("dynamic_particle_max_weight", 0.0)),
                6,
            ),
            "high_path_threshold": round(
                float(hybrid.get("high_path_threshold", 0.75)),
                6,
            ),
            "high_path_particle_boost": round(
                float(hybrid.get("high_path_particle_boost", 0.0)),
                6,
            ),
        },
        "independent_draw_path_model": {
            "enabled": bool(independent_path_model.get("enabled", False)),
            "probabilities": independent_path_probabilities,
            "control_probabilities": independent_path_control_probabilities,
            "next_draw_paths": _draw_path_dict(
                independent_path_model.get(
                    "next_draw_paths",
                    result["next_draw_paths"],
                )
            ),
            "path_outcome_probabilities": {
                path_name: _probability_dict(row)
                for path_name, row in zip(
                    PATH_NAMES,
                    independent_path_model.get(
                        "path_outcome_probabilities",
                        [[0.0, 0.0, 0.0]] * 4,
                    ),
                )
            },
            "control_path_outcome_probabilities": {
                path_name: _probability_dict(row)
                for path_name, row in zip(
                    PATH_NAMES,
                    independent_path_model.get(
                        "control_path_outcome_probabilities",
                        [[0.0, 0.0, 0.0]] * 4,
                    ),
                )
            },
            "path_support": {
                path_name: round(float(value), 6)
                for path_name, value in zip(
                    PATH_NAMES,
                    independent_path_model.get(
                        "path_support",
                        [0.0, 0.0, 0.0, 0.0],
                    ),
                )
            },
            "reliability": round(
                float(independent_path_model.get("reliability", 0.0)),
                6,
            ),
            "minimum_reliability": round(
                float(
                    independent_path_model.get(
                        "minimum_reliability",
                        result["settings"].get(
                            "independent_path_model_min_reliability",
                            0.55,
                        ),
                    )
                ),
                6,
            ),
            "configured_max_weight": round(
                float(independent_path_model.get("configured_max_weight", 0.0)),
                6,
            ),
            "configured_late_max_weight": round(
                float(
                    independent_path_model.get(
                        "configured_late_max_weight",
                        result["settings"].get(
                            "independent_path_model_late_max_weight",
                            0.24,
                        ),
                    )
                ),
                6,
            ),
            "effective_weight": round(
                float(independent_path_model.get("effective_weight", 0.0)),
                6,
            ),
            "direction_agreement": round(
                float(independent_path_model.get("direction_agreement", 0.5)),
                6,
            ),
            "weighted_support": round(
                float(independent_path_model.get("weighted_support", 0.0)),
                6,
            ),
            "residual_shrinkage": round(
                float(independent_path_model.get("residual_shrinkage", 0.0)),
                6,
            ),
            "shoe_depth": round(
                float(independent_path_model.get("shoe_depth", 0.0)),
                6,
            ),
            "depth_factor": round(
                float(independent_path_model.get("depth_factor", 0.0)),
                6,
            ),
            "depth_consistency": round(
                float(independent_path_model.get("depth_consistency", 0.0)),
                6,
            ),
            "dynamic_max_weight": round(
                float(independent_path_model.get("dynamic_max_weight", 0.0)),
                6,
            ),
            "dynamic_prior_strength": {
                path_name: round(float(value), 6)
                for path_name, value in zip(
                    PATH_NAMES,
                    independent_path_model.get(
                        "dynamic_prior_strength",
                        [0.0, 0.0, 0.0, 0.0],
                    ),
                )
            },
            "residual_adjustment": _probability_dict(
                independent_path_model.get(
                    "residual_adjustment",
                    [0.0, 0.0, 0.0],
                )
            ),
            "max_adjustment": round(
                float(independent_path_model.get("max_adjustment", 0.0)),
                6,
            ),
            "uses_additional_simulations": bool(
                independent_path_model.get("uses_additional_simulations", False)
            ),
        },
        "shoe_context": {
            "hand_number": hand_number,
            "state_complete": state_complete,
            "state_source": state_source,
            "tracked_card_hands": tracked_card_hands,
            "exact_state_enabled": bool(
                hybrid.get("exact_state_enabled", False)
            ),
        },
        "recommend": recommend,
        "raw_recommend": raw_recommend,
        "recommend_text": (
            "觀望"
            if recommend == "O"
            else "莊"
            if recommend == "B"
            else "閒"
        ),
        "is_observe": is_observe,
        "confidence": round(confidence, 6),
        "confidence_pct": round(confidence * 100.0, 1),
        "decision_edge": round(
            float(result["edge"]),
            8,
        ),
        "signal_level": str(result["signal_level"]),
        "decision_source": str(
            result["decision_source"]
        ),
        "validated_signal": bool(
            result["validated_signal"]
        ),
        "quality_pass": bool(
            result.get("quality_pass", False)
        ),
        "lower_bound": round(
            float(result["lower_bound"]),
            8,
        ),
        "centered_edge": round(
            float(result["center"]),
            8,
        ),
        "center_se": round(
            float(result["center_se"]),
            8,
        ),
        "replica_count": int(result["replicas"]),
        "replica_directions": list(
            result["replica_directions"]
        ),
        "replica_agreement": round(
            float(result["replica_agreement"]),
            6,
        ),
        "split_agreement": round(
            float(result["split_agreement"]),
            6,
        ),
        "effective_replicas": round(
            float(result["effective_replicas"]),
            4,
        ),
        "stability": str(result["stability"]),
        "weakness_reason": str(
            result["weakness_reason"]
        ),
        "current_point_draw_paths": _draw_path_dict(
            result["draw_paths"]
        ),
        "next_hand_draw_paths": _draw_path_dict(
            result["next_draw_paths"]
        ),
        "raw_next_hand_draw_paths": _draw_path_dict(
            result.get("next_draw_paths_raw", result["next_draw_paths"])
        ),
        "current_path_transition_prior": _draw_path_dict(
            result.get("path_transition_prior", result["next_draw_paths"])
        ),
        "top_points": list(result["top_points"]),
        "draw_path_diagnostics": {
            "coverage": round(
                float(result["average_path_coverage"]),
                6,
            ),
            "legacy_coverage": round(
                float(
                    result[
                        "average_legacy_path_coverage"
                    ]
                ),
                6,
            ),
            "ess_quality": round(
                float(
                    result["average_path_ess_quality"]
                ),
                6,
            ),
            "quality_score": round(
                float(result.get("average_path_quality", 0.0)),
                6,
            ),
            "fusion_gain": round(
                float(result.get("average_path_fusion_gain", 0.0)),
                6,
            ),
            "hybrid_path_gate": round(
                float(hybrid.get("path_gate", 0.0)),
                6,
            ),
            "quality_pass": bool(
                result.get("path_quality_pass", False)
            ),
            "quality_threshold": round(
                float(result.get("path_quality_threshold", 0.0)),
                6,
            ),
            "decision_quality_score": round(
                float(result.get("path_quality_score", 0.0)),
                6,
            ),
            "path_ess_floor": round(
                float(result.get("path_ess_floor", 0.0)),
                6,
            ),
            "current_path_agreement_floor": round(
                float(result.get("current_path_agreement_floor", 0.0)),
                6,
            ),
            "current_path_agreement_quality": round(
                float(result.get("current_path_agreement_quality", 0.0)),
                6,
            ),
            "transition_prior_weight": round(
                float(result.get("average_path_transition_weight", 0.0)),
                6,
            ),
            "transition_prior_quality": round(
                float(result.get("average_path_transition_quality", 0.0)),
                6,
            ),
            "base_min_validated_edge": round(
                float(result.get("base_min_validated_edge", 0.0)),
                8,
            ),
            "effective_min_validated_edge": round(
                float(result.get("effective_min_validated_edge", 0.0)),
                8,
            ),
            "validated_edge_relief": round(
                float(result.get("validated_edge_relief", 0.0)),
                6,
            ),
            "advantage_continuity_score": round(
                float(result.get("advantage_continuity_score", 0.0)),
                6,
            ),
            "advantage_continuity_bonus": round(
                float(result.get("advantage_continuity_bonus", 0.0)),
                8,
            ),
            "decision_confidence": round(
                float(result.get("decision_confidence", 0.5)),
                6,
            ),
            "decision_aggression_score": round(
                float(result.get("decision_aggression_score", 0.0)),
                6,
            ),
            "candidates": [
                round(float(item), 2)
                for item in result[
                    "average_path_candidates"
                ]
            ],
            "ess": [
                round(float(item), 2)
                for item in result["average_path_ess"]
            ],
            "allocated": [
                round(float(item), 2)
                for item in result[
                    "average_path_allocated"
                ]
            ],
            "current_path_agreement": round(
                float(
                    result[
                        "average_current_path_agreement"
                    ]
                ),
                6,
            ),
            "next_draw_agreement": round(
                float(
                    result["average_draw_agreement"]
                ),
                6,
            ),
        },
        "point_particle_filter": {
            "particles_per_replica": particle_count,
            "particle_limit": 2000,
            "replicas": int(result["replicas"]),
            "target_matches": int(
                result["settings"]["target_matches"]
            ),
            "target_ess": round(
                float(result["settings"]["target_ess"]),
                3,
            ),
            "path_target_matches": int(
                result["settings"][
                    "path_target_matches"
                ]
            ),
            "minimum_particles_per_draw_path": int(
                result["settings"]["min_path_particles"]
            ),
            "independent_path_model_enabled": bool(
                result["settings"]["independent_path_model_enabled"]
            ),
            "independent_path_model_max_weight": round(
                float(result["settings"]["independent_path_model_max_weight"]),
                6,
            ),
            "independent_path_model_min_reliability": round(
                float(
                    result["settings"][
                        "independent_path_model_min_reliability"
                    ]
                ),
                6,
            ),
            "independent_path_model_max_adjustment": round(
                float(
                    result["settings"][
                        "independent_path_model_max_adjustment"
                    ]
                ),
                6,
            ),
            "independent_path_model_late_max_weight": round(
                float(
                    result["settings"][
                        "independent_path_model_late_max_weight"
                    ]
                ),
                6,
            ),
            "independent_path_model_depth_start": round(
                float(
                    result["settings"][
                        "independent_path_model_depth_start"
                    ]
                ),
                6,
            ),
            "independent_path_model_depth_full": round(
                float(
                    result["settings"][
                        "independent_path_model_depth_full"
                    ]
                ),
                6,
            ),
            "independent_path_model_residual_shrink_power": round(
                float(
                    result["settings"][
                        "independent_path_model_residual_shrink_power"
                    ]
                ),
                6,
            ),
            "third_card_likelihood_power": round(
                float(result["settings"]["third_card_likelihood_power"]),
                6,
            ),
            "no_draw_allocation_protection": round(
                float(result["settings"]["no_draw_allocation_protection"]),
                6,
            ),
            "path_transition_prior_weight": round(
                float(result["settings"]["path_transition_prior_weight"]),
                6,
            ),
            "path_transition_self_bias": round(
                float(result["settings"]["path_transition_self_bias"]),
                6,
            ),
            "forecast_simulations_per_replica": max(
                int(
                    result["settings"][
                        "predict_simulations_per_replica"
                    ]
                )
                + int(
                    result["settings"][
                        "point_joint_simulations_per_replica"
                    ]
                ),
                particle_count * 2,
            ),
            "average_matches": round(
                float(result["average_matches"]),
                3,
            ),
            "average_effective_sample_size": round(
                float(result["average_ess"]),
                3,
            ),
            "average_acceptance": round(
                float(result["average_acceptance"]),
                8,
            ),
            "average_attempts": round(
                float(result["average_attempts"]),
                3,
            ),
            "average_diversity": round(
                float(result["average_diversity"]),
                6,
            ),
            "total_forecast_simulations": int(
                result["total_forecast_simulations"]
            ),
            "total_condition_attempts": int(
                result["total_condition_attempts"]
            ),
            "state_digest": str(
                result["state_digest"]
            ),
            "conditional_generator": str(
                result["conditional_generator"]
            ),
            "variance_reduction": str(
                result["variance_reduction"]
            ),
            "depth_profile": str(
                result["depth_profile"]
            ),
        },
        "shoe_state_database": {
            **get_shoe_state_database().database_info(),
            "probabilities": db_probabilities,
            "average_samples": round(
                float(result["database_samples"]),
                3,
            ),
            "effective_weight": round(
                float(
                    result["average_database_weight"]
                ),
                8,
            ),
            "holdout": dict(DB_HOLDOUT),
        },
        "reason": (
            f"V5.4.1 HYBRID：{particle_count}粒子×"
            f"{int(result['replicas'])}副本；"
            "每次只使用最新一局最終點數、該局補牌路徑與本次明確傳入的牌靴事實獨立模擬；"
            "第三張牌條件似然、雙方不補粒子保護、路徑覆蓋/ESS/當前路徑共識已提高決策權重；"
            "高補牌品質時會動態提高粒子融合上限並降低驗證門檻，方向連續性則以同一次分析內的"
            "副本/分割/補牌分支一致性加分；另以當前補牌路徑加入較強但受品質限制的單步先驗，"
            "不讀取更早牌路、不使用學習式Markov、長龍、"
            "上一局推薦、勝敗紀錄或持久粒子方向，且不增加模擬輪數。"
            f"決策來源={result['decision_source']}；"
            f"{result['reason']}。"
        ),
        "debug": None,
    }
    response.update(bet_sizing)

    if DEBUG_V5_RESULT:
        response["debug"] = {
            "settings": result["settings"],
            "configured_settings": result[
                "configured_settings"
            ],
            "votes": result["votes"],
            "weighted_votes": result[
                "weighted_votes"
            ],
            "outlier_count": result["outlier_count"],
            "robust_mad": result["robust_mad"],
            "hybrid": hybrid,
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
    result = predict(
        history,
        venue,
        room,
        shoe_id,
        user_id,
    )
    return {
        "ok": bool(result.get("ok")),
        "model": result.get("engine"),
        "independent_samples": int(
            result.get("used_observation_count", 0)
        ),
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
        "fresh_particle_mode": True,
        "message": (
            "V5.4.1每次重建粒子；請由store.reset_shoe清除"
            "實體牌靴資訊。"
        ),
    }


def clear_model_cache(
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "ok": True,
        "removed": 0,
        "fresh_particle_mode": True,
    }
