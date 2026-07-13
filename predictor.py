"""Point-conditioned baccarat predictor with card-depletion particle filtering
and optional DeepSeek calibration.

Primary model
-------------
* Each observation contains final Player/Banker points.
* A particle filter tracks many possible remaining 8-deck shoe states.
* The next-hand distribution is simulated locally from those weighted shoes.

DeepSeek layer
--------------
* DeepSeek is retained as a low-weight calibration/confirmation layer.
* Async cache mode is enabled by default so LINE replies do not wait for API.
* The particle filter remains the primary model even when DeepSeek is available.

Public compatibility
--------------------
    predict(history_or_observations, venue='', room='', shoe_id='', user_id='')

Preferred observations
----------------------
    [{'player': 6, 'banker': 5}, {'player': 2, 'banker': 8}]

Accepted text examples
----------------------
    P6B5
    閒6莊5
    6,5
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from particle_filter_points import PointParticleFilter

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        value = int(os.getenv(name, str(default)).strip())
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _env_float(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        value = float(os.getenv(name, str(default)).strip())
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


# ---------------------------------------------------------------------------
# Core settings
# ---------------------------------------------------------------------------

MAX_FILTERS = _env_int("PF_MAX_FILTERS", 32, 1)
RECOMMEND_TIE = _env_bool("PF_RECOMMEND_TIE", False)
MAX_DISPLAY_CONFIDENCE = _env_float(
    "PF_MAX_DISPLAY_CONFIDENCE",
    0.64,
    0.50,
    0.80,
)

# DeepSeek is deliberately a low-weight layer.
USE_DEEPSEEK = _env_bool("USE_DEEPSEEK", True)
DEEPSEEK_WEIGHT = _env_float("DEEPSEEK_WEIGHT", 0.08, 0.0, 0.30)
DEEPSEEK_MIN_OBSERVATIONS = _env_int("DEEPSEEK_MIN_OBSERVATIONS", 3, 0)
DEEPSEEK_TIMEOUT_SECONDS = _env_float(
    "DEEPSEEK_TIMEOUT_SECONDS",
    2.5,
    0.5,
    15.0,
)
DEEPSEEK_ASYNC_MODE = _env_bool("DEEPSEEK_ASYNC_MODE", True)
DEEPSEEK_CALL_INTERVAL = _env_int("DEEPSEEK_CALL_INTERVAL", 2, 1)
DEEPSEEK_CACHE_MAX_AGE_HANDS = _env_int(
    "DEEPSEEK_CACHE_MAX_AGE_HANDS",
    2,
    0,
)
DEEPSEEK_FAILURE_COOLDOWN_HANDS = _env_int(
    "DEEPSEEK_FAILURE_COOLDOWN_HANDS",
    3,
    0,
)
DEEPSEEK_CONFIRM_ONLY = _env_bool("DEEPSEEK_CONFIRM_ONLY", False)
DEEPSEEK_CONFLICT_WEIGHT_SCALE = _env_float(
    "DEEPSEEK_CONFLICT_WEIGHT_SCALE",
    0.50,
    0.0,
    1.0,
)
DEBUG_AI_RESULT = _env_bool("DEBUG_AI_RESULT", False)

# ---------------------------------------------------------------------------
# Optional DeepSeek client
# ---------------------------------------------------------------------------

_DEEPSEEK_IMPORT_ERROR = ""
try:
    from deepseek_client import DeepSeekClient  # type: ignore
except Exception as exc:  # pragma: no cover
    _DEEPSEEK_IMPORT_ERROR = str(exc)

    class DeepSeekClient:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            self.import_error = _DEEPSEEK_IMPORT_ERROR

        def calibrate(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
            return {
                "error": True,
                "message": f"DeepSeekClient unavailable: {self.import_error}",
            }


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_FILTERS: "OrderedDict[str, PointParticleFilter]" = OrderedDict()
_FILTER_LOCK = threading.RLock()
_OBS_COUNTS: Dict[str, int] = {}

_DEEPSEEK_CLIENT: Optional[Any] = None
_DEEPSEEK_CLIENT_LOCK = threading.RLock()
_DEEPSEEK_EXECUTOR = ThreadPoolExecutor(
    max_workers=_env_int("DEEPSEEK_WORKERS", 2, 1)
)
_DEEPSEEK_CACHE_LOCK = threading.RLock()
_DEEPSEEK_CACHE: Dict[str, Dict[str, Any]] = {}
_DEEPSEEK_FUTURES: Dict[str, Future[Any]] = {}


# ---------------------------------------------------------------------------
# Particle-filter helpers
# ---------------------------------------------------------------------------


def _key(user_id: str, venue: str, room: str, shoe_id: str) -> str:
    return "|".join(
        [
            user_id or "anonymous",
            venue or "global",
            room or "global",
            shoe_id or "global",
        ]
    )


def _get_filter(key: str) -> PointParticleFilter:
    with _FILTER_LOCK:
        if key in _FILTERS:
            _FILTERS.move_to_end(key)
            return _FILTERS[key]

        while len(_FILTERS) >= MAX_FILTERS:
            old_key, _ = _FILTERS.popitem(last=False)
            _OBS_COUNTS.pop(old_key, None)
            with _DEEPSEEK_CACHE_LOCK:
                _DEEPSEEK_CACHE.pop(old_key, None)
                _DEEPSEEK_FUTURES.pop(old_key, None)

        particle_filter = PointParticleFilter(key)
        _FILTERS[key] = particle_filter
        _OBS_COUNTS[key] = 0
        return particle_filter


def parse_point_observation(value: Any) -> Optional[Dict[str, int]]:
    if isinstance(value, Mapping):
        player = value.get("player", value.get("P", value.get("閒")))
        banker = value.get("banker", value.get("B", value.get("莊")))
        try:
            return {
                "player": int(player) % 10,
                "banker": int(banker) % 10,
            }
        except Exception:
            return None

    text = str(value or "").strip().upper()
    patterns = [
        r"(?:P|PLAYER|閒|闲)\s*([0-9])\D+(?:B|BANKER|莊|庄)\s*([0-9])",
        r"(?:B|BANKER|莊|庄)\s*([0-9])\D+(?:P|PLAYER|閒|闲)\s*([0-9])",
        r"^\s*([0-9])\s*[,/\- ]\s*([0-9])\s*$",
    ]

    match = re.search(patterns[0], text)
    if match:
        return {
            "player": int(match.group(1)),
            "banker": int(match.group(2)),
        }

    match = re.search(patterns[1], text)
    if match:
        return {
            "player": int(match.group(2)),
            "banker": int(match.group(1)),
        }

    match = re.search(patterns[2], text)
    if match:
        return {
            "player": int(match.group(1)),
            "banker": int(match.group(2)),
        }

    return None


def _clean_observations(
    values: Union[str, Iterable[Any], None],
) -> List[Dict[str, int]]:
    if values is None:
        return []

    if isinstance(values, str):
        chunks = [
            item
            for item in re.split(r"[;|\n]+", values)
            if item.strip()
        ]
    else:
        chunks = list(values)

    observations: List[Dict[str, int]] = []
    for item in chunks:
        parsed = parse_point_observation(item)
        if parsed is not None:
            observations.append(parsed)
    return observations


def _cap_probs(probs: Mapping[str, float]) -> Dict[str, float]:
    tie = max(0.0, min(0.30, float(probs.get("T", 0.0))))
    banker = max(0.0, float(probs.get("B", 0.0)))
    player = max(0.0, float(probs.get("P", 0.0)))

    non_tie = max(1e-12, banker + player)
    banker_share = banker / non_tie
    player_share = player / non_tie

    if max(banker_share, player_share) > MAX_DISPLAY_CONFIDENCE:
        if banker_share >= player_share:
            banker_share = MAX_DISPLAY_CONFIDENCE
            player_share = 1.0 - MAX_DISPLAY_CONFIDENCE
        else:
            player_share = MAX_DISPLAY_CONFIDENCE
            banker_share = 1.0 - MAX_DISPLAY_CONFIDENCE

    return {
        "B": banker_share * (1.0 - tie),
        "P": player_share * (1.0 - tie),
        "T": tie,
    }


def _normalize_triplet(values: Mapping[str, Any]) -> Dict[str, float]:
    numbers = {
        name: max(0.0, float(values.get(name, 0.0) or 0.0))
        for name in ("B", "P", "T")
    }
    total = sum(numbers.values())
    if total <= 0.0:
        return {"B": 0.4586, "P": 0.4462, "T": 0.0952}
    return {name: value / total for name, value in numbers.items()}


def _observation_outcome(observation: Mapping[str, int]) -> str:
    player = int(observation["player"])
    banker = int(observation["banker"])
    return "B" if banker > player else "P" if player > banker else "T"


def _observations_fingerprint(
    observations: Sequence[Mapping[str, int]],
) -> str:
    compact = "|".join(
        f"{int(item['player'])}{int(item['banker'])}"
        for item in observations
    )
    return hashlib.sha1(compact.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# DeepSeek parsing and cache
# ---------------------------------------------------------------------------


def _get_deepseek_client() -> Any:
    global _DEEPSEEK_CLIENT
    with _DEEPSEEK_CLIENT_LOCK:
        if _DEEPSEEK_CLIENT is None:
            _DEEPSEEK_CLIENT = DeepSeekClient()
        return _DEEPSEEK_CLIENT


def _as_probability(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if 1.0 < number <= 100.0:
        number /= 100.0
    return max(0.0, min(1.0, number))


def _extract_json_mapping(value: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return value

    if not isinstance(value, str):
        return None

    text = value.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    candidates = [text]
    object_match = re.search(r"\{.*\}", text, flags=re.S)
    if object_match:
        candidates.append(object_match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, Mapping):
            return parsed

    return None


def _extract_deepseek_probs(
    raw: Any,
    fallback: Mapping[str, float],
) -> Optional[Dict[str, float]]:
    mapping = _extract_json_mapping(raw)
    if mapping is None:
        return None

    for key in (
        "probabilities",
        "probs",
        "prediction",
        "result",
        "distribution",
        "data",
        "content",
        "message",
    ):
        nested = mapping.get(key)
        nested_mapping = _extract_json_mapping(nested)
        if nested_mapping is not None:
            parsed = _extract_deepseek_probs(nested_mapping, fallback)
            if parsed is not None:
                return parsed

    key_sets = [
        ("B", "P", "T"),
        ("b", "p", "t"),
        ("banker", "player", "tie"),
        ("banker_prob", "player_prob", "tie_prob"),
        ("banker_probability", "player_probability", "tie_probability"),
        ("banker_rate", "player_rate", "tie_rate"),
    ]

    for banker_key, player_key, tie_key in key_sets:
        banker = _as_probability(mapping.get(banker_key))
        player = _as_probability(mapping.get(player_key))
        tie = _as_probability(mapping.get(tie_key))
        if banker is not None and player is not None:
            if tie is None:
                tie = float(fallback.get("T", 0.0952))
            return _normalize_triplet(
                {"B": banker, "P": player, "T": tie}
            )

    recommendation = str(
        mapping.get("recommend")
        or mapping.get("recommendation")
        or mapping.get("side")
        or mapping.get("direction")
        or ""
    ).strip().upper()

    aliases = {
        "BANKER": "B",
        "PLAYER": "P",
        "TIE": "T",
        "莊": "B",
        "庄": "B",
        "閒": "P",
        "闲": "P",
        "和": "T",
    }
    recommendation = aliases.get(recommendation, recommendation)

    if recommendation in {"B", "P", "T"}:
        confidence = _as_probability(mapping.get("confidence"))
        confidence = 0.55 if confidence is None else max(
            0.34,
            min(0.75, confidence),
        )
        remainder = 1.0 - confidence
        others = [name for name in ("B", "P", "T") if name != recommendation]
        result = {
            recommendation: confidence,
            others[0]: remainder * 0.55,
            others[1]: remainder * 0.45,
        }
        return _normalize_triplet(result)

    return None


def _deepseek_payload(
    observations: Sequence[Mapping[str, int]],
    particle_probs: Mapping[str, float],
    particle_diagnostics: Mapping[str, Any],
    user_id: str,
    venue: str,
    room: str,
    shoe_id: str,
) -> Dict[str, Any]:
    recent = [
        {
            "player": int(item["player"]),
            "banker": int(item["banker"]),
            "outcome": _observation_outcome(item),
        }
        for item in observations[-24:]
    ]

    return {
        "task": "baccarat_point_particle_filter_calibration",
        "instruction": (
            "Use only the supplied point observations and particle-filter "
            "distribution. Return strict JSON only with numeric keys B, P, T "
            "that sum to 1. Do not provide betting advice, prose, markdown, "
            "or guaranteed-win claims."
        ),
        "classes": ["B", "P", "T"],
        "user_id": user_id,
        "venue": venue,
        "room": room,
        "shoe_id": shoe_id,
        "observation_count": len(observations),
        "recent_point_observations": recent,
        "particle_filter_probabilities": dict(particle_probs),
        "particle_effective_sample_size": particle_diagnostics.get(
            "effective_sample_size"
        ),
        "particle_simulations": particle_diagnostics.get("simulations"),
        "timeout_seconds": DEEPSEEK_TIMEOUT_SECONDS,
    }


def _call_deepseek(
    payload: Mapping[str, Any],
    fallback: Mapping[str, float],
) -> Dict[str, Any]:
    started = time.monotonic()
    try:
        raw = _get_deepseek_client().calibrate(payload)
    except Exception as exc:
        return {
            "ok": False,
            "status": "call_error",
            "error": str(exc),
            "elapsed_seconds": round(time.monotonic() - started, 4),
        }

    if isinstance(raw, Mapping) and raw.get("error"):
        return {
            "ok": False,
            "status": "api_error",
            "error": str(raw.get("message") or raw),
            "raw": dict(raw) if DEBUG_AI_RESULT else None,
            "elapsed_seconds": round(time.monotonic() - started, 4),
        }

    parsed = _extract_deepseek_probs(raw, fallback)
    if parsed is None:
        return {
            "ok": False,
            "status": "unrecognized_response",
            "raw": raw if DEBUG_AI_RESULT else None,
            "elapsed_seconds": round(time.monotonic() - started, 4),
        }

    return {
        "ok": True,
        "status": "ready",
        "probabilities": parsed,
        "raw": raw if DEBUG_AI_RESULT else None,
        "elapsed_seconds": round(time.monotonic() - started, 4),
    }


def _store_deepseek_result(
    key: str,
    fingerprint: str,
    observation_count: int,
    result: Mapping[str, Any],
) -> None:
    with _DEEPSEEK_CACHE_LOCK:
        previous = _DEEPSEEK_CACHE.get(key) or {}
        failure_count = (
            0
            if result.get("ok")
            else int(previous.get("failure_count", 0)) + 1
        )
        cooldown_until = (
            observation_count + DEEPSEEK_FAILURE_COOLDOWN_HANDS
            if not result.get("ok")
            else observation_count
        )
        _DEEPSEEK_CACHE[key] = {
            **dict(result),
            "fingerprint": fingerprint,
            "observation_count": observation_count,
            "failure_count": failure_count,
            "cooldown_until_observation": cooldown_until,
            "updated_at": time.time(),
        }


def _deepseek_future_callback(
    key: str,
    fingerprint: str,
    observation_count: int,
    future: Future[Any],
) -> None:
    try:
        result = future.result()
    except Exception as exc:
        result = {
            "ok": False,
            "status": "background_error",
            "error": str(exc),
        }

    _store_deepseek_result(
        key,
        fingerprint,
        observation_count,
        result,
    )

    with _DEEPSEEK_CACHE_LOCK:
        _DEEPSEEK_FUTURES.pop(key, None)


def _cached_deepseek(
    key: str,
    observation_count: int,
) -> Tuple[Optional[Dict[str, float]], Dict[str, Any]]:
    with _DEEPSEEK_CACHE_LOCK:
        cached = dict(_DEEPSEEK_CACHE.get(key) or {})

    if not cached:
        return None, {"status": "cache_empty"}

    age_hands = max(
        0,
        observation_count - int(cached.get("observation_count", 0)),
    )
    if age_hands > DEEPSEEK_CACHE_MAX_AGE_HANDS:
        return None, {
            "status": "cache_stale",
            "age_hands": age_hands,
        }

    if not cached.get("ok"):
        return None, {
            "status": str(cached.get("status") or "cached_error"),
            "age_hands": age_hands,
            "error": cached.get("error"),
        }

    probabilities = cached.get("probabilities")
    if not isinstance(probabilities, Mapping):
        return None, {
            "status": "cache_invalid",
            "age_hands": age_hands,
        }

    return _normalize_triplet(probabilities), {
        "status": "ready_cache",
        "age_hands": age_hands,
        "elapsed_seconds": cached.get("elapsed_seconds"),
        "fingerprint": cached.get("fingerprint"),
    }


def _maybe_start_deepseek(
    key: str,
    observations: Sequence[Mapping[str, int]],
    particle_probs: Mapping[str, float],
    particle_diagnostics: Mapping[str, Any],
    user_id: str,
    venue: str,
    room: str,
    shoe_id: str,
) -> str:
    observation_count = len(observations)

    if not USE_DEEPSEEK or DEEPSEEK_WEIGHT <= 0.0:
        return "disabled"

    if observation_count < DEEPSEEK_MIN_OBSERVATIONS:
        return "not_enough_observations"

    with _DEEPSEEK_CACHE_LOCK:
        cached = _DEEPSEEK_CACHE.get(key) or {}
        future = _DEEPSEEK_FUTURES.get(key)

        if future is not None and not future.done():
            return "background_running"

        cooldown_until = int(
            cached.get("cooldown_until_observation", 0) or 0
        )
        if observation_count < cooldown_until:
            return "failure_cooldown"

        last_called = int(cached.get("observation_count", -10_000))
        if observation_count - last_called < DEEPSEEK_CALL_INTERVAL:
            return "interval_skip"

    fingerprint = _observations_fingerprint(observations)
    payload = _deepseek_payload(
        observations,
        particle_probs,
        particle_diagnostics,
        user_id,
        venue,
        room,
        shoe_id,
    )

    if DEEPSEEK_ASYNC_MODE:
        future = _DEEPSEEK_EXECUTOR.submit(
            _call_deepseek,
            payload,
            particle_probs,
        )
        with _DEEPSEEK_CACHE_LOCK:
            _DEEPSEEK_FUTURES[key] = future

        future.add_done_callback(
            lambda completed: _deepseek_future_callback(
                key,
                fingerprint,
                observation_count,
                completed,
            )
        )
        return "background_started"

    future = _DEEPSEEK_EXECUTOR.submit(
        _call_deepseek,
        payload,
        particle_probs,
    )
    try:
        result = future.result(timeout=DEEPSEEK_TIMEOUT_SECONDS)
    except TimeoutError:
        result = {
            "ok": False,
            "status": "timeout",
            "error": (
                f"DeepSeek exceeded {DEEPSEEK_TIMEOUT_SECONDS:.1f}s"
            ),
        }

    _store_deepseek_result(
        key,
        fingerprint,
        observation_count,
        result,
    )
    return str(result.get("status") or "completed")


def _fuse_particle_and_deepseek(
    particle_probs: Mapping[str, float],
    deepseek_probs: Optional[Mapping[str, float]],
) -> Tuple[Dict[str, float], float, bool]:
    particle = _normalize_triplet(particle_probs)
    if deepseek_probs is None or DEEPSEEK_WEIGHT <= 0.0:
        return particle, 0.0, False

    deepseek = _normalize_triplet(deepseek_probs)
    particle_side = max(("B", "P"), key=lambda name: particle[name])
    deepseek_side = max(("B", "P"), key=lambda name: deepseek[name])
    conflict = particle_side != deepseek_side

    if DEEPSEEK_CONFIRM_ONLY and conflict:
        return particle, 0.0, True

    weight = DEEPSEEK_WEIGHT
    if conflict:
        weight *= DEEPSEEK_CONFLICT_WEIGHT_SCALE

    fused = {
        name: particle[name] * (1.0 - weight)
        + deepseek[name] * weight
        for name in ("B", "P", "T")
    }
    return _normalize_triplet(fused), weight, conflict


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict(
    history: Union[str, Iterable[Any]],
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
    user_id: str = "",
) -> Dict[str, Any]:
    observations = _clean_observations(history)
    key = _key(user_id, venue, room, shoe_id)
    particle_filter = _get_filter(key)

    applied = _OBS_COUNTS.get(key, 0)
    if len(observations) < applied:
        reset_uid_model(user_id, venue, room, shoe_id)
        particle_filter = _get_filter(key)
        applied = 0

    updates: List[Dict[str, Any]] = []
    for observation in observations[applied:]:
        updates.append(
            particle_filter.update(
                observation["player"],
                observation["banker"],
            )
        )
    _OBS_COUNTS[key] = len(observations)

    raw_particle = particle_filter.predict()
    particle_probs = _cap_probs(raw_particle["probabilities"])

    cached_ai_probs, cache_info = _cached_deepseek(
        key,
        len(observations),
    )

    trigger_status = _maybe_start_deepseek(
        key=key,
        observations=observations,
        particle_probs=particle_probs,
        particle_diagnostics=raw_particle,
        user_id=user_id,
        venue=venue,
        room=room,
        shoe_id=shoe_id,
    )

    # In synchronous mode, retrieve the just-written cache immediately.
    if not DEEPSEEK_ASYNC_MODE:
        cached_ai_probs, cache_info = _cached_deepseek(
            key,
            len(observations),
        )

    fused_probs, effective_ai_weight, ai_conflict = (
        _fuse_particle_and_deepseek(
            particle_probs,
            cached_ai_probs,
        )
    )
    probs = _cap_probs(fused_probs)

    allowed = ("B", "P", "T") if RECOMMEND_TIE else ("B", "P")
    recommend = max(allowed, key=lambda name: probs[name])
    recommend_text = {
        "B": "莊",
        "P": "閒",
        "T": "和",
    }[recommend]

    bp_total = max(1e-12, probs["B"] + probs["P"])
    confidence = max(probs["B"], probs["P"]) / bp_total
    edge = abs(probs["B"] - probs["P"]) / bp_total
    signal = (
        "HIGH"
        if confidence >= 0.58
        else "MEDIUM"
        if confidence >= 0.54
        else "LOW"
    )

    ai_used = cached_ai_probs is not None and effective_ai_weight > 0.0
    ai_status = (
        "ready_cache"
        if ai_used
        else trigger_status
        if trigger_status not in {"interval_skip", "failure_cooldown"}
        else str(cache_info.get("status") or trigger_status)
    )

    reason = (
        f"CARD_PF；觀測={len(observations)}；"
        f"粒子ESS={raw_particle['effective_sample_size']:.1f}；"
        f"下一手模擬={raw_particle['simulations']}；"
        f"DeepSeek={ai_status}；"
        f"AI有效權重={effective_ai_weight:.3f}"
    )

    return {
        "ok": True,
        "engine": "CARD_POINT_PARTICLE_FILTER_DEEPSEEK",
        "user_id": user_id,
        "venue": venue,
        "room": room,
        "shoe_id": shoe_id,
        "round_no": len(observations) + 1,
        "history_len": len(observations),
        "banker_rate": round(probs["B"] * 100, 1),
        "player_rate": round(probs["P"] * 100, 1),
        "tie_rate": round(probs["T"] * 100, 1),
        "probabilities": probs,
        "particle_probabilities": particle_probs,
        "deepseek_probabilities": cached_ai_probs,
        "recommend": recommend,
        "recommend_text": recommend_text,
        "is_observe": False,
        "observe_reason": "",
        "confidence": round(confidence, 4),
        "confidence_pct": round(confidence * 100, 1),
        "decision_edge": round(edge, 6),
        "signal_level": signal,
        "reason": reason,
        "point_particle_filter": raw_particle,
        "applied_updates": updates,
        "ai_used": ai_used,
        "ai_status": ai_status,
        "ai_result": (
            {
                "cache": cache_info,
                "trigger_status": trigger_status,
                "conflict": ai_conflict,
                "effective_weight": effective_ai_weight,
            }
            if DEBUG_AI_RESULT
            else None
        ),
        "deepseek": {
            "enabled": USE_DEEPSEEK,
            "async_mode": DEEPSEEK_ASYNC_MODE,
            "status": ai_status,
            "trigger_status": trigger_status,
            "cache_status": cache_info.get("status"),
            "cache_age_hands": cache_info.get("age_hands"),
            "configured_weight": DEEPSEEK_WEIGHT,
            "effective_weight": effective_ai_weight,
            "conflict": ai_conflict,
            "import_error": _DEEPSEEK_IMPORT_ERROR,
        },
        "ml_trained": len(observations) > 0,
        "ml_samples": len(observations),
        "tf_available": False,
        "lstm_status": "disabled_point_pf",
        "global_lstm_status": "disabled_point_pf",
        "configured_weights": {
            "particle_filter": 1.0 - DEEPSEEK_WEIGHT,
            "deepseek": DEEPSEEK_WEIGHT,
        },
        "effective_weights": {
            "particle_filter": round(1.0 - effective_ai_weight, 6),
            "deepseek": round(effective_ai_weight, 6),
        },
        "debug": None,
    }


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
        "ok": True,
        "history_len": result["history_len"],
        "model": "CARD_POINT_PARTICLE_FILTER_DEEPSEEK",
        "ai_status": result["ai_status"],
    }


def reset_uid_model(
    user_id: str,
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
) -> Dict[str, Any]:
    key = _key(user_id, venue, room, shoe_id)

    with _FILTER_LOCK:
        removed = _FILTERS.pop(key, None) is not None
        _OBS_COUNTS.pop(key, None)

    with _DEEPSEEK_CACHE_LOCK:
        _DEEPSEEK_CACHE.pop(key, None)
        future = _DEEPSEEK_FUTURES.pop(key, None)
        if future is not None and not future.done():
            future.cancel()

    return {
        "ok": True,
        "removed": int(removed),
        "training_key": key,
    }


def clear_model_cache(
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    with _FILTER_LOCK, _DEEPSEEK_CACHE_LOCK:
        if not user_id:
            count = len(_FILTERS)
            _FILTERS.clear()
            _OBS_COUNTS.clear()
            _DEEPSEEK_CACHE.clear()

            for future in _DEEPSEEK_FUTURES.values():
                if not future.done():
                    future.cancel()
            _DEEPSEEK_FUTURES.clear()

            return {
                "ok": True,
                "removed": count,
            }

        prefix = f"{user_id}|"
        keys = [key for key in _FILTERS if key.startswith(prefix)]

        for key in keys:
            _FILTERS.pop(key, None)
            _OBS_COUNTS.pop(key, None)
            _DEEPSEEK_CACHE.pop(key, None)

            future = _DEEPSEEK_FUTURES.pop(key, None)
            if future is not None and not future.done():
                future.cancel()

        return {
            "ok": True,
            "removed": len(keys),
        }


def get_model_cache_info() -> Dict[str, Any]:
    with _DEEPSEEK_CACHE_LOCK:
        ai_cache = {
            key: {
                "status": value.get("status"),
                "ok": value.get("ok"),
                "observation_count": value.get("observation_count"),
                "failure_count": value.get("failure_count"),
            }
            for key, value in _DEEPSEEK_CACHE.items()
        }

    return {
        "size": len(_FILTERS),
        "keys": list(_FILTERS.keys()),
        "engine": "CARD_POINT_PARTICLE_FILTER_DEEPSEEK",
        "deepseek_enabled": USE_DEEPSEEK,
        "deepseek_async_mode": DEEPSEEK_ASYNC_MODE,
        "deepseek_cache": ai_cache,
    }
