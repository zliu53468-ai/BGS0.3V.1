"""Independent point-conditioned baccarat particle-filter predictor.

Core behavior
-------------
* Every prediction uses only the newest Player/Banker final-point observation.
* A brand-new set of particle filters is created for that observation.
* No particle weights, inferred shoe depletion, or card counts are carried from
  the previous hand into the next hand.
* Multiple fresh replicas are used so the same point result can represent many
  possible two-card / three-card draw paths and many possible remaining shoes.

DeepSeek behavior
-----------------
* DeepSeek is retained as a low-weight calibration layer.
* Its payload contains only the newest point observation and the current fresh
  particle-filter distribution.
* Cached AI results are isolated by the exact point fingerprint; a result from
  another point pair can never be blended into the current prediction.

Public compatibility
--------------------
    predict(history_or_observations, venue='', room='', shoe_id='', user_id='')

Accepted observations
---------------------
    {'player': 6, 'banker': 5}
    P6B5
    閒6莊5
    6,5
    65          # first digit = Player, second digit = Banker
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

# PF_MAX_FILTERS is retained for environment compatibility. There are no
# persistent particle filters in independent mode; it now caps AI cache size.
MAX_FILTERS = _env_int("PF_MAX_FILTERS", 32, 1)
MAX_AI_CACHE_ENTRIES = max(16, MAX_FILTERS * 16)

# Each replica starts from a completely fresh PointParticleFilter. Different
# replica keys create independent random streams when the PF seeds from key.
INDEPENDENT_REPLICAS = min(
    8,
    _env_int("PF_INDEPENDENT_REPLICAS", 3, 1),
)
ESS_WEIGHTED_REPLICAS = _env_bool("PF_ESS_WEIGHTED_REPLICAS", True)

RECOMMEND_TIE = _env_bool("PF_RECOMMEND_TIE", False)
MAX_DISPLAY_CONFIDENCE = _env_float(
    "PF_MAX_DISPLAY_CONFIDENCE",
    0.64,
    0.50,
    0.80,
)

# DeepSeek remains deliberately low-weight.
USE_DEEPSEEK = _env_bool("USE_DEEPSEEK", True)
DEEPSEEK_WEIGHT = _env_float("DEEPSEEK_WEIGHT", 0.08, 0.0, 0.30)

# Independent mode has exactly one active observation, so the default must be 1.
CONFIGURED_DEEPSEEK_MIN_OBSERVATIONS = _env_int(
    "DEEPSEEK_MIN_OBSERVATIONS",
    1,
    0,
)
# A single isolated point is the complete input in this mode. Existing Render
# values such as 3 or 5 are safely clamped so DeepSeek is not disabled forever.
DEEPSEEK_MIN_OBSERVATIONS = min(1, CONFIGURED_DEEPSEEK_MIN_OBSERVATIONS)
DEEPSEEK_TIMEOUT_SECONDS = _env_float(
    "DEEPSEEK_TIMEOUT_SECONDS",
    2.5,
    0.5,
    15.0,
)
DEEPSEEK_ASYNC_MODE = _env_bool("DEEPSEEK_ASYNC_MODE", True)
DEEPSEEK_CACHE_MAX_AGE_SECONDS = _env_int(
    "DEEPSEEK_CACHE_MAX_AGE_SECONDS",
    3600,
    0,
)
DEEPSEEK_FAILURE_COOLDOWN_SECONDS = _env_int(
    "DEEPSEEK_FAILURE_COOLDOWN_SECONDS",
    30,
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
# Runtime state: DeepSeek only. Particle filters are never retained.
# ---------------------------------------------------------------------------

_DEEPSEEK_CLIENT: Optional[Any] = None
_DEEPSEEK_CLIENT_LOCK = threading.RLock()
_DEEPSEEK_EXECUTOR = ThreadPoolExecutor(
    max_workers=_env_int("DEEPSEEK_WORKERS", 2, 1)
)
_DEEPSEEK_CACHE_LOCK = threading.RLock()
_DEEPSEEK_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_DEEPSEEK_FUTURES: Dict[str, Future[Any]] = {}


# ---------------------------------------------------------------------------
# Point parsing and independent particle-filter helpers
# ---------------------------------------------------------------------------


def _session_key(user_id: str, venue: str, room: str, shoe_id: str) -> str:
    return "|".join(
        [
            user_id or "anonymous",
            venue or "global",
            room or "global",
            shoe_id or "global",
        ]
    )


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
        r"^\s*([0-9])([0-9])\s*$",
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

    for pattern in patterns[2:]:
        match = re.search(pattern, text)
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


def _normalize_triplet(values: Mapping[str, Any]) -> Dict[str, float]:
    numbers: Dict[str, float] = {}
    for name in ("B", "P", "T"):
        try:
            numbers[name] = max(0.0, float(values.get(name, 0.0) or 0.0))
        except (TypeError, ValueError):
            numbers[name] = 0.0

    total = sum(numbers.values())
    if total <= 0.0:
        return {"B": 0.4586, "P": 0.4462, "T": 0.0952}
    return {name: value / total for name, value in numbers.items()}


def _cap_probs(probs: Mapping[str, float]) -> Dict[str, float]:
    normalized = _normalize_triplet(probs)
    tie = max(0.0, min(0.30, normalized["T"]))
    banker = max(0.0, normalized["B"])
    player = max(0.0, normalized["P"])

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


def _observation_outcome(observation: Mapping[str, int]) -> str:
    player = int(observation["player"])
    banker = int(observation["banker"])
    return "B" if banker > player else "P" if player > banker else "T"


def _point_fingerprint(observation: Mapping[str, int]) -> str:
    compact = f"P{int(observation['player'])}B{int(observation['banker'])}"
    return hashlib.sha1(compact.encode("utf-8")).hexdigest()[:16]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _fresh_particle_prediction(
    observation: Optional[Mapping[str, int]],
    session_key: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run fresh independent PF replicas for only the current point pair."""

    weighted_totals = {"B": 0.0, "P": 0.0, "T": 0.0}
    total_weight = 0.0
    ess_values: List[float] = []
    total_simulations = 0
    replica_diagnostics: List[Dict[str, Any]] = []
    updates: List[Dict[str, Any]] = []

    point_text = (
        f"P{int(observation['player'])}B{int(observation['banker'])}"
        if observation is not None
        else "NO_POINT"
    )

    for replica_index in range(INDEPENDENT_REPLICAS):
        # A new object is intentionally created on every call and every replica.
        # Nothing is inserted into a global filter dictionary.
        replica_key = (
            f"{session_key}|INDEPENDENT|{point_text}|"
            f"replica={replica_index}"
        )
        particle_filter = PointParticleFilter(replica_key)

        update_info: Optional[Dict[str, Any]] = None
        if observation is not None:
            raw_update = particle_filter.update(
                int(observation["player"]),
                int(observation["banker"]),
            )
            update_info = (
                dict(raw_update)
                if isinstance(raw_update, Mapping)
                else {"result": raw_update}
            )
            updates.append(
                {
                    "replica": replica_index + 1,
                    "observation": dict(observation),
                    "update": update_info,
                }
            )

        raw_prediction = particle_filter.predict()
        if not isinstance(raw_prediction, Mapping):
            raw_prediction = {}

        probabilities = _normalize_triplet(
            raw_prediction.get("probabilities", {})
            if isinstance(raw_prediction.get("probabilities", {}), Mapping)
            else {}
        )
        ess = max(
            0.0,
            _safe_float(raw_prediction.get("effective_sample_size"), 0.0),
        )
        simulations = max(
            0,
            _safe_int(raw_prediction.get("simulations"), 0),
        )
        weight = max(1.0, ess) if ESS_WEIGHTED_REPLICAS else 1.0

        for side in ("B", "P", "T"):
            weighted_totals[side] += probabilities[side] * weight
        total_weight += weight
        ess_values.append(ess)
        total_simulations += simulations

        replica_item: Dict[str, Any] = {
            "replica": replica_index + 1,
            "probabilities": probabilities,
            "effective_sample_size": ess,
            "simulations": simulations,
        }
        if DEBUG_AI_RESULT:
            replica_item["update"] = update_info
        replica_diagnostics.append(replica_item)

    if total_weight <= 0.0:
        aggregate_probs = {"B": 0.4586, "P": 0.4462, "T": 0.0952}
    else:
        aggregate_probs = _normalize_triplet(
            {
                side: weighted_totals[side] / total_weight
                for side in ("B", "P", "T")
            }
        )

    average_ess = (
        sum(ess_values) / len(ess_values)
        if ess_values
        else 0.0
    )

    aggregate = {
        "probabilities": aggregate_probs,
        "effective_sample_size": average_ess,
        "simulations": total_simulations,
        "replica_count": INDEPENDENT_REPLICAS,
        "independent_mode": True,
        "accumulated_state": False,
        "conditioned_point": dict(observation) if observation is not None else None,
        "replicas": replica_diagnostics,
    }
    return aggregate, updates


# ---------------------------------------------------------------------------
# DeepSeek parsing and exact-point cache
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
        nested_mapping = _extract_json_mapping(mapping.get(key))
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
            return _normalize_triplet({"B": banker, "P": player, "T": tie})

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
        confidence = 0.55 if confidence is None else max(0.34, min(0.75, confidence))
        remainder = 1.0 - confidence
        others = [name for name in ("B", "P", "T") if name != recommendation]
        return _normalize_triplet(
            {
                recommendation: confidence,
                others[0]: remainder * 0.55,
                others[1]: remainder * 0.45,
            }
        )

    return None


def _deepseek_payload(
    observation: Mapping[str, int],
    particle_probs: Mapping[str, float],
    particle_diagnostics: Mapping[str, Any],
    user_id: str,
    venue: str,
    room: str,
    shoe_id: str,
) -> Dict[str, Any]:
    current = {
        "player": int(observation["player"]),
        "banker": int(observation["banker"]),
        "outcome": _observation_outcome(observation),
    }

    return {
        "task": "baccarat_independent_point_particle_filter_calibration",
        "instruction": (
            "Calibrate only this single current point observation and the fresh "
            "independent particle-filter distribution. Do not use, infer, or "
            "continue any previous-hand trend, streak, shoe state, card count, "
            "or cached distribution. Return strict JSON only with numeric keys "
            "B, P, T that sum to 1. Do not provide prose or guaranteed claims."
        ),
        "classes": ["B", "P", "T"],
        "user_id": user_id,
        "venue": venue,
        "room": room,
        "shoe_id": shoe_id,
        "independent_mode": True,
        "observation_count": 1,
        "current_point_observation": current,
        "particle_filter_probabilities": dict(particle_probs),
        "particle_effective_sample_size": particle_diagnostics.get(
            "effective_sample_size"
        ),
        "particle_simulations": particle_diagnostics.get("simulations"),
        "particle_replicas": particle_diagnostics.get("replica_count"),
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


def _cache_key(session_key: str, fingerprint: str) -> str:
    return f"{session_key}|POINT={fingerprint}"


def _store_deepseek_result(
    cache_key: str,
    fingerprint: str,
    result: Mapping[str, Any],
) -> None:
    with _DEEPSEEK_CACHE_LOCK:
        previous = _DEEPSEEK_CACHE.get(cache_key) or {}
        failure_count = (
            0
            if result.get("ok")
            else int(previous.get("failure_count", 0)) + 1
        )
        _DEEPSEEK_CACHE[cache_key] = {
            **dict(result),
            "fingerprint": fingerprint,
            "failure_count": failure_count,
            "updated_at": time.time(),
        }
        _DEEPSEEK_CACHE.move_to_end(cache_key)
        while len(_DEEPSEEK_CACHE) > MAX_AI_CACHE_ENTRIES:
            old_key, _ = _DEEPSEEK_CACHE.popitem(last=False)
            old_future = _DEEPSEEK_FUTURES.pop(old_key, None)
            if old_future is not None and not old_future.done():
                old_future.cancel()


def _deepseek_future_callback(
    cache_key: str,
    fingerprint: str,
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

    _store_deepseek_result(cache_key, fingerprint, result)
    with _DEEPSEEK_CACHE_LOCK:
        _DEEPSEEK_FUTURES.pop(cache_key, None)


def _cached_deepseek(
    cache_key: str,
    fingerprint: str,
) -> Tuple[Optional[Dict[str, float]], Dict[str, Any]]:
    with _DEEPSEEK_CACHE_LOCK:
        cached = dict(_DEEPSEEK_CACHE.get(cache_key) or {})
        if cache_key in _DEEPSEEK_CACHE:
            _DEEPSEEK_CACHE.move_to_end(cache_key)

    if not cached:
        return None, {"status": "cache_empty"}

    if cached.get("fingerprint") != fingerprint:
        return None, {"status": "cache_fingerprint_mismatch"}

    age_seconds = max(0.0, time.time() - _safe_float(cached.get("updated_at")))
    if (
        DEEPSEEK_CACHE_MAX_AGE_SECONDS > 0
        and age_seconds > DEEPSEEK_CACHE_MAX_AGE_SECONDS
    ):
        return None, {
            "status": "cache_stale",
            "age_seconds": round(age_seconds, 3),
        }

    if not cached.get("ok"):
        return None, {
            "status": str(cached.get("status") or "cached_error"),
            "age_seconds": round(age_seconds, 3),
            "error": cached.get("error"),
        }

    probabilities = cached.get("probabilities")
    if not isinstance(probabilities, Mapping):
        return None, {
            "status": "cache_invalid",
            "age_seconds": round(age_seconds, 3),
        }

    return _normalize_triplet(probabilities), {
        "status": "ready_cache_exact_point",
        "age_seconds": round(age_seconds, 3),
        "elapsed_seconds": cached.get("elapsed_seconds"),
        "fingerprint": cached.get("fingerprint"),
    }


def _maybe_start_deepseek(
    cache_key: str,
    fingerprint: str,
    observation: Optional[Mapping[str, int]],
    particle_probs: Mapping[str, float],
    particle_diagnostics: Mapping[str, Any],
    user_id: str,
    venue: str,
    room: str,
    shoe_id: str,
) -> str:
    if not USE_DEEPSEEK or DEEPSEEK_WEIGHT <= 0.0:
        return "disabled"
    if observation is None:
        return "no_current_observation"
    with _DEEPSEEK_CACHE_LOCK:
        cached = dict(_DEEPSEEK_CACHE.get(cache_key) or {})
        future = _DEEPSEEK_FUTURES.get(cache_key)

    if future is not None and not future.done():
        return "background_running_exact_point"

    now = time.time()
    age_seconds = max(0.0, now - _safe_float(cached.get("updated_at"), now))

    if cached.get("ok") and cached.get("fingerprint") == fingerprint:
        if (
            DEEPSEEK_CACHE_MAX_AGE_SECONDS == 0
            or age_seconds <= DEEPSEEK_CACHE_MAX_AGE_SECONDS
        ):
            return "cache_fresh_exact_point"

    if (
        cached
        and not cached.get("ok")
        and cached.get("fingerprint") == fingerprint
        and age_seconds < DEEPSEEK_FAILURE_COOLDOWN_SECONDS
    ):
        return "failure_cooldown_exact_point"

    payload = _deepseek_payload(
        observation,
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
            _DEEPSEEK_FUTURES[cache_key] = future
        future.add_done_callback(
            lambda completed: _deepseek_future_callback(
                cache_key,
                fingerprint,
                completed,
            )
        )
        return "background_started_exact_point"

    future = _DEEPSEEK_EXECUTOR.submit(
        _call_deepseek,
        payload,
        particle_probs,
    )
    try:
        result = future.result(timeout=DEEPSEEK_TIMEOUT_SECONDS)
    except TimeoutError:
        future.cancel()
        result = {
            "ok": False,
            "status": "timeout",
            "error": f"DeepSeek exceeded {DEEPSEEK_TIMEOUT_SECONDS:.1f}s",
        }

    _store_deepseek_result(cache_key, fingerprint, result)
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
        name: particle[name] * (1.0 - weight) + deepseek[name] * weight
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
    latest_observation = observations[-1] if observations else None
    history_len = len(observations)
    round_no = history_len + 1
    session_key = _session_key(user_id, venue, room, shoe_id)

    # Critical fix: no _get_filter(), no _OBS_COUNTS, no incremental update.
    # Every call starts fresh and updates only the newest point observation.
    raw_particle, updates = _fresh_particle_prediction(
        latest_observation,
        session_key,
    )
    particle_probs = _cap_probs(raw_particle["probabilities"])

    if latest_observation is not None:
        fingerprint = _point_fingerprint(latest_observation)
        ai_cache_key = _cache_key(session_key, fingerprint)
        cached_ai_probs, cache_info = _cached_deepseek(
            ai_cache_key,
            fingerprint,
        )
        trigger_status = _maybe_start_deepseek(
            cache_key=ai_cache_key,
            fingerprint=fingerprint,
            observation=latest_observation,
            particle_probs=particle_probs,
            particle_diagnostics=raw_particle,
            user_id=user_id,
            venue=venue,
            room=room,
            shoe_id=shoe_id,
        )

        if not DEEPSEEK_ASYNC_MODE:
            cached_ai_probs, cache_info = _cached_deepseek(
                ai_cache_key,
                fingerprint,
            )
    else:
        fingerprint = ""
        ai_cache_key = ""
        cached_ai_probs = None
        cache_info = {"status": "no_current_observation"}
        trigger_status = "no_current_observation"

    fused_probs, effective_ai_weight, ai_conflict = (
        _fuse_particle_and_deepseek(particle_probs, cached_ai_probs)
    )
    probs = _cap_probs(fused_probs)

    allowed = ("B", "P", "T") if RECOMMEND_TIE else ("B", "P")
    recommend = max(allowed, key=lambda name: probs[name])
    recommend_text = {"B": "莊", "P": "閒", "T": "和"}[recommend]

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
    if ai_used:
        ai_status = "ready_cache_exact_point"
    elif trigger_status in {
        "cache_fresh_exact_point",
        "failure_cooldown_exact_point",
    }:
        ai_status = str(cache_info.get("status") or trigger_status)
    else:
        ai_status = trigger_status

    current_point_text = (
        f"閒{latest_observation['player']}莊{latest_observation['banker']}"
        if latest_observation is not None
        else "無"
    )
    reason = (
        f"INDEPENDENT_CARD_PF；本次只使用={current_point_text}；"
        f"不沿用前手牌靴；獨立粒子組={raw_particle['replica_count']}；"
        f"平均ESS={raw_particle['effective_sample_size']:.1f}；"
        f"本次模擬={raw_particle['simulations']}；"
        f"DeepSeek={ai_status}；AI有效權重={effective_ai_weight:.3f}"
    )

    return {
        "ok": True,
        "engine": "CARD_POINT_INDEPENDENT_PARTICLE_FILTER_DEEPSEEK",
        "user_id": user_id,
        "venue": venue,
        "room": room,
        "shoe_id": shoe_id,
        "round_no": round_no,
        "history_len": history_len,
        "independent_mode": True,
        "accumulated_state": False,
        "conditioning_observation": (
            dict(latest_observation)
            if latest_observation is not None
            else None
        ),
        "ignored_prior_observations": max(0, history_len - 1),
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
                "cache_key": ai_cache_key,
                "fingerprint": fingerprint,
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
            "independent_point_cache": True,
            "status": ai_status,
            "trigger_status": trigger_status,
            "cache_status": cache_info.get("status"),
            "cache_age_seconds": cache_info.get("age_seconds"),
            "configured_weight": DEEPSEEK_WEIGHT,
            "configured_min_observations": CONFIGURED_DEEPSEEK_MIN_OBSERVATIONS,
            "effective_min_observations": DEEPSEEK_MIN_OBSERVATIONS,
            "effective_weight": effective_ai_weight,
            "conflict": ai_conflict,
            "import_error": _DEEPSEEK_IMPORT_ERROR,
        },
        "ml_trained": latest_observation is not None,
        "ml_samples": 1 if latest_observation is not None else 0,
        "tf_available": False,
        "lstm_status": "disabled_independent_point_pf",
        "global_lstm_status": "disabled_independent_point_pf",
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
    # Kept for app.py compatibility. Independent mode never trains on history;
    # it evaluates only the latest point as one isolated observation.
    result = predict(history, venue, room, shoe_id, user_id)
    return {
        "ok": True,
        "history_len": result["history_len"],
        "independent_samples": result["ml_samples"],
        "model": "CARD_POINT_INDEPENDENT_PARTICLE_FILTER_DEEPSEEK",
        "ai_status": result["ai_status"],
    }


def reset_uid_model(
    user_id: str,
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
) -> Dict[str, Any]:
    session_key = _session_key(user_id, venue, room, shoe_id)
    prefix = f"{session_key}|POINT="

    removed = 0
    with _DEEPSEEK_CACHE_LOCK:
        keys = [key for key in _DEEPSEEK_CACHE if key.startswith(prefix)]
        future_keys = [key for key in _DEEPSEEK_FUTURES if key.startswith(prefix)]

        for key in keys:
            _DEEPSEEK_CACHE.pop(key, None)
            removed += 1
        for key in future_keys:
            future = _DEEPSEEK_FUTURES.pop(key, None)
            if future is not None and not future.done():
                future.cancel()

    return {
        "ok": True,
        "removed": removed,
        "training_key": session_key,
        "independent_mode": True,
        "persistent_particle_filter_removed": False,
    }


def clear_model_cache(user_id: Optional[str] = None) -> Dict[str, Any]:
    with _DEEPSEEK_CACHE_LOCK:
        if not user_id:
            count = len(_DEEPSEEK_CACHE)
            _DEEPSEEK_CACHE.clear()
            for future in _DEEPSEEK_FUTURES.values():
                if not future.done():
                    future.cancel()
            _DEEPSEEK_FUTURES.clear()
            return {
                "ok": True,
                "removed": count,
                "independent_mode": True,
            }

        prefix = f"{user_id}|"
        keys = [key for key in _DEEPSEEK_CACHE if key.startswith(prefix)]
        future_keys = [key for key in _DEEPSEEK_FUTURES if key.startswith(prefix)]

        for key in keys:
            _DEEPSEEK_CACHE.pop(key, None)
        for key in future_keys:
            future = _DEEPSEEK_FUTURES.pop(key, None)
            if future is not None and not future.done():
                future.cancel()

        return {
            "ok": True,
            "removed": len(keys),
            "independent_mode": True,
        }


def get_model_cache_info() -> Dict[str, Any]:
    with _DEEPSEEK_CACHE_LOCK:
        ai_cache = {
            key: {
                "status": value.get("status"),
                "ok": value.get("ok"),
                "fingerprint": value.get("fingerprint"),
                "failure_count": value.get("failure_count"),
            }
            for key, value in _DEEPSEEK_CACHE.items()
        }

    return {
        "size": 0,
        "keys": [],
        "persistent_particle_filters": 0,
        "independent_replicas_per_prediction": INDEPENDENT_REPLICAS,
        "engine": "CARD_POINT_INDEPENDENT_PARTICLE_FILTER_DEEPSEEK",
        "deepseek_enabled": USE_DEEPSEEK,
        "deepseek_async_mode": DEEPSEEK_ASYNC_MODE,
        "deepseek_cache": ai_cache,
    }
