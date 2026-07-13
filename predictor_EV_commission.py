"""Independent point-conditioned baccarat particle-filter predictor.

Core behavior
-------------
* Every prediction uses only the newest Player/Banker final-point observation.
* A brand-new set of particle filters is created for that observation.
* No particle weights, inferred shoe depletion, or card counts are carried from
  the previous hand into the next hand.
* A fresh particle population represents many possible two-card / three-card
  draw paths and many possible remaining shoes. Low-latency mode uses one such
  population per request instead of repeating the whole filter three times.
* Final entry decisions use payout-aware EV: Banker commission is deducted,
  Player payout is configurable, and Banker/Player wagers push on Tie.

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


# ---------------------------------------------------------------------------
# Pre-import low-latency caps
# ---------------------------------------------------------------------------
# particle_filter_points commonly reads these values at import time.  Cap the
# expensive defaults before importing it so an old Render environment such as
# PF_N=220 / PF_UPD_SIMS=70 does not keep every LINE reply unnecessarily slow.
_PREFLIGHT_LOW_LATENCY = os.getenv("PF_LOW_LATENCY_MODE", "true").strip().lower() in {
    "1", "true", "yes", "on"
}


def _cap_preimport_env(name: str, default: int, maximum: int, minimum: int = 1) -> None:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    os.environ[name] = str(max(minimum, min(maximum, value)))


if _PREFLIGHT_LOW_LATENCY:
    _cap_preimport_env("PF_N", 120, 120, 32)
    _cap_preimport_env("PF_UPD_SIMS", 32, 32, 4)
    _cap_preimport_env("PF_PRED_SIMS", 8, 8, 2)

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

# Low-latency mode is enabled by default.  It deliberately overrides an old
# PF_INDEPENDENT_REPLICAS=3 setting and runs one fresh filter per request.
LOW_LATENCY_MODE = _env_bool("PF_LOW_LATENCY_MODE", True)
REQUESTED_INDEPENDENT_REPLICAS = min(
    4,
    _env_int("PF_INDEPENDENT_REPLICAS", 1, 1),
)
INDEPENDENT_REPLICAS = (
    1 if LOW_LATENCY_MODE else REQUESTED_INDEPENDENT_REPLICAS
)
ESS_WEIGHTED_REPLICAS = _env_bool("PF_ESS_WEIGHTED_REPLICAS", True)
RETURN_DIAGNOSTICS = _env_bool("PF_RETURN_DIAGNOSTICS", False)

# Neutralise the ordinary baccarat Banker base-rate edge before deciding B/P.
# Raw simulator probabilities remain available in raw_* response fields.
BIAS_NEUTRALIZE = _env_bool("PF_BIAS_NEUTRALIZE", True)
BASE_BANKER_PROB = _env_float("PF_BASE_BANKER_PROB", 0.4586, 0.30, 0.60)
BASE_PLAYER_PROB = _env_float("PF_BASE_PLAYER_PROB", 0.4462, 0.30, 0.60)
MIN_DIRECTION_EDGE = _env_float("PF_MIN_DIRECTION_EDGE", 0.0015, 0.0, 0.10)
OBSERVE_ON_NEUTRAL = _env_bool("PF_OBSERVE_ON_NEUTRAL", True)

# Expected-value decision layer. Probabilities used here are the model's raw
# calibrated probabilities before display-only Banker base-rate neutralisation.
# Standard baccarat settlement is Banker win +0.95 after 5% commission,
# Player win +1.00, and Tie pushes Banker/Player bets.
EV_DECISION_ENABLED = _env_bool("PF_EV_DECISION_ENABLED", True)
BANKER_COMMISSION_RATE = _env_float(
    "PF_BANKER_COMMISSION_RATE",
    0.05,
    0.0,
    0.20,
)
PLAYER_COMMISSION_RATE = _env_float(
    "PF_PLAYER_COMMISSION_RATE",
    0.0,
    0.0,
    0.20,
)
TIE_PAYOUT = _env_float("PF_TIE_PAYOUT", 8.0, 1.0, 20.0)
MIN_ENTRY_EV = _env_float("PF_MIN_ENTRY_EV", 0.0, -0.20, 0.50)
MIN_EV_GAP = _env_float("PF_MIN_EV_GAP", 0.0010, 0.0, 0.20)
ALLOW_NEGATIVE_EV = _env_bool("PF_ALLOW_NEGATIVE_EV", False)

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
CONFIGURED_DEEPSEEK_ASYNC_MODE = _env_bool("DEEPSEEK_ASYNC_MODE", True)
# In low-latency mode an old DEEPSEEK_ASYNC_MODE=False value is ignored so the
# LINE webhook never waits for the external API.
DEEPSEEK_ASYNC_MODE = (
    True if LOW_LATENCY_MODE else CONFIGURED_DEEPSEEK_ASYNC_MODE
)
DEEPSEEK_GLOBAL_POINT_CACHE = _env_bool(
    "DEEPSEEK_GLOBAL_POINT_CACHE",
    True,
)
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
    max_workers=(1 if LOW_LATENCY_MODE else _env_int("DEEPSEEK_WORKERS", 2, 1))
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


def _bias_neutralize_bp(probs: Mapping[str, float]) -> Dict[str, float]:
    """Remove the normal Banker base-rate edge from the B/P decision score.

    A fresh full-shoe simulation naturally starts around B=45.86%, P=44.62%.
    Comparing those raw values directly makes weak/no-information cases choose
    Banker.  Dividing each side by its own baseline compares relative evidence
    instead: baseline-only evidence becomes an honest 50/50 B/P score.
    """

    normalized = _normalize_triplet(probs)
    if not BIAS_NEUTRALIZE:
        return normalized

    tie = max(0.0, min(0.30, normalized["T"]))
    non_tie_mass = max(0.0, 1.0 - tie)
    banker_lift = normalized["B"] / max(1e-12, BASE_BANKER_PROB)
    player_lift = normalized["P"] / max(1e-12, BASE_PLAYER_PROB)
    lift_total = banker_lift + player_lift

    if lift_total <= 1e-12:
        banker_share = 0.5
        player_share = 0.5
    else:
        banker_share = banker_lift / lift_total
        player_share = player_lift / lift_total

    return {
        "B": banker_share * non_tie_mass,
        "P": player_share * non_tie_mass,
        "T": tie,
    }


def _expected_values(probs: Mapping[str, float]) -> Dict[str, float]:
    """Return net expected profit per one-unit wager for each side.

    Banker/Player wagers push on Tie, so Tie contributes zero to their EV.
    A winning Banker wager returns net profit 1 - commission; a losing wager
    loses one unit. Tie EV uses the configured net-to-one payout.
    """

    normalized = _normalize_triplet(probs)
    banker_win_profit = 1.0 - BANKER_COMMISSION_RATE
    player_win_profit = 1.0 - PLAYER_COMMISSION_RATE

    return {
        "B": normalized["B"] * banker_win_profit - normalized["P"],
        "P": normalized["P"] * player_win_profit - normalized["B"],
        "T": normalized["T"] * TIE_PAYOUT
        - normalized["B"]
        - normalized["P"],
    }


def _ev_decision(
    probs: Mapping[str, float],
    allowed: Sequence[str],
) -> Dict[str, Any]:
    """Choose the highest-EV side and decide whether it is worth entering."""

    ev = _expected_values(probs)
    ranked = sorted(allowed, key=lambda side: ev[side], reverse=True)
    best_side = ranked[0]
    second_side = ranked[1] if len(ranked) > 1 else best_side
    best_ev = ev[best_side]
    second_ev = ev[second_side]
    ev_gap = best_ev - second_ev

    reasons: List[str] = []
    if not ALLOW_NEGATIVE_EV and best_ev <= MIN_ENTRY_EV:
        reasons.append(
            f"最高EV={best_ev * 100:.3f}%未高於進場門檻"
            f"{MIN_ENTRY_EV * 100:.3f}%"
        )
    if ev_gap < MIN_EV_GAP:
        reasons.append(
            f"最佳與次佳EV差={ev_gap * 100:.3f}%低於門檻"
            f"{MIN_EV_GAP * 100:.3f}%"
        )

    return {
        "side": best_side,
        "values": ev,
        "best_ev": best_ev,
        "second_best_ev": second_ev,
        "ev_gap": ev_gap,
        "is_observe": bool(reasons),
        "observe_reason": "；".join(reasons),
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
        seed_scope = "GLOBAL_POINT" if DEEPSEEK_GLOBAL_POINT_CACHE else session_key
        replica_key = (
            f"{seed_scope}|INDEPENDENT|{point_text}|"
            f"replica={replica_index}"
        )
        particle_filter = PointParticleFilter(replica_key)

        update_info: Optional[Dict[str, Any]] = None
        if observation is not None:
            raw_update = particle_filter.update(
                int(observation["player"]),
                int(observation["banker"]),
            )
            # Large update diagnostics are intentionally omitted in normal
            # operation because serialising them can noticeably delay LINE.
            if RETURN_DIAGNOSTICS:
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

        if RETURN_DIAGNOSTICS:
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
            "or previous-hand distribution. The ordinary baccarat Banker base "
            "advantage must not by itself be treated as directional evidence. "
            "Return probability calibration only; payout commission and wager "
            "EV are calculated locally by deterministic code after calibration. "
            "Return strict JSON only with numeric keys B, P, T that sum to 1. "
            "Do not provide prose or guaranteed claims."
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
        "settlement_rules": {
            "banker_commission_rate": BANKER_COMMISSION_RATE,
            "player_commission_rate": PLAYER_COMMISSION_RATE,
            "tie_payout_to_one": TIE_PAYOUT,
            "banker_player_tie_is_push": True,
        },
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
    # Independent point calibration is session-agnostic. Reusing the exact same
    # point cache avoids repeated DeepSeek calls for every UID/room.
    if DEEPSEEK_GLOBAL_POINT_CACHE:
        return f"GLOBAL|POINT={fingerprint}"
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
    started_at = time.monotonic()
    observations = _clean_observations(history)
    latest_observation = observations[-1] if observations else None
    history_len = len(observations)
    round_no = history_len + 1
    session_key = _session_key(user_id, venue, room, shoe_id)

    # Critical fix: no _get_filter(), no _OBS_COUNTS, no incremental update.
    # Every call starts fresh and updates only the newest point observation.
    particle_started_at = time.monotonic()
    raw_particle, updates = _fresh_particle_prediction(
        latest_observation,
        session_key,
    )
    particle_elapsed_ms = round(
        (time.monotonic() - particle_started_at) * 1000.0,
        2,
    )
    raw_particle_probs = _cap_probs(raw_particle["probabilities"])
    particle_probs = _cap_probs(_bias_neutralize_bp(raw_particle_probs))

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
            particle_probs=raw_particle_probs,
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

    fused_model_probs, effective_ai_weight, ai_conflict = (
        _fuse_particle_and_deepseek(raw_particle_probs, cached_ai_probs)
    )
    # Keep an uncapped normalized distribution for payout/commission EV.
    # Display caps and Banker base-rate neutralisation are presentation and
    # anti-bias layers; they must not be used to settle wager economics.
    ev_probabilities = _normalize_triplet(fused_model_probs)
    raw_fused_probs = _cap_probs(ev_probabilities)
    probs = _cap_probs(_bias_neutralize_bp(ev_probabilities))

    allowed = ("B", "P", "T") if RECOMMEND_TIE else ("B", "P")
    probability_recommend = max(allowed, key=lambda name: probs[name])

    if EV_DECISION_ENABLED:
        ev_decision = _ev_decision(ev_probabilities, allowed)
        recommend = str(ev_decision["side"])
        ev_values = dict(ev_decision["values"])
        best_ev = float(ev_decision["best_ev"])
        ev_gap = float(ev_decision["ev_gap"])
        is_observe = bool(ev_decision["is_observe"])
        observe_reason = str(ev_decision["observe_reason"])
    else:
        recommend = probability_recommend
        ev_values = _expected_values(ev_probabilities)
        best_ev = float(ev_values[recommend])
        sorted_ev = sorted(
            (ev_values[side] for side in allowed),
            reverse=True,
        )
        ev_gap = (
            sorted_ev[0] - sorted_ev[1]
            if len(sorted_ev) > 1
            else 0.0
        )
        bp_total_for_edge = max(1e-12, probs["B"] + probs["P"])
        neutral_edge = abs(probs["B"] - probs["P"]) / bp_total_for_edge
        is_observe = bool(
            OBSERVE_ON_NEUTRAL
            and recommend in {"B", "P"}
            and neutral_edge < MIN_DIRECTION_EDGE
        )
        observe_reason = (
            "莊閒相對證據不足，已扣除百家樂原始莊家基準優勢"
            if is_observe
            else ""
        )

    bp_total = max(1e-12, probs["B"] + probs["P"])
    confidence = max(probs["B"], probs["P"]) / bp_total
    edge = abs(probs["B"] - probs["P"]) / bp_total
    recommend_text = (
        "觀望"
        if is_observe
        else {"B": "莊", "P": "閒", "T": "和"}[recommend]
    )

    # Signal strength follows usable post-commission EV when enabled.
    if EV_DECISION_ENABLED:
        signal = (
            "HIGH"
            if best_ev >= 0.03
            else "MEDIUM"
            if best_ev >= 0.012
            else "LOW"
        )
    else:
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
        f"莊基準校正={'ON' if BIAS_NEUTRALIZE else 'OFF'}；"
        f"EV判斷={'ON' if EV_DECISION_ENABLED else 'OFF'}；"
        f"莊抽水={BANKER_COMMISSION_RATE * 100:.1f}%；"
        f"莊EV={ev_values['B'] * 100:.3f}%；"
        f"閒EV={ev_values['P'] * 100:.3f}%；"
        f"最佳EV={best_ev * 100:.3f}%；"
        f"PF耗時={particle_elapsed_ms:.2f}ms；"
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
        "raw_particle_probabilities": raw_particle_probs,
        "raw_fused_probabilities": raw_fused_probs,
        "ev_probabilities": ev_probabilities,
        "deepseek_probabilities": cached_ai_probs,
        "recommend": recommend,
        "probability_recommend": probability_recommend,
        "recommend_text": recommend_text,
        "is_observe": is_observe,
        "observe_reason": observe_reason,
        "confidence": round(confidence, 4),
        "confidence_pct": round(confidence * 100, 1),
        "decision_edge": round(edge, 6),
        "expected_value": round(best_ev, 8),
        "expected_value_pct": round(best_ev * 100.0, 4),
        "ev_gap": round(ev_gap, 8),
        "ev_gap_pct": round(ev_gap * 100.0, 4),
        "ev_values": {
            side: round(value, 8)
            for side, value in ev_values.items()
        },
        "ev_values_pct": {
            side: round(value * 100.0, 4)
            for side, value in ev_values.items()
        },
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
            "configured_async_mode": CONFIGURED_DEEPSEEK_ASYNC_MODE,
            "global_exact_point_cache": DEEPSEEK_GLOBAL_POINT_CACHE,
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
        "performance": {
            "low_latency_mode": LOW_LATENCY_MODE,
            "particle_elapsed_ms": particle_elapsed_ms,
            "total_elapsed_ms": round(
                (time.monotonic() - started_at) * 1000.0,
                2,
            ),
            "replicas": INDEPENDENT_REPLICAS,
            "diagnostics_returned": RETURN_DIAGNOSTICS,
            "preimport_caps": {
                "PF_N": os.getenv("PF_N"),
                "PF_UPD_SIMS": os.getenv("PF_UPD_SIMS"),
                "PF_PRED_SIMS": os.getenv("PF_PRED_SIMS"),
            },
        },
        "ev_decision": {
            "enabled": EV_DECISION_ENABLED,
            "probability_source": "raw_calibrated_before_display_bias_correction",
            "banker_commission_rate": BANKER_COMMISSION_RATE,
            "banker_win_net_payout": 1.0 - BANKER_COMMISSION_RATE,
            "player_commission_rate": PLAYER_COMMISSION_RATE,
            "player_win_net_payout": 1.0 - PLAYER_COMMISSION_RATE,
            "tie_payout_to_one": TIE_PAYOUT,
            "banker_player_tie_is_push": True,
            "minimum_entry_ev": MIN_ENTRY_EV,
            "minimum_ev_gap": MIN_EV_GAP,
            "allow_negative_ev": ALLOW_NEGATIVE_EV,
            "best_side": recommend,
            "best_ev": round(best_ev, 8),
            "ev_gap": round(ev_gap, 8),
            "values": {
                side: round(value, 8)
                for side, value in ev_values.items()
            },
        },
        "bias_correction": {
            "enabled": BIAS_NEUTRALIZE,
            "base_banker_probability": BASE_BANKER_PROB,
            "base_player_probability": BASE_PLAYER_PROB,
            "minimum_direction_edge": MIN_DIRECTION_EDGE,
            "observe_on_neutral": OBSERVE_ON_NEUTRAL,
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

    # Global exact-point calibration is not per-UID model state, so a UID reset
    # leaves it intact. There is still no persistent particle filter to reset.
    if DEEPSEEK_GLOBAL_POINT_CACHE:
        return {
            "ok": True,
            "removed": 0,
            "training_key": session_key,
            "independent_mode": True,
            "persistent_particle_filter_removed": False,
            "global_point_cache_preserved": True,
        }

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
        "low_latency_mode": LOW_LATENCY_MODE,
        "bias_neutralize": BIAS_NEUTRALIZE,
        "engine": "CARD_POINT_INDEPENDENT_PARTICLE_FILTER_DEEPSEEK",
        "deepseek_enabled": USE_DEEPSEEK,
        "deepseek_async_mode": DEEPSEEK_ASYNC_MODE,
        "deepseek_cache": ai_cache,
    }
