"""Baccarat predictor: B/P LSTM + 10M pattern DB + cross-shoe memory.

Compatibility goals
-------------------
* Input remains a Big Road sequence containing B / P / T.
* ``predict`` and ``fit_history`` signatures are unchanged.
* Per-UID / venue / room / shoe isolation is retained.
* Common response keys used by older app.py versions are preserved.
* The active statistical model is LSTM only.
* GRU, TCN/1D-CNN, GBM, and Markov remain disabled.
* DeepSeek remains available as a low-weight calibration/confirmation layer.

Model changes
-------------
* The LSTM target is binary B/P. A tie is retained in the input features but
  is skipped as a training target, so rare ties no longer compress the B/P
  decision boundary.
* A cross-shoe base LSTM is maintained per UID + venue + room while the Python
  process remains alive. Finished shoes are archived into this memory.
* A local shoe LSTM is initialized from the base LSTM when possible and then
  updated with a replay buffer rather than only the latest hand.
* Early, middle, and late shoe phases use different base/local weights and
  different entry/uncertainty thresholds.
* A short-vs-long road regime-change detector can pause recommendations for a
  small cooldown window after a material rhythm transition.
* Monte Carlo Dropout estimates uncertainty for both base and local LSTMs.

Important limitation
--------------------
B/P/T history alone cannot guarantee a long-run predictive edge or a stable
70 percent next-hand hit rate in a fair baccarat game. This module is designed
for stricter walk-forward evaluation and uncertainty filtering, not guaranteed
betting returns.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

# Render / CPU stability settings must be set before importing TensorFlow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np

try:
    from pattern_database_shape import PatternDatabase  # type: ignore
except Exception:
    PatternDatabase = None  # type: ignore

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: Optional[int] = None) -> int:
    try:
        value = int(os.getenv(name, str(default)).strip())
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _env_float(
    name: str,
    default: float,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    try:
        value = float(os.getenv(name, str(default)).strip())
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


# ---------------------------------------------------------------------------
# Core settings
# ---------------------------------------------------------------------------

PREDICT_ENGINE = os.getenv(
    "PREDICT_ENGINE", "LSTM_SHAPE_PATTERN_DB_MC_DEEPSEEK"
).strip().upper()
PER_UID_MODELS = _env_bool("PER_UID_MODELS", True)
REQUIRE_USER_ID = _env_bool("REQUIRE_USER_ID", False)
MAX_UID_MODELS = _env_int("MAX_UID_MODELS", 16, minimum=1)
MAX_SCOPE_MEMORIES = _env_int("MAX_SCOPE_MEMORIES", 16, minimum=1)

# Active local/base model: LSTM only.
USE_LSTM = True
LSTM_SEQUENCE_LENGTH = _env_int("LSTM_SEQUENCE_LENGTH", 16, minimum=4)
LSTM_UNITS = _env_int("LSTM_UNITS", 24, minimum=8)
LSTM_SECOND_UNITS = _env_int("LSTM_SECOND_UNITS", 12, minimum=4)
LSTM_DENSE_UNITS = _env_int("LSTM_DENSE_UNITS", 12, minimum=3)
LSTM_DROPOUT = _env_float("LSTM_DROPOUT", 0.40, 0.05, 0.80)
LSTM_LEARNING_RATE = _env_float("LSTM_LEARNING_RATE", 0.0002, 0.000001)
LSTM_EPOCHS = _env_int("LSTM_EPOCHS", 4, minimum=1)
LSTM_ONLINE_EPOCHS = _env_int("LSTM_ONLINE_EPOCHS", 1, minimum=1)
LSTM_MIN_SAMPLES = _env_int("LSTM_MIN_SAMPLES", 10, minimum=2)
LSTM_RETRAIN_INTERVAL = _env_int("LSTM_RETRAIN_INTERVAL", 3, minimum=1)
LOCAL_FREEZE_FIRST_LSTM = _env_bool("LOCAL_FREEZE_FIRST_LSTM", True)

# Replay buffer. Targets are B/P only; T targets are skipped.
REPLAY_RECENT_WINDOW = _env_int("REPLAY_RECENT_WINDOW", 24, minimum=4)
REPLAY_BATCH_SAMPLES = _env_int("REPLAY_BATCH_SAMPLES", 32, minimum=4)
REPLAY_RECENT_RATIO = _env_float("REPLAY_RECENT_RATIO", 0.45, 0.0, 1.0)
REPLAY_RECENCY_DECAY = _env_float("REPLAY_RECENCY_DECAY", 0.995, 0.80, 1.0)

# External large-scale pattern database. The database contains aggregated
# suffix/context counts built from the user's real historical shoe dataset.
USE_PATTERN_DATABASE = _env_bool("USE_PATTERN_DATABASE", True)
PATTERN_DB_PATH = os.getenv("PATTERN_DB_PATH", "pattern_10m.sqlite3").strip()
PATTERN_DB_WEIGHT = _env_float("PATTERN_DB_WEIGHT", 0.25, 0.0, 1.0)
PATTERN_DB_MAX_ORDER = _env_int("PATTERN_DB_MAX_ORDER", 24, minimum=1)
PATTERN_DB_MIN_MATCHES = _env_int("PATTERN_DB_MIN_MATCHES", 12, minimum=1)
PATTERN_DB_SMOOTHING = _env_float("PATTERN_DB_SMOOTHING", 4.0, 0.0, 1000.0)
PATTERN_DB_FIRST_HAND_ENABLED = _env_bool("PATTERN_DB_FIRST_HAND_ENABLED", True)

# Recommendations are always returned. Entry thresholds and regime/AI conflict
# checks remain diagnostic only and no longer produce an observe result.
DISABLE_OBSERVE = _env_bool("DISABLE_OBSERVE", True)

# Cross-shoe base memory. It is process-memory based; a Render restart clears it.
GLOBAL_MEMORY_ENABLED = _env_bool("GLOBAL_MEMORY_ENABLED", True)
GLOBAL_MEMORY_MIN_SAMPLES = _env_int("GLOBAL_MEMORY_MIN_SAMPLES", 24, minimum=4)
GLOBAL_MEMORY_MAX_SHOES = _env_int("GLOBAL_MEMORY_MAX_SHOES", 40, minimum=1)
GLOBAL_MEMORY_MAX_SAMPLES = _env_int("GLOBAL_MEMORY_MAX_SAMPLES", 1800, minimum=32)
GLOBAL_MEMORY_EPOCHS = _env_int("GLOBAL_MEMORY_EPOCHS", 3, minimum=1)
GLOBAL_MEMORY_ONLINE_EPOCHS = _env_int(
    "GLOBAL_MEMORY_ONLINE_EPOCHS", 1, minimum=1
)
GLOBAL_MEMORY_BATCH_SIZE = _env_int("GLOBAL_MEMORY_BATCH_SIZE", 16, minimum=1)

# Monte Carlo Dropout.
LSTM_MC_SIMULATIONS = _env_int("LSTM_MC_SIMULATIONS", 64, minimum=8)
GLOBAL_MC_SIMULATIONS = _env_int("GLOBAL_MC_SIMULATIONS", 32, minimum=8)
LSTM_MC_BATCH_SIZE = _env_int("LSTM_MC_BATCH_SIZE", 64, minimum=1)
LSTM_MC_TEMPERATURE = _env_float(
    "LSTM_MC_TEMPERATURE", 1.10, minimum=0.25, maximum=3.0
)
LSTM_MC_VOTE_BLEND = _env_float(
    "LSTM_MC_VOTE_BLEND", 0.0, minimum=0.0, maximum=0.50
)
LSTM_MC_UNCERTAINTY_SCALE = _env_float(
    "LSTM_MC_UNCERTAINTY_SCALE", 3.0, minimum=0.0, maximum=20.0
)
LSTM_MC_MIN_RELIABILITY = _env_float(
    "LSTM_MC_MIN_RELIABILITY", 0.35, minimum=0.0, maximum=1.0
)

# Training controls.
NEURAL_BATCH_SIZE = _env_int("NEURAL_BATCH_SIZE", 4, minimum=1)
NEURAL_VALIDATION_MIN_SAMPLES = _env_int(
    "NEURAL_VALIDATION_MIN_SAMPLES", 48, minimum=4
)
NEURAL_EARLY_STOP_PATIENCE = _env_int(
    "NEURAL_EARLY_STOP_PATIENCE", 2, minimum=0
)
NEURAL_CLASS_WEIGHT = _env_bool("NEURAL_CLASS_WEIGHT", False)
NEURAL_CLASS_WEIGHT_MAX = _env_float("NEURAL_CLASS_WEIGHT_MAX", 1.50, 1.0)
NEURAL_VERBOSE = _env_int("NEURAL_VERBOSE", 0, minimum=0)

# Small-sample B/P calibration.
MODEL_CALIBRATION_STRENGTH = _env_float(
    "MODEL_CALIBRATION_STRENGTH", 40.0, 0.0
)
COMPONENT_MAX_PROB = _env_float("COMPONENT_MAX_PROB", 0.68, 0.50, 0.95)
FINAL_MAX_PROB = _env_float("FINAL_MAX_PROB", 0.65, 0.50, 0.95)

# Tie is estimated separately for display. It is not an LSTM output class.
B_PRIOR = _env_float("B_PRIOR", 0.4586, 0.0001)
P_PRIOR = _env_float("P_PRIOR", 0.4462, 0.0001)
T_PRIOR = _env_float("T_PRIOR", 0.0952, 0.0001)
TIE_PRIOR_STRENGTH = _env_float("TIE_PRIOR_STRENGTH", 24.0, 0.0)
TIE_RATE_MIN = _env_float("TIE_RATE_MIN", 0.06, 0.0, 0.30)
TIE_RATE_MAX = _env_float("TIE_RATE_MAX", 0.14, 0.0, 0.40)
FALLBACK_PRIOR_STRENGTH = _env_float(
    "FALLBACK_PRIOR_STRENGTH",
    _env_float("LSTM_FALLBACK_PRIOR_STRENGTH", 18.0, 0.0),
    0.0,
)

# Shoe phases.
EARLY_PHASE_END = _env_int("EARLY_PHASE_END", 12, minimum=4)
MID_PHASE_END = _env_int("MID_PHASE_END", 36, minimum=EARLY_PHASE_END + 1)

# Phase model weights. They are renormalized when a component is unavailable.
EARLY_GLOBAL_WEIGHT = _env_float("EARLY_GLOBAL_WEIGHT", 0.85, 0.0, 1.0)
EARLY_LOCAL_WEIGHT = _env_float("EARLY_LOCAL_WEIGHT", 0.10, 0.0, 1.0)
EARLY_DEEPSEEK_WEIGHT = _env_float("EARLY_DEEPSEEK_WEIGHT", 0.05, 0.0, 1.0)
MID_GLOBAL_WEIGHT = _env_float("MID_GLOBAL_WEIGHT", 0.40, 0.0, 1.0)
MID_LOCAL_WEIGHT = _env_float("MID_LOCAL_WEIGHT", 0.55, 0.0, 1.0)
MID_DEEPSEEK_WEIGHT = _env_float("MID_DEEPSEEK_WEIGHT", 0.05, 0.0, 1.0)
LATE_GLOBAL_WEIGHT = _env_float("LATE_GLOBAL_WEIGHT", 0.35, 0.0, 1.0)
LATE_LOCAL_WEIGHT = _env_float("LATE_LOCAL_WEIGHT", 0.60, 0.0, 1.0)
LATE_DEEPSEEK_WEIGHT = _env_float("LATE_DEEPSEEK_WEIGHT", 0.05, 0.0, 1.0)

# Legacy weights remain supported. LSTM_WEIGHT scales base + local together;
# DEEPSEEK_WEIGHT can disable or cap the phase DeepSeek contribution.
LSTM_WEIGHT = _env_float("LSTM_WEIGHT", 0.72, 0.0)
DEEPSEEK_WEIGHT = _env_float("DEEPSEEK_WEIGHT", 0.03, 0.0)
GRU_WEIGHT = 0.0
TCN_WEIGHT = 0.0
GBM_WEIGHT = 0.0

# Phase-specific entry thresholds.
EARLY_MIN_ENTRY_EDGE = _env_float("EARLY_MIN_ENTRY_EDGE", 0.055, 0.0, 0.50)
MID_MIN_ENTRY_EDGE = _env_float("MID_MIN_ENTRY_EDGE", 0.035, 0.0, 0.50)
LATE_MIN_ENTRY_EDGE = _env_float("LATE_MIN_ENTRY_EDGE", 0.045, 0.0, 0.50)
EARLY_MIN_BP_CONFIDENCE = _env_float(
    "EARLY_MIN_BP_CONFIDENCE", 0.56, 0.50, 1.0
)
MID_MIN_BP_CONFIDENCE = _env_float(
    "MID_MIN_BP_CONFIDENCE", 0.54, 0.50, 1.0
)
LATE_MIN_BP_CONFIDENCE = _env_float(
    "LATE_MIN_BP_CONFIDENCE", 0.55, 0.50, 1.0
)
EARLY_MC_MAX_UNCERTAINTY = _env_float(
    "EARLY_MC_MAX_UNCERTAINTY", 0.030, 0.0, 1.0
)
MID_MC_MAX_UNCERTAINTY = _env_float(
    "MID_MC_MAX_UNCERTAINTY", 0.040, 0.0, 1.0
)
LATE_MC_MAX_UNCERTAINTY = _env_float(
    "LATE_MC_MAX_UNCERTAINTY", 0.030, 0.0, 1.0
)
REQUIRE_MODEL_READY_FOR_ENTRY = _env_bool(
    "REQUIRE_MODEL_READY_FOR_ENTRY", True
)

# Regime-change detector.
REGIME_SHORT_WINDOW = _env_int("REGIME_SHORT_WINDOW", 6, minimum=3)
REGIME_LONG_WINDOW = _env_int(
    "REGIME_LONG_WINDOW", 18, minimum=REGIME_SHORT_WINDOW + 2
)
REGIME_CHANGE_THRESHOLD = _env_float(
    "REGIME_CHANGE_THRESHOLD", 0.18, 0.01, 1.0
)
REGIME_CHANGE_COOLDOWN_HANDS = _env_int(
    "REGIME_CHANGE_COOLDOWN_HANDS", 2, minimum=0
)
REGIME_CROSSING_RELEASE_RATIO = _env_float(
    "REGIME_CROSSING_RELEASE_RATIO", 0.90, 0.10, 1.0
)

# DeepSeek remains, but mainly confirms the LSTM direction.
USE_DEEPSEEK = _env_bool("USE_DEEPSEEK", True)
DEEPSEEK_MIN_HISTORY = _env_int("DEEPSEEK_MIN_HISTORY", 12, minimum=0)
DEEPSEEK_TIMEOUT_SECONDS = _env_float(
    "DEEPSEEK_TIMEOUT_SECONDS", 8.0, minimum=1.0
)
DEEPSEEK_CONFLICT_OBSERVE = _env_bool("DEEPSEEK_CONFLICT_OBSERVE", True)
DEEPSEEK_CONFLICT_OVERRIDE_EDGE = _env_float(
    "DEEPSEEK_CONFLICT_OVERRIDE_EDGE", 0.06, 0.0, 0.50
)

# Removed components retained as compatibility constants.
USE_GRU = False
USE_TCN = False
USE_GBM = False
GRU_SEQUENCE_LENGTH = _env_int("GRU_SEQUENCE_LENGTH", 16, minimum=4)
GRU_UNITS = _env_int("GRU_UNITS", 32, minimum=4)
TCN_SEQUENCE_LENGTH = _env_int("TCN_SEQUENCE_LENGTH", 20, minimum=4)
TCN_FILTERS = _env_int("TCN_FILTERS", 24, minimum=4)
GBM_BACKEND = "DISABLED_LSTM_ONLY"

RANDOM_SEED = _env_int("RANDOM_SEED", 42)
DEBUG_PREDICTOR = _env_bool("DEBUG_PREDICTOR", False)
DEBUG_AI_RESULT = _env_bool("DEBUG_AI_RESULT", False)

CLASS_NAMES: Tuple[str, str, str] = ("B", "P", "T")
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {idx: name for name, idx in CLASS_TO_INDEX.items()}
BP_NAMES: Tuple[str, str] = ("B", "P")
BP_TO_INDEX = {name: idx for idx, name in enumerate(BP_NAMES)}

# 3 result one-hot + run + switch + ABA + pair block + short/medium balance
# + tie rate + local switch rates + lifecycle progress.
LSTM_FEATURE_DIM = 14

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Optional TensorFlow import
# ---------------------------------------------------------------------------

TF_AVAILABLE = False
TF_IMPORT_ERROR = ""
tf = None
Sequential = None
Input = None
Masking = None
LSTM = None
Dense = None
Dropout = None
Adam = None
EarlyStopping = None

try:
    import tensorflow as tf  # type: ignore[assignment]
    from tensorflow.keras.callbacks import EarlyStopping  # type: ignore[assignment]
    from tensorflow.keras.layers import (  # type: ignore[assignment]
        Dense,
        Dropout,
        Input,
        LSTM,
        Masking,
    )
    from tensorflow.keras.models import Sequential  # type: ignore[assignment]
    from tensorflow.keras.optimizers import Adam  # type: ignore[assignment]

    TF_AVAILABLE = True
    tf.random.set_seed(RANDOM_SEED)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass
except Exception as exc:  # pragma: no cover - deployment dependent
    TF_IMPORT_ERROR = str(exc)
    logger.warning("TensorFlow unavailable; LSTM uses fallback probabilities: %s", exc)


# ---------------------------------------------------------------------------
# Disabled backend compatibility
# ---------------------------------------------------------------------------

LIGHTGBM_AVAILABLE = False
XGBOOST_AVAILABLE = False
SKLEARN_GBM_AVAILABLE = False
GBM_IMPORT_ERRORS: Dict[str, str] = {
    "disabled": "GRU/TCN/GBM removed; B/P LSTM-only mode"
}
LGBMClassifier = None
XGBClassifier = None
HistGradientBoostingClassifier = None


# ---------------------------------------------------------------------------
# Optional DeepSeek client import
# ---------------------------------------------------------------------------

_DEEPSEEK_IMPORT_ERROR = ""
try:
    from deepseek_client import DeepSeekClient  # type: ignore
except Exception as exc:  # pragma: no cover - project dependent
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
# Generic helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _normalize(
    values: Sequence[float], fallback: Optional[Sequence[float]] = None
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total <= 0.0:
        arr = np.asarray(
            fallback if fallback is not None else np.ones(len(arr)),
            dtype=np.float64,
        )
        arr = np.maximum(arr, 0.0)
        total = float(arr.sum())
    if total <= 0.0:
        return np.ones(len(arr), dtype=np.float64) / max(1, len(arr))
    return arr / total


def _bp_prior_probs() -> np.ndarray:
    return _normalize([B_PRIOR, P_PRIOR], fallback=[1.0, 1.0])


def _prior_probs() -> np.ndarray:
    return _normalize([B_PRIOR, P_PRIOR, T_PRIOR], fallback=[1.0, 1.0, 1.0])


def _clean_history(history: Union[str, Iterable[Any], None]) -> List[str]:
    if history is None:
        return []
    if isinstance(history, str):
        raw = history.strip().upper()
        allowed = {"B", "P", "T", ",", " ", "|", "/", "-", "_"}
        if all(char in allowed for char in raw):
            items = [char for char in raw if char in CLASS_TO_INDEX]
        else:
            items = [part.strip().upper() for part in raw.replace("|", ",").split(",")]
    else:
        items = [str(value).strip().upper() for value in history]
    return [item for item in items if item in CLASS_TO_INDEX]


def _history_fingerprint(history: Sequence[str]) -> str:
    return hashlib.sha1("".join(history).encode("utf-8")).hexdigest()[:16]


def _is_extension(previous: Sequence[str], current: Sequence[str]) -> bool:
    return len(current) >= len(previous) and list(current[: len(previous)]) == list(previous)


def _last_bp_side(history: Sequence[str]) -> Optional[str]:
    for item in reversed(history):
        if item in {"B", "P"}:
            return item
    return None


def _continuation_to_bp_probs(
    continuation_probs: Sequence[float], history: Sequence[str]
) -> np.ndarray:
    """Map [continue, switch] probabilities back to absolute B/P."""
    cs = _normalize(continuation_probs, fallback=[0.5, 0.5])
    last_side = _last_bp_side(history)
    if last_side == "B":
        return np.asarray([cs[0], cs[1]], dtype=np.float64)
    if last_side == "P":
        return np.asarray([cs[1], cs[0]], dtype=np.float64)
    return np.asarray([0.5, 0.5], dtype=np.float64)


def _bp_fallback_probs(history: Sequence[str]) -> np.ndarray:
    """Neutral fallback; never follows the side that currently appears more."""
    del history
    return np.asarray([0.5, 0.5], dtype=np.float64)


def _estimate_tie_rate(history: Sequence[str]) -> float:
    prior = float(_prior_probs()[CLASS_TO_INDEX["T"]])
    tie_count = float(sum(item == "T" for item in history))
    denominator = float(len(history)) + TIE_PRIOR_STRENGTH
    estimated = (
        (tie_count + prior * TIE_PRIOR_STRENGTH) / denominator
        if denominator > 0
        else prior
    )
    lower = min(TIE_RATE_MIN, TIE_RATE_MAX)
    upper = max(TIE_RATE_MIN, TIE_RATE_MAX)
    return _clamp(estimated, lower, upper)


def _bp_to_triplet(bp_probs: Sequence[float], tie_rate: float) -> np.ndarray:
    bp = _normalize(bp_probs, fallback=_bp_prior_probs())
    tie = _clamp(tie_rate, 0.0, 0.60)
    remaining = max(0.0, 1.0 - tie)
    return _normalize(
        [float(bp[0]) * remaining, float(bp[1]) * remaining, tie],
        fallback=_prior_probs(),
    )


def _triplet_to_bp(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2:
        return _bp_prior_probs()
    return _normalize([arr[0], arr[1]], fallback=_bp_prior_probs())


def _to_prob_dict(
    values: Sequence[float], digits: Optional[int] = None
) -> Dict[str, float]:
    probs = _normalize(values, fallback=_prior_probs())
    if len(probs) == 2:
        result = {"B": float(probs[0]), "P": float(probs[1])}
    else:
        result = {
            name: float(probs[idx])
            for idx, name in enumerate(CLASS_NAMES[: len(probs)])
        }
    if digits is not None:
        result = {key: round(value, digits) for key, value in result.items()}
    return result


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _as_probability(value: Any) -> Optional[float]:
    number = _safe_float(value)
    if number is None:
        return None
    if 1.0 < number <= 100.0:
        number /= 100.0
    return _clamp(number, 0.0, 1.0)


def _cap_binary_probability(values: Sequence[float], maximum: float) -> np.ndarray:
    probs = _normalize(values, fallback=_bp_prior_probs())
    maximum = _clamp(maximum, 0.50, 0.99)
    winner = int(np.argmax(probs))
    if float(probs[winner]) <= maximum:
        return probs
    result = np.asarray([1.0 - maximum, 1.0 - maximum], dtype=np.float64)
    result[winner] = maximum
    result[1 - winner] = 1.0 - maximum
    return result


def _calibrate_binary_probs(
    values: Sequence[float], sample_count: int
) -> np.ndarray:
    raw = _normalize(values, fallback=_bp_prior_probs())
    prior = _bp_prior_probs()
    strength = max(0.0, MODEL_CALIBRATION_STRENGTH)
    reliability = (
        float(sample_count) / (float(sample_count) + strength)
        if strength > 0.0
        else 1.0
    )
    calibrated = prior * (1.0 - reliability) + raw * reliability
    return _cap_binary_probability(calibrated, COMPONENT_MAX_PROB)


def _signal_level(confidence: float, edge: float) -> str:
    if confidence >= 0.62 and edge >= 0.12:
        return "HIGH"
    if confidence >= 0.56 and edge >= 0.06:
        return "MEDIUM"
    return "LOW"


def _class_weight_mapping(labels: np.ndarray) -> Optional[Dict[int, float]]:
    if not NEURAL_CLASS_WEIGHT or len(labels) == 0:
        return None
    counts = Counter(int(value) for value in labels.tolist())
    if len(counts) <= 1:
        return None
    total = float(len(labels))
    present = float(len(counts))
    return {
        class_index: min(
            NEURAL_CLASS_WEIGHT_MAX,
            total / (present * max(1.0, float(count))),
        )
        for class_index, count in counts.items()
    }


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class ComponentState:
    model: Any = None
    trained: bool = False
    training_samples: int = 0
    last_train_history_len: int = 0
    last_train_fingerprint: str = ""
    train_count: int = 0
    last_loss: Optional[float] = None
    last_accuracy: Optional[float] = None
    status: str = "not_trained"
    backend: str = ""
    class_labels: List[int] = field(default_factory=lambda: [0, 1])
    initialized_from_base: bool = False

    def reset(self) -> None:
        self.model = None
        self.trained = False
        self.training_samples = 0
        self.last_train_history_len = 0
        self.last_train_fingerprint = ""
        self.train_count = 0
        self.last_loss = None
        self.last_accuracy = None
        self.status = "reset"
        self.backend = ""
        self.class_labels = [0, 1]
        self.initialized_from_base = False


@dataclass
class UIDModelState:
    key: str
    scope_key: str
    shoe_id: str
    lstm: ComponentState = field(default_factory=ComponentState)
    gru: ComponentState = field(default_factory=ComponentState)
    tcn: ComponentState = field(default_factory=ComponentState)
    gbm: ComponentState = field(default_factory=ComponentState)
    last_history: List[str] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def reset_local(self) -> None:
        self.lstm.reset()
        self.gru.reset()
        self.tcn.reset()
        self.gbm.reset()
        self.last_history = []


@dataclass
class ScopeMemoryState:
    key: str
    base_lstm: ComponentState = field(default_factory=ComponentState)
    shoe_histories: List[List[str]] = field(default_factory=list)
    archived_fingerprints: List[str] = field(default_factory=list)
    total_archived_samples: int = 0
    active_training_key: str = ""
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


_MODEL_CACHE: "OrderedDict[str, UIDModelState]" = OrderedDict()
_SCOPE_MEMORY_CACHE: "OrderedDict[str, ScopeMemoryState]" = OrderedDict()
_MODEL_CACHE_LOCK = threading.RLock()
_SCOPE_MEMORY_CACHE_LOCK = threading.RLock()
_TF_LOCK = threading.RLock()
_DEEPSEEK_LOCK = threading.RLock()
_DEEPSEEK_CLIENT: Optional[Any] = None
_PATTERN_DB_LOCK = threading.RLock()
_PATTERN_DB_CLIENT: Optional[Any] = None


def _make_scope_key(user_id: str, venue: str, room: str) -> str:
    identity = str(user_id or "anonymous").strip() or "anonymous"
    if not PER_UID_MODELS:
        identity = "global"
    return "|".join(
        [
            identity,
            str(venue or "global").strip() or "global",
            str(room or "global").strip() or "global",
        ]
    )


def _make_training_key(user_id: str, venue: str, room: str, shoe_id: str) -> str:
    return "|".join(
        [
            _make_scope_key(user_id, venue, room),
            str(shoe_id or "global").strip() or "global",
        ]
    )


def _get_scope_memory(scope_key: str) -> ScopeMemoryState:
    with _SCOPE_MEMORY_CACHE_LOCK:
        state = _SCOPE_MEMORY_CACHE.get(scope_key)
        if state is not None:
            _SCOPE_MEMORY_CACHE.move_to_end(scope_key)
            return state
        while len(_SCOPE_MEMORY_CACHE) >= MAX_SCOPE_MEMORIES:
            _SCOPE_MEMORY_CACHE.popitem(last=False)
        state = ScopeMemoryState(key=scope_key)
        _SCOPE_MEMORY_CACHE[scope_key] = state
        return state


def _get_model_state(
    training_key: str, scope_key: str, shoe_id: str
) -> UIDModelState:
    with _MODEL_CACHE_LOCK:
        state = _MODEL_CACHE.get(training_key)
        if state is not None:
            _MODEL_CACHE.move_to_end(training_key)
            return state
        while len(_MODEL_CACHE) >= MAX_UID_MODELS:
            _MODEL_CACHE.popitem(last=False)
        state = UIDModelState(
            key=training_key,
            scope_key=scope_key,
            shoe_id=str(shoe_id or "global"),
        )
        _MODEL_CACHE[training_key] = state
        return state


def _component_info(component: ComponentState) -> Dict[str, Any]:
    return {
        "trained": component.trained,
        "training_samples": component.training_samples,
        "last_train_history_len": component.last_train_history_len,
        "train_count": component.train_count,
        "status": component.status,
        "backend": component.backend,
        "last_loss": component.last_loss,
        "last_accuracy": component.last_accuracy,
        "class_labels": list(component.class_labels),
        "initialized_from_base": component.initialized_from_base,
    }


def clear_model_cache(user_id: Optional[str] = None) -> Dict[str, Any]:
    with _MODEL_CACHE_LOCK, _SCOPE_MEMORY_CACHE_LOCK:
        if not user_id:
            removed = len(_MODEL_CACHE)
            memory_removed = len(_SCOPE_MEMORY_CACHE)
            _MODEL_CACHE.clear()
            _SCOPE_MEMORY_CACHE.clear()
            return {
                "ok": True,
                "removed": removed,
                "memory_removed": memory_removed,
                "user_id": None,
            }
        prefix = f"{str(user_id).strip()}|"
        model_keys = [key for key in _MODEL_CACHE if key.startswith(prefix)]
        memory_keys = [key for key in _SCOPE_MEMORY_CACHE if key.startswith(prefix)]
        for key in model_keys:
            _MODEL_CACHE.pop(key, None)
        for key in memory_keys:
            _SCOPE_MEMORY_CACHE.pop(key, None)
        return {
            "ok": True,
            "removed": len(model_keys),
            "memory_removed": len(memory_keys),
            "user_id": user_id,
        }


def reset_uid_model(
    user_id: str,
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
) -> Dict[str, Any]:
    key = _make_training_key(user_id, venue, room, shoe_id)
    with _MODEL_CACHE_LOCK:
        removed = _MODEL_CACHE.pop(key, None) is not None
    return {"ok": True, "removed": int(removed), "training_key": key}


def get_model_cache_info() -> Dict[str, Any]:
    with _MODEL_CACHE_LOCK, _SCOPE_MEMORY_CACHE_LOCK:
        return {
            "size": len(_MODEL_CACHE),
            "max_size": MAX_UID_MODELS,
            "scope_memory_size": len(_SCOPE_MEMORY_CACHE),
            "per_uid_models": PER_UID_MODELS,
            "keys": list(_MODEL_CACHE.keys()),
            "models": {
                key: {
                    "lstm": _component_info(state.lstm),
                    "gru": _component_info(state.gru),
                    "tcn": _component_info(state.tcn),
                    "gbm": _component_info(state.gbm),
                }
                for key, state in _MODEL_CACHE.items()
            },
            "scope_memories": {
                key: {
                    "base_lstm": _component_info(state.base_lstm),
                    "archived_shoes": len(state.shoe_histories),
                    "total_archived_samples": state.total_archived_samples,
                    "active_training_key": state.active_training_key,
                }
                for key, state in _SCOPE_MEMORY_CACHE.items()
            },
        }


# ---------------------------------------------------------------------------
# LSTM data and model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NeuralConfig:
    name: str
    enabled: bool
    sequence_length: int
    units: int
    dropout: float
    dense_units: int
    learning_rate: float
    epochs: int
    online_epochs: int
    min_samples: int
    retrain_interval: int


def _neural_config() -> NeuralConfig:
    return NeuralConfig(
        "lstm",
        USE_LSTM,
        LSTM_SEQUENCE_LENGTH,
        LSTM_UNITS,
        LSTM_DROPOUT,
        LSTM_DENSE_UNITS,
        LSTM_LEARNING_RATE,
        LSTM_EPOCHS,
        LSTM_ONLINE_EPOCHS,
        LSTM_MIN_SAMPLES,
        LSTM_RETRAIN_INTERVAL,
    )


def _switch_rate(sequence: Sequence[str]) -> float:
    bp = [item for item in sequence if item in BP_TO_INDEX]
    if len(bp) < 2:
        return 0.5
    changes = sum(current != previous for previous, current in zip(bp, bp[1:]))
    return changes / max(1, len(bp) - 1)


def _road_feature_sequence(
    sequence: Sequence[str], sequence_length: int
) -> np.ndarray:
    """Encode only road shape, never absolute Banker/Player advantage.

    B/P mirror pairs therefore produce the same tensor. The first three values
    represent SAME / CHANGE / TIE relative to the previous non-tie result.
    """
    recent = list(sequence[-sequence_length:])
    encoded = np.zeros((sequence_length, LSTM_FEATURE_DIM), dtype=np.float32)
    if not recent:
        return encoded

    rows: List[List[float]] = []
    previous_bp: Optional[str] = None
    run_length = 0
    shape_events: List[str] = []

    for index, item in enumerate(recent):
        same = change = tie = 0.0
        if item == "T":
            tie = 1.0
            if previous_bp is not None:
                shape_events.append("T")
        elif item in {"B", "P"}:
            if previous_bp is None:
                run_length = 1
            elif item == previous_bp:
                same = 1.0
                run_length += 1
                shape_events.append("S")
            else:
                change = 1.0
                run_length = 1
                shape_events.append("C")
            previous_bp = item

        tail4 = shape_events[-4:]
        tail8 = shape_events[-8:]
        tail12 = shape_events[-12:]
        switch4 = tail4.count("C") / max(1.0, float(sum(x != "T" for x in tail4)))
        switch8 = tail8.count("C") / max(1.0, float(sum(x != "T" for x in tail8)))
        switch12 = tail12.count("C") / max(1.0, float(sum(x != "T" for x in tail12)))
        tie8 = tail8.count("T") / max(1.0, float(len(tail8)))
        alternating = 1.0 if len(tail4) >= 3 and tail4[-3:] == ["C", "C", "C"] else 0.0
        pair_rhythm = 1.0 if len(tail8) >= 4 and tail8[-4:] in (["S", "C", "S", "C"], ["C", "S", "C", "S"]) else 0.0
        run_stability = min(1.0, run_length / 8.0)
        short_long_gap = abs(switch4 - switch12)
        progress = min(1.0, (index + 1) / max(1.0, float(sequence_length)))

        rows.append([
            same,
            change,
            tie,
            run_stability,
            switch4,
            switch8,
            switch12,
            tie8,
            alternating,
            pair_rhythm,
            short_long_gap,
            1.0 - switch4,
            1.0 - switch12,
            progress,
        ])

    row_array = np.asarray(rows, dtype=np.float32)
    encoded[-len(row_array):] = row_array
    return encoded

def _all_binary_samples(
    history: Sequence[str], sequence_length: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build continuation/switch targets instead of absolute B/P targets."""
    X: List[np.ndarray] = []
    y: List[int] = []
    target_indices: List[int] = []
    previous_bp: Optional[str] = None
    for target_index, target in enumerate(history):
        if target == "T":
            continue
        if target not in {"B", "P"}:
            continue
        if previous_bp is None:
            previous_bp = target
            continue
        window = history[max(0, target_index - sequence_length):target_index]
        X.append(_road_feature_sequence(window, sequence_length))
        y.append(0 if target == previous_bp else 1)
        target_indices.append(target_index)
        previous_bp = target
    if not X:
        return (
            np.empty((0, sequence_length, LSTM_FEATURE_DIM), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        np.asarray(target_indices, dtype=np.int64),
    )

def _select_replay_indices(
    target_indices: np.ndarray,
    history: Sequence[str],
    max_samples: int,
) -> np.ndarray:
    count = len(target_indices)
    if count <= max_samples:
        return np.arange(count, dtype=np.int64)

    recent_cutoff = max(0, len(history) - REPLAY_RECENT_WINDOW)
    recent_positions = np.where(target_indices >= recent_cutoff)[0]
    old_positions = np.where(target_indices < recent_cutoff)[0]

    recent_target = min(
        len(recent_positions),
        max(1, int(round(max_samples * REPLAY_RECENT_RATIO))),
    )
    old_target = max_samples - recent_target

    seed = (int(_history_fingerprint(history), 16) + RANDOM_SEED) % (2**31 - 1)
    rng = np.random.default_rng(seed)

    if len(recent_positions) > recent_target:
        recent_selected = rng.choice(recent_positions, recent_target, replace=False)
    else:
        recent_selected = recent_positions

    old_target = min(old_target, len(old_positions))
    if old_target > 0:
        old_selected = rng.choice(old_positions, old_target, replace=False)
    else:
        old_selected = np.empty((0,), dtype=np.int64)

    selected = np.unique(np.concatenate([old_selected, recent_selected])).astype(np.int64)
    if len(selected) < max_samples:
        remaining = np.setdiff1d(np.arange(count), selected, assume_unique=False)
        need = min(max_samples - len(selected), len(remaining))
        if need > 0:
            selected = np.concatenate([selected, remaining[-need:]])
    return np.sort(selected)


def _prepare_local_replay_data(
    history: Sequence[str], sequence_length: int, initial: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    X_all, y_all, target_indices = _all_binary_samples(history, sequence_length)
    total_samples = len(y_all)
    if total_samples == 0:
        return X_all, y_all, np.empty((0,), dtype=np.float64), 0

    if initial:
        selected = np.arange(total_samples, dtype=np.int64)
    else:
        selected = _select_replay_indices(
            target_indices, history, max(4, REPLAY_BATCH_SAMPLES)
        )

    X = X_all[selected]
    y = y_all[selected]
    selected_targets = target_indices[selected]
    ages = np.maximum(0, len(history) - 1 - selected_targets)
    sample_weight = np.power(REPLAY_RECENCY_DECAY, ages.astype(np.float64))

    class_map = _class_weight_mapping(y)
    if class_map:
        sample_weight *= np.asarray(
            [class_map.get(int(label), 1.0) for label in y], dtype=np.float64
        )
    return X, y, sample_weight, total_samples


def _prepare_global_data(
    histories: Sequence[Sequence[str]], sequence_length: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    for history in histories:
        X, y, _ = _all_binary_samples(history, sequence_length)
        if len(y):
            X_parts.append(X)
            y_parts.append(y)
    if not X_parts:
        return (
            np.empty((0, sequence_length, LSTM_FEATURE_DIM), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float64),
            0,
        )
    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    total = len(y_all)
    if total > GLOBAL_MEMORY_MAX_SAMPLES:
        seed_text = "|".join(_history_fingerprint(h) for h in histories)
        seed = (int(hashlib.sha1(seed_text.encode()).hexdigest()[:16], 16) + RANDOM_SEED) % (
            2**31 - 1
        )
        rng = np.random.default_rng(seed)
        selected = np.sort(
            rng.choice(total, GLOBAL_MEMORY_MAX_SAMPLES, replace=False)
        )
        X_all = X_all[selected]
        y_all = y_all[selected]
    weights = np.ones(len(y_all), dtype=np.float64)
    class_map = _class_weight_mapping(y_all)
    if class_map:
        weights *= np.asarray(
            [class_map.get(int(label), 1.0) for label in y_all], dtype=np.float64
        )
    return X_all, y_all, weights, total


def _build_lstm_model(config: NeuralConfig) -> Any:
    if not (config.enabled and TF_AVAILABLE):
        return None
    model = Sequential(
        [
            Input(shape=(config.sequence_length, LSTM_FEATURE_DIM)),
            Masking(mask_value=0.0),
            LSTM(config.units, return_sequences=True, name="base_lstm_layer"),
            Dropout(config.dropout),
            LSTM(LSTM_SECOND_UNITS, return_sequences=False, name="adapt_lstm_layer"),
            Dropout(config.dropout),
            Dense(config.dense_units, activation="relu", name="direction_dense"),
            Dropout(min(0.50, config.dropout * 0.50)),
            Dense(2, activation="softmax", name="bp_output"),
        ],
        name="baccarat_bp_lstm_mc",
    )
    _compile_lstm_model(model, config.learning_rate)
    return model


def _compile_lstm_model(model: Any, learning_rate: float) -> None:
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def _set_first_lstm_trainable(model: Any, trainable: bool) -> None:
    try:
        layer = model.get_layer("base_lstm_layer")
        if bool(layer.trainable) != bool(trainable):
            layer.trainable = bool(trainable)
            _compile_lstm_model(model, LSTM_LEARNING_RATE)
    except Exception:
        pass


def _copy_base_weights(local: ComponentState, base: ComponentState) -> bool:
    if not (base.trained and base.model is not None and local.model is not None):
        return False
    try:
        local.model.set_weights(base.model.get_weights())
        local.initialized_from_base = True
        return True
    except Exception as exc:
        logger.warning("Unable to copy base LSTM weights: %s", exc)
        return False


def _fit_component(
    component: ComponentState,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    epochs: int,
    batch_size: int,
    validation_allowed: bool,
    backend: str,
) -> Dict[str, Any]:
    callbacks: List[Any] = []
    validation_split = 0.0
    if validation_allowed and len(X) >= NEURAL_VALIDATION_MIN_SAMPLES:
        validation_split = 0.20
        if NEURAL_EARLY_STOP_PATIENCE > 0:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=NEURAL_EARLY_STOP_PATIENCE,
                    restore_best_weights=True,
                )
            )
    batch_size = min(max(1, batch_size), len(X))
    with _TF_LOCK:
        fit_result = component.model.fit(
            X,
            y,
            epochs=max(1, epochs),
            batch_size=batch_size,
            verbose=NEURAL_VERBOSE,
            shuffle=True,
            validation_split=validation_split,
            callbacks=callbacks,
            sample_weight=sample_weight if len(sample_weight) else None,
        )
    component.trained = True
    component.train_count += 1
    component.status = "trained"
    component.backend = backend
    metrics = getattr(fit_result, "history", {}) or {}
    if metrics.get("loss"):
        component.last_loss = float(metrics["loss"][-1])
    if metrics.get("accuracy"):
        component.last_accuracy = float(metrics["accuracy"][-1])
    return {
        "trained": True,
        "status": component.status,
        "samples_used": len(X),
        "epochs": max(1, epochs),
        "train_count": component.train_count,
        "loss": component.last_loss,
        "accuracy": component.last_accuracy,
        "backend": backend,
    }


def _train_local_if_needed(
    component: ComponentState,
    base_component: ComponentState,
    config: NeuralConfig,
    history: Sequence[str],
    force: bool = False,
) -> Dict[str, Any]:
    if not TF_AVAILABLE:
        component.status = "tensorflow_unavailable"
        return {
            "trained": False,
            "status": component.status,
            "samples": 0,
            "error": TF_IMPORT_ERROR,
        }

    _, y_all, _ = _all_binary_samples(history, config.sequence_length)
    total_samples = len(y_all)
    component.training_samples = total_samples
    if total_samples < config.min_samples:
        component.status = "not_enough_bp_samples"
        return {
            "trained": component.trained,
            "status": component.status,
            "samples": total_samples,
            "required_samples": config.min_samples,
        }

    if (
        not force
        and component.trained
        and len(history) - component.last_train_history_len < config.retrain_interval
    ):
        component.status = "ready"
        return {
            "trained": True,
            "status": component.status,
            "samples": total_samples,
            "skipped": True,
        }

    initial = component.model is None or not component.trained or force
    if initial:
        component.model = _build_lstm_model(config)
        if component.model is None:
            component.status = "model_build_failed"
            return {"trained": False, "status": component.status, "samples": total_samples}
        copied = _copy_base_weights(component, base_component)
        if copied and LOCAL_FREEZE_FIRST_LSTM:
            _set_first_lstm_trainable(component.model, False)
        epochs = config.epochs
    else:
        if component.initialized_from_base and LOCAL_FREEZE_FIRST_LSTM:
            _set_first_lstm_trainable(component.model, False)
        epochs = config.online_epochs

    X, y, weights, total_samples = _prepare_local_replay_data(
        history, config.sequence_length, initial=initial
    )
    if len(y) == 0:
        component.status = "no_bp_samples"
        return {"trained": component.trained, "status": component.status, "samples": 0}

    result = _fit_component(
        component,
        X,
        y,
        weights,
        epochs=epochs,
        batch_size=NEURAL_BATCH_SIZE,
        validation_allowed=initial,
        backend="tensorflow_bp_lstm_local_replay",
    )
    component.training_samples = total_samples
    component.last_train_history_len = len(history)
    component.last_train_fingerprint = _history_fingerprint(history)
    result.update(
        {
            "samples": total_samples,
            "replay_samples": len(y),
            "initialized_from_base": component.initialized_from_base,
            "first_lstm_frozen": bool(
                component.initialized_from_base and LOCAL_FREEZE_FIRST_LSTM
            ),
        }
    )
    return result


def _train_base_memory(memory: ScopeMemoryState, config: NeuralConfig) -> Dict[str, Any]:
    if not GLOBAL_MEMORY_ENABLED:
        memory.base_lstm.status = "global_memory_disabled"
        return {"trained": False, "status": memory.base_lstm.status, "samples": 0}
    if not TF_AVAILABLE:
        memory.base_lstm.status = "tensorflow_unavailable"
        return {
            "trained": False,
            "status": memory.base_lstm.status,
            "samples": 0,
            "error": TF_IMPORT_ERROR,
        }

    X, y, weights, total_samples = _prepare_global_data(
        memory.shoe_histories, config.sequence_length
    )
    memory.total_archived_samples = total_samples
    memory.base_lstm.training_samples = total_samples
    if total_samples < GLOBAL_MEMORY_MIN_SAMPLES:
        memory.base_lstm.status = "not_enough_global_samples"
        return {
            "trained": memory.base_lstm.trained,
            "status": memory.base_lstm.status,
            "samples": total_samples,
            "required_samples": GLOBAL_MEMORY_MIN_SAMPLES,
        }

    initial = memory.base_lstm.model is None or not memory.base_lstm.trained
    if initial:
        memory.base_lstm.model = _build_lstm_model(config)
        epochs = GLOBAL_MEMORY_EPOCHS
    else:
        _set_first_lstm_trainable(memory.base_lstm.model, True)
        epochs = GLOBAL_MEMORY_ONLINE_EPOCHS
    if memory.base_lstm.model is None:
        memory.base_lstm.status = "model_build_failed"
        return {"trained": False, "status": memory.base_lstm.status, "samples": total_samples}

    result = _fit_component(
        memory.base_lstm,
        X,
        y,
        weights,
        epochs=epochs,
        batch_size=GLOBAL_MEMORY_BATCH_SIZE,
        validation_allowed=initial,
        backend="tensorflow_bp_lstm_cross_shoe",
    )
    memory.base_lstm.training_samples = total_samples
    memory.base_lstm.last_train_history_len = sum(len(h) for h in memory.shoe_histories)
    memory.base_lstm.last_train_fingerprint = hashlib.sha1(
        "|".join(_history_fingerprint(h) for h in memory.shoe_histories).encode()
    ).hexdigest()[:16]
    result.update(
        {
            "samples": total_samples,
            "archived_shoes": len(memory.shoe_histories),
            "memory_persistent": False,
        }
    )
    return result


def _archive_shoe_history(
    memory: ScopeMemoryState,
    history: Sequence[str],
    config: NeuralConfig,
) -> Dict[str, Any]:
    cleaned = _clean_history(history)
    fingerprint = _history_fingerprint(cleaned)
    bp_count = sum(item in BP_TO_INDEX for item in cleaned[1:])
    if not GLOBAL_MEMORY_ENABLED:
        return {"archived": False, "status": "global_memory_disabled"}
    if bp_count < max(4, LSTM_MIN_SAMPLES):
        return {
            "archived": False,
            "status": "shoe_too_short",
            "bp_samples": bp_count,
        }
    with memory.lock:
        if fingerprint in memory.archived_fingerprints:
            return {"archived": False, "status": "already_archived"}
        memory.shoe_histories.append(list(cleaned))
        memory.archived_fingerprints.append(fingerprint)
        if len(memory.shoe_histories) > GLOBAL_MEMORY_MAX_SHOES:
            overflow = len(memory.shoe_histories) - GLOBAL_MEMORY_MAX_SHOES
            del memory.shoe_histories[:overflow]
            del memory.archived_fingerprints[:overflow]
        training = _train_base_memory(memory, config)
    return {
        "archived": True,
        "status": "archived",
        "bp_samples": bp_count,
        "base_training": training,
    }


def _activate_training_key(
    memory: ScopeMemoryState,
    training_key: str,
    config: NeuralConfig,
) -> Dict[str, Any]:
    previous_key = ""
    with memory.lock:
        previous_key = memory.active_training_key
        memory.active_training_key = training_key
    if previous_key and previous_key != training_key:
        with _MODEL_CACHE_LOCK:
            previous_state = _MODEL_CACHE.get(previous_key)
        if previous_state is not None and previous_state.last_history:
            return _archive_shoe_history(memory, previous_state.last_history, config)
    return {"archived": False, "status": "same_or_first_shoe"}


# ---------------------------------------------------------------------------
# Monte Carlo prediction
# ---------------------------------------------------------------------------


def _temperature_scale_rows(values: np.ndarray, temperature: float) -> np.ndarray:
    probs = np.asarray(values, dtype=np.float64)
    probs = np.maximum(probs, 1e-12)
    logits = np.log(probs) / max(1e-6, float(temperature))
    logits -= np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(logits)
    totals = np.maximum(exp_values.sum(axis=1, keepdims=True), 1e-12)
    return exp_values / totals


def _empty_mc_diag(simulations: int, fallback: Sequence[float]) -> Dict[str, Any]:
    return {
        "enabled": True,
        "simulations_requested": simulations,
        "simulations_completed": 0,
        "uncertainty": None,
        "reliability": 0.0,
        "std": {"B": 0.0, "P": 0.0},
        "vote_probs": _to_prob_dict(fallback, digits=6),
    }


def _predict_component_mc(
    component: ComponentState,
    config: NeuralConfig,
    history: Sequence[str],
    simulations: int,
    label: str,
) -> Tuple[np.ndarray, str, bool, Dict[str, Any]]:
    fallback = _bp_fallback_probs(history)
    diag = _empty_mc_diag(simulations, fallback)
    if not TF_AVAILABLE:
        return fallback, "tensorflow_unavailable", False, diag
    if not component.trained or component.model is None:
        return fallback, component.status, False, diag
    if not history:
        return fallback, "empty_history", False, diag

    X = _road_feature_sequence(history, config.sequence_length).reshape(
        1, config.sequence_length, LSTM_FEATURE_DIM
    )
    try:
        with _TF_LOCK:
            deterministic_tensor = component.model(X, training=False)
            if hasattr(deterministic_tensor, "numpy"):
                deterministic_tensor = deterministic_tensor.numpy()
            deterministic_cs = _normalize(
                np.asarray(deterministic_tensor[0], dtype=np.float64),
                fallback=[0.5, 0.5],
            )
            deterministic = _continuation_to_bp_probs(deterministic_cs, history)

            seed_source = f"{label}:{_history_fingerprint(history)}"
            fingerprint_seed = int(
                hashlib.sha1(seed_source.encode()).hexdigest()[:16], 16
            ) % (2**31 - 1)
            tf.random.set_seed((RANDOM_SEED + fingerprint_seed) % (2**31 - 1))

            samples: List[np.ndarray] = []
            remaining = int(simulations)
            while remaining > 0:
                current_batch = min(remaining, max(1, LSTM_MC_BATCH_SIZE))
                batch_input = np.repeat(X, current_batch, axis=0)
                stochastic = component.model(batch_input, training=True)
                if hasattr(stochastic, "numpy"):
                    stochastic = stochastic.numpy()
                samples.append(
                    _temperature_scale_rows(
                        np.asarray(stochastic, dtype=np.float64),
                        LSTM_MC_TEMPERATURE,
                    )
                )
                remaining -= current_batch
            matrix_cs = np.concatenate(samples, axis=0)
            last_side = _last_bp_side(history)
            if last_side == "P":
                matrix = matrix_cs[:, [1, 0]]
            elif last_side == "B":
                matrix = matrix_cs
            else:
                matrix = np.repeat(np.asarray([[0.5, 0.5]]), len(matrix_cs), axis=0)

        mc_mean = _normalize(matrix.mean(axis=0), fallback=fallback)
        mc_std = np.std(matrix, axis=0)
        winners = np.argmax(matrix, axis=1)
        vote_counts = np.bincount(winners, minlength=2).astype(float)
        vote_probs = _normalize(vote_counts, fallback=mc_mean)
        blended = _normalize(
            mc_mean * (1.0 - LSTM_MC_VOTE_BLEND)
            + vote_probs * LSTM_MC_VOTE_BLEND,
            fallback=mc_mean,
        )
        uncertainty = float(np.mean(mc_std))
        reliability = _clamp(
            1.0 - uncertainty * LSTM_MC_UNCERTAINTY_SCALE,
            LSTM_MC_MIN_RELIABILITY,
            1.0,
        )
        adjusted = _normalize(
            blended * reliability + fallback * (1.0 - reliability),
            fallback=deterministic,
        )
        calibrated = _calibrate_binary_probs(adjusted, component.training_samples)
        diag = {
            "enabled": True,
            "component": label,
            "simulations_requested": simulations,
            "simulations_completed": int(len(matrix)),
            "temperature": round(LSTM_MC_TEMPERATURE, 6),
            "vote_blend": round(LSTM_MC_VOTE_BLEND, 6),
            "uncertainty": round(uncertainty, 8),
            "reliability": round(reliability, 6),
            "deterministic_probs": _to_prob_dict(deterministic, digits=6),
            "mean_probs": _to_prob_dict(mc_mean, digits=6),
            "std": {
                "B": round(float(mc_std[0]), 6),
                "P": round(float(mc_std[1]), 6),
            },
            "vote_probs": _to_prob_dict(vote_probs, digits=6),
            "final_bp_probs": _to_prob_dict(calibrated, digits=6),
        }
        return calibrated, "ready_mc", True, diag
    except Exception as exc:
        logger.exception("%s Monte Carlo prediction failed", label)
        diag["error"] = str(exc)
        return fallback, f"predict_error:{exc}", False, diag


# ---------------------------------------------------------------------------
# Large pattern database
# ---------------------------------------------------------------------------


def _get_pattern_db_client() -> Optional[Any]:
    global _PATTERN_DB_CLIENT
    if not USE_PATTERN_DATABASE or PatternDatabase is None:
        return None
    with _PATTERN_DB_LOCK:
        if _PATTERN_DB_CLIENT is None:
            _PATTERN_DB_CLIENT = PatternDatabase(
                path=PATTERN_DB_PATH,
                max_order=PATTERN_DB_MAX_ORDER,
                min_matches=PATTERN_DB_MIN_MATCHES,
                smoothing=PATTERN_DB_SMOOTHING,
                b_prior=0.5,
            )
        return _PATTERN_DB_CLIENT


def _pattern_db_probs(history: Sequence[str]) -> Tuple[np.ndarray, str, bool, Dict[str, Any]]:
    fallback = _bp_fallback_probs(history)
    if not USE_PATTERN_DATABASE:
        return fallback, "disabled", False, {"enabled": False}
    if not history and not PATTERN_DB_FIRST_HAND_ENABLED:
        return fallback, "first_hand_disabled", False, {"enabled": True}
    client = _get_pattern_db_client()
    if client is None:
        status = "module_unavailable" if PatternDatabase is None else "client_unavailable"
        return fallback, status, False, {"enabled": True, "status": status}
    try:
        lookup = client.lookup(history)
        probs = _normalize(lookup.probs, fallback=fallback)
        info = {
            "enabled": True,
            "path": PATTERN_DB_PATH,
            "status": lookup.status,
            "available": bool(lookup.available),
            "context": lookup.context,
            "order": int(lookup.order),
            "matches": int(lookup.matches),
            "b_count": int(lookup.b_count),
            "p_count": int(lookup.p_count),
            "continue_count": int(getattr(lookup, "continue_count", 0)),
            "switch_count": int(getattr(lookup, "switch_count", 0)),
            "continue_prob": round(float(getattr(lookup, "continue_prob", 0.5)), 6),
            "switch_prob": round(float(getattr(lookup, "switch_prob", 0.5)), 6),
            "last_side": str(getattr(lookup, "last_side", "")),
            "shape_context": str(getattr(lookup, "shape_context", lookup.context)),
            "bp_probs": _to_prob_dict(probs, digits=6),
        }
        return probs, lookup.status, bool(lookup.available), info
    except Exception as exc:
        logger.warning("Pattern database lookup failed: %s", exc)
        return fallback, f"lookup_error:{exc}", False, {
            "enabled": True, "path": PATTERN_DB_PATH, "error": str(exc)
        }


# ---------------------------------------------------------------------------
# Regime and phase logic
# ---------------------------------------------------------------------------


def _phase_name(history_len: int) -> str:
    if history_len <= EARLY_PHASE_END:
        return "EARLY"
    if history_len <= MID_PHASE_END:
        return "MID"
    return "LATE"


def _phase_settings(phase: str) -> Dict[str, float]:
    if phase == "EARLY":
        return {
            "global_weight": EARLY_GLOBAL_WEIGHT,
            "local_weight": EARLY_LOCAL_WEIGHT,
            "deepseek_weight": EARLY_DEEPSEEK_WEIGHT,
            "min_edge": EARLY_MIN_ENTRY_EDGE,
            "min_confidence": EARLY_MIN_BP_CONFIDENCE,
            "max_uncertainty": EARLY_MC_MAX_UNCERTAINTY,
        }
    if phase == "MID":
        return {
            "global_weight": MID_GLOBAL_WEIGHT,
            "local_weight": MID_LOCAL_WEIGHT,
            "deepseek_weight": MID_DEEPSEEK_WEIGHT,
            "min_edge": MID_MIN_ENTRY_EDGE,
            "min_confidence": MID_MIN_BP_CONFIDENCE,
            "max_uncertainty": MID_MC_MAX_UNCERTAINTY,
        }
    return {
        "global_weight": LATE_GLOBAL_WEIGHT,
        "local_weight": LATE_LOCAL_WEIGHT,
        "deepseek_weight": LATE_DEEPSEEK_WEIGHT,
        "min_edge": LATE_MIN_ENTRY_EDGE,
        "min_confidence": LATE_MIN_BP_CONFIDENCE,
        "max_uncertainty": LATE_MC_MAX_UNCERTAINTY,
    }


def _run_lengths(bp: Sequence[str]) -> List[int]:
    if not bp:
        return []
    lengths: List[int] = []
    side = bp[0]
    length = 1
    for item in bp[1:]:
        if item == side:
            length += 1
        else:
            lengths.append(length)
            side = item
            length = 1
    lengths.append(length)
    return lengths


def _regime_vector(sequence: Sequence[str]) -> np.ndarray:
    bp = [item for item in sequence if item in BP_TO_INDEX]
    if not bp:
        return np.asarray([0.5, 0.2, 0.0], dtype=np.float64)
    switch = _switch_rate(bp)
    lengths = _run_lengths(bp)
    mean_run = float(np.mean(lengths)) if lengths else 1.0
    run_std = float(np.std(lengths)) if lengths else 0.0
    return np.asarray([
        switch,
        min(1.0, mean_run / 5.0),
        min(1.0, run_std / 4.0),
    ], dtype=np.float64)


def _regime_score_at(history: Sequence[str], end: int) -> float:
    prefix = list(history[:end])
    if len(prefix) < REGIME_LONG_WINDOW:
        return 0.0
    short = _regime_vector(prefix[-REGIME_SHORT_WINDOW:])
    long = _regime_vector(prefix[-REGIME_LONG_WINDOW:])
    diff = np.abs(short - long)
    return float(diff[0] * 0.45 + diff[1] * 0.35 + diff[2] * 0.20)


def _regime_change_info(history: Sequence[str]) -> Dict[str, Any]:
    if len(history) < REGIME_LONG_WINDOW + 1:
        return {
            "detected": False,
            "score": 0.0,
            "threshold": REGIME_CHANGE_THRESHOLD,
            "cooldown_remaining": 0,
            "hands_since_change": None,
            "short_window": REGIME_SHORT_WINDOW,
            "long_window": REGIME_LONG_WINDOW,
        }

    start = max(REGIME_LONG_WINDOW + 1, len(history) - REGIME_CHANGE_COOLDOWN_HANDS - 2)
    scores: List[Tuple[int, float]] = []
    for end in range(start, len(history) + 1):
        scores.append((end, _regime_score_at(history, end)))

    event_end: Optional[int] = None
    for index, (end, score) in enumerate(scores):
        previous = scores[index - 1][1] if index > 0 else _regime_score_at(history, end - 1)
        if (
            score >= REGIME_CHANGE_THRESHOLD
            and previous < REGIME_CHANGE_THRESHOLD * REGIME_CROSSING_RELEASE_RATIO
        ):
            event_end = end

    current_score = scores[-1][1] if scores else 0.0
    if event_end is None:
        return {
            "detected": False,
            "score": round(current_score, 6),
            "threshold": REGIME_CHANGE_THRESHOLD,
            "cooldown_remaining": 0,
            "hands_since_change": None,
            "short_window": REGIME_SHORT_WINDOW,
            "long_window": REGIME_LONG_WINDOW,
        }

    hands_since = len(history) - event_end
    detected = hands_since <= REGIME_CHANGE_COOLDOWN_HANDS
    cooldown_remaining = max(0, REGIME_CHANGE_COOLDOWN_HANDS - hands_since)
    return {
        "detected": detected,
        "score": round(current_score, 6),
        "threshold": REGIME_CHANGE_THRESHOLD,
        "cooldown_remaining": cooldown_remaining,
        "hands_since_change": hands_since,
        "event_end": event_end,
        "short_window": REGIME_SHORT_WINDOW,
        "long_window": REGIME_LONG_WINDOW,
    }


def _combine_model_probs(
    phase: str,
    global_probs: Sequence[float],
    local_probs: Sequence[float],
    pattern_probs: Sequence[float],
    deepseek_probs: Optional[Sequence[float]],
    availability: Mapping[str, bool],
    fallback: Sequence[float],
) -> Tuple[np.ndarray, Dict[str, float]]:
    settings = _phase_settings(phase)
    weights = {
        "pattern_db": (
            PATTERN_DB_WEIGHT if availability.get("pattern_db", False) else 0.0
        ),
        "global_lstm": (
            settings["global_weight"] * LSTM_WEIGHT
            if availability.get("global_lstm", False)
            else 0.0
        ),
        "local_lstm": (
            settings["local_weight"] * LSTM_WEIGHT
            if availability.get("local_lstm", False)
            else 0.0
        ),
        "deepseek": (
            min(settings["deepseek_weight"], DEEPSEEK_WEIGHT)
            if availability.get("deepseek", False) and DEEPSEEK_WEIGHT > 0.0
            else 0.0
        ),
    }
    total = sum(weights.values())
    if total <= 0.0:
        return _normalize(fallback, fallback=_bp_prior_probs()), {
            **weights,
            "lstm": 0.0,
            "gru": 0.0,
            "tcn": 0.0,
            "gbm": 0.0,
            "markov": 0.0,
        }
    effective = {key: value / total for key, value in weights.items()}
    final = np.zeros(2, dtype=np.float64)
    final += _normalize(pattern_probs, fallback=fallback) * effective["pattern_db"]
    final += _normalize(global_probs, fallback=fallback) * effective["global_lstm"]
    final += _normalize(local_probs, fallback=fallback) * effective["local_lstm"]
    if deepseek_probs is not None:
        final += _normalize(deepseek_probs, fallback=fallback) * effective["deepseek"]
    final = _cap_binary_probability(_normalize(final, fallback=fallback), FINAL_MAX_PROB)
    effective.update(
        {
            "lstm": effective["global_lstm"] + effective["local_lstm"],
            "gru": 0.0,
            "tcn": 0.0,
            "gbm": 0.0,
            "markov": 0.0,
        }
    )
    return final, effective


# ---------------------------------------------------------------------------
# DeepSeek
# ---------------------------------------------------------------------------


def _get_deepseek_client() -> Any:
    global _DEEPSEEK_CLIENT
    if _DEEPSEEK_CLIENT is None:
        _DEEPSEEK_CLIENT = DeepSeekClient()
    return _DEEPSEEK_CLIENT


def _extract_bp_from_mapping(
    data: Mapping[str, Any], base_probs: Sequence[float]
) -> Optional[np.ndarray]:
    for nested_key in ("probabilities", "probs", "prediction", "result", "distribution"):
        nested = data.get(nested_key)
        if isinstance(nested, Mapping):
            parsed = _extract_bp_from_mapping(nested, base_probs)
            if parsed is not None:
                return parsed

    bp_key_sets = [
        ("B", "P"),
        ("b", "p"),
        ("banker", "player"),
        ("banker_prob", "player_prob"),
        ("banker_probability", "player_probability"),
        ("banker_rate", "player_rate"),
    ]
    for keys in bp_key_sets:
        values = [_as_probability(data.get(key)) for key in keys]
        if all(value is not None for value in values):
            return _normalize([float(value) for value in values], fallback=base_probs)

    triplet_sets = [
        ("B", "P", "T"),
        ("b", "p", "t"),
        ("banker", "player", "tie"),
        ("banker_prob", "player_prob", "tie_prob"),
        ("banker_rate", "player_rate", "tie_rate"),
    ]
    for keys in triplet_sets:
        values = [_as_probability(data.get(key)) for key in keys]
        if all(value is not None for value in values):
            return _normalize([float(values[0]), float(values[1])], fallback=base_probs)

    recommendation = str(
        data.get("recommend") or data.get("recommendation") or data.get("side") or ""
    ).strip().upper()
    aliases = {"BANKER": "B", "PLAYER": "P", "莊": "B", "闲": "P", "閒": "P"}
    recommendation = aliases.get(recommendation, recommendation)
    if recommendation in BP_TO_INDEX:
        confidence = _as_probability(data.get("confidence"))
        confidence = 0.58 if confidence is None else _clamp(confidence, 0.50, 0.90)
        result = np.asarray([1.0 - confidence, 1.0 - confidence], dtype=np.float64)
        result[BP_TO_INDEX[recommendation]] = confidence
        result[1 - BP_TO_INDEX[recommendation]] = 1.0 - confidence
        return result
    return None


def _deepseek_probs(
    history: Sequence[str],
    user_id: str,
    venue: str,
    room: str,
    shoe_id: str,
    component_probs: Mapping[str, Sequence[float]],
    local_probs: Sequence[float],
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]], str]:
    if not USE_DEEPSEEK or DEEPSEEK_WEIGHT <= 0.0:
        return None, None, "disabled"
    if len(history) < DEEPSEEK_MIN_HISTORY:
        return None, None, "not_enough_history"

    payload: Dict[str, Any] = {
        "task": "baccarat_next_banker_or_player_probability",
        "classes": ["B", "P"],
        "instruction": (
            "Analyze only road shape: continuation versus transition, run lengths, chop rhythm, "
            "and regime change. Do not use which side has appeared more often as a reason. "
            "Return next non-tie B and P probabilities only; probabilities must sum to 1."
        ),
        "user_id": user_id,
        "venue": venue,
        "room": room,
        "shoe_id": shoe_id,
        "history_len": len(history),
        "history_tail": "".join(history[-48:]),
        "component_bp_probs": {
            key: _to_prob_dict(value, digits=6) for key, value in component_probs.items()
        },
        "local_bp_probs": _to_prob_dict(local_probs, digits=6),
        "timeout_seconds": DEEPSEEK_TIMEOUT_SECONDS,
    }
    try:
        with _DEEPSEEK_LOCK:
            raw = _get_deepseek_client().calibrate(payload)
    except Exception as exc:
        logger.warning("DeepSeek calibration failed: %s", exc)
        return None, {"error": True, "message": str(exc)}, "call_error"
    if not isinstance(raw, Mapping):
        return None, {"error": True, "message": "DeepSeek response is not a mapping"}, "invalid_response"
    if raw.get("error"):
        return None, dict(raw), "api_error"
    parsed = _extract_bp_from_mapping(raw, local_probs)
    if parsed is None:
        return None, dict(raw), "unrecognized_response"
    return parsed, dict(raw), "ready"


# ---------------------------------------------------------------------------
# Disabled component stubs
# ---------------------------------------------------------------------------


def _select_gbm_backend() -> str:
    return ""


def _disabled_component_result(name: str) -> Dict[str, Any]:
    return {
        "trained": False,
        "status": "disabled_lstm_only",
        "samples": 0,
        "model": name,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_history(
    history: Union[str, Iterable[Any]],
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
    user_id: str = "",
    force: bool = True,
) -> Dict[str, Any]:
    """Explicitly fit/update the current-shoe B/P LSTM for one UID."""
    cleaned = _clean_history(history)
    uid = str(user_id or "").strip()
    if REQUIRE_USER_ID and not uid:
        return {"ok": False, "error": "user_id is required when REQUIRE_USER_ID=1"}

    scope_key = _make_scope_key(uid, venue, room)
    training_key = _make_training_key(uid, venue, room, shoe_id)
    config = _neural_config()
    memory = _get_scope_memory(scope_key)
    activation = _activate_training_key(memory, training_key, config)
    state = _get_model_state(training_key, scope_key, shoe_id)

    with state.lock:
        if state.last_history and not _is_extension(state.last_history, cleaned):
            archive = _archive_shoe_history(memory, state.last_history, config)
            state.reset_local()
        else:
            archive = {"archived": False, "status": "extension"}
        local_training = _train_local_if_needed(
            state.lstm, memory.base_lstm, config, cleaned, force=force
        )
        state.gru.status = "disabled_lstm_only"
        state.tcn.status = "disabled_lstm_only"
        state.gbm.status = "disabled_lstm_only"
        state.last_history = list(cleaned)

    results = {
        "lstm": local_training,
        "global_lstm": _component_info(memory.base_lstm),
        "gru": _disabled_component_result("gru"),
        "tcn": _disabled_component_result("tcn"),
        "gbm": _disabled_component_result("gbm"),
    }
    return {
        "ok": True,
        "training_key": training_key,
        "scope_key": scope_key,
        "user_id": uid,
        "history_len": len(cleaned),
        "bp_training_samples": state.lstm.training_samples,
        "tf_available": TF_AVAILABLE,
        "gbm_backend": "DISABLED_LSTM_ONLY",
        "models": results,
        "lstm": local_training,
        "shoe_activation": activation,
        "shoe_archive": archive,
        "cross_shoe_memory": {
            "enabled": GLOBAL_MEMORY_ENABLED,
            "archived_shoes": len(memory.shoe_histories),
            "samples": memory.total_archived_samples,
            "base_trained": memory.base_lstm.trained,
            "persistent_across_restart": False,
        },
    }


def predict(
    history: Union[str, Iterable[Any]],
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
    user_id: str = "",
) -> Dict[str, Any]:
    """Predict B/P from first record with pattern DB + phase-aware LSTM + DeepSeek."""
    cleaned = _clean_history(history)
    uid = str(user_id or "").strip()
    if REQUIRE_USER_ID and not uid:
        return {
            "ok": False,
            "error": "user_id is required when REQUIRE_USER_ID=1",
            "recommend": "NONE",
            "recommend_text": "觀望",
        }

    scope_key = _make_scope_key(uid, venue, room)
    training_key = _make_training_key(uid, venue, room, shoe_id)
    config = _neural_config()
    memory = _get_scope_memory(scope_key)
    activation = _activate_training_key(memory, training_key, config)
    state = _get_model_state(training_key, scope_key, shoe_id)

    try:
        with state.lock:
            reset_detected = bool(
                state.last_history and not _is_extension(state.last_history, cleaned)
            )
            if reset_detected:
                archive_result = _archive_shoe_history(
                    memory, state.last_history, config
                )
                state.reset_local()
            else:
                archive_result = {"archived": False, "status": "no_reset"}

            local_training = _train_local_if_needed(
                state.lstm, memory.base_lstm, config, cleaned, force=False
            )
            state.gru.status = "disabled_lstm_only"
            state.tcn.status = "disabled_lstm_only"
            state.gbm.status = "disabled_lstm_only"

            local_bp, local_status, local_available, local_mc = _predict_component_mc(
                state.lstm,
                config,
                cleaned,
                simulations=LSTM_MC_SIMULATIONS,
                label="local_lstm",
            )
            with memory.lock:
                global_bp, global_status, global_available, global_mc = (
                    _predict_component_mc(
                        memory.base_lstm,
                        config,
                        cleaned,
                        simulations=GLOBAL_MC_SIMULATIONS,
                        label="global_lstm",
                    )
                )
            state.last_history = list(cleaned)

        fallback_bp = _bp_fallback_probs(cleaned)
        phase = _phase_name(len(cleaned))
        phase_settings = _phase_settings(phase)
        pattern_bp, pattern_status, pattern_available, pattern_info = (
            _pattern_db_probs(cleaned)
        )

        # Local LSTM aggregate used as DeepSeek context before DeepSeek is called.
        pre_ai_availability = {
            "pattern_db": pattern_available,
            "global_lstm": global_available,
            "local_lstm": local_available,
            "deepseek": False,
        }
        pre_ai_bp, pre_ai_weights = _combine_model_probs(
            phase,
            global_bp,
            local_bp,
            pattern_bp,
            None,
            pre_ai_availability,
            fallback_bp,
        )

        ai_bp, ai_result, ai_status = _deepseek_probs(
            history=cleaned,
            user_id=uid,
            venue=venue,
            room=room,
            shoe_id=shoe_id,
            component_probs={
                "pattern_db": pattern_bp,
                "global_lstm": global_bp,
                "local_lstm": local_bp,
                "phase_lstm": pre_ai_bp,
            },
            local_probs=pre_ai_bp,
        )

        availability = {
            "pattern_db": pattern_available,
            "global_lstm": global_available,
            "local_lstm": local_available,
            "deepseek": ai_bp is not None,
            "gru": False,
            "tcn": False,
            "gbm": False,
        }
        final_bp, effective_weights = _combine_model_probs(
            phase,
            global_bp,
            local_bp,
            pattern_bp,
            ai_bp,
            availability,
            fallback_bp,
        )

        b_bp = float(final_bp[0])
        p_bp = float(final_bp[1])
        confidence = max(b_bp, p_bp)
        edge = abs(b_bp - p_bp)
        signal_level = _signal_level(confidence, edge)
        primary_side = "B" if b_bp >= p_bp else "P"

        tie_rate = _estimate_tie_rate(cleaned)
        final_triplet = _bp_to_triplet(final_bp, tie_rate)
        pattern_triplet = _bp_to_triplet(pattern_bp, tie_rate)
        global_triplet = _bp_to_triplet(global_bp, tie_rate)
        local_triplet = _bp_to_triplet(local_bp, tie_rate)
        pre_ai_triplet = _bp_to_triplet(pre_ai_bp, tie_rate)
        ai_triplet = _bp_to_triplet(ai_bp, tie_rate) if ai_bp is not None else None

        b_prob = float(final_triplet[CLASS_TO_INDEX["B"]])
        p_prob = float(final_triplet[CLASS_TO_INDEX["P"]])
        t_prob = float(final_triplet[CLASS_TO_INDEX["T"]])

        regime_change = _regime_change_info(cleaned)
        local_uncertainty = local_mc.get("uncertainty")
        global_uncertainty = global_mc.get("uncertainty")
        uncertainty_values: List[float] = []
        if local_available and local_uncertainty is not None:
            uncertainty_values.append(float(local_uncertainty))
        if global_available and global_uncertainty is not None:
            uncertainty_values.append(float(global_uncertainty))
        combined_uncertainty = (
            float(np.average(uncertainty_values)) if uncertainty_values else None
        )

        lstm_reference_bp = pre_ai_bp
        lstm_side = "B" if float(lstm_reference_bp[0]) >= float(lstm_reference_bp[1]) else "P"
        deepseek_side = None
        if ai_bp is not None:
            deepseek_side = "B" if float(ai_bp[0]) >= float(ai_bp[1]) else "P"
        deepseek_conflict = bool(
            deepseek_side is not None and deepseek_side != lstm_side
        )

        # Observation filters are retained as diagnostics only. The user requested
        # a B/P recommendation from the first record onward, so no condition can
        # replace the direction with NONE/observe.
        diagnostic_flags: List[str] = []
        model_ready = pattern_available or global_available or local_available
        if not model_ready:
            diagnostic_flags.append("模型與規律資料庫尚未就緒，使用B/P先驗")
        if edge < phase_settings["min_edge"]:
            diagnostic_flags.append("edge_below_old_threshold")
        if confidence < phase_settings["min_confidence"]:
            diagnostic_flags.append("confidence_below_old_threshold")
        if (
            combined_uncertainty is not None
            and combined_uncertainty > phase_settings["max_uncertainty"]
        ):
            diagnostic_flags.append("mc_uncertainty_above_old_threshold")
        if regime_change.get("detected"):
            diagnostic_flags.append("regime_change_detected")
        if deepseek_conflict:
            diagnostic_flags.append("deepseek_conflict")

        is_observe = False
        recommend = primary_side
        recommend_text = "莊" if recommend == "B" else "閒"
        observe_reason = ""


        active_labels = [
            name.upper()
            for name in ("pattern_db", "global_lstm", "local_lstm", "deepseek")
            if availability.get(name, False) and effective_weights.get(name, 0.0) > 0.0
        ]
        active_text = "+".join(active_labels) if active_labels else "BP_PRIOR_FALLBACK"
        reason = (
            f"{active_text}；phase={phase}; pattern_db={pattern_status}; "
            f"global={global_status}; local={local_status}; DeepSeek={ai_status}; "
            f"B/P信心={confidence * 100:.1f}%; edge={edge * 100:.1f}%; "
            f"MC不確定度={combined_uncertainty}; "
            f"regime_change={regime_change.get('detected', False)}"
        )
        if diagnostic_flags:
            reason += "；診斷=" + ",".join(diagnostic_flags)

        disabled_triplet = _bp_to_triplet(fallback_bp, tie_rate)
        configured_weights = {
            "pattern_db": PATTERN_DB_WEIGHT,
            "lstm": LSTM_WEIGHT,
            "deepseek": DEEPSEEK_WEIGHT,
            "global_lstm": phase_settings["global_weight"],
            "local_lstm": phase_settings["local_weight"],
            "phase_deepseek": phase_settings["deepseek_weight"],
            "gru": 0.0,
            "tcn": 0.0,
            "gbm": 0.0,
            "markov": 0.0,
        }
        model_training = {
            "lstm": local_training,
            "global_lstm": _component_info(memory.base_lstm),
            "gru": _disabled_component_result("gru"),
            "tcn": _disabled_component_result("tcn"),
            "gbm": _disabled_component_result("gbm"),
        }

        result: Dict[str, Any] = {
            "ok": True,
            "engine": "LSTM_SHAPE_PATTERN_DB_MC_DEEPSEEK",
            "predict_engine_env": PREDICT_ENGINE,
            "user_id": uid,
            "uid_isolated": bool(PER_UID_MODELS and uid),
            "venue": venue,
            "room": room,
            "shoe_id": shoe_id,
            "round_no": len(cleaned) + 1,
            "history_len": len(cleaned),
            "history_tail": "".join(cleaned[-36:]),
            "banker_rate": round(b_prob * 100.0, 1),
            "player_rate": round(p_prob * 100.0, 1),
            "tie_rate": round(t_prob * 100.0, 1),
            "probabilities": _to_prob_dict(final_triplet, digits=6),
            "bp_probabilities": _to_prob_dict(final_bp, digits=6),
            "recommend": recommend,
            "recommend_text": recommend_text,
            "is_observe": is_observe,
            "observe_reason": observe_reason,
            "confidence": round(confidence, 4),
            "confidence_pct": round(confidence * 100.0, 1),
            "decision_edge": round(edge, 6),
            "signal_level": signal_level,
            "pattern_label": "牌路形狀資料庫＋延續/轉折LSTM＋跨靴記憶＋Replay＋MC",
            "regime": phase,
            "ngram_label": "",
            "ngram_sample": 0,
            "big_road_label": (
                "B/P/T作為輸入；僅B/P作為訓練目標；和局目標跳過"
            ),
            "big_eye_label": "",
            "small_road_label": "",
            "cockroach_label": "",
            "road_consensus_label": (
                "DeepSeek與LSTM一致" if not deepseek_conflict else "DeepSeek與LSTM衝突"
            ),
            "road_consensus_ratio": 1.0 if ai_bp is not None and not deepseek_conflict else 0.5,
            "road_conflict_ratio": 1.0 if deepseek_conflict else 0.0,
            "road_family": {},
            "down3_family": {},
            "down3_family_label": "",
            "dense_board": {},
            "final_confirmation": {
                "lstm_side": lstm_side,
                "deepseek_side": deepseek_side,
                "conflict": deepseek_conflict,
                "override_edge": DEEPSEEK_CONFLICT_OVERRIDE_EDGE,
                "passed": not is_observe,
            },
            "road_lifecycle": {
                "phase": phase,
                "early_end": EARLY_PHASE_END,
                "mid_end": MID_PHASE_END,
                "phase_settings": phase_settings,
            },
            "adaptive_road_memory": {
                "enabled": GLOBAL_MEMORY_ENABLED,
                "scope_key": scope_key,
                "archived_shoes": len(memory.shoe_histories),
                "archived_samples": memory.total_archived_samples,
                "base_trained": memory.base_lstm.trained,
                "persistent_across_restart": False,
                "activation": activation,
                "archive_result": archive_result,
            },
            "pattern_replay_memory": {
                "enabled": True,
                "recent_window": REPLAY_RECENT_WINDOW,
                "batch_samples": REPLAY_BATCH_SAMPLES,
                "recent_ratio": REPLAY_RECENT_RATIO,
                "recency_decay": REPLAY_RECENCY_DECAY,
                "local_training": local_training,
            },
            "road_rhythm": {
                "regime_change": regime_change,
                "short_switch_rate": _switch_rate(cleaned[-REGIME_SHORT_WINDOW:]),
                "long_switch_rate": _switch_rate(cleaned[-REGIME_LONG_WINDOW:]),
            },
            "long_anchor": {
                "global_available": global_available,
                "global_status": global_status,
                "global_bp_probs": _to_prob_dict(global_bp, digits=6),
            },
            "online_model_performance": {},
            "live_walk_forward_performance": {},
            "ask_road_memory": {},
            "walk_forward_enabled": True,
            "direction_core": "LSTM_SHAPE_CONTINUE_SWITCH_REGIME",
            "direction_locked": False,
            "reason": reason,
            "configured_weights": {
                key: round(float(value), 6)
                for key, value in configured_weights.items()
            },
            "effective_weights": {
                key: round(float(value), 6)
                for key, value in effective_weights.items()
            },
            "dynamic_weights": {
                key: round(float(value), 6)
                for key, value in effective_weights.items()
            },
            "local_effective_weights": {
                key: round(float(value), 6) for key, value in pre_ai_weights.items()
            },
            "component_probs": {
                "pattern_db": _to_prob_dict(pattern_triplet, digits=6),
                "global_lstm": _to_prob_dict(global_triplet, digits=6),
                "local_lstm": _to_prob_dict(local_triplet, digits=6),
                "lstm": _to_prob_dict(pre_ai_triplet, digits=6),
                "gru": _to_prob_dict(disabled_triplet, digits=6),
                "tcn": _to_prob_dict(disabled_triplet, digits=6),
                "gbm": _to_prob_dict(disabled_triplet, digits=6),
                "deepseek": _to_prob_dict(ai_triplet, digits=6)
                if ai_triplet is not None
                else None,
                "local": _to_prob_dict(pre_ai_triplet, digits=6),
                "final": _to_prob_dict(final_triplet, digits=6),
            },
            "pattern_database": pattern_info,
            "pattern_db_status": pattern_status,
            "pattern_db_available": pattern_available,
            "pattern_db_matches": int(pattern_info.get("matches") or 0),
            "pattern_db_order": int(pattern_info.get("order") or 0),
            "monte_carlo": {
                "local": local_mc,
                "global": global_mc,
                "combined_uncertainty": combined_uncertainty,
            },
            "mc_simulations": LSTM_MC_SIMULATIONS,
            "mc_simulations_completed": int(
                local_mc.get("simulations_completed") or 0
            ),
            "markov": {
                "enabled": False,
                "removed": True,
                "reason": "Markov removed from prediction fusion",
            },
            "markov_label": "Markov 已移除，不參與方向判斷",
            "ai_used": ai_bp is not None,
            "ai_status": ai_status,
            "ai_result": ai_result if DEBUG_AI_RESULT else None,
            "ml_trained": bool(state.lstm.trained or memory.base_lstm.trained),
            "ml_samples": max(
                state.lstm.training_samples,
                memory.base_lstm.training_samples,
            ),
            "tf_available": TF_AVAILABLE,
            "tf_import_error": TF_IMPORT_ERROR if not TF_AVAILABLE else "",
            "lstm_status": local_status,
            "global_lstm_status": global_status,
            "gru_status": "disabled_lstm_only",
            "tcn_status": "disabled_lstm_only",
            "gbm_status": "disabled_lstm_only",
            "gbm_backend": "DISABLED_LSTM_ONLY",
            "lstm_training": local_training,
            "global_lstm_training": _component_info(memory.base_lstm),
            "gru_training": model_training["gru"],
            "tcn_training": model_training["tcn"],
            "gbm_training": model_training["gbm"],
            "model_training": model_training,
            "lstm_sequence_length": LSTM_SEQUENCE_LENGTH,
            "gru_sequence_length": GRU_SEQUENCE_LENGTH,
            "tcn_sequence_length": TCN_SEQUENCE_LENGTH,
            "lstm_units": LSTM_UNITS,
            "lstm_second_units": LSTM_SECOND_UNITS,
            "gru_units": GRU_UNITS,
            "tcn_filters": TCN_FILTERS,
            "training_key": training_key,
            "scope_key": scope_key,
            "model_cache_size": len(_MODEL_CACHE),
            "ml_predictions": {
                "lr": round(float(disabled_triplet[0]), 6),
                "rf": round(float(disabled_triplet[0]), 6),
                "lstm": round(float(pre_ai_triplet[0]), 6),
                "pattern_db": round(float(pattern_triplet[0]), 6),
                "global_lstm": round(float(global_triplet[0]), 6),
                "local_lstm": round(float(local_triplet[0]), 6),
                "gru": round(float(disabled_triplet[0]), 6),
                "tcn": round(float(disabled_triplet[0]), 6),
                "gbm": round(float(disabled_triplet[0]), 6),
                "ensemble": round(b_prob, 6),
                "lstm_probs": _to_prob_dict(pre_ai_triplet, digits=6),
                "pattern_db_probs": _to_prob_dict(pattern_triplet, digits=6),
                "global_lstm_probs": _to_prob_dict(global_triplet, digits=6),
                "local_lstm_probs": _to_prob_dict(local_triplet, digits=6),
                "gru_probs": _to_prob_dict(disabled_triplet, digits=6),
                "tcn_probs": _to_prob_dict(disabled_triplet, digits=6),
                "gbm_probs": _to_prob_dict(disabled_triplet, digits=6),
            },
        }

        if DEBUG_PREDICTOR:
            result["debug"] = {
                "cleaned_history": cleaned,
                "reset_detected": reset_detected,
                "availability": availability,
                "state": {
                    "local_lstm": _component_info(state.lstm),
                    "global_lstm": _component_info(memory.base_lstm),
                    "gru": _component_info(state.gru),
                    "tcn": _component_info(state.tcn),
                    "gbm": _component_info(state.gbm),
                },
                "deepseek_import_error": _DEEPSEEK_IMPORT_ERROR,
                "regime_change": regime_change,
            }
        else:
            result["debug"] = None
        return result

    except Exception as exc:
        logger.exception("Predictor failed for key=%s", training_key)
        fallback_bp = _bp_fallback_probs(cleaned)
        tie_rate = _estimate_tie_rate(cleaned)
        fallback = _bp_to_triplet(fallback_bp, tie_rate)
        b_prob, p_prob, t_prob = [float(value) for value in fallback]
        confidence = max(float(fallback_bp[0]), float(fallback_bp[1]))
        return {
            "ok": False,
            "error": str(exc),
            "engine": "LSTM_SHAPE_PATTERN_DB_MC_DEEPSEEK",
            "user_id": uid,
            "venue": venue,
            "room": room,
            "shoe_id": shoe_id,
            "round_no": len(cleaned) + 1,
            "history_len": len(cleaned),
            "banker_rate": round(b_prob * 100.0, 1),
            "player_rate": round(p_prob * 100.0, 1),
            "tie_rate": round(t_prob * 100.0, 1),
            "probabilities": _to_prob_dict(fallback, digits=6),
            "bp_probabilities": _to_prob_dict(fallback_bp, digits=6),
            "recommend": "B" if float(fallback_bp[0]) >= float(fallback_bp[1]) else "P",
            "recommend_text": "莊" if float(fallback_bp[0]) >= float(fallback_bp[1]) else "閒",
            "is_observe": False,
            "observe_reason": "",
            "confidence": round(confidence, 4),
            "confidence_pct": round(confidence * 100.0, 1),
            "decision_edge": round(abs(float(fallback_bp[0]) - float(fallback_bp[1])), 6),
            "signal_level": "LOW",
            "reason": "模型執行失敗，使用B/P先驗機率持續給出方向",
            "training_key": training_key,
            "scope_key": scope_key,
            "tf_available": TF_AVAILABLE,
            "ai_used": False,
            "monte_carlo": {
                "enabled": True,
                "simulations_requested": LSTM_MC_SIMULATIONS,
                "simulations_completed": 0,
            },
            "markov": {"enabled": False, "removed": True},
            "markov_label": "Markov 已移除",
            "debug": {"exception": repr(exc)} if DEBUG_PREDICTOR else None,
        }
