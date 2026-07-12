"""Baccarat predictor: per-UID GRU + TCN/1D-CNN + GBM ensemble.

Core design retained from the previous predictor:

* Input is a Big Road sequence containing B / P / T.
* Every LINE UID / venue / room / shoe has isolated online model state.
* Public ``predict`` and ``fit_history`` signatures remain compatible.
* Common response keys used by app.py are preserved.
* Markov has been removed from the actual prediction path.
* LSTM code is retained as an optional A/B-test component, but is disabled by
  default because it overlaps strongly with GRU and consumes more resources.
* GBM backend supports LightGBM, XGBoost, or an automatic sklearn fallback.

Important limitation:
B/P/T history alone cannot guarantee a stable predictive edge in a fair
baccarat game. This module is an experimental sequence classifier and should
be evaluated with strict walk-forward testing rather than training accuracy.
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


def _env_csv_ints(name: str, default: Sequence[int]) -> Tuple[int, ...]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return tuple(int(v) for v in default)
    values: List[int] = []
    for part in raw.split(","):
        try:
            values.append(max(1, int(part.strip())))
        except (TypeError, ValueError):
            continue
    return tuple(values) if values else tuple(int(v) for v in default)


# ---------------------------------------------------------------------------
# Core settings
# ---------------------------------------------------------------------------

PREDICT_ENGINE = os.getenv(
    "PREDICT_ENGINE", "GRU_TCN_GBM_ENSEMBLE"
).strip().upper()
PER_UID_MODELS = _env_bool("PER_UID_MODELS", True)
REQUIRE_USER_ID = _env_bool("REQUIRE_USER_ID", False)
MAX_UID_MODELS = _env_int("MAX_UID_MODELS", 12, minimum=1)

# LSTM is retained for A/B testing, but disabled by default.
USE_LSTM = _env_bool("USE_LSTM", False)
LSTM_SEQUENCE_LENGTH = _env_int("LSTM_SEQUENCE_LENGTH", 16, minimum=4)
LSTM_UNITS = _env_int("LSTM_UNITS", 24, minimum=4)
LSTM_DROPOUT = _env_float("LSTM_DROPOUT", 0.30, 0.0, 0.80)
LSTM_DENSE_UNITS = _env_int("LSTM_DENSE_UNITS", 16, minimum=3)
LSTM_LEARNING_RATE = _env_float("LSTM_LEARNING_RATE", 0.0005, 0.000001)
LSTM_EPOCHS = _env_int("LSTM_EPOCHS", 6, minimum=1)
LSTM_ONLINE_EPOCHS = _env_int("LSTM_ONLINE_EPOCHS", 1, minimum=1)
LSTM_MIN_SAMPLES = _env_int("LSTM_MIN_SAMPLES", 18, minimum=2)
LSTM_RETRAIN_INTERVAL = _env_int("LSTM_RETRAIN_INTERVAL", 6, minimum=1)

USE_GRU = _env_bool("USE_GRU", True)
GRU_SEQUENCE_LENGTH = _env_int("GRU_SEQUENCE_LENGTH", 16, minimum=4)
GRU_UNITS = _env_int("GRU_UNITS", 32, minimum=4)
GRU_DROPOUT = _env_float("GRU_DROPOUT", 0.30, 0.0, 0.80)
GRU_DENSE_UNITS = _env_int("GRU_DENSE_UNITS", 16, minimum=3)
GRU_LEARNING_RATE = _env_float("GRU_LEARNING_RATE", 0.0005, 0.000001)
GRU_EPOCHS = _env_int("GRU_EPOCHS", 6, minimum=1)
GRU_ONLINE_EPOCHS = _env_int("GRU_ONLINE_EPOCHS", 1, minimum=1)
GRU_MIN_SAMPLES = _env_int("GRU_MIN_SAMPLES", 18, minimum=2)
GRU_RETRAIN_INTERVAL = _env_int("GRU_RETRAIN_INTERVAL", 6, minimum=1)

USE_TCN = _env_bool("USE_TCN", True)
TCN_SEQUENCE_LENGTH = _env_int("TCN_SEQUENCE_LENGTH", 20, minimum=4)
TCN_FILTERS = _env_int("TCN_FILTERS", 24, minimum=4)
TCN_KERNEL_SIZE = _env_int("TCN_KERNEL_SIZE", 3, minimum=2)
TCN_DROPOUT = _env_float("TCN_DROPOUT", 0.25, 0.0, 0.80)
TCN_DENSE_UNITS = _env_int("TCN_DENSE_UNITS", 16, minimum=3)
TCN_LEARNING_RATE = _env_float("TCN_LEARNING_RATE", 0.0005, 0.000001)
TCN_EPOCHS = _env_int("TCN_EPOCHS", 6, minimum=1)
TCN_ONLINE_EPOCHS = _env_int("TCN_ONLINE_EPOCHS", 1, minimum=1)
TCN_MIN_SAMPLES = _env_int("TCN_MIN_SAMPLES", 18, minimum=2)
TCN_RETRAIN_INTERVAL = _env_int("TCN_RETRAIN_INTERVAL", 6, minimum=1)

NEURAL_BATCH_SIZE = _env_int("NEURAL_BATCH_SIZE", 8, minimum=1)
NEURAL_VALIDATION_MIN_SAMPLES = _env_int(
    "NEURAL_VALIDATION_MIN_SAMPLES", 36, minimum=4
)
NEURAL_EARLY_STOP_PATIENCE = _env_int(
    "NEURAL_EARLY_STOP_PATIENCE", 2, minimum=0
)
NEURAL_CLASS_WEIGHT = _env_bool("NEURAL_CLASS_WEIGHT", True)
NEURAL_CLASS_WEIGHT_MAX = _env_float(
    "NEURAL_CLASS_WEIGHT_MAX", 3.0, 1.0
)
NEURAL_VERBOSE = _env_int("NEURAL_VERBOSE", 0, minimum=0)

USE_GBM = _env_bool("USE_GBM", True)
GBM_BACKEND = os.getenv("GBM_BACKEND", "AUTO").strip().upper()
GBM_HISTORY_WINDOW = _env_int("GBM_HISTORY_WINDOW", 24, minimum=6)
GBM_CONTEXT_WINDOWS = _env_csv_ints(
    "GBM_CONTEXT_WINDOWS", (4, 6, 8, 12, 16, 24)
)
GBM_MIN_CONTEXT = _env_int("GBM_MIN_CONTEXT", 6, minimum=2)
GBM_MIN_SAMPLES = _env_int("GBM_MIN_SAMPLES", 18, minimum=4)
GBM_RETRAIN_INTERVAL = _env_int("GBM_RETRAIN_INTERVAL", 5, minimum=1)
GBM_N_ESTIMATORS = _env_int("GBM_N_ESTIMATORS", 70, minimum=10)
GBM_LEARNING_RATE = _env_float("GBM_LEARNING_RATE", 0.05, 0.001, 1.0)
GBM_MAX_DEPTH = _env_int("GBM_MAX_DEPTH", 3, minimum=1)
GBM_NUM_LEAVES = _env_int("GBM_NUM_LEAVES", 7, minimum=3)
GBM_MIN_CHILD_SAMPLES = _env_int("GBM_MIN_CHILD_SAMPLES", 5, minimum=1)
GBM_SUBSAMPLE = _env_float("GBM_SUBSAMPLE", 0.85, 0.30, 1.0)
GBM_COLSAMPLE = _env_float("GBM_COLSAMPLE", 0.85, 0.30, 1.0)

# Probability calibration prevents small per-shoe datasets from producing
# misleading 80-99% component confidence.
MODEL_CALIBRATION_STRENGTH = _env_float(
    "MODEL_CALIBRATION_STRENGTH", 70.0, 0.0
)
COMPONENT_MAX_PROB = _env_float("COMPONENT_MAX_PROB", 0.72, 0.34, 0.95)
FINAL_MAX_PROB = _env_float("FINAL_MAX_PROB", 0.68, 0.34, 0.95)

# Optional DeepSeek compatibility layer. It is supported but disabled by
# default so the statistical ensemble remains deterministic.
USE_DEEPSEEK = _env_bool("USE_DEEPSEEK", False)
DEEPSEEK_WEIGHT = _env_float("DEEPSEEK_WEIGHT", 0.0, 0.0)
DEEPSEEK_MIN_HISTORY = _env_int("DEEPSEEK_MIN_HISTORY", 8, minimum=0)
DEEPSEEK_TIMEOUT_SECONDS = _env_float(
    "DEEPSEEK_TIMEOUT_SECONDS", 8.0, minimum=1.0
)

# Default fusion: GRU is the primary sequence model; TCN captures local road
# shapes; GBM consumes engineered rhythm/streak features. LSTM stays at zero
# unless explicitly enabled and assigned a positive weight.
LSTM_WEIGHT = _env_float("LSTM_WEIGHT", 0.0, 0.0)
GRU_WEIGHT = _env_float("GRU_WEIGHT", 0.40, 0.0)
TCN_WEIGHT = _env_float("TCN_WEIGHT", 0.32, 0.0)
GBM_WEIGHT = _env_float("GBM_WEIGHT", 0.28, 0.0)

FALLBACK_PRIOR_STRENGTH = _env_float(
    "FALLBACK_PRIOR_STRENGTH",
    _env_float("LSTM_FALLBACK_PRIOR_STRENGTH", 12.0, 0.0),
    0.0,
)

B_PRIOR = _env_float("B_PRIOR", 0.4586, 0.0001)
P_PRIOR = _env_float("P_PRIOR", 0.4462, 0.0001)
T_PRIOR = _env_float("T_PRIOR", 0.0952, 0.0001)

RANDOM_SEED = _env_int("RANDOM_SEED", 42)
DEBUG_PREDICTOR = _env_bool("DEBUG_PREDICTOR", False)
DEBUG_AI_RESULT = _env_bool("DEBUG_AI_RESULT", False)

CLASS_NAMES: Tuple[str, str, str] = ("B", "P", "T")
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {idx: name for name, idx in CLASS_TO_INDEX.items()}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Optional TensorFlow import
# ---------------------------------------------------------------------------

TF_AVAILABLE = False
TF_IMPORT_ERROR = ""
tf = None
Model = None
Sequential = None
Input = None
LSTM = None
GRU = None
Dense = None
Dropout = None
Conv1D = None
GlobalAveragePooling1D = None
Adam = None
EarlyStopping = None

if USE_LSTM or USE_GRU or USE_TCN:
    try:
        import tensorflow as tf  # type: ignore[assignment]
        from tensorflow.keras.callbacks import EarlyStopping  # type: ignore[assignment]
        from tensorflow.keras.layers import (  # type: ignore[assignment]
            Conv1D,
            Dense,
            Dropout,
            GlobalAveragePooling1D,
            GRU,
            Input,
            LSTM,
        )
        from tensorflow.keras.models import Model, Sequential  # type: ignore[assignment]
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
        logger.warning(
            "TensorFlow unavailable; neural components use fallback probabilities: %s",
            exc,
        )
else:
    TF_IMPORT_ERROR = "all neural components disabled"


# ---------------------------------------------------------------------------
# Optional GBM imports
# ---------------------------------------------------------------------------

LIGHTGBM_AVAILABLE = False
XGBOOST_AVAILABLE = False
SKLEARN_GBM_AVAILABLE = False
GBM_IMPORT_ERRORS: Dict[str, str] = {}
LGBMClassifier = None
XGBClassifier = None
HistGradientBoostingClassifier = None

try:
    from lightgbm import LGBMClassifier  # type: ignore[assignment]

    LIGHTGBM_AVAILABLE = True
except Exception as exc:  # pragma: no cover - deployment dependent
    GBM_IMPORT_ERRORS["lightgbm"] = str(exc)

try:
    from xgboost import XGBClassifier  # type: ignore[assignment]

    XGBOOST_AVAILABLE = True
except Exception as exc:  # pragma: no cover - deployment dependent
    GBM_IMPORT_ERRORS["xgboost"] = str(exc)

try:
    from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore[assignment]

    SKLEARN_GBM_AVAILABLE = True
except Exception as exc:  # pragma: no cover - deployment dependent
    GBM_IMPORT_ERRORS["sklearn"] = str(exc)


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
        if fallback is None:
            arr = np.ones(len(arr), dtype=np.float64)
        else:
            arr = np.asarray(fallback, dtype=np.float64)
            arr = np.maximum(arr, 0.0)
        total = float(arr.sum())
    if total <= 0.0:
        return np.ones(len(arr), dtype=np.float64) / max(1, len(arr))
    return arr / total


def _prior_probs() -> np.ndarray:
    return _normalize([B_PRIOR, P_PRIOR, T_PRIOR], fallback=[1.0, 1.0, 1.0])


def _empirical_fallback_probs(history: Sequence[str]) -> np.ndarray:
    prior = _prior_probs()
    pseudo = prior * FALLBACK_PRIOR_STRENGTH
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    for item in history:
        counts[CLASS_TO_INDEX[item]] += 1.0
    return _normalize(counts + pseudo, fallback=prior)


def _to_prob_dict(
    values: Sequence[float], digits: Optional[int] = None
) -> Dict[str, float]:
    probs = _normalize(values, fallback=_prior_probs())
    result = {name: float(probs[idx]) for idx, name in enumerate(CLASS_NAMES)}
    if digits is not None:
        result = {key: round(value, digits) for key, value in result.items()}
    return result


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
    return len(current) >= len(previous) and list(current[: len(previous)]) == list(
        previous
    )


def _signal_level(confidence: float, edge: float) -> str:
    if confidence >= 0.62 and edge >= 0.10:
        return "HIGH"
    if confidence >= 0.56 and edge >= 0.045:
        return "MEDIUM"
    return "LOW"


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _as_probability(value: Any) -> Optional[float]:
    number = _safe_float(value)
    if number is None:
        return None
    if 1.0 < number <= 100.0:
        number /= 100.0
    return _clamp(number, 0.0, 1.0)




def _cap_probability_vector(
    values: Sequence[float],
    maximum: float,
    fallback: Sequence[float],
) -> np.ndarray:
    """Cap one class without breaking normalization or class ordering."""
    probs = _normalize(values, fallback=fallback)
    max_index = int(np.argmax(probs))
    max_value = float(probs[max_index])
    if max_value <= maximum:
        return probs

    remainder_before = max(1e-12, 1.0 - max_value)
    remainder_after = max(0.0, 1.0 - maximum)
    result = probs.copy()
    result[max_index] = maximum
    for index in range(len(result)):
        if index == max_index:
            continue
        result[index] = probs[index] / remainder_before * remainder_after
    return _normalize(result, fallback=fallback)


def _calibrate_component_probs(
    values: Sequence[float],
    sample_count: int,
    fallback: Sequence[float],
) -> np.ndarray:
    """Shrink small-sample predictions toward the fixed long-term prior."""
    raw = _normalize(values, fallback=fallback)
    base = _prior_probs()
    strength = max(0.0, MODEL_CALIBRATION_STRENGTH)
    reliability = (
        float(sample_count) / (float(sample_count) + strength)
        if strength > 0.0
        else 1.0
    )
    calibrated = base * (1.0 - reliability) + raw * reliability
    return _cap_probability_vector(calibrated, COMPONENT_MAX_PROB, base)


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


def _sample_weights(labels: np.ndarray) -> Optional[np.ndarray]:
    mapping = _class_weight_mapping(labels)
    if not mapping:
        return None
    return np.asarray([mapping.get(int(label), 1.0) for label in labels], dtype=float)


# ---------------------------------------------------------------------------
# Per-UID model state
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
    class_labels: List[int] = field(default_factory=list)

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
        self.class_labels = []


@dataclass
class UIDModelState:
    key: str
    lstm: ComponentState = field(default_factory=ComponentState)
    gru: ComponentState = field(default_factory=ComponentState)
    tcn: ComponentState = field(default_factory=ComponentState)
    gbm: ComponentState = field(default_factory=ComponentState)
    last_history: List[str] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def reset(self) -> None:
        self.lstm.reset()
        self.gru.reset()
        self.tcn.reset()
        self.gbm.reset()
        self.last_history = []


_MODEL_CACHE: "OrderedDict[str, UIDModelState]" = OrderedDict()
_MODEL_CACHE_LOCK = threading.RLock()
_DEEPSEEK_LOCK = threading.RLock()
_DEEPSEEK_CLIENT: Optional[Any] = None


def _make_training_key(user_id: str, venue: str, room: str, shoe_id: str) -> str:
    identity = str(user_id or "anonymous").strip() or "anonymous"
    if not PER_UID_MODELS:
        identity = "global"
    return "|".join(
        [
            identity,
            str(venue or "global").strip() or "global",
            str(room or "global").strip() or "global",
            str(shoe_id or "global").strip() or "global",
        ]
    )


def _get_model_state(training_key: str) -> UIDModelState:
    with _MODEL_CACHE_LOCK:
        state = _MODEL_CACHE.get(training_key)
        if state is not None:
            _MODEL_CACHE.move_to_end(training_key)
            return state

        while len(_MODEL_CACHE) >= MAX_UID_MODELS:
            evicted_key, _ = _MODEL_CACHE.popitem(last=False)
            logger.info("Evicted inactive UID model: %s", evicted_key)

        state = UIDModelState(key=training_key)
        _MODEL_CACHE[training_key] = state
        return state


def clear_model_cache(user_id: Optional[str] = None) -> Dict[str, Any]:
    """Clear every cached model, or only models belonging to one UID."""
    with _MODEL_CACHE_LOCK:
        if not user_id:
            removed = len(_MODEL_CACHE)
            _MODEL_CACHE.clear()
            return {"ok": True, "removed": removed, "user_id": None}

        prefix = f"{str(user_id).strip()}|"
        keys = [key for key in _MODEL_CACHE if key.startswith(prefix)]
        for key in keys:
            _MODEL_CACHE.pop(key, None)
        return {"ok": True, "removed": len(keys), "user_id": user_id}


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
    }


def get_model_cache_info() -> Dict[str, Any]:
    with _MODEL_CACHE_LOCK:
        return {
            "size": len(_MODEL_CACHE),
            "max_size": MAX_UID_MODELS,
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
        }


# ---------------------------------------------------------------------------
# Neural sequence models
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


def _neural_configs() -> Dict[str, NeuralConfig]:
    return {
        "lstm": NeuralConfig(
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
        ),
        "gru": NeuralConfig(
            "gru",
            USE_GRU,
            GRU_SEQUENCE_LENGTH,
            GRU_UNITS,
            GRU_DROPOUT,
            GRU_DENSE_UNITS,
            GRU_LEARNING_RATE,
            GRU_EPOCHS,
            GRU_ONLINE_EPOCHS,
            GRU_MIN_SAMPLES,
            GRU_RETRAIN_INTERVAL,
        ),
        "tcn": NeuralConfig(
            "tcn",
            USE_TCN,
            TCN_SEQUENCE_LENGTH,
            TCN_FILTERS,
            TCN_DROPOUT,
            TCN_DENSE_UNITS,
            TCN_LEARNING_RATE,
            TCN_EPOCHS,
            TCN_ONLINE_EPOCHS,
            TCN_MIN_SAMPLES,
            TCN_RETRAIN_INTERVAL,
        ),
    }


def _one_hot_sequence(sequence: Sequence[str]) -> np.ndarray:
    encoded = np.zeros((len(sequence), len(CLASS_NAMES)), dtype=np.float32)
    for row, item in enumerate(sequence):
        encoded[row, CLASS_TO_INDEX[item]] = 1.0
    return encoded


def _prepare_sequence_data(
    history: Sequence[str],
    sequence_length: int,
    target_start_index: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(history) <= sequence_length:
        return (
            np.empty((0, sequence_length, len(CLASS_NAMES)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    start = sequence_length
    if target_start_index is not None:
        start = max(start, int(target_start_index))

    X: List[np.ndarray] = []
    y: List[int] = []
    for target_index in range(start, len(history)):
        window = history[target_index - sequence_length : target_index]
        X.append(_one_hot_sequence(window))
        y.append(CLASS_TO_INDEX[history[target_index]])

    if not X:
        return (
            np.empty((0, sequence_length, len(CLASS_NAMES)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def _build_neural_model(config: NeuralConfig) -> Any:
    if not (config.enabled and TF_AVAILABLE):
        return None

    if config.name == "lstm":
        model = Sequential(
            [
                Input(shape=(config.sequence_length, len(CLASS_NAMES))),
                LSTM(config.units, return_sequences=False),
                Dropout(config.dropout),
                Dense(config.dense_units, activation="relu"),
                Dense(len(CLASS_NAMES), activation="softmax"),
            ],
            name="baccarat_lstm",
        )
    elif config.name == "gru":
        model = Sequential(
            [
                Input(shape=(config.sequence_length, len(CLASS_NAMES))),
                GRU(config.units, return_sequences=False),
                Dropout(config.dropout),
                Dense(config.dense_units, activation="relu"),
                Dense(len(CLASS_NAMES), activation="softmax"),
            ],
            name="baccarat_gru",
        )
    elif config.name == "tcn":
        inputs = Input(shape=(config.sequence_length, len(CLASS_NAMES)))
        x = Conv1D(
            filters=config.units,
            kernel_size=TCN_KERNEL_SIZE,
            padding="causal",
            dilation_rate=1,
            activation="relu",
        )(inputs)
        x = Dropout(config.dropout)(x)
        x = Conv1D(
            filters=config.units,
            kernel_size=TCN_KERNEL_SIZE,
            padding="causal",
            dilation_rate=2,
            activation="relu",
        )(x)
        x = Dropout(config.dropout)(x)
        x = Conv1D(
            filters=max(8, config.units // 2),
            kernel_size=TCN_KERNEL_SIZE,
            padding="causal",
            dilation_rate=4,
            activation="relu",
        )(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(config.dense_units, activation="relu")(x)
        outputs = Dense(len(CLASS_NAMES), activation="softmax")(x)
        model = Model(inputs=inputs, outputs=outputs, name="baccarat_tcn")
    else:
        raise ValueError(f"Unknown neural model: {config.name}")

    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _train_neural_if_needed(
    component: ComponentState,
    config: NeuralConfig,
    history: Sequence[str],
    force: bool = False,
) -> Dict[str, Any]:
    if not config.enabled:
        component.status = "disabled"
        return {"trained": False, "status": component.status, "samples": 0}
    if not TF_AVAILABLE:
        component.status = "tensorflow_unavailable"
        return {
            "trained": False,
            "status": component.status,
            "samples": 0,
            "error": TF_IMPORT_ERROR,
        }

    total_samples = max(0, len(history) - config.sequence_length)
    if total_samples < config.min_samples:
        component.status = "not_enough_samples"
        component.training_samples = total_samples
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
            "samples": component.training_samples,
            "skipped": True,
        }

    initial_training = component.model is None or not component.trained or force
    if initial_training:
        component.model = _build_neural_model(config)
        target_start = config.sequence_length
        epochs = config.epochs
    else:
        target_start = max(config.sequence_length, component.last_train_history_len)
        epochs = config.online_epochs

    if component.model is None:
        component.status = "model_build_failed"
        return {"trained": False, "status": component.status, "samples": 0}

    X, y = _prepare_sequence_data(
        history,
        config.sequence_length,
        target_start_index=target_start,
    )
    if len(X) == 0:
        component.status = "no_new_samples"
        return {
            "trained": component.trained,
            "status": component.status,
            "samples": component.training_samples,
        }

    callbacks: List[Any] = []
    validation_split = 0.0
    if initial_training and len(X) >= NEURAL_VALIDATION_MIN_SAMPLES:
        validation_split = 0.20
        if NEURAL_EARLY_STOP_PATIENCE > 0:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=NEURAL_EARLY_STOP_PATIENCE,
                    restore_best_weights=True,
                )
            )

    batch_size = min(max(1, NEURAL_BATCH_SIZE), len(X))
    fit_history = component.model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=NEURAL_VERBOSE,
        shuffle=True,
        validation_split=validation_split,
        callbacks=callbacks,
        class_weight=_class_weight_mapping(y),
    )

    component.trained = True
    component.training_samples = total_samples
    component.last_train_history_len = len(history)
    component.last_train_fingerprint = _history_fingerprint(history)
    component.train_count += 1
    component.status = "trained"
    component.backend = "tensorflow"

    metrics = getattr(fit_history, "history", {}) or {}
    if metrics.get("loss"):
        component.last_loss = float(metrics["loss"][-1])
    if metrics.get("accuracy"):
        component.last_accuracy = float(metrics["accuracy"][-1])

    return {
        "trained": True,
        "status": component.status,
        "samples": component.training_samples,
        "new_samples": len(X),
        "epochs": epochs,
        "train_count": component.train_count,
        "loss": component.last_loss,
        "accuracy": component.last_accuracy,
    }


def _predict_neural(
    component: ComponentState,
    config: NeuralConfig,
    history: Sequence[str],
) -> Tuple[np.ndarray, str, bool]:
    fallback = _empirical_fallback_probs(history)
    if not config.enabled:
        return fallback, "disabled", False
    if not TF_AVAILABLE:
        return fallback, "tensorflow_unavailable", False
    if not component.trained or component.model is None:
        return fallback, component.status, False
    if len(history) < config.sequence_length:
        return fallback, "history_shorter_than_sequence", False

    window = history[-config.sequence_length :]
    X = _one_hot_sequence(window).reshape(
        1, config.sequence_length, len(CLASS_NAMES)
    )
    try:
        predicted = component.model.predict(X, verbose=0)[0]
        calibrated = _calibrate_component_probs(
            predicted, component.training_samples, fallback
        )
        return calibrated, "ready", True
    except Exception as exc:
        logger.exception("%s prediction failed", config.name.upper())
        return fallback, f"predict_error:{exc}", False


# ---------------------------------------------------------------------------
# GBM features and model
# ---------------------------------------------------------------------------


def _current_streak(history: Sequence[str]) -> Tuple[str, int]:
    if not history:
        return "", 0
    last = history[-1]
    length = 1
    for item in reversed(history[:-1]):
        if item != last:
            break
        length += 1
    return last, length


def _run_statistics(history: Sequence[str]) -> Dict[str, float]:
    if not history:
        return {
            "runs": 0.0,
            "longest_b": 0.0,
            "longest_p": 0.0,
            "longest_t": 0.0,
            "mean_run": 0.0,
        }

    runs: List[Tuple[str, int]] = []
    side = history[0]
    length = 1
    for item in history[1:]:
        if item == side:
            length += 1
        else:
            runs.append((side, length))
            side = item
            length = 1
    runs.append((side, length))

    return {
        "runs": float(len(runs)),
        "longest_b": float(max((n for s, n in runs if s == "B"), default=0)),
        "longest_p": float(max((n for s, n in runs if s == "P"), default=0)),
        "longest_t": float(max((n for s, n in runs if s == "T"), default=0)),
        "mean_run": float(sum(n for _, n in runs) / max(1, len(runs))),
    }


def _alternation_rate(history: Sequence[str]) -> float:
    if len(history) < 2:
        return 0.5
    comparisons = 0
    changes = 0
    for previous, current in zip(history, history[1:]):
        if previous == "T" or current == "T":
            continue
        comparisons += 1
        changes += int(previous != current)
    return changes / comparisons if comparisons else 0.5


def _extract_gbm_features(history: Sequence[str]) -> np.ndarray:
    """Create fixed-length sequence/rhythm features without a Markov model."""
    features: List[float] = []
    prior = _prior_probs()

    # Positional one-hot encoding for the latest fixed window.
    padded = [""] * max(0, GBM_HISTORY_WINDOW - len(history)) + list(
        history[-GBM_HISTORY_WINDOW:]
    )
    for item in padded:
        features.extend(
            [
                1.0 if item == "B" else 0.0,
                1.0 if item == "P" else 0.0,
                1.0 if item == "T" else 0.0,
            ]
        )

    # Multi-scale context statistics.
    for window_size in GBM_CONTEXT_WINDOWS:
        window = list(history[-window_size:])
        denominator = max(1, len(window))
        counts = Counter(window)
        features.extend(
            [
                counts.get("B", 0) / denominator,
                counts.get("P", 0) / denominator,
                counts.get("T", 0) / denominator,
                (counts.get("B", 0) - counts.get("P", 0)) / denominator,
                _alternation_rate(window),
            ]
        )
        side, streak_len = _current_streak(window)
        features.extend(
            [
                1.0 if side == "B" else 0.0,
                1.0 if side == "P" else 0.0,
                1.0 if side == "T" else 0.0,
                min(1.0, streak_len / max(1.0, float(window_size))),
            ]
        )

    recent = list(history[-GBM_HISTORY_WINDOW:])
    stats = _run_statistics(recent)
    norm = max(1.0, float(len(recent)))
    side, streak_len = _current_streak(recent)
    features.extend(
        [
            min(1.0, len(history) / 100.0),
            1.0 if side == "B" else 0.0,
            1.0 if side == "P" else 0.0,
            1.0 if side == "T" else 0.0,
            min(1.0, streak_len / 10.0),
            stats["runs"] / norm,
            stats["longest_b"] / norm,
            stats["longest_p"] / norm,
            stats["longest_t"] / norm,
            stats["mean_run"] / norm,
            _alternation_rate(recent),
            1.0 if len(recent) >= 2 and recent[-1] != recent[-2] else 0.0,
            1.0
            if len(recent) >= 3 and recent[-1] == recent[-3] != recent[-2]
            else 0.0,
            float(prior[0]),
            float(prior[1]),
            float(prior[2]),
        ]
    )

    return np.asarray(features, dtype=np.float32)


def _prepare_gbm_data(history: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    X: List[np.ndarray] = []
    y: List[int] = []
    for target_index in range(GBM_MIN_CONTEXT, len(history)):
        X.append(_extract_gbm_features(history[:target_index]))
        y.append(CLASS_TO_INDEX[history[target_index]])
    if not X:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def _select_gbm_backend() -> str:
    requested = GBM_BACKEND
    if requested in {"LIGHTGBM", "LGBM"}:
        return "LIGHTGBM" if LIGHTGBM_AVAILABLE else ""
    if requested in {"XGBOOST", "XGB"}:
        return "XGBOOST" if XGBOOST_AVAILABLE else ""
    if requested in {"SKLEARN", "HISTGB"}:
        return "SKLEARN" if SKLEARN_GBM_AVAILABLE else ""

    # AUTO priority: LightGBM is fastest for this small tabular workload;
    # XGBoost follows; sklearn is a dependency-safe final fallback.
    if LIGHTGBM_AVAILABLE:
        return "LIGHTGBM"
    if XGBOOST_AVAILABLE:
        return "XGBOOST"
    if SKLEARN_GBM_AVAILABLE:
        return "SKLEARN"
    return ""


def _build_gbm_model(backend: str, class_count: int) -> Any:
    if backend == "LIGHTGBM":
        return LGBMClassifier(
            objective="multiclass" if class_count > 2 else "binary",
            num_class=3 if class_count > 2 else None,
            n_estimators=GBM_N_ESTIMATORS,
            learning_rate=GBM_LEARNING_RATE,
            max_depth=GBM_MAX_DEPTH,
            num_leaves=GBM_NUM_LEAVES,
            min_child_samples=GBM_MIN_CHILD_SAMPLES,
            subsample=GBM_SUBSAMPLE,
            colsample_bytree=GBM_COLSAMPLE,
            random_state=RANDOM_SEED,
            n_jobs=1,
            verbosity=-1,
        )
    if backend == "XGBOOST":
        kwargs: Dict[str, Any] = {
            "n_estimators": GBM_N_ESTIMATORS,
            "learning_rate": GBM_LEARNING_RATE,
            "max_depth": GBM_MAX_DEPTH,
            "subsample": GBM_SUBSAMPLE,
            "colsample_bytree": GBM_COLSAMPLE,
            "min_child_weight": max(1, GBM_MIN_CHILD_SAMPLES),
            "random_state": RANDOM_SEED,
            "n_jobs": 1,
            "tree_method": "hist",
            "eval_metric": "mlogloss" if class_count > 2 else "logloss",
        }
        if class_count > 2:
            kwargs.update({"objective": "multi:softprob", "num_class": 3})
        else:
            kwargs.update({"objective": "binary:logistic"})
        return XGBClassifier(**kwargs)
    if backend == "SKLEARN":
        return HistGradientBoostingClassifier(
            learning_rate=GBM_LEARNING_RATE,
            max_iter=GBM_N_ESTIMATORS,
            max_depth=GBM_MAX_DEPTH,
            min_samples_leaf=GBM_MIN_CHILD_SAMPLES,
            random_state=RANDOM_SEED,
        )
    return None


def _train_gbm_if_needed(
    component: ComponentState,
    history: Sequence[str],
    force: bool = False,
) -> Dict[str, Any]:
    if not USE_GBM:
        component.status = "disabled"
        return {"trained": False, "status": component.status, "samples": 0}

    backend = _select_gbm_backend()
    if not backend:
        component.status = "backend_unavailable"
        return {
            "trained": False,
            "status": component.status,
            "samples": 0,
            "errors": dict(GBM_IMPORT_ERRORS),
        }

    X, y = _prepare_gbm_data(history)
    total_samples = len(y)
    component.training_samples = total_samples
    if total_samples < GBM_MIN_SAMPLES:
        component.status = "not_enough_samples"
        return {
            "trained": component.trained,
            "status": component.status,
            "samples": total_samples,
            "required_samples": GBM_MIN_SAMPLES,
            "backend": backend,
        }

    present_classes = sorted(set(int(v) for v in y.tolist()))
    if len(present_classes) < 2:
        component.status = "not_enough_classes"
        return {
            "trained": component.trained,
            "status": component.status,
            "samples": total_samples,
            "classes": present_classes,
            "backend": backend,
        }

    # Encode any two-class subset (for example B/P when no tie occurred yet)
    # into contiguous labels required by LightGBM/XGBoost binary objectives.
    label_to_encoded = {label: index for index, label in enumerate(present_classes)}
    encoded_y = np.asarray(
        [label_to_encoded[int(label)] for label in y], dtype=np.int64
    )

    if (
        not force
        and component.trained
        and len(history) - component.last_train_history_len < GBM_RETRAIN_INTERVAL
    ):
        component.status = "ready"
        return {
            "trained": True,
            "status": component.status,
            "samples": total_samples,
            "skipped": True,
            "backend": component.backend or backend,
        }

    # Tree ensembles are retrained on the complete current shoe so they retain
    # both early and recent context. The dataset is intentionally small.
    model = _build_gbm_model(backend, class_count=len(present_classes))
    if model is None:
        component.status = "model_build_failed"
        return {"trained": False, "status": component.status, "samples": total_samples}

    try:
        weights = _sample_weights(encoded_y)
        if weights is None:
            model.fit(X, encoded_y)
        else:
            model.fit(X, encoded_y, sample_weight=weights)
    except Exception as exc:
        logger.exception("GBM training failed with backend=%s", backend)
        component.status = f"train_error:{exc}"
        return {
            "trained": component.trained,
            "status": component.status,
            "samples": total_samples,
            "backend": backend,
        }

    component.model = model
    component.trained = True
    component.last_train_history_len = len(history)
    component.last_train_fingerprint = _history_fingerprint(history)
    component.train_count += 1
    component.status = "trained"
    component.backend = backend
    component.class_labels = list(present_classes)

    return {
        "trained": True,
        "status": component.status,
        "samples": total_samples,
        "train_count": component.train_count,
        "backend": backend,
        "classes": present_classes,
        "feature_count": int(X.shape[1]),
    }


def _predict_gbm(
    component: ComponentState,
    history: Sequence[str],
) -> Tuple[np.ndarray, str, bool]:
    fallback = _empirical_fallback_probs(history)
    if not USE_GBM:
        return fallback, "disabled", False
    if not component.trained or component.model is None:
        return fallback, component.status, False
    if len(history) < GBM_MIN_CONTEXT:
        return fallback, "history_shorter_than_context", False

    X = _extract_gbm_features(history).reshape(1, -1)
    try:
        raw = np.asarray(component.model.predict_proba(X)[0], dtype=np.float64)
        encoded_classes = getattr(
            component.model, "classes_", np.arange(len(raw))
        )
        original_labels = component.class_labels or [
            int(value) for value in encoded_classes
        ]
        mapped = np.zeros(len(CLASS_NAMES), dtype=np.float64)
        for probability, encoded_class in zip(raw, encoded_classes):
            encoded_index = int(encoded_class)
            if 0 <= encoded_index < len(original_labels):
                original_index = int(original_labels[encoded_index])
                if 0 <= original_index < len(mapped):
                    mapped[original_index] = float(probability)
        calibrated = _calibrate_component_probs(
            mapped, component.training_samples, fallback
        )
        return calibrated, "ready", True
    except Exception as exc:
        logger.exception("GBM prediction failed")
        return fallback, f"predict_error:{exc}", False


# ---------------------------------------------------------------------------
# DeepSeek compatibility layer
# ---------------------------------------------------------------------------


def _get_deepseek_client() -> Any:
    global _DEEPSEEK_CLIENT
    if _DEEPSEEK_CLIENT is None:
        _DEEPSEEK_CLIENT = DeepSeekClient()
    return _DEEPSEEK_CLIENT


def _extract_triplet_from_mapping(
    data: Mapping[str, Any],
    base_probs: Sequence[float],
) -> Optional[np.ndarray]:
    for nested_key in (
        "probabilities",
        "probs",
        "prediction",
        "result",
        "distribution",
    ):
        nested = data.get(nested_key)
        if isinstance(nested, Mapping):
            parsed = _extract_triplet_from_mapping(nested, base_probs)
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
    for keys in key_sets:
        values = [_as_probability(data.get(key)) for key in keys]
        if all(value is not None for value in values):
            return _normalize([float(value) for value in values], fallback=base_probs)

    adjustment_keys = ("banker_adjust", "player_adjust", "tie_adjust")
    if any(key in data for key in adjustment_keys):
        adjusted = np.asarray(base_probs, dtype=np.float64).copy()
        for index, key in enumerate(adjustment_keys):
            value = _safe_float(data.get(key, 0.0))
            adjusted[index] += _clamp(value or 0.0, -0.20, 0.20)
        return _normalize(adjusted, fallback=base_probs)

    recommendation = str(
        data.get("recommend")
        or data.get("recommendation")
        or data.get("side")
        or ""
    ).strip().upper()
    aliases = {
        "BANKER": "B",
        "PLAYER": "P",
        "TIE": "T",
        "莊": "B",
        "闲": "P",
        "閒": "P",
        "和": "T",
    }
    recommendation = aliases.get(recommendation, recommendation)
    if recommendation in CLASS_TO_INDEX:
        confidence = _as_probability(data.get("confidence"))
        confidence = 0.60 if confidence is None else _clamp(confidence, 0.34, 0.95)
        base = _normalize(base_probs, fallback=_prior_probs())
        target_index = CLASS_TO_INDEX[recommendation]
        remaining = max(0.0, 1.0 - confidence)
        other_total = float(base.sum() - base[target_index])
        result = np.zeros(len(CLASS_NAMES), dtype=np.float64)
        result[target_index] = confidence
        for idx in range(len(CLASS_NAMES)):
            if idx == target_index:
                continue
            ratio = base[idx] / other_total if other_total > 0 else 0.5
            result[idx] = remaining * ratio
        return _normalize(result, fallback=base)

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
        "task": "baccarat_next_result_probability",
        "classes": list(CLASS_NAMES),
        "instruction": (
            "Return next-hand B/P/T probabilities. Do not return betting advice. "
            "Probabilities must sum to 1."
        ),
        "user_id": user_id,
        "venue": venue,
        "room": room,
        "shoe_id": shoe_id,
        "history_len": len(history),
        "history_tail": "".join(history[-48:]),
        "component_probs": {
            key: _to_prob_dict(value, digits=6)
            for key, value in component_probs.items()
        },
        "local_probs": _to_prob_dict(local_probs, digits=6),
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

    parsed = _extract_triplet_from_mapping(raw, local_probs)
    if parsed is None:
        return None, dict(raw), "unrecognized_response"
    return parsed, dict(raw), "ready"


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def _configured_weights() -> Dict[str, float]:
    return {
        "lstm": max(0.0, LSTM_WEIGHT),
        "gru": max(0.0, GRU_WEIGHT),
        "tcn": max(0.0, TCN_WEIGHT),
        "gbm": max(0.0, GBM_WEIGHT),
        "deepseek": max(0.0, DEEPSEEK_WEIGHT),
        "markov": 0.0,
    }


def _fusion(
    component_probs: Mapping[str, Sequence[float]],
    availability: Mapping[str, bool],
    fallback: Sequence[float],
) -> Tuple[np.ndarray, Dict[str, float]]:
    configured = _configured_weights()
    active_weights: Dict[str, float] = {}
    for name in ("lstm", "gru", "tcn", "gbm", "deepseek"):
        active_weights[name] = (
            configured.get(name, 0.0) if availability.get(name, False) else 0.0
        )

    total = sum(active_weights.values())
    if total <= 0.0:
        return _normalize(fallback, fallback=_prior_probs()), {
            **{key: 0.0 for key in active_weights},
            "markov": 0.0,
        }

    effective = {key: value / total for key, value in active_weights.items()}
    final = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    for name, weight in effective.items():
        final += _normalize(
            component_probs.get(name, fallback), fallback=fallback
        ) * weight

    effective["markov"] = 0.0
    return _normalize(final, fallback=fallback), effective


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
    """Explicitly fit/update all isolated model components for one UID."""
    cleaned = _clean_history(history)
    if REQUIRE_USER_ID and not str(user_id or "").strip():
        return {
            "ok": False,
            "error": "user_id is required when REQUIRE_USER_ID=1",
        }

    key = _make_training_key(user_id, venue, room, shoe_id)
    state = _get_model_state(key)
    configs = _neural_configs()

    with state.lock:
        if state.last_history and not _is_extension(state.last_history, cleaned):
            state.reset()

        results = {
            "lstm": _train_neural_if_needed(
                state.lstm, configs["lstm"], cleaned, force=force
            ),
            "gru": _train_neural_if_needed(
                state.gru, configs["gru"], cleaned, force=force
            ),
            "tcn": _train_neural_if_needed(
                state.tcn, configs["tcn"], cleaned, force=force
            ),
            "gbm": _train_gbm_if_needed(state.gbm, cleaned, force=force),
        }
        state.last_history = list(cleaned)

    return {
        "ok": True,
        "training_key": key,
        "user_id": user_id,
        "history_len": len(cleaned),
        "tf_available": TF_AVAILABLE,
        "gbm_backend": state.gbm.backend or _select_gbm_backend(),
        "models": results,
        # Old compatibility key.
        "lstm": results["lstm"],
    }


def predict(
    history: Union[str, Iterable[Any]],
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
    user_id: str = "",
) -> Dict[str, Any]:
    """Predict next B/P/T probabilities and recommend B or P.

    Every user gets an isolated model key:
        user_id | venue | room | shoe_id

    Pass the LINE UID as ``user_id`` to guarantee isolation.
    """
    cleaned = _clean_history(history)
    uid = str(user_id or "").strip()

    if REQUIRE_USER_ID and not uid:
        return {
            "ok": False,
            "error": "user_id is required when REQUIRE_USER_ID=1",
            "recommend": "NONE",
            "recommend_text": "觀望",
        }

    training_key = _make_training_key(uid, venue, room, shoe_id)
    state = _get_model_state(training_key)
    configs = _neural_configs()

    try:
        with state.lock:
            reset_detected = bool(
                state.last_history and not _is_extension(state.last_history, cleaned)
            )
            if reset_detected:
                state.reset()

            training_results = {
                "lstm": _train_neural_if_needed(
                    state.lstm, configs["lstm"], cleaned, force=False
                ),
                "gru": _train_neural_if_needed(
                    state.gru, configs["gru"], cleaned, force=False
                ),
                "tcn": _train_neural_if_needed(
                    state.tcn, configs["tcn"], cleaned, force=False
                ),
                "gbm": _train_gbm_if_needed(state.gbm, cleaned, force=False),
            }

            lstm_probs, lstm_status, lstm_available = _predict_neural(
                state.lstm, configs["lstm"], cleaned
            )
            gru_probs, gru_status, gru_available = _predict_neural(
                state.gru, configs["gru"], cleaned
            )
            tcn_probs, tcn_status, tcn_available = _predict_neural(
                state.tcn, configs["tcn"], cleaned
            )
            gbm_probs, gbm_status, gbm_available = _predict_gbm(
                state.gbm, cleaned
            )
            state.last_history = list(cleaned)

        fallback = _empirical_fallback_probs(cleaned)
        local_components: Dict[str, np.ndarray] = {
            "lstm": lstm_probs,
            "gru": gru_probs,
            "tcn": tcn_probs,
            "gbm": gbm_probs,
        }
        local_availability = {
            "lstm": lstm_available,
            "gru": gru_available,
            "tcn": tcn_available,
            "gbm": gbm_available,
        }

        local_probs, local_effective = _fusion(
            component_probs=local_components,
            availability={**local_availability, "deepseek": False},
            fallback=fallback,
        )

        ai_probs, ai_result, ai_status = _deepseek_probs(
            history=cleaned,
            user_id=uid,
            venue=venue,
            room=room,
            shoe_id=shoe_id,
            component_probs=local_components,
            local_probs=local_probs,
        )

        all_components: Dict[str, Sequence[float]] = {
            **local_components,
            "deepseek": ai_probs if ai_probs is not None else fallback,
        }
        availability = {
            **local_availability,
            "deepseek": ai_probs is not None,
        }
        final_probs, effective_weights = _fusion(
            component_probs=all_components,
            availability=availability,
            fallback=fallback,
        )
        final_probs = _cap_probability_vector(
            final_probs, FINAL_MAX_PROB, fallback
        )

        b_prob = float(final_probs[CLASS_TO_INDEX["B"]])
        p_prob = float(final_probs[CLASS_TO_INDEX["P"]])
        t_prob = float(final_probs[CLASS_TO_INDEX["T"]])

        # ------------------------------------------------------------------
        # NEW: avoid consistently recommending the same side when the edge
        # is extremely small and the signal is low. The recommendation is
        # then determined by a reproducible random seed derived from the
        # history fingerprint, so the same shoe always yields the same
        # outcome but different shoes can vary.
        # ------------------------------------------------------------------
        bp_total = max(1e-12, b_prob + p_prob)
        confidence = max(b_prob, p_prob) / bp_total
        edge = abs(b_prob - p_prob)
        signal_level = _signal_level(confidence, edge)

        # Convert fingerprint to a local seed – deterministic per history.
        fp = _history_fingerprint(cleaned)
        local_seed = int(fp, 16) % (2**31)
        rng = random.Random(local_seed)

        if signal_level == "LOW" and edge < 0.02:
            # Randomly pick B or P with equal probability to break the
            # monotonic side bias while retaining full reproducibility.
            recommend = "B" if rng.random() < 0.5 else "P"
            recommend_text = "莊" if recommend == "B" else "閒"
            reason_extra = (
                f" (邊緣極小 edge={edge:.4f}，使用歷史指紋隨機打破固定偏好)"
            )
        else:
            recommend = "B" if b_prob >= p_prob else "P"
            recommend_text = "莊" if recommend == "B" else "閒"
            reason_extra = ""

        component_prob_dict = {
            "lstm": _to_prob_dict(lstm_probs, digits=6),
            "gru": _to_prob_dict(gru_probs, digits=6),
            "tcn": _to_prob_dict(tcn_probs, digits=6),
            "gbm": _to_prob_dict(gbm_probs, digits=6),
            "deepseek": _to_prob_dict(ai_probs, digits=6)
            if ai_probs is not None
            else None,
            "local": _to_prob_dict(local_probs, digits=6),
            "final": _to_prob_dict(final_probs, digits=6),
        }

        active_labels = [
            name.upper()
            for name, available in availability.items()
            if available and effective_weights.get(name, 0.0) > 0.0
        ]
        active_text = "+".join(active_labels) if active_labels else "PRIOR_FALLBACK"
        reason = (
            f"{active_text} 融合；"
            f"LSTM={lstm_status}; GRU={gru_status}; TCN={tcn_status}; "
            f"GBM={gbm_status}/{state.gbm.backend or _select_gbm_backend() or 'NONE'}; "
            f"DeepSeek={ai_status}; 莊閒方向信心={confidence * 100:.1f}%"
            f"{reason_extra}"
        )

        configured_weights = _configured_weights()
        result: Dict[str, Any] = {
            "ok": True,
            "engine": "LSTM_GRU_TCN_GBM_DEEPSEEK",
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
            "probabilities": _to_prob_dict(final_probs, digits=6),
            "recommend": recommend,
            "recommend_text": recommend_text,
            "is_observe": False,
            "observe_reason": "",
            "confidence": round(confidence, 4),
            "confidence_pct": round(confidence * 100.0, 1),
            "decision_edge": round(edge, 6),
            "signal_level": signal_level,
            "pattern_label": "GRU+TCN/1D-CNN+GBM 融合（LSTM可選）",
            "regime": "GRU_TCN_GBM_ENSEMBLE",
            "ngram_label": "",
            "ngram_sample": 0,
            "big_road_label": "大路 B/P/T 序列作為神經網路與GBM特徵輸入",
            "big_eye_label": "",
            "small_road_label": "",
            "cockroach_label": "",
            "road_consensus_label": "",
            "road_consensus_ratio": 0.5,
            "road_conflict_ratio": 0.5,
            "road_family": {},
            "down3_family": {},
            "down3_family_label": "",
            "dense_board": {},
            "final_confirmation": {},
            "road_lifecycle": {},
            "adaptive_road_memory": {},
            "pattern_replay_memory": {},
            "road_rhythm": {},
            "long_anchor": {},
            "online_model_performance": {},
            "live_walk_forward_performance": {},
            "ask_road_memory": {},
            "walk_forward_enabled": False,
            "direction_core": "GRU_TCN_GBM_ENSEMBLE",
            "direction_locked": False,
            "reason": reason,
            "configured_weights": {
                key: round(value, 6) for key, value in configured_weights.items()
            },
            "effective_weights": {
                key: round(value, 6) for key, value in effective_weights.items()
            },
            "dynamic_weights": {
                key: round(value, 6) for key, value in effective_weights.items()
            },
            "local_effective_weights": {
                key: round(value, 6) for key, value in local_effective.items()
            },
            "component_probs": component_prob_dict,
            # Markov compatibility fields remain but no Markov calculation is used.
            "markov": {
                "enabled": False,
                "removed": True,
                "reason": "Markov removed from prediction fusion",
            },
            "markov_label": "Markov 已移除，不參與方向判斷",
            "ai_used": ai_probs is not None,
            "ai_status": ai_status,
            "ai_result": ai_result if DEBUG_AI_RESULT else None,
            "ml_trained": any(
                [
                    state.lstm.trained,
                    state.gru.trained,
                    state.tcn.trained,
                    state.gbm.trained,
                ]
            ),
            "ml_samples": max(
                state.lstm.training_samples,
                state.gru.training_samples,
                state.tcn.training_samples,
                state.gbm.training_samples,
            ),
            "tf_available": TF_AVAILABLE,
            "tf_import_error": TF_IMPORT_ERROR if not TF_AVAILABLE else "",
            "lstm_status": lstm_status,
            "gru_status": gru_status,
            "tcn_status": tcn_status,
            "gbm_status": gbm_status,
            "gbm_backend": state.gbm.backend or _select_gbm_backend(),
            "lstm_training": training_results["lstm"],
            "gru_training": training_results["gru"],
            "tcn_training": training_results["tcn"],
            "gbm_training": training_results["gbm"],
            "model_training": training_results,
            "lstm_sequence_length": LSTM_SEQUENCE_LENGTH,
            "gru_sequence_length": GRU_SEQUENCE_LENGTH,
            "tcn_sequence_length": TCN_SEQUENCE_LENGTH,
            "lstm_units": LSTM_UNITS,
            "gru_units": GRU_UNITS,
            "tcn_filters": TCN_FILTERS,
            "training_key": training_key,
            "model_cache_size": len(_MODEL_CACHE),
            "ml_predictions": {
                "lr": round(float(gbm_probs[CLASS_TO_INDEX["B"]]), 6),
                "rf": round(float(tcn_probs[CLASS_TO_INDEX["B"]]), 6),
                "lstm": round(float(lstm_probs[CLASS_TO_INDEX["B"]]), 6),
                "gru": round(float(gru_probs[CLASS_TO_INDEX["B"]]), 6),
                "tcn": round(float(tcn_probs[CLASS_TO_INDEX["B"]]), 6),
                "gbm": round(float(gbm_probs[CLASS_TO_INDEX["B"]]), 6),
                "ensemble": round(b_prob, 6),
                "lstm_probs": _to_prob_dict(lstm_probs, digits=6),
                "gru_probs": _to_prob_dict(gru_probs, digits=6),
                "tcn_probs": _to_prob_dict(tcn_probs, digits=6),
                "gbm_probs": _to_prob_dict(gbm_probs, digits=6),
            },
        }

        if DEBUG_PREDICTOR:
            result["debug"] = {
                "cleaned_history": cleaned,
                "reset_detected": reset_detected,
                "availability": availability,
                "state": {
                    "lstm": _component_info(state.lstm),
                    "gru": _component_info(state.gru),
                    "tcn": _component_info(state.tcn),
                    "gbm": _component_info(state.gbm),
                },
                "gbm_import_errors": GBM_IMPORT_ERRORS,
                "deepseek_import_error": _DEEPSEEK_IMPORT_ERROR,
            }
        else:
            result["debug"] = None

        return result

    except Exception as exc:
        logger.exception("Predictor failed for key=%s", training_key)
        fallback = _empirical_fallback_probs(cleaned)
        b_prob, p_prob, t_prob = [float(value) for value in fallback]
        recommend = "B" if b_prob >= p_prob else "P"
        bp_total = max(1e-12, b_prob + p_prob)
        confidence = max(b_prob, p_prob) / bp_total
        return {
            "ok": False,
            "error": str(exc),
            "engine": "LSTM_GRU_TCN_GBM_DEEPSEEK",
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
            "recommend": recommend,
            "recommend_text": "莊" if recommend == "B" else "閒",
            "is_observe": False,
            "observe_reason": "",
            "confidence": round(confidence, 4),
            "confidence_pct": round(confidence * 100.0, 1),
            "decision_edge": round(abs(b_prob - p_prob), 6),
            "signal_level": "LOW",
            "reason": "模型執行失敗，暫時使用 B/P/T 基礎與當前樣本先驗機率",
            "training_key": training_key,
            "tf_available": TF_AVAILABLE,
            "ai_used": False,
            "markov": {"enabled": False, "removed": True},
            "markov_label": "Markov 已移除",
            "debug": {"exception": repr(exc)} if DEBUG_PREDICTOR else None,
        }
