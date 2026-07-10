Baccarat predictor: per-UID LSTM + Markov + DeepSeek hybrid model.

Core design
-----------
1. Input: Big Road sequence containing B / P / T.
2. LSTM: sequence_length=12, units=128, 3-class softmax output.
3. Markov: transition probabilities from the latest result with alpha smoothing.
4. Fusion: LSTM 0.55 + Markov 0.30 + DeepSeek 0.15.
5. Each LINE UID / venue / room / shoe gets an isolated model state.

The public ``predict`` signature and the most commonly used response fields are
kept compatible with the previous predictor.py so app.py can continue calling:

    predict(history, venue=..., room=..., shoe_id=..., user_id=line_uid)
"""

from __future__ import annotations

import hashlib
import json
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


# ---------------------------------------------------------------------------
# Core settings
# ---------------------------------------------------------------------------
PREDICT_ENGINE = os.getenv("PREDICT_ENGINE", "LSTM_MARKOV_DEEPSEEK").strip().upper()
PER_UID_MODELS = _env_bool("PER_UID_MODELS", True)
REQUIRE_USER_ID = _env_bool("REQUIRE_USER_ID", False)
MAX_UID_MODELS = _env_int("MAX_UID_MODELS", 12, minimum=1)

USE_LSTM = _env_bool("USE_LSTM", True)
LSTM_SEQUENCE_LENGTH = _env_int("LSTM_SEQUENCE_LENGTH", 12, minimum=2)
LSTM_UNITS = _env_int("LSTM_UNITS", 128, minimum=4)
LSTM_DROPOUT = _env_float("LSTM_DROPOUT", 0.20, minimum=0.0, maximum=0.80)
LSTM_DENSE_UNITS = _env_int("LSTM_DENSE_UNITS", 32, minimum=3)
LSTM_LEARNING_RATE = _env_float("LSTM_LEARNING_RATE", 0.001, minimum=0.000001)
LSTM_EPOCHS = _env_int("LSTM_EPOCHS", 8, minimum=1)
LSTM_ONLINE_EPOCHS = _env_int("LSTM_ONLINE_EPOCHS", 2, minimum=1)
LSTM_BATCH_SIZE = _env_int("LSTM_BATCH_SIZE", 8, minimum=1)
LSTM_MIN_SAMPLES = _env_int("LSTM_MIN_SAMPLES", 12, minimum=1)
LSTM_RETRAIN_INTERVAL = _env_int("LSTM_RETRAIN_INTERVAL", 5, minimum=1)
LSTM_VALIDATION_MIN_SAMPLES = _env_int("LSTM_VALIDATION_MIN_SAMPLES", 30, minimum=4)
LSTM_EARLY_STOP_PATIENCE = _env_int("LSTM_EARLY_STOP_PATIENCE", 2, minimum=0)
LSTM_CLASS_WEIGHT = _env_bool("LSTM_CLASS_WEIGHT", True)
LSTM_CLASS_WEIGHT_MAX = _env_float("LSTM_CLASS_WEIGHT_MAX", 3.0, minimum=1.0)
LSTM_VERBOSE = _env_int("LSTM_VERBOSE", 0, minimum=0)
LSTM_FALLBACK_PRIOR_STRENGTH = _env_float(
    "LSTM_FALLBACK_PRIOR_STRENGTH", 12.0, minimum=0.0
)

MARKOV_ALPHA = _env_float("MARKOV_ALPHA", 2.5, minimum=0.000001)

USE_DEEPSEEK = _env_bool("USE_DEEPSEEK", True)
DEEPSEEK_WEIGHT = _env_float("DEEPSEEK_WEIGHT", 0.15, minimum=0.0)
DEEPSEEK_MIN_HISTORY = _env_int("DEEPSEEK_MIN_HISTORY", 6, minimum=0)
DEEPSEEK_TIMEOUT_SECONDS = _env_float("DEEPSEEK_TIMEOUT_SECONDS", 8.0, minimum=1.0)
DEEPSEEK_FALLBACK_MODE = os.getenv("DEEPSEEK_FALLBACK_MODE", "REDISTRIBUTE").strip().upper()

LSTM_WEIGHT = _env_float("LSTM_WEIGHT", 0.55, minimum=0.0)
MARKOV_WEIGHT = _env_float("MARKOV_WEIGHT", 0.30, minimum=0.0)

B_PRIOR = _env_float("B_PRIOR", 0.4586, minimum=0.0001)
P_PRIOR = _env_float("P_PRIOR", 0.4462, minimum=0.0001)
T_PRIOR = _env_float("T_PRIOR", 0.0952, minimum=0.0001)

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
Sequential = None
Input = None
LSTM = None
Dense = None
Dropout = None
Adam = None
EarlyStopping = None

if USE_LSTM:
    try:
        import tensorflow as tf  # type: ignore[assignment]
        from tensorflow.keras.callbacks import EarlyStopping  # type: ignore[assignment]
        from tensorflow.keras.layers import Dense, Dropout, Input, LSTM  # type: ignore[assignment]
        from tensorflow.keras.models import Sequential  # type: ignore[assignment]
        from tensorflow.keras.optimizers import Adam  # type: ignore[assignment]

        TF_AVAILABLE = True
        tf.random.set_seed(RANDOM_SEED)
        try:
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except Exception:
            pass
    except Exception as exc:  # pragma: no cover - depends on deployment image
        TF_IMPORT_ERROR = str(exc)
        logger.warning("TensorFlow unavailable; LSTM uses a prior fallback: %s", exc)
else:
    TF_IMPORT_ERROR = "USE_LSTM=0"


# ---------------------------------------------------------------------------
# Optional DeepSeek client import
# ---------------------------------------------------------------------------
_DEEPSEEK_IMPORT_ERROR = ""
try:
    from deepseek_client import DeepSeekClient  # type: ignore
except Exception as exc:  # pragma: no cover - depends on project files
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


def _normalize(values: Sequence[float], fallback: Optional[Sequence[float]] = None) -> np.ndarray:
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


def _to_prob_dict(values: Sequence[float], digits: Optional[int] = None) -> Dict[str, float]:
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
        # Supports "BPTBP", "B,P,T,B,P", and text containing whitespace.
        if all(char in {"B", "P", "T", ",", " ", "|", "/", "-", "_"} for char in raw):
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
    if number > 1.0 and number <= 100.0:
        number /= 100.0
    return _clamp(number, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Per-UID model state
# ---------------------------------------------------------------------------
@dataclass
class UIDModelState:
    key: str
    model: Any = None
    trained: bool = False
    training_samples: int = 0
    last_train_history_len: int = 0
    last_train_fingerprint: str = ""
    last_history: List[str] = field(default_factory=list)
    train_count: int = 0
    last_loss: Optional[float] = None
    last_accuracy: Optional[float] = None
    status: str = "not_trained"
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def reset(self) -> None:
        self.model = None
        self.trained = False
        self.training_samples = 0
        self.last_train_history_len = 0
        self.last_train_fingerprint = ""
        self.last_history = []
        self.train_count = 0
        self.last_loss = None
        self.last_accuracy = None
        self.status = "reset"


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


def get_model_cache_info() -> Dict[str, Any]:
    with _MODEL_CACHE_LOCK:
        return {
            "size": len(_MODEL_CACHE),
            "max_size": MAX_UID_MODELS,
            "per_uid_models": PER_UID_MODELS,
            "keys": list(_MODEL_CACHE.keys()),
            "models": {
                key: {
                    "trained": state.trained,
                    "training_samples": state.training_samples,
                    "last_train_history_len": state.last_train_history_len,
                    "train_count": state.train_count,
                    "status": state.status,
                }
                for key, state in _MODEL_CACHE.items()
            },
        }


# ---------------------------------------------------------------------------
# LSTM model
# ---------------------------------------------------------------------------
def _build_lstm_model() -> Any:
    if not (USE_LSTM and TF_AVAILABLE):
        return None

    model = Sequential(
        [
            Input(shape=(LSTM_SEQUENCE_LENGTH, len(CLASS_NAMES))),
            LSTM(LSTM_UNITS, return_sequences=False),
            Dropout(LSTM_DROPOUT),
            Dense(LSTM_DENSE_UNITS, activation="relu"),
            Dense(len(CLASS_NAMES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=LSTM_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _one_hot_sequence(sequence: Sequence[str]) -> np.ndarray:
    encoded = np.zeros((len(sequence), len(CLASS_NAMES)), dtype=np.float32)
    for row, item in enumerate(sequence):
        encoded[row, CLASS_TO_INDEX[item]] = 1.0
    return encoded


def _prepare_lstm_data(
    history: Sequence[str],
    target_start_index: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(history) <= LSTM_SEQUENCE_LENGTH:
        return (
            np.empty((0, LSTM_SEQUENCE_LENGTH, len(CLASS_NAMES)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    start = LSTM_SEQUENCE_LENGTH
    if target_start_index is not None:
        start = max(start, int(target_start_index))

    X: List[np.ndarray] = []
    y: List[int] = []
    for target_index in range(start, len(history)):
        window = history[target_index - LSTM_SEQUENCE_LENGTH : target_index]
        X.append(_one_hot_sequence(window))
        y.append(CLASS_TO_INDEX[history[target_index]])

    if not X:
        return (
            np.empty((0, LSTM_SEQUENCE_LENGTH, len(CLASS_NAMES)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def _class_weights(labels: np.ndarray) -> Optional[Dict[int, float]]:
    if not LSTM_CLASS_WEIGHT or len(labels) == 0:
        return None

    counts = Counter(int(value) for value in labels.tolist())
    if len(counts) <= 1:
        return None

    total = float(len(labels))
    present_classes = float(len(counts))
    return {
        class_index: min(
            LSTM_CLASS_WEIGHT_MAX,
            total / (present_classes * max(1.0, float(count))),
        )
        for class_index, count in counts.items()
    }


def _empirical_fallback_probs(history: Sequence[str]) -> np.ndarray:
    prior = _prior_probs()
    pseudo = prior * LSTM_FALLBACK_PRIOR_STRENGTH
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    for item in history:
        counts[CLASS_TO_INDEX[item]] += 1.0
    return _normalize(counts + pseudo, fallback=prior)


def _train_lstm_if_needed(state: UIDModelState, history: Sequence[str], force: bool = False) -> Dict[str, Any]:
    if not USE_LSTM:
        state.status = "disabled"
        return {"trained": False, "status": state.status, "samples": 0}
    if not TF_AVAILABLE:
        state.status = "tensorflow_unavailable"
        return {
            "trained": False,
            "status": state.status,
            "samples": 0,
            "error": TF_IMPORT_ERROR,
        }

    total_samples = max(0, len(history) - LSTM_SEQUENCE_LENGTH)
    if total_samples < LSTM_MIN_SAMPLES:
        state.status = "not_enough_samples"
        state.training_samples = total_samples
        return {
            "trained": state.trained,
            "status": state.status,
            "samples": total_samples,
            "required_samples": LSTM_MIN_SAMPLES,
        }

    if (
        not force
        and state.trained
        and len(history) - state.last_train_history_len < LSTM_RETRAIN_INTERVAL
    ):
        state.status = "ready"
        return {
            "trained": True,
            "status": state.status,
            "samples": state.training_samples,
            "skipped": True,
        }

    initial_training = state.model is None or not state.trained or force
    if initial_training:
        # A forced fit rebuilds the model and trains on the complete history.
        state.model = _build_lstm_model()
        target_start = LSTM_SEQUENCE_LENGTH
        epochs = LSTM_EPOCHS
    else:
        # Online update: only train on targets that arrived after the previous fit.
        target_start = max(LSTM_SEQUENCE_LENGTH, state.last_train_history_len)
        epochs = LSTM_ONLINE_EPOCHS

    if state.model is None:
        state.status = "model_build_failed"
        return {"trained": False, "status": state.status, "samples": 0}

    X, y = _prepare_lstm_data(history, target_start_index=target_start)
    if len(X) == 0:
        state.status = "no_new_samples"
        return {
            "trained": state.trained,
            "status": state.status,
            "samples": state.training_samples,
        }

    callbacks: List[Any] = []
    validation_split = 0.0
    if initial_training and len(X) >= LSTM_VALIDATION_MIN_SAMPLES:
        validation_split = 0.20
        if LSTM_EARLY_STOP_PATIENCE > 0:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=LSTM_EARLY_STOP_PATIENCE,
                    restore_best_weights=True,
                )
            )

    batch_size = min(max(1, LSTM_BATCH_SIZE), len(X))
    fit_history = state.model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=LSTM_VERBOSE,
        shuffle=True,
        validation_split=validation_split,
        callbacks=callbacks,
        class_weight=_class_weights(y),
    )

    state.trained = True
    state.training_samples = total_samples
    state.last_train_history_len = len(history)
    state.last_train_fingerprint = _history_fingerprint(history)
    state.train_count += 1
    state.status = "trained"

    metrics = getattr(fit_history, "history", {}) or {}
    if metrics.get("loss"):
        state.last_loss = float(metrics["loss"][-1])
    if metrics.get("accuracy"):
        state.last_accuracy = float(metrics["accuracy"][-1])

    return {
        "trained": True,
        "status": state.status,
        "samples": state.training_samples,
        "new_samples": len(X),
        "epochs": epochs,
        "train_count": state.train_count,
        "loss": state.last_loss,
        "accuracy": state.last_accuracy,
    }


def _predict_lstm(state: UIDModelState, history: Sequence[str]) -> Tuple[np.ndarray, str]:
    fallback = _empirical_fallback_probs(history)
    if not (USE_LSTM and TF_AVAILABLE and state.trained and state.model is not None):
        return fallback, state.status
    if len(history) < LSTM_SEQUENCE_LENGTH:
        return fallback, "history_shorter_than_sequence"

    window = history[-LSTM_SEQUENCE_LENGTH:]
    X = _one_hot_sequence(window).reshape(
        1, LSTM_SEQUENCE_LENGTH, len(CLASS_NAMES)
    )
    try:
        predicted = state.model.predict(X, verbose=0)[0]
        return _normalize(predicted, fallback=fallback), "ready"
    except Exception as exc:
        logger.exception("LSTM prediction failed for %s", state.key)
        return fallback, f"predict_error:{exc}"


# ---------------------------------------------------------------------------
# Markov model
# ---------------------------------------------------------------------------
def _markov_probs(history: Sequence[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    prior = _prior_probs()
    transitions = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.float64)

    for previous, current in zip(history, history[1:]):
        transitions[CLASS_TO_INDEX[previous], CLASS_TO_INDEX[current]] += 1.0

    if not history:
        return prior, {
            "last": "",
            "sample": 0,
            "alpha": MARKOV_ALPHA,
            "counts": {name: 0 for name in CLASS_NAMES},
        }

    last = history[-1]
    row = transitions[CLASS_TO_INDEX[last]]
    sample = int(row.sum())

    # Dirichlet alpha smoothing around realistic B/P/T priors.
    # Total pseudo-count strength is MARKOV_ALPHA.
    smoothed = row + prior * MARKOV_ALPHA
    probs = _normalize(smoothed, fallback=prior)

    return probs, {
        "last": last,
        "sample": sample,
        "alpha": MARKOV_ALPHA,
        "counts": {
            name: int(row[CLASS_TO_INDEX[name]]) for name in CLASS_NAMES
        },
        "transition_matrix": {
            source: {
                target: int(transitions[CLASS_TO_INDEX[source], CLASS_TO_INDEX[target]])
                for target in CLASS_NAMES
            }
            for source in CLASS_NAMES
        },
    }


# ---------------------------------------------------------------------------
# DeepSeek calibration
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
    # Common nested response shapes.
    for nested_key in ("probabilities", "probs", "prediction", "result", "distribution"):
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

    # Compatibility with the existing deepseek_client.py calibrate() output.
    adjustment_keys = ("banker_adjust", "player_adjust", "tie_adjust")
    if any(key in data for key in adjustment_keys):
        adjusted = np.asarray(base_probs, dtype=np.float64).copy()
        for index, key in enumerate(adjustment_keys):
            value = _safe_float(data.get(key, 0.0))
            adjusted[index] += _clamp(value or 0.0, -0.20, 0.20)
        return _normalize(adjusted, fallback=base_probs)

    # Last-resort support: recommendation + confidence.
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
    lstm_probs: Sequence[float],
    markov_probs: Sequence[float],
    local_probs: Sequence[float],
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]], str]:
    if not USE_DEEPSEEK:
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
        "sequence_length": LSTM_SEQUENCE_LENGTH,
        "lstm_probs": _to_prob_dict(lstm_probs, digits=6),
        "markov_probs": _to_prob_dict(markov_probs, digits=6),
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
# Fusion and public API
# ---------------------------------------------------------------------------
def _configured_weights() -> Dict[str, float]:
    normalized = _normalize(
        [LSTM_WEIGHT, MARKOV_WEIGHT, DEEPSEEK_WEIGHT],
        fallback=[0.55, 0.30, 0.15],
    )
    return {
        "lstm": float(normalized[0]),
        "markov": float(normalized[1]),
        "deepseek": float(normalized[2]),
    }


def _effective_weights(deepseek_available: bool) -> Dict[str, float]:
    weights = _configured_weights()
    if deepseek_available:
        return weights

    if DEEPSEEK_FALLBACK_MODE == "PRIOR":
        # Keep the configured DeepSeek weight; its component will use priors.
        return weights

    # Default REDISTRIBUTE: move unavailable DeepSeek weight to local models
    # while preserving the LSTM:Markov ratio.
    local_total = weights["lstm"] + weights["markov"]
    if local_total <= 0:
        return {"lstm": 0.5, "markov": 0.5, "deepseek": 0.0}
    return {
        "lstm": weights["lstm"] / local_total,
        "markov": weights["markov"] / local_total,
        "deepseek": 0.0,
    }


def _fusion(
    lstm_probs: Sequence[float],
    markov_probs: Sequence[float],
    deepseek_probs: Optional[Sequence[float]],
) -> Tuple[np.ndarray, Dict[str, float]]:
    deepseek_available = deepseek_probs is not None
    weights = _effective_weights(deepseek_available)
    ai_component = (
        _normalize(deepseek_probs, fallback=_prior_probs())
        if deepseek_available
        else _prior_probs()
    )

    final = (
        _normalize(lstm_probs, fallback=_prior_probs()) * weights["lstm"]
        + _normalize(markov_probs, fallback=_prior_probs()) * weights["markov"]
        + ai_component * weights["deepseek"]
    )
    return _normalize(final, fallback=_prior_probs()), weights


def fit_history(
    history: Union[str, Iterable[Any]],
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
    user_id: str = "",
    force: bool = True,
) -> Dict[str, Any]:
    """Explicitly fit/update the isolated LSTM model for one UID context."""
    cleaned = _clean_history(history)
    if REQUIRE_USER_ID and not str(user_id or "").strip():
        return {"ok": False, "error": "user_id is required when REQUIRE_USER_ID=1"}

    key = _make_training_key(user_id, venue, room, shoe_id)
    state = _get_model_state(key)
    with state.lock:
        if state.last_history and not _is_extension(state.last_history, cleaned):
            state.reset()
        result = _train_lstm_if_needed(state, cleaned, force=force)
        state.last_history = list(cleaned)
    return {
        "ok": True,
        "training_key": key,
        "user_id": user_id,
        "history_len": len(cleaned),
        "tf_available": TF_AVAILABLE,
        "lstm": result,
    }


def predict(
    history: Union[str, Iterable[Any]],
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
    user_id: str = "",
) -> Dict[str, Any]:
    """
    Predict the next B/P/T probabilities and recommend B or P.

    Every user gets an isolated model key:
        user_id | venue | room | shoe_id

    The caller must pass the LINE UID as ``user_id`` to guarantee isolation.
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

    try:
        with state.lock:
            # New shoe/history replacement under the same key: reset only this UID model.
            reset_detected = bool(
                state.last_history and not _is_extension(state.last_history, cleaned)
            )
            if reset_detected:
                state.reset()

            training_result = _train_lstm_if_needed(state, cleaned, force=False)
            lstm_probs, lstm_status = _predict_lstm(state, cleaned)
            state.last_history = list(cleaned)

        markov_probs, markov_info = _markov_probs(cleaned)

        local_weights = _normalize([LSTM_WEIGHT, MARKOV_WEIGHT], fallback=[0.55, 0.30])
        local_probs = _normalize(
            lstm_probs * local_weights[0] + markov_probs * local_weights[1],
            fallback=_prior_probs(),
        )

        ai_probs, ai_result, ai_status = _deepseek_probs(
            history=cleaned,
            user_id=uid,
            venue=venue,
            room=room,
            shoe_id=shoe_id,
            lstm_probs=lstm_probs,
            markov_probs=markov_probs,
            local_probs=local_probs,
        )

        final_probs, effective_weights = _fusion(
            lstm_probs=lstm_probs,
            markov_probs=markov_probs,
            deepseek_probs=ai_probs,
        )

        b_prob = float(final_probs[CLASS_TO_INDEX["B"]])
        p_prob = float(final_probs[CLASS_TO_INDEX["P"]])
        t_prob = float(final_probs[CLASS_TO_INDEX["T"]])

        # Requirement: recommendation is B or P. Tie probability is displayed but
        # never becomes the recommended betting side.
        recommend = "B" if b_prob >= p_prob else "P"
        recommend_text = "莊" if recommend == "B" else "閒"
        bp_total = max(1e-12, b_prob + p_prob)
        confidence = max(b_prob, p_prob) / bp_total
        edge = abs(b_prob - p_prob)
        signal_level = _signal_level(confidence, edge)

        component_probs = {
            "lstm": _to_prob_dict(lstm_probs, digits=6),
            "markov": _to_prob_dict(markov_probs, digits=6),
            "deepseek": _to_prob_dict(ai_probs, digits=6) if ai_probs is not None else None,
            "final": _to_prob_dict(final_probs, digits=6),
        }

        reason = (
            f"LSTM({effective_weights['lstm']:.2f}) + "
            f"Markov({effective_weights['markov']:.2f}) + "
            f"DeepSeek({effective_weights['deepseek']:.2f}); "
            f"LSTM={lstm_status}; DeepSeek={ai_status}; "
            f"莊閒方向信心={confidence * 100:.1f}%"
        )

        result: Dict[str, Any] = {
            "ok": True,
            "engine": "LSTM_MARKOV_DEEPSEEK",
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
            "pattern_label": "LSTM+Markov+DeepSeek 三分類融合",
            "regime": "LSTM_MARKOV_DEEPSEEK",
            "ngram_label": "",
            "ngram_sample": 0,
            "big_road_label": "大路 B/P/T 序列作為 LSTM 輸入",
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
            "direction_core": "LSTM_MARKOV_DEEPSEEK",
            "direction_locked": False,
            "reason": reason,
            "configured_weights": {
                key: round(value, 6) for key, value in _configured_weights().items()
            },
            "effective_weights": {
                key: round(value, 6) for key, value in effective_weights.items()
            },
            "dynamic_weights": {
                key: round(value, 6) for key, value in effective_weights.items()
            },
            "component_probs": component_probs,
            "markov": markov_info,
            "markov_label": f"最後一手{markov_info.get('last') or '無'}轉移樣本{markov_info.get('sample', 0)}",
            "ai_used": ai_probs is not None,
            "ai_status": ai_status,
            "ai_result": ai_result if DEBUG_AI_RESULT else None,
            "ml_trained": state.trained,
            "ml_samples": state.training_samples,
            "tf_available": TF_AVAILABLE,
            "tf_import_error": TF_IMPORT_ERROR if not TF_AVAILABLE else "",
            "lstm_status": lstm_status,
            "lstm_training": training_result,
            "lstm_sequence_length": LSTM_SEQUENCE_LENGTH,
            "lstm_units": LSTM_UNITS,
            "training_key": training_key,
            "model_cache_size": len(_MODEL_CACHE),
            # Compatibility with older app/debug displays.
            "ml_predictions": {
                "lr": 0.5,
                "rf": 0.5,
                "lstm": round(float(lstm_probs[CLASS_TO_INDEX["B"]]), 6),
                "ensemble": round(b_prob, 6),
                "lstm_probs": _to_prob_dict(lstm_probs, digits=6),
            },
        }

        if DEBUG_PREDICTOR:
            result["debug"] = {
                "cleaned_history": cleaned,
                "reset_detected": reset_detected,
                "state": {
                    "trained": state.trained,
                    "training_samples": state.training_samples,
                    "last_train_history_len": state.last_train_history_len,
                    "last_train_fingerprint": state.last_train_fingerprint,
                    "train_count": state.train_count,
                    "last_loss": state.last_loss,
                    "last_accuracy": state.last_accuracy,
                    "status": state.status,
                },
                "deepseek_import_error": _DEEPSEEK_IMPORT_ERROR,
            }
        else:
            result["debug"] = None

        return result

    except Exception as exc:
        logger.exception("Predictor failed for key=%s", training_key)
        fallback = _prior_probs()
        b_prob, p_prob, t_prob = [float(value) for value in fallback]
        recommend = "B" if b_prob >= p_prob else "P"
        bp_total = max(1e-12, b_prob + p_prob)
        confidence = max(b_prob, p_prob) / bp_total
        return {
            "ok": False,
            "error": str(exc),
            "engine": "LSTM_MARKOV_DEEPSEEK",
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
            "reason": "模型執行失敗，暫時使用 B/P/T 基礎先驗機率",
            "training_key": training_key,
            "tf_available": TF_AVAILABLE,
            "ai_used": False,
            "debug": {"exception": repr(exc)} if DEBUG_PREDICTOR else None,
        }
