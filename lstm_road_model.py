"""Short-shoe LSTM sequence model for Baccarat Big-Road B/P prediction.

Formal direction is owned by the LSTM family. Ties are skipped. Before enough
within-shoe labels exist, the module returns a deliberately weak, symmetric
sequence prior rather than handing direction to another model. Shoe state is not
used here; it is applied later to confidence / sizing only.
"""
from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
from threading import RLock
from typing import Any, Iterable, Mapping, Sequence
import math
import os

import numpy as np

MODEL_ID = "LSTM-SHORT-SHOE-BP-V2"
PAD_TOKEN = 0
B_TOKEN = 1
P_TOKEN = 2
VOCAB_SIZE = 3
WINDOW_SIZE = max(20, min(32, int(os.getenv("LSTM_ROAD_WINDOW", "24") or "24")))
MIN_HISTORY = max(14, min(24, int(os.getenv("LSTM_ROAD_MIN_HISTORY", "16") or "16")))
MIN_CONTEXT = max(4, min(8, int(os.getenv("LSTM_ROAD_MIN_CONTEXT", "5") or "5")))
MIN_ONLINE_SAMPLES = max(10, min(24, int(os.getenv("LSTM_ROAD_MIN_ONLINE_SAMPLES", "12") or "12")))
ONLINE_LEARNING_RATE = max(1e-6, min(1e-3, float(os.getenv("LSTM_ROAD_ONLINE_LR", "0.0001") or "0.0001")))
L2_REGULARIZATION = max(0.0, min(1e-2, float(os.getenv("LSTM_ROAD_L2", "0.0002") or "0.0002")))
BOOTSTRAP_EPOCHS = max(1, min(3, int(os.getenv("LSTM_ROAD_BOOTSTRAP_EPOCHS", "2") or "2")))
ONLINE_EPOCHS = 1
REPLAY_WINDOW = max(12, min(48, int(os.getenv("LSTM_ROAD_REPLAY_WINDOW", "24") or "24")))
RECENCY_DECAY = max(0.90, min(0.999, float(os.getenv("LSTM_ROAD_RECENCY_DECAY", "0.97") or "0.97")))
SOFTMAX_TEMPERATURE = max(1.0, min(2.0, float(os.getenv("LSTM_ROAD_TEMPERATURE", "1.20") or "1.20")))
MAX_IN_SHOE_CONFIDENCE = max(0.55, min(0.75, float(os.getenv("LSTM_ROAD_MAX_IN_SHOE_CONF", "0.66") or "0.66")))
MAX_COLD_START_CONFIDENCE = max(0.505, min(0.56, float(os.getenv("LSTM_ROAD_MAX_COLD_CONF", "0.535") or "0.535")))
MAX_SCOPE_MODELS = max(4, min(128, int(os.getenv("LSTM_ROAD_MAX_SCOPES", "32") or "32")))

_TF = None
_TF_IMPORT_ERROR: str | None = None
_CACHE_LOCK = RLock()
_MODEL_CACHE: "OrderedDict[str, LSTMRoadModel]" = OrderedDict()


def _clip(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(number):
        return lo
    return max(lo, min(hi, number))


def normalize_bp(history: str | Iterable[Any] | None) -> list[str]:
    if history is None:
        return []
    if isinstance(history, str):
        compact = history.replace("|", "").replace(",", "").replace(" ", "").upper()
        if compact and all(char in {"B", "P", "T"} for char in compact):
            return [char for char in compact if char in {"B", "P"}][-2000:]
        items: Iterable[Any] = [part for part in history.replace("|", ",").split(",") if part.strip()]
    else:
        items = history
    result: list[str] = []
    for item in items:
        if isinstance(item, Mapping):
            raw = item.get("outcome") or item.get("actual") or item.get("actual_outcome") or item.get("virtual_outcome")
        else:
            raw = item
        value = str(raw or "").upper().strip()
        if value in {"B", "P"}:
            result.append(value)
    return result[-2000:]


def _token(side: str) -> int:
    return B_TOKEN if str(side).upper() == "B" else P_TOKEN


def _encode_window(sequence: Sequence[str], window_size: int = WINDOW_SIZE) -> np.ndarray:
    width = max(8, int(window_size))
    tokens = [_token(side) for side in sequence[-width:]]
    padded = [PAD_TOKEN] * max(0, width - len(tokens)) + tokens
    return np.asarray(padded[-width:], dtype=np.int32)


def _training_examples(sequence: Sequence[str], *, window_size: int = WINDOW_SIZE, min_context: int = MIN_CONTEXT) -> tuple[np.ndarray, np.ndarray]:
    cleaned = [side for side in sequence if side in {"B", "P"}]
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for target_index in range(max(1, int(min_context)), len(cleaned)):
        xs.append(_encode_window(cleaned[:target_index], window_size))
        ys.append(0 if cleaned[target_index] == "B" else 1)
    if not xs:
        return np.empty((0, max(8, int(window_size))), dtype=np.int32), np.empty((0,), dtype=np.int32)
    return np.stack(xs).astype(np.int32), np.asarray(ys, dtype=np.int32)


def _sample_weights(labels: np.ndarray, *, recency_decay: float = RECENCY_DECAY) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int32)
    if y.size == 0:
        return np.empty((0,), dtype=np.float32)
    counts = {0: int(np.sum(y == 0)), 1: int(np.sum(y == 1))}
    n = float(len(y))
    class_weight = {cls: (n / (2.0 * count) if count > 0 else 1.0) for cls, count in counts.items()}
    class_weight = {cls: max(0.60, min(1.60, weight)) for cls, weight in class_weight.items()}
    ages = np.arange(len(y) - 1, -1, -1, dtype=np.float64)
    recency = np.power(float(recency_decay), ages)
    weights = np.asarray([class_weight[int(label)] for label in y], dtype=np.float64) * recency
    mean = float(np.mean(weights)) if weights.size else 1.0
    if mean > 1e-12:
        weights /= mean
    return weights.astype(np.float32)


def sequence_features(history: str | Iterable[Any] | None) -> dict[str, float | int]:
    seq = normalize_bp(history)
    if not seq:
        return {"rounds": 0, "current_run_length": 0, "recent_switch_rate": 0.5, "recent_b_ratio": 0.5}
    current = seq[-1]
    run = 1
    for side in reversed(seq[:-1]):
        if side != current:
            break
        run += 1
    recent = seq[-12:]
    switches = sum(left != right for left, right in zip(recent, recent[1:]))
    return {"rounds": len(seq), "current_run_length": int(run), "recent_switch_rate": float(switches / max(1, len(recent) - 1)), "recent_b_ratio": float(sum(side == "B" for side in recent) / max(1, len(recent)))}


def _cold_start_prior(sequence: Sequence[str]) -> dict[str, Any]:
    recent = [side for side in sequence[-12:] if side in {"B", "P"}]
    if not recent:
        p_b = 0.5
    else:
        b = sum(side == "B" for side in recent)
        posterior_b = (b + 2.0) / (len(recent) + 4.0)
        p_b = 0.5 + 0.28 * (posterior_b - 0.5)
    p_b = _clip(p_b, 1.0 - MAX_COLD_START_CONFIDENCE, MAX_COLD_START_CONFIDENCE)
    p_p = 1.0 - p_b
    direction = "B" if p_b >= p_p else "P"
    confidence = max(p_b, p_p)
    return {"direction": direction, "probabilities": {"B": float(p_b), "P": float(p_p)}, "confidence": float(confidence), "raw_confidence": float(confidence), "source": "lstm_cold_start_prior"}


def _temperature_scale(probabilities: Sequence[float], temperature: float) -> tuple[float, float]:
    p = np.asarray(probabilities, dtype=np.float64)
    p = np.clip(p, 1e-9, 1.0)
    logits = np.log(p) / max(1.0, float(temperature))
    logits -= np.max(logits)
    exp = np.exp(logits)
    total = float(np.sum(exp))
    if total <= 1e-12:
        return 0.5, 0.5
    scaled = exp / total
    return float(scaled[0]), float(scaled[1])


def _tensorflow():
    global _TF, _TF_IMPORT_ERROR
    if _TF is not None:
        return _TF
    if _TF_IMPORT_ERROR is not None:
        return None
    try:
        import tensorflow as tf
    except Exception as exc:
        _TF_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        return None
    _TF = tf
    return _TF


def tensorflow_status() -> dict[str, Any]:
    tf = _tensorflow()
    return {"available": tf is not None, "error": _TF_IMPORT_ERROR, "backend": "tensorflow.keras" if tf is not None else "unavailable"}


class LSTMRoadModel:
    def __init__(self, *, scope_key: str, window_size: int = WINDOW_SIZE, min_history: int = MIN_HISTORY, weight_path: str | None = None) -> None:
        self.scope_key = str(scope_key or "GLOBAL")
        self.window_size = max(20, min(32, int(window_size)))
        self.min_history = max(14, min(24, int(min_history)))
        self.weight_path = str(weight_path or os.getenv("LSTM_ROAD_MODEL_PATH", "") or "").strip()
        self._lock = RLock()
        self._model = None
        self._pretrained_loaded = False
        self._bootstrap_done = False
        self._trained_rounds = 0
        self._online_updates = 0
        self._load_error: str | None = None
        self._last_sequence: tuple[str, ...] = ()
        self._reset_count = 0

    def _seed(self) -> int:
        return int(sha256(self.scope_key.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF

    def _build_model(self):
        tf = _tensorflow()
        if tf is None:
            return None
        seed = self._seed()
        keras = tf.keras
        inputs = keras.Input(shape=(self.window_size,), dtype="int32", name="bp_window")
        x = keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=6, mask_zero=True, embeddings_initializer=keras.initializers.GlorotUniform(seed=seed), name="bp_embedding")(inputs)
        x = keras.layers.LSTM(24, dropout=0.10, recurrent_dropout=0.0, kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 1), recurrent_initializer=keras.initializers.Orthogonal(seed=seed + 2), bias_initializer="zeros", name="road_lstm")(x)
        x = keras.layers.Dense(10, activation="relu", kernel_regularizer=keras.regularizers.l2(L2_REGULARIZATION), kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 3), name="road_dense")(x)
        outputs = keras.layers.Dense(2, activation="softmax", kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 4), name="bp_softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name="bgs_short_shoe_lstm")
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=ONLINE_LEARNING_RATE, clipnorm=1.0), loss="sparse_categorical_crossentropy")
        return model

    def _reset_for_new_history(self) -> None:
        self._model = None
        self._pretrained_loaded = False
        self._bootstrap_done = False
        self._trained_rounds = 0
        self._online_updates = 0
        self._load_error = None
        self._last_sequence = ()
        self._reset_count += 1

    def _guard_history(self, sequence: Sequence[str]) -> None:
        current = tuple(sequence)
        if not self._last_sequence:
            return
        prefix_length = min(len(current), len(self._last_sequence))
        same_prefix = current[:prefix_length] == self._last_sequence[:prefix_length]
        if len(current) < len(self._last_sequence) or not same_prefix:
            self._reset_for_new_history()

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        self._model = self._build_model()
        if self._model is None:
            return None
        if self.weight_path:
            try:
                if os.path.exists(self.weight_path):
                    self._model.load_weights(self.weight_path)
                    self._pretrained_loaded = True
                    self._bootstrap_done = True
                else:
                    self._load_error = f"weights_not_found:{self.weight_path}"
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"
        return self._model

    def _bootstrap_online(self, sequence: Sequence[str]) -> bool:
        model = self._ensure_model()
        if model is None:
            return False
        if self._bootstrap_done:
            return True
        x_train, y_train = _training_examples(sequence, window_size=self.window_size, min_context=MIN_CONTEXT)
        if len(y_train) < MIN_ONLINE_SAMPLES:
            return False
        weights = _sample_weights(y_train)
        model.fit(x_train, y_train, sample_weight=weights, epochs=BOOTSTRAP_EPOCHS, batch_size=min(8, len(y_train)), shuffle=False, verbose=0)
        self._bootstrap_done = True
        self._trained_rounds = len(sequence)
        self._online_updates += int(len(y_train))
        return True

    def _update_newly_observed(self, sequence: Sequence[str]) -> None:
        if not self._bootstrap_done or len(sequence) <= self._trained_rounds:
            return
        x_all, y_all = _training_examples(sequence, window_size=self.window_size, min_context=MIN_CONTEXT)
        if len(y_all) <= 0:
            self._trained_rounds = len(sequence)
            return
        x_train = x_all[-REPLAY_WINDOW:]
        y_train = y_all[-REPLAY_WINDOW:]
        weights = _sample_weights(y_train)
        self._model.fit(x_train, y_train, sample_weight=weights, epochs=ONLINE_EPOCHS, batch_size=min(8, len(y_train)), shuffle=False, verbose=0)
        self._online_updates += int(max(1, len(sequence) - self._trained_rounds))
        self._trained_rounds = len(sequence)

    def predict(self, history: str | Iterable[Any] | None, *, allow_online_update: bool = True) -> dict[str, Any]:
        sequence = normalize_bp(history)
        n = len(sequence)
        cold = _cold_start_prior(sequence)
        base = {"model_id": MODEL_ID, "available": True, "direction": str(cold["direction"]), "probabilities": dict(cold["probabilities"]), "confidence": float(cold["confidence"]), "raw_confidence": float(cold["raw_confidence"]), "window_size": self.window_size, "min_history": self.min_history, "min_online_samples": int(MIN_ONLINE_SAMPLES), "sequence_length": n, "pretrained_loaded": bool(self._pretrained_loaded), "online_updates": int(self._online_updates), "weight_path": self.weight_path or None, "weight_load_error": self._load_error, "features": sequence_features(sequence), "reset_count": int(self._reset_count), "cold_start_prior": dict(cold), "semantics": "short_shoe_lstm_BP_only_with_weak_internal_cold_start_prior"}
        with self._lock:
            self._guard_history(sequence)
            if n < self.min_history:
                self._last_sequence = tuple(sequence)
                base["reason"] = "lstm_warmup_internal_prior"
                base["training_ready"] = False
                return base
            model = self._ensure_model()
            if model is None:
                self._last_sequence = tuple(sequence)
                base["reason"] = "tensorflow_unavailable_internal_prior"
                base["tensorflow"] = tensorflow_status()
                base["training_ready"] = False
                return base
            try:
                if not self._pretrained_loaded and not self._bootstrap_online(sequence):
                    self._last_sequence = tuple(sequence)
                    base["reason"] = "lstm_waiting_for_balanced_training_sample_floor"
                    base["training_ready"] = False
                    return base
                if self._pretrained_loaded and self._trained_rounds == 0:
                    self._trained_rounds = min(len(sequence), MIN_CONTEXT)
                if allow_online_update:
                    self._update_newly_observed(sequence)
                raw = model(_encode_window(sequence, self.window_size)[None, :], training=False).numpy()[0]
                p_b, p_p = _temperature_scale(raw, SOFTMAX_TEMPERATURE)
                raw_direction = "B" if p_b >= p_p else "P"
                raw_confidence = max(p_b, p_p)
                maturity = _clip((n - self.min_history + 1) / max(1.0, float(self.window_size - self.min_history + 1)))
                maturity_scale = 0.45 + 0.55 * maturity
                confidence_cap = 0.75 if self._pretrained_loaded else MAX_IN_SHOE_CONFIDENCE
                confidence = min(confidence_cap, 0.5 + (raw_confidence - 0.5) * maturity_scale)
                direction = raw_direction
                final_b = confidence if direction == "B" else 1.0 - confidence
                final_p = 1.0 - final_b
                self._last_sequence = tuple(sequence)
                return {**base, "available": True, "direction": direction, "probabilities": {"B": float(final_b), "P": float(final_p)}, "confidence": float(confidence), "raw_confidence": float(raw_confidence), "raw_probabilities": {"B": float(p_b), "P": float(p_p)}, "maturity": float(maturity), "maturity_scale": float(maturity_scale), "pretrained_loaded": bool(self._pretrained_loaded), "online_updates": int(self._online_updates), "trained_rounds": int(self._trained_rounds), "training_ready": True, "reason": "short_shoe_lstm_available"}
            except Exception as exc:
                self._last_sequence = tuple(sequence)
                base["reason"] = "lstm_inference_failed_internal_prior"
                base["error"] = f"{type(exc).__name__}: {exc}"
                base["training_ready"] = False
                return base


def _cache_key(scope_key: str) -> str:
    return sha256(str(scope_key or "GLOBAL").encode("utf-8")).hexdigest()[:24]


def get_lstm_road_model(scope_key: str = "") -> LSTMRoadModel:
    key = _cache_key(scope_key)
    with _CACHE_LOCK:
        model = _MODEL_CACHE.pop(key, None)
        if model is None:
            model = LSTMRoadModel(scope_key=scope_key or key)
        _MODEL_CACHE[key] = model
        while len(_MODEL_CACHE) > MAX_SCOPE_MODELS:
            _MODEL_CACHE.popitem(last=False)
        return model


def predict_lstm_road(history: str | Iterable[Any] | None, *, scope_key: str = "", allow_online_update: bool = True) -> dict[str, Any]:
    return get_lstm_road_model(scope_key).predict(history, allow_online_update=allow_online_update)


def clear_lstm_model_cache() -> None:
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()


__all__ = ["MODEL_ID", "WINDOW_SIZE", "MIN_HISTORY", "MIN_CONTEXT", "MIN_ONLINE_SAMPLES", "ONLINE_LEARNING_RATE", "L2_REGULARIZATION", "REPLAY_WINDOW", "RECENCY_DECAY", "SOFTMAX_TEMPERATURE", "MAX_IN_SHOE_CONFIDENCE", "MAX_COLD_START_CONFIDENCE", "normalize_bp", "sequence_features", "tensorflow_status", "LSTMRoadModel", "get_lstm_road_model", "predict_lstm_road", "clear_lstm_model_cache"]
