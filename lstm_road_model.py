"""Lightweight LSTM sequence model for Baccarat Big-Road B/P prediction.

The LSTM is the primary directional model once available. Ties are skipped.
It supports optional pretrained weights plus conservative in-shoe online fitting.
Shoe composition is never used here to choose B/P direction.
"""
from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
from threading import RLock
from typing import Any, Iterable, Mapping, Sequence
import math
import os

import numpy as np

MODEL_ID = "LSTM-ROAD-BP-V1"
B_TOKEN = 0
P_TOKEN = 1
PAD_TOKEN = 2
VOCAB_SIZE = 3
WINDOW_SIZE = max(16, min(32, int(os.getenv("LSTM_ROAD_WINDOW", "20") or "20")))
MIN_HISTORY = max(8, min(12, int(os.getenv("LSTM_ROAD_MIN_HISTORY", "10") or "10")))
MIN_CONTEXT = max(3, min(8, int(os.getenv("LSTM_ROAD_MIN_CONTEXT", "4") or "4")))
MIN_ONLINE_SAMPLES = max(4, int(os.getenv("LSTM_ROAD_MIN_ONLINE_SAMPLES", "6") or "6"))
ONLINE_LEARNING_RATE = max(1e-6, min(5e-3, float(os.getenv("LSTM_ROAD_ONLINE_LR", "0.0002") or "0.0002")))
L2_REGULARIZATION = max(0.0, min(1e-2, float(os.getenv("LSTM_ROAD_L2", "0.0001") or "0.0001")))
BOOTSTRAP_EPOCHS = max(1, min(4, int(os.getenv("LSTM_ROAD_BOOTSTRAP_EPOCHS", "2") or "2")))
ONLINE_EPOCHS = 1
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
    width = max(4, int(window_size))
    tokens = [_token(side) for side in sequence[-width:]]
    padded = [PAD_TOKEN] * max(0, width - len(tokens)) + tokens
    return np.asarray(padded[-width:], dtype=np.int32)


def _training_examples(sequence: Sequence[str], *, window_size: int = WINDOW_SIZE, min_context: int = MIN_CONTEXT) -> tuple[np.ndarray, np.ndarray]:
    cleaned = [side for side in sequence if side in {"B", "P"}]
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for target_index in range(max(1, int(min_context)), len(cleaned)):
        xs.append(_encode_window(cleaned[:target_index], window_size))
        ys.append(_token(cleaned[target_index]))
    if not xs:
        return np.empty((0, max(4, int(window_size))), dtype=np.int32), np.empty((0,), dtype=np.int32)
    return np.stack(xs).astype(np.int32), np.asarray(ys, dtype=np.int32)


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
    return {
        "rounds": len(seq),
        "current_run_length": int(run),
        "recent_switch_rate": float(switches / max(1, len(recent) - 1)),
        "recent_b_ratio": float(sum(side == "B" for side in recent) / max(1, len(recent))),
    }


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
        self.window_size = max(16, min(32, int(window_size)))
        self.min_history = max(8, min(12, int(min_history)))
        self.weight_path = str(weight_path or os.getenv("LSTM_ROAD_MODEL_PATH", "") or "").strip()
        self._lock = RLock()
        self._model = None
        self._pretrained_loaded = False
        self._bootstrap_done = False
        self._trained_rounds = 0
        self._online_updates = 0
        self._load_error: str | None = None

    def _seed(self) -> int:
        return int(sha256(self.scope_key.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF

    def _build_model(self):
        tf = _tensorflow()
        if tf is None:
            return None
        seed = self._seed()
        keras = tf.keras
        inputs = keras.Input(shape=(self.window_size,), dtype="int32", name="bp_window")
        x = keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=4, embeddings_initializer=keras.initializers.GlorotUniform(seed=seed), name="bp_embedding")(inputs)
        x = keras.layers.LSTM(
            16,
            dropout=0.15,
            recurrent_dropout=0.0,
            kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 1),
            recurrent_initializer=keras.initializers.Orthogonal(seed=seed + 2),
            bias_initializer="zeros",
            name="road_lstm",
        )(x)
        x = keras.layers.Dense(
            8,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(L2_REGULARIZATION),
            kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 3),
            name="road_dense",
        )(x)
        outputs = keras.layers.Dense(2, activation="softmax", kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 4), name="bp_softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name="bgs_lstm_road")
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=ONLINE_LEARNING_RATE, clipnorm=1.0), loss="sparse_categorical_crossentropy")
        return model

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
        model.fit(x_train, y_train, epochs=BOOTSTRAP_EPOCHS, batch_size=min(8, len(y_train)), shuffle=False, verbose=0)
        self._bootstrap_done = True
        self._trained_rounds = len(sequence)
        self._online_updates += int(len(y_train))
        return True

    def _update_newly_observed(self, sequence: Sequence[str]) -> None:
        if not self._bootstrap_done or len(sequence) <= self._trained_rounds:
            return
        first_target = max(MIN_CONTEXT, self._trained_rounds)
        xs: list[np.ndarray] = []
        ys: list[int] = []
        for target_index in range(first_target, len(sequence)):
            xs.append(_encode_window(sequence[:target_index], self.window_size))
            ys.append(_token(sequence[target_index]))
        if xs:
            x_train = np.stack(xs).astype(np.int32)
            y_train = np.asarray(ys, dtype=np.int32)
            self._model.fit(x_train, y_train, epochs=ONLINE_EPOCHS, batch_size=min(4, len(y_train)), shuffle=False, verbose=0)
            self._online_updates += int(len(y_train))
        self._trained_rounds = len(sequence)

    def predict(self, history: str | Iterable[Any] | None, *, allow_online_update: bool = True) -> dict[str, Any]:
        sequence = normalize_bp(history)
        n = len(sequence)
        base = {
            "model_id": MODEL_ID,
            "available": False,
            "direction": None,
            "probabilities": {"B": 0.5, "P": 0.5},
            "confidence": 0.5,
            "raw_confidence": 0.5,
            "window_size": self.window_size,
            "min_history": self.min_history,
            "sequence_length": n,
            "pretrained_loaded": bool(self._pretrained_loaded),
            "online_updates": int(self._online_updates),
            "weight_path": self.weight_path or None,
            "weight_load_error": self._load_error,
            "features": sequence_features(sequence),
            "semantics": "lstm_softmax_over_resolved_big_road_BP_only",
        }
        if n < self.min_history:
            base["reason"] = "cold_start_insufficient_history"
            return base
        with self._lock:
            model = self._ensure_model()
            if model is None:
                base["reason"] = "tensorflow_unavailable"
                base["tensorflow"] = tensorflow_status()
                return base
            try:
                if not self._pretrained_loaded:
                    if not self._bootstrap_online(sequence):
                        base["reason"] = "insufficient_online_training_samples"
                        return base
                elif self._trained_rounds == 0:
                    self._trained_rounds = MIN_CONTEXT
                if allow_online_update:
                    self._update_newly_observed(sequence)
                raw = model(_encode_window(sequence, self.window_size)[None, :], training=False).numpy()[0]
                p_b = _clip(raw[B_TOKEN])
                p_p = _clip(raw[P_TOKEN])
                total = p_b + p_p
                if total <= 1e-12:
                    p_b = p_p = 0.5
                else:
                    p_b, p_p = p_b / total, p_p / total
                direction = "B" if p_b >= p_p else "P"
                confidence = p_b if direction == "B" else p_p
                maturity = _clip((n - self.min_history + 1) / max(1.0, float(self.window_size - self.min_history + 1)))
                return {
                    **base,
                    "available": True,
                    "direction": direction,
                    "probabilities": {"B": float(p_b), "P": float(p_p)},
                    "confidence": float(confidence),
                    "raw_confidence": float(confidence),
                    "maturity": float(maturity),
                    "pretrained_loaded": bool(self._pretrained_loaded),
                    "online_updates": int(self._online_updates),
                    "trained_rounds": int(self._trained_rounds),
                    "reason": "pretrained_or_online_lstm_available",
                }
            except Exception as exc:
                base["reason"] = "lstm_inference_failed"
                base["error"] = f"{type(exc).__name__}: {exc}"
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


__all__ = [
    "MODEL_ID",
    "WINDOW_SIZE",
    "MIN_HISTORY",
    "MIN_CONTEXT",
    "MIN_ONLINE_SAMPLES",
    "ONLINE_LEARNING_RATE",
    "L2_REGULARIZATION",
    "normalize_bp",
    "sequence_features",
    "tensorflow_status",
    "LSTMRoadModel",
    "get_lstm_road_model",
    "predict_lstm_road",
    "clear_lstm_model_cache",
]
