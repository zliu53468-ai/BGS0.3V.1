"""Production LSTM + exact-shoe + cut-depth fusion for baccarat B/P.

Design goals
------------
* 50-70 hand baccarat shoe.
* Big-Road resolved B/P sequence is learned by a compact LSTM.
* Exact remaining shoe composition contributes a physical non-replacement B/P
  logit inside the same final fusion decision.
* Cut-card depth has no artificial B/P sign.  It changes how much the fusion
  trusts the sequence branch versus exact composition as the shoe approaches
  the configured cut point.
* Ties are skipped for sequence learning.
* No Markov/LinUCB/HSMM/hazard model is required for formal direction.
* The model is safe on cold start: zero-initialized neural head, balanced replay,
  PAD masking, prefix-reset protection, and bounded logits prevent one-sided
  lock-in from a handful of early results.

This cannot guarantee profitable baccarat prediction.  The implementation is
intended to be stable, inspectable, and deployable with short-shoe data.
"""
from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
from threading import RLock
from typing import Any, Iterable, Mapping, Sequence
import math
import os

import numpy as np

from shoe_composition import STANDARD_EIGHT_DECK_BASELINE, analyze_shoe_composition
from shoe_constants import SHOE_DECKS
from shoe_depth_estimator import (
    DEFAULT_CUT_CARD_REMAINING,
    TARGET_HANDS_MAX,
    TARGET_HANDS_MIN,
    ShoeDepthEstimator,
    build_shoe_depth_features,
)

MODEL_ID = "LSTM-SHOE-CUT-FUSION-V2"
PAD_TOKEN = 0
B_TOKEN = 1
P_TOKEN = 2
VOCAB_SIZE = 3
WINDOW_SIZE = max(18, min(32, int(os.getenv("LSTM_ROAD_WINDOW", "24") or "24")))
MIN_HISTORY = max(14, min(20, int(os.getenv("LSTM_ROAD_MIN_HISTORY", "16") or "16")))
MIN_CONTEXT = max(4, min(10, int(os.getenv("LSTM_ROAD_MIN_CONTEXT", "6") or "6")))
MIN_ONLINE_SAMPLES = max(8, int(os.getenv("LSTM_ROAD_MIN_ONLINE_SAMPLES", "10") or "10"))
FULL_MATURITY_ROUNDS = max(
    MIN_HISTORY + 8,
    min(50, int(os.getenv("LSTM_ROAD_FULL_MATURITY", "36") or "36")),
)
REPLAY_WINDOW = max(20, min(56, int(os.getenv("LSTM_ROAD_REPLAY_WINDOW", "36") or "36")))
ONLINE_LEARNING_RATE = max(
    1e-6,
    min(2e-3, float(os.getenv("LSTM_ROAD_ONLINE_LR", "0.00015") or "0.00015")),
)
L2_REGULARIZATION = max(
    0.0,
    min(1e-2, float(os.getenv("LSTM_ROAD_L2", "0.0002") or "0.0002")),
)
BOOTSTRAP_EPOCHS = max(1, min(3, int(os.getenv("LSTM_ROAD_BOOTSTRAP_EPOCHS", "2") or "2")))
ONLINE_EPOCHS = 1
MAX_SCOPE_MODELS = max(4, min(128, int(os.getenv("LSTM_ROAD_MAX_SCOPES", "32") or "32")))
MAX_NEURAL_LOGIT = 0.70
MAX_SHOE_DEVIATION_LOGIT = 0.55
MAX_STRUCTURE_LOGIT = 0.28
MAX_FINAL_LOGIT = 0.50

# Resolve the standard baccarat physical prior over B/P only.  This is tiny and
# prevents an arbitrary Player tie-break when every other signal is neutral.
_BASE_B = float(STANDARD_EIGHT_DECK_BASELINE["B"])
_BASE_P = float(STANDARD_EIGHT_DECK_BASELINE["P"])
_BASE_RESOLVED_B = _BASE_B / (_BASE_B + _BASE_P)
BASE_RESOLVED_LOGIT = math.log(_BASE_RESOLVED_B / (1.0 - _BASE_RESOLVED_B))

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


def _safe_logit(probability: float) -> float:
    p = _clip(probability, 1e-6, 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _sigmoid(logit: float) -> float:
    value = max(-20.0, min(20.0, float(logit)))
    return 1.0 / (1.0 + math.exp(-value))


def normalize_bp(history: str | Iterable[Any] | None) -> list[str]:
    if history is None:
        return []
    if isinstance(history, str):
        compact = (
            history.replace("|", "")
            .replace(",", "")
            .replace(" ", "")
            .upper()
        )
        if compact and all(char in {"B", "P", "T"} for char in compact):
            return [char for char in compact if char in {"B", "P"}][-2000:]
        items: Iterable[Any] = [
            part for part in history.replace("|", ",").split(",") if part.strip()
        ]
    else:
        items = history
    result: list[str] = []
    for item in items:
        if isinstance(item, Mapping):
            raw = (
                item.get("outcome")
                or item.get("actual")
                or item.get("actual_outcome")
                or item.get("virtual_outcome")
            )
        else:
            raw = item
        value = str(raw or "").upper().strip()
        if value in {"B", "P"}:
            result.append(value)
    return result[-2000:]


def _token(side: str) -> int:
    return B_TOKEN if str(side).upper() == "B" else P_TOKEN


def _encode_window(
    sequence: Sequence[str],
    window_size: int = WINDOW_SIZE,
) -> np.ndarray:
    width = max(8, int(window_size))
    tokens = [_token(side) for side in sequence[-width:]]
    padded = [PAD_TOKEN] * max(0, width - len(tokens)) + tokens
    return np.asarray(padded[-width:], dtype=np.int32)


def _training_examples(
    sequence: Sequence[str],
    *,
    window_size: int = WINDOW_SIZE,
    min_context: int = MIN_CONTEXT,
    replay_window: int = REPLAY_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    cleaned = [side for side in sequence if side in {"B", "P"}]
    first_target = max(1, int(min_context))
    start_target = max(first_target, len(cleaned) - max(8, int(replay_window)))
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for target_index in range(start_target, len(cleaned)):
        xs.append(_encode_window(cleaned[:target_index], window_size))
        ys.append(0 if cleaned[target_index] == "B" else 1)
    if not xs:
        return (
            np.empty((0, max(8, int(window_size))), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )
    return np.stack(xs).astype(np.int32), np.asarray(ys, dtype=np.int32)


def _balanced_sample_weights(labels: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    labels = np.asarray(labels, dtype=np.int32)
    if labels.size == 0:
        return np.empty((0,), dtype=np.float32), {
            "b_samples": 0,
            "p_samples": 0,
            "balanced": False,
        }
    b_count = int(np.sum(labels == 0))
    p_count = int(np.sum(labels == 1))
    if b_count <= 0 or p_count <= 0:
        return np.ones(labels.shape, dtype=np.float32), {
            "b_samples": b_count,
            "p_samples": p_count,
            "balanced": False,
        }
    total = float(labels.size)
    b_weight = _clip(total / (2.0 * b_count), 0.55, 2.20)
    p_weight = _clip(total / (2.0 * p_count), 0.55, 2.20)
    weights = np.where(labels == 0, b_weight, p_weight).astype(np.float32)
    return weights, {
        "b_samples": b_count,
        "p_samples": p_count,
        "b_sample_weight": float(b_weight),
        "p_sample_weight": float(p_weight),
        "balanced": True,
    }


def sequence_features(history: str | Iterable[Any] | None) -> dict[str, float | int | str]:
    seq = normalize_bp(history)
    if not seq:
        return {
            "rounds": 0,
            "current_run_length": 0,
            "recent_switch_rate": 0.5,
            "recent_b_ratio": 0.5,
            "recent_p_ratio": 0.5,
            "last_side": "",
            "structure_logit": 0.0,
        }
    current = seq[-1]
    run = 1
    for side in reversed(seq[:-1]):
        if side != current:
            break
        run += 1
    recent = seq[-12:]
    switches = sum(left != right for left, right in zip(recent, recent[1:]))
    switch_rate = switches / max(1, len(recent) - 1)
    b_ratio = sum(side == "B" for side in recent) / max(1, len(recent))
    side_sign = 1.0 if current == "B" else -1.0
    imbalance = 2.0 * (b_ratio - 0.5)
    persistence = side_sign * (1.0 - 2.0 * switch_rate)
    run_term = side_sign * min(1.0, run / 5.0)
    structure_logit = max(
        -MAX_STRUCTURE_LOGIT,
        min(
            MAX_STRUCTURE_LOGIT,
            0.16 * imbalance + 0.08 * persistence + 0.05 * run_term,
        ),
    )
    return {
        "rounds": len(seq),
        "current_run_length": int(run),
        "recent_switch_rate": float(switch_rate),
        "recent_b_ratio": float(b_ratio),
        "recent_p_ratio": float(1.0 - b_ratio),
        "last_side": current,
        "structure_logit": float(structure_logit),
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
    return {
        "available": tf is not None,
        "error": _TF_IMPORT_ERROR,
        "backend": "tensorflow.keras" if tf is not None else "unavailable",
    }


def _shoe_fusion_state(
    history: Sequence[str],
    shoe_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    context = dict(shoe_context or {})
    try:
        decks = max(1, min(16, int(context.get("decks", SHOE_DECKS) or SHOE_DECKS)))
    except (TypeError, ValueError):
        decks = int(SHOE_DECKS)
    cut_override = context.get("cut_card_remaining_cards")
    exact = dict(analyze_shoe_composition(context))
    exact_available = bool(exact.get("available"))

    if exact_available:
        remaining = float(exact.get("remaining_cards", 0.0) or 0.0)
        reliability = 1.0
        depth_source = str(exact.get("source") or "exact_composition")
    else:
        supplied = context.get("remaining_cards")
        try:
            supplied_remaining = float(supplied) if supplied is not None else None
        except (TypeError, ValueError):
            supplied_remaining = None
        if supplied_remaining is not None and supplied_remaining >= 0.0:
            remaining = supplied_remaining
            reliability = _clip(context.get("remaining_cards_reliability", 0.70))
            depth_source = str(context.get("remaining_cards_source") or "supplied_remaining_cards")
        else:
            estimate = ShoeDepthEstimator(
                shoe_decks=decks,
                cut_card_remaining_cards=cut_override,
            ).estimate(history).as_dict()
            remaining = float(estimate["remaining_cards"])
            reliability = _clip(estimate.get("remaining_cards_reliability", 0.60))
            depth_source = "round_count_estimate"

    depth = build_shoe_depth_features(
        remaining,
        shoe_decks=decks,
        reliability=reliability,
        source=depth_source,
        hand_count=len(history),
        cut_card_remaining_cards=cut_override,
    )

    shoe_logit = 0.0
    shoe_probability_b = None
    shoe_probability_p = None
    shoe_direction = None
    if exact_available:
        probabilities = dict(exact.get("probabilities") or {})
        p_b = max(1e-9, float(probabilities.get("B", 0.0) or 0.0))
        p_p = max(1e-9, float(probabilities.get("P", 0.0) or 0.0))
        bp_mass = p_b + p_p
        if bp_mass > 1e-12:
            resolved_b = p_b / bp_mass
            raw_logit = _safe_logit(resolved_b)
            shoe_logit = max(
                -MAX_SHOE_DEVIATION_LOGIT,
                min(MAX_SHOE_DEVIATION_LOGIT, raw_logit - BASE_RESOLVED_LOGIT),
            )
            shoe_probability_b = float(resolved_b)
            shoe_probability_p = float(1.0 - resolved_b)
            shoe_direction = "B" if resolved_b >= 0.5 else "P"

    return {
        "exact_composition_available": exact_available,
        "exact_composition_source": str(exact.get("source") or "none"),
        "remaining_cards": float(remaining),
        "remaining_ratio": float(depth["remaining_ratio"]),
        "penetration": float(depth["penetration"]),
        "shoe_stage": str(depth["shoe_stage"]),
        "remaining_cards_reliability": float(reliability),
        "cut_card_remaining_cards": float(depth["cut_card_remaining_cards"]),
        "cut_progress": float(depth["cut_progress"]),
        "cut_proximity": float(depth["cut_proximity"]),
        "cards_until_cut": float(depth["cards_until_cut"]),
        "estimated_hands_until_cut": float(depth["estimated_hands_until_cut"]),
        "target_hands_min": int(TARGET_HANDS_MIN),
        "target_hands_max": int(TARGET_HANDS_MAX),
        "shoe_resolved_probability_b": shoe_probability_b,
        "shoe_resolved_probability_p": shoe_probability_p,
        "shoe_direction": shoe_direction,
        "shoe_logit_deviation": float(shoe_logit),
        "shoe_analysis": exact,
        "depth_feature_source": depth_source,
    }


class LSTMRoadModel:
    """Per-shoe compact LSTM with deterministic physical late fusion."""

    def __init__(
        self,
        *,
        scope_key: str,
        window_size: int = WINDOW_SIZE,
        min_history: int = MIN_HISTORY,
        weight_path: str | None = None,
    ) -> None:
        self.scope_key = str(scope_key or "GLOBAL")
        self.window_size = max(18, min(32, int(window_size)))
        self.min_history = max(14, min(20, int(min_history)))
        self.weight_path = str(
            weight_path or os.getenv("LSTM_ROAD_MODEL_PATH", "") or ""
        ).strip()
        self._lock = RLock()
        self._model = None
        self._pretrained_loaded = False
        self._bootstrap_done = False
        self._trained_rounds = 0
        self._online_updates = 0
        self._load_error: str | None = None
        self._last_sequence: list[str] = []
        self._reset_count = 0
        self._last_reset_reason = "initial"
        self._training_balance: dict[str, Any] = {}

    def _seed(self) -> int:
        return int(
            sha256(self.scope_key.encode("utf-8")).hexdigest()[:8],
            16,
        ) & 0x7FFFFFFF

    def _build_model(self):
        tf = _tensorflow()
        if tf is None:
            return None
        seed = self._seed()
        keras = tf.keras
        inputs = keras.Input(
            shape=(self.window_size,),
            dtype="int32",
            name="bp_window",
        )
        # PAD=0 is masked, so early 16-24 hand windows do not teach the model
        # that padding is a real baccarat state.
        x = keras.layers.Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=6,
            mask_zero=True,
            embeddings_initializer=keras.initializers.GlorotUniform(seed=seed),
            name="bp_embedding",
        )(inputs)
        x = keras.layers.LSTM(
            20,
            dropout=0.12,
            recurrent_dropout=0.0,
            kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 1),
            recurrent_initializer=keras.initializers.Orthogonal(seed=seed + 2),
            bias_initializer="zeros",
            name="road_lstm",
        )(x)
        x = keras.layers.Dense(
            10,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(L2_REGULARIZATION),
            kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 3),
            name="road_dense",
        )(x)
        # Zero head is intentional: an untrained new shoe starts at 50/50
        # instead of inheriting a random B/P bias from initialization.
        outputs = keras.layers.Dense(
            2,
            activation="softmax",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="bp_softmax",
        )(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name="bgs_lstm_shoe_cut")
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=ONLINE_LEARNING_RATE,
                clipnorm=1.0,
            ),
            loss="sparse_categorical_crossentropy",
        )
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

    def _reset_for_history(self, reason: str) -> None:
        self._model = None
        self._pretrained_loaded = False
        self._bootstrap_done = False
        self._trained_rounds = 0
        self._online_updates = 0
        self._last_sequence = []
        self._training_balance = {}
        self._reset_count += 1
        self._last_reset_reason = str(reason)

    def _ensure_history_alignment(self, sequence: Sequence[str]) -> None:
        current = list(sequence)
        previous = list(self._last_sequence)
        if previous:
            same_prefix = (
                len(current) >= len(previous)
                and current[: len(previous)] == previous
            )
            same_exact = current == previous
            if not same_exact and not same_prefix:
                self._reset_for_history("history_prefix_changed_or_new_shoe")
        self._last_sequence = current

    def _fit_replay(self, sequence: Sequence[str], *, epochs: int) -> bool:
        model = self._ensure_model()
        if model is None:
            return False
        x_train, y_train = _training_examples(
            sequence,
            window_size=self.window_size,
            min_context=MIN_CONTEXT,
            replay_window=REPLAY_WINDOW,
        )
        if len(y_train) < MIN_ONLINE_SAMPLES:
            self._training_balance = {
                "sample_count": int(len(y_train)),
                "balanced": False,
                "reason": "insufficient_samples",
            }
            return False
        sample_weights, balance = _balanced_sample_weights(y_train)
        self._training_balance = {
            **balance,
            "sample_count": int(len(y_train)),
        }
        # Do not fit a one-class replay window.  This is the key anti-lock rule:
        # a short Player-heavy opening cannot train the whole network into P.
        if not bool(balance.get("balanced")):
            self._training_balance["reason"] = "single_class_replay_skipped"
            return False
        model.fit(
            x_train,
            y_train,
            sample_weight=sample_weights,
            epochs=max(1, int(epochs)),
            batch_size=min(8, len(y_train)),
            shuffle=False,
            verbose=0,
        )
        self._online_updates += int(len(y_train))
        self._trained_rounds = len(sequence)
        return True

    def _neural_state(
        self,
        sequence: Sequence[str],
        *,
        allow_online_update: bool,
    ) -> dict[str, Any]:
        n = len(sequence)
        model = self._ensure_model()
        if model is None:
            return {
                "available": False,
                "probability_b": 0.5,
                "probability_p": 0.5,
                "logit": 0.0,
                "maturity": 0.0,
                "reason": "tensorflow_unavailable",
            }

        if n >= self.min_history and not self._bootstrap_done:
            trained = self._fit_replay(sequence, epochs=BOOTSTRAP_EPOCHS)
            self._bootstrap_done = bool(trained or self._pretrained_loaded)
        elif (
            allow_online_update
            and self._bootstrap_done
            and n > self._trained_rounds
        ):
            self._fit_replay(sequence, epochs=ONLINE_EPOCHS)

        if not self._bootstrap_done:
            return {
                "available": False,
                "probability_b": 0.5,
                "probability_p": 0.5,
                "logit": 0.0,
                "maturity": 0.0,
                "reason": (
                    "cold_start_balanced_replay_not_ready"
                    if n < self.min_history
                    else "balanced_replay_not_ready"
                ),
            }

        raw = model(
            _encode_window(sequence, self.window_size)[None, :],
            training=False,
        ).numpy()[0]
        p_b = _clip(raw[0], 1e-6, 1.0 - 1e-6)
        p_p = _clip(raw[1], 1e-6, 1.0 - 1e-6)
        total = p_b + p_p
        p_b, p_p = p_b / total, p_p / total
        neural_logit = max(
            -MAX_NEURAL_LOGIT,
            min(MAX_NEURAL_LOGIT, _safe_logit(p_b)),
        )
        maturity = _clip(
            (n - self.min_history + 1)
            / max(1.0, float(FULL_MATURITY_ROUNDS - self.min_history + 1))
        )
        if self._pretrained_loaded:
            maturity = max(0.65, maturity)
        return {
            "available": True,
            "probability_b": float(p_b),
            "probability_p": float(p_p),
            "logit": float(neural_logit),
            "maturity": float(maturity),
            "reason": "balanced_online_lstm_available",
        }

    def predict(
        self,
        history: str | Iterable[Any] | None,
        *,
        shoe_context: Mapping[str, Any] | None = None,
        allow_online_update: bool = True,
    ) -> dict[str, Any]:
        sequence = normalize_bp(history)
        features = sequence_features(sequence)
        shoe = _shoe_fusion_state(sequence, shoe_context)

        with self._lock:
            self._ensure_history_alignment(sequence)
            neural = self._neural_state(
                sequence,
                allow_online_update=allow_online_update,
            )

        maturity = float(neural.get("maturity", 0.0) or 0.0)
        cut = _clip(shoe.get("cut_progress", 0.0))
        exact_available = bool(shoe.get("exact_composition_available"))
        neural_logit = float(neural.get("logit", 0.0) or 0.0)
        shoe_logit = float(shoe.get("shoe_logit_deviation", 0.0) or 0.0)
        structure_logit = float(features.get("structure_logit", 0.0) or 0.0)

        # As the configured cut point approaches, sequence pattern weight tapers
        # while exact composition receives a modestly larger role.  Cut depth
        # itself never has a positive/negative B/P sign.
        lstm_weight = maturity * (0.95 - 0.25 * cut)
        shoe_weight = (
            (0.50 + 0.25 * cut)
            * float(shoe.get("remaining_cards_reliability", 0.0) or 0.0)
            if exact_available
            else 0.0
        )
        structure_weight = 0.08 + 0.52 * (1.0 - maturity)

        fused_logit_unclipped = (
            BASE_RESOLVED_LOGIT
            + lstm_weight * neural_logit
            + shoe_weight * shoe_logit
            + structure_weight * structure_logit
        )
        fused_logit = max(
            -MAX_FINAL_LOGIT,
            min(MAX_FINAL_LOGIT, fused_logit_unclipped),
        )
        p_b = _sigmoid(fused_logit)
        p_p = 1.0 - p_b
        direction = "B" if p_b >= p_p else "P"
        confidence = p_b if direction == "B" else p_p

        return {
            "model_id": MODEL_ID,
            "available": True,
            "direction": direction,
            "probabilities": {"B": float(p_b), "P": float(p_p)},
            "confidence": float(confidence),
            "raw_confidence": float(confidence),
            "window_size": self.window_size,
            "min_history": self.min_history,
            "full_maturity_rounds": int(FULL_MATURITY_ROUNDS),
            "sequence_length": len(sequence),
            "pretrained_loaded": bool(self._pretrained_loaded),
            "online_updates": int(self._online_updates),
            "trained_rounds": int(self._trained_rounds),
            "weight_path": self.weight_path or None,
            "weight_load_error": self._load_error,
            "features": features,
            "neural": neural,
            "shoe_fusion": shoe,
            "fusion": {
                "base_resolved_banker_logit": float(BASE_RESOLVED_LOGIT),
                "lstm_logit": float(neural_logit),
                "shoe_composition_logit_deviation": float(shoe_logit),
                "structure_logit": float(structure_logit),
                "lstm_weight": float(lstm_weight),
                "shoe_weight": float(shoe_weight),
                "structure_weight": float(structure_weight),
                "cut_progress": float(cut),
                "fused_logit_unclipped": float(fused_logit_unclipped),
                "fused_logit": float(fused_logit),
                "direction": direction,
                "direction_authority": "single_lstm_shoe_cut_fusion",
            },
            "training_balance": dict(self._training_balance),
            "reset_count": int(self._reset_count),
            "last_reset_reason": self._last_reset_reason,
            "tensorflow": tensorflow_status(),
            "reason": (
                "lstm_plus_exact_shoe_plus_cut_fusion"
                if bool(neural.get("available"))
                else "cold_start_shoe_cut_structural_fusion_until_lstm_matures"
            ),
            "formal_direction_source": "lstm_road_model",
            "fallback_required": False,
            "target_hand_window": {
                "min": int(TARGET_HANDS_MIN),
                "max": int(TARGET_HANDS_MAX),
            },
            "default_cut_card_remaining_cards": int(DEFAULT_CUT_CARD_REMAINING),
            "semantics": (
                "single_BP_decision_from_balanced_masked_LSTM_plus_exact_nonreplacement_"
                "shoe_logit_with_50_70_hand_cut_depth_weighting"
            ),
        }


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


def predict_lstm_road(
    history: str | Iterable[Any] | None,
    *,
    scope_key: str = "",
    shoe_context: Mapping[str, Any] | None = None,
    allow_online_update: bool = True,
) -> dict[str, Any]:
    return get_lstm_road_model(scope_key).predict(
        history,
        shoe_context=shoe_context,
        allow_online_update=allow_online_update,
    )


def clear_lstm_model_cache() -> None:
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()


__all__ = [
    "MODEL_ID",
    "WINDOW_SIZE",
    "MIN_HISTORY",
    "MIN_CONTEXT",
    "MIN_ONLINE_SAMPLES",
    "FULL_MATURITY_ROUNDS",
    "REPLAY_WINDOW",
    "ONLINE_LEARNING_RATE",
    "L2_REGULARIZATION",
    "BASE_RESOLVED_LOGIT",
    "normalize_bp",
    "sequence_features",
    "tensorflow_status",
    "LSTMRoadModel",
    "get_lstm_road_model",
    "predict_lstm_road",
    "clear_lstm_model_cache",
]
