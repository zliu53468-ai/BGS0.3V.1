"""BGS 兩臂 Dynamic Contextual LinUCB 決策核心。

正式決策只有 Player(P) / Banker(B)。固定 16 維 Context 仍以剩餘牌組特徵
優先；缺少精確牌組時使用中性值。此版專門修正 50～70 局短靴中，早期單次
reward 讓某一臂長時間被鎖死的問題：模型內部 L2 正規化、兩臂舊統計 forgetting、
依有效樣本數調整探索，以及移除固定 Player 平手偏置。OCR/掃描與 LLM 均不參與。
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Mapping, Sequence
import json
import math
import os
import time

import numpy as np

ARMS = ("P", "B")
CONTEXT_DIM = 16
CONTEXT_FEATURE_NAMES = (
    "remaining_cards_ratio",
    "remaining_A_ratio", "remaining_2_ratio", "remaining_3_ratio",
    "remaining_4_ratio", "remaining_5_ratio", "remaining_6_ratio",
    "remaining_7_ratio", "remaining_8_ratio", "remaining_9_ratio",
    "remaining_10JQK_ratio", "high_vs_four_ratio_delta",
    "recent_8_banker_ratio", "recent_12_banker_ratio",
    "current_run_length_norm", "recent_12_switch_rate",
)

SHOE_DECKS = max(1, int(os.getenv("SHOE_DECKS", "8") or "8"))
LINUCB_ALPHA = max(0.0, float(os.getenv("LINUCB_ALPHA", "0.5") or "0.5"))
LINUCB_RIDGE = max(1e-6, float(os.getenv("LINUCB_RIDGE", "1.0") or "1.0"))
LINUCB_UPDATE_WEIGHT = max(1e-3, float(os.getenv("LINUCB_UPDATE_WEIGHT", "1.0") or "1.0"))
# 短靴必須讓早期樣本逐步退場，否則一臂初期一次輸局就可能整靴無法再探索。
LINUCB_FORGETTING = max(0.70, min(1.0, float(os.getenv("LINUCB_FORGETTING", "0.90") or "0.90")))
LINUCB_ARM_ALPHA_MAX_SCALE = max(1.0, min(2.5, float(os.getenv("LINUCB_ARM_ALPHA_MAX_SCALE", "1.60") or "1.60")))
LINUCB_SCORE_TIE_EPSILON = max(1e-12, float(os.getenv("LINUCB_SCORE_TIE_EPSILON", "0.000001") or "0.000001"))
ESTIMATED_CARDS_PER_ROUND = max(4.0, min(6.0, float(os.getenv("ESTIMATED_CARDS_PER_ROUND", "4.8") or "4.8")))
PROBABILITY_MIN = 0.48
PROBABILITY_MAX = 0.58
# 換版會讓 Render 上 V1/V2 已鎖住的 A/b 狀態自動失效，避免部署後仍沿用舊偏置。
STATE_VERSION = "LINUCB-2ARM-SHORTSHOE-DYNAMIC-V3"
_LOCK = RLock()


def _clip(value: Any, lo: float, hi: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(number):
        return lo
    return max(lo, min(hi, number))


def _normalize_history(history: Iterable[Any] | str | None) -> list[str]:
    if history is None:
        return []
    if isinstance(history, str):
        compact = history.replace("|", "").replace(",", "").replace(" ", "").upper()
        if compact and all(c in {"B", "P", "T"} for c in compact):
            return list(compact)[-2000:]
        items: Iterable[Any] = [p for p in history.replace("|", ",").split(",") if p.strip()]
    else:
        items = history
    out: list[str] = []
    for item in items:
        raw = (
            item.get("outcome") or item.get("actual") or item.get("actual_outcome") or item.get("virtual_outcome")
            if isinstance(item, Mapping) else item
        )
        value = str(raw or "").upper().strip()
        if value in {"B", "P", "T"}:
            out.append(value)
    return out[-2000:]


def _card_bucket(card: Any) -> int | None:
    try:
        if isinstance(card, str):
            value = card.strip().upper()
            if value == "A": return 0
            if value in {"10", "T", "J", "Q", "K"}: return 9
            numeric = int(value)
        else:
            numeric = int(card)
    except (TypeError, ValueError):
        return None
    if numeric == 1: return 0
    if 2 <= numeric <= 9: return numeric - 1
    if numeric in {0, 10, 11, 12, 13}: return 9
    return None


def _initial_bucket_counts(decks: int) -> np.ndarray:
    return np.asarray([4.0 * decks] * 9 + [16.0 * decks], dtype=np.float64)


def _remaining_counts_from_context(ctx: Mapping[str, Any], decks: int) -> tuple[np.ndarray | None, str]:
    initial = _initial_bucket_counts(decks)
    raw = ctx.get("remaining_counts")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) == 10:
        try:
            z = np.asarray([max(0.0, float(v)) for v in raw], dtype=np.float64)
            a_to_ten = np.asarray(list(z[1:10]) + [z[0]], dtype=np.float64)
            return np.minimum(a_to_ten, initial), "exact_remaining_counts"
        except (TypeError, ValueError):
            pass
    observed = ctx.get("observed_cards")
    if isinstance(observed, Sequence) and not isinstance(observed, (str, bytes)):
        counts, seen = initial.copy(), 0
        for card in observed:
            bucket = _card_bucket(card)
            if bucket is not None:
                counts[bucket] = max(0.0, counts[bucket] - 1.0)
                seen += 1
        if seen:
            return counts, "observed_cards_estimate"
    return None, "neutral_rank_ratios"


def _banker_ratio(seq: Sequence[str], window: int) -> float:
    bp = [v for v in seq[-window:] if v in {"B", "P"}]
    return float(sum(v == "B" for v in bp) / len(bp)) if bp else 0.5


def _run_length(seq: Sequence[str]) -> int:
    bp = [v for v in seq if v in {"B", "P"}]
    if not bp: return 0
    last, n = bp[-1], 0
    for value in reversed(bp):
        if value != last: break
        n += 1
    return n


def _switch_rate(seq: Sequence[str], window: int = 12) -> float:
    bp = [v for v in seq if v in {"B", "P"}][-window:]
    if len(bp) < 2: return 0.5
    return float(sum(bp[i] != bp[i - 1] for i in range(1, len(bp))) / (len(bp) - 1))


def _model_x(vector: Sequence[float]) -> np.ndarray:
    """API 保留原始 X；LinUCB 內部只做尺度正規化，不更改 16 維特徵意義。"""
    x = np.asarray(vector, dtype=np.float64).reshape(CONTEXT_DIM)
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    norm = float(np.linalg.norm(x))
    return x / norm if norm > 1e-12 else x


@dataclass(frozen=True)
class ContextSnapshot:
    vector: np.ndarray
    metadata: dict[str, Any]


class ContextGenerator:
    def __init__(self, decks: int = SHOE_DECKS):
        self.decks = max(1, int(decks or SHOE_DECKS))

    # 牌組特徵優先；X[12:16] 只作牌路輔助。缺精確牌組仍產生中性 X，不觀望。
    def build(self, history: Iterable[Any] | str | None, shoe_context: Mapping[str, Any] | None = None) -> ContextSnapshot:
        ctx = dict(shoe_context or {})
        decks = max(1, int(ctx.get("decks", self.decks) or self.decks))
        full_cards = 52 * decks
        raw = _normalize_history(history)
        counts, count_source = _remaining_counts_from_context(ctx, decks)
        if counts is not None:
            remaining = float(np.sum(counts)); remaining_source = count_source
        else:
            try: supplied = float(ctx.get("remaining_cards", 0.0) or 0.0)
            except (TypeError, ValueError): supplied = 0.0
            if supplied > 0.0:
                remaining = min(float(full_cards), supplied)
                remaining_source = str(ctx.get("remaining_cards_source") or "supplied_remaining_cards")
            else:
                remaining = max(0.0, float(full_cards) - min(float(full_cards), len(raw) * ESTIMATED_CARDS_PER_ROUND))
                remaining_source = "round_count_estimate"
        remaining_ratio = _clip(remaining / float(full_cards), 0.0, 1.0)
        initial = _initial_bucket_counts(decks)
        if counts is None:
            rank_ratios = np.ones(10, dtype=np.float64)
            estimated = initial * remaining_ratio
        else:
            rank_ratios = np.clip(np.divide(counts, initial, out=np.ones_like(initial), where=initial > 0), 0.0, 1.5)
            estimated = counts
        high_vs_four = _clip(float(rank_ratios[9] - rank_ratios[3]), -1.5, 1.5)
        recent8, recent12 = _banker_ratio(raw, 8), _banker_ratio(raw, 12)
        run_len = _run_length(raw)
        run_norm = _clip(run_len / 12.0, 0.0, 1.0)
        switch12 = _switch_rate(raw, 12)
        vector = np.asarray([
            remaining_ratio, *rank_ratios.tolist(), high_vs_four,
            recent8, recent12, run_norm, switch12,
        ], dtype=np.float64)
        if vector.shape != (CONTEXT_DIM,):
            raise RuntimeError(f"context dimension mismatch: {vector.shape}")
        vector = np.nan_to_num(vector, nan=0.5, posinf=1.0, neginf=-1.0)
        estimated_0_to_9 = [float(estimated[9]), *[float(v) for v in estimated[:9]]]
        return ContextSnapshot(vector=vector, metadata={
            "decks": decks, "full_cards": full_cards, "remaining_cards": float(remaining),
            "remaining_cards_ratio": float(remaining_ratio), "remaining_cards_source": remaining_source,
            "rank_ratio_source": count_source, "rank_ratios_a_to_10jqk": [float(v) for v in rank_ratios],
            "estimated_remaining_counts_0_to_9": estimated_0_to_9,
            "raw_round_count": len(raw), "bp_round_count": sum(v in {"B", "P"} for v in raw),
            "recent8_banker_ratio": recent8, "recent12_banker_ratio": recent12,
            "run_length": run_len, "run_length_norm": run_norm, "recent12_switch_rate": switch12,
            "raw_context_l2_norm": float(np.linalg.norm(vector)), "model_context_l2_norm": float(np.linalg.norm(_model_x(vector))),
            "context_scaling": "l2_internal_raw_api_preserved",
            "feature_priority": "remaining_shoe_first_road_auxiliary",
        })


def _state_path() -> Path:
    candidates: list[Path] = []
    configured = str(os.getenv("LINUCB_STATE_FILE", "") or "").strip()
    if configured: candidates.append(Path(configured).expanduser())
    candidates += [Path("/var/data/contextual_linucb_state.json"), Path(__file__).resolve().parent / "data" / "contextual_linucb_state.json", Path("/tmp/contextual_linucb_state.json")]
    for candidate in candidates:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            probe = candidate.parent / f".linucb_write_{time.time_ns()}"
            probe.write_text("ok", encoding="utf-8"); probe.unlink(missing_ok=True)
            return candidate
        except OSError:
            continue
    return Path("/tmp/contextual_linucb_state.json")


STATE_FILE = _state_path()


def _new_arm() -> dict[str, Any]:
    return {"A": (np.eye(CONTEXT_DIM) * LINUCB_RIDGE).tolist(), "b": np.zeros(CONTEXT_DIM).tolist(), "n": 0, "effective_n": 0.0}


def _new_scope_state() -> dict[str, Any]:
    now = int(time.time())
    return {"arms": {a: _new_arm() for a in ARMS}, "pending": {}, "updates": 0, "last_selected": "", "selection_streak": 0, "created_at": now, "updated_at": now}


def _read_state() -> dict[str, Any]:
    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if not isinstance(payload, dict): raise ValueError
    except Exception:
        payload = {}
    # 不沿用 V1/V2 鎖邊狀態。
    if payload.get("version") != STATE_VERSION or payload.get("dim") != CONTEXT_DIM:
        payload = {}
    return {"version": STATE_VERSION, "dim": CONTEXT_DIM, "alpha": LINUCB_ALPHA, "ridge": LINUCB_RIDGE, "forgetting": LINUCB_FORGETTING, "scopes": payload.get("scopes") if isinstance(payload.get("scopes"), dict) else {}}


def _write_state(payload: Mapping[str, Any]) -> None:
    tmp = STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE_FILE)


def make_scope_key(*, user_id: str = "", venue: str = "", room: str = "", shoe_id: str = "") -> str:
    raw = "|".join((str(user_id or "").strip(), str(venue or "").upper().strip(), str(room or "").strip(), str(shoe_id or "").strip()))
    return sha256((raw or "GLOBAL").encode("utf-8")).hexdigest()[:24]


def _history_fingerprint(history: Sequence[str]) -> str:
    return sha256("".join(history).encode("utf-8")).hexdigest()[:24]


def _arm_arrays(state: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    try:
        A = np.asarray(state.get("A"), dtype=np.float64).reshape(CONTEXT_DIM, CONTEXT_DIM)
        b = np.asarray(state.get("b"), dtype=np.float64).reshape(CONTEXT_DIM)
        if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)): raise ValueError
        return A, b
    except Exception:
        return np.eye(CONTEXT_DIM) * LINUCB_RIDGE, np.zeros(CONTEXT_DIM)


class ContextualLinUCB:
    def __init__(self, alpha: float = LINUCB_ALPHA):
        # alpha / Ridge / forgetting 都針對每靴 50～70 個樣本，避免早期單局把整靴鎖死。
        self.alpha = max(0.0, float(alpha)); self.generator = ContextGenerator()

    def _score(self, arm_state: Mapping[str, Any], x: np.ndarray, alpha_scale: float) -> dict[str, float]:
        A, b = _arm_arrays(arm_state)
        try:
            theta, solved_x = np.linalg.solve(A, b), np.linalg.solve(A, x)
        except np.linalg.LinAlgError:
            A = A + np.eye(CONTEXT_DIM) * LINUCB_RIDGE
            theta, solved_x = np.linalg.solve(A, b), np.linalg.solve(A, x)
        mean = float(x @ theta); uncertainty = float(math.sqrt(max(0.0, x @ solved_x)))
        effective_alpha = self.alpha * max(0.5, min(2.5, float(alpha_scale)))
        return {"score": mean + effective_alpha * uncertainty, "mean": mean, "uncertainty": uncertainty, "effective_alpha": effective_alpha, "raw_n": float(arm_state.get("n", 0) or 0), "effective_n": float(arm_state.get("effective_n", arm_state.get("n", 0)) or 0.0)}

    def _decay_arms(self, scope: dict[str, Any]) -> None:
        identity = np.eye(CONTEXT_DIM) * LINUCB_RIDGE
        arms = scope.setdefault("arms", {})
        for arm in ARMS:
            state = dict(arms.get(arm) or _new_arm()); A, b = _arm_arrays(state)
            state["A"] = (identity + LINUCB_FORGETTING * (A - identity)).tolist()
            state["b"] = (LINUCB_FORGETTING * b).tolist()
            state["effective_n"] = LINUCB_FORGETTING * float(state.get("effective_n", state.get("n", 0)) or 0.0)
            arms[arm] = state

    def _update_scope(self, scope: dict[str, Any], *, action: str, context_vector: Sequence[float], actual_outcome: str) -> dict[str, Any]:
        action, actual = str(action or "").upper().strip(), str(actual_outcome or "").upper().strip()
        if action not in ARMS or actual not in {"B", "P", "T"}:
            return {"updated": False, "reason": "invalid_feedback"}
        x = _model_x(context_vector); self._decay_arms(scope)
        scope["updates"] = int(scope.get("updates", 0) or 0) + 1; scope["updated_at"] = int(time.time())
        if actual == "T":
            # 和局 reward=0，但不把它當 B/P 方向樣本加入 A，避免無訊號卻降低 uncertainty。
            return {"updated": True, "action": action, "actual_outcome": actual, "reward": 0.0, "directional_sample_applied": False, "forgetting": LINUCB_FORGETTING, "reason": "tie_reward_zero_no_directional_information"}
        reward = (0.95 if action == "B" else 1.0) if action == actual else -1.0
        state = dict(scope.get("arms", {}).get(action) or _new_arm()); A, b = _arm_arrays(state)
        A = A + LINUCB_UPDATE_WEIGHT * np.outer(x, x); b = b + LINUCB_UPDATE_WEIGHT * reward * x
        state.update({"A": A.tolist(), "b": b.tolist(), "n": int(state.get("n", 0) or 0) + 1, "effective_n": float(state.get("effective_n", 0.0) or 0.0) + 1.0})
        scope.setdefault("arms", {})[action] = state
        return {"updated": True, "action": action, "actual_outcome": actual, "reward": reward, "directional_sample_applied": True, "update_weight": LINUCB_UPDATE_WEIGHT, "forgetting": LINUCB_FORGETTING, "context_l2_normalized": True}

    def update(self, *, scope_key: str, action: str, context_vector: Sequence[float], actual_outcome: str, clear_pending: bool = True) -> dict[str, Any]:
        with _LOCK:
            root = _read_state(); scope = dict(root["scopes"].get(scope_key) or _new_scope_state())
            result = self._update_scope(scope, action=action, context_vector=context_vector, actual_outcome=actual_outcome)
            if clear_pending: scope["pending"] = {}
            root["scopes"][scope_key] = scope; _write_state(root); return result

    def _apply_pending(self, scope: dict[str, Any], raw_history: Sequence[str]) -> dict[str, Any]:
        pending = dict(scope.get("pending") or {})
        if not pending: return {"updated": False, "reason": "no_pending_prediction"}
        previous_len = int(pending.get("raw_round_count", 0) or 0)
        if len(raw_history) <= previous_len: return {"updated": False, "reason": "no_new_resolved_round"}
        if _history_fingerprint(list(raw_history[:previous_len])) != str(pending.get("history_fingerprint") or ""):
            scope["pending"] = {}
            return {"updated": False, "reason": "history_reset_or_misaligned", "previous_len": previous_len, "current_len": len(raw_history)}
        result = self._update_scope(scope, action=str(pending.get("action") or ""), context_vector=list(pending.get("context_vector") or []), actual_outcome=raw_history[previous_len])
        scope["pending"] = {}
        result.update({"history_aligned": True, "resolved_history_index": previous_len, "history_rounds_after_append": len(raw_history)})
        return result

    def _choose(self, scope: Mapping[str, Any], scores: Mapping[str, Mapping[str, float]], scope_key: str, fingerprint: str) -> tuple[str, str]:
        gap = float(scores["B"]["score"]) - float(scores["P"]["score"])
        if abs(gap) > LINUCB_SCORE_TIE_EPSILON:
            return ("B" if gap > 0 else "P"), "max_ucb_score"
        # 只在真正數值平手時處理，不會覆蓋 LinUCB 有意義的 score 差。
        arms = dict(scope.get("arms") or {})
        b_n = float((arms.get("B") or {}).get("effective_n", 0.0) or 0.0); p_n = float((arms.get("P") or {}).get("effective_n", 0.0) or 0.0)
        if abs(b_n - p_n) > 1e-9: return ("B" if b_n < p_n else "P"), "tie_less_sampled_arm"
        last = str(scope.get("last_selected") or "").upper().strip()
        if last in ARMS: return ("P" if last == "B" else "B"), "tie_opposite_previous_arm"
        token = sha256(f"{scope_key}|{fingerprint}".encode("utf-8")).digest()[0]
        return ("B" if token % 2 else "P"), "tie_deterministic_balanced_hash"

    # 絕對沒有觀望臂：永遠比較 P/B 的 UCB score；Dynamic forgetting 只避免鎖邊。
    def predict(self, *, history: Iterable[Any] | str | None, shoe_context: Mapping[str, Any] | None, scope_key: str) -> dict[str, Any]:
        raw = _normalize_history(history); snapshot = self.generator.build(raw, shoe_context)
        raw_x, x, fingerprint = snapshot.vector, _model_x(snapshot.vector), _history_fingerprint(raw)
        with _LOCK:
            root = _read_state(); scope = dict(root["scopes"].get(scope_key) or _new_scope_state())
            feedback = self._apply_pending(scope, raw)
            n_bp = sum(v in {"B", "P"} for v in raw)
            base_scale = 1.35 if n_bp < 8 else 1.15 if n_bp < 15 else 1.0
            arms = dict(scope.get("arms") or {})
            eff = {a: max(0.0, float((arms.get(a) or {}).get("effective_n", (arms.get(a) or {}).get("n", 0)) or 0.0)) for a in ARMS}
            total_eff = sum(eff.values()); scores: dict[str, dict[str, float]] = {}
            for arm in ARMS:
                imbalance = math.sqrt(max(1.0, total_eff + 2.0) / max(1.0, eff[arm] + 1.0))
                scale = base_scale * _clip(imbalance, 0.85, LINUCB_ARM_ALPHA_MAX_SCALE)
                scores[arm] = self._score(arms.get(arm, {}), x, scale); scores[arm]["alpha_scale"] = scale
            direction, reason = self._choose(scope, scores, scope_key, fingerprint)
            other = "P" if direction == "B" else "B"
            margin = max(0.0, float(scores[direction]["score"]) - float(scores[other]["score"]))
            p = _clip(0.50 + 0.08 * math.tanh(margin), PROBABILITY_MIN, PROBABILITY_MAX)
            probabilities = {"B": p, "P": 1.0 - p, "T": 0.0} if direction == "B" else {"P": p, "B": 1.0 - p, "T": 0.0}
            previous = str(scope.get("last_selected") or "").upper().strip()
            streak = int(scope.get("selection_streak", 0) or 0) + 1 if previous == direction else 1
            scope.update({"last_selected": direction, "selection_streak": streak, "updated_at": int(time.time()), "pending": {"action": direction, "context_vector": [float(v) for v in raw_x], "raw_round_count": len(raw), "history_fingerprint": fingerprint, "created_at": int(time.time())}})
            root["scopes"][scope_key] = scope; _write_state(root)
        return {
            "model": "two_arm_dynamic_contextual_linucb", "version": STATE_VERSION,
            "direction": direction, "selected_arm": direction, "arm_index": 1 if direction == "B" else 0,
            "probabilities": probabilities, "selected_win_probability": p, "confidence": p,
            "context_vector": [float(v) for v in raw_x], "model_context_vector": [float(v) for v in x],
            "context_feature_names": list(CONTEXT_FEATURE_NAMES), "context_dim": CONTEXT_DIM,
            "context_metadata": snapshot.metadata, "scores": scores, "alpha": self.alpha, "ridge": LINUCB_RIDGE,
            "forgetting": LINUCB_FORGETTING, "feedback_update": feedback, "scope_key": scope_key,
            "arms": list(ARMS), "selection_reason": reason, "selection_streak": streak,
            "effective_arm_samples": eff, "short_shoe_target_rounds": "50-70",
            "history_round_count": len(raw), "bp_history_round_count": n_bp, "history_fingerprint": fingerprint,
            "anti_lock": {"enabled": True, "method": "dynamic_forgetting_l2_context_adaptive_exploration", "fixed_player_tie_bias_removed": True, "tie_is_non_directional": True, "old_v1_v2_state_reused": False},
        }


_DEFAULT_BANDIT = ContextualLinUCB()


def predict_bandit(*, history: Iterable[Any] | str | None, shoe_context: Mapping[str, Any] | None, scope_key: str) -> dict[str, Any]:
    return _DEFAULT_BANDIT.predict(history=history, shoe_context=shoe_context, scope_key=scope_key)


def update_bandit(*, scope_key: str, action: str, context_vector: Sequence[float], actual_outcome: str, clear_pending: bool = True) -> dict[str, Any]:
    return _DEFAULT_BANDIT.update(scope_key=scope_key, action=action, context_vector=context_vector, actual_outcome=actual_outcome, clear_pending=clear_pending)


__all__ = [
    "ARMS", "CONTEXT_DIM", "CONTEXT_FEATURE_NAMES", "ContextGenerator", "ContextualLinUCB",
    "ESTIMATED_CARDS_PER_ROUND", "LINUCB_ALPHA", "LINUCB_ARM_ALPHA_MAX_SCALE", "LINUCB_FORGETTING",
    "LINUCB_RIDGE", "LINUCB_SCORE_TIE_EPSILON", "LINUCB_UPDATE_WEIGHT", "PROBABILITY_MIN",
    "PROBABILITY_MAX", "SHOE_DECKS", "STATE_VERSION", "make_scope_key", "predict_bandit", "update_bandit",
]
