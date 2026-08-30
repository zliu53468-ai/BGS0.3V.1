"""BGS 兩臂 Contextual LinUCB 決策核心。

正式決策只有 Player(P) / Banker(B) 兩個手臂。Context 固定 16 維，
優先使用剩餘牌組資訊；無精確牌組時採中性/估計值，仍會產生可用決策。

此模組不依賴 OCR、截圖或牌路掃描程式，也不依賴任何 LLM。
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


ARMS = ("P", "B")  # 臂 0 = Player、臂 1 = Banker
CONTEXT_DIM = 16
# 後四維刻意改成「轉折敏感」特徵，避免短樣本只會跟目前領先方。
# X[12] 近期莊比例（已置中且依樣本數衰減）
# X[13] 動量差：最近4局 vs 最近12局（正=最近更偏莊，負=最近開始往閒轉）
# X[14] 龍長衰竭訊號：龍越長越接近可能轉折
# X[15] 切換加速：最近切換率相對整段的偏高程度
CONTEXT_FEATURE_NAMES = (
    "remaining_cards_ratio",
    "remaining_A_ratio",
    "remaining_2_ratio",
    "remaining_3_ratio",
    "remaining_4_ratio",
    "remaining_5_ratio",
    "remaining_6_ratio",
    "remaining_7_ratio",
    "remaining_8_ratio",
    "remaining_9_ratio",
    "remaining_10JQK_ratio",
    "high_vs_four_ratio_delta",
    "recent_8_banker_centered_damped",
    "momentum_4_vs_12",
    "streak_exhaustion_signal",
    "switch_acceleration",
)

SHOE_DECKS = max(1, int(os.getenv("SHOE_DECKS", "8") or "8"))
LINUCB_ALPHA = max(0.0, float(os.getenv("LINUCB_ALPHA", "0.5") or "0.5"))
LINUCB_RIDGE = max(1e-6, float(os.getenv("LINUCB_RIDGE", "1.0") or "1.0"))
LINUCB_UPDATE_WEIGHT = max(
    1e-3, float(os.getenv("LINUCB_UPDATE_WEIGHT", "1.0") or "1.0")
)
ESTIMATED_CARDS_PER_ROUND = max(
    4.0, min(6.0, float(os.getenv("ESTIMATED_CARDS_PER_ROUND", "4.8") or "4.8"))
)
PROBABILITY_MIN = 0.48
PROBABILITY_MAX = 0.58
STATE_VERSION = "LINUCB-2ARM-SHORTSHOE-TURN-V2"
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
        if compact and all(char in {"B", "P", "T"} for char in compact):
            return list(compact)[-2000:]
        raw_items: Iterable[Any] = [
            item for item in history.replace("|", ",").split(",") if item.strip()
        ]
    else:
        raw_items = history
    cleaned: list[str] = []
    for item in raw_items:
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
        if value in {"B", "P", "T"}:
            cleaned.append(value)
    return cleaned[-2000:]


def _card_bucket(card: Any) -> int | None:
    """回傳 A..9,10/J/Q/K 的 0..9 bucket。"""
    if isinstance(card, str):
        value = card.strip().upper()
        if value == "A":
            return 0
        if value in {"10", "T", "J", "Q", "K"}:
            return 9
        try:
            numeric = int(value)
        except ValueError:
            return None
    else:
        try:
            numeric = int(card)
        except (TypeError, ValueError):
            return None
    if numeric == 1:
        return 0
    if 2 <= numeric <= 9:
        return numeric - 1
    if numeric in {0, 10, 11, 12, 13}:
        return 9
    return None


def _initial_bucket_counts(decks: int) -> np.ndarray:
    # A..9 每副各 4 張；10/J/Q/K 合併為 16 張。
    return np.asarray([4.0 * decks] * 9 + [16.0 * decks], dtype=np.float64)


def _remaining_counts_from_context(
    shoe_context: Mapping[str, Any], decks: int
) -> tuple[np.ndarray | None, str]:
    initial = _initial_bucket_counts(decks)
    raw_counts = shoe_context.get("remaining_counts")
    if isinstance(raw_counts, Sequence) and not isinstance(raw_counts, (str, bytes)) and len(raw_counts) == 10:
        try:
            # 專案既有 remaining_counts 採點數 0..9：0 代表 10/J/Q/K，1 代表 A。
            zero_to_nine = np.asarray(
                [max(0.0, float(value)) for value in raw_counts], dtype=np.float64
            )
            a_to_ten = np.asarray(
                list(zero_to_nine[1:10]) + [zero_to_nine[0]], dtype=np.float64
            )
            return np.minimum(a_to_ten, initial), "exact_remaining_counts"
        except (TypeError, ValueError):
            pass

    observed_cards = shoe_context.get("observed_cards")
    if isinstance(observed_cards, Sequence) and not isinstance(observed_cards, (str, bytes)):
        counts = initial.copy()
        seen = 0
        for card in observed_cards:
            bucket = _card_bucket(card)
            if bucket is None:
                continue
            counts[bucket] = max(0.0, counts[bucket] - 1.0)
            seen += 1
        if seen > 0:
            return counts, "observed_cards_estimate"
    return None, "neutral_rank_ratios"


def _history_banker_ratio(sequence: Sequence[str], window: int) -> float:
    recent = [value for value in sequence[-window:] if value in {"B", "P"}]
    if not recent:
        return 0.5
    return float(sum(value == "B" for value in recent) / len(recent))


def _run_length(sequence: Sequence[str]) -> int:
    bp = [value for value in sequence if value in {"B", "P"}]
    if not bp:
        return 0
    last = bp[-1]
    length = 0
    for value in reversed(bp):
        if value != last:
            break
        length += 1
    return length


def _switch_rate(sequence: Sequence[str], window: int = 12) -> float:
    bp = [value for value in sequence if value in {"B", "P"}][-window:]
    if len(bp) < 2:
        return 0.5
    switches = sum(bp[index] != bp[index - 1] for index in range(1, len(bp)))
    return float(switches / (len(bp) - 1))


@dataclass(frozen=True)
class ContextSnapshot:
    vector: np.ndarray
    metadata: dict[str, Any]


class ContextGenerator:
    """固定產生 16 維 Context；缺少精確牌組資料時使用中性值而不中斷決策。"""

    def __init__(self, decks: int = SHOE_DECKS):
        self.decks = max(1, int(decks or SHOE_DECKS))

    # 牌組特徵優先於純牌路特徵：百家樂每局實際是從有限牌靴不放回抽牌，
    # 因此 X[0:12] 放剩餘牌組/估計牌組；龍長與切換率只放在後段作輔助。
    def build(
        self,
        history: Iterable[Any] | str | None,
        shoe_context: Mapping[str, Any] | None = None,
    ) -> ContextSnapshot:
        context = dict(shoe_context or {})
        decks = max(1, int(context.get("decks", self.decks) or self.decks))
        full_cards = 52 * decks
        raw_history = _normalize_history(history)

        counts, counts_source = _remaining_counts_from_context(context, decks)
        if counts is not None:
            remaining_cards = float(np.sum(counts))
        else:
            try:
                supplied_remaining = float(context.get("remaining_cards", 0.0) or 0.0)
            except (TypeError, ValueError):
                supplied_remaining = 0.0
            if supplied_remaining > 0.0:
                remaining_cards = min(float(full_cards), supplied_remaining)
                remaining_source = str(
                    context.get("remaining_cards_source") or "supplied_remaining_cards"
                )
            else:
                estimated_used = min(
                    float(full_cards), len(raw_history) * ESTIMATED_CARDS_PER_ROUND
                )
                remaining_cards = max(0.0, float(full_cards) - estimated_used)
                remaining_source = "round_count_estimate"
        if counts is not None:
            remaining_source = counts_source

        remaining_cards_ratio = _clip(remaining_cards / float(full_cards), 0.0, 1.0)
        initial = _initial_bucket_counts(decks)
        if counts is None:
            # 無精確點數牌資料時，依需求使用 1.0 中性值；不因缺資料停止決策。
            rank_ratios = np.ones(10, dtype=np.float64)
            estimated_counts_a_to_ten = initial * remaining_cards_ratio
        else:
            rank_ratios = np.divide(
                counts,
                initial,
                out=np.ones_like(initial),
                where=initial > 0,
            )
            rank_ratios = np.clip(rank_ratios, 0.0, 1.5)
            estimated_counts_a_to_ten = counts

        high_ratio = float(rank_ratios[9])
        four_ratio = float(rank_ratios[3])
        high_vs_four = _clip(high_ratio - four_ratio, -1.5, 1.5)

        # --- 轉折敏感的牌路輔助特徵（解決短樣本一直跟領先方）---
        bp = [v for v in raw_history if v in {"B", "P"}]
        n_bp = len(bp)
        # 樣本信心：前 4 局幾乎不信近期比例，12 局後才接近滿分
        sample_conf = _clip((n_bp - 4) / 12.0, 0.0, 1.0)

        recent8 = _history_banker_ratio(raw_history, 8)
        recent12 = _history_banker_ratio(raw_history, 12)
        recent4 = _history_banker_ratio(raw_history, 4)
        # 置中後再依樣本數衰減，避免靴前幾局「誰多就押誰」
        recent8_centered_damped = (recent8 - 0.5) * sample_conf

        # 動量差：最近 4 局相對最近 12 局的偏移（轉折時會先出現符號變化）
        momentum_4_vs_12 = _clip((recent4 - recent12) * 2.0, -1.0, 1.0)

        run_len = _run_length(raw_history)
        # 龍長衰竭：從第 3 連開始累積，越長越暗示可能轉折（不是鼓勵續龍）
        streak_exhaustion = _clip(max(0, run_len - 3) / 8.0, 0.0, 1.0)

        switch12 = _switch_rate(raw_history, 12)
        switch6 = _switch_rate(raw_history, 6)
        # 切換加速：近期切換比稍長窗口更頻繁 → 轉折/混亂訊號
        switch_acceleration = _clip((switch6 - switch12) * 2.0, -1.0, 1.0)

        vector = np.asarray(
            [
                remaining_cards_ratio,
                *rank_ratios.tolist(),
                high_vs_four,
                recent8_centered_damped,
                momentum_4_vs_12,
                streak_exhaustion,
                switch_acceleration,
            ],
            dtype=np.float64,
        )
        if vector.shape != (CONTEXT_DIM,):
            raise RuntimeError(f"context dimension mismatch: {vector.shape}")
        vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)

        # 轉回專案慣用 0..9 剩餘張數順序，方便既有 API 診斷欄位沿用。
        estimated_zero_to_nine = [
            float(estimated_counts_a_to_ten[9]),
            *[float(value) for value in estimated_counts_a_to_ten[:9]],
        ]
        return ContextSnapshot(
            vector=vector,
            metadata={
                "decks": decks,
                "full_cards": full_cards,
                "remaining_cards": float(remaining_cards),
                "remaining_cards_ratio": float(remaining_cards_ratio),
                "remaining_cards_source": remaining_source,
                "rank_ratio_source": counts_source,
                "rank_ratios_a_to_10jqk": [float(value) for value in rank_ratios],
                "estimated_remaining_counts_0_to_9": estimated_zero_to_nine,
                "raw_round_count": len(raw_history),
                "bp_round_count": n_bp,
                "sample_confidence": float(sample_conf),
                "recent8_banker_ratio": float(recent8),
                "recent12_banker_ratio": float(recent12),
                "recent4_banker_ratio": float(recent4),
                "momentum_4_vs_12": float(momentum_4_vs_12),
                "run_length": int(run_len),
                "streak_exhaustion_signal": float(streak_exhaustion),
                "switch6": float(switch6),
                "switch12": float(switch12),
                "switch_acceleration": float(switch_acceleration),
                "feature_priority": "remaining_shoe_first_turn_sensitive_road_auxiliary",
            },
        )


def _state_path() -> Path:
    candidates = [
        Path("/var/data/contextual_linucb_state.json"),
        Path(__file__).resolve().parent / "data" / "contextual_linucb_state.json",
        Path("/tmp/contextual_linucb_state.json"),
    ]
    for candidate in candidates:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            probe = candidate.parent / f".linucb_write_{time.time_ns()}"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return candidate
        except OSError:
            continue
    return Path("/tmp/contextual_linucb_state.json")


STATE_FILE = _state_path()


def _new_scope_state() -> dict[str, Any]:
    identity = np.eye(CONTEXT_DIM, dtype=np.float64) * LINUCB_RIDGE
    return {
        "arms": {
            arm: {"A": identity.tolist(), "b": np.zeros(CONTEXT_DIM).tolist(), "n": 0}
            for arm in ARMS
        },
        "pending": {},
        "updates": 0,
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
    }


def _read_state() -> dict[str, Any]:
    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError
    except Exception:
        payload = {}
    if payload.get("version") != STATE_VERSION or payload.get("dim") != CONTEXT_DIM:
        payload = {}
    return {
        "version": STATE_VERSION,
        "dim": CONTEXT_DIM,
        "alpha": LINUCB_ALPHA,
        "ridge": LINUCB_RIDGE,
        "scopes": payload.get("scopes") if isinstance(payload.get("scopes"), dict) else {},
    }


def _write_state(payload: Mapping[str, Any]) -> None:
    data = dict(payload)
    temporary = STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    temporary.replace(STATE_FILE)


def make_scope_key(
    *, user_id: str = "", venue: str = "", room: str = "", shoe_id: str = ""
) -> str:
    raw = "|".join(
        (
            str(user_id or "").strip(),
            str(venue or "").upper().strip(),
            str(room or "").strip(),
            str(shoe_id or "").strip(),
        )
    )
    return sha256((raw or "GLOBAL").encode("utf-8")).hexdigest()[:24]


def _history_fingerprint(history: Sequence[str]) -> str:
    return sha256("".join(history).encode("utf-8")).hexdigest()[:24]


class ContextualLinUCB:
    """兩臂 Ridge LinUCB；每臂獨立維護 A 與 b，支援跨請求持久化線上更新。"""

    def __init__(self, alpha: float = LINUCB_ALPHA):
        # 50～70 局短靴樣本很少，因此 alpha 預設 0.5，避免探索項壓過樣本訊號；
        # Ridge 也維持顯著正則化，防止單靴少量結果讓係數瞬間爆大。
        self.alpha = max(0.0, float(alpha))
        self.generator = ContextGenerator()

    def _arm_score(
        self,
        arm_state: Mapping[str, Any],
        x: np.ndarray,
        *,
        alpha_scale: float = 1.0,
    ) -> dict[str, float]:
        try:
            A = np.asarray(arm_state.get("A"), dtype=np.float64).reshape(CONTEXT_DIM, CONTEXT_DIM)
            b = np.asarray(arm_state.get("b"), dtype=np.float64).reshape(CONTEXT_DIM)
        except Exception:
            A = np.eye(CONTEXT_DIM, dtype=np.float64) * LINUCB_RIDGE
            b = np.zeros(CONTEXT_DIM, dtype=np.float64)
        try:
            theta = np.linalg.solve(A, b)
            solved_x = np.linalg.solve(A, x)
        except np.linalg.LinAlgError:
            A = A + np.eye(CONTEXT_DIM, dtype=np.float64) * LINUCB_RIDGE
            theta = np.linalg.solve(A, b)
            solved_x = np.linalg.solve(A, x)
        mean = float(np.dot(x, theta))
        uncertainty = float(math.sqrt(max(0.0, np.dot(x, solved_x))))
        # 短樣本時略為提高探索，讓另一側比較有機會被選到，改善轉折反應
        effective_alpha = self.alpha * max(0.5, min(1.6, float(alpha_scale)))
        score = mean + effective_alpha * uncertainty
        return {
            "score": score,
            "mean": mean,
            "uncertainty": uncertainty,
            "effective_alpha": float(effective_alpha),
        }

    def _update_scope(
        self,
        scope: dict[str, Any],
        *,
        action: str,
        context_vector: Sequence[float],
        actual_outcome: str,
    ) -> dict[str, Any]:
        action = str(action or "").upper().strip()
        actual = str(actual_outcome or "").upper().strip()
        if action not in ARMS or actual not in {"B", "P", "T"}:
            return {"updated": False, "reason": "invalid_feedback"}
        x = np.asarray(context_vector, dtype=np.float64).reshape(CONTEXT_DIM)
        if actual == "T":
            reward = 0.0
        elif action == actual:
            reward = 0.95 if action == "B" else 1.0
        else:
            reward = -1.0

        arm_state = dict(scope.get("arms", {}).get(action) or {})
        try:
            A = np.asarray(arm_state.get("A"), dtype=np.float64).reshape(CONTEXT_DIM, CONTEXT_DIM)
            b = np.asarray(arm_state.get("b"), dtype=np.float64).reshape(CONTEXT_DIM)
        except Exception:
            A = np.eye(CONTEXT_DIM, dtype=np.float64) * LINUCB_RIDGE
            b = np.zeros(CONTEXT_DIM, dtype=np.float64)
        weight = LINUCB_UPDATE_WEIGHT
        A = A + weight * np.outer(x, x)
        b = b + weight * reward * x
        scope.setdefault("arms", {})[action] = {
            "A": A.tolist(),
            "b": b.tolist(),
            "n": int(arm_state.get("n", 0) or 0) + 1,
        }
        scope["updates"] = int(scope.get("updates", 0) or 0) + 1
        scope["updated_at"] = int(time.time())
        return {
            "updated": True,
            "action": action,
            "actual_outcome": actual,
            "reward": float(reward),
            "update_weight": float(weight),
        }

    def update(
        self,
        *,
        scope_key: str,
        action: str,
        context_vector: Sequence[float],
        actual_outcome: str,
        clear_pending: bool = True,
    ) -> dict[str, Any]:
        with _LOCK:
            state = _read_state()
            scopes = state["scopes"]
            scope = dict(scopes.get(scope_key) or _new_scope_state())
            result = self._update_scope(
                scope,
                action=action,
                context_vector=context_vector,
                actual_outcome=actual_outcome,
            )
            if clear_pending:
                scope["pending"] = {}
            scopes[scope_key] = scope
            _write_state(state)
            return result

    def _apply_pending_feedback(
        self,
        scope: dict[str, Any],
        raw_history: Sequence[str],
    ) -> dict[str, Any]:
        pending = dict(scope.get("pending") or {})
        if not pending:
            return {"updated": False, "reason": "no_pending_prediction"}
        previous_len = int(pending.get("raw_round_count", 0) or 0)
        if len(raw_history) <= previous_len:
            return {"updated": False, "reason": "no_new_resolved_round"}
        prefix = list(raw_history[:previous_len])
        if _history_fingerprint(prefix) != str(pending.get("history_fingerprint") or ""):
            scope["pending"] = {}
            return {"updated": False, "reason": "history_reset_or_misaligned"}
        # 只用 pending 決策後第一個新實際結果更新，避免跨多局時錯配 arm/context。
        actual = raw_history[previous_len]
        result = self._update_scope(
            scope,
            action=str(pending.get("action") or ""),
            context_vector=list(pending.get("context_vector") or []),
            actual_outcome=actual,
        )
        scope["pending"] = {}
        return result

    # 去掉觀望臂的原因：本系統業務規格要求每局都必須在 P/B 二選一；
    # 因此 LinUCB 只比較兩個可下注手臂，資料不足時由 Ridge prior + exploration term
    # 提供穩定分數，不會用第三臂規避決策。
    def predict(
        self,
        *,
        history: Iterable[Any] | str | None,
        shoe_context: Mapping[str, Any] | None,
        scope_key: str,
    ) -> dict[str, Any]:
        raw_history = _normalize_history(history)
        snapshot = self.generator.build(raw_history, shoe_context)
        x = snapshot.vector

        with _LOCK:
            state = _read_state()
            scopes = state["scopes"]
            scope = dict(scopes.get(scope_key) or _new_scope_state())
            feedback = self._apply_pending_feedback(scope, raw_history)

            # 靴前 12 局樣本少：探索係數提高，降低「一直黏在目前領先方」的機率
            n_bp = sum(1 for value in raw_history if value in {"B", "P"})
            if n_bp < 8:
                alpha_scale = 1.45
            elif n_bp < 15:
                alpha_scale = 1.25
            else:
                alpha_scale = 1.0

            scores = {
                arm: self._arm_score(
                    scope.get("arms", {}).get(arm, {}),
                    x,
                    alpha_scale=alpha_scale,
                )
                for arm in ARMS
            }
            # 相同分數時 arm 0(Player) 穩定勝出；不使用隨機數，確保同樣狀態可重現。
            direction = "B" if scores["B"]["score"] > scores["P"]["score"] else "P"
            other = "P" if direction == "B" else "B"
            margin = max(0.0, scores[direction]["score"] - scores[other]["score"])
            selected_probability = _clip(
                0.50 + 0.08 * math.tanh(margin),
                PROBABILITY_MIN,
                PROBABILITY_MAX,
            )
            probabilities = (
                {"B": selected_probability, "P": 1.0 - selected_probability, "T": 0.0}
                if direction == "B"
                else {"P": selected_probability, "B": 1.0 - selected_probability, "T": 0.0}
            )

            scope["pending"] = {
                "action": direction,
                "context_vector": [float(value) for value in x],
                "raw_round_count": len(raw_history),
                "history_fingerprint": _history_fingerprint(raw_history),
                "created_at": int(time.time()),
            }
            scope["updated_at"] = int(time.time())
            scopes[scope_key] = scope
            _write_state(state)

        return {
            "model": "two_arm_contextual_linucb",
            "version": STATE_VERSION,
            "direction": direction,
            "selected_arm": direction,
            "arm_index": 1 if direction == "B" else 0,
            "probabilities": probabilities,
            "selected_win_probability": float(selected_probability),
            "confidence": float(selected_probability),
            "context_vector": [float(value) for value in x],
            "context_feature_names": list(CONTEXT_FEATURE_NAMES),
            "context_dim": CONTEXT_DIM,
            "context_metadata": snapshot.metadata,
            "scores": scores,
            "alpha": float(self.alpha),
            "ridge": float(LINUCB_RIDGE),
            "feedback_update": feedback,
            "scope_key": scope_key,
            "arms": list(ARMS),
            "short_shoe_target_rounds": "50-70",
        }


_DEFAULT_BANDIT = ContextualLinUCB()


def predict_bandit(
    *,
    history: Iterable[Any] | str | None,
    shoe_context: Mapping[str, Any] | None,
    scope_key: str,
) -> dict[str, Any]:
    return _DEFAULT_BANDIT.predict(
        history=history, shoe_context=shoe_context, scope_key=scope_key
    )


def update_bandit(
    *,
    scope_key: str,
    action: str,
    context_vector: Sequence[float],
    actual_outcome: str,
    clear_pending: bool = True,
) -> dict[str, Any]:
    return _DEFAULT_BANDIT.update(
        scope_key=scope_key,
        action=action,
        context_vector=context_vector,
        actual_outcome=actual_outcome,
        clear_pending=clear_pending,
    )


__all__ = [
    "ARMS",
    "CONTEXT_DIM",
    "CONTEXT_FEATURE_NAMES",
    "ContextGenerator",
    "ContextualLinUCB",
    "LINUCB_ALPHA",
    "LINUCB_RIDGE",
    "PROBABILITY_MIN",
    "PROBABILITY_MAX",
    "SHOE_DECKS",
    "STATE_VERSION",
    "make_scope_key",
    "predict_bandit",
    "update_bandit",
]
