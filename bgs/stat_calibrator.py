# -*- coding: utf-8 -*-
"""
bgs/stat_calibrator.py

百萬局百家樂統計校準模組

用途：
1. 讀取 bgs/calibrator_stats.json
2. 接收 PF 粒子濾波器輸出的基礎機率
3. 根據牌靴階段、上一局結果、上一局點數、連莊連閒狀態做校準
4. 輸出修正後的 banker / player / tie 機率

注意：
這不是保證勝率模組。
它的作用是校準、降噪、降低偏莊偏閒與機率跳動。
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple


logger = logging.getLogger(__name__)


DEFAULT_STATS = {
    "global": {
        "banker": 0.4586,
        "player": 0.4462,
        "tie": 0.0952,
        "samples": 1000000,
    },
    "stage": {
        "early": {
            "banker": 0.4586,
            "player": 0.4462,
            "tie": 0.0952,
            "samples": 300000,
        },
        "mid": {
            "banker": 0.4586,
            "player": 0.4462,
            "tie": 0.0952,
            "samples": 400000,
        },
        "late": {
            "banker": 0.4586,
            "player": 0.4462,
            "tie": 0.0952,
            "samples": 300000,
        },
    },
    "last_result": {},
    "streak": {},
    "points": {},
}


class StatCalibrator:
    """
    使用方式：

        from bgs.stat_calibrator import StatCalibrator

        CALIBRATOR = StatCalibrator()

        probs = {
            "banker": 0.489,
            "player": 0.498,
            "tie": 0.013,
        }

        context = {
            "shoe_pos": 0.45,
            "last_result": "P",
            "banker_point": 5,
            "player_point": 6,
            "streak_side": "P",
            "streak_len": 2,
        }

        calibrated = CALIBRATOR.adjust(probs, context)
    """

    def __init__(
        self,
        stats_path: Optional[str] = None,
        enable: Optional[bool] = None,
        blend_weight: Optional[float] = None,
        min_samples: Optional[int] = None,
        max_shift: Optional[float] = None,
        tie_keep_base: Optional[bool] = None,
    ):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        self.stats_path = stats_path or os.getenv(
            "CALIBRATOR_STATS_PATH",
            os.path.join(base_dir, "calibrator_stats.json"),
        )

        self.enable = self._env_bool(
            "STAT_CALIBRATOR_ENABLE",
            True if enable is None else enable,
        )

        # 校準權重：越高越相信百萬局統計
        # 建議先用 0.10 ~ 0.18
        self.blend_weight = self._env_float(
            "STAT_CALIBRATOR_BLEND",
            0.16 if blend_weight is None else blend_weight,
        )

        # 樣本數太少的 bucket 不採用，避免過度擬合
        self.min_samples = self._env_int(
            "STAT_CALIBRATOR_MIN_SAMPLES",
            3000 if min_samples is None else min_samples,
        )

        # 限制每次最多修正多少機率
        self.max_shift = self._env_float(
            "STAT_CALIBRATOR_MAX_SHIFT",
            0.018 if max_shift is None else max_shift,
        )

        # 和局通常不作為下注主決策，所以預設保留 PF 原始 tie
        self.tie_keep_base = self._env_bool(
            "STAT_CALIBRATOR_TIE_KEEP_BASE",
            True if tie_keep_base is None else tie_keep_base,
        )

        self.stats = self._load_stats()

    # ---------------------------------------------------------
    # Public
    # ---------------------------------------------------------

    def adjust(
        self,
        base_probs: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        校準 PF 原始機率。

        base_probs 支援：
            {"banker": 0.49, "player": 0.50, "tie": 0.01}
            {"B": 0.49, "P": 0.50, "T": 0.01}
            {"莊": 0.49, "閒": 0.50, "和": 0.01}

        context 支援：
            shoe_pos: 0.0~1.0
            round_index: 第幾局
            total_round_est: 預估一靴總局數，預設 70
            last_result: B / P / T / 莊 / 閒 / 和
            banker_point: 0~9
            player_point: 0~9
            streak_side: B / P / 莊 / 閒
            streak_len: 連續幾局
        """

        base = self._normalize_probs(base_probs)

        if not self.enable:
            return base

        context = context or {}

        candidates = self._collect_candidates(context)

        if not candidates:
            return base

        target = self._weighted_average(candidates)

        if not target:
            return base

        adjusted = self._blend(base, target)
        adjusted = self._limit_shift(base, adjusted)
        adjusted = self._normalize_probs(adjusted)

        return adjusted

    def decide_stage(self, context: Optional[Dict[str, Any]] = None) -> str:
        context = context or {}

        shoe_pos = context.get("shoe_pos")

        if shoe_pos is None:
            round_index = context.get("round_index")
            total_round_est = context.get("total_round_est", 70)

            try:
                shoe_pos = float(round_index) / max(float(total_round_est), 1.0)
            except Exception:
                shoe_pos = 0.5

        try:
            shoe_pos = float(shoe_pos)
        except Exception:
            shoe_pos = 0.5

        if shoe_pos < 0.33:
            return "early"
        if shoe_pos < 0.70:
            return "mid"
        return "late"

    # ---------------------------------------------------------
    # Candidate collection
    # ---------------------------------------------------------

    def _collect_candidates(
        self,
        context: Dict[str, Any],
    ) -> List[Tuple[float, Dict[str, Any]]]:

        candidates: List[Tuple[float, Dict[str, Any]]] = []

        # 全域統計：穩定底盤
        global_stat = self.stats.get("global")
        if self._valid_stat(global_stat):
            candidates.append((0.20, global_stat))

        # 牌靴階段
        stage = self.decide_stage(context)
        stage_stat = self.stats.get("stage", {}).get(stage)
        if self._valid_stat(stage_stat):
            candidates.append((0.25, stage_stat))

        # 上一局結果
        last_result = self._normalize_result(context.get("last_result"))
        if last_result:
            last_stat = self.stats.get("last_result", {}).get(last_result)
            if self._valid_stat(last_stat):
                candidates.append((0.20, last_stat))

        # 連莊 / 連閒
        skey = self._make_streak_key(context)
        if skey:
            streak_stat = self.stats.get("streak", {}).get(skey)
            if self._valid_stat(streak_stat):
                candidates.append((0.25, streak_stat))

        # 點數條件
        pkey = self._make_point_key(context)
        if pkey:
            point_stat = self.stats.get("points", {}).get(pkey)
            if self._valid_stat(point_stat):
                candidates.append((0.35, point_stat))

        return candidates

    def _make_streak_key(self, context: Dict[str, Any]) -> Optional[str]:
        side = self._normalize_result(context.get("streak_side"))
        length = context.get("streak_len")

        if side not in ("B", "P"):
            return None

        try:
            length = int(length)
        except Exception:
            return None

        if length < 2:
            return None

        if length >= 4:
            return f"{side}4+"

        return f"{side}{length}"

    def _make_point_key(self, context: Dict[str, Any]) -> Optional[str]:
        last_result = self._normalize_result(context.get("last_result"))
        banker_point = context.get("banker_point")
        player_point = context.get("player_point")

        if last_result not in ("B", "P", "T"):
            return None

        try:
            banker_point = int(banker_point)
            player_point = int(player_point)
        except Exception:
            return None

        if not (0 <= banker_point <= 9):
            return None
        if not (0 <= player_point <= 9):
            return None

        return f"{last_result}_B{banker_point}_P{player_point}"

    # ---------------------------------------------------------
    # Math
    # ---------------------------------------------------------

    def _weighted_average(
        self,
        candidates: List[Tuple[float, Dict[str, Any]]],
    ) -> Optional[Dict[str, float]]:

        total_weight = 0.0
        banker = 0.0
        player = 0.0
        tie = 0.0

        for base_weight, stat in candidates:
            if not self._valid_stat(stat):
                continue

            samples = float(stat.get("samples", 0))

            # 樣本數越高，可信度略高，但封頂避免過度放大
            sample_factor = min(1.0, samples / 100000.0)
            weight = base_weight * (0.5 + 0.5 * sample_factor)

            banker += weight * float(stat.get("banker", 0.0))
            player += weight * float(stat.get("player", 0.0))
            tie += weight * float(stat.get("tie", 0.0))
            total_weight += weight

        if total_weight <= 0:
            return None

        return self._normalize_probs({
            "banker": banker / total_weight,
            "player": player / total_weight,
            "tie": tie / total_weight,
        })

    def _blend(
        self,
        base: Dict[str, float],
        target: Dict[str, float],
    ) -> Dict[str, float]:

        w = self._clamp(float(self.blend_weight), 0.0, 1.0)

        banker = base["banker"] * (1.0 - w) + target["banker"] * w
        player = base["player"] * (1.0 - w) + target["player"] * w

        if self.tie_keep_base:
            tie = base["tie"]
        else:
            tie = base["tie"] * (1.0 - w) + target["tie"] * w

        return {
            "banker": banker,
            "player": player,
            "tie": tie,
        }

    def _limit_shift(
        self,
        base: Dict[str, float],
        adjusted: Dict[str, float],
    ) -> Dict[str, float]:

        max_shift = abs(float(self.max_shift))

        out = {}

        for key in ("banker", "player", "tie"):
            old = float(base.get(key, 0.0))
            new = float(adjusted.get(key, old))

            diff = new - old

            if diff > max_shift:
                new = old + max_shift
            elif diff < -max_shift:
                new = old - max_shift

            out[key] = new

        return out

    def _normalize_probs(self, probs: Dict[str, float]) -> Dict[str, float]:
        if not isinstance(probs, dict):
            return {
                "banker": 0.4586,
                "player": 0.4462,
                "tie": 0.0952,
            }

        b = probs.get("banker", probs.get("B", probs.get("莊", 0.0)))
        p = probs.get("player", probs.get("P", probs.get("閒", 0.0)))
        t = probs.get("tie", probs.get("T", probs.get("和", 0.0)))

        try:
            b = float(b)
            p = float(p)
            t = float(t)
        except Exception:
            b, p, t = 0.4586, 0.4462, 0.0952

        b = max(0.0, b)
        p = max(0.0, p)
        t = max(0.0, t)

        total = b + p + t

        if total <= 0:
            return {
                "banker": 0.4586,
                "player": 0.4462,
                "tie": 0.0952,
            }

        return {
            "banker": b / total,
            "player": p / total,
            "tie": t / total,
        }

    # ---------------------------------------------------------
    # Utils
    # ---------------------------------------------------------

    def _load_stats(self) -> Dict[str, Any]:
        if not os.path.exists(self.stats_path):
            logger.warning(
                "[StatCalibrator] stats file not found: %s, using DEFAULT_STATS",
                self.stats_path,
            )
            return DEFAULT_STATS

        try:
            with open(self.stats_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                logger.warning("[StatCalibrator] invalid stats format, using DEFAULT_STATS")
                return DEFAULT_STATS

            merged = dict(DEFAULT_STATS)
            merged.update(data)

            logger.info("[StatCalibrator] loaded stats from %s", self.stats_path)
            return merged

        except Exception as e:
            logger.exception("[StatCalibrator] failed to load stats: %s", e)
            return DEFAULT_STATS

    def _valid_stat(self, stat: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(stat, dict):
            return False

        try:
            samples = int(stat.get("samples", 0))
        except Exception:
            return False

        if samples < int(self.min_samples):
            return False

        for key in ("banker", "player", "tie"):
            if key not in stat:
                return False
            try:
                float(stat.get(key))
            except Exception:
                return False

        return True

    def _normalize_result(self, value: Any) -> Optional[str]:
        if value is None:
            return None

        s = str(value).strip().upper()

        mapping = {
            "B": "B",
            "BANKER": "B",
            "莊": "B",
            "庄": "B",

            "P": "P",
            "PLAYER": "P",
            "閒": "P",
            "闲": "P",

            "T": "T",
            "TIE": "T",
            "和": "T",
        }

        return mapping.get(s)

    def _clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _env_bool(self, name: str, default: bool) -> bool:
        raw = os.getenv(name)

        if raw is None:
            return bool(default)

        return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")

    def _env_float(self, name: str, default: float) -> float:
        raw = os.getenv(name)

        if raw is None:
            return float(default)

        try:
            return float(raw)
        except Exception:
            return float(default)

    def _env_int(self, name: str, default: int) -> int:
        raw = os.getenv(name)

        if raw is None:
            return int(default)

        try:
            return int(raw)
        except Exception:
            return int(default)


def adjust_probs(
    base_probs: Dict[str, float],
    context: Optional[Dict[str, Any]] = None,
    stats_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    簡易函式版呼叫。

    用法：
        from bgs.stat_calibrator import adjust_probs

        probs = adjust_probs(base_probs, context)
    """

    calibrator = StatCalibrator(stats_path=stats_path)
    return calibrator.adjust(base_probs, context)
