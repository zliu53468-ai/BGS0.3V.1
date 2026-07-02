"""
backtest_runner.py

把 predictor.predict() 當黑盒逐手回測。
預設使用 baccarat_simulator.py 產生 8 副牌百家樂模擬靴。

重要防呆：
1. 先 predict(history)，再 history.append(truth)，避免資訊洩漏。
2. 每靴 history 重新開始。
3. 每靴傳不同 shoe_id，避免 ML cache 混靴。
4. 預設關閉 AI_BLEND，避免 DeepSeek API 影響速度、成本與可重現性。

執行：
    python backtest_runner.py --shoes 2000 --seed 42

如果要測 ML：
    python backtest_runner.py --shoes 500 --mode ml

如果要輸出 CSV：
    python backtest_runner.py --shoes 2000 --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

# 必須在 import predictor 前設定，因為 predictor 會在 import 時讀環境變數。
BACKTEST_MODE = os.getenv("BACKTEST_MODE", "road")

# 預設回測先關 DeepSeek，避免 API 成本與不可重現。
os.environ["AI_BLEND"] = os.getenv("BACKTEST_AI_BLEND", "0")

# 預設第一輪先測純牌路；要開 ML 可用 --mode ml 或設定 BACKTEST_ML_WEIGHT。
if BACKTEST_MODE == "ml":
    os.environ["ML_WEIGHT"] = os.getenv("BACKTEST_ML_WEIGHT", "0.08")
else:
    os.environ["ML_WEIGHT"] = os.getenv("BACKTEST_ML_WEIGHT", "0")

# 回測時保守一點，避免小樣本大幅調權重。
os.environ.setdefault("USE_ONLINE_WEIGHTING", "1")
os.environ.setdefault("ONLINE_WEIGHT_WINDOW", "36")
os.environ.setdefault("ONLINE_WEIGHT_MIN_COUNT", "12")
os.environ.setdefault("ONLINE_WEIGHT_ALPHA", "0.22")
os.environ.setdefault("ONLINE_BAYES_ALPHA", "6.0")

from baccarat_simulator import simulate_many_shoes  # noqa: E402
from predictor import clear_model_cache, predict  # noqa: E402

try:
    from scipy import stats  # type: ignore
except Exception:  # pragma: no cover
    stats = None


@dataclass
class BacktestResult:
    shoes: int = 0
    rounds: int = 0
    correct: int = 0
    total_bets: int = 0
    observed: int = 0
    tie_push: int = 0
    skipped_tie_recommend: int = 0
    rec_b: int = 0
    rec_p: int = 0
    hit_b: int = 0
    hit_p: int = 0
    profit_units: float = 0.0
    max_win_streak: int = 0
    max_loss_streak: int = 0

    def merge(self, other: "BacktestResult") -> None:
        self.shoes += other.shoes
        self.rounds += other.rounds
        self.correct += other.correct
        self.total_bets += other.total_bets
        self.observed += other.observed
        self.tie_push += other.tie_push
        self.skipped_tie_recommend += other.skipped_tie_recommend
        self.rec_b += other.rec_b
        self.rec_p += other.rec_p
        self.hit_b += other.hit_b
        self.hit_p += other.hit_p
        self.profit_units += other.profit_units
        self.max_win_streak = max(self.max_win_streak, other.max_win_streak)
        self.max_loss_streak = max(self.max_loss_streak, other.max_loss_streak)


def exact_binom_pvalue_greater(k: int, n: int, p0: float) -> float:
    """scipy 不存在時的精確二項右尾 p-value fallback。"""
    if n <= 0:
        return 1.0
    # sum_{i=k}^{n} C(n,i)p^i(1-p)^(n-i)，用遞推避免超大組合數。
    prob0 = (1 - p0) ** n
    probs = [prob0]
    for i in range(0, n):
        prev = probs[-1]
        nxt = prev * (n - i) / (i + 1) * p0 / (1 - p0)
        probs.append(nxt)
    return float(min(1.0, sum(probs[k:])))


def binom_pvalue_greater(k: int, n: int, p0: float = 0.5068) -> float:
    if n <= 0:
        return 1.0
    if stats is not None:
        return float(stats.binomtest(k, n, p0, alternative="greater").pvalue)
    return exact_binom_pvalue_greater(k, n, p0)


def normalize_rec(raw: Any) -> str:
    rec = str(raw or "NONE").upper()
    if rec in {"莊", "BANKER"}:
        return "B"
    if rec in {"閒", "PLAYER"}:
        return "P"
    if rec in {"和", "TIE"}:
        return "T"
    if rec in {"NONE", "OBSERVE", "觀望", ""}:
        return "NONE"
    return rec if rec in {"B", "P", "T"} else "NONE"


def backtest_one_shoe(shoe: List[str], shoe_index: int, min_history: int = 6) -> BacktestResult:
    history: List[str] = []
    r = BacktestResult(shoes=1, rounds=len(shoe))

    current_win_streak = 0
    current_loss_streak = 0

    for round_index, truth in enumerate(shoe, start=1):
        truth = str(truth).upper()

        if len(history) >= min_history:
            # 重要：先預測，再把 truth 放進 history。這行以前不要移到 history.append 後面。
            pred = predict(
                history,
                venue="BACKTEST",
                room="SIM",
                shoe_id=f"shoe_{shoe_index}",
                user_id="backtest",
            )
            rec = normalize_rec(pred.get("recommend"))

            if rec == "NONE":
                r.observed += 1
            elif rec == "T":
                # 第一版主測 B/P，和局推薦先不納入 B/P 命中率。
                r.skipped_tie_recommend += 1
            elif rec in {"B", "P"}:
                if truth == "T":
                    # 百家樂押莊/閒遇和通常 push，不算輸贏。
                    r.tie_push += 1
                elif truth in {"B", "P"}:
                    r.total_bets += 1
                    if rec == "B":
                        r.rec_b += 1
                    else:
                        r.rec_p += 1

                    if rec == truth:
                        r.correct += 1
                        current_win_streak += 1
                        current_loss_streak = 0
                        if rec == "B":
                            r.hit_b += 1
                            r.profit_units += 0.95  # 莊抽 5% 佣金
                        else:
                            r.hit_p += 1
                            r.profit_units += 1.0
                    else:
                        current_loss_streak += 1
                        current_win_streak = 0
                        r.profit_units -= 1.0

                    r.max_win_streak = max(r.max_win_streak, current_win_streak)
                    r.max_loss_streak = max(r.max_loss_streak, current_loss_streak)

        history.append(truth)

    return r


def run_backtest(
    n_shoes: int = 2000,
    seed: Optional[int] = None,
    min_history: int = 6,
    p0: float = 0.5068,
    csv_path: Optional[str] = None,
) -> Dict[str, Any]:
    clear_model_cache()
    shoes = simulate_many_shoes(n_shoes=n_shoes, seed=seed)
    total = BacktestResult()

    csv_rows: List[Dict[str, Any]] = []

    for i, shoe in enumerate(shoes, start=1):
        one = backtest_one_shoe(shoe, shoe_index=i, min_history=min_history)
        total.merge(one)
        if csv_path:
            row = asdict(one)
            row["shoe_index"] = i
            row["accuracy"] = one.correct / one.total_bets if one.total_bets else 0.0
            row["ev_per_bet"] = one.profit_units / one.total_bets if one.total_bets else 0.0
            csv_rows.append(row)

    accuracy = total.correct / total.total_bets if total.total_bets else 0.0
    p_value = binom_pvalue_greater(total.correct, total.total_bets, p0=p0)
    entry_rate = total.total_bets / max(1, total.rounds)
    observe_rate = total.observed / max(1, total.rounds)
    ev_per_bet = total.profit_units / total.total_bets if total.total_bets else 0.0
    b_acc = total.hit_b / total.rec_b if total.rec_b else 0.0
    p_acc = total.hit_p / total.rec_p if total.rec_p else 0.0

    summary: Dict[str, Any] = {
        "總靴數": total.shoes,
        "總局數": total.rounds,
        "總下注局數": total.total_bets,
        "總觀望局數": total.observed,
        "和局Push次數": total.tie_push,
        "和局推薦略過次數": total.skipped_tie_recommend,
        "進場率": round(entry_rate, 6),
        "觀望率": round(observe_rate, 6),
        "整體命中率": round(accuracy, 6),
        "推薦莊次數": total.rec_b,
        "推薦閒次數": total.rec_p,
        "推薦莊命中率": round(b_acc, 6),
        "推薦閒命中率": round(p_acc, 6),
        "總損益units": round(total.profit_units, 4),
        "平均每注EV": round(ev_per_bet, 6),
        "基準線p0": p0,
        "p_value_greater": round(p_value, 10),
        "最大連贏": total.max_win_streak,
        "最大連錯": total.max_loss_streak,
        "判讀": "顯著高於基準線" if p_value < 0.05 else "沒有顯著高於基準線",
    }

    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else [])
            if csv_rows:
                writer.writeheader()
                writer.writerows(csv_rows)
        summary["CSV"] = csv_path

    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n========== Baccarat Predictor Backtest ==========")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shoes", type=int, default=2000, help="模擬靴數，建議 1000~2000 起跳")
    parser.add_argument("--seed", type=int, default=None, help="可重現測試用 seed")
    parser.add_argument("--min-history", type=int, default=6, help="幾手後開始呼叫 predict")
    parser.add_argument("--p0", type=float, default=0.5068, help="二項檢定基準線，預設排除和局後莊勝率")
    parser.add_argument("--csv", type=str, default=None, help="輸出每靴統計 CSV 路徑")
    parser.add_argument("--mode", choices=["road", "ml"], default=BACKTEST_MODE, help="road=關 ML；ml=開 ML_WEIGHT")
    args = parser.parse_args()

    # 若 CLI 指定 mode=ml，提醒使用者：因 predictor 已經 import，建議用環境變數啟動。
    if args.mode != BACKTEST_MODE:
        print("提醒：predictor 已在程式啟動時讀取環境變數。要切換 mode，建議用：")
        print(f"  BACKTEST_MODE={args.mode} python backtest_runner.py --shoes {args.shoes}")

    summary = run_backtest(
        n_shoes=args.shoes,
        seed=args.seed,
        min_history=args.min_history,
        p0=args.p0,
        csv_path=args.csv,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
