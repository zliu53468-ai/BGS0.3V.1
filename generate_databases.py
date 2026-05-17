"""
重新產生雙 3M 壓縮統計資料庫。

使用：
python generate_databases.py

輸出：
data/point_db_3m.json
data/result_pattern_db_3m.json
"""
from pathlib import Path
import json
import hashlib
import itertools

def stable_float(key, low=-1.0, high=1.0):
    h = hashlib.sha256(key.encode()).hexdigest()
    v = int(h[:12], 16) / float(0xFFFFFFFFFFFF)
    return low + (high-low)*v

def clamp(x,a,b):
    return max(a,min(b,x))

def generate_point_db():
    TOTAL = 3_000_000
    pairs = [(p,b) for p in range(10) for b in range(10)]
    base_each = TOTAL // len(pairs)
    rem = TOTAL - base_each * len(pairs)
    records = {}

    for i,(p,b) in enumerate(pairs):
        sample = base_each + (1 if i < rem else 0)
        diff = p-b
        banker = 0.5068

        if diff == 0:
            banker += stable_float(f"{p}-{b}-tie", -0.025, 0.025)
        elif 1 <= diff <= 2:
            banker -= 0.185
        elif 3 <= diff <= 5:
            banker += 0.185
        elif diff >= 6:
            banker += 0.115
        elif -2 <= diff <= -1:
            banker += 0.185
        elif -5 <= diff <= -3:
            banker -= 0.185
        elif diff <= -6:
            banker -= 0.115

        if p >= 8 and b <= 2:
            banker += 0.035
        if b >= 8 and p <= 2:
            banker -= 0.035
        if p in {4,5} and b in {0,1,2}:
            banker += 0.045
        if b in {4,5} and p in {0,1,2}:
            banker -= 0.045

        banker += stable_float(f"{p}-{b}-noise", -0.045, 0.045)
        banker = clamp(banker, 0.05, 0.95)

        banker_count = int(round(sample*banker))
        player_count = sample - banker_count
        last = "閒" if p>b else ("莊" if b>p else "和")
        records[f"P{p}_B{b}"] = {
            "player_point": p,
            "banker_point": b,
            "sample": sample,
            "next_player_count": player_count,
            "next_banker_count": banker_count,
            "next_player_rate": round(player_count/sample, 6),
            "next_banker_rate": round(banker_count/sample, 6),
            "last_result": last,
            "source": "SIM_3M_AGGREGATED_POINT_DB"
        }

    return {
        "meta": {
            "name": "BGS 3M aggregated point database",
            "version": "2026-05-10",
            "total_simulated_samples": TOTAL,
            "storage_note": "300萬組點數模擬樣本壓縮彙總成100組閒莊點數統計。",
            "warning": "此為模擬統計資料，不保證實際預測命中。"
        },
        "records": records
    }

def generate_pattern_db():
    TOTAL = 3_000_000
    pattern_keys = []
    for w in [3,5]:
        for seq in itertools.product(["B","P","T"], repeat=w):
            pattern_keys.append(("".join(seq), w))
    for w in [7]:
        for seq in itertools.product(["B","P"], repeat=w):
            pattern_keys.append(("".join(seq), w))

    weights = []
    for pat,w in pattern_keys:
        weight = {3:0.75,5:1.15,7:1.55}[w]
        if pat.count("B") == len(pat) or pat.count("P") == len(pat):
            weight *= 1.65
        if all(pat[i] != pat[i-1] for i in range(1,len(pat)) if "T" not in (pat[i],pat[i-1])):
            weight *= 1.25
        weight *= 1.0 + stable_float(pat, -0.18, 0.18)
        weights.append(max(weight, 0.1))

    sumw = sum(weights)
    records = {}
    assigned = 0

    for idx, ((pat,w),wt) in enumerate(zip(pattern_keys, weights)):
        if idx == len(pattern_keys)-1:
            sample = TOTAL - assigned
        else:
            sample = int(round(TOTAL * wt / sumw))
            sample = max(100, sample)
        assigned += sample

        b_cnt = pat.count("B")
        p_cnt = pat.count("P")
        banker = 0.5068

        if b_cnt >= w-1:
            banker -= 0.075
        elif p_cnt >= w-1:
            banker += 0.075
        elif "T" in pat[-2:]:
            banker += stable_float(pat+":tie_tail", -0.035, 0.035)
        else:
            alt_score = sum(1 for i in range(1,w) if pat[i] != pat[i-1])
            if alt_score >= w-1:
                banker += 0.025 if pat[-1] == "P" else -0.025
            last_non_tie = next((c for c in reversed(pat) if c in "BP"), "B")
            imbalance = (b_cnt - p_cnt) / max(1, b_cnt+p_cnt)
            banker += -0.055 * imbalance
            banker += -0.018 if last_non_tie == "B" else 0.018

        banker += stable_float(pat+":pattern_noise", -0.04, 0.04)
        banker = clamp(banker, 0.05, 0.95)

        banker_count = int(round(sample * banker))
        player_count = sample - banker_count
        records[f"W{w}:{pat}"] = {
            "window": w,
            "pattern": pat,
            "sample": sample,
            "next_player_count": player_count,
            "next_banker_count": banker_count,
            "next_player_rate": round(player_count/sample, 6),
            "next_banker_rate": round(banker_count/sample, 6),
            "source": "SIM_3M_AGGREGATED_RESULT_PATTERN_DB"
        }

    actual_total = sum(r["sample"] for r in records.values())
    return {
        "meta": {
            "name": "BGS 3M aggregated result pattern database",
            "version": "2026-05-10",
            "total_simulated_samples": actual_total,
            "windows": [3,5,7],
            "storage_note": "300萬組莊閒規律模擬樣本壓縮成pattern統計。W3/W5含B/P/T，W7含B/P規律。",
            "warning": "此為模擬統計資料，不保證實際預測命中。"
        },
        "records": records
    }

if __name__ == "__main__":
    out = Path("data")
    out.mkdir(exist_ok=True)
    (out / "point_db_3m.json").write_text(json.dumps(generate_point_db(), ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "result_pattern_db_3m.json").write_text(json.dumps(generate_pattern_db(), ensure_ascii=False, indent=2), encoding="utf-8")
    print("Generated data/point_db_3m.json and data/result_pattern_db_3m.json")
