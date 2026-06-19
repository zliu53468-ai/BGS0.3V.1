#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_combo_db_per_condition_3m.py
用途：重新產生 data/combo_db_3m.json
key 格式：P6_B5|SC:BOTH_DRAW / PLAYER_DRAW / BANKER_DRAW / NONE_DRAW，以及 P6_B5|BASE

測試：
PER_CONDITION_SAMPLES=10000 INCLUDE_BASE=1 python generate_combo_db_per_condition_3m.py

正式高樣本：
PER_CONDITION_SAMPLES=3000000 INCLUDE_BASE=1 python generate_combo_db_per_condition_3m.py
"""

import itertools
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

OUTPUT_PATH = Path(os.getenv("COMBO_OUTPUT_PATH", "data/combo_db_3m.json"))
PER_CONDITION_SAMPLES = int(os.getenv("PER_CONDITION_SAMPLES", "10000"))
DECKS = int(os.getenv("DECKS", "8"))
SEED = int(os.getenv("COMBO_SEED", "42"))
INCLUDE_BASE = int(os.getenv("INCLUDE_BASE", "1")) == 1
SAVE_EVERY_KEY = int(os.getenv("SAVE_EVERY_KEY", "1")) == 1
USE_REALISTIC_DRAW_RULE_FILTER = int(os.getenv("USE_REALISTIC_DRAW_RULE_FILTER", "1")) == 1

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T"]
VALUES = {"A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 0}
RANK_COUNTS_PER_DECK = {r: (16 if r == "T" else 4) for r in RANKS}
SCENARIOS = ["NONE_DRAW", "PLAYER_DRAW", "BANKER_DRAW", "BOTH_DRAW"]
SCENARIO_FLAGS = {
    "NONE_DRAW": (False, False),
    "PLAYER_DRAW": (True, False),
    "BANKER_DRAW": (False, True),
    "BOTH_DRAW": (True, True),
}
SCENARIO_ZH = {
    "NONE_DRAW": "雙方不補",
    "PLAYER_DRAW": "閒補莊不補",
    "BANKER_DRAW": "莊補閒不補",
    "BOTH_DRAW": "莊閒皆補",
}

def total(cards: Tuple[str, ...]) -> int:
    return sum(VALUES[c] for c in cards) % 10

def fresh_shoe_counts() -> Dict[str, int]:
    return {r: RANK_COUNTS_PER_DECK[r] * DECKS for r in RANKS}

def draw_one(rng: random.Random, shoe: Dict[str, int]) -> str:
    n = sum(shoe.values())
    x = rng.randrange(n)
    acc = 0
    for r in RANKS:
        c = shoe.get(r, 0)
        if c <= 0:
            continue
        acc += c
        if x < acc:
            shoe[r] -= 1
            return r
    raise RuntimeError("draw failed")

def remove_cards(shoe: Dict[str, int], cards: Tuple[str, ...]) -> bool:
    for c in cards:
        if shoe.get(c, 0) <= 0:
            return False
        shoe[c] -= 1
    return True

def banker_should_draw(banker_two_total: int, player_third_value: int = None, player_drew: bool = False) -> bool:
    b = banker_two_total
    if not player_drew:
        return b <= 5
    pt = int(player_third_value)
    if b <= 2:
        return True
    if b == 3:
        return pt != 8
    if b == 4:
        return 2 <= pt <= 7
    if b == 5:
        return 4 <= pt <= 7
    if b == 6:
        return pt in (6, 7)
    return False

def deal_baccarat_round(rng: random.Random, shoe: Dict[str, int]):
    p = [draw_one(rng, shoe), draw_one(rng, shoe)]
    b = [draw_one(rng, shoe), draw_one(rng, shoe)]
    p2 = total(tuple(p))
    b2 = total(tuple(b))
    if p2 in (8, 9) or b2 in (8, 9):
        pt, bt = p2, b2
    else:
        p_drew = p2 <= 5
        p_third_value = None
        if p_drew:
            c = draw_one(rng, shoe)
            p.append(c)
            p_third_value = VALUES[c]
        if banker_should_draw(b2, p_third_value, p_drew):
            b.append(draw_one(rng, shoe))
        pt, bt = total(tuple(p)), total(tuple(b))
    if bt > pt:
        winner = "BANKER"
    elif pt > bt:
        winner = "PLAYER"
    else:
        winner = "TIE"
    return pt, bt, winner, tuple(p), tuple(b)

def valid_current_state(player_cards: Tuple[str, ...], banker_cards: Tuple[str, ...]) -> bool:
    p2 = total(player_cards[:2])
    b2 = total(banker_cards[:2])
    p_drew = len(player_cards) == 3
    b_drew = len(banker_cards) == 3
    if p2 in (8, 9) or b2 in (8, 9):
        return (not p_drew) and (not b_drew)
    expected_p_draw = p2 <= 5
    if p_drew != expected_p_draw:
        return False
    p_third_value = VALUES[player_cards[2]] if p_drew else None
    expected_b_draw = banker_should_draw(b2, p_third_value, p_drew)
    return b_drew == expected_b_draw

def state_weight(cards: Tuple[str, ...]) -> int:
    w = 1
    for c in cards:
        w *= RANK_COUNTS_PER_DECK[c]
    return int(w)

def weighted_choice(rng: random.Random, items):
    total_w = sum(x[2] for x in items)
    x = rng.randrange(total_w)
    acc = 0
    for p_cards, b_cards, w in items:
        acc += w
        if x < acc:
            return p_cards, b_cards, w
    return items[-1]

def build_current_state_pools():
    print("[1/3] Building current-state pools ...")
    pools = defaultdict(list)
    hand_cache = {2: list(itertools.product(RANKS, repeat=2)), 3: list(itertools.product(RANKS, repeat=3))}
    checked = kept = 0
    for scenario, (p_draw, b_draw) in SCENARIO_FLAGS.items():
        p_count = 3 if p_draw else 2
        b_count = 3 if b_draw else 2
        for p_cards in hand_cache[p_count]:
            for b_cards in hand_cache[b_count]:
                checked += 1
                if USE_REALISTIC_DRAW_RULE_FILTER and not valid_current_state(p_cards, b_cards):
                    continue
                pt = total(p_cards)
                bt = total(b_cards)
                tmp = fresh_shoe_counts()
                if not remove_cards(tmp, p_cards + b_cards):
                    continue
                pools[(pt, bt, scenario)].append((p_cards, b_cards, state_weight(p_cards + b_cards)))
                kept += 1
    print(f"[1/3] Pools ready. checked={checked:,}, kept={kept:,}, keys={len(pools):,}")
    return pools

def simulate_next_after_condition(rng: random.Random, state_pool, n: int):
    banker = player = tie = 0
    for _ in range(n):
        p_cards, b_cards, _w = weighted_choice(rng, state_pool)
        shoe = fresh_shoe_counts()
        if not remove_cards(shoe, p_cards + b_cards):
            continue
        _pt, _bt, winner, _np, _nb = deal_baccarat_round(rng, shoe)
        if winner == "BANKER":
            banker += 1
        elif winner == "PLAYER":
            player += 1
        else:
            tie += 1
    return {"banker": banker, "player": player, "tie": tie}

def make_record(key, pp, bp, scenario, counts, state_pool_size):
    b = int(counts.get("banker", 0))
    p = int(counts.get("player", 0))
    t = int(counts.get("tie", 0))
    sample_size = b + p + t
    no_tie = b + p
    banker_prob = b / no_tie if no_tie else 0.5
    player_prob = p / no_tie if no_tie else 0.5
    return {
        "available": True,
        "key": key,
        "player_point": int(pp),
        "banker_point": int(bp),
        "scenario": scenario,
        "scenario_zh": SCENARIO_ZH.get(scenario, scenario),
        "banker_prob": round(banker_prob, 8),
        "player_prob": round(player_prob, 8),
        "tie_rate": round(t / sample_size, 8) if sample_size else 0.0,
        "banker_wins": b,
        "player_wins": p,
        "tie_count": t,
        "sample_size": int(sample_size),
        "no_tie_sample_size": int(no_tie),
        "state_pool_size": int(state_pool_size),
        "source": "STRATIFIED_PER_CONDITION_8_DECK_BACCARAT_MC_V9",
    }

def save_db(db, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(path)

def load_existing(path: Path):
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}

def build_base_pool_for_point(pools, pp, bp):
    merged = []
    for sc in SCENARIOS:
        merged.extend(pools.get((pp, bp, sc), []))
    return merged

def main():
    start = time.time()
    rng = random.Random(SEED)
    print("=" * 72)
    print("V9 combo_db per-condition generator")
    print(f"OUTPUT_PATH={OUTPUT_PATH}")
    print(f"PER_CONDITION_SAMPLES={PER_CONDITION_SAMPLES:,}")
    print("=" * 72)
    pools = build_current_state_pools()
    db = load_existing(OUTPUT_PATH)
    db.setdefault("__meta__", {})
    db["__meta__"].update({
        "version": "combo_db_per_condition_v9",
        "decks": DECKS,
        "per_condition_samples_target": PER_CONDITION_SAMPLES,
        "include_base": INCLUDE_BASE,
        "seed": SEED,
        "scenario_keys": SCENARIOS,
        "description": "每個點數+補牌情境 key 獨立抽樣到指定樣本數。",
        "updated_at_unix": int(time.time()),
    })
    tasks = []
    for pp in range(10):
        for bp in range(10):
            for sc in SCENARIOS:
                tasks.append((pp, bp, sc, f"P{pp}_B{bp}|SC:{sc}"))
            if INCLUDE_BASE:
                tasks.append((pp, bp, "BASE", f"P{pp}_B{bp}|BASE"))
    total_tasks = len(tasks)
    print(f"[2/3] Total keys to build: {total_tasks}")
    done = skipped = 0
    for idx, (pp, bp, sc, key) in enumerate(tasks, start=1):
        existing = db.get(key)
        if isinstance(existing, dict) and int(existing.get("sample_size", 0)) >= PER_CONDITION_SAMPLES:
            skipped += 1
            continue
        pool = build_base_pool_for_point(pools, pp, bp) if sc == "BASE" else pools.get((pp, bp, sc), [])
        if not pool:
            db[key] = {
                "available": False,
                "key": key,
                "player_point": pp,
                "banker_point": bp,
                "scenario": sc,
                "banker_prob": 0.5,
                "player_prob": 0.5,
                "sample_size": 0,
                "source": "NO_REALISTIC_STATE_POOL_FOR_CONDITION",
            }
            print(f"[{idx}/{total_tasks}] {key} -> no pool")
            if SAVE_EVERY_KEY:
                save_db(db, OUTPUT_PATH)
            continue
        print(f"[{idx}/{total_tasks}] Building {key} | pool={len(pool):,} | n={PER_CONDITION_SAMPLES:,}")
        counts = simulate_next_after_condition(rng, pool, PER_CONDITION_SAMPLES)
        db[key] = make_record(key, pp, bp, sc, counts, len(pool))
        done += 1
        if SAVE_EVERY_KEY:
            save_db(db, OUTPUT_PATH)
    db["__meta__"]["completed_at_unix"] = int(time.time())
    db["__meta__"]["elapsed_seconds"] = round(time.time() - start, 2)
    db["__meta__"]["records"] = len([k for k in db.keys() if not str(k).startswith("__")])
    save_db(db, OUTPUT_PATH)
    print("[3/3] Done")
    print(f"records={db['__meta__']['records']:,}, done={done:,}, skipped={skipped:,}")
    print(f"output={OUTPUT_PATH}")

if __name__ == "__main__":
    main()
