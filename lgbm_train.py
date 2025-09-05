#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train LightGBM (3-class: B/P/T) from CSV logged by server.py

CSV format:
user_id,ts,history_before,label

Env (all optional):
- TRAIN_DATA_PATH   default /mnt/data/logs/rounds.csv
- LGBM_OUT_PATH     default /opt/models/lgbm.txt
- FEAT_WIN          default 20        # must match server.py
- VAL_SPLIT         default 0.15
- LGBM_ROUNDS       default 1200
- LGBM_EARLY        default 100
- LGBM_LR           default 0.05
"""

import os, csv
from typing import List, Tuple
import numpy as np
import lightgbm as lgb

TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "/mnt/data/logs/rounds.csv")
LGBM_OUT_PATH   = os.getenv("LGBM_OUT_PATH", "/opt/models/lgbm.txt")
os.makedirs(os.path.dirname(LGBM_OUT_PATH), exist_ok=True)

FEAT_WIN   = int(os.getenv("FEAT_WIN", "20"))
VAL_SPLIT  = float(os.getenv("VAL_SPLIT", "0.15"))
LGBM_ROUNDS= int(os.getenv("LGBM_ROUNDS", "1200"))
LGBM_EARLY = int(os.getenv("LGBM_EARLY", "100"))
LGBM_LR    = float(os.getenv("LGBM_LR", "0.05"))

LUT = {'B':0,'P':1,'T':2}

def load_rows(path:str):
    rows = []
    if not os.path.exists(path):
        print(f"[ERROR] data file not found: {path}")
        return rows
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        for line in r:
            if not line or len(line) < 4: continue
            ts = int(line[1])
            hist = (line[2] or "").strip().upper()
            lab  = (line[3] or "").strip().upper()
            if lab in LUT:
                rows.append((ts, hist, lab))
    rows.sort(key=lambda x: x[0])
    return rows

def seq_to_vec(seq:str, K:int) -> np.ndarray:
    seq = [ch for ch in seq if ch in LUT]
    take = seq[-K:]
    vec = []
    for ch in take:
        one = [0.0,0.0,0.0]; one[LUT[ch]] = 1.0
        vec.extend(one)
    need = K*3 - len(vec)
    if need > 0:
        vec = [0.0]*need + vec
    return np.array(vec, dtype=np.float32)

def build_xy(rows, K:int):
    X, y = [], []
    for _, hist, lab in rows:
        X.append(seq_to_vec(hist, K))
        y.append(LUT[lab])
    return np.vstack(X), np.array(y, dtype=np.int32)

def time_split(rows, val_ratio:float):
    n = len(rows)
    k = max(1, int(n*(1.0 - val_ratio)))
    return rows[:k], rows[k:]

def class_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=3).astype(np.float64)
    counts[counts==0] = 1.0
    inv = 1.0 / counts
    w = inv[y]
    w *= (len(w) / w.sum())
    return w.astype(np.float32)

def main():
    rows = load_rows(TRAIN_DATA_PATH)
    if len(rows) < 200:
        print(f"[WARN] few samples: {len(rows)}; training anyway.")
    tr, va = time_split(rows, VAL_SPLIT)
    Xtr, ytr = build_xy(tr, FEAT_WIN)
    Xva, yva = build_xy(va, FEAT_WIN)

    wtr = class_weights(ytr)
    ltr = lgb.Dataset(Xtr, label=ytr, weight=wtr, free_raw_data=False)
    lva = lgb.Dataset(Xva, label=yva, reference=ltr, free_raw_data=False) if len(yva)>0 else None

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "learning_rate": LGBM_LR,
        "metric": "multi_logloss",
        "num_leaves": 63,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "max_depth": -1,
        "verbosity": -1,
        # "device": "gpu"  # 若有 GPU 環境可打開
    }

    valid_sets = [ltr]
    valid_names= ["train"]
    if lva is not None and len(yva)>0:
        valid_sets.append(lva)
        valid_names.append("valid")

    booster = lgb.train(
        params,
        ltr,
        num_boost_round=LGBM_ROUNDS,
        valid_sets=valid_sets,
        valid_names=valid_names,
        early_stopping_rounds=LGBM_EARLY if lva is not None and len(yva)>0 else None,
        verbose_eval=50
    )

    booster.save_model(LGBM_OUT_PATH)
    print(f"[OK] saved LGBM model -> {LGBM_OUT_PATH}")

if __name__ == "__main__":
    main()
