#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train LightGBM (multiclass 3: B/P/T) aligned with server v15.3
- Features: one-hot over last FEAT_WIN results (B,P,T)
- CSV source: user_id, ts, history_before, label
- Saves to text model (server loads with lgb.Booster(model_file))
"""

import os, csv
from typing import List, Tuple
import numpy as np
import lightgbm as lgb

TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "/data/logs/rounds.csv")
LGBM_OUT_PATH   = os.getenv("LGBM_OUT_PATH",   "/data/models/lgbm.txt")
os.makedirs(os.path.dirname(LGBM_OUT_PATH), exist_ok=True)

FEAT_WIN   = int(os.getenv("FEAT_WIN", "20"))
VAL_SPLIT  = float(os.getenv("VAL_SPLIT", "0.15"))
MAX_HISTORY= int(os.getenv("MAX_HISTORY", "400"))
LR         = float(os.getenv("LGBM_LR", "0.05"))
ROUNDS     = int(os.getenv("LGBM_ROUNDS", "1200"))
EARLY_STOP = int(os.getenv("LGBM_EARLY", "100"))
LUT = {'B':0,'P':1,'T':2}

def load_rows(path: str):
    rows = []
    if not os.path.exists(path):
        print(f"[ERROR] data file not found: {path}")
        return rows
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        for line in r:
            if not line or len(line) < 4: continue
            try:
                ts = int(line[1])
            except Exception:
                continue
            hist = (line[2] or "").strip().upper()
            lab  = (line[3] or "").strip().upper()
            if lab in LUT:
                rows.append((ts, hist, lab))
    rows.sort(key=lambda x: x[0])
    return rows

def seq_to_vec(seq: str, K: int) -> np.ndarray:
    seq = [ch for ch in seq if ch in LUT]
    if len(seq) > MAX_HISTORY:
        seq = seq[-MAX_HISTORY:]
    take = seq[-K:]
    vec = []
    for ch in take:
        one = [0.0,0.0,0.0]; one[LUT[ch]] = 1.0
        vec.extend(one)
    need = K*3 - len(vec)
    if need > 0:
        vec = [0.0]*need + vec
    return np.array(vec, dtype=np.float32)

def build_xy(rows, K: int):
    X, y = [], []
    for _, hist, lab in rows:
        X.append(seq_to_vec(hist, K))
        y.append(LUT[lab])
    return np.vstack(X), np.array(y, dtype=np.int32)

def time_split(rows, val_ratio: float):
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
    if len(rows) < 50:
        print(f"[WARN] few samples: {len(rows)}; training anyway.")
    tr, va = time_split(rows, VAL_SPLIT)
    Xtr, ytr = build_xy(tr, FEAT_WIN)
    Xva, yva = build_xy(va, FEAT_WIN) if va else (None, None)

    wtr = class_weights(ytr)
    ltr = lgb.Dataset(Xtr, label=ytr, weight=wtr, free_raw_data=False)
    lva = lgb.Dataset(Xva, label=yva, reference=ltr, free_raw_data=False) if Xva is not None else None

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "learning_rate": LR,
        "metric": "multi_logloss",
        "num_leaves": 63,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "max_depth": -1,
        "verbosity": -1,
    }

    valid_sets = [ltr]; valid_names = ["train"]
    if lva is not None: valid_sets.append(lva); valid_names.append("valid")

    booster = lgb.train(
        params,
        ltr,
        num_boost_round=ROUNDS,
        valid_sets=valid_sets,
        valid_names=valid_names,
        early_stopping_rounds=EARLY_STOP if lva is not None else None,
        verbose_eval=50
    )

    booster.save_model(LGBM_OUT_PATH)
    print(f"[OK] saved LGBM -> {LGBM_OUT_PATH}")

if __name__ == "__main__":
    main()
