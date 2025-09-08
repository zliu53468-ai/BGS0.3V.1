#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train XGBoost (multiclass 3: B/P/T) aligned with server v15.3
- Features: one-hot over last FEAT_WIN results (B,P,T)
- CSV source: user_id, ts, history_before, label
- Saves Booster to JSON (server loads with xgb.Booster().load_model)
"""

import os, csv
from typing import List, Tuple
import numpy as np
import xgboost as xgb

# ---- ENV ----
TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "/data/logs/rounds.csv")
XGB_OUT_PATH    = os.getenv("XGB_OUT_PATH",    "/data/models/xgb.json")
os.makedirs(os.path.dirname(XGB_OUT_PATH), exist_ok=True)

FEAT_WIN   = int(os.getenv("FEAT_WIN", "20"))
VAL_SPLIT  = float(os.getenv("VAL_SPLIT", "0.15"))
MAX_HISTORY= int(os.getenv("MAX_HISTORY", "400"))
LR         = float(os.getenv("XGB_LR", "0.05"))
ROUNDS     = int(os.getenv("XGB_ROUNDS", "600"))
EARLY_STOP = int(os.getenv("XGB_EARLY", "60"))
LUT = {'B':0, 'P':1, 'T':2}

def load_rows(path: str):
    rows = []
    if not os.path.exists(path):
        print(f"[ERROR] file not found: {path}")
        return rows
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        # user_id,ts,history_before,label
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
    """left-pad one-hot(B,P,T) window K*3"""
    seq = [ch for ch in seq if ch in LUT]
    if len(seq) > MAX_HISTORY:
        seq = seq[-MAX_HISTORY:]
    take = seq[-K:]
    vec = []
    for ch in take:
        one = [0.0, 0.0, 0.0]; one[LUT[ch]] = 1.0
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
    if not rows:
        print("[ERROR] no data"); return
    tr, va = time_split(rows, VAL_SPLIT)
    Xtr, ytr = build_xy(tr, FEAT_WIN)
    Xva, yva = build_xy(va, FEAT_WIN) if va else (None, None)

    wtr = class_weights(ytr)
    dtr = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
    dva = xgb.DMatrix(Xva, label=yva) if Xva is not None else None

    num_features = Xtr.shape[1]
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eta": LR,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
    }

    watch = [(dtr, "train")]
    if dva is not None: watch.append((dva, "valid"))

    booster = xgb.train(
        params,
        dtr,
        num_boost_round=ROUNDS,
        evals=watch,
        early_stopping_rounds=EARLY_STOP if dva is not None else None,
        verbose_eval=50
    )
    booster.save_model(XGB_OUT_PATH)
    print(f"[OK] saved XGB -> {XGB_OUT_PATH}")

if __name__ == "__main__":
    main()
