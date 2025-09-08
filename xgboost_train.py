#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train XGBoost (3-class: B/P/T) from CSV logged by server.py

CSV format:
user_id,ts,history_before,label

Env (all optional):
- TRAIN_DATA_PATH   default /mnt/data/logs/rounds.csv
- XGB_OUT_PATH      default /opt/models/xgb.json
- FEAT_WIN          default 20        # must match server.py
- VAL_SPLIT         default 0.15      # time-based split
- XGB_ROUNDS        default 600
- XGB_EARLY         default 50
- XGB_LR            default 0.08
"""

import os, csv
from typing import List, Tuple
import numpy as np

import xgboost as xgb

TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "/mnt/data/logs/rounds.csv")
XGB_OUT_PATH    = os.getenv("XGB_OUT_PATH", "/opt/models/xgb.json")
os.makedirs(os.path.dirname(XGB_OUT_PATH), exist_ok=True)

FEAT_WIN   = int(os.getenv("FEAT_WIN", "20"))
VAL_SPLIT  = float(os.getenv("VAL_SPLIT", "0.15"))
XGB_ROUNDS = int(os.getenv("XGB_ROUNDS", "600"))
XGB_EARLY  = int(os.getenv("XGB_EARLY", "50"))
XGB_LR     = float(os.getenv("XGB_LR", "0.08"))

LUT = {'B':0,'P':1,'T':2}

def load_rows(path:str) -> List[Tuple[int,str,str]]:
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
    """最近 K 手 -> one-hot 展平（K*3）"""
    seq = [ch for ch in seq if ch in LUT]
    take = seq[-K:]
    vec = []
    for ch in take:
        one = [0.0,0.0,0.0]; one[LUT[ch]] = 1.0
        vec.extend(one)
    # 左側補零
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
    """per-sample weights: inverse freq"""
    counts = np.bincount(y, minlength=3).astype(np.float64)
    counts[counts==0] = 1.0
    inv = 1.0 / counts
    w = inv[y]
    # normalize to mean 1
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
    dtr = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
    dva = xgb.DMatrix(Xva, label=yva) if len(yva) > 0 else None

    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eta": XGB_LR,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "min_child_weight": 2.0,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        # "gpu_id": 0, "tree_method": "gpu_hist",  # 如果有 GPU 可改這行
    }

    evals = [(dtr, "train")]
    if dva is not None and len(yva) > 0:
        evals.append((dva, "valid"))

    booster = xgb.train(
        params,
        dtr,
        num_boost_round=XGB_ROUNDS,
        evals=evals,
        early_stopping_rounds=XGB_EARLY if dva is not None and len(yva)>0 else None,
        verbose_eval=50
    )
    booster.save_model(XGB_OUT_PATH)
    print(f"[OK] saved XGB model -> {XGB_OUT_PATH}")

if __name__ == "__main__":
    main()
