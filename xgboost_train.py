#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train XGBoost (3-class: B/P/T) aligned with server v15

CSV schema:
user_id,ts,history_before,label

ENV (all optional):
- TRAIN_DATA_PATH   default /data/logs/rounds.csv
- XGB_OUT_PATH      default /data/models/xgb.json   # server v15 expects JSON OK
- FEAT_WIN          default 20                      # must match server.py
- VAL_SPLIT         default 0.15                    # time-based split
- XGB_ROUNDS        default 1200
- XGB_EARLY         default 100
- XGB_LR            default 0.05
- XGB_MAX_DEPTH     default 6
- XGB_SUBSAMPLE     default 0.9
- XGB_COLSAMPLE     default 0.9
- SEED              default 42
"""

import os, csv
from typing import List, Tuple
import numpy as np
import xgboost as xgb

TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "/data/logs/rounds.csv")
XGB_OUT_PATH    = os.getenv("XGB_OUT_PATH", "/data/models/xgb.json")
os.makedirs(os.path.dirname(XGB_OUT_PATH), exist_ok=True)

FEAT_WIN       = int(os.getenv("FEAT_WIN", "20"))
VAL_SPLIT      = float(os.getenv("VAL_SPLIT", "0.15"))
XGB_ROUNDS     = int(os.getenv("XGB_ROUNDS", "1200"))
XGB_EARLY      = int(os.getenv("XGB_EARLY", "100"))
XGB_LR         = float(os.getenv("XGB_LR", "0.05"))
XGB_MAX_DEPTH  = int(os.getenv("XGB_MAX_DEPTH", "6"))
XGB_SUBSAMPLE  = float(os.getenv("XGB_SUBSAMPLE", "0.9"))
XGB_COLSAMPLE  = float(os.getenv("XGB_COLSAMPLE", "0.9"))
SEED           = int(os.getenv("SEED", "42"))

LUT = {'B':0, 'P':1, 'T':2}

def load_rows(path:str):
    rows=[]
    if not os.path.exists(path):
        print(f"[ERROR] data file not found: {path}")
        return rows
    with open(path, "r", encoding="utf-8") as f:
        r=csv.reader(f)
        for line in r:
            if not line or len(line)<4: continue
            try:
                ts=int(line[1])
            except Exception:
                continue
            hist=(line[2] or "").strip().upper()
            lab =(line[3] or "").strip().upper()
            if lab in LUT:
                rows.append((ts,hist,lab))
    rows.sort(key=lambda x:x[0])  # time order
    return rows

def seq_to_vec(seq:str, K:int)->np.ndarray:
    seq=[ch for ch in seq if ch in LUT]
    take=seq[-K:]
    vec=[]
    for ch in take:
        one=[0.0,0.0,0.0]; one[LUT[ch]]=1.0
        vec.extend(one)
    need=K*3-len(vec)
    if need>0: vec=[0.0]*need+vec
    return np.array(vec, dtype=np.float32)

def build_xy(rows, K:int):
    X=[]; y=[]
    for _, hist, lab in rows:
        X.append(seq_to_vec(hist, K))
        y.append(LUT[lab])
    return np.vstack(X), np.array(y, dtype=np.int32)

def time_split(rows, val_ratio:float):
    n=len(rows)
    k=max(1, int(n*(1.0-val_ratio)))
    return rows[:k], rows[k:]

def class_weights(y: np.ndarray)->np.ndarray:
    cnt=np.bincount(y, minlength=3).astype(np.float64)
    cnt[cnt==0]=1.0
    inv=1.0/cnt
    w=inv[y]
    w*= (len(w)/w.sum())
    return w.astype(np.float32)

def main():
    rows=load_rows(TRAIN_DATA_PATH)
    if len(rows)<200:
        print(f"[WARN] few samples: {len(rows)}; training anyway.")
    tr, va = time_split(rows, VAL_SPLIT)
    Xtr,ytr = build_xy(tr, FEAT_WIN)
    Xva,yva = build_xy(va, FEAT_WIN) if len(va)>0 else (None,None)

    wtr = class_weights(ytr)
    dtr = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
    dval= xgb.DMatrix(Xva, label=yva) if Xva is not None else None

    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "learning_rate": XGB_LR,
        "max_depth": XGB_MAX_DEPTH,
        "subsample": XGB_SUBSAMPLE,
        "colsample_bytree": XGB_COLSAMPLE,
        "seed": SEED,
        "tree_method": "hist"
    }

    evals=[(dtr,"train")]
    if dval is not None: evals.append((dval,"valid"))

    booster=xgb.train(
        params, dtr,
        num_boost_round=XGB_ROUNDS,
        evals=evals,
        early_stopping_rounds=(XGB_EARLY if dval is not None else None),
        verbose_eval=50
    )

    # Save as JSON to match server v15 default path
    booster.save_model(XGB_OUT_PATH)
    print(f"[OK] saved XGB model -> {XGB_OUT_PATH}")

if __name__=="__main__":
    main()
