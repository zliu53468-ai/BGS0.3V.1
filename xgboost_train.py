#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train XGBoost (3-class B/P/T) with one-hot(seq), plus Big-Road features.
ENV (optional):
- TRAIN_DATA_PATH  default /data/logs/rounds.csv
- XGB_OUT_PATH     default /data/models/xgb.json
- FEAT_WIN         default 20
- VAL_SPLIT        default 0.15
"""
import os, csv
from typing import List, Tuple
import numpy as np
import xgboost as xgb
from br_features import map_to_big_road, bp_only, run_hist, hazard_from_hist, exp_decay_freq

TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "/data/logs/rounds.csv")
XGB_OUT_PATH    = os.getenv("XGB_OUT_PATH", "/data/models/xgb.json")
os.makedirs(os.path.dirname(XGB_OUT_PATH), exist_ok=True)

FEAT_WIN  = int(os.getenv("FEAT_WIN","20"))
VAL_SPLIT = float(os.getenv("VAL_SPLIT","0.15"))
LUT = {'B':0,'P':1,'T':2}

def read_rows(path:str):
    rows=[]
    if not os.path.exists(path):
        print("[WARN] data not found:", path); return rows
    with open(path,"r",encoding="utf-8") as f:
        r=csv.reader(f)
        for line in r:
            if len(line)<4: continue
            hist=(line[2] or "").strip().upper()
            lab =(line[3] or "").strip().upper()
            if lab in LUT:
                rows.append((hist, lab))
    return rows

def seq_to_onehot(seq:str, K:int)->List[float]:
    seq=[ch for ch in seq if ch in LUT]
    vec=[]
    for ch in seq[-K:]:
        one=[0.0,0.0,0.0]; one[LUT[ch]]=1.0
        vec.extend(one)
    need=K*3-len(vec)
    if need>0: vec=[0.0]*need + vec
    return vec

def br_extra_feats(seq: List[str]) -> List[float]:
    # 取 Big-Road 的列深、牆阻與 early-dragon 指示、BP run hazard 估計等
    _, feat = map_to_big_road(seq)
    seq_bp = bp_only(seq)
    hist   = run_hist(seq_bp)
    cur_run=1
    if seq_bp:
        last=seq_bp[-1]; i=len(seq_bp)-2
        while i>=0 and seq_bp[i]==last: cur_run+=1; i-=1
    hz = hazard_from_hist(cur_run, hist)
    ew = exp_decay_freq(seq)
    return [
        float(feat.get("col_depth",0)),
        1.0 if feat.get("blocked",False) else 0.0,
        1.0 if feat.get("early_dragon_hint",False) else 0.0,
        float(hz),
        float(ew[2])  # long-run T intensity
    ]

def build_xy(rows)->Tuple[np.ndarray,np.ndarray]:
    X=[]; y=[]
    for hist, lab in rows:
        seq=[c for c in hist if c in LUT]
        X.append(seq_to_onehot(hist, FEAT_WIN) + br_extra_feats(seq))
        y.append(LUT[lab])
    return np.array(X,dtype=np.float32), np.array(y,dtype=np.int32)

def time_split(rows, val_ratio:float):
    n=len(rows); k=max(1, int(n*(1.0-val_ratio)))
    return rows[:k], rows[k:]

def main():
    rows=read_rows(TRAIN_DATA_PATH)
    if len(rows)<100:
        print(f"[WARN] few samples: {len(rows)}")
    tr,va=time_split(rows, VAL_SPLIT)
    Xtr,ytr=build_xy(tr); Xva,yva=build_xy(va) if va else (None,None)

    dtr=xgb.DMatrix(Xtr, label=ytr)
    dva=xgb.DMatrix(Xva, label=yva) if Xva is not None else None

    params=dict(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        eta=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9,
        min_child_weight=2, reg_lambda=1.0
    )
    watch=[(dtr,"train")] + ([(dva,"valid")] if dva is not None else [])
    booster = xgb.train(params, dtr, num_boost_round=800, evals=watch, early_stopping_rounds=80 if dva else None, verbose_eval=50)
    booster.save_model(XGB_OUT_PATH)
    print("[OK] XGB saved ->", XGB_OUT_PATH)

if __name__=="__main__":
    main()
