#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train LightGBM (3-class B/P/T) with one-hot(seq) + Big-Road features.
ENV:
- TRAIN_DATA_PATH  /data/logs/rounds.csv
- LGBM_OUT_PATH    /data/models/lgbm.txt
- FEAT_WIN         20
- VAL_SPLIT        0.15
"""
import os, csv
from typing import List, Tuple
import numpy as np
import lightgbm as lgb
from br_features import map_to_big_road, bp_only, run_hist, hazard_from_hist, exp_decay_freq

TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "/data/logs/rounds.csv")
LGBM_OUT_PATH   = os.getenv("LGBM_OUT_PATH", "/data/models/lgbm.txt")
os.makedirs(os.path.dirname(LGBM_OUT_PATH), exist_ok=True)

FEAT_WIN  = int(os.getenv("FEAT_WIN","20"))
VAL_SPLIT = float(os.getenv("VAL_SPLIT","0.15"))
LUT={'B':0,'P':1,'T':2}

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
        float(ew[2])
    ]

def build_xy(rows)->Tuple[np.ndarray,np.ndarray]:
    X=[]; y=[]
    for hist,lab in rows:
        seq=[c for c in hist if c in LUT]
        X.append(seq_to_onehot(hist, FEAT_WIN) + br_extra_feats(seq))
        y.append(LUT[lab])
    return np.array(X,dtype=np.float32), np.array(y,dtype=np.int32)

def time_split(rows, val_ratio:float):
    n=len(rows); k=max(1, int(n*(1.0-val_ratio)))
    return rows[:k], rows[k:]

def class_weights(y):
    import numpy as np
    counts=np.bincount(y, minlength=3).astype(np.float64)
    counts[counts==0]=1.0
    w=(1.0/counts)[y]
    w *= (len(w)/w.sum())
    return w.astype(np.float32)

def main():
    rows=read_rows(TRAIN_DATA_PATH)
    if len(rows)<100:
        print(f"[WARN] few samples: {len(rows)}")
    tr,va=time_split(rows, VAL_SPLIT)
    Xtr,ytr=build_xy(tr); Xva,yva=build_xy(va) if va else (None,None)

    wtr=class_weights(ytr)
    ltr=lgb.Dataset(Xtr,label=ytr,weight=wtr,free_raw_data=False)
    lva=lgb.Dataset(Xva,label=yva,reference=ltr,free_raw_data=False) if Xva is not None else None

    params=dict(
        objective="multiclass",
        num_class=3,
        learning_rate=0.05,
        metric="multi_logloss",
        num_leaves=63,
        min_data_in_leaf=30,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        max_depth=-1,
        verbosity=-1
    )
    valid_sets=[ltr]; valid_names=["train"]
    if lva is not None: valid_sets.append(lva); valid_names.append("valid")

    booster=lgb.train(
        params, ltr, num_boost_round=1000,
        valid_sets=valid_sets, valid_names=valid_names,
        early_stopping_rounds=80 if lva is not None else None,
        verbose_eval=50
    )
    booster.save_model(LGBM_OUT_PATH)
    print("[OK] LGBM saved ->", LGBM_OUT_PATH)

if __name__=="__main__":
    main()
