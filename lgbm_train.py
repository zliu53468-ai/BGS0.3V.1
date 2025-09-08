#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train LightGBM (3-class: B/P/T) with Big-Road features
Env:
- TRAIN_DATA_PATH  (default /data/logs/rounds.csv)
- LGBM_OUT_PATH    (default /data/models/lgbm.txt)
- FEAT_WIN         (default 20)
"""

import os, csv
from typing import List, Tuple, Dict, Any
import numpy as np
import lightgbm as lgb

TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "/data/logs/rounds.csv")
LGBM_OUT_PATH   = os.getenv("LGBM_OUT_PATH",   "/data/models/lgbm.txt")
os.makedirs(os.path.dirname(LGBM_OUT_PATH), exist_ok=True)

CLASS_ORDER=("B","P","T"); LUT={'B':0,'P':1,'T':2}
FEAT_WIN=int(os.getenv("FEAT_WIN","20"))

def map_to_big_road(seq: List[str], rows:int=6, cols:int=20) -> Dict[str,Any]:
    grid=[["" for _ in range(cols)] for _ in range(rows)]
    if not seq:
        return {"cur_run":0,"col_depth":0,"blocked":False,"early_dragon_hint":False}
    r=c=0; last=None
    for ch in seq:
        if last is None:
            grid[r][c]=ch; last=ch; continue
        if ch==last:
            if r+1<rows and grid[r+1][c]=="":
                r+=1
            else:
                c=min(cols-1,c+1)
                while c<cols and grid[r][c]!="":
                    c=min(cols-1,c+1)
                if c>=cols: c=cols-1
        else:
            last=ch
            c=min(cols-1,c+1); r=0
            while c<cols and grid[r][c]!="":
                c=min(cols-1,c+1)
            if c>=cols: c=cols-1
        if grid[r][c]=="": grid[r][c]=ch
    cur_depth=0
    for rr in range(rows):
        if grid[rr][c]!="": cur_depth=rr+1
    blocked=(r==rows-1) or (r+1<rows and grid[r+1][c]!="" and last==grid[r][c])
    def last_run_len(s: List[str])->int:
        if not s: return 0
        ch=s[-1]; i=len(s)-2; n=1
        while i>=0 and s[i]==ch: n+=1; i-=1
        return n
    def early_dragon_hint(s: List[str])->bool:
        k=min(6,len(s))
        if k<4: return False
        t=s[-k:]; return max(t.count("B"), t.count("P"))>=k-1
    return {"cur_run":last_run_len(seq),"col_depth":cur_depth,"blocked":blocked,"early_dragon_hint":early_dragon_hint(seq)}

def seq_to_onehot_tail(seq: List[str], K:int)->List[float]:
    v=[]
    tail=seq[-K:]
    for lab in tail:
        v.extend([1.0 if lab==c else 0.0 for c in CLASS_ORDER])
    need=K*3 - len(v)
    if need>0: v=[0.0]*need + v
    return v

def build_row(history_before:str, label:str)->Tuple[List[float], int]:
    seq=[ch for ch in (history_before or "").strip().upper() if ch in LUT]
    v=seq_to_onehot_tail(seq, FEAT_WIN)
    f=map_to_big_road(seq)
    v.extend([
        float(f["cur_run"]), float(f["col_depth"]),
        1.0 if f["blocked"] else 0.0,
        1.0 if f["early_dragon_hint"] else 0.0
    ])
    return v, LUT[label]

def load_dataset(path:str)->Tuple[np.ndarray,np.ndarray]:
    X=[];Y=[]
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path,"r",encoding="utf-8") as f:
        r=csv.reader(f)
        for row in r:
            if not row or len(row)<4: continue
            hist=row[2].strip().upper()
            lab =(row[3] or "").strip().upper()
            if lab not in LUT: continue
            v,y=build_row(hist, lab)
            X.append(v); Y.append(y)
    return np.array(X,dtype=np.float32), np.array(Y,dtype=np.int32)

def class_weights(y: np.ndarray) -> np.ndarray:
    cnt=np.bincount(y, minlength=3).astype(np.float64)
    cnt[cnt==0]=1.0
    inv=1.0/cnt
    w=inv[y]; w*= (len(w)/w.sum())
    return w.astype(np.float32)

def main():
    X,Y=load_dataset(TRAIN_DATA_PATH)
    if len(Y)<200:
        print(f"[WARN] few samples: {len(Y)}")
    k=int(len(Y)*0.85)
    Xtr,Ytr = X[:k], Y[:k]
    Xva,Yva = X[k:], Y[k:]
    wtr = class_weights(Ytr)
    ltr = lgb.Dataset(Xtr, label=Ytr, weight=wtr, free_raw_data=False)
    lva = lgb.Dataset(Xva, label=Yva, reference=ltr, free_raw_data=False) if len(Yva)>0 else None

    print(f"[INFO] features={Xtr.shape[1]}, train={len(Ytr)}, valid={len(Yva)}")

    params = {
        "objective":"multiclass",
        "num_class":3,
        "learning_rate":0.05,
        "metric":"multi_logloss",
        "num_leaves":63,
        "min_data_in_leaf":30,
        "feature_fraction":0.9,
        "bagging_fraction":0.9,
        "bagging_freq":1,
        "max_depth":-1,
        "verbosity":-1,
    }
    booster = lgb.train(
        params, ltr, num_boost_round=1200,
        valid_sets=[ltr] + ([lva] if lva is not None else []),
        valid_names=["train"] + (["valid"] if lva is not None else []),
        early_stopping_rounds=100 if lva is not None else None,
        verbose_eval=50
    )
    booster.save_model(LGBM_OUT_PATH)
    print(f"[OK] saved LGBM -> {LGBM_OUT_PATH}")

if __name__=="__main__":
    main()
