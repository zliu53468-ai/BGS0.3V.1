#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train XGBoost (3-class: B/P/T) with Big-Road features
Data: CSV rows [user_id, ts, history_before, label] from server logging
Env:
- TRAIN_DATA_PATH  (default /data/logs/rounds.csv)
- XGB_OUT_PATH     (default /data/models/xgb.json)
- FEAT_WIN         (default 20)  # must match server.py
"""

import os, csv
from typing import List, Tuple, Dict, Any
import numpy as np
import xgboost as xgb

TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "/data/logs/rounds.csv")
XGB_OUT_PATH    = os.getenv("XGB_OUT_PATH",   "/data/models/xgb.json")
os.makedirs(os.path.dirname(XGB_OUT_PATH), exist_ok=True)

CLASS_ORDER = ("B","P","T"); LUT = {'B':0,'P':1,'T':2}
FEAT_WIN = int(os.getenv("FEAT_WIN","20"))

# --- 同 server.py 的大路邏輯（簡化版） ---
def map_to_big_road(seq: List[str], rows:int=6, cols:int=20) -> Dict[str,Any]:
    grid=[["" for _ in range(cols)] for _ in range(rows)]
    if not seq:
        return {"cur_run":0,"col_depth":0,"blocked":False,"r":0,"c":0,"early_dragon_hint":False}
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
    blocked = (r==rows-1) or (r+1<rows and grid[r+1][c]!="" and last==grid[r][c])
    # run len
    def last_run_len(s: List[str])->int:
        if not s: return 0
        ch=s[-1]; i=len(s)-2; n=1
        while i>=0 and s[i]==ch: n+=1; i-=1
        return n
    def early_dragon_hint(s: List[str])->bool:
        k=min(6,len(s))
        if k<4: return False
        t=s[-k:]; return max(t.count("B"), t.count("P"))>=k-1
    return {
        "cur_run": last_run_len(seq),
        "col_depth": cur_depth,
        "blocked": blocked,
        "early_dragon_hint": early_dragon_hint(seq)
    }

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
    # features
    v = seq_to_onehot_tail(seq, FEAT_WIN)
    feat = map_to_big_road(seq)
    v.extend([
        float(feat["cur_run"]), float(feat["col_depth"]),
        1.0 if feat["blocked"] else 0.0,
        1.0 if feat["early_dragon_hint"] else 0.0
    ])
    y = LUT[label]
    return v, y

def load_dataset(path:str)->Tuple[np.ndarray, np.ndarray]:
    X=[]; Y=[]
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path,"r",encoding="utf-8") as f:
        r=csv.reader(f)
        for row in r:
            if not row or len(row)<4: continue
            hist=row[2].strip().upper()
            lab =(row[3] or "").strip().upper()
            if lab not in LUT: continue
            v,y = build_row(hist, lab)
            X.append(v); Y.append(y)
    return np.array(X,dtype=np.float32), np.array(Y,dtype=np.int32)

def main():
    X,Y = load_dataset(TRAIN_DATA_PATH)
    if len(Y)<200:
        print(f"[WARN] few samples: {len(Y)}")
    # 時序切分（後段做驗證）
    k = int(len(Y)*0.85)
    Xtr, Ytr = X[:k], Y[:k]
    Xva, Yva = X[k:], Y[k:]
    dtr = xgb.DMatrix(Xtr, label=Ytr)
    dva = xgb.DMatrix(Xva, label=Yva) if len(Yva)>0 else None

    num_feat = Xtr.shape[1]
    print(f"[INFO] features={num_feat}, train={len(Ytr)}, valid={len(Yva)}")

    params = {
        "objective":"multi:softprob",
        "num_class":3,
        "max_depth":6,
        "eta":0.05,
        "subsample":0.9,
        "colsample_bytree":0.9,
        "eval_metric":"mlogloss",
        "min_child_weight":2,
        "tree_method":"hist",
    }
    watch=[(dtr,"train")]
    if dva is not None: watch.append((dva,"valid"))

    bst = xgb.train(params, dtr, num_boost_round=800, evals=watch,
                    early_stopping_rounds=80 if dva is not None else None,
                    verbose_eval=50)
    bst.save_model(XGB_OUT_PATH)
    print(f"[OK] saved XGB -> {XGB_OUT_PATH}")

if __name__=="__main__":
    main()
