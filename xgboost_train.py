# xgboost_train.py — train XGBoost on Big Road features

import os, csv, random
from typing import List, Tuple
import numpy as np

SEED=int(os.getenv("SEED","42")); random.seed(SEED); np.random.seed(SEED)

MAP = {"B":0,"P":1,"T":2,"莊":0,"閒":1,"和":2}

FEAT_WIN   = int(os.getenv("FEAT_WIN", "40"))
GRID_ROWS  = int(os.getenv("GRID_ROWS", "6"))
GRID_COLS  = int(os.getenv("GRID_COLS", "20"))
USE_FULL_SHOE = int(os.getenv("USE_FULL_SHOE","1"))
LOCAL_WEIGHT  = float(os.getenv("LOCAL_WEIGHT", "0.65"))
GLOBAL_WEIGHT = float(os.getenv("GLOBAL_WEIGHT", "0.35"))

def parse_seq(s:str)->List[int]:
    s=(s or "").strip().upper()
    if not s: return []
    toks=s.split()
    seq=list(s) if len(toks)==1 else toks
    out=[]
    for ch in seq:
        if ch in MAP: out.append(MAP[ch])
    return out

def big_road_grid(seq: List[int], rows:int=6, cols:int=20):
    gs = np.zeros((rows, cols), dtype=np.int8)
    gt = np.zeros((rows, cols), dtype=np.int16)
    r=c=0; last_bp=None
    for v in seq:
        if v==2:
            if 0<=r<rows and 0<=c<cols: gt[r,c]+=1
            continue
        cur = +1 if v==0 else -1
        if last_bp is None:
            r=c=0; gs[r,c]=cur; last_bp=cur; continue
        if cur==last_bp:
            nr=r+1; nc=c
            if nr>=rows or gs[nr,nc]!=0: nr=r; nc=c+1
            r,c=nr,nc
            if 0<=r<rows and 0<=c<cols: gs[r,c]=cur
        else:
            c=c+1; r=0; last_bp=cur
            if c<cols: gs[r,c]=cur
    return gs, gt, (r,c)

def _global_aggregates(seq: List[int]) -> np.ndarray:
    n=len(seq)
    if n==0:
        return np.array([0.49,0.49,0.02, 0.5,0.5, 0,0,0,0, 0.5,0.5,0.5,0.5, 0.0], dtype=np.float32)
    arr=np.array(seq, dtype=np.int16)
    cnt=np.bincount(arr, minlength=3).astype(np.float32); freq=cnt/n
    bp=arr[arr!=2]
    altern=0.5 if len(bp)<2 else float(np.mean(bp[1:]!=bp[:-1]))
    def run_stats(side):
        x=(bp==side).astype(np.int8)
        if x.size==0: return 0.0,0.0
        runs=[]; cur=0
        for v in x:
            if v==1: cur+=1
            elif cur>0: runs.append(cur); cur=0
        if cur>0: runs.append(cur)
        if not runs: return 0.0,0.0
        r=np.array(runs, dtype=np.float32)
        return float(r.mean()), float(r.var()) if r.size>1 else 0.0
    b_mean,b_var = run_stats(0); p_mean,p_var = run_stats(1)
    b2b=p2p=b2p=p2b=0; cb=cp=0
    for i in range(len(bp)-1):
        a,b=bp[i], bp[i+1]
        if a==0: cb+=1; b2b+=(b==0); b2p+=(b==1)
        else:    cp+=1; p2p+=(b==1); p2b+=(b==0)
    B2B=(b2b/cb) if cb>0 else 0.5
    P2P=(p2p/cp) if cp>0 else 0.5
    B2P=(b2p/cb) if cb>0 else 0.5
    P2B=(p2b/cp) if cp>0 else 0.5
    tie_rate=float((arr==2).mean())
    return np.array([freq[0],freq[1],freq[2], altern,1.0-altern,
                     b_mean,b_var,p_mean,p_var, B2B,P2P,B2P,P2B, tie_rate], dtype=np.float32)

def _local_bigroad_feat(seq: List[int], rows:int, cols:int, win:int) -> np.ndarray:
    sub = seq[-win:] if len(seq)>win else seq[:]
    gs, gt, (r,c) = big_road_grid(sub, rows, cols)
    grid_sign_flat = gs.flatten().astype(np.float32)
    grid_tie_flat  = np.clip(gt.flatten(),0,3).astype(np.float32)/3.0
    bp_only=[x for x in sub if x in (0,1)]
    streak_len=0; streak_side=0.0
    if bp_only:
        last=bp_only[-1]
        for v in reversed(bp_only):
            if v==last: streak_len+=1
            else: break
        streak_side=+1.0 if last==0 else -1.0
    col_heights=[]
    for cc in range(cols-1,-1,-1):
        h=int((gs[:,cc]!=0).sum())
        if h>0: col_heights.append(h)
        if len(col_heights)>=6: break
    while len(col_heights)<6: col_heights.append(0)
    col_heights=np.array(col_heights, dtype=np.float32)/rows
    cur_col_height=float((gs[:,c]!=0).sum())/rows if 0<=c<cols else 0.0
    cur_col_side=float(gs[0,c]) if 0<=c<cols else 0.0
    cnt=np.bincount(sub, minlength=3).astype(np.float32); freq=cnt/max(1,len(sub))
    return np.concatenate([grid_sign_flat, grid_tie_flat,
                           np.array([streak_len/rows, streak_side], dtype=np.float32),
                           col_heights,
                           np.array([cur_col_height, cur_col_side], dtype=np.float32),
                           freq], axis=0)

def big_road_features(seq: List[int], rows:int=6, cols:int=20, win:int=40) -> np.ndarray:
    local=_local_bigroad_feat(seq, rows, cols, win).astype(np.float32)
    if USE_FULL_SHOE:
        glob=_global_aggregates(seq).astype(np.float32)
        lw=max(0.0, LOCAL_WEIGHT); gw=max(0.0, GLOBAL_WEIGHT); s=lw+gw
        if s==0: lw,gw=1.0,0.0
        else: lw,gw=lw/s,gw/s
        return np.concatenate([local*lw, glob*gw], axis=0).astype(np.float32)
    else:
        return local

def load_samples(path:str)->Tuple[np.ndarray, np.ndarray]:
    X=[]; y=[]
    if not os.path.exists(path):
        print(f"[WARN] train file not found: {path}"); return np.zeros((0,4),dtype=np.float32), np.zeros((0,),dtype=np.int64)
    with open(path,"r",encoding="utf-8") as f:
        r=csv.DictReader(f)
        has_next=("next" in (r.fieldnames or []))
        for row in r:
            hist = parse_seq(row.get("history",""))
            if not hist: continue
            if has_next:
                nxt = MAP.get((row.get("next","") or "").strip().upper(), None)
                if nxt is None: continue
                X.append(big_road_features(hist, GRID_ROWS, GRID_COLS, FEAT_WIN)); y.append(int(nxt))
            else:
                for i in range(1, len(hist)):
                    X.append(big_road_features(hist[:i], GRID_ROWS, GRID_COLS, FEAT_WIN)); y.append(int(hist[i]))
    X=np.stack(X,0).astype(np.float32); y=np.array(y,dtype=np.int64)
    return X,y

def main():
    import xgboost as xgb
    TRAIN_PATH=os.getenv("TRAIN_PATH","data/train.csv")
    XGB_OUT_PATH=os.getenv("XGB_OUT_PATH","data/models/xgb.json")

    X,y=load_samples(TRAIN_PATH)
    if X.shape[0]==0:
        print("[ERROR] no training data."); return

    # train/val split
    idx=np.arange(len(y)); np.random.shuffle(idx)
    n=int(0.9*len(idx)); tr,va=idx[:n], idx[n:]
    dtr=xgb.DMatrix(X[tr], label=y[tr]); dva=xgb.DMatrix(X[va], label=y[va])

    params={
        "num_class":3, "objective":"multi:softprob",
        "max_depth":6, "eta":0.1, "subsample":0.9, "colsample_bytree":0.9,
        "eval_metric":"mlogloss", "seed":SEED, "tree_method":"hist"
    }
    evallist=[(dtr,"train"),(dva,"valid")]
    bst=xgb.train(params, dtr, num_boost_round=400, evals=evallist, early_stopping_rounds=40, verbose_eval=50)
    os.makedirs(os.path.dirname(XGB_OUT_PATH), exist_ok=True)
    bst.save_model(XGB_OUT_PATH)
    print(f"[OK] saved XGB to {XGB_OUT_PATH}")

if __name__=="__main__":
    main()
