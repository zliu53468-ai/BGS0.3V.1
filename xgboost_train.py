# xgboost_train.py
import os, csv, json, argparse
import numpy as np

MAP = {"B":0, "P":1, "T":2}
INV = {0:"B", 1:"P", 2:"T"}

FEAT_WIN  = int(os.getenv("FEAT_WIN", "40"))
GRID_ROWS = int(os.getenv("GRID_ROWS","6"))
GRID_COLS = int(os.getenv("GRID_COLS","20"))
OUT_PATH  = os.getenv("XGB_OUT_PATH","/data/models/xgb.json")
SEED      = int(os.getenv("SEED","42"))

def parse_history(s: str):
    s = (s or "").strip().upper()
    toks = s.split()
    seq = list(s) if len(toks)==1 else toks
    out=[]
    for ch in seq:
        if ch in MAP: out.append(MAP[ch])
    return out

def big_road_grid(seq, rows=6, cols=20):
    import numpy as np
    grid_sign = np.zeros((rows, cols), dtype=np.int8)
    grid_ties = np.zeros((rows, cols), dtype=np.int16)
    r = 0; c = 0
    last_bp = None
    for v in seq:
        if v == 2:
            if 0 <= r < rows and 0 <= c < cols:
                grid_ties[r, c] += 1
            continue
        cur_bp = +1 if v==0 else -1
        if last_bp is None:
            r,c=0,0; grid_sign[r,c]=cur_bp; last_bp=cur_bp; continue
        if cur_bp == last_bp:
            nr=r+1; nc=c
            if nr>=rows or grid_sign[nr,nc]!=0:
                nr=r; nc=c+1
            r,c=nr,nc
            if 0 <= r < rows and 0 <= c < cols:
                grid_sign[r,c]=cur_bp
        else:
            c=c+1; r=0
            if c<cols: grid_sign[r,c]=cur_bp
            last_bp=cur_bp
    return grid_sign, grid_ties, (r,c)

def big_road_features(seq, rows=6, cols=20, win=40):
    import numpy as np
    sub = seq[-win:] if len(seq)>win else seq[:]
    gs, gt, (r,c) = big_road_grid(sub, rows, cols)
    grid_sign_flat = gs.flatten().astype(np.float32)
    grid_tie_flat  = np.clip(gt.flatten(), 0, 3).astype(np.float32)/3.0
    bp_only = [x for x in sub if x in (0,1)]
    streak_len = 0; streak_side = 0.0
    if bp_only:
        last = bp_only[-1]
        for v in reversed(bp_only):
            if v==last: streak_len += 1
            else: break
        streak_side = +1.0 if last==0 else -1.0
    col_heights=[]
    for cc in range(cols-1,-1,-1):
        h = int((gs[:,cc]!=0).sum())
        if h>0: col_heights.append(h)
        if len(col_heights)>=6: break
    while len(col_heights)<6: col_heights.append(0)
    col_heights = np.array(col_heights, dtype=np.float32)/rows
    cur_col_height = float((gs[:,c]!=0).sum())/rows if 0<=c<cols else 0.0
    cur_col_side   = float(gs[0,c]) if 0<=c<cols else 0.0
    cnt = np.bincount(sub, minlength=3).astype(np.float32)
    freq = cnt / max(1,len(sub))
    feat = np.concatenate([grid_sign_flat, grid_tie_flat,
                           np.array([streak_len/rows, streak_side], dtype=np.float32),
                           col_heights,
                           np.array([cur_col_height, cur_col_side], dtype=np.float32),
                           freq], axis=0)
    return feat

def load_dataset():
    X=[]; y=[]
    # CSV
    csv_path="data/train.csv"
    if os.path.exists(csv_path):
        with open(csv_path, newline='', encoding="utf-8") as f:
            for row in csv.DictReader(f):
                hist = row.get("history") or row.get("seq") or row.get("sequence")
                nxt  = row.get("next") or row.get("label") or row.get("y")
                if not hist or not nxt: continue
                seq = parse_history(hist); nxt = nxt.strip().upper()[:1]
                if nxt not in MAP: continue
                X.append(big_road_features(seq, GRID_ROWS, GRID_COLS, FEAT_WIN))
                y.append(MAP[nxt])
    # TXT
    txt_path="data/train.txt"
    if os.path.exists(txt_path):
        with open(txt_path, encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if "->" not in line: continue
                left,right = line.split("->",1)
                seq = parse_history(left)
                nxt = right.strip().upper()[:1]
                if nxt not in MAP: continue
                X.append(big_road_features(seq, GRID_ROWS, GRID_COLS, FEAT_WIN))
                y.append(MAP[nxt])
    if not X:
        raise SystemExit("No data found. Provide data/train.csv (history,next) or data/train.txt (HIST -> LABEL)")
    return np.stack(X,0), np.array(y, dtype=np.int32)

def main():
    import xgboost as xgb
    X,y = load_dataset()
    n = len(X)
    tr = int(n*0.8)
    dtr = xgb.DMatrix(X[:tr], label=y[:tr])
    dva = xgb.DMatrix(X[tr:], label=y[tr:]) if tr<n else None
    params = {
        "objective":"multi:softprob",
        "num_class":3,
        "eta":0.1,
        "max_depth":6,
        "subsample":0.9,
        "colsample_bytree":0.9,
        "eval_metric":"mlogloss",
        "seed":SEED
    }
    evallist=[(dtr,"train")]
    if dva is not None: evallist.append((dva,"valid"))
    bst = xgb.train(params, dtr, num_boost_round=500, evals=evallist,
                    early_stopping_rounds=50 if dva is not None else None,
                    verbose_eval=50)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    bst.save_model(OUT_PATH)
    print(f"Saved XGB model to {OUT_PATH}")

if __name__=="__main__":
    main()
