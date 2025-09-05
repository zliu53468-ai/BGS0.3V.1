# train_tree_models.py
# 目的：用與上線推論相同的特徵(seq_features)訓練 XGBoost / LightGBM
# 產物：models/xgb_model.pkl、models/lgbm_model.pkl（joblib 格式）
import os, math, json, argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple
from collections import Counter

# ===== 特徵工程（與 app.py 完全一致）=====
SYMBOL = {"B":0, "P":1, "T":2}
def _ratio_lastN(seq: List[str], N: int) -> Tuple[float,float,float]:
    s = seq[-N:] if len(seq)>=N else seq
    if not s: return (0.33,0.33,0.34)
    n=len(s); return (s.count("B")/n, s.count("P")/n, s.count("T")/n)

def _streak_tail(seq: List[str]) -> int:
    if not seq: return 0
    t, c = seq[-1], 1
    for i in range(len(seq)-2, -1, -1):
        if seq[i]==t: c+=1
        else: break
    return c

def _alt_streak(seq: List[str]) -> int:
    # 計算最後是否 B/P 交替，並回傳交替長度
    if len(seq) < 2: return 0
    c=1
    for i in range(len(seq)-1, 0, -1):
        a,b = seq[i], seq[i-1]
        if {"B","P"}=={a,b} and a!=b: c+=1
        else: break
    return c

def seq_features(seq: List[str], win: int=20) -> np.ndarray:
    """
    輸入：完整歷史序列（如 ['B','P','B','T','B', ...]）
    輸出：固定 26 維特徵（與 app.py 相同）：
      [ n, tail, alt, last(0/1/2), max_streak_in_win,
        b_all, p_all, t_all, b_win, p_win, t_win,   # 比例
        onehot(last5) (5*3=15) ]
    """
    n = len(seq)
    b_all,p_all,t_all = _ratio_lastN(seq, n)
    b_n,p_n,t_n       = _ratio_lastN(seq, win)
    tail  = _streak_tail(seq)
    alt   = _alt_streak(seq)
    last  = SYMBOL.get(seq[-1], -1) if n>0 else -1

    # 移動窗內最大連段
    max_streak = 0
    cur = 0
    start_idx = max(0, n - win)
    for i in range(start_idx, n):
        if i==start_idx or seq[i]==seq[i-1]:
            cur += 1
        else:
            max_streak = max(max_streak, cur)
            cur = 1
    max_streak = max(max_streak, cur)

    # 最後5手 one-hot 攤平（不足左側補 -1→全0）
    k=5
    lastK = seq[-k:]
    lastK_vec = [SYMBOL.get(s, -1) for s in lastK]
    lastK_vec = ([-1]*(k-len(lastK_vec))) + lastK_vec
    lastK_oh = np.zeros((k,3), dtype=float)
    for i,v in enumerate(lastK_vec):
        if 0<=v<3: lastK_oh[i,v]=1.0

    feats = np.array([n, tail, alt, last, max_streak,
                      b_all, p_all, t_all, b_n, p_n, t_n], dtype=float)
    return np.concatenate([feats, lastK_oh.reshape(-1)])  # 11 + 15 = 26 維

# ===== 數據載入與切分 =====
def load_sequences_from_jsonl(path: str) -> List[List[str]]:
    """
    自訂資料格式：每一行一個 JSON 物件，包含鍵 'seq'，例如：
    {"seq": "BPBBPT..."} 或 {"seq": ["B","P","B","B","P","T", ...]}
    """
    seqs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            obj = json.loads(line)
            s = obj.get("seq")
            if isinstance(s, str):
                s = [ch for ch in s if ch in {"B","P","T"}]
            if isinstance(s, list):
                s = [x for x in s if x in {"B","P","T"}]
                if len(s) >= 2:
                    seqs.append(s)
    return seqs

def build_supervised(seqs: List[List[str]], feat_win: int=20, N_min: int=6):
    """
    產生 (X, y) 監督學習資料。
    對每條序列的每個位置 i，使用 seq[:i] 做特徵，y=seq[i]（下一手）
    只保留長度 >= N_min 的前綴。
    """
    X_list, y_list = [], []
    for seq in seqs:
        for i in range(1, len(seq)):
            prefix = seq[:i]
            if len(prefix) < N_min: 
                continue
            x = seq_features(prefix, feat_win)
            y = {"B":0, "P":1, "T":2}[seq[i]]
            X_list.append(x)
            y_list.append(y)
    X = np.asarray(X_list, dtype=float)
    y = np.asarray(y_list, dtype=int)
    return X, y

# ===== 模型訓練 =====
def train_models(X, y, outdir="models", seed=42):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, log_loss
    import joblib

    # XGBoost
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=3,
            random_state=seed,
            tree_method="hist",
            reg_alpha=0.0,
            reg_lambda=1.0,
        )
    except Exception as e:
        xgb = None
        print(f"[Warn] XGBoost 不可用：{e}")

    # LightGBM
    try:
        import lightgbm as lgb
        lgbm = lgb.LGBMClassifier(
            n_estimators=800,
            max_depth=-1,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multiclass",
            class_weight=None,
            random_state=seed,
        )
    except Exception as e:
        lgbm = None
        print(f"[Warn] LightGBM 不可用：{e}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, random_state=seed, stratify=y)

    os.makedirs(outdir, exist_ok=True)

    def fit_eval_save(model, name, path):
        if model is None: 
            return None
        model.fit(X_tr, y_tr)
        # 評估
        y_pred = model.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        if hasattr(model, "predict_proba"):
            ll = log_loss(y_te, model.predict_proba(X_te))
        else:
            ll = float("nan")
        print(f"[{name}] acc={acc:.4f} logloss={ll:.4f}")
        # 存檔
        joblib.dump(model, path)
        print(f"[{name}] saved -> {path}")
        return model

    fit_eval_save(xgb,  "XGB",  os.path.join(outdir, "xgb_model.pkl"))
    fit_eval_save(lgbm, "LGBM", os.path.join(outdir, "lgbm_model.pkl"))

# ===== 範例資料產生（如無自有資料）=====
def gen_synthetic(num=1200, min_len=20, max_len=50, seed=7):
    rng = np.random.default_rng(seed)
    seqs = []
    for _ in range(num):
        L = rng.integers(min_len, max_len+1)
        cur = rng.choice(["B","P"])
        streak = 0
        target = int(rng.integers(3,8))
        seq = []
        for i in range(L):
            # 偶爾切換莊/閒
            if streak >= target or rng.random() < 0.18:
                cur = "P" if cur=="B" else "B"
                streak = 0
                target = int(rng.integers(3,8))
            seq.append(cur)
            streak += 1
            # 小機率出現和
            if rng.random() < 0.05:
                seq.append("T")
    ...
