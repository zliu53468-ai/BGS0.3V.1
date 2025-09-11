# server.py
# BGS AI — Baccarat Predictor (Big Road 6x20 + Heuristic + Optional XGB/LGBM/RNN Ensemble)
# API:
#   POST /predict  { "history": "B P B T P ..." } or "BPBT..."
# Response:
#   { "probs": {"banker":..,"player":..,"tie":..}, "suggestion":"B/P/T/WAIT", "why":"..." }

import os, logging, math, json
from typing import List, Tuple, Optional, Dict
import numpy as np
from flask import Flask, request, jsonify

log = logging.getLogger("bgs-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
app = Flask(__name__)

# ======= ENV =======
FEAT_WIN   = int(os.getenv("FEAT_WIN", "40"))        # RNN 序列長度；大路特徵也用最近 FEAT_WIN 手
GRID_ROWS  = int(os.getenv("GRID_ROWS", "6"))
GRID_COLS  = int(os.getenv("GRID_COLS", "20"))

ENS_W_HEU  = float(os.getenv("ENS_W_HEU", "0.50"))   # 規則法權重
ENS_W_XGB  = float(os.getenv("ENS_W_XGB", "0.25"))
ENS_W_LGB  = float(os.getenv("ENS_W_LGB", "0.20"))
ENS_W_RNN  = float(os.getenv("ENS_W_RNN", "0.05"))

MIN_EDGE   = float(os.getenv("MIN_EDGE", "0.05"))     # 建議下注所需邊際
TEMP       = float(os.getenv("TEMP", "1.00"))         # 溫度（<1 更尖銳）
CLIP_T_MIN = float(os.getenv("CLIP_T_MIN", "0.02"))
CLIP_T_MAX = float(os.getenv("CLIP_T_MAX", "0.18"))
SEED       = int(os.getenv("SEED", "42"))

np.random.seed(SEED)

# ======= Optional models (lazy load) =======
XGB_MODEL = None
LGB_MODEL = None
RNN_MODEL = None

def _load_xgb():
    global XGB_MODEL
    try:
        import xgboost as xgb, os
        path = os.getenv("XGB_OUT_PATH", "/data/models/xgb.json")
        if os.path.exists(path):
            booster = xgb.Booster()
            booster.load_model(path)
            XGB_MODEL = booster
            log.info("[MODEL] XGB loaded: %s", path)
    except Exception as e:
        log.warning("[MODEL] XGB load failed: %s", e)

def _load_lgb():
    global LGB_MODEL
    try:
        import lightgbm as lgb, os
        path = os.getenv("LGBM_OUT_PATH", "/data/models/lgbm.txt")
        if os.path.exists(path):
            LGB_MODEL = lgb.Booster(model_file=path)
            log.info("[MODEL] LGBM loaded: %s", path)
    except Exception as e:
        log.warning("[MODEL] LGBM load failed: %s", e)

def _load_rnn():
    global RNN_MODEL
    try:
        import torch, torch.nn as nn, os
        class TinyRNN(nn.Module):
            def __init__(self, in_dim=3, hid=32, out_dim=3):
                super().__init__()
                self.gru = nn.GRU(input_size=in_dim, hidden_size=hid, num_layers=1, batch_first=True)
                self.fc  = nn.Linear(hid, out_dim)
            def forward(self, x):
                o, _ = self.gru(x)       # x: [B,T,3]
                last = o[:, -1, :]
                return self.fc(last)
        path = os.getenv("RNN_OUT_PATH", "/data/models/rnn.pt")
        if os.path.exists(path):
            RNN_MODEL = TinyRNN()
            state = __import__("torch").load(path, map_location="cpu")
            RNN_MODEL.load_state_dict(state)
            RNN_MODEL.eval()
            log.info("[MODEL] RNN loaded: %s", path)
    except Exception as e:
        log.warning("[MODEL] RNN load failed: %s", e)

_load_xgb()
_load_lgb()
_load_rnn()

# ======= Common utils & Big Road (一致性：訓練/預測完全同邏輯) =======
MAP = {"B":0, "P":1, "T":2}
INV = {0:"B", 1:"P", 2:"T"}

def parse_history(s: str) -> List[int]:
    s = (s or "").strip().upper()
    if not s:
        return []
    tokens = s.split()
    seq = list(s) if len(tokens)==1 else tokens
    out = []
    for ch in seq:
        ch = ch.strip().upper()
        if ch in MAP:
            out.append(MAP[ch])
    return out

def big_road_grid(seq: List[int], rows:int=6, cols:int=20):
    """
    大路 6x20：不同結果(B/P)→換新列， 相同結果→往下。Tie(T) 不移動，只記在當前格的 tie 計數。
    超出最底 row 會「切到右一列」繼續（常見 Big Road wrap 規則）。
    回傳：
      grid_sign: (rows, cols) 取值 {0=空, +1=Banker, -1=Player}
      grid_ties: (rows, cols) 該格累積的 Tie 次數
      cur_pos:   (r, c) 當前游標所在（最後一手所在格）
    """
    grid_sign = np.zeros((rows, cols), dtype=np.int8)
    grid_ties = np.zeros((rows, cols), dtype=np.int16)
    r = 0; c = 0
    last_bp = None
    for v in seq:
        if v == 2:  # Tie：只疊在當前格
            if 0 <= r < rows and 0 <= c < cols:
                grid_ties[r, c] += 1
            continue
        cur_bp = +1 if v==0 else -1  # B=+1, P=-1
        if last_bp is None:
            # 第一個 B/P，從 (0,0) 開始
            r, c = 0, 0
            grid_sign[r, c] = cur_bp
            last_bp = cur_bp
            continue
        if cur_bp == last_bp:
            # 相同結果 → 往下
            nr = r + 1
            nc = c
            if nr >= rows or grid_sign[nr, nc] != 0:
                # 到底或被占 → 右移一列、維持同列段
                nr = r
                nc = c + 1
            r, c = nr, nc
            if 0 <= r < rows and 0 <= c < cols:
                grid_sign[r, c] = cur_bp
        else:
            # 不同結果 → 換新列（右一列的 row=0）
            c = c + 1
            r = 0
            if c < cols:
                grid_sign[r, c] = cur_bp
            last_bp = cur_bp
    # 取得當前位置（最後一個非 T 的格）
    cur_pos = (r, c) if last_bp is not None else (0, 0)
    return grid_sign, grid_ties, cur_pos

def big_road_features(seq: List[int], rows:int=6, cols:int=20, win:int=40) -> np.ndarray:
    """
    將最近 win 手映射到大路，再抽特徵：
      - grid_sign_flat 6x20 → 120 維（B=+1, P=-1, 空=0）
      - grid_tie_flat  6x20 → 120 維（剪裁到 3 以內）
      - 連莊長度、連閒長度、目前 streak（B/P）
      - 最近 6 列的列高、當前列高、當前列 B/P 標記
      - 最近 win 中的 B/P/T 頻率
    """
    sub = seq[-win:] if len(seq)>win else seq[:]
    gs, gt, (r,c) = big_road_grid(sub, rows, cols)
    grid_sign_flat = gs.flatten().astype(np.float32)
    grid_tie_flat  = np.clip(gt.flatten(), 0, 3).astype(np.float32) / 3.0

    # 當前 streak（以 B/P 為主，忽略 T）
    bp_only = [x for x in sub if x in (0,1)]
    streak_len = 0
    streak_side = 0.0
    if bp_only:
        last = bp_only[-1]
        for v in reversed(bp_only):
            if v == last:
                streak_len += 1
            else:
                break
        streak_side = +1.0 if last==0 else -1.0   # B=+1, P=-1

    # 最近幾列的列高（從最右往左）
    col_heights = []
    for cc in range(cols-1, -1, -1):
        h = int((gs[:,cc]!=0).sum())
        if h>0: col_heights.append(h)
        if len(col_heights) >= 6: break
    while len(col_heights) < 6:
        col_heights.append(0)
    col_heights = np.array(col_heights, dtype=np.float32) / rows

    cur_col_height = float((gs[:, c]!=0).sum()) / rows if 0 <= c < cols else 0.0
    cur_col_side   = float(gs[0, c]) if 0 <= c < cols else 0.0  # 頂端符號代表該列的 B/P
    # 頻率
    cnt = np.bincount(sub, minlength=3).astype(np.float32)
    freq = cnt / max(1, len(sub))

    feat = np.concatenate([
        grid_sign_flat, grid_tie_flat,
        np.array([streak_len/rows, streak_side], dtype=np.float32),
        col_heights,
        np.array([cur_col_height, cur_col_side], dtype=np.float32),
        freq
    ], axis=0)
    return feat

def one_hot_seq(seq: List[int], win:int) -> np.ndarray:
    # RNN 用：最近 win 手 one-hot → [1, win, 3]
    sub = seq[-win:] if len(seq)>win else seq[:]
    pad = [ -1 ] * max(0, win - len(sub))
    final = (pad + sub)[-win:]
    oh = []
    for v in final:
        a = [0,0,0]
        if v in (0,1,2): a[v]=1
        oh.append(a)
    arr = np.array(oh, dtype=np.float32)[np.newaxis, :, :]
    return arr

def softmax(x: np.ndarray, temp: float=1.0) -> np.ndarray:
    x = x / max(1e-9, temp)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

# ======= Heuristic estimator（含大路訊號） =======
def heuristic_probs(seq: List[int]) -> Tuple[np.ndarray, str]:
    n = len(seq)
    if n == 0:
        return np.array([0.49,0.49,0.02], dtype=np.float32), "prior only"

    # 大路特徵
    feat = big_road_features(seq, GRID_ROWS, GRID_COLS, FEAT_WIN)
    # 基本頻率
    sub = seq[-FEAT_WIN:] if len(seq)>FEAT_WIN else seq
    cnt = np.bincount(sub, minlength=3).astype(np.float32)
    freq = cnt / max(1, len(sub))
    p0 = 0.90 * freq + 0.10 * np.array([0.49,0.49,0.02], dtype=np.float32)

    # 簡單大路訊號：若當前列高度高（接近底部），偏右移 → 易轉列 → 偏向相反
    gs, _, (r,c) = big_road_grid(sub, GRID_ROWS, GRID_COLS)
    cur_h = (gs[:, c]!=0).sum() if 0 <= c < GRID_COLS else 0
    cur_side = gs[0, c] if 0 <= c < GRID_COLS else 0  # +1=B, -1=P
    if cur_side != 0:
        # 高柱 + 已接近最底 -> 提示轉列：提高 Opp 的機率
        near_bottom = (cur_h >= GRID_ROWS-1)
        boost = 0.05 if near_bottom else 0.02
        if cur_side > 0:  # B 柱
            p0[1] += boost    # 給 P
            p0[0] -= boost/2  # 稍降 B
        else:
            p0[0] += boost
            p0[1] -= boost/2

    # T 上下限夾住
    p0[2] = np.clip(p0[2], CLIP_T_MIN, CLIP_T_MAX)
    # normalize
    p0 = np.clip(p0, 1e-6, None); p0 = p0 / p0.sum()
    return p0.astype(np.float32), "heuristic(big-road)"

# ======= Model inference（與訓練一致） =======
def xgb_probs(seq: List[int]) -> Optional[np.ndarray]:
    if XGB_MODEL is None: return None
    import xgboost as xgb
    feat = big_road_features(seq, GRID_ROWS, GRID_COLS, FEAT_WIN).astype(np.float32)
    d = xgb.DMatrix(feat.reshape(1,-1))
    p = XGB_MODEL.predict(d)  # shape [1,3]
    p = np.array(p[0], dtype=np.float32)
    return p

def lgb_probs(seq: List[int]) -> Optional[np.ndarray]:
    if LGB_MODEL is None: return None
    feat = big_road_features(seq, GRID_ROWS, GRID_COLS, FEAT_WIN).astype(np.float32).reshape(1,-1)
    p = LGB_MODEL.predict(feat)[0]  # [3]
    return np.array(p, dtype=np.float32)

def rnn_probs(seq: List[int]) -> Optional[np.ndarray]:
    if RNN_MODEL is None: return None
    import torch
    x = one_hot_seq(seq, FEAT_WIN)               # [1,win,3]
    with torch.no_grad():
        logits = RNN_MODEL(torch.from_numpy(x))  # [1,3]
        p = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return p.astype(np.float32)

def fuse_probs(ph: np.ndarray,
               px: Optional[np.ndarray],
               pl: Optional[np.ndarray],
               pr: Optional[np.ndarray]) -> np.ndarray:
    w_heu = ENS_W_HEU
    w_xgb = ENS_W_XGB if px is not None else 0.0
    w_lgb = ENS_W_LGB if pl is not None else 0.0
    w_rnn = ENS_W_RNN if pr is not None else 0.0
    total = w_heu + w_xgb + w_lgb + w_rnn
    if total <= 0: return ph
    p = w_heu*ph
    if px is not None: p += w_xgb*px
    if pl is not None: p += w_lgb*pl
    if pr is not None: p += w_rnn*pr
    p = p / total
    # 溫度 + T 夾住
    p = softmax(np.log(np.clip(p,1e-9,None)), TEMP)
    p[2] = np.clip(p[2], CLIP_T_MIN, CLIP_T_MAX)
    p = np.clip(p, 1e-6, None); p = p / p.sum()
    return p.astype(np.float32)

def make_suggestion(p: np.ndarray) -> Tuple[str,str]:
    b, p_, t = float(p[0]), float(p[1]), float(p[2])
    top_idx = int(np.argmax(p))
    labels = ["B","P","T"]
    top = labels[top_idx]
    # 下注條件：與第二名差距 >= MIN_EDGE，且非過高 T
    sorted_probs = sorted([(b,"B"), (p_,"P"), (t,"T")], reverse=True)
    edge = sorted_probs[0][0] - sorted_probs[1][0]
    if top == "T" and t < max(0.05, CLIP_T_MIN + 0.01):
        return "WAIT", f"tie low ({t:.2f})"
    if edge >= MIN_EDGE:
        return top, f"edge {edge:.2f} (top={top}, p={sorted_probs[0][0]:.2f})"
    return "WAIT", f"edge {edge:.2f} < {MIN_EDGE:.2f}"

# ======= API =======
@app.route("/", methods=["GET"])
def root():
    return "BGS AI server ok", 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True, silent=True) or {}
        hist = data.get("history", "")
        seq = parse_history(hist)
        ph, reason = heuristic_probs(seq)
        px = xgb_probs(seq)
        pl = lgb_probs(seq)
        pr = rnn_probs(seq)
        p = fuse_probs(ph, px, pl, pr)
        sug, why = make_suggestion(p)
        return jsonify({
            "probs": {"banker": round(float(p[0]),4),
                      "player": round(float(p[1]),4),
                      "tie":    round(float(p[2]),4)},
            "suggestion": sug,
            "why": f"{reason}; {why}",
            "models": {"xgb": px is not None, "lgbm": pl is not None, "rnn": pr is not None},
            "feat_win": FEAT_WIN,
            "grid": {"rows": GRID_ROWS, "cols": GRID_COLS}
        })
    except Exception as e:
        log.exception("predict error")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT","8000"))
    app.run(host="0.0.0.0", port=port)
