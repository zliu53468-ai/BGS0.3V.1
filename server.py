# server.py — BGS AI (Big Road 6x20 + Heuristic + XGB/LGBM + RNN)
# + LINE Webhook（Emoji & 快速回覆）
# + 30 分鐘免費試用 / 單一密碼開通（只從環境變數讀取）
# + /predict 同款 Emoji 文本
# + /health 健康檢查
# 啟動：
#   gunicorn server:app --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 180 --graceful-timeout 45

import os, logging, time
from typing import List, Tuple, Optional, Dict
import numpy as np
from flask import Flask, request, jsonify, abort

# ===== Logging / App =====
log = logging.getLogger("bgs-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
app = Flask(__name__)

# ===== Config =====
FEAT_WIN   = int(os.getenv("FEAT_WIN", "40"))
GRID_ROWS  = int(os.getenv("GRID_ROWS", "6"))
GRID_COLS  = int(os.getenv("GRID_COLS", "20"))

# Ensemble 權重（有對應模型檔才會參與）
ENS_W_HEU  = float(os.getenv("ENS_W_HEU", "0.55"))
ENS_W_XGB  = float(os.getenv("ENS_W_XGB", "0.25"))
ENS_W_LGB  = float(os.getenv("ENS_W_LGB", "0.20"))
ENS_W_RNN  = float(os.getenv("ENS_W_RNN", "0.15"))  # 預設啟用 RNN 權重

MIN_EDGE   = float(os.getenv("MIN_EDGE", "0.07"))
TEMP       = float(os.getenv("TEMP", "0.95"))
CLIP_T_MIN = float(os.getenv("CLIP_T_MIN", "0.02"))
CLIP_T_MAX = float(os.getenv("CLIP_T_MAX", "0.12"))
SEED       = int(os.getenv("SEED", "42"))
np.random.seed(SEED)

# ===== 試用 / 開通設定 =====
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "30"))
ADMIN_CONTACT = os.getenv("ADMIN_CONTACT", "@jins888")
# 唯一密碼只從環境變數讀取，程式碼不內嵌任何預設值
ADMIN_ACTIVATION_SECRET = os.getenv("ADMIN_ACTIVATION_SECRET", "")  # 例：在 Render 設定

# ===== LINE =====
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
try:
    from linebot import LineBotApi, WebhookHandler
    from linebot.exceptions import InvalidSignatureError, LineBotApiError
    from linebot.models import (
        MessageEvent, TextMessage, FollowEvent, TextSendMessage,
        QuickReply, QuickReplyButton, MessageAction
    )
    line_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN) if LINE_CHANNEL_ACCESS_TOKEN else None
    line_handler = WebhookHandler(LINE_CHANNEL_SECRET) if LINE_CHANNEL_SECRET else None
except Exception as e:
    line_api = None
    line_handler = None
    log.warning("LINE SDK not fully available: %s", e)

# ===== Session =====
SESS: Dict[str, Dict[str, object]] = {}  # { user_id: {bankroll, seq, trial_start, premium} }

# ===== Optional models (lazy load) =====
XGB_MODEL = None
LGB_MODEL = None
RNN_MODEL = None

def _load_xgb():
    global XGB_MODEL
    try:
        import xgboost as xgb, os
        path = os.getenv("XGB_OUT_PATH", "/data/models/xgb.json")
        if os.path.exists(path):
            booster = xgb.Booster(); booster.load_model(path)
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
    """載入 /data/models/rnn.pt（Tiny GRU）"""
    global RNN_MODEL
    try:
        import torch
        import torch.nn as nn
        class TinyRNN(nn.Module):
            def __init__(self, in_dim=3, hid=32, out_dim=3):
                super().__init__()
                self.gru = nn.GRU(input_size=in_dim, hidden_size=hid, num_layers=1, batch_first=True)
                self.fc  = nn.Linear(hid, out_dim)
            def forward(self, x):
                o,_ = self.gru(x); return self.fc(o[:, -1, :])
        path = os.getenv("RNN_OUT_PATH", "/data/models/rnn.pt")
        if os.path.exists(path):
            RNN_MODEL = TinyRNN()
            import torch as _torch
            state = _torch.load(path, map_location="cpu")
            RNN_MODEL.load_state_dict(state); RNN_MODEL.eval()
            log.info("[MODEL] RNN loaded: %s", path)
        else:
            log.warning("[MODEL] RNN file not found at %s (skipping)", path)
    except Exception as e:
        log.warning("[MODEL] RNN load failed: %s", e)

_load_xgb(); _load_lgb(); _load_rnn()

# ===== Big Road & Features =====
MAP = {"B":0, "P":1, "T":2}
INV = {0:"B", 1:"P", 2:"T"}

def parse_history(s: str) -> List[int]:
    s = (s or "").strip().upper()
    if not s: return []
    toks = s.split()
    seq = list(s) if len(toks)==1 else toks
    out=[]
    for ch in seq:
        ch = ch.strip().upper()
        if ch in MAP: out.append(MAP[ch])
    return out

def big_road_grid(seq: List[int], rows:int=6, cols:int=20):
    grid_sign = np.zeros((rows, cols), dtype=np.int8)
    grid_ties = np.zeros((rows, cols), dtype=np.int16)
    r = 0; c = 0; last_bp = None
    for v in seq:
        if v == 2:
            if 0 <= r < rows and 0 <= c < cols: grid_ties[r, c] += 1
            continue
        cur_bp = +1 if v==0 else -1
        if last_bp is None:
            r,c=0,0; grid_sign[r,c]=cur_bp; last_bp=cur_bp; continue
        if cur_bp == last_bp:
            nr=r+1; nc=c
            if nr>=rows or grid_sign[nr,nc]!=0:
                nr=r; nc=c+1
            r,c=nr,nc
            if 0<=r<rows and 0<=c<cols: grid_sign[r,c]=cur_bp
        else:
            c=c+1; r=0
            if c<cols: grid_sign[r,c]=cur_bp
            last_bp = cur_bp
    return grid_sign, grid_ties, (r,c)

def big_road_features(seq: List[int], rows:int=6, cols:int=20, win:int=40) -> np.ndarray:
    sub = seq[-win:] if len(seq)>win else seq[:]
    gs, gt, (r,c) = big_road_grid(sub, rows, cols)
    grid_sign_flat = gs.flatten().astype(np.float32)
    grid_tie_flat  = np.clip(gt.flatten(), 0, 3).astype(np.float32) / 3.0
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
    return np.concatenate([
        grid_sign_flat, grid_tie_flat,
        np.array([streak_len/rows, streak_side], dtype=np.float32),
        col_heights,
        np.array([cur_col_height, cur_col_side], dtype=np.float32),
        freq
    ], axis=0)

def one_hot_seq(seq: List[int], win:int) -> np.ndarray:
    sub = seq[-win:] if len(seq)>win else seq[:]
    pad = [-1]*max(0, win-len(sub))
    final = (pad+sub)[-win:]
    oh=[]
    for v in final:
        a=[0,0,0]
        if v in (0,1,2): a[v]=1
        oh.append(a)
    return np.array(oh, dtype=np.float32)[np.newaxis, :, :]

def softmax(x: np.ndarray, temp: float=1.0) -> np.ndarray:
    x = x / max(1e-9, temp)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

# ===== Heuristic =====
def heuristic_probs(seq: List[int]) -> Tuple[np.ndarray, str]:
    if not seq:
        return np.array([0.49,0.49,0.02], dtype=np.float32), "prior"
    sub = seq[-FEAT_WIN:] if len(seq)>FEAT_WIN else seq
    cnt = np.bincount(sub, minlength=3).astype(np.float32)
    freq = cnt / max(1,len(sub))
    p0 = 0.90*freq + 0.10*np.array([0.49,0.49,0.02], dtype=np.float32)
    # 大路「高柱轉列」微調
    gs,_,(r,c) = big_road_grid(sub, GRID_ROWS, GRID_COLS)
    cur_h = (gs[:,c]!=0).sum() if 0<=c<GRID_COLS else 0
    cur_side = gs[0,c] if 0<=c<GRID_COLS else 0
    if cur_side != 0:
        near_bottom = (cur_h >= GRID_ROWS-1)
        boost = 0.05 if near_bottom else 0.02
        if cur_side > 0:  # B 柱
            p0[1] += boost; p0[0] -= boost/2
        else:
            p0[0] += boost; p0[1] -= boost/2
    p0[2] = np.clip(p0[2], CLIP_T_MIN, CLIP_T_MAX)
    p0 = np.clip(p0, 1e-6, None); p0 = p0 / p0.sum()
    return p0.astype(np.float32), "heuristic(big-road)"

# ===== Models =====
def xgb_probs(seq: List[int]) -> Optional[np.ndarray]:
    if XGB_MODEL is None: return None
    import xgboost as xgb
    feat = big_road_features(seq, GRID_ROWS, GRID_COLS, FEAT_WIN).astype(np.float32)
    d = xgb.DMatrix(feat.reshape(1,-1))
    p = XGB_MODEL.predict(d)[0]
    return np.array(p, dtype=np.float32)

def lgb_probs(seq: List[int]) -> Optional[np.ndarray]:
    if LGB_MODEL is None: return None
    feat = big_road_features(seq, GRID_ROWS, GRID_COLS, FEAT_WIN).astype(np.float32).reshape(1,-1)
    p = LGB_MODEL.predict(feat)[0]
    return np.array(p, dtype=np.float32)

def rnn_probs(seq: List[int]) -> Optional[np.ndarray]:
    if RNN_MODEL is None: return None
    import torch
    x = one_hot_seq(seq, FEAT_WIN)
    with torch.no_grad():
        logits = RNN_MODEL(torch.from_numpy(x))
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
    p = softmax(np.log(np.clip(p,1e-9,None)), TEMP)
    p[2] = np.clip(p[2], CLIP_T_MIN, CLIP_T_MAX)
    p = np.clip(p, 1e-6, None); p = p / p.sum()
    return p.astype(np.float32)

def decide_bet(p: np.ndarray) -> Tuple[str, float, float]:
    """回傳 (建議: '莊'/'閒'/'和'/'觀望', 邊際, 建議下注比例%)"""
    arr = [(float(p[0]),"莊"), (float(p[1]),"閒"), (float(p[2]),"和")]
    arr.sort(reverse=True, key=lambda x: x[0])
    top_p, top_lab = arr[0]
    edge = top_p - arr[1][0]
    if top_lab == "和" and p[2] < max(0.05, CLIP_T_MIN + 0.01):
        return "觀望", edge, 0.0
    if edge >= max(0.10, MIN_EDGE+0.02):
        bet_pct = 0.30
    elif edge >= max(0.08, MIN_EDGE):
        bet_pct = 0.20
    elif edge >= max(0.05, MIN_EDGE-0.01):
        bet_pct = 0.10
    else:
        return "觀望", edge, 0.0
    return top_lab, edge, bet_pct

# ===== Emoji 訊息樣式 =====
def fmt_line_reply(n_hand:int, p:np.ndarray, sug:str, edge:float, bankroll:int, bet_pct:float) -> str:
    b, pl, t = p[0], p[1], p[2]
    lines = []
    lines.append(f"📊 已解析 {n_hand} 手（0 ms）")
    lines.append(f"📈 機率：莊 {b:.3f}｜閒 {pl:.3f}｜和 {t:.3f}")
    badge = "🎯" if sug != "觀望" else "🟡"
    lines.append(f"👉 下一手建議：{sug} {badge}")
    if bankroll and bet_pct>0:
        bet_amt = int(round(bankroll * bet_pct))
        lines.append(f"💵 本金：{bankroll:,}")
        lines.append(f"✅ 建議下注：{bet_amt:,} ＝ {bankroll:,} × {bet_pct*100:.1f}%")
        lines.append(f"🧮 10%={int(round(bankroll*0.10)):,}｜20%={int(round(bankroll*0.20)):,}｜30%={int(round(bankroll*0.30)):,}")
    lines.append("📝 直接輸入下一手結果（莊/閒/和 或 B/P/T），我會再幫你算下一局。")
    return "\n".join(lines)

def fmt_trial_over() -> str:
    return (
        "⛔ 免費試用已結束。\n"
        f"📬 請先聯繫管理員官方 LINE：{ADMIN_CONTACT} 取得開通密碼後再使用。\n"
        "🔐 開通方式：收到密碼後，直接輸入：\n"
        "【開通 你的密碼】（例如：開通 abc123）"
    )

def quick_reply_buttons():
    try:
        return QuickReply(items=[
            QuickReplyButton(action=MessageAction(label="莊", text="莊")),
            QuickReplyButton(action=MessageAction(label="閒", text="閒")),
            QuickReplyButton(action=MessageAction(label="和", text="和")),
            QuickReplyButton(action=MessageAction(label="開始分析", text="開始分析")),
        ])
    except Exception:
        return None

# ===== API =====
@app.route("/", methods=["GET"])
def root():
    return "BGS AI server ok", 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok"), 200

@app.route("/predict", methods=["POST"])
def predict_api():
    # /predict 不做試用限制（供你前端/內部）
    data = request.get_json(silent=True) or {}
    history = data.get("history", "")
    bankroll = int(data.get("bankroll", 0) or 0)
    seq = parse_history(history)
    ph,_ = heuristic_probs(seq)
    p = fuse_probs(ph, xgb_probs(seq), lgb_probs(seq), rnn_probs(seq))
    sug, edge, bet_pct = decide_bet(p)
    text = fmt_line_reply(len(seq), p, sug, edge, bankroll, bet_pct)
    return jsonify({
        "hands": len(seq),
        "probs": {"banker": round(float(p[0]),3), "player": round(float(p[1]),3), "tie": round(float(p[2]),3)},
        "suggestion": sug,
        "edge": round(edge,3),
        "bet_pct": bet_pct,
        "bet_amount": int(round(bankroll*bet_pct)) if bankroll and bet_pct>0 else 0,
        "message": text
    })

# ===== LINE webhook =====
@app.route("/line-webhook", methods=["POST"])
def line_webhook():
    if not line_handler or not line_api:
        abort(503, "LINE not configured")
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        line_handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400, "Invalid signature")
    return "OK", 200

@line_handler.add(FollowEvent)
def on_follow(event):
    uid = event.source.user_id
    now = int(time.time())
    SESS[uid] = {"bankroll": 0, "seq": [], "trial_start": now, "premium": False}
    mins = TRIAL_MINUTES
    msg = (
        "🤖 歡迎加入！\n"
        f"🎁 已啟用 {mins} 分鐘免費試用，現在就開始吧！\n"
        "請先輸入你的本金（例如：5000 或 20000），我會用它計算下注建議。💡\n"
        "接著貼上歷史（B/P/T 或 莊/閒/和），然後輸入『開始分析』即可！📊\n"
        "🔐 試用到期後，請聯繫管理員取得開通密碼："
        f"{ADMIN_CONTACT}"
    )
    line_api.reply_message(event.reply_token, TextSendMessage(text=msg, quick_reply=quick_reply_buttons()))

@line_handler.add(MessageEvent, message=TextMessage)
def on_text(event):
    uid = event.source.user_id
    text = (event.message.text or "").strip()
    sess = SESS.setdefault(uid, {"bankroll": 0, "seq": [], "trial_start": int(time.time()), "premium": False})

    # ===== 檢查試用到期（未開通者鎖定）=====
    if not sess.get("premium", False):
        start = int(sess.get("trial_start", int(time.time())))
        elapsed_min = (int(time.time()) - start) / 60.0
        if elapsed_min >= TRIAL_MINUTES:
            # 只接受「開通 <密碼>」且密碼需等於 ADMIN_ACTIVATION_SECRET
            if text.startswith("開通") or text.lower().startswith("activate"):
                code = text.split(" ",1)[1].strip() if " " in text else ""
                if validate_activation_code(code):
                    sess["premium"] = True
                    safe_reply(event.reply_token, "✅ 已開通成功！現在可以繼續使用所有功能。🎉", uid)
                else:
                    safe_reply(event.reply_token, "❌ 開通密碼不正確，請向管理員索取正確密碼。", uid)
            else:
                safe_reply(event.reply_token, fmt_trial_over(), uid)
            return

    # ===== 正常流程 =====
    # 1) 數字 → 設定本金
    if text.isdigit():
        sess["bankroll"] = int(text)
        msg = f"👍 已設定本金：{int(text):,} 元。\n接著貼上歷史（B/P/T 或 莊/閒/和），然後輸入『開始分析』即可！🚀"
        safe_reply(event.reply_token, msg, uid)
        return

    # 2) 開通密碼（試用未到期也可先開通）
    if text.startswith("開通") or text.lower().startswith("activate"):
        code = text.split(" ",1)[1].strip() if " " in text else ""
        if validate_activation_code(code):
            sess["premium"] = True
            safe_reply(event.reply_token, "✅ 已開通成功！現在可以繼續使用所有功能。🎉", uid)
        else:
            safe_reply(event.reply_token, "❌ 開通密碼不正確，請向管理員索取正確密碼。", uid)
        return

    # 3) 歷史或單手結果
    zh2eng = {"莊":"B","閒":"P","和":"T"}
    norm = "".join(zh2eng.get(ch, ch) for ch in text.upper())
    seq = parse_history(norm)

    if seq and ("開始分析" not in text):
        if len(seq) == 1:
            sess.setdefault("seq", [])
            sess["seq"].append(seq[0])
        else:
            sess["seq"] = seq
        n = len(sess["seq"])
        msg = f"✅ 已接收歷史共 {n} 手，目前累計 {n} 手。\n輸入『開始分析』即可啟動。🧪"
        safe_reply(event.reply_token, msg, uid)
        return

    # 4) 開始分析
    if ("開始分析" in text) or (text in ["分析", "開始", "GO", "go"]):
        sseq: List[int] = sess.get("seq", [])
        bankroll: int = int(sess.get("bankroll", 0) or 0)
        ph,_ = heuristic_probs(sseq)
        p = fuse_probs(ph, xgb_probs(sseq), lgb_probs(sseq), rnn_probs(sseq))
        sug, edge, bet_pct = decide_bet(p)
        reply = fmt_line_reply(len(sseq), p, sug, edge, bankroll, bet_pct)
        safe_reply(event.reply_token, reply, uid)
        return

    # 5) 說明
    msg = (
        "🧭 指令說明：\n"
        "• 輸入『數字』設定本金（例：5000）\n"
        "• 貼上歷史：B/P/T 或 莊/閒/和（可有空白）\n"
        "• 輸入『開始分析』取得建議\n"
        "• 試用到期後輸入：『開通 你的密碼』\n"
        f"• 管理員官方 LINE：{ADMIN_CONTACT}"
    )
    safe_reply(event.reply_token, msg, uid)

def validate_activation_code(code: str) -> bool:
    # 只有當 ADMIN_ACTIVATION_SECRET 設定且完全相等才通過；否則一律拒絕
    if not ADMIN_ACTIVATION_SECRET:
        # 若你忘了設定密碼，為安全起見，直接拒絕
        return False
    return bool(code) and (code == ADMIN_ACTIVATION_SECRET)

def safe_reply(reply_token: str, text: str, uid: Optional[str] = None):
    try:
        line_api.reply_message(reply_token, TextSendMessage(text=text, quick_reply=quick_reply_buttons()))
    except Exception as e:
        log.warning("[LINE] reply failed, try push: %s", e)
        if uid:
            try:
                line_api.push_message(uid, TextSendMessage(text=text, quick_reply=quick_reply_buttons()))
            except Exception as e2:
                log.error("[LINE] push failed: %s", e2)

# ===== Main =====
@app.route("/health", methods=["GET"])
def _health_dup_for_gunicorn():
    # 某些平台可能探測兩次，保險起見
    return jsonify(status="ok"), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT","8000"))
    app.run(host="0.0.0.0", port=port)
