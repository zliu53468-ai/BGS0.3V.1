# server.py — LiveBoot Baccarat AI (XGB/LGBM/RNN 投票 + 平均機率)
# 功能：
# • 3 模型投票決策（XGB / LGBM / RNN），平均機率作最終機率
# • 配注比例 = 邊際分級(10/20/30%) × 投票信心（0.5~1.0倍）
# • LINE Webhook（Emoji & 快速回覆）
# • 30 分鐘試用 / 單一密碼開通（環境變數 ADMIN_ACTIVATION_SECRET）
# • /predict 回傳同款 Emoji 文本
# • /health 健檢
# 啟動（Render）：
#   gunicorn server:app --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 180 --graceful-timeout 45

import os, logging, time
from typing import List, Tuple, Optional, Dict
import numpy as np
from flask import Flask, request, jsonify, abort

# ===== Logging / App =====
log = logging.getLogger("liveboot-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
app = Flask(__name__)

# ===== Config =====
FEAT_WIN   = int(os.getenv("FEAT_WIN", "40"))
GRID_ROWS  = int(os.getenv("GRID_ROWS", "6"))
GRID_COLS  = int(os.getenv("GRID_COLS", "20"))

# 配注分級與限制
MIN_EDGE   = float(os.getenv("MIN_EDGE", "0.07"))   # 最小邊際建倉門檻
TEMP       = float(os.getenv("TEMP", "0.95"))
CLIP_T_MIN = float(os.getenv("CLIP_T_MIN", "0.02"))
CLIP_T_MAX = float(os.getenv("CLIP_T_MAX", "0.12"))
SEED       = int(os.getenv("SEED", "42"))
np.random.seed(SEED)

# 試用 / 開通
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "30"))
ADMIN_CONTACT = os.getenv("ADMIN_CONTACT", "@jins888")
ADMIN_ACTIVATION_SECRET = os.getenv("ADMIN_ACTIVATION_SECRET", "")  # 你在 Render 填的唯一密碼
SHOW_REMAINING_TIME = int(os.getenv("SHOW_REMAINING_TIME", "1"))

# ===== LINE =====
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
try:
    from linebot import LineBotApi, WebhookHandler
    from linebot.exceptions import InvalidSignatureError
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

# ===== Session（in-memory）=====
# { user_id: {"bankroll": int, "seq": List[int], "trial_start": int, "premium": bool} }
SESS: Dict[str, Dict[str, object]] = {}

# ===== 模型（Lazy Load）=====
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
            state = torch.load(path, map_location="cpu")
            RNN_MODEL.load_state_dict(state); RNN_MODEL.eval()
            log.info("[MODEL] RNN loaded: %s", path)
        else:
            log.warning("[MODEL] RNN file not found at %s", path)
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
    import numpy as _np
    grid_sign = _np.zeros((rows, cols), dtype=_np.int8)
    grid_ties = _np.zeros((rows, cols), dtype=_np.int16)
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

def softmax_log(p: np.ndarray, temp: float=1.0) -> np.ndarray:
    # 避免過度尖銳：對機率做溫度縮放（log-space）
    x = np.log(np.clip(p,1e-9,None)) / max(1e-9, temp)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

# ===== 取得各模型機率 =====
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

# ===== 三模型投票 + 平均機率融合 =====
def vote_and_average(seq: List[int]) -> Tuple[np.ndarray, Dict[str,str], Dict[str,int]]:
    """回傳 (p_avg, vote_labels, vote_counts)
    p_avg: 三模型的平均機率（有載入的才參與；至少一個）
    vote_labels: 各模型的投票（'XGB':'莊' 等）
    vote_counts: 各類票數 {'莊':v_b,'閒':v_p,'和':v_t}
    """
    preds = []
    vote_labels = {}
    vote_counts = {'莊':0,'閒':0,'和':0}
    label_map = ["莊","閒","和"]

    px = xgb_probs(seq)
    if px is not None:
        preds.append(px); vote_labels['XGB'] = label_map[int(px.argmax())]; vote_counts[vote_labels['XGB']]+=1
    pl = lgb_probs(seq)
    if pl is not None:
        preds.append(pl); vote_labels['LGBM'] = label_map[int(pl.argmax())]; vote_counts[vote_labels['LGBM']]+=1
    pr = rnn_probs(seq)
    if pr is not None:
        preds.append(pr); vote_labels['RNN']  = label_map[int(pr.argmax())]; vote_counts[vote_labels['RNN']]+=1

    if not preds:
        # 全部模型都沒有 → 退回簡單 heuristic 當保底
        ph, _ = heuristic_probs(seq)
        return ph, {}, {'莊':0,'閒':0,'和':0}

    P = np.stack(preds, axis=0).astype(np.float32)
    # 平均之前做個溫度縮放以避免單一模型過尖（可關閉）
    P = np.stack([softmax_log(p, TEMP) for p in P], axis=0)
    p_avg = P.mean(axis=0)
    # clip tie
    p_avg[2] = np.clip(p_avg[2], CLIP_T_MIN, CLIP_T_MAX)
    p_avg = np.clip(p_avg, 1e-6, None); p_avg = p_avg / p_avg.sum()
    return p_avg, vote_labels, vote_counts

# ===== Heuristic（保底用；三模型都缺時）=====
def heuristic_probs(seq: List[int]) -> Tuple[np.ndarray, str]:
    if not seq:
        return np.array([0.49,0.49,0.02], dtype=np.float32), "prior"
    sub = seq[-FEAT_WIN:] if len(seq)>FEAT_WIN else seq
    cnt = np.bincount(sub, minlength=3).astype(np.float32)
    freq = cnt / max(1,len(sub))
    p0 = 0.90*freq + 0.10*np.array([0.49,0.49,0.02], dtype=np.float32)
    # tie clip
    p0[2] = np.clip(p0[2], CLIP_T_MIN, CLIP_T_MAX)
    p0 = np.clip(p0,1e-6,None); p0 = p0/p0.sum()
    return p0, "heuristic"

# ===== 配注（依投票%）=====
def edge_to_base_pct(edge: float) -> float:
    if edge >= max(0.10, MIN_EDGE+0.02): return 0.30
    if edge >= max(0.08, MIN_EDGE):      return 0.20
    if edge >= max(0.05, MIN_EDGE-0.01): return 0.10
    return 0.0

def decide_bet_from_votes(p: np.ndarray, votes: Dict[str,int], models_used:int) -> Tuple[str,float,float, float]:
    """回傳 (建議, 邊際, 最終下注比例, 投票信心)"""
    labels = ["莊","閒","和"]
    arr = [(float(p[0]),"莊"), (float(p[1]),"閒"), (float(p[2]),"和")]
    arr.sort(reverse=True, key=lambda x: x[0])
    (p1, lab1), (p2, _) = arr[0], arr[1]
    edge = p1 - p2

    # 投票信心：最高票 / 參與模型數
    max_votes = max(votes.get("莊",0), votes.get("閒",0), votes.get("和",0)) if models_used>0 else 0
    vote_conf = (max_votes / models_used) if models_used>0 else 0.0

    # 和的保護：機率過低不主推
    if lab1 == "和" and p[2] < max(0.05, CLIP_T_MIN + 0.01):
        return "觀望", edge, 0.0, vote_conf

    base_pct = edge_to_base_pct(edge)
    if base_pct == 0.0:
        return "觀望", edge, 0.0, vote_conf

    # 依投票強度調整倉位（1/3→0.66倍, 2/3→0.83倍, 3/3→1.0倍）
    scale = 0.5 + 0.5*vote_conf
    bet_pct = base_pct * scale
    # 上下限保護
    bet_pct = float(np.clip(bet_pct, 0.05 if base_pct>0 else 0.0, 0.30))
    return lab1, edge, bet_pct, vote_conf

# ===== Emoji 訊息 =====
def fmt_line_reply(n_hand:int, p:np.ndarray, sug:str, edge:float,
                   bankroll:int, bet_pct:float, vote_labels:Dict[str,str],
                   vote_counts:Dict[str,int], models_used:int, remain_min:Optional[int]) -> str:
    b, pl, t = p[0], p[1], p[2]
    lines = []
    lines.append(f"📊 已解析 {n_hand} 手（0 ms）")
    lines.append(f"📈 平均機率：莊 {b:.3f}｜閒 {pl:.3f}｜和 {t:.3f}")

    if models_used>0:
        vB, vP, vT = vote_counts.get('莊',0), vote_counts.get('閒',0), vote_counts.get('和',0)
        vote_line = f"🗳️ 投票（{models_used} 模型）：莊 {vB}｜閒 {vP}｜和 {vT}"
        who = []
        for k in ["XGB","LGBM","RNN"]:
            if k in vote_labels: who.append(f"{k}→{vote_labels[k]}")
        if who: vote_line += "｜" + "，".join(who)
        lines.append(vote_line)

    badge = "🎯" if sug != "觀望" else "🟡"
    lines.append(f"👉 下一手建議：{sug} {badge}（邊際 {edge:.3f}）")

    if bankroll and bet_pct>0:
        bet_amt = int(round(bankroll * bet_pct))
        lines.append(f"💵 本金：{bankroll:,}")
        lines.append(f"✅ 建議下注：{bet_amt:,} ＝ {bankroll:,} × {bet_pct*100:.1f}%")
        lines.append(f"🧮 10%={int(round(bankroll*0.10)):,}｜20%={int(round(bankroll*0.20)):,}｜30%={int(round(bankroll*0.30)):,}")

    if remain_min is not None and SHOW_REMAINING_TIME:
        lines.append(f"⏳ 試用剩餘：約 {max(0, remain_min)} 分鐘")

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
    return "LiveBoot ok", 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok"), 200

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json(silent=True) or {}
    history = data.get("history", "")
    bankroll = int(data.get("bankroll", 0) or 0)
    seq = parse_history(history)

    p_avg, vote_labels, vote_counts = vote_and_average(seq)
    models_used = len(vote_labels)
    sug, edge, bet_pct, vote_conf = decide_bet_from_votes(p_avg, vote_counts, models_used)
    text = fmt_line_reply(len(seq), p_avg, sug, edge, bankroll, bet_pct, vote_labels, vote_counts, models_used, None)

    return jsonify({
        "hands": len(seq),
        "probs": {"banker": round(float(p_avg[0]),3), "player": round(float(p_avg[1]),3), "tie": round(float(p_avg[2]),3)},
        "suggestion": sug,
        "edge": round(float(edge),3),
        "bet_pct": float(bet_pct),
        "bet_amount": int(round(bankroll*bet_pct)) if bankroll and bet_pct>0 else 0,
        "votes": {"models_used": models_used, **vote_counts},
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
        "先輸入本金（例如：5000 或 20000），再貼歷史（B/P/T 或 莊/閒/和），輸入『開始分析』即可！📊\n"
        "🔐 試用到期後，請聯繫管理員取得開通密碼："
        f"{ADMIN_CONTACT}"
    )
    line_api.reply_message(event.reply_token, TextSendMessage(text=msg, quick_reply=quick_reply_buttons()))

@line_handler.add(MessageEvent, message=TextMessage)
def on_text(event):
    uid = event.source.user_id
    text = (event.message.text or "").strip()
    sess = SESS.setdefault(uid, {"bankroll": 0, "seq": [], "trial_start": int(time.time()), "premium": False})

    # 試用檢查
    if not sess.get("premium", False):
        start = int(sess.get("trial_start", int(time.time())))
        now   = int(time.time())
        elapsed_min = (now - start) // 60
        remain_min = max(0, TRIAL_MINUTES - elapsed_min)
        if elapsed_min >= TRIAL_MINUTES:
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
    else:
        remain_min = None

    # 本金設定
    if text.isdigit():
        sess["bankroll"] = int(text)
        msg = f"👍 已設定本金：{int(text):,} 元。\n貼上歷史（B/P/T 或 莊/閒/和）後輸入『開始分析』即可！🚀"
        safe_reply(event.reply_token, msg, uid)
        return

    # 開通碼
    if text.startswith("開通") or text.lower().startswith("activate"):
        code = text.split(" ",1)[1].strip() if " " in text else ""
        if validate_activation_code(code):
            sess["premium"] = True
            safe_reply(event.reply_token, "✅ 已開通成功！現在可以繼續使用所有功能。🎉", uid)
        else:
            safe_reply(event.reply_token, "❌ 開通密碼不正確，請向管理員索取正確密碼。", uid)
        return

    # 歷史/單手
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

    # 分析
    if ("開始分析" in text) or (text in ["分析", "開始", "GO", "go"]):
        sseq: List[int] = sess.get("seq", [])
        bankroll: int = int(sess.get("bankroll", 0) or 0)
        p_avg, vote_labels, vote_counts = vote_and_average(sseq)
        models_used = len(vote_labels)
        sug, edge, bet_pct, vote_conf = decide_bet_from_votes(p_avg, vote_counts, models_used)
        reply = fmt_line_reply(len(sseq), p_avg, sug, edge, bankroll, bet_pct, vote_labels, vote_counts, models_used, remain_min)
        safe_reply(event.reply_token, reply, uid)
        return

    # 說明
    msg = (
        "🧭 指令說明：\n"
        "• 輸入『數字』設定本金（例：5000）\n"
        "• 貼上歷史：B/P/T 或 莊/閒/和（可含空白）\n"
        "• 輸入『開始分析』取得建議（採 XGB/LGBM/RNN 投票＋平均機率）\n"
        "• 試用到期後輸入：『開通 你的密碼』\n"
        f"• 管理員官方 LINE：{ADMIN_CONTACT}"
    )
    safe_reply(event.reply_token, msg, uid)

def validate_activation_code(code: str) -> bool:
    # 只有當 ADMIN_ACTIVATION_SECRET 設定且完全相等才通過；否則一律拒絕
    if not ADMIN_ACTIVATION_SECRET:
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

# 再多一個 health 以防部分平台重複探測
@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify(status="ok"), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT","8000"))
    app.run(host="0.0.0.0", port=port)
