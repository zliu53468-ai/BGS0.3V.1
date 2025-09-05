
# app.py — LINE Bot（按鈕版｜無OCR）
# 集成：規則法 +（可選）RNN +（可選）XGBoost +（可選）LightGBM
# - 使用者以按鈕輸入莊/閒/和；輸入「開始分析」後才回覆下注建議
# - 模型檔若不存在則自動跳過，照樣可用（只用規則法）
import os, logging, math
from typing import Dict, List, Tuple, Optional

import numpy as np

from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    FollowEvent, PostbackEvent,
    FlexSendMessage, BubbleContainer, BoxComponent,
    ButtonComponent, TextComponent, PostbackAction
)

# ===== App & Logging =====
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bgs-bot")

# ===== LINE Creds =====
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN) if LINE_CHANNEL_ACCESS_TOKEN else None
line_handler = WebhookHandler(LINE_CHANNEL_SECRET) if LINE_CHANNEL_SECRET else None

# ===== Sessions =====
analysis_enabled: Dict[str, bool] = {}      # 是否「開始分析」
user_history_seq: Dict[str, List[str]] = {} # 使用者輸入的歷史序列（B/P/T）

# ===== 模型參數（可用環境變數覆蓋） =====
RNN_PATH   = os.getenv("RNN_PATH",   "models/rnn_model.h5")
XGB_PATH   = os.getenv("XGB_PATH",   "models/xgb_model.pkl")
LGBM_PATH  = os.getenv("LGBM_PATH",  "models/lgbm_model.pkl")
RNN_MIN_SEQ = int(os.getenv("RNN_MIN_SEQ", "10"))
XGB_MIN_SEQ = int(os.getenv("XGB_MIN_SEQ", "6"))
LGBM_MIN_SEQ= int(os.getenv("LGBM_MIN_SEQ","6"))
RULE_W = float(os.getenv("RULE_W", "0.5"))
RNN_W  = float(os.getenv("RNN_W",  "0.3"))
XGB_W  = float(os.getenv("XGB_W",  "0.15"))
LGBM_W = float(os.getenv("LGBM_W", "0.05"))
FEAT_WIN = int(os.getenv("FEAT_WIN","20"))   # 特徵窗口長度（XGB/LGBM 使用）

# ===== 可選：RNN 載入 =====
rnn_model = None
try:
    from tensorflow.keras.models import load_model
    if os.path.exists(RNN_PATH):
        rnn_model = load_model(RNN_PATH)
        logger.info(f"[RNN] loaded: {RNN_PATH}")
    else:
        logger.info(f"[RNN] not found: {RNN_PATH}")
except Exception as e:
    logger.warning(f"[RNN] unavailable: {e}")

# ===== 可選：XGBoost / LightGBM 載入 =====
xgb_model = None
lgbm_model = None
try:
    import joblib
    if os.path.exists(XGB_PATH):
        xgb_model = joblib.load(XGB_PATH)
        logger.info(f"[XGB] loaded: {XGB_PATH}")
    else:
        logger.info(f"[XGB] not found: {XGB_PATH}")
    if os.path.exists(LGBM_PATH):
        lgbm_model = joblib.load(LGBM_PATH)
        logger.info(f"[LGBM] loaded: {LGBM_PATH}")
    else:
        logger.info(f"[LGBM] not found: {LGBM_PATH}")
except Exception as e:
    logger.warning(f"[XGB/LGBM] unavailable: {e}")

# ===== 工具：統一 class 對應（假設分類順序為 [B,P,T]；若你的模型 classes_ 不同，請用環境變數覆蓋） =====
CLASS_ORDER = os.getenv("CLASS_ORDER", "B,P,T").split(",")
CLASS_TO_IDX = {c:i for i,c in enumerate(CLASS_ORDER)}
def proba_to_dict(proba: np.ndarray) -> Dict[str,float]:
    # 將 predict_proba 的輸出 map 到 banker/player/tie
    # 假設 proba 對應於 CLASS_ORDER（例如 [B,P,T]），否則請調整 CLASS_ORDER
    pB = float(proba[CLASS_TO_IDX.get("B",0)]) if len(proba)>CLASS_TO_IDX.get("B",0) else 0.33
    pP = float(proba[CLASS_TO_IDX.get("P",1)]) if len(proba)>CLASS_TO_IDX.get("P",1) else 0.33
    pT = float(proba[CLASS_TO_IDX.get("T",2)]) if len(proba)>CLASS_TO_IDX.get("T",2) else 0.34
    s = pB+pP+pT
    if s<=0: return {"banker":0.34,"player":0.34,"tie":0.32}
    return {"banker":pB/s, "player":pP/s, "tie":pT/s}

# ===== 規則特徵 & 輕量規則 =====
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

def rule_probs(seq: List[str]) -> Dict[str,float]:
    if not seq:
        return {"banker":0.34, "player":0.34, "tie":0.32}
    pb,pp,pt = _ratio_lastN(seq, len(seq))
    # 尾部連莊/連閒微調
    tail = _streak_tail(seq)
    if seq[-1] in {"B","P"}:
        boost = min(0.10, 0.03*(tail-1))
        if seq[-1]=="B": pb += boost
        else: pp += boost
    # 和局合理範圍
    pt = max(0.02, min(0.15, pt))
    s = pb+pp+pt
    return {"banker":pb/s, "player":pp/s, "tie":pt/s}

# ===== 特徵工程（給 XGB / LGBM ）=====
SYMBOL = {"B":0, "P":1, "T":2}
def seq_features(seq: List[str], win: int=20) -> np.ndarray:
    """將序列轉成固定長度特徵向量（需與你訓練時一致）"""
    n = len(seq)
    b_all,p_all,t_all = _ratio_lastN(seq, n)
    b_n,p_n,t_n       = _ratio_lastN(seq, win)
    tail  = _streak_tail(seq)
    alt   = _alt_streak(seq)
    last  = SYMBOL.get(seq[-1], -1) if n>0 else -1
    # 移動窗內的連續長度估計
    max_streak = 0
    cur = 0
    for i in range(max(0,n-win), n):
        if i==max(0,n-win) or seq[i]==seq[i-1]:
            cur += 1
        else:
            max_streak = max(max_streak, cur)
            cur = 1
    max_streak = max(max_streak, cur)
    # onehot of last k (k=5)
    k=5
    lastK = seq[-k:]
    lastK_vec = [SYMBOL.get(s, -1) for s in lastK]
    # pad to k with -1
    lastK_vec = ([-1]*(k-len(lastK_vec))) + lastK_vec
    # 轉換為三類 onehot 再攤平成 15 維
    lastK_oh = np.zeros((k,3), dtype=float)
    for i,v in enumerate(lastK_vec):
        if 0<=v<3: lastK_oh[i,v]=1.0
    feats = np.array([
        n, tail, alt, last, max_streak,
        b_all, p_all, t_all, b_n, p_n, t_n
    ], dtype=float)
    return np.concatenate([feats, lastK_oh.reshape(-1)])  # 維度：11 + 15 = 26

# ===== RNN 概率 =====
def seq_to_onehot(seq: List[str], N: int) -> np.ndarray:
    seq = seq[-N:]
    arr = np.zeros((N, 3), dtype="float32")
    start = N - len(seq)
    for i, s in enumerate(seq, start):
        if s in ("B","P","T"):
            arr[i, SYMBOL[s]] = 1.0
    return arr[None, ...]

def rnn_probs(seq: List[str]) -> Optional[Dict[str,float]]:
    if rnn_model is None or len(seq) < RNN_MIN_SEQ:
        return None
    try:
        x = seq_to_onehot(seq, RNN_MIN_SEQ)
        y = rnn_model.predict(x, verbose=0)[0]
        y = np.asarray(y, dtype=float)
        s = float(np.sum(y)); s = s if s>0 else 1.0
        return {"banker": float(y[0]/s), "player": float(y[1]/s), "tie": float(y[2]/s)}
    except Exception as e:
        logger.warning(f"[RNN] predict failed: {e}")
        return None

# ===== XGB / LGBM 概率 =====
def xgb_probs(seq: List[str]) -> Optional[Dict[str,float]]:
    if xgb_model is None or len(seq) < XGB_MIN_SEQ:
        return None
    try:
        X = seq_features(seq, FEAT_WIN).reshape(1, -1)
        if hasattr(xgb_model, "predict_proba"):
            proba = xgb_model.predict_proba(X)[0]
        else:
            # 退而求其次：假設輸出為 logits
            logits = xgb_model.predict(X)
            e = np.exp(logits - np.max(logits))
            proba = e / e.sum()
        return proba_to_dict(proba)
    except Exception as e:
        logger.warning(f"[XGB] predict failed: {e}")
        return None

def lgbm_probs(seq: List[str]) -> Optional[Dict[str,float]]:
    if lgbm_model is None or len(seq) < LGBM_MIN_SEQ:
        return None
    try:
        X = seq_features(seq, FEAT_WIN).reshape(1, -1)
        if hasattr(lgbm_model, "predict_proba"):
            proba = lgbm_model.predict_proba(X)[0]
        else:
            preds = lgbm_model.predict(X)  # 若為 multiclass，可能回傳 (n,3)
            proba = preds[0] if np.ndim(preds)>1 else preds
        return proba_to_dict(np.asarray(proba))
    except Exception as e:
        logger.warning(f"[LGBM] predict failed: {e}")
        return None

# ===== 集成 =====
def normalize_weights(avail: List[Tuple[float, Dict[str,float]]]) -> List[Tuple[float, Dict[str,float]]]:
    s = sum(w for w,_ in avail)
    if s <= 0:  # fallback 等分
        k = len(avail)
        return [ (1.0/k, p) for _,p in avail ]
    return [ (w/s, p) for w,p in avail ]

def ensemble_probs(seq: List[str]) -> Dict[str,float]:
    parts: List[Tuple[float, Dict[str,float]]] = []
    parts.append( (RULE_W, rule_probs(seq)) )
    r = rnn_probs(seq);   if r: parts.append( (RNN_W,  r) )
    x = xgb_probs(seq);   if x: parts.append( (XGB_W,  x) )
    l = lgbm_probs(seq);  if l: parts.append( (LGBM_W, l) )
    parts = normalize_weights(parts)
    b=p=t=0.0
    for w,pd in parts:
        b += w * pd["banker"]
        p += w * pd["player"]
        t += w * pd["tie"]
    s=b+p+t
    return {"banker":b/s, "player":p/s, "tie":t/s}

def render_reply(seq: List[str], probs: Dict[str,float]) -> str:
    b=probs["banker"]; p=probs["player"]; t=probs["tie"]
    side = "莊" if b>=p else "閒"
    side_prob = max(b,p)
    diff = abs(b-p)
    if diff < 0.05:
        suggest = "觀望（勝率差距不足 5%）"
    else:
        suggest = f"建議：{side}（勝率 {side_prob*100:.1f}%）"
    return (
        f"已解析 {len(seq)} 手\n"
        f"機率：莊 {b*100:.1f}%｜閒 {p*100:.1f}%｜和 {t*100:.1f}%\n"
        f"{suggest}"
    )

# ===== UI =====
def make_baccarat_buttons(prompt_text: str, title_text: str) -> FlexSendMessage:
    buttons = [
        ButtonComponent(action=PostbackAction(label="莊", data="choice=banker"), style="primary", color="#E53935", height="sm", flex=1),
        ButtonComponent(action=PostbackAction(label="閒", data="choice=player"), style="primary", color="#1E88E5", height="sm", flex=1),
        ButtonComponent(action=PostbackAction(label="和", data="choice=tie"),     style="primary", color="#43A047", height="sm", flex=1),
    ]
    bubble = BubbleContainer(
        size="mega",
        header=BoxComponent(layout="vertical", contents=[TextComponent(text=title_text, weight="bold", size="lg", align="center")]),
        body=BoxComponent(layout="vertical", contents=[TextComponent(text=prompt_text, size="md")]),
        footer=BoxComponent(layout="horizontal", spacing="sm", contents=buttons),
    )
    return FlexSendMessage(alt_text=title_text, contents=bubble)

# ===== Routes =====
@app.get("/")
def index():
    return "BGS AI（按鈕＋集成模型）運行中 ✅，/line-webhook 已就緒", 200

@app.get("/health")
def health():
    return jsonify(ok=True), 200

@app.post("/line-webhook")
def line_webhook():
    if not (line_bot_api and line_handler):
        return "Line credentials missing", 200
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        line_handler.handle(body, signature)
    except InvalidSignatureError:
        logger.exception("Invalid signature")
    return "OK", 200

# ===== Handlers =====
if line_handler and line_bot_api:

    @line_handler.add(FollowEvent)
    def on_follow(event: FollowEvent):
        uid = getattr(event.source, "user_id", "unknown")
        analysis_enabled[uid] = False
        user_history_seq[uid] = []
        welcome = (
            "歡迎加入 BGS AI 助手 🎉\n\n"
            "先用按鈕輸入歷史莊/閒/和；輸入「開始分析」後，我才會開始回覆下注建議。\n"
            "隨時輸入「結束分析」可清除資料並重新開始。"
        )
        flex = make_baccarat_buttons("請點擊下方按鈕依序輸入過往莊/閒/和結果：", "🤖請開始輸入歷史數據")
        line_bot_api.reply_message(event.reply_token, [TextSendMessage(text=welcome), flex])

    @line_handler.add(MessageEvent, message=TextMessage)
    def on_text(event: MessageEvent):
        uid = getattr(event.source, "user_id", "unknown")
        text = (event.message.text or "").strip()

        if text in {"結束分析","结束分析"}:
            analysis_enabled[uid] = False
            user_history_seq[uid] = []
            msg = "已結束本輪分析，所有歷史數據已刪除。\n請使用下方按鈕重新輸入歷史數據。"
            flex = make_baccarat_buttons("請點擊下方按鈕依序輸入過往莊/閒/和結果：", "🤖請開始輸入歷史數據")
            line_bot_api.reply_message(event.reply_token, [TextSendMessage(text=msg), flex])
            return

        if text in {"開始分析","开始分析","開始","开始","START","分析"}:
            analysis_enabled[uid] = True
            seq = user_history_seq.get(uid, [])
            if len(seq) >= 5:
                probs = ensemble_probs(seq)
                msg = "已開始分析 ✅\n" + render_reply(seq, probs)
            else:
                msg = "已開始分析 ✅\n目前資料不足（至少 5 手）。先繼續用按鈕輸入歷史結果，我會再給出建議。"
            flex = make_baccarat_buttons("持續點擊下方按鈕輸入新一手結果：", "下注選擇")
            line_bot_api.reply_message(event.reply_token, [TextSendMessage(text=msg), flex])
            return

        hint = "請先使用下方按鈕輸入歷史莊/閒/和；\n輸入「開始分析」後，我才會開始回覆下注建議。"
        flex = make_baccarat_buttons("請點擊下方按鈕依序輸入過往莊/閒/和結果：", "🤖請開始輸入歷史數據")
        line_bot_api.reply_message(event.reply_token, [TextSendMessage(text=hint), flex])

    @line_handler.add(PostbackEvent)
    def on_postback(event: PostbackEvent):
        uid = getattr(event.source, "user_id", "unknown")
        data = event.postback.data or ""
        params = dict(x.split("=",1) for x in data.split("&") if "=" in x)
        choice = params.get("choice")
        map_ = {"banker":"B","player":"P","tie":"T"}
        if choice not in map_:
            line_bot_api.reply_message(event.reply_token, [
                TextSendMessage(text="收到未知操作，請重新選擇。"),
                make_baccarat_buttons("請點擊下方按鈕輸入：","下注選擇")
            ])
            return

        # 累積歷史
        seq = user_history_seq.get(uid, [])
        seq.append(map_[choice])
        user_history_seq[uid] = seq

        # 已啟用分析 → 回覆建議；未啟用 → 只記錄
        if analysis_enabled.get(uid):
            probs = ensemble_probs(seq)
            text = render_reply(seq, probs)
        else:
            text = f"已記錄：{len(seq)} 手（例：{''.join(seq[-12:])}）\n輸入「開始分析」後，我才會開始回覆下注建議。"

        flex = make_baccarat_buttons("持續點擊下方按鈕輸入新一手結果：", "下注選擇")
        line_bot_api.reply_message(event.reply_token, [TextSendMessage(text=text), flex])

# ===== Main =====
if __name__ == "__main__":
    port = int(os.getenv("PORT","5000"))
    app.run(host="0.0.0.0", port=port)
