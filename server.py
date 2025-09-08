# server.py — LINE 卡住熱修版 (H1)
# 變更：
# - safe_reply_or_push(): reply 失敗自動 push
# - debounce_guard(): 防連點/防重送（1.2 秒）
# - hander 全面套用 safe_reply_or_push，縮短回覆鏈

import os, csv, time, logging, math
from typing import Any, Dict, List, Optional, Tuple
from flask import Flask, request, jsonify, Response

app = Flask(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO").upper(), format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-backend")

# ===== 可寫目錄自動降級（沿用你現有修正） =====
def _is_writable_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        p = os.path.join(path, ".wtest")
        with open(p, "w") as f: f.write("ok")
        os.remove(p); return True
    except Exception as e:
        log.warning("dir not writable: %s (%s)", path, e); return False

def _resolve_base_dir() -> str:
    user_path = os.getenv("DATA_BASE") or os.path.dirname(os.getenv("DATA_LOG_PATH",""))
    if user_path and _is_writable_dir(user_path): return user_path
    if _is_writable_dir("/tmp/bgs"): return "/tmp/bgs"
    local_path = os.path.join(os.getcwd(), "data")
    if _is_writable_dir(local_path): return local_path
    return "/tmp"

DATA_BASE_DIR = _resolve_base_dir()

def _resolve_csv_path() -> str:
    env_path = os.getenv("DATA_LOG_PATH","").strip()
    if env_path:
        parent = os.path.dirname(env_path)
        if parent and _is_writable_dir(parent): return env_path
        log.warning("DATA_LOG_PATH not writable, fallback in use.")
    logs_dir = os.path.join(DATA_BASE_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return os.path.join(logs_dir, "rounds.csv")

DATA_CSV_PATH = _resolve_csv_path()

# ===== 參數與常數 =====
CLASS_ORDER = ("B","P","T")
LAB_ZH = {"B":"莊","P":"閒","T":"和"}
THEORETICAL = {"B":0.458,"P":0.446,"T":0.096}
MAX_HISTORY = int(os.getenv("MAX_HISTORY","400"))
FEAT_WIN = int(os.getenv("FEAT_WIN","120"))

def parse_history(payload) -> List[str]:
    out: List[str] = []
    if payload is None: return out
    if isinstance(payload, list):
        for s in payload:
            if isinstance(s,str) and s.strip().upper() in CLASS_ORDER:
                out.append(s.strip().upper())
    elif isinstance(payload,str):
        if any(ch in payload for ch in [" ", ","]):
            for t in payload.replace(",", " ").split():
                up=t.strip().upper()
                if up in CLASS_ORDER: out.append(up)
        else:
            for ch in payload:
                up=ch.upper()
                if up in CLASS_ORDER: out.append(up)
    return out[-MAX_HISTORY:] if len(out)>MAX_HISTORY else out

# ===== 簡化：只保留必要機率路徑（演算法保持你的上一版配置） =====
def exp_decay_freq(seq: List[str], gamma: float = None) -> List[float]:
    if not seq: return [1/3,1/3,1/3]
    if gamma is None: gamma = float(os.getenv("EW_GAMMA","0.96"))
    wB=wP=wT=0.0; w=1.0
    for r in reversed(seq):
        if r=="B": wB+=w
        elif r=="P": wP+=w
        else: wT+=w
        w*=gamma
    a=float(os.getenv("LAPLACE","0.5"))
    wB+=a; wP+=a; wT+=a; S=wB+wP+wT
    return [wB/S, wP/S, wT/S]

def recent_freq(seq: List[str], win: int) -> List[float]:
    if not seq: return [1/3,1/3,1/3]
    cut = seq[-win:] if win>0 else seq
    a=float(os.getenv("LAPLACE","0.5"))
    nB=cut.count("B")+a; nP=cut.count("P")+a; nT=cut.count("T")+a
    tot=max(1,len(cut))+3*a
    return [nB/tot, nP/tot, nT/tot]

def norm(v: List[float]) -> List[float]:
    s=sum(v); s=s if s>1e-12 else 1.0
    return [max(0.0,x)/s for x in v]

def temperature(p: List[float], tau: float) -> List[float]:
    if tau<=1e-6: return p
    ex=[pow(max(pi,1e-9), 1.0/tau) for pi in p]; s=sum(ex)
    return [e/s for e in ex]

def estimate_probs(seq: List[str]) -> List[float]:
    # 簡化版：短窗 + 指數長窗 混合，再與理論先驗融合
    W_S=int(os.getenv("WIN_SHORT","6"))
    short=recent_freq(seq, W_S)
    long =exp_decay_freq(seq, float(os.getenv("EW_GAMMA","0.96")))
    PRIOR=[THEORETICAL["B"], THEORETICAL["P"], THEORETICAL["T"]]
    a=float(os.getenv("W_SHORT","0.45"))
    b=float(os.getenv("W_LONG","0.35"))
    c=float(os.getenv("W_PRIOR","0.20"))
    p=[a*short[i]+b*long[i]+c*PRIOR[i] for i in range(3)]
    p=norm(p)
    p=temperature(p, float(os.getenv("TEMP","1.06")))
    # 安全上下限
    floor=float(os.getenv("EPSILON_FLOOR","0.06"))
    cap  =float(os.getenv("MAX_CAP","0.86"))
    p=[min(cap, max(floor, x)) for x in p]
    return norm(p)

def recommend(p: List[float]) -> str:
    return CLASS_ORDER[p.index(max(p))]

# ===== Health / Predict =====
@app.get("/")
def index(): return "ok", 200

@app.get("/health")
def health(): return jsonify(status="healthy", storage=DATA_CSV_PATH)

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    seq  = parse_history(data.get("history"))
    p = estimate_probs(seq)
    return jsonify({
        "history_len": len(seq),
        "probabilities": {"B":p[0], "P":p[1], "T":p[2]},
        "recommendation": recommend(p)
    })

# ===== CSV 記錄（沿用，可關閉） =====
def append_round_csv(uid: str, history_before: str, label: str) -> None:
    if os.getenv("EXPORT_LOGS","1")!="1": return
    try:
        parent=os.path.dirname(DATA_CSV_PATH); os.makedirs(parent, exist_ok=True)
        with open(DATA_CSV_PATH,"a",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([uid, int(time.time()), history_before, label])
    except Exception as e:
        log.warning("append_round_csv failed: %s", e)

# ===== LINE SDK =====
LINE_TOKEN  = os.getenv("LINE_CHANNEL_ACCESS_TOKEN","")
LINE_SECRET = os.getenv("LINE_CHANNEL_SECRET","")

USE_LINE=False
try:
    from linebot import LineBotApi, WebhookHandler
    from linebot.models import MessageEvent, TextMessage, TextSendMessage, PostbackEvent, PostbackAction, QuickReply, QuickReplyButton, FlexSendMessage
    USE_LINE = bool(LINE_TOKEN and LINE_SECRET)
except Exception as e:
    log.warning("LINE SDK not ready: %s", e); USE_LINE=False

if USE_LINE:
    line_bot_api = LineBotApi(LINE_TOKEN)
    handler = WebhookHandler(LINE_SECRET)
else:
    line_bot_api = None; handler = None

USER_HISTORY: Dict[str, List[str]] = {}
USER_READY  : Dict[str, bool]      = {}
_LAST_HIT: Dict[str, float]        = {}   # 防重複/防連點

def debounce_guard(uid: str, key: str, window: float = 1.2) -> bool:
    """在 window 秒內，同一 uid+key 只接受一次"""
    now = time.time()
    k = f"{uid}:{key}"
    last = _LAST_HIT.get(k, 0.0)
    if now - last < window:
        return True  # 應該忽略
    _LAST_HIT[k] = now
    return False

def quick_reply_bar():
    return QuickReply(items=[
        QuickReplyButton(action=PostbackAction(label="莊", data="B")),
        QuickReplyButton(action=PostbackAction(label="閒", data="P")),
        QuickReplyButton(action=PostbackAction(label="和", data="T")),
        QuickReplyButton(action=PostbackAction(label="開始分析", data="START")),
        QuickReplyButton(action=PostbackAction(label="結束分析", data="END")),
        QuickReplyButton(action=PostbackAction(label="返回", data="UNDO")),
    ])

def safe_reply_or_push(event, messages):
    """優先 reply；失敗則改 push，避免使用者看不到回覆"""
    try:
        line_bot_api.reply_message(event.reply_token, messages)
    except Exception as e:
        try:
            uid = event.source.user_id
            log.warning("reply failed (%s), fallback to push", e)
            line_bot_api.push_message(uid, messages)
        except Exception as e2:
            log.error("push also failed: %s", e2)

@app.post("/line-webhook")
def line_webhook():
    if not USE_LINE or handler is None:
        return "ok", 200
    signature = request.headers.get("X-Line-Signature","")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except Exception as e:
        log.error("LINE handle error: %s", e)
    return "ok", 200

if USE_LINE and handler is not None:
    @handler.add(MessageEvent, message=TextMessage)
    def handle_text(event):
        uid = event.source.user_id
        USER_HISTORY.setdefault(uid, [])
        USER_READY.setdefault(uid, False)

        seq = USER_HISTORY[uid]
        msg = (
            "請用下方快速回覆輸入：莊 / 閒 / 和。\n"
            f"目前已輸入：{len(seq)} 手（莊{seq.count('B')}｜閒{seq.count('P')}｜和{seq.count('T')}）。\n"
            "按「開始分析」後會給出下注建議。"
        )
        safe_reply_or_push(event, TextSendMessage(text=msg, quick_reply=quick_reply_bar()))

    @handler.add(PostbackEvent)
    def handle_postback(event):
        uid  = event.source.user_id
        data = (event.postback.data or "").upper()

        USER_HISTORY.setdefault(uid, [])
        USER_READY.setdefault(uid, False)
        seq  = USER_HISTORY[uid]
        ready= USER_READY[uid]

        # 防重複/防連點：同一操作 1.2 秒內忽略
        if debounce_guard(uid, key=data, window=float(os.getenv("DEBOUNCE_SEC","1.2"))):
            safe_reply_or_push(event, TextSendMessage(text="⏳ 收到重複操作，稍等 1 秒再按一次即可。", quick_reply=quick_reply_bar()))
            return

        if data == "START":
            USER_READY[uid] = True
            safe_reply_or_push(event, TextSendMessage(text="🔎 已開始分析，請持續輸入莊/閒/和。", quick_reply=quick_reply_bar()))
            return

        if data == "END":
            USER_HISTORY[uid] = []
            USER_READY[uid]   = False
            safe_reply_or_push(event, TextSendMessage(text="✅ 已結束分析，紀錄已清空。", quick_reply=quick_reply_bar()))
            return

        if data == "UNDO":
            if seq:
                removed = seq.pop()
                msg = f"↩ 已返回一步（移除：{LAB_ZH.get(removed, removed)}）。目前 {len(seq)} 手。"
            else:
                msg = "沒有可返回的紀錄。"
            safe_reply_or_push(event, TextSendMessage(text=msg, quick_reply=quick_reply_bar()))
            return

        if data not in CLASS_ORDER:
            safe_reply_or_push(event, TextSendMessage(text="請用按鈕輸入（莊/閒/和）。", quick_reply=quick_reply_bar()))
            return

        # 記錄 + 匯出
        history_before = "".join(seq)
        seq.append(data)
        if len(seq) > MAX_HISTORY: seq[:] = seq[-MAX_HISTORY:]
        USER_HISTORY[uid] = seq
        append_round_csv(uid, history_before, data)

        # 若尚未 START，僅顯示統計
        if not ready:
            msg = f"已記錄 {len(seq)} 手。按「開始分析」後才會給出下注建議。"
            safe_reply_or_push(event, TextSendMessage(text=msg, quick_reply=quick_reply_bar()))
            return

        # 計算建議（盡量保持 < 500ms）
        t0 = time.time()
        p = estimate_probs(seq)
        rec = recommend(p)
        dt = int((time.time() - t0)*1000)
        msg = (
            f"已解析 {len(seq)} 手\n"
            f"機率：莊 {p[0]:.3f}｜閒 {p[1]:.3f}｜和 {p[2]:.3f}\n"
            f"建議：{LAB_ZH[rec]}（{dt} ms）"
        )
        safe_reply_or_push(event, TextSendMessage(text=msg, quick_reply=quick_reply_bar()))
        return

# ===== Entrypoint =====
if __name__ == "__main__":
    port = int(os.getenv("PORT","8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
