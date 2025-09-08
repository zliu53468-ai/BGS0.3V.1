# server.py — BGS Backend (CHOP+ + Pattern Hooks + Flex Menu)
# - 可寫路徑自動降級：/tmp/bgs
# - /health /storage /predict API
# - 機率融合：短窗 + 指數長窗 + 先驗
# - 路型鉤子：單跳/雙跳/1-2/2-1/提早斷龍（乘法小倍率，可用環境變數調整/關閉）
# - 安全層：溫度 + 機率上下限 + 去過熱
# - CSV 記錄（EXPORT_LOGS=1 時）
# - LINE Webhook（有 token/secret 才啟用），UI 改為 Flex 3×2 彩色按鈕

import os
import csv
import time
import logging
from collections import deque
from typing import List, Dict, Tuple, Optional

from flask import Flask, request, jsonify

# -------------------- Flask / Logging --------------------
app = Flask(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s"
)
log = logging.getLogger("bgs-backend")

# -------------------- 可寫路徑 --------------------
def _is_writable_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        test = os.path.join(path, ".wtest")
        with open(test, "w") as f:
            f.write("ok")
        os.remove(test)
        return True
    except Exception as e:
        log.warning("dir not writable: %s (%s)", path, e)
        return False

def _resolve_base_dir() -> str:
    user = os.getenv("DATA_BASE", "").strip()
    if user and _is_writable_dir(user):
        return user
    if _is_writable_dir("/tmp/bgs"):
        return "/tmp/bgs"
    local = os.path.join(os.getcwd(), "data")
    if _is_writable_dir(local):
        return local
    return "/tmp"

DATA_BASE_DIR = _resolve_base_dir()

def _resolve_csv_path() -> str:
    custom = os.getenv("DATA_LOG_PATH", "").strip()
    if custom:
        parent = os.path.dirname(custom)
        if parent and _is_writable_dir(parent):
            return custom
        log.warning("DATA_LOG_PATH not writable, fallback used.")
    logs_dir = os.path.join(DATA_BASE_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return os.path.join(logs_dir, "rounds.csv")

DATA_CSV_PATH = _resolve_csv_path()

# -------------------- 常數/先驗 --------------------
CLASS_ORDER = ("B", "P", "T")
LAB_ZH = {"B": "莊", "P": "閒", "T": "和"}
THEORETICAL = {"B": 0.458, "P": 0.446, "T": 0.096}
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "400"))

# -------------------- 小工具 --------------------
def parse_history(payload) -> List[str]:
    out: List[str] = []
    if payload is None:
        return out
    if isinstance(payload, list):
        for s in payload:
            if isinstance(s, str) and s.strip().upper() in CLASS_ORDER:
                out.append(s.strip().upper())
    elif isinstance(payload, str):
        if any(ch in payload for ch in [" ", ","]):
            for t in payload.replace(",", " ").split():
                up = t.strip().upper()
                if up in CLASS_ORDER:
                    out.append(up)
        else:
            for ch in payload:
                up = ch.upper()
                if up in CLASS_ORDER:
                    out.append(up)
    return out[-MAX_HISTORY:] if len(out) > MAX_HISTORY else out

def norm(v: List[float]) -> List[float]:
    s = sum(v)
    s = s if s > 1e-12 else 1.0
    return [max(0.0, x) / s for x in v]

def temperature(p: List[float], tau: float) -> List[float]:
    if tau <= 1e-9:
        return p
    ex = [pow(max(pi, 1e-12), 1.0 / tau) for pi in p]
    s = sum(ex)
    return [e / s for e in ex]

# -------------------- 頻率估計 --------------------
def recent_freq(seq: List[str], win: int) -> List[float]:
    if not seq:
        return [1/3, 1/3, 1/3]
    cut = seq[-win:] if win > 0 else seq
    a = float(os.getenv("LAPLACE", "0.5"))
    nB = cut.count("B") + a
    nP = cut.count("P") + a
    nT = cut.count("T") + a
    tot = max(1, len(cut)) + 3 * a
    return [nB/tot, nP/tot, nT/tot]

def exp_decay_freq(seq: List[str], gamma: Optional[float] = None) -> List[float]:
    if not seq:
        return [1/3, 1/3, 1/3]
    if gamma is None:
        gamma = float(os.getenv("EW_GAMMA", "0.96"))
    wB = wP = wT = 0.0
    w = 1.0
    for r in reversed(seq):
        if r == "B": wB += w
        elif r == "P": wP += w
        else: wT += w
        w *= gamma
    a = float(os.getenv("LAPLACE", "0.5"))
    wB += a; wP += a; wT += a
    S = wB + wP + wT
    return [wB/S, wP/S, wT/S]

# -------------------- 路型鉤子（Pattern Hooks） --------------------
def _run_length_tail(seq: List[str], k: int = 1):
    if not seq:
        return None, 0
    blocks = deque()
    cur, cnt = seq[-1], 1
    for x in reversed(seq[:-1]):
        if x == cur:
            cnt += 1
        else:
            blocks.appendleft((cur, cnt))
            cur, cnt = x, 1
    blocks.appendleft((cur, cnt))
    return blocks[-k] if 0 < k <= len(blocks) else (None, 0)

def pattern_boost(seq: List[str]) -> Dict[str, float]:
    m = {"B": 1.0, "P": 1.0, "T": 1.0}
    n = len(seq)
    if n < 3:
        return m

    alt_boost = float(os.getenv("HOOK_ALT", "1.06"))
    dbl_boost = float(os.getenv("HOOK_DBLJUMP", "1.04"))
    cyc_boost = float(os.getenv("HOOK_CYCLE", "1.06"))
    dlen_th   = int(os.getenv("HOOK_DRAGON_LEN", "3"))
    dragon_k  = float(os.getenv("HOOK_DRAGON_K", "0.06"))

    a, b = seq[-1], seq[-2]

    # 交替單跳 ..BPBP
    if a != b and n >= 4 and seq[-3] != seq[-2] and seq[-4] != seq[-3]:
        if a == "P": m["B"] *= alt_boost
        if a == "B": m["P"] *= alt_boost

    # 雙跳 ..BBPP / ..PPBB → 多半換邊
    if n >= 4 and seq[-1] == seq[-2] and seq[-3] != seq[-2] and seq[-3] == seq[-4]:
        if a == "P": m["B"] *= dbl_boost
        if a == "B": m["P"] *= dbl_boost

    # 1-2 / 2-1 循環（近 6 手）
    if n >= 6:
        last6 = "".join(seq[-6:])
        if last6 in ("BPPBPP", "PPBPPB", "PBBPBB", "BBPBBP"):
            if a == "P": m["B"] *= cyc_boost
            if a == "B": m["P"] *= cyc_boost

    # 提早斷龍：run >= dlen_th
    sym, run = _run_length_tail(seq, 1)
    if run >= dlen_th and sym in ("B", "P"):
        hazard = 1.0 + min(0.25, dragon_k * (run - (dlen_th - 1)))
        if sym == "B": m["P"] *= hazard
        if sym == "P": m["B"] *= hazard

    return m

# -------------------- 機率與建議 --------------------
def estimate_probs(seq: List[str]) -> List[float]:
    if not seq:
        base = [THEORETICAL["B"], THEORETICAL["P"], THEORETICAL["T"]]
        return norm(base)

    W_S = int(os.getenv("WIN_SHORT", "6"))
    gamma = float(os.getenv("EW_GAMMA", "0.96"))

    short = recent_freq(seq, W_S)
    longv = exp_decay_freq(seq, gamma)
    prior = [THEORETICAL["B"], THEORETICAL["P"], THEORETICAL["T"]]

    a = float(os.getenv("REC_W", "0.20"))
    b = float(os.getenv("LONG_W", "0.20"))
    c = float(os.getenv("PRIOR_W", "0.20"))
    p = [a*short[i] + b*longv[i] + c*prior[i] for i in range(3)]
    p = norm(p)

    hook = pattern_boost(seq)
    p = [p[0]*hook["B"], p[1]*hook["P"], p[2]*hook["T"]]
    p = norm(p)

    p = temperature(p, float(os.getenv("TEMP", "1.06")))
    floor = float(os.getenv("EPSILON_FLOOR", "0.06"))
    cap   = float(os.getenv("MAX_CAP", "0.86"))
    p = [min(cap, max(floor, x)) for x in p]
    return norm(p)

def recommend(p: List[float]) -> str:
    return CLASS_ORDER[p.index(max(p))]

# -------------------- CSV 記錄 --------------------
def append_round_csv(uid: str, history_before: str, label: str) -> None:
    if os.getenv("EXPORT_LOGS", "1") != "1":
        return
    try:
        parent = os.path.dirname(DATA_CSV_PATH)
        os.makedirs(parent, exist_ok=True)
        with open(DATA_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([uid, int(time.time()), history_before, label])
    except Exception as e:
        log.warning("append_round_csv failed: %s", e)

# -------------------- HTTP 端點 --------------------
@app.get("/")
def index():
    return "ok", 200

@app.get("/health")
def health():
    return jsonify(status="healthy", storage=DATA_CSV_PATH), 200

@app.get("/storage")
def storage():
    return jsonify(base_dir=DATA_BASE_DIR, csv_path=DATA_CSV_PATH), 200

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    seq = parse_history(data.get("history"))
    p = estimate_probs(seq)
    return jsonify({
        "history_len": len(seq),
        "probabilities": {"B": p[0], "P": p[1], "T": p[2]},
        "recommendation": recommend(p)
    }), 200

# -------------------- LINE（可選；有憑證才啟用） --------------------
LINE_TOKEN  = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
USE_LINE = False
try:
    if LINE_TOKEN and LINE_SECRET:
        from linebot import LineBotApi, WebhookHandler
        from linebot.models import (
            MessageEvent, TextMessage, TextSendMessage,
            PostbackEvent, PostbackAction, QuickReply, QuickReplyButton,
            FlexSendMessage
        )
        USE_LINE = True
        line_bot_api = LineBotApi(LINE_TOKEN)
        handler = WebhookHandler(LINE_SECRET)
    else:
        line_bot_api = None
        handler = None
except Exception as e:
    log.warning("LINE SDK not ready: %s", e)
    line_bot_api = None
    handler = None
    USE_LINE = False

USER_HISTORY: Dict[str, List[str]] = {}
USER_READY: Dict[str, bool] = {}
_LAST_HIT: Dict[str, float] = {}

def debounce_guard(uid: str, key: str, window: float = 1.2) -> bool:
    now = time.time()
    k = f"{uid}:{key}"
    last = _LAST_HIT.get(k, 0.0)
    if now - last < window:
        return True
    _LAST_HIT[k] = now
    return False

# === Flex Menu (3x2 彩色按鈕) — 修正版 ===
def flex_menu():
    from linebot.models import FlexSendMessage
    bubble = {
      "type": "bubble",
      "size": "mega",
      "body": {
        "type": "box",
        "layout": "vertical",
        "spacing": "lg",
        "contents": [
          {"type": "text", "text": "🤖 請開始輸入歷史數據", "weight": "bold", "size": "md"},
          {"type": "text", "text": "先輸入莊/閒/和；按「開始分析」後才會給出下注建議。", "wrap": True, "size": "sm", "color": "#6B7280"},
          {
            "type": "box", "layout": "horizontal", "spacing": "sm", "contents": [
              {"type": "button", "style": "primary", "height": "sm", "color": "#EF4444",
               "action": {"type": "postback", "label": "莊", "data": "B", "displayText": "莊"}},
              {"type": "button", "style": "primary", "height": "sm", "color": "#3B82F6",
               "action": {"type": "postback", "label": "閒", "data": "P", "displayText": "閒"}},
              {"type": "button", "style": "primary", "height": "sm", "color": "#22C55E",
               "action": {"type": "postback", "label": "和", "data": "T", "displayText": "和"}}
            ]
          },
          {
            "type": "box", "layout": "horizontal", "spacing": "sm", "contents": [
              {"type": "button", "style": "primary", "height": "sm", "color": "#E5E7EB",
               "action": {"type": "postback", "label": "開始...", "data": "START", "displayText": "開始分析"}},
              {"type": "button", "style": "primary", "height": "sm", "color": "#E5E7EB",
               "action": {"type": "postback", "label": "結束...", "data": "END", "displayText": "結束分析"}},
              {"type": "button", "style": "primary", "height": "sm", "color": "#E5E7EB",
               "action": {"type": "postback", "label": "返回", "data": "UNDO", "displayText": "返回一步"}}
            ]
          }
        ]
      }
    }
    return FlexSendMessage(alt_text="請開始輸入歷史數據", contents=bubble)

def safe_reply_or_push(event, messages):
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
    signature = request.headers.get("X-Line-Signature", "")
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
        safe_reply_or_push(event, flex_menu())

    @handler.add(PostbackEvent)
    def handle_postback(event):
        uid  = event.source.user_id
        data = (event.postback.data or "").upper()
        USER_HISTORY.setdefault(uid, [])
        USER_READY.setdefault(uid, False)
        seq  = USER_HISTORY[uid]
        ready= USER_READY[uid]

        if debounce_guard(uid, key=data, window=float(os.getenv("DEBOUNCE_SEC","1.2"))):
            safe_reply_or_push(event, flex_menu())
            return

        if data == "START":
            USER_READY[uid] = True
            safe_reply_or_push(event, flex_menu()); return
        if data == "END":
            USER_HISTORY[uid] = []
            USER_READY[uid] = False
            safe_reply_or_push(event, flex_menu()); return
        if data == "UNDO":
            if seq:
                seq.pop()
            safe_reply_or_push(event, flex_menu()); return

        if data not in CLASS_ORDER:
            safe_reply_or_push(event, flex_menu()); return

        # 記錄 + 匯出
        history_before = "".join(seq)
        seq.append(data)
        if len(seq) > MAX_HISTORY:
            seq[:] = seq[-MAX_HISTORY:]
        USER_HISTORY[uid] = seq
        append_round_csv(uid, history_before, data)

        if not ready:
            safe_reply_or_push(event, flex_menu()); return

        t0 = time.time()
        p = estimate_probs(seq)
        rec = recommend(p)
        dt = int((time.time() - t0) * 1000)
        from linebot.models import TextSendMessage
        result_text = (
            f"已解析 {len(seq)} 手\n"
            f"機率：莊 {p[0]:.3f}｜閒 {p[1]:.3f}｜和 {p[2]:.3f}\n"
            f"建議：{LAB_ZH[rec]}（{dt} ms）"
        )
        safe_reply_or_push(event, [TextSendMessage(text=result_text), flex_menu()])

# -------------------- Entrypoint --------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
