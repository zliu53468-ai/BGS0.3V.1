#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BGS LINE Bot backend — Buttons only + Ensemble voting with anti-stuck
- /, /health, /healthz
- /predict  (後端測)
- /line-webhook  (莊/閒/和/開始分析/結束分析)
"""

import os, logging
from typing import Any, Dict, List, Optional
from flask import Flask, request, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bgs-backend")

# ========= 基礎常數 =========
CLASS_ORDER = ("B", "P", "T")
LAB_ZH = {"B": "莊", "P": "閒", "T": "和"}

# 標準八副牌理論機率
THEORETICAL_PROBS: Dict[str, float] = {"B": 0.458, "P": 0.446, "T": 0.096}

# ========= 解析工具 =========
def parse_history(payload) -> List[str]:
    if payload is None: return []
    seq: List[str] = []
    if isinstance(payload, list):
        for s in payload:
            if isinstance(s, str) and s.strip().upper() in CLASS_ORDER:
                seq.append(s.strip().upper())
    elif isinstance(payload, str):
        for ch in payload:
            up = ch.upper()
            if up in CLASS_ORDER: seq.append(up)
    return seq

# ========= 可選模型：安全匯入（沒裝也能啟動） =========
try:
    import torch  # type: ignore
    import torch.nn as tnn  # type: ignore
except Exception:
    torch = None  # type: ignore
    tnn   = None  # type: ignore

try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None  # type: ignore

try:
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None  # type: ignore

if tnn is not None:
    class TinyRNN(tnn.Module):  # type: ignore
        def __init__(self, in_dim=3, hidden=16, out_dim=3):
            super().__init__()
            self.rnn = tnn.GRU(in_dim, hidden, batch_first=True)
            self.fc  = tnn.Linear(hidden, out_dim)
        def forward(self, x):
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :])
else:
    TinyRNN = None  # type: ignore

# ========= 載入模型檔（若提供） =========
RNN_MODEL: Optional[Any] = None
if TinyRNN is not None and torch is not None:
    p = os.getenv("RNN_PATH", "")
    if p and os.path.exists(p):
        try:
            _m = TinyRNN()
            _m.load_state_dict(torch.load(p, map_location="cpu"))
            _m.eval()
            RNN_MODEL = _m
            logger.info("Loaded RNN from %s", p)
        except Exception as e:
            logger.warning("Load RNN failed: %s", e)

XGB_MODEL = None
if xgb is not None:
    p = os.getenv("XGB_PATH", "")
    if p and os.path.exists(p):
        try:
            booster = xgb.Booster()
            booster.load_model(p)
            XGB_MODEL = booster
            logger.info("Loaded XGB from %s", p)
        except Exception as e:
            logger.warning("Load XGB failed: %s", e)

LGBM_MODEL = None
if lgb is not None:
    p = os.getenv("LGBM_PATH", "")
    if p and os.path.exists(p):
        try:
            booster = lgb.Booster(model_file=p)
            LGBM_MODEL = booster
            logger.info("Loaded LGBM from %s", p)
        except Exception as e:
            logger.warning("Load LGBM failed: %s", e)

# ========= 單模型推論 =========
def rnn_predict(seq: List[str]) -> Optional[List[float]]:
    if RNN_MODEL is None or torch is None or not seq: return None
    try:
        def onehot(label: str): return [1 if label == lab else 0 for lab in CLASS_ORDER]
        inp = torch.tensor([[onehot(ch) for ch in seq]], dtype=torch.float32)
        with torch.no_grad():
            logits = RNN_MODEL(inp)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0].tolist()
        return [float(p) for p in probs]
    except Exception as e:
        logger.warning("RNN inference failed: %s", e)
        return None

def xgb_predict(seq: List[str]) -> Optional[List[float]]:
    if XGB_MODEL is None or not seq: return None
    try:
        import numpy as np
        K = int(os.getenv("FEAT_WIN", "20"))
        vec: List[float] = []
        for label in seq[-K:]:
            vec.extend([1.0 if label == lab else 0.0 for lab in CLASS_ORDER])
        pad = K*3 - len(vec)
        if pad > 0: vec = [0.0]*pad + vec
        dmatrix = xgb.DMatrix(np.array([vec], dtype=float))
        prob = XGB_MODEL.predict(dmatrix)[0]
        if isinstance(prob, (list, tuple)) and len(prob) == 3:
            return [float(prob[0]), float(prob[1]), float(prob[2])]
        elif len(prob) == 2:
            return [float(prob[0]), float(prob[1]), 0.05]
        return None
    except Exception as e:
        logger.warning("XGB inference failed: %s", e)
        return None

def lgbm_predict(seq: List[str]) -> Optional[List[float]]:
    if LGBM_MODEL is None or not seq: return None
    try:
        K = int(os.getenv("FEAT_WIN", "20"))
        vec: List[float] = []
        for label in seq[-K:]:
            vec.extend([1.0 if label == lab else 0.0 for lab in CLASS_ORDER])
        pad = K*3 - len(vec)
        if pad > 0: vec = [0.0]*pad + vec
        prob = LGBM_MODEL.predict([vec])[0]
        if isinstance(prob, (list, tuple)) and len(prob) == 3:
            return [float(prob[0]), float(prob[1]), float(prob[2])]
        elif len(prob) == 2:
            return [float(prob[0]), float(prob[1]), 0.05]
        return None
    except Exception as e:
        logger.warning("LGBM inference failed: %s", e)
        return None

# ========= Anti-Stuck 集成投票 =========
def norm(v: List[float]) -> List[float]:
    s = sum(v);  s = s if s > 1e-12 else 1.0
    return [max(0.0, x)/s for x in v]

def blend(a: List[float], b: List[float], w: float) -> List[float]:
    # w: 權重給 b（0~1）
    return [ (1-w)*a[i] + w*b[i] for i in range(3) ]

def recent_freq(seq: List[str], win: int) -> List[float]:
    if not seq:
        return [1/3, 1/3, 1/3]
    cut = seq[-win:] if win>0 else seq
    nB = cut.count("B"); nP = cut.count("P"); nT = cut.count("T")
    tot = max(1, len(cut))
    # 拉普拉斯平滑，避免某一類=0
    alpha = float(os.getenv("LAPLACE", "0.5"))
    nB += alpha; nP += alpha; nT += alpha; tot += 3*alpha
    return [nB/tot, nP/tot, nT/tot]

def temperature_scale(p: List[float], tau: float) -> List[float]:
    if tau <= 1e-6: return p
    import math
    ex = [pow(max(pi,1e-9), 1.0/tau) for pi in p]
    s  = sum(ex)
    return [e/s for e in ex]

def ensemble_with_anti_stuck(seq: List[str]) -> List[float]:
    # 1) 個別模型
    rule = [THEORETICAL_PROBS["B"], THEORETICAL_PROBS["P"], THEORETICAL_PROBS["T"]]
    pr_rnn = rnn_predict(seq)
    pr_xgb = xgb_predict(seq)
    pr_lgb = lgbm_predict(seq)

    # 2) 權重
    w_rule = float(os.getenv("RULE_W", "0.50"))
    w_rnn  = float(os.getenv("RNN_W",  "0.25"))
    w_xgb  = float(os.getenv("XGB_W",  "0.15"))
    w_lgb  = float(os.getenv("LGBM_W", "0.10"))

    # 3) 初步融合（缺模型自動跳過）
    total = w_rule + (w_rnn if pr_rnn else 0) + (w_xgb if pr_xgb else 0) + (w_lgb if pr_lgb else 0)
    base = [w_rule*rule[i] for i in range(3)]
    if pr_rnn: base = [base[i] + w_rnn*pr_rnn[i] for i in range(3)]
    if pr_xgb: base = [base[i] + w_xgb*pr_xgb[i] for i in range(3)]
    if pr_lgb: base = [base[i] + w_lgb*pr_lgb[i] for i in range(3)]
    probs = [b / max(total, 1e-9) for b in base]

    # 4) 近期頻率融合（防單邊卡死）
    REC_WIN = int(os.getenv("REC_WIN", "16"))
    REC_W   = float(os.getenv("REC_W", "0.35"))   # 與近期頻率的融合權重
    probs   = blend(probs, recent_freq(seq, REC_WIN), REC_W)

    # 5) 先驗（理論機率）再拉回一點，避免長期漂移
    PRIOR_W = float(os.getenv("PRIOR_W", "0.20"))
    probs   = blend(probs, [rule[0], rule[1], rule[2]], PRIOR_W)

    # 6) 最小/最大限制 + 溫度平滑（避免某一類過度接近1）
    EPS      = float(os.getenv("EPSILON_FLOOR", "0.06"))  # 每類最低機率
    CAP      = float(os.getenv("MAX_CAP", "0.88"))        # 單類上限
    probs    = [min(CAP, max(EPS, p)) for p in probs]
    probs    = norm(probs)
    TAU      = float(os.getenv("TEMP", "1.15"))
    probs    = temperature_scale(probs, TAU)

    return norm(probs)

def recommend_from_probs(probs: List[float]) -> str:
    # 如果差距太小，仍給出最高者，但你也可改為回「觀望」
    MIN_GAP = float(os.getenv("MIN_GAP", "0.03"))  # 3% 門檻
    mx  = max(probs); mn = min(probs)
    idx = probs.index(mx)
    # 這裡維持回傳 B/P/T；若想觀望可改成回傳 'N'
    # if mx - sorted(probs)[-2] < MIN_GAP: return 'N'
    return CLASS_ORDER[idx]

# ========= Health & Predict =========
@app.route("/", methods=["GET"])
def index(): return "ok"

@app.route("/health", methods=["GET"])
def health(): return jsonify(status="healthy", version="v2-ensemble-anti-stuck")

@app.route("/healthz", methods=["GET"])
def healthz(): return jsonify(status="healthy")

@app.route("/predict", methods=["POST"])
def predict():
    data: Dict[str, Any] = request.get_json(silent=True) or {}
    seq = parse_history(data.get("history"))
    probs = ensemble_with_anti_stuck(seq)
    rec   = recommend_from_probs(probs)
    labels = list(CLASS_ORDER)
    return jsonify({
        "history_len": len(seq),
        "probabilities": {labels[i]: probs[i] for i in range(3)},
        "recommendation": rec
    })

# ========= LINE Webhook（保持你先前的互動邏輯，不改樣式） =========
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET       = os.getenv("LINE_CHANNEL_SECRET", "")

USE_LINE = False
try:
    from linebot import LineBotApi, WebhookHandler  # type: ignore
    from linebot.models import (  # type: ignore
        MessageEvent, TextMessage, TextSendMessage,
        PostbackEvent, PostbackAction,
        FlexSendMessage,
        QuickReply, QuickReplyButton
    )
    USE_LINE = bool(LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET)
except Exception as e:
    logger.warning("LINE SDK not available or env not set: %s", e)
    USE_LINE = False

if USE_LINE:
    line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
    handler = WebhookHandler(LINE_CHANNEL_SECRET)
else:
    line_bot_api = None
    handler = None

USER_HISTORY: Dict[str, List[str]] = {}
USER_READY:   Dict[str, bool]      = {}

def flex_buttons_card() -> FlexSendMessage:
    contents = {
        "type": "bubble",
        "body": {
            "type": "box", "layout": "vertical", "spacing": "md",
            "contents": [
                {"type": "text", "text": "🤖 請開始輸入歷史數據", "weight": "bold", "size": "lg"},
                {"type": "text", "text": "先輸入莊/閒/和；按「開始分析」後才會給出下注建議。", "wrap": True, "size": "sm", "color": "#555555"},
                {"type": "box", "layout": "horizontal", "spacing": "sm",
                 "contents": [
                    {"type":"button","style":"primary","color":"#E74C3C","action":{"type":"postback","label":"莊","data":"B"}},
                    {"type":"button","style":"primary","color":"#2980B9","action":{"type":"postback","label":"閒","data":"P"}},
                    {"type":"button","style":"primary","color":"#27AE60","action":{"type":"postback","label":"和","data":"T"}}
                 ]},
                {"type": "box", "layout": "horizontal", "spacing": "sm",
                 "contents": [
                    {"type":"button","style":"secondary","action":{"type":"postback","label":"開始分析","data":"START"}},
                    {"type":"button","style":"secondary","action":{"type":"postback","label":"結束分析","data":"END"}}
                 ]}
            ]
        }
    }
    return FlexSendMessage(alt_text="請開始輸入歷史數據", contents=contents)

def quick_reply_bar() -> QuickReply:
    return QuickReply(items=[
        QuickReplyButton(action=PostbackAction(label="莊", data="B")),
        QuickReplyButton(action=PostbackAction(label="閒", data="P")),
        QuickReplyButton(action=PostbackAction(label="和", data="T")),
        QuickReplyButton(action=PostbackAction(label="開始分析", data="START")),
        QuickReplyButton(action=PostbackAction(label="結束分析", data="END")),
    ])

@app.route("/line-webhook", methods=["POST"])
def line_webhook():
    if not USE_LINE or handler is None:
        logger.warning("LINE webhook hit but LINE SDK/env not configured.")
        return "ok", 200
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except Exception as e:
        logger.error("LINE handle error: %s", e)
        return "ok", 200
    return "ok", 200

if USE_LINE and handler is not None:
    @handler.add(MessageEvent, message=TextMessage)
    def handle_text(event):
        uid = event.source.user_id
        USER_HISTORY.setdefault(uid, [])
        USER_READY.setdefault(uid, False)
        msg = "請使用下方按鈕輸入：莊/閒/和；按「開始分析」後才會給出下注建議。"
        line_bot_api.reply_message(
            event.reply_token,
            [TextSendMessage(text=msg, quick_reply=quick_reply_bar()), flex_buttons_card()]
        )

    @handler.add(PostbackEvent)
    def handle_postback(event):
        uid  = event.source.user_id
        data = (event.postback.data or "").upper()
        seq  = USER_HISTORY.get(uid, [])
        ready= USER_READY.get(uid, False)

        if data == "START":
            USER_READY[uid] = True
            line_bot_api.reply_message(
                event.reply_token,
                [TextSendMessage(text="🔎 已開始分析。請繼續輸入莊/閒/和，我會根據資料給出建議。", quick_reply=quick_reply_bar()),
                 flex_buttons_card()]
            )
            return

        if data == "END":
            USER_HISTORY[uid] = []
            USER_READY[uid]   = False
            line_bot_api.reply_message(
                event.reply_token,
                [TextSendMessage(text="✅ 已結束分析，紀錄已清空。", quick_reply=quick_reply_bar()),
                 flex_buttons_card()]
            )
            return

        if data not in CLASS_ORDER:
            line_bot_api.reply_message(
                event.reply_token,
                [TextSendMessage(text="請用按鈕輸入（莊/閒/和），或選開始/結束分析。", quick_reply=quick_reply_bar()),
                 flex_buttons_card()]
            )
            return

        # 累積牌路
        seq.append(data); USER_HISTORY[uid] = seq

        # 尚未開始分析：只提示進度
        if not ready:
            s = "".join(seq[-20:])  # 顯示末20手
            line_bot_api.reply_message(
                event.reply_token,
                [TextSendMessage(text=f"已記錄 {len(seq)} 手：{s}\n按「開始分析」後才會給出下注建議。", quick_reply=quick_reply_bar()),
                 flex_buttons_card()]
            )
            return

        # 已開始分析：做集成 + 反鬼打牆
        probs = ensemble_with_anti_stuck(seq)
        rec   = recommend_from_probs(probs)
        msg = (
            f"已解析 {len(seq)} 手\n"
            f"機率：莊 {probs[0]:.3f}｜閒 {probs[1]:.3f}｜和 {probs[2]:.3f}\n"
            f"建議：{LAB_ZH[rec]}"
        )
        line_bot_api.reply_message(
            event.reply_token,
            [TextSendMessage(text=msg, quick_reply=quick_reply_bar()),
             flex_buttons_card()]
        )

# ========= Entrypoint =========
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
