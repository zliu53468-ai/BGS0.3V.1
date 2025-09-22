# -*- coding: utf-8 -*-
"""
server.py — 完整百家樂AI + LINE webhook（Render可用直接覆蓋）
"""

import os, sys, re, time, json, math, random, logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    _has_flask = True
except Exception:
    _has_flask = False
    Flask = None
    def jsonify(*_, **__): raise RuntimeError("Flask not available")
    def CORS(*_, **__): pass

if _has_flask:
    app = Flask(__name__)
    CORS(app)

    @app.get("/")
    def root():
        return "✅ BGS PF Server OK", 200

    @app.get("/health")
    def health():
        return jsonify(ok=True, ts=time.time(), msg="API normal"), 200
else:
    class _DummyApp:
        def get(self,*a,**k): def deco(f): return f; return deco
        def post(self,*a,**k): def deco(f): return f; return deco
        def run(self,*a,**k): print("Flask not installed; dummy app.")
    app = _DummyApp()

# ---------- Redis / Fallback ----------
try:
    import redis
except Exception:
    redis = None

REDIS_URL = os.getenv("REDIS_URL", "")
rcli = None
if redis and REDIS_URL:
    try:
        rcli = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
        rcli.ping()
    except Exception:
        rcli = None

SESS: Dict[str, Dict[str, Any]] = {}
SESSION_EXPIRE = 3600

def _rget(k: str) -> Optional[str]:
    try:
        if rcli: return rcli.get(k)
    except Exception: pass
    return None

def _rset(k: str, v: str, ex: Optional[int]=None):
    try:
        if rcli: rcli.set(k, v, ex=ex)
    except Exception: pass

# ---------- 參數強化 ----------
os.environ.setdefault("PF_N", "80")
os.environ.setdefault("PF_RESAMPLE", "0.73")
os.environ.setdefault("PF_DIR_EPS", "0.012")
os.environ.setdefault("EDGE_ENTER", "0.007")
os.environ.setdefault("WATCH_INSTAB_THRESH", "0.16")
os.environ.setdefault("TIE_PROB_MAX", "0.18")
os.environ.setdefault("PF_BACKEND", "mc")
os.environ.setdefault("DECKS", "6")
os.environ.setdefault("PF_UPD_SIMS", "36")
os.environ.setdefault("PF_PRED_SIMS", "30")
os.environ.setdefault("MIN_BET_PCT", "0.08")
os.environ.setdefault("MAX_BET_PCT", "0.26")
os.environ.setdefault("PROB_SMA_ALPHA", "0.39")
os.environ.setdefault("PROB_TEMP", "0.95")
os.environ.setdefault("UNCERT_MARGIN_MAX", "1")
os.environ.setdefault("UNCERT_RATIO", "0.22")

# ---------- PF import ----------
OutcomePF = None
try:
    from bgs.pfilter import OutcomePF
except Exception:
    try:
        cur = os.path.dirname(os.path.abspath(__file__))
        if cur not in sys.path: sys.path.insert(0, cur)
        from pfilter import OutcomePF
    except Exception:
        OutcomePF = None

class _DummyPF:
    def update_outcome(self, outcome): pass
    def predict(self, **k): return np.array([0.48,0.47,0.05], dtype=np.float32)
    def update_point_history(self, p_pts, b_pts): pass

def _get_pf_from_sess(sess: Dict[str, Any]) -> Any:
    if OutcomePF:
        if sess.get("pf") is None:
            try:
                sess["pf"] = OutcomePF(
                    decks=int(os.getenv("DECKS","6")),
                    seed=int(os.getenv("SEED","42")) + int(time.time() % 1000),
                    n_particles=int(os.getenv("PF_N","80")),
                    sims_lik=max(1,int(os.getenv("PF_UPD_SIMS","36"))),
                    resample_thr=float(os.getenv("PF_RESAMPLE","0.73")),
                    backend=os.getenv("PF_BACKEND","mc"),
                    dirichlet_eps=float(os.getenv("PF_DIR_EPS","0.012")),
                )
            except Exception:
                sess["pf"] = _DummyPF()
        return sess["pf"]
    return _DummyPF()

def _is_long_dragon(sess: Dict[str,Any], dragon_len=7) -> Optional[str]:
    pred = sess.get("hist_real", [])
    if len(pred) < dragon_len: return None
    lastn = pred[-dragon_len:]
    if all(x=="莊" for x in lastn): return "莊"
    if all(x=="閒" for x in lastn): return "閒"
    return None

def handle_points_and_predict(sess: Dict[str,Any], p_pts: int, b_pts: int) -> str:
    if not (0 <= int(p_pts) <= 9 and 0 <= int(b_pts) <= 9):
        return "❌ 點數數據異常（僅接受 0~9）。請重新輸入，例如：65 / 和 / 閒6莊5"

    pf = _get_pf_from_sess(sess)
    pf.update_point_history(p_pts, b_pts)
    sess["hand_idx"] = int(sess.get("hand_idx", 0)) + 1
    margin = abs(p_pts - b_pts)
    last_gap = float(sess.get("last_prob_gap", 0.0))
    w = 1.0 + 0.95 * (abs(p_pts - b_pts) / 9.0)
    REP_CAP = 3
    rep = max(1, min(REP_CAP, int(round(w))))
    if p_pts == b_pts:
        try: pf.update_outcome(2)
        except Exception: pass
    else:
        outcome = 1 if p_pts > b_pts else 0
        for _ in range(rep):
            try: pf.update_outcome(outcome)
            except Exception: pass

    last_real = sess.get("hist_real", [])
    cooling = False
    if len(last_real)>=1 and last_real[-1]=="和":
        cooling = True

    sims_pred = int(os.getenv("PF_PRED_SIMS","30"))
    p_raw = pf.predict(sims_per_particle=sims_pred)
    p_adj = p_raw / np.sum(p_raw)
    p_temp = np.exp(np.log(np.clip(p_adj,1e-9,1.0)) / float(os.getenv("PROB_TEMP","0.95")))
    p_temp = p_temp / np.sum(p_temp)
    if "prob_sma" not in sess: sess["prob_sma"] = None
    alpha = float(os.getenv("PROB_SMA_ALPHA","0.39"))
    def ema(prev, cur, alpha): return cur if prev is None else alpha*cur + (1-alpha)*prev
    sess["prob_sma"] = ema(sess["prob_sma"], p_temp, alpha)
    p_final = sess["prob_sma"] if sess["prob_sma"] is not None else p_temp

    dragon = _is_long_dragon(sess, dragon_len=7)
    if dragon:
        choice_text = dragon
        edge = abs(float(p_final[0]) - float(p_final[1]))
    else:
        pB, pP, pT = float(p_final[0]), float(p_final[1]), float(p_final[2])
        edge = abs(pB - pP)
        if pB >= pP: choice_text = "莊"
        else:        choice_text = "閒"

    if np.isnan(p_final).any() or np.sum(p_final) < 0.99:
        if random.random() < 0.5: choice_text = "莊"
        else:                     choice_text = "閒"
        edge = 0.02

    watch = False
    reasons = []
    if cooling:
        watch = True; reasons.append("和局冷卻")
    elif edge < float(os.getenv("EDGE_ENTER","0.007")):
        watch = True; reasons.append("機率差過小")
    elif float(p_final[2]) > float(os.getenv("TIE_PROB_MAX","0.18")):
        watch = True; reasons.append("和局風險高")
    elif abs(edge - last_gap) > float(os.getenv("WATCH_INSTAB_THRESH","0.16")):
        watch = True; reasons.append("勝率波動大")

    bankroll = int(sess.get("bankroll", 0))
    bet_pct = 0.0
    if not watch:
        if edge < 0.015:
            bet_pct = 0.08
        elif edge < 0.03:
            bet_pct = 0.14
        else:
            bet_pct = 0.26
    bet_amt = int(round(bankroll * bet_pct)) if bankroll>0 and bet_pct>0 else 0

    st = sess.setdefault("stats", {"bets": 0, "wins": 0, "push": 0, "sum_edge": 0.0, "payout": 0})
    if p_pts == b_pts:
        st["push"] += 1
        real_label = "和"
    else:
        real_label = "閒" if p_pts > b_pts else "莊"
        if not watch:
            st["bets"] += 1
            st["sum_edge"] += float(edge)
            if choice_text == real_label:
                if real_label == "莊":
                    st["payout"] += int(round(bet_amt * 0.95))
                else:
                    st["payout"] += int(bet_amt)
                st["wins"] += 1
            else:
                st["payout"] -= int(bet_amt)
    pred_label = "觀望" if watch else choice_text
    if "hist_pred" not in sess: sess["hist_pred"] = []
    if "hist_real" not in sess: sess["hist_real"] = []
    sess["hist_pred"].append(pred_label)
    sess["hist_real"].append(real_label)
    if len(sess["hist_pred"])>200: sess["hist_pred"]=sess["hist_pred"][-200:]
    if len(sess["hist_real"])>200: sess["hist_real"]=sess["hist_real"][-200:]
    sess["last_pts_text"] = f"上局結果: {'和 '+str(p_pts) if p_pts==b_pts else '閒 '+str(p_pts)+' 莊 '+str(b_pts)}"
    sess["last_prob_gap"] = edge

    def _acc_ex_tie(sess, last_n=None):
        pred, real = sess.get("hist_pred", []), sess.get("hist_real", [])
        if last_n: pred, real = pred[-last_n:], real[-last_n:]
        pairs = [(p,r) for p,r in zip(pred,real) if r in ("莊","閒") and p in ("莊","閒")]
        if not pairs: return (0,0,0.0)
        hit = sum(1 for p,r in pairs if p==r)
        tot = len(pairs)
        return (hit, tot, 100.0*hit/tot)
    hit, tot, acc = _acc_ex_tie(sess, 30)
    acc_txt = f"📊 近30手命中率：{acc:.1f}%（{hit}/{tot}）" if tot > 0 else "📊 近30手命中率：尚無資料"

    strat = f"⚠️ 觀望（{'、'.join(reasons)}）" if watch else (
        f"🟡 低信心配注 {bet_pct*100:.1f}%" if bet_pct<0.13 else
        f"🟠 中信心配注 {bet_pct*100:.1f}%" if bet_pct<0.22 else
        f"🟢 高信心配注 {bet_pct*100:.1f}%"
    )

    msg = [
        sess["last_pts_text"],
        "開始分析下局....",
        "",
        "【預測結果】",
        f"閒：{p_final[1]*100:.2f}%",
        f"莊：{p_final[0]*100:.2f}%",
        f"和：{p_final[2]*100:.2f}%",
        f"本次預測結果：{pred_label} (優勢: {edge*100:.2f}%)",
        f"建議下注金額：{bet_amt:,}",
        f"配注策略：{strat}",
        acc_txt,
        "—",
        "🔁 連續模式：請直接輸入下一局點數（例：65 / 和 / 閒6莊5）",
    ]

    return "\n".join(msg)

# ================== LINE webhook流程(可用) ===================

from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

@app.route("/line-webhook", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except Exception as e:
        print("LINE webhook error:", e)
    return "ok", 200

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_id = event.source.user_id
    text = event.message.text.strip()
    sess = SESS.setdefault(user_id, {"bankroll": 10000})
    try:
        # 支援格式：65、閒6莊5、莊5閒6
        m = re.match(r"^(\d{2})$", text)
        if m:
            p_pts, b_pts = int(text[0]), int(text[1])
            reply = handle_points_and_predict(sess, p_pts, b_pts)
        elif re.search("閒(\d+).*莊(\d+)", text):
            mm = re.search("閒(\d+).*莊(\d+)", text)
            p_pts, b_pts = int(mm.group(1)), int(mm.group(2))
            reply = handle_points_and_predict(sess, p_pts, b_pts)
        elif re.search("莊(\d+).*閒(\d+)", text):
            mm = re.search("莊(\d+).*閒(\d+)", text)
            b_pts, p_pts = int(mm.group(1)), int(mm.group(2))
            reply = handle_points_and_predict(sess, p_pts, b_pts)
        elif "和" in text:
            reply = "和局目前不需輸入點數，請直接輸入如：65"
        else:
            reply = "請輸入正確格式，例如 65 代表閒6莊5，或 閒6莊5 / 莊5閒6"
    except Exception as e:
        reply = f"❌ 輸入格式有誤: {e}"

    try:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply)
        )
    except Exception as e:
        print("LINE reply_message error:", e)

# ---------- MAIN ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT","8000"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
    log = logging.getLogger("bgs-server")
    log.info("Starting BGS-PF on port %s", port)
    app.run(host="0.0.0.0", port=port, debug=False)
