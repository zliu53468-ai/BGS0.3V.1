# -*- coding: utf-8 -*-
"""
server.py — BGS百家樂AI 多步驟/館別桌號/本金/30分試用/永久帳號/粒子濾波動態預測
"""
import os, sys, re, time, json, math, random, logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import threading  # NEW: for async LINE reply

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
        def get(self, *a, **k): 
            def deco(f): return f
            return deco
        def post(self, *a, **k): 
            def deco(f): return f
            return deco
        def run(self, *a, **k): print("Flask not installed; dummy app.")
    app = _DummyApp()

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

# ---- 既有參數（維持你的原本預設） ----
os.environ.setdefault("PF_N", "80")
os.environ.setdefault("PF_RESAMPLE", "0.73")
os.environ.setdefault("PF_DIR_EPS", "0.012")
os.environ.setdefault("EDGE_ENTER", "0.007")  # 仍保留，但決策已改用 EV 門檻
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

# ---- EV 決策（含抽水） ----
os.environ.setdefault("BANKER_COMMISSION", "0.05")  # 莊家抽水 5%
os.environ.setdefault("EDGE_ENTER_EV", "0.010")     # EV 進場門檻；保守 0.012~0.015

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

TRIAL_SECONDS = 1800  # 30分鐘
OPENCODE = "aaa8881688"   # 開通碼
ADMIN_LINE = "https://lin.ee/Dlm6Y3u"  # 你的LINE

def _now(): return int(time.time())

def _get_user_info(user_id):
    k = f"bgsu:{user_id}"
    if rcli:
        s = rcli.get(k)
        if s:
            return json.loads(s)
    return SESS.get(user_id, {})

def _set_user_info(user_id, info):
    k = f"bgsu:{user_id}"
    if rcli:
        rcli.set(k, json.dumps(info), ex=86400)
    SESS[user_id] = info

def _is_trial_valid(user_id):
    info = _get_user_info(user_id)
    if info.get("is_opened"): return True
    if not info.get("trial_start"): return False
    return (_now() - int(info["trial_start"])) < TRIAL_SECONDS

def _start_trial(user_id):
    info = _get_user_info(user_id)
    if info.get("is_opened"): return
    if not info.get("trial_start"):
        info["trial_start"] = _now()
        _set_user_info(user_id, info)

def _set_opened(user_id):
    info = _get_user_info(user_id)
    info["is_opened"] = True
    _set_user_info(user_id, info)

def _left_trial_sec(user_id):
    info = _get_user_info(user_id)
    if info.get("is_opened"): return "永久"
    if not info.get("trial_start"): return "尚未啟動"
    left = TRIAL_SECONDS - (_now() - int(info["trial_start"]))
    return f"{left//60} 分 {left%60} 秒" if left > 0 else "已到期"

def _is_long_dragon(sess: Dict[str,Any], dragon_len=7) -> Optional[str]:
    # 仍保留函式，但不再覆蓋 EV 決策
    pred = sess.get("hist_real", [])
    if len(pred) < dragon_len: return None
    lastn = pred[-dragon_len:]
    if all(x=="莊" for x in lastn): return "莊"
    if all(x=="閒" for x in lastn): return "閒"
    return None

# ---------------- 主預測（EV 決策；獨立模式不做EMA/溫度） ----------------
def handle_points_and_predict(sess: Dict[str,Any], p_pts: int, b_pts: int) -> str:
    if not (0 <= int(p_pts) <= 9 and 0 <= int(b_pts) <= 9):
        return "❌ 點數數據異常（僅接受 0~9）。請重新輸入，例如：65 / 和 / 閒6莊5"

    pf = _get_pf_from_sess(sess)
    pf.update_point_history(p_pts, b_pts)
    sess["hand_idx"] = int(sess.get("hand_idx", 0)) + 1

    # 保留你既有的 update_outcome 設計（learn 模式才會生效）
    w = 1.0 + 0.95 * (abs(p_pts - b_pts) / 9.0)
    REP_CAP = 3
    rep = max(1, min(REP_CAP, int(round(w))))
    if p_pts == b_pts:
        try: pf.update_outcome(2)
        except Exception: pass
        real_label = "和"
    else:
        outcome = 1 if p_pts > b_pts else 0
        real_label = "閒" if p_pts > b_pts else "莊"
        for _ in range(rep):
            try: pf.update_outcome(outcome)
            except Exception: pass

    # 和後冷卻（保留）
    last_real = sess.get("hist_real", [])
    cooling = len(last_real)>=1 and last_real[-1]=="和"

    # 機率取得
    sims_pred = int(os.getenv("PF_PRED_SIMS","30"))
    p_raw = pf.predict(sims_per_particle=sims_pred)
    p_adj = p_raw / np.sum(p_raw)

    # ★ indep 模式不做溫度/EMA；learn 模式才做（避免跨手殘留偏差）
    if os.getenv("MODEL_MODE","indep").strip().lower() == "indep":
        p_final = p_adj
    else:
        p_temp = np.exp(np.log(np.clip(p_adj,1e-9,1.0)) / float(os.getenv("PROB_TEMP","0.95")))
        p_temp = p_temp / np.sum(p_temp)
        if "prob_sma" not in sess: sess["prob_sma"] = None
        alpha = float(os.getenv("PROB_SMA_ALPHA","0.39"))
        def ema(prev, cur, a): return cur if prev is None else a*cur + (1-a)*prev
        sess["prob_sma"] = ema(sess["prob_sma"], p_temp, alpha)
        p_final = sess["prob_sma"] if sess["prob_sma"] is not None else p_temp

    # ===== 以 EV（含莊家抽水）選邊 =====
    pB, pP, pT = float(p_final[0]), float(p_final[1]), float(p_final[2])
    BCOMM = float(os.getenv("BANKER_COMMISSION", "0.05"))
    ev_b = pB * (1.0 - BCOMM) - pP   # 壓莊 EV（和局當退回）
    ev_p = pP - pB                    # 壓閒 EV
    ev_choice = "莊" if ev_b > ev_p else "閒"
    edge_ev = ev_b if ev_b > ev_p else ev_p

    # 不再用龍覆蓋，避免偏性：一律交由 EV 決策
    choice_text = ev_choice

    # 例外防護
    if np.isnan(p_final).any() or np.sum(p_final) < 0.99:
        choice_text = "莊" if random.random() < 0.5 else "閒"
        edge_ev = 0.02

    # ===== 風控：EV 門檻 + 其他既有條件 =====
    watch = False
    reasons = []
    if cooling:
        watch = True; reasons.append("和局冷卻")

    EDGE_ENTER_EV = float(os.getenv("EDGE_ENTER_EV", "0.010"))
    if edge_ev < EDGE_ENTER_EV:
        watch = True; reasons.append("EV優勢過小")

    if float(p_final[2]) > float(os.getenv("TIE_PROB_MAX","0.18")):
        watch = True; reasons.append("和局風險高")

    # 優勢波動檢查（用 EV）
    last_gap = float(sess.get("last_prob_gap", 0.0))
    if abs(edge_ev - last_gap) > float(os.getenv("WATCH_INSTAB_THRESH","0.16")):
        watch = True; reasons.append("勝率波動大")

    # ===== 配注：以 EV 強度分級 =====
    bankroll = int(sess.get("bankroll", 0))
    bet_pct = 0.0
    if not watch:
        if edge_ev < 0.010:
            bet_pct = 0.08
        elif edge_ev < 0.020:
            bet_pct = 0.14
        else:
            bet_pct = 0.26
    bet_amt = int(round(bankroll * bet_pct)) if bankroll>0 and bet_pct>0 else 0

    # ===== 統計 =====
    st = sess.setdefault("stats", {"bets": 0, "wins": 0, "push": 0, "sum_edge": 0.0, "payout": 0})
    if real_label == "和":
        st["push"] += 1
    else:
        if not watch:
            st["bets"] += 1
            st["sum_edge"] += float(edge_ev)
            if choice_text == real_label:
                if real_label == "莊":
                    st["payout"] += int(round(bet_amt * 0.95))
                else:
                    st["payout"] += int(bet_amt)
                st["wins"] += 1
            else:
                st["payout"] -= int(bet_amt)

    # ===== 歷史/輸出 =====
    pred_label = "觀望" if watch else choice_text
    if "hist_pred" not in sess: sess["hist_pred"] = []
    if "hist_real" not in sess: sess["hist_real"] = []
    sess["hist_pred"].append(pred_label)
    sess["hist_real"].append(real_label)
    if len(sess["hist_pred"])>200: sess["hist_pred"]=sess["hist_pred"][-200:]
    if len(sess["hist_real"])>200: sess["hist_real"]=sess["hist_real"][-200:]
    sess["last_pts_text"] = f"上局結果: {'和 '+str(p_pts) if p_pts==b_pts else '閒 '+str(p_pts)+' 莊 '+str(b_pts)}"

    sess["last_prob_gap"] = edge_ev  # 記錄 EV 優勢（供波動檢查）

    # log（可選）：為什麼觀望
    logging.info("WATCH=%s reasons=%s EV=%.4f pT=%.3f", watch, ",".join(reasons), edge_ev, pT)

    # 命中率（近30手，排除和）
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
        f"本次預測結果：{'觀望' if watch else choice_text} (EV優勢: {edge_ev*100:.2f}%)",
        f"建議下注金額：{bet_amt:,}",
        f"配注策略：{strat}",
        acc_txt,
        "—",
        "🔁 連續模式：請直接輸入下一局點數（例：65 / 和 / 閒6莊5）",
    ]
    return "\n".join(msg)

# ========== LINE webhook主流程（非阻塞回覆，避免逾時） ==========
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_TIMEOUT = float(os.getenv("LINE_TIMEOUT", "2.0"))  # NEW: 連線逾時（秒）

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN, timeout=LINE_TIMEOUT)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

def _async_reply(token, text):
    try:
        line_bot_api.reply_message(token, TextSendMessage(text=text))
    except Exception as e:
        print("LINE reply_message error:", e)

@app.route("/line-webhook", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except Exception as e:
        print("LINE webhook error:", e)
    return "ok", 200

def welcome_text(uid):
    left = _left_trial_sec(uid)
    return (
        "👋 歡迎使用 BGS AI 預測分析！\n"
        "【使用步驟】\n"
        "1️⃣ 選擇館別（輸入 1~10）\n"
        "2️⃣ 輸入桌號（例：DG01）\n"
        "3️⃣ 輸入本金（例：5000）\n"
        "4️⃣ 每局回報點數（例：65 / 和 / 閒6莊5）\n"
        f"💾 試用剩餘：{left}\n\n"
        "【請選擇遊戲館別】\n"
        "1. WM\n2. PM\n3. DG\n4. SA\n5. KU\n6. 歐博/卡利\n7. KG\n8. 金利\n9. 名人\n10. MT真人\n"
        "(請直接輸入數字1-10)"
    )

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_id = event.source.user_id
    text = event.message.text.strip()
    info = _get_user_info(user_id)

    # 開通碼解封（不變）
    if text.startswith("開通"):
        pwd = text[2:].strip()
        if pwd == OPENCODE:
            _set_opened(user_id)
            reply = "✅ 已開通成功！"
        else:
            reply = "❌ 開通碼錯誤，請重新輸入。"
        threading.Thread(target=_async_reply, args=(event.reply_token, reply), daemon=True).start()
        return

    # 試用驗證（不變）
    if not _is_trial_valid(user_id):
        msg = (
            "⛔ 試用期已到\n"
            f"📬 請聯繫管理員開通登入帳號\n👉 加入官方 LINE：{ADMIN_LINE}"
        )
        threading.Thread(target=_async_reply, args=(event.reply_token, msg), daemon=True).start()
        return

    _start_trial(user_id)

    sess = SESS.setdefault(user_id, {"bankroll": 0})
    sess["user_id"] = user_id

    # ===== 多步驟引導（不變） =====
    if not sess.get("hall_id"):
        if text.isdigit() and 1 <= int(text) <= 10:
            sess["hall_id"] = int(text)
            hall_map = ["WM", "PM", "DG", "SA", "KU", "歐博/卡利", "KG", "金利", "名人", "MT真人"]
            hall_name = hall_map[int(text)-1]
            reply = f"✅ 已選 [{hall_name}]\n請輸入桌號（例：DG01，格式：2字母+2數字）"
        else:
            reply = welcome_text(user_id)
        threading.Thread(target=_async_reply, args=(event.reply_token, reply), daemon=True).start()
        return

    if not sess.get("table_id"):
        m = re.match(r"^[a-zA-Z]{2}\d{2}$", text)
        if m:
            sess["table_id"] = text.upper()
            reply = f"✅ 已設桌號 [{sess['table_id']}]\n請輸入您的本金（例：5000）"
        else:
            reply = "請輸入正確格式的桌號（例：DG01，格式：2字母+2數字）"
        threading.Thread(target=_async_reply, args=(event.reply_token, reply), daemon=True).start()
        return

    if not sess.get("bankroll") or sess["bankroll"] <= 0:
        m = re.match(r"^(\d{3,7})$", text)
        if m:
            sess["bankroll"] = int(text)
            reply = f"👍 已設定本金：{sess['bankroll']:,}\n請輸入上一局點數（例：65 / 和 / 閒6莊5），之後能連續傳手。"
        else:
            reply = "請輸入正確格式的本金（例：5000）"
        threading.Thread(target=_async_reply, args=(event.reply_token, reply), daemon=True).start()
        return

    # 4. 每局輸入點數，進入預測主邏輯（不變入口）
    try:
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

    # 非阻塞回覆，Webhook 立即返回 200（在 callback() 中）
    threading.Thread(target=_async_reply, args=(event.reply_token, reply), daemon=True).start()

if __name__ == "__main__":
    port = int(os.getenv("PORT","8000"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
    log = logging.getLogger("bgs-server")
    log.info("Starting BGS-PF on port %s", port)
    app.run(host="0.0.0.0", port=port, debug=False)
