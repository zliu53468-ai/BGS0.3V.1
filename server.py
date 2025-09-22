# -*- coding: utf-8 -*-
"""
server.py — Right-side base + Left-side prediction tweaks + 30min Trial Gate
- 保留右側整體流程/指令/按鈕/Redis/REST/LINE webhook
- 採用左側去敏化 PF + should_bet 保守入場
- 回覆訊息為你指定的段落樣式
- 新增 30 分鐘試用與「開通 <密碼>」解除限制（讀 TRIAL_MINUTES / ADMIN_ACTIVATION_SECRET）
"""

import os, sys, re, time, json, math, random, logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

# ---------- Optional deps (Flask/LINE/Redis) ----------
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    _has_flask = True
except Exception:
    _has_flask = False
    Flask = None  # type: ignore
    def jsonify(*_, **__): raise RuntimeError("Flask not available")
    def CORS(*_, **__): pass

try:
    import redis
except Exception:
    redis = None

# ---------- Version & logging ----------
VERSION = "bgs-right+left-pred-trial-2025-09-22"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL","INFO"),
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s"
)
log = logging.getLogger("bgs-bot")

# ---------- Flask ----------
if _has_flask:
    app = Flask(__name__)
    CORS(app)
else:
    app = None  # type: ignore

# ---------- Env helpers ----------
def env_flag(name: str, default: int = 0) -> bool:
    v = os.getenv(name, str(default)).strip().lower()
    return v in ("1","true","yes","y","on")

# ---------- Core helpers ----------
def softmax(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=float)
    x = np.clip(x, 1e-12, None)
    x = x / np.sum(x)
    return x

def ema(prev: Optional[np.ndarray], cur: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None: return cur
    return alpha*cur + (1-alpha)*prev

# ====== 可調參數區（右側保留 + 左側去敏化預設） ======
# Trial / Activation
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "30"))  # 預設 30
ADMIN_ACTIVATION_SECRET = os.getenv("ADMIN_ACTIVATION_SECRET", "")  # 可留空

# Thompson
TS_EN = env_flag("TS_EN", 0)
TS_ALPHA = float(os.getenv("TS_ALPHA","2"))
TS_BETA  = float(os.getenv("TS_BETA","2"))

# 入場/觀望條件（用左側保守值）
EDGE_ENTER = float(os.getenv("EDGE_ENTER","0.015"))
TIE_PROB_MAX = float(os.getenv("TIE_PROB_MAX","0.16"))
WATCH_EN = env_flag("WATCH_EN", 1)
WATCH_INSTAB_THRESH = float(os.getenv("WATCH_INSTAB_THRESH","0.12"))

# 不確定性懲罰/點差權重（保留右側；係數取合理預設）
UNCERT_PENALTY_EN = env_flag("UNCERT_PENALTY_EN", 1)
UNCERT_MARGIN_MAX = int(os.getenv("UNCERT_MARGIN_MAX","2"))
UNCERT_RATIO = float(os.getenv("UNCERT_RATIO","0.25"))

W_BASE   = float(os.getenv("W_BASE","0.8"))
W_MIN    = float(os.getenv("W_MIN","0.5"))
W_MAX    = float(os.getenv("W_MAX","2.0"))
W_ALPHA  = float(os.getenv("W_ALPHA","0.7"))
W_SIG_K  = float(os.getenv("W_SIG_K","1.5"))
W_SIG_MID= float(os.getenv("W_SIG_MID","2.0"))
W_GAMMA  = float(os.getenv("W_GAMMA","0.6"))
W_GAP_CAP= float(os.getenv("W_GAP_CAP","0.08"))

DEPTH_W_EN  = env_flag("DEPTH_W_EN", 1)
DEPTH_W_MAX = float(os.getenv("DEPTH_W_MAX","1.5"))

# 溫度/平滑（保留右側）
TEMP_EN = env_flag("TEMP_EN", 1)
TEMP    = float(os.getenv("TEMP","0.95"))
SMOOTH_EN = env_flag("SMOOTH_EN", 1)
SMOOTH_ALPHA = float(os.getenv("SMOOTH_ALPHA","0.7"))

# 抽水與決策模式
BANKER_COMMISSION = float(os.getenv("BANKER_COMMISSION","0.05"))
DECIDE_MODE = os.getenv("DECIDE_MODE","prob")  # prob|ev

# ---------- PF dummy（右側簡化版） ----------
class PFHealth:
    def __init__(self): self.is_dummy=True
PF_HEALTH = PFHealth()

class OutcomePF:
    def __init__(self, decks=6, seed=42, n_particles=60, sims_lik=30, resample_thr=0.7, backend="mc", dirichlet_eps=0.003):
        random.seed(seed); np.random.seed(seed)
        self.n=n_particles; self.sims_lik=sims_lik; self.resample_thr=resample_thr
        self.backend=backend; self.dirichlet_eps=dirichlet_eps
        self.hist=[]
    def update_point_history(self, p_pts, b_pts):
        self.hist.append((p_pts,b_pts))
    def update_outcome(self, outcome):
        pass
    def predict(self) -> np.ndarray:
        if len(self.hist)>=4:
            xs = [1,1,1]
            for p,b in self.hist[-6:]:
                if p==b: xs[2]+=1
                elif p>b: xs[1]+=1
                else: xs[0]+=1
            return softmax(np.array(xs, dtype=float))
        return np.array([0.45,0.45,0.10])

# ---------- Redis / Session（右側保留） ----------
REDIS_URL = os.getenv("REDIS_URL")
_r = None
if REDIS_URL and redis:
    try:
        _r = redis.from_url(REDIS_URL, decode_responses=True)
        log.info("Redis connected")
    except Exception as e:
        log.warning("Redis connect fail: %s", e)

def load_sess(uid: str) -> Dict[str,Any]:
    sess: Dict[str,Any] = {}
    if _r:
        j = _r.get(f"sess:{uid}")
        if j: 
            try: sess = json.loads(j)
            except: sess={}
    sess.setdefault("stats", {"bets":0,"wins":0,"sum_edge":0.0,"payout":0,"push":0})
    sess.setdefault("hand_idx", 0)
    return sess

def save_sess(uid: str, sess: Dict[str,Any]):
    if _r:
        try: _r.set(f"sess:{uid}", json.dumps(sess) )
        except Exception as e: log.warning("Redis save sess err: %s", e)

# ---------- Trial helpers ----------
_trial_local: Dict[str, Dict[str,str]] = {}  # 無 Redis 時的本地備援

def _now_ts() -> int:
    return int(time.time())

def _trial_key(uid: str) -> str:
    return f"trial:{uid}"

def trial_start_if_needed(uid: str):
    """若未開始，立即紀錄 trial_start（永久綁定 uid）"""
    ts = str(_now_ts())
    if TRIAL_MINUTES <= 0:  # 關閉試用
        return
    if _r:
        if _r.get(_trial_key(uid)) is None:
            _r.set(_trial_key(uid), ts)
    else:
        _trial_local.setdefault(uid, {"trial_start": ts})

def trial_seconds_remaining(uid: str) -> int:
    """回傳剩餘秒數（<0 代表到期）"""
    limit_s = TRIAL_MINUTES * 60
    if TRIAL_MINUTES <= 0:
        return 9_999_999
    if _r:
        v = _r.get(_trial_key(uid))
        if v is None:
            return limit_s
        start = int(v)
    else:
        start = int(_trial_local.get(uid, {}).get("trial_start", _now_ts()))
    return limit_s - (_now_ts() - start)

def is_vip(sess: Dict[str,Any]) -> bool:
    return bool(sess.get("vip", False))

# ---------- PF 管理（右側保留；左側去敏化預設） ----------
def _get_pf_from_sess(sess: Dict[str,Any]) -> OutcomePF:
    if sess.get("pf") is None:
        try:
            sess["pf"] = OutcomePF(
                decks=int(os.getenv("DECKS","6")),
                seed=int(os.getenv("SEED","42")),
                n_particles=int(os.getenv("PF_N","120")),                 # 60 -> 120
                sims_lik=max(1,int(os.getenv("PF_UPD_SIMS","40"))),       # 30 -> 40
                resample_thr=float(os.getenv("PF_RESAMPLE","0.85")),      # 0.7 -> 0.85
                backend=os.getenv("PF_BACKEND","mc"),
                dirichlet_eps=float(os.getenv("PF_DIR_EPS","0.025")),     # 0.003 -> 0.025
            )
            log.info("Per-session PF init ok")
        except Exception as e:
            log.error("Per-session PF init fail: %s; use dummy", e)
            sess["pf"] = OutcomePF()
            PF_HEALTH.is_dummy = True
    return sess["pf"]

# ---------- 下注/配注（右側保留） ----------
def base_bet_pct(edge: float, maxp: float) -> float:
    k1 = float(os.getenv("BET_K1","0.60"))
    k2 = float(os.getenv("BET_K2","0.40"))
    return max(0.05, min(0.4, k1*edge + k2*(maxp-0.5)))

def thompson_scale_pct(pct: float) -> float:
    if not TS_EN: return pct
    s = np.random.beta(TS_ALPHA, TS_BETA)
    return max(0.05, min(0.4, pct * (0.8 + 0.4*s)))

def bet_amount(bankroll: int, pct: float) -> int:
    return int(round(max(0, bankroll) * pct))

# ---------- 輸入解析（右側保留） ----------
def parse_last_hand_points(text: str) -> Optional[Tuple[int,int]]:
    t = text.strip()
    if t in ("和","和局"):
        x = random.randint(0,9)
        return (x,x)
    m = re.search(r"([閒Pp])\s*([0-9])\s*[莊Bb]\s*([0-9])", t)
    if m:
        try: return (int(m.group(2)), int(m.group(3)))
        except: return None
    m = re.search(r"([0-9])\s*([0-9])", t)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None

def validate_input_data(p_pts: int, b_pts: int) -> bool:
    return (0 <= p_pts <= 9) and (0 <= b_pts <= 9)

# ---------- 命中率/優勢工具（取左側權重邏輯） ----------
def calc_margin_weight(p_pts: int, b_pts: int, last_prob_gap: float) -> float:
    margin = abs(int(p_pts) - int(b_pts))
    sig = 1.0/(1.0 + math.exp(-W_SIG_K * (margin - W_SIG_MID)))
    part_m = W_ALPHA * sig
    gap_norm = min(max(float(last_prob_gap),0.0), W_GAP_CAP) / max(W_GAP_CAP,1e-6)
    part_g = W_GAMMA * gap_norm
    w = W_BASE + part_m + part_g
    return max(W_MIN, min(W_MAX, w))

def should_bet(prob: Tuple[float,float,float], last_gap: float, cur_gap: float, idx: int) -> bool:
    """
    保守入場：max_prob>=0.52、T<=0.16、機率差變動不大、cur_gap>=1.8%、前5手略放寬
    """
    pB, pP, pT = prob
    max_prob = max(pB, pP)
    gap_change = abs(cur_gap - last_gap) if last_gap > 0 else 0.0
    conds = [
        max_prob >= 0.52,
        pT <= 0.16,
        gap_change <= 0.15,
        cur_gap >= 0.018,
        (idx > 5) or (max_prob >= 0.54)
    ]
    return all(conds)

def decide_bp(prob: np.ndarray) -> Tuple[str, float, float, float]:
    pB, pP, pT = float(prob[0]), float(prob[1]), float(prob[2])
    if DECIDE_MODE == "prob":
        choice = "莊" if pB>=pP else "閒"
        maxp = max(pB,pP)
        edge = max(0.0, maxp - (1.0-maxp-pT))
    else:
        evB = pB*(1.0 - BANKER_COMMISSION) - (pP)
        evP = pP - pB
        if evB>=evP:
            choice="莊"; edge=max(0.0, evB)
        else:
            choice="閒"; edge=max(0.0, evP)
        maxp = max(pB,pP)
    prob_gap = abs(pB - pP)
    return choice, edge, maxp, prob_gap

_prev_prob_sma: Optional[np.ndarray] = None

def analysis_confidence(prob: np.ndarray, last_gap: float, cur_gap: float) -> float:
    # 仍保留右側流程（不輸出顯示）
    pB, pP, pT = float(prob[0]), float(prob[1]), float(prob[2])
    maxp = max(pB,pP); gap=abs(pB-pP)
    x = 0.4*max(0, maxp-0.5)/0.5 + 0.4*max(0, gap)/0.5 + 0.2*max(0, 0.16-pT)/0.16
    x = max(0.0, min(1.0, x))
    return x

# ---------- 主邏輯（右側保留，僅換 watch 規則 + 訊息格式） ----------
def handle_points_and_predict(sess: Dict[str,Any], p_pts: int, b_pts: int) -> str:
    global _prev_prob_sma

    if not validate_input_data(p_pts, b_pts):
        return "❌ 點數數據異常（僅接受 0~9）。請重新輸入，例如：65 / 和 / 閒6莊5"

    pf = _get_pf_from_sess(sess)
    pf.update_point_history(p_pts, b_pts)

    # 1) 初步預測
    prob = pf.predict()  # [B,P,T]
    prob = softmax(prob)

    # 2) 不確定性懲罰 + 平滑/溫度
    last_gap = float(sess.get("last_prob_gap", 0.0))
    if UNCERT_PENALTY_EN:
        margin = min(abs(p_pts-b_pts), UNCERT_MARGIN_MAX)
        punish = (1.0 - UNCERT_RATIO * (margin/UNCERT_MARGIN_MAX))
        prob = softmax(prob * np.array([1.0,1.0,1.0]) * punish)

    if SMOOTH_EN:
        _prev_prob_sma = ema(_prev_prob_sma, prob, SMOOTH_ALPHA)
        prob = _prev_prob_sma

    if TEMP_EN and TEMP>0:
        t = max(1e-6, TEMP)
        prob = softmax(prob ** (1.0/t))

    # 3) 方向/優勢
    choice, edge, maxp, prob_gap = decide_bp(prob)

    # 4) watch 規則（用 should_bet）
    hand_idx = int(sess.get("hand_idx", 0))
    p_final = prob
    watch = not should_bet(tuple(p_final.tolist()), last_gap, prob_gap, hand_idx)

    # 5) 資金與配注
    bankroll = int(sess.get("bankroll", 0))
    _ = analysis_confidence(p_final, last_gap, prob_gap)  # 保留內部用
    pct_base = base_bet_pct(edge, maxp)
    bet_pct  = thompson_scale_pct(pct_base)
    bet_amt  = bet_amount(bankroll, bet_pct)

    if watch:
        choice_text = "觀望"
        bet_pct = 0.0
        bet_amt = 0
        strat = "⚠️ 觀望"
    else:
        choice_text = choice
        if bet_pct < 0.28:   strat = f"🟡 低信心配注 {bet_pct*100:.1f}%"
        elif bet_pct < 0.34: strat = f"🟠 中信心配注 {bet_pct*100:.1f}%"
        else:                strat = f"🟢 高信心配注 {bet_pct*100:.1f}%"

    # 6) 更新統計（右側保留）
    st = sess["stats"]
    if p_pts == b_pts:
        real_label = "和"; st["push"] += 1
    else:
        real_label = "閒" if p_pts>b_pts else "莊"
        if not watch:
            st["bets"] += 1
            st["sum_edge"] += float(edge)
            if choice_text == real_label:
                if real_label == "莊":
                    st["payout"] += int(round(bet_amt * BANKER_COMMISSION))
                else:
                    st["payout"] += int(bet_amt)
                st["wins"] += 1
            else:
                st["payout"] -= int(bet_amt)

    sess["hand_idx"] = int(sess.get("hand_idx",0)) + 1
    if p_pts == b_pts:
        last_txt = f"上局結果: 和 {p_pts}"
    else:
        last_txt = f"上局結果: 閒 {p_pts} 莊 {b_pts}"
    sess["last_pts_text"] = last_txt
    sess["last_prob_gap"] = prob_gap

    # 7) 訊息輸出（你指定的樣式）
    msg = [
        last_txt,
        "開始分析下局....",
        "",
        "【預測結果】",
        f"閒：{p_final[1]*100:.2f}%",
        f"莊：{p_final[0]*100:.2f}%",
        f"和：{p_final[2]*100:.2f}%",
        f"本次預測結果：{choice_text}(優勢: {edge*100:.1f}%)",
        f"建議下注金額：{bet_amt}",
        f"配注策略：{strat}",
        "—",
        "🔁 連續模式：請直接輸入下一局點數（例：65 / 和 / 閒6莊5）",
    ]

    if PF_HEALTH.is_dummy:
        msg.insert(0, "⚠️ 模型載入為簡化版（Dummy）。請確認 bgs.pfilter 是否可用。")

    return "\n".join(msg)

# ---------- 簡易 REST ----------
@app.get("/")
def root():
    return f"✅ BGS PF Server OK ({VERSION})", 200

@app.get("/health")
def health():
    try:
        return jsonify(ok=True, version=VERSION, dummy=PF_HEALTH.is_dummy), 200
    except Exception:
        return jsonify(ok=True, version=VERSION, dummy=True), 200

@app.post("/predict")
def predict_rest():
    data = request.get_json(force=True, silent=True) or {}
    uid = str(data.get("uid","demo"))
    # Trial gate for REST
    sess = load_sess(uid)
    if not is_vip(sess):
        trial_start_if_needed(uid)
        sec_left = trial_seconds_remaining(uid)
        if sec_left <= 0:
            return jsonify(ok=False, err="trial_expired", minutes=TRIAL_MINUTES), 403
    try:
        p_pts = int(data.get("p",0)); b_pts = int(data.get("b",0))
    except:
        return jsonify(ok=False, err="invalid p/b"), 400
    msg = handle_points_and_predict(sess, p_pts, b_pts)
    save_sess(uid, sess)
    return jsonify(ok=True, msg=msg), 200

# ---------- LINE webhook（右側保留 + Trial Gate + 歡迎＆選館訊息） ----------
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_CHANNEL_TOKEN  = os.getenv("LINE_CHANNEL_TOKEN")

if LINE_CHANNEL_SECRET and LINE_CHANNEL_TOKEN and _has_flask:
    try:
        from linebot import LineBotApi, WebhookHandler
        from linebot.models import MessageEvent, TextMessage, TextSendMessage, FollowEvent
        from linebot.exceptions import InvalidSignatureError, LineBotApiError
        _line_ok=True
    except Exception:
        _line_ok=False

    line_bot_api = LineBotApi(LINE_CHANNEL_TOKEN) if _line_ok else None
    handler = WebhookHandler(LINE_CHANNEL_SECRET) if _line_ok else None

    def reply_text(token: str, text: str, user_id: Optional[str]=None):
        if not _line_ok or not line_bot_api: return
        try:
            line_bot_api.reply_message(token, TextSendMessage(text=text))
        except LineBotApiError:
            try:
                if user_id:
                    line_bot_api.push_message(user_id, TextSendMessage(text=text))
            except Exception as e:
                log.warning("LINE push fail: %s", e)

    def _trial_notice_card(sec_left: int) -> str:
        # 你截圖的風格：顯示試用剩餘或到期提示
        if sec_left <= 0:
            return (
                "試用期已到 ❌\n"
                "📣 請聯繫管理員開通\n"
                "加入官方 LINE ：https://lin.ee/Dlm6y3u\n"
                "—\n"
                "輸入：開通 你的密碼"
            )
        else:
            mm, ss = divmod(max(0,sec_left), 60)
            return f"⏳ 試用剩餘：{mm} 分 {ss} 秒"

    @app.post("/line-webhook")
    def line_webhook():
        body = request.get_data(as_text=True)
        sig = request.headers.get("X-Line-Signature","")
        try:
            handler.handle(body, sig)
        except InvalidSignatureError:
            # 無憑證/測試時允許直通
            return "ok", 200
        return "ok", 200

    @handler.add(FollowEvent)
    def on_follow(event):
        uid = event.source.user_id if event.source else "demo"
        sess = load_sess(uid)
        if not is_vip(sess):
            trial_start_if_needed(uid)
            sec_left = trial_seconds_remaining(uid)
            card = _trial_notice_card(sec_left)
        else:
            card = "✅ 已開通完整功能。"

        welcome = (
            "👋 歡迎使用 BGS AI 預測分析！\n"
            "使用步驟：\n"
            "1️⃣ 選擇館別（輸入 1~10）\n"
            "2️⃣ 輸入桌號（例：DG01）\n"
            "3️⃣ 輸入本金（例：5000）\n"
            "4️⃣ 每局回報點數（例：65 / 和 / 閒6莊5）即可連續預測！\n\n"
            "【請選擇遊戲館別】\n"
            "1. WM\n2. PM\n3. DG\n4. SA\n5. KU\n6. 歐博/卡利\n7. KG\n8. 全利\n9. 名人\n10. MT真人\n"
            "(請直接輸入數字 1-10)"
        )
        reply_text(event.reply_token, f"{welcome}\n\n{card}", user_id=uid)

    @handler.add(MessageEvent, message=TextMessage)
    def on_text(event):
        uid = event.source.user_id if event.source else "demo"
        text = (event.message.text or "").strip()
        sess = load_sess(uid)

        # ===== 開通碼處理 =====
        if text.startswith("開通"):
            secret = text.replace("開通", "").strip()
            if ADMIN_ACTIVATION_SECRET and secret == ADMIN_ACTIVATION_SECRET:
                sess["vip"] = True
                save_sess(uid, sess)
                reply_text(event.reply_token, "✅ 已開通完整功能，無試用時間限制。", user_id=uid)
                return
            else:
                reply_text(event.reply_token, "❌ 開通碼錯誤。若忘記請洽管理員。", user_id=uid)
                return

        # ===== 試用限制（非 VIP 才檢查）=====
        if not is_vip(sess):
            trial_start_if_needed(uid)
            sec_left = trial_seconds_remaining(uid)
            if sec_left <= 0:
                # 顯示到期卡片（跟你截圖一致風格）
                reply_text(event.reply_token, _trial_notice_card(sec_left), user_id=uid)
                return

        # ===== 原本右側流程：本金/館別/點數 =====
        # 簡化引導指令：如果輸入「遊戲設定」或步驟數字，送歡迎+館別清單
        if text in ("遊戲設定","/start"):
            tip = (
                "👋 歡迎使用 BGS AI 預測分析！\n"
                "使用步驟：\n"
                "1️⃣ 選擇館別（輸入 1~10）\n"
                "2️⃣ 輸入桌號（例：DG01）\n"
                "3️⃣ 輸入本金（例：5000）\n"
                "4️⃣ 每局回報點數（例：65 / 和 / 閒6莊5）即可連續預測！\n\n"
                "【請選擇遊戲館別】\n"
                "1. WM\n2. PM\n3. DG\n4. SA\n5. KU\n6. 歐博/卡利\n7. KG\n8. 全利\n9. 名人\n10. MT真人\n"
                "(請直接輸入數字 1-10)"
            )
            if not is_vip(sess):
                sec_left = trial_seconds_remaining(uid)
                tip = f"{tip}\n\n{_trial_notice_card(sec_left)}"
            reply_text(event.reply_token, tip, user_id=uid)
            save_sess(uid, sess)
            return

        # 本金設定流程
        if sess.get("phase","await_pts") == "await_bankroll":
            try:
                bk = int(re.sub(r"[^0-9]","", text))
                if bk>0:
                    sess["bankroll"]=bk; sess["phase"]="await_pts"
                    save_sess(uid, sess)
                    reply_text(event.reply_token, "✅ 已設定本金。請回報上一局點數（例：65 / 和 / 閒6莊5）", user_id=uid); return
            except: pass
            reply_text(event.reply_token, "請輸入數字本金（例：5000）。", user_id=uid); return

        # 點數 or 其他指令
        pts = parse_last_hand_points(text)
        if pts is None:
            # 還沒設定本金 → 先要求設定
            if not int(sess.get("bankroll",0)):
                sess["phase"] = "await_bankroll"; save_sess(uid, sess)
                msg = "請先輸入本金（例：5000），再回報點數。"
                if not is_vip(sess):
                    sec_left = trial_seconds_remaining(uid)
                    msg = f"{msg}\n{_trial_notice_card(sec_left)}"
                reply_text(event.reply_token, msg, user_id=uid); return
            # 非點數輸入 → 提示格式
            msg = "點數格式錯誤（例：65 / 和 / 閒6莊5）。"
            if not is_vip(sess):
                sec_left = trial_seconds_remaining(uid)
                msg = f"{msg}\n{_trial_notice_card(sec_left)}"
            reply_text(event.reply_token, msg, user_id=uid); return

        # 必要的本金保護
        if not int(sess.get("bankroll",0)):
            sess["phase"] = "await_bankroll"; save_sess(uid, sess)
            reply_text(event.reply_token, "請先輸入本金（例：5000），再回報點數。", user_id=uid); return

        # 產生預測訊息
        msg = handle_points_and_predict(sess, int(pts[0]), int(pts[1]))
        # 在每次回覆尾端附上試用剩餘提示（未開通時）
        if not is_vip(sess):
            sec_left = trial_seconds_remaining(uid)
            tail = _trial_notice_card(sec_left)
            if sec_left > 0:
                msg = f"{msg}\n\n{tail}"
        sess["phase"] = "await_pts"; save_sess(uid, sess)
        reply_text(event.reply_token, msg, user_id=uid)

else:
    @app.post("/line-webhook")
    def line_webhook_min():
        return "ok", 200

# ---------- Main ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT","8000"))
    log.info("Starting %s on port %s (DECIDE_MODE=%s, COMM=%.3f, TRIAL=%dm)...",
             VERSION, port, os.getenv("DECIDE_MODE","prob"), BANKER_COMMISSION, TRIAL_MINUTES)
    app.run(host="0.0.0.0", port=port, debug=False)
