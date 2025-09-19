# -*- coding: utf-8 -*-
"""
server.py — Render 免費版優化 + 點差連續加權 + 不確定性懲罰
附加：手數深度權重、點差可靠度表(5桶)、Thompson 配注縮放（皆可用環境變數開關）
含 LINE webhook（有憑證→驗簽處理；否則自動退回 200）

不改你的主流程：解析點數 → PF.update_outcome → PF.predict → EV 決策 → 配注映射
"""

import os, sys, re, time, json, math, random, logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

# ---------- 可選依賴（Flask/LINE/Redis） ----------
try:
    from flask import Flask, request, jsonify, abort
    from flask_cors import CORS
    _has_flask = True
except Exception:
    _has_flask = False
    Flask = None  # type: ignore
    def jsonify(*_, **__): raise RuntimeError("Flask not available")
    def abort(*_, **__): raise RuntimeError("Flask not available")
    def CORS(*_, **__): pass

try:
    import redis
except Exception:
    redis = None

# ---------- 版本 & 日誌 ----------
VERSION = "pf-adv-render-free-2025-09-19-line+cmds"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-server")

# ---------- Flask 初始化 ----------
if _has_flask:
    app = Flask(__name__)
    CORS(app)
else:
    class _DummyApp:
        def get(self, *a, **k):
            def deco(f): return f
            return deco
        def post(self, *a, **k):
            def deco(f): return f
            return deco
        def run(self, *a, **k):
            log.warning("Flask not installed; dummy app.")
    app = _DummyApp()

# ---------- Redis / Fallback ----------
REDIS_URL = os.getenv("REDIS_URL", "")
rcli = None
if redis and REDIS_URL:
    try:
        rcli = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
        rcli.ping()
        log.info("Redis connected.")
    except Exception as e:
        rcli = None
        log.warning("Redis connect fail: %s => fallback memory store", e)

SESS: Dict[str, Dict[str, Any]] = {}  # fallback

SESSION_EXPIRE = 3600
def _rget(k: str) -> Optional[str]:
    try:
        if rcli: return rcli.get(k)
    except Exception as e:
        log.warning("Redis GET err: %s", e)
    return None

def _rset(k: str, v: str, ex: Optional[int] = None):
    try:
        if rcli: rcli.set(k, v, ex=ex)
    except Exception as e:
        log.warning("Redis SET err: %s", e)

# ---------- 旗標 ----------
def env_flag(name: str, default: int=0) -> int:
    v = os.getenv(name)
    if v is None: return 1 if default else 0
    v = v.strip().lower()
    if v in ("1","true","t","yes","y","on"): return 1
    if v in ("0","false","f","no","n","off"): return 0
    try: return 1 if int(float(v))!=0 else 0
    except: return 1 if default else 0

# ---------- 會話 ----------
def now_sess(uid: str) -> Dict[str, Any]:
    if rcli:
        j = _rget(f"sess:{uid}")
        if j:
            try: return json.loads(j)
            except: pass
    s = SESS.get(uid)
    if s: return s
    s = {
        "bankroll": 0,
        "phase": "choose_game",
        "game": None, "table": None,
        "trial_start": int(time.time()),
        "premium": True,  # 省略試用流程
        "last_pts_text": None,
        "last_prob_gap": 0.0,
        "hand_idx": 0,  # 每桌手數索引
        # margin reliability (5桶: 0,1,2,3,4+): alpha/beta
        "mrel": {"a":[1.0,1.0,1.0,1.0,1.0], "b":[1.0,1.0,1.0,1.0,1.0]},
    }
    SESS[uid] = s
    return s

def save_sess(uid: str, s: Dict[str, Any]):
    if rcli:
        _rset(f"sess:{uid}", json.dumps(s), ex=SESSION_EXPIRE)
    else:
        SESS[uid] = s

# ---------- 常用工具 ----------
def _norm(s: str) -> str:
    """全半形與空白標準化，轉大寫，便於比對指令。"""
    if not s: return ""
    s = s.translate(str.maketrans("　０１２３４５６７８９：－—～〜", " 0123456789:----"))
    s = re.sub(r"\s+", " ", s.strip())
    return s.upper()

def _quick_reply():
    try:
        from linebot.models import QuickReply, QuickReplyButton, MessageAction
        return QuickReply(items=[
            QuickReplyButton(action=MessageAction(label="遊戲設定 🎮", text="遊戲設定")),
            QuickReplyButton(action=MessageAction(label="結束分析 🧹", text="結束分析")),
            QuickReplyButton(action=MessageAction(label="報莊 🅱️", text="B")),
            QuickReplyButton(action=MessageAction(label="報閒 🅿️", text="P")),
            QuickReplyButton(action=MessageAction(label="報和 ⚪", text="T")),
        ])
    except Exception:
        return None

# ---------- 解析點數（放寬版） ----------
def parse_last_hand_points(text: str) -> Optional[Tuple[int,int]]:
    """
    更寬鬆版解析器：
    - 支援：65 / 6-5 / 6:5 / 閒6莊5 / 莊5閒6 / B6P5 / P6B5 / 和 / TIE / DRAW / T
    - 忽略桌號/emoji/符號，只取最後兩個單一位數（0-9）保底
    """
    if not text:
        return None

    s = str(text)
    s = s.translate(str.maketrans("０１２３４５６７８９：－—～〜", "0123456789:----"))
    s = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\r\n\t]", "", s)
    u = re.sub(r"\s+", " ", s).strip().upper()

    # 和局
    if re.search(r"\b(和|TIE|DRAW|^T$)\b", u):
        m = re.search(r"(?:和|TIE|DRAW|T)\s*:?\s*([0-9])", u)
        if m:
            d = int(m.group(1)); return (d, d)
        return (0, 0)

    # 閒x莊y / 莊y閒x
    m = re.search(r"(閒|P)\s*:?\s*([0-9]).*?(莊|B)\s*:?\s*([0-9])", u)
    if m: return (int(m.group(2)), int(m.group(4)))
    m = re.search(r"(莊|B)\s*:?\s*([0-9]).*?(閒|P)\s*:?\s*([0-9])", u)
    if m: return (int(m.group(4)), int(m.group(2)))

    # B6P5 / P6B5（容許符號）
    m = re.search(r"B\D*([0-9])\D*P\D*([0-9])", u)
    if m: return (int(m.group(2)), int(m.group(1)))
    m = re.search(r"P\D*([0-9])\D*B\D*([0-9])", u)
    if m: return (int(m.group(1)), int(m.group(2)))

    # 保底：最後兩個數字
    digits = re.findall(r"[0-9]", u)
    if len(digits) >= 2:
        return (int(digits[-2]), int(digits[-1]))
    return None

# ---------- PF 匯入（本地 pfilter 或套件） ----------
try:
    from bgs.pfilter import OutcomePF  # type: ignore
except Exception:
    try:
        cur = os.path.dirname(os.path.abspath(__file__))
        if cur not in sys.path: sys.path.insert(0, cur)
        from pfilter import OutcomePF  # type: ignore
        log.info("OutcomePF from local pfilter.py")
    except Exception as e:
        OutcomePF = None # type: ignore
        log.error("OutcomePF import failed: %s", e)

# ---------- Render 安全預設 ----------
os.environ.setdefault("PF_BACKEND", "mc")
os.environ.setdefault("DECKS", "6")
os.environ.setdefault("PF_N", "60")
os.environ.setdefault("PF_UPD_SIMS", "30")
os.environ.setdefault("PF_PRED_SIMS", "20")
os.environ.setdefault("PF_RESAMPLE", "0.7")
os.environ.setdefault("PF_DIR_EPS", "0.003")

# ---------- 初始化 PF ----------
if OutcomePF:
    try:
        PF = OutcomePF(
            decks=int(os.getenv("DECKS","6")),
            seed=int(os.getenv("SEED","42")),
            n_particles=int(os.getenv("PF_N","60")),
            sims_lik=max(1,int(os.getenv("PF_UPD_SIMS","30"))),
            resample_thr=float(os.getenv("PF_RESAMPLE","0.7")),
            backend=os.getenv("PF_BACKEND","mc"),
            dirichlet_eps=float(os.getenv("PF_DIR_EPS","0.003")),
        )
        log.info("PF init ok: n=%s backend=%s", getattr(PF,"n_particles","?"), getattr(PF,"backend","?"))
    except Exception as e:
        log.error("PF init fail: %s", e)
        class _Dummy:  # minimal fallback
            def update_outcome(self, outcome): pass
            def predict(self, **k): return np.array([0.48,0.47,0.05], dtype=np.float32)
        PF = _Dummy()
else:
    class _Dummy:
        def update_outcome(self, outcome): pass
        def predict(self, **k): return np.array([0.48,0.47,0.05], dtype=np.float32)
    PF = _Dummy()

# ---------- 決策/配注 ----------
EDGE_ENTER = float(os.getenv("EDGE_ENTER","0.03"))
MIN_BET_PCT = float(os.getenv("MIN_BET_PCT","0.05"))
MAX_BET_PCT = float(os.getenv("MAX_BET_PCT","0.40"))
PROB_SMA_ALPHA = float(os.getenv("PROB_SMA_ALPHA","0.45"))
PROB_TEMP = float(os.getenv("PROB_TEMP","1.0"))

# Thompson Scaling（只縮放配注，不改選邊）
TS_EN = env_flag("TS_EN", 0)
TS_ALPHA = float(os.getenv("TS_ALPHA","2"))
TS_BETA  = float(os.getenv("TS_BETA","2"))

# 不確定性懲罰
UNCERT_PENALTY_EN = env_flag("UNCERT_PENALTY_EN", 1)
UNCERT_MARGIN_MAX = int(os.getenv("UNCERT_MARGIN_MAX","1"))
UNCERT_RATIO = float(os.getenv("UNCERT_RATIO","0.33"))

# 連續點差加權（含上一手機率差）
W_BASE = float(os.getenv("W_BASE","1.0"))
W_MIN  = float(os.getenv("W_MIN","0.5"))
W_MAX  = float(os.getenv("W_MAX","2.8"))
W_ALPHA= float(os.getenv("W_ALPHA","0.95"))
W_SIG_K= float(os.getenv("W_SIG_K","1.10"))
W_SIG_MID=float(os.getenv("W_SIG_MID","1.8"))
W_GAMMA= float(os.getenv("W_GAMMA","1.0"))
W_GAP_CAP=float(os.getenv("W_GAP_CAP","0.06"))

# 手數深度權重
DEPTH_W_EN  = env_flag("DEPTH_W_EN", 1)
DEPTH_W_MAX = float(os.getenv("DEPTH_W_MAX","1.3"))

# 點差可靠度表（5桶）
MREL_EN = env_flag("MREL_EN", 1)
MREL_LR = float(os.getenv("MREL_LR","0.02"))

INV = {0:"莊", 1:"閒"}

# ---------- 平滑工具 ----------
def softmax_temp(p: np.ndarray, t: float) -> np.ndarray:
    t = max(1e-6, float(t))
    x = np.log(np.clip(p,1e-9,1.0)) / t
    x = np.exp(x - np.max(x))
    x = x / np.sum(x)
    return x

def ema(prev: Optional[np.ndarray], cur: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None: return cur
    return alpha*cur + (1-alpha)*prev

# ---------- 連續權重：點差 + 上一手機率差 ----------
def calc_margin_weight(p_pts: int, b_pts: int, last_prob_gap: float) -> float:
    margin = abs(int(p_pts) - int(b_pts))
    sig = 1.0/(1.0 + math.exp(-W_SIG_K * (margin - W_SIG_MID)))
    part_m = W_ALPHA * sig
    gap_norm = min(max(float(last_prob_gap),0.0), W_GAP_CAP) / max(W_GAP_CAP,1e-6)
    part_g = W_GAMMA * gap_norm
    w = W_BASE + part_m + part_g
    return max(W_MIN, min(W_MAX, w))

# ---------- 可靠度表（5桶：0,1,2,3,4+） ----------
def margin_bucket(margin: int) -> int:
    return 4 if margin>=4 else margin

def mrel_score(sess: Dict[str,Any], margin: int) -> float:
    if not MREL_EN: return 1.0
    b = margin_bucket(margin)
    a = sess["mrel"]["a"][b]; bb = sess["mrel"]["b"][b]
    return (a)/(a+bb)  # Beta 均值

def mrel_update(sess: Dict[str,Any], margin: int, correct: bool):
    if not MREL_EN: return
    b = margin_bucket(margin)
    if correct:
        sess["mrel"]["a"][b] = max(1.0, sess["mrel"]["a"][b] + MREL_LR)
    else:
        sess["mrel"]["b"][b] = max(1.0, sess["mrel"]["b"][b] + MREL_LR)

# ---------- EV 決策（只莊/閒；含 5% 抽水） ----------
def decide_bp(prob: np.ndarray) -> Tuple[str, float, float]:
    pB, pP = float(prob[0]), float(prob[1])
    evB, evP = 0.95*pB - pP, pP - pB
    side = 0 if evB>evP else 1
    edge = max(abs(evB), abs(evP))
    return (INV[side], edge, max(pB,pP))

def bet_amount(bankroll: int, pct: float) -> int:
    if bankroll<=0 or pct<=0: return 0
    return int(round(bankroll * pct))

def confidence_to_pct(edge: float, max_prob: float) -> float:
    base_conf = min(1.0, edge*15.0)
    prob_conf = max(0.0, (max_prob-0.45)*2.5)
    total = 0.5*base_conf + 0.5*prob_conf
    pct = MIN_BET_PCT + (total**0.8) * (MAX_BET_PCT-MIN_BET_PCT)
    return max(MIN_BET_PCT, min(MAX_BET_PCT, pct))

def thompson_scale(pct: float) -> float:
    if not TS_EN: return pct
    a = max(1e-3, TS_ALPHA); b = max(1e-3, TS_BETA)
    s = np.random.beta(a, b)  # 只縮放配注，不改方向
    return max(MIN_BET_PCT, min(MAX_BET_PCT, pct*s))

# ---------- 主流程：收到上一局點數，更新並預測 ----------
_prev_prob_sma: Optional[np.ndarray] = None

def handle_points_and_predict(sess: Dict[str,Any], p_pts: int, b_pts: int) -> str:
    global _prev_prob_sma

    # 手數+1（每桌綁定）
    sess["hand_idx"] = int(sess.get("hand_idx", 0)) + 1
    margin = abs(p_pts - b_pts)

    # 1) 更新 PF（連續權重 + 不確定懲罰 + 深度權重）
    last_gap = float(sess.get("last_prob_gap", 0.0))
    w = calc_margin_weight(p_pts, b_pts, last_gap)

    if DEPTH_W_EN and sess["hand_idx"]>0:
        depth_boost = 1.0 + min(sess["hand_idx"]/70.0, (DEPTH_W_MAX-1.0))
        w *= depth_boost

    rep = max(1, min(3, int(round(w))))

    if p_pts == b_pts:
        try:
            PF.update_outcome(2)  # tie
        except Exception as e:
            log.warning("PF tie update err: %s", e)
    else:
        outcome = 1 if p_pts > b_pts else 0
        for _ in range(rep):
            try:
                PF.update_outcome(outcome)
            except Exception as e:
                log.warning("PF update err: %s", e)

        if UNCERT_PENALTY_EN and margin <= UNCERT_MARGIN_MAX:
            rev = 0 if outcome==1 else 1
            if random.random() < UNCERT_RATIO:
                try:
                    PF.update_outcome(rev)
                except Exception as e:
                    log.warning("PF uncert reverse update err: %s", e)

    # 2) 預測
    sims_pred = max(0, int(os.getenv("PF_PRED_SIMS","20")))
    p_raw = PF.predict(sims_per_particle=sims_pred)

    # 可靠度表加權（基於 margin 桶）
    rel = mrel_score(sess, margin)
    p_adj = np.array([p_raw[0]*rel, p_raw[1]*rel, p_raw[2]], dtype=np.float32)
    p_adj = p_adj / np.sum(p_adj)

    # 溫度 & 平滑
    p_temp = softmax_temp(p_adj, PROB_TEMP)
    _prev_prob_sma = ema(_prev_prob_sma, p_temp, PROB_SMA_ALPHA)
    p_final = _prev_prob_sma if _prev_prob_sma is not None else p_temp

    # 記錄給下一手用
    sess["last_prob_gap"] = abs(float(p_final[0]) - float(p_final[1]))

    # 3) 決策與配注
    choice, edge, maxp = decide_bp(p_final)
    bankroll = int(sess.get("bankroll", 0))
    bet_pct = confidence_to_pct(edge, maxp)
    bet_pct = thompson_scale(bet_pct)
    bet_amt = bet_amount(bankroll, bet_pct)

    # 4) 更新可靠度表（用上一手預測 vs 當前 outcome）
    if p_pts != b_pts and _prev_prob_sma is not None:
        prev_choice = 1 if _prev_prob_sma[1] >= _prev_prob_sma[0] else 0
        correct = (prev_choice == (1 if p_pts>b_pts else 0))
        mrel_update(sess, margin, correct)

    # 5) 輸出
    if p_pts == b_pts:
        sess["last_pts_text"] = f"上局結果: 和局 (閒{p_pts} 莊{b_pts})"
    else:
        sess["last_pts_text"] = f"上局結果: 閒{p_pts} 莊{b_pts}"

    msg = [
        sess["last_pts_text"],
        "開始分析下局....",
        "【預測結果】",
        f"閒：{p_final[1]*100:.2f}%",
        f"莊：{p_final[0]*100:.2f}%",
        f"本次預測結果：{choice}",
        f"建議下注：{bet_amt:,}",
        f"(edge={edge*100:.1f}%, maxp={maxp*100:.1f}%, rep={rep}, rel={rel:.2f})",
    ]
    return "\n".join(msg)

# ---------- REST 路由 ----------
@app.get("/")
def root():
    return f"✅ BGS PF Server OK ({VERSION})", 200

@app.get("/health")
def health():
    return jsonify(ok=True, ts=time.time(), version=VERSION), 200

@app.post("/predict")
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    uid = str(data.get("uid","guest"))
    text = str(data.get("text","")).strip()
    sess = now_sess(uid)
    if "bankroll" in data:
        try:
            bk = int(data["bankroll"])
            if bk>0: sess["bankroll"] = bk
        except: pass
    pts = parse_last_hand_points(text)
    if pts is None:
        return jsonify(ok=False, err="無法解析點數（例：閒6莊5 / 65 / 和）"), 400
    msg = handle_points_and_predict(sess, pts[0], pts[1])
    save_sess(uid, sess)
    return jsonify(ok=True, msg=msg), 200

# ---------- LINE webhook（自動降級 + 指令） ----------
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
try:
    from linebot import LineBotApi, WebhookHandler
    from linebot.exceptions import InvalidSignatureError
    from linebot.models import MessageEvent, TextMessage, TextSendMessage
    _has_line = True
except Exception:
    _has_line = False

if _has_line and LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN:
    line_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
    line_handler = WebhookHandler(LINE_CHANNEL_SECRET)

    @app.post("/line-webhook")
    def line_webhook():
        signature = request.headers.get("X-Line-Signature", "")
        body = request.get_data(as_text=True)
        try:
            line_handler.handle(body, signature)
        except InvalidSignatureError:
            return "invalid signature", 400
        except Exception as e:
            log.warning("line webhook err: %s", e)
            return "ok", 200
        return "ok", 200

    @line_handler.add(MessageEvent, message=TextMessage)
    def on_text(event):
        uid = getattr(event.source, "user_id", "guest")
        raw = (event.message.text or "")
        norm = _norm(raw)
        sess = now_sess(uid)

        log.info("[LINE] uid=%s text=%r norm=%r phase=%s", uid, raw, norm, sess.get("phase"))

        # 指令：遊戲設定
        if norm in ("遊戲設定", "設定", "SETUP", "GAME"):
            sess["phase"] = "choose_game"
            save_sess(uid, sess)
            try:
                line_api.reply_message(
                    event.reply_token,
                    TextSendMessage(
                        text="【請選擇遊戲館別】\n1.WM 2.PM 3.DG 4.SA 5.KU 6.歐博/卡利 7.KG 8.全利 9.名人 10.MT真人\n（直接輸入數字）\n或先輸入本金（例如：5000）",
                        quick_reply=_quick_reply(),
                    ),
                )
            except Exception as e:
                log.warning("reply err: %s", e)
            return

        # 指令：結束分析 / 清空 / RESET
        if norm in ("結束分析", "清空", "RESET"):
            keep_premium = bool(sess.get("premium", True))
            keep_trial   = int(sess.get("trial_start", int(time.time())))
            SESS.pop(uid, None)
            sess = now_sess(uid)
            sess["premium"] = keep_premium
            sess["trial_start"] = keep_trial
            save_sess(uid, sess)
            try:
                line_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text="🧹 已清空。輸入『遊戲設定』或直接輸入本金（例：5000）。", quick_reply=_quick_reply()),
                )
            except Exception as e:
                log.warning("reply err: %s", e)
            return

        # 本金設定（純數字）
        if norm.isdigit():
            try:
                bk = int(norm)
                if bk>0:
                    sess["bankroll"] = bk
                    save_sess(uid, sess)
                    line_api.reply_message(
                        event.reply_token,
                        TextSendMessage(
                            text=f"👍 已設定本金：{bk:,}\n請直接輸入上一局點數（例：65 / 和 / 閒6莊5）。",
                            quick_reply=_quick_reply(),
                        ),
                    )
                    return
            except Exception:
                pass

        # 點數解析
        pts = parse_last_hand_points(raw)
        if pts is None:
            try:
                line_api.reply_message(
                    event.reply_token,
                    TextSendMessage(
                        text="指令無法辨識。\n可輸入：\n・本金（例：5000）\n・上一局點數（例：65 / 和 / 閒6莊5）\n・『遊戲設定』或『結束分析』",
                        quick_reply=_quick_reply(),
                    ),
                )
            except Exception as e:
                log.warning("reply err: %s", e)
            return

        # 未設定本金先提醒
        if not int(sess.get("bankroll", 0)):
            try:
                line_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text="請先輸入本金（例：5000），再回報點數。", quick_reply=_quick_reply()),
                )
            except Exception as e:
                log.warning("reply err: %s", e)
            return

        # 正式預測
        msg = handle_points_and_predict(sess, int(pts[0]), int(pts[1]))
        save_sess(uid, sess)
        try:
            line_api.reply_message(event.reply_token, TextSendMessage(text=msg, quick_reply=_quick_reply()))
        except Exception as e:
            log.warning("reply err: %s", e)
else:
    # 無 SDK 或無憑證 → 安全退回僅回 200 的最小端點（避免 404）
    @app.post("/line-webhook")
    def line_webhook_min():
        return "ok", 200

# ---------- Main ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT","8000"))
    log.info("Starting %s on port %s", VERSION, port)
    app.run(host="0.0.0.0", port=port, debug=False)
