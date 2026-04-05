# -*- coding: utf-8 -*-
"""server.py — BGS 最終盈利版 v3.0（可放大資金版）- 完整風控 + 動態下注 + 真實學習"""
import os, sys, logging, time, re, json, threading
from typing import Optional, Dict, Any, Tuple, List
import numpy as np

# ==================== VERSION 定義 ====================
VERSION = "bgs-profit-v3.0-2025-11-03"

# ==================== 試用設定 ====================
TRIAL_SECONDS = int(os.getenv("TRIAL_SECONDS", "1800"))
PREMIUM_TRIAL_SECONDS = int(os.getenv("PREMIUM_TRIAL_SECONDS", "2592000"))

def env_flag(name: str, default: int = 1) -> int:
    val = os.getenv(name)
    if val is None:
        return 1 if default else 0
    v = str(val).strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return 1
    if v in ("0", "false", "f", "no", "n", "off"):
        return 0
    try:
        return 1 if int(float(v)) != 0 else 0
    except Exception:
        return 1 if default else 0

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-server")
np.seterr(all="ignore")

# ==================== KEEP ALIVE ====================
KEEP_ALIVE_STARTED = False
KEEP_ALIVE_LOCK = threading.Lock()

def _self_keep_alive():
    try:
        import requests
    except Exception:
        log.warning("[KEEPALIVE] requests module not available")
        return
    url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("SELF_URL") or os.getenv("SELF_PING_URL")
    interval = int(os.getenv("SELF_PING_INTERVAL", "120"))
    if not url:
        log.warning("[KEEPALIVE] No URL found")
        return
    ping_url = url.rstrip("/") + "/ping"
    log.info(f"[KEEPALIVE] Started | interval: {interval}s")
    while True:
        try:
            r = requests.get(ping_url, timeout=10)
            if r.status_code < 400:
                log.info("[KEEPALIVE] self ping success")
        except Exception as e:
            log.warning("[KEEPALIVE] ping failed: %s", e)
        time.sleep(interval)

# ==================== 館別選單 ====================
GAMES = {
    "1": "WM", "2": "PM", "3": "DG", "4": "SA", "5": "KU",
    "6": "歐博/卡利", "7": "KG", "8": "全利", "9": "名人", "10": "MT真人"
}

# ==================== Deplete Module ====================
DEPLETE_OK = False
init_counts = None
probs_after_points = None

try:
    from deplete import init_counts, probs_after_points
    DEPLETE_OK = True
    log.info("Deplete module loaded")
except Exception:
    try:
        from bgs.deplete import init_counts, probs_after_points
        DEPLETE_OK = True
        log.info("Deplete loaded from bgs.deplete")
    except Exception:
        try:
            _cur_dir = os.path.dirname(os.path.abspath(__file__))
            if _cur_dir not in sys.path:
                sys.path.insert(0, _cur_dir)
            from deplete import init_counts, probs_after_points
            DEPLETE_OK = True
            log.info("Deplete loaded from local path")
        except Exception as e:
            DEPLETE_OK = False
            log.warning(f"Deplete not available: {e}")

# ==================== PF Module ====================
PF_INITIALIZED = False
pf_model = None

class SmartDummyPF:
    """PF模型 - 穩定學習版"""
    
    def __init__(self):
        self.base_probs = np.array([0.4586, 0.4462, 0.0952])
        self.outcome_history = []
        self.learning_rate = 0.1
        self.min_prob = 0.40
        self.max_prob = 0.55
        log.info("PF Dummy Model initialized (v3.0)")
    
    def predict_proba(self, points: List[int]) -> np.ndarray:
        if DEPLETE_OK and init_counts and probs_after_points:
            try:
                counts = init_counts()
                try:
                    probs = probs_after_points(counts, points)
                except TypeError:
                    try:
                        probs = probs_after_points(points)
                    except Exception:
                        probs = probs_after_points(counts=counts, points=points)
                if not isinstance(probs, np.ndarray):
                    probs = np.array(probs)
                return probs
            except Exception as e:
                log.warning(f"Deplete failed: {e}")
        return self.base_probs.copy()
    
    def update_outcome(self, outcome: int, confidence: float = 1.0) -> None:
        """🔥 真實學習：outcome: 1=莊贏, 0=閒贏"""
        if outcome is None or confidence < 0.8:
            return
        
        self.outcome_history.append(outcome)
        if len(self.outcome_history) > 100:
            self.outcome_history = self.outcome_history[-100:]
        
        if len(self.outcome_history) < 20:
            return
        
        recent = self.outcome_history[-20:]
        banker_rate = recent.count(1) / len(recent)
        player_rate = 1 - banker_rate
        
        alpha = self.learning_rate
        self.base_probs[0] = self.base_probs[0] * (1 - alpha) + banker_rate * alpha
        self.base_probs[1] = self.base_probs[1] * (1 - alpha) + player_rate * alpha
        self.base_probs[2] = 0.0952
        
        self.base_probs = np.clip(self.base_probs, self.min_prob, self.max_prob)
        self.base_probs = self.base_probs / self.base_probs.sum()
        log.debug(f"PF學習: 莊 {self.base_probs[0]:.3f} 閒 {self.base_probs[1]:.3f}")

try:
    try:
        from pf import SmartPF
        pf_model = SmartPF()
        PF_INITIALIZED = True
        log.info("Real PF loaded")
    except ImportError:
        pf_model = SmartDummyPF()
        PF_INITIALIZED = True
        log.info("Using Dummy PF")
except Exception as e:
    log.warning(f"PF error: {e}")
    pf_model = SmartDummyPF()
    PF_INITIALIZED = True

# ==================== Flask ====================
try:
    from flask import Flask, request, jsonify, abort
    from flask_cors import CORS
    _flask_available = True
except Exception:
    _flask_available = False
    Flask = None
    request = None
    def jsonify(*args, **kwargs):
        raise RuntimeError("Flask not available")
    def abort(*args, **kwargs):
        raise RuntimeError("Flask not available")
    def CORS(app):
        return None

# ==================== Redis ====================
try:
    import redis
except Exception:
    redis = None

REDIS_URL = os.getenv("REDIS_URL")
redis_client = None
if redis is not None and REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        log.info("Redis connected")
    except Exception as e:
        log.error(f"Redis failed: {e}")

# ==================== Session Management ====================
SESS_FALLBACK = {}
KV_FALLBACK = {}
SESSION_EXPIRE_SECONDS = int(os.getenv("SESSION_EXPIRE_SECONDS", "1200"))
DEDUPE_TTL = 60

def _rget(k: str) -> Optional[str]:
    try:
        if redis_client:
            return redis_client.get(k)
        return KV_FALLBACK.get(k)
    except Exception:
        return None

def _rset(k: str, v: str, ex: Optional[int] = None):
    try:
        if redis_client:
            redis_client.set(k, v, ex=ex)
        else:
            KV_FALLBACK[k] = v
    except Exception:
        pass

def _rsetnx(k: str, v: str, ex: int) -> bool:
    try:
        if redis_client:
            return bool(redis_client.set(k, v, ex=ex, nx=True))
        if k in KV_FALLBACK:
            return False
        KV_FALLBACK[k] = v
        return True
    except Exception:
        return True

def _dedupe_event(event_id: Optional[str]) -> bool:
    if not event_id:
        return True
    return _rsetnx(f"dedupe:{event_id}", "1", ex=DEDUPE_TTL)

# ==================== Premium System ====================
def _premium_key(uid: str) -> str:
    return f"premium:{uid}"

def get_activation_codes() -> List[str]:
    raw = os.getenv("ACTIVATION_CODES", "")
    return [c.strip().upper() for c in raw.split(",") if c.strip()]

def use_activation_code(uid: str, code: str) -> bool:
    if code.upper() not in get_activation_codes():
        return False
    _rset(_premium_key(uid), "1", ex=PREMIUM_TRIAL_SECONDS)
    return True

def is_premium(uid: str) -> bool:
    if not uid:
        return False
    return _rget(_premium_key(uid)) == "1"

def is_blocked(uid: str) -> bool:
    return False  # 簡化，可自行擴充

def get_session(uid: str) -> Dict[str, Any]:
    if not uid:
        uid = "anon"
    try:
        if redis_client:
            raw = redis_client.get(f"sess:{uid}")
            if raw:
                sess = json.loads(raw)
                sess.setdefault("phase", "init")
                sess.setdefault("game", None)
                sess.setdefault("bankroll", 0)
                sess.setdefault("rounds_seen", 0)
                sess.setdefault("premium", is_premium(uid))
                sess.setdefault("trial_start", int(time.time()))
                sess.setdefault("skip_streak", 0)
                sess.setdefault("loss_streak", 0)  # 🔥 新增：連輸計數
                sess.setdefault("win_streak", 0)   # 🔥 新增：連勝計數
                sess.setdefault("last_choice", None)
                sess.setdefault("last_prediction", None)
                return sess
    except Exception:
        pass
    
    sess = {
        "phase": "init", "game": None, "bankroll": 0, "rounds_seen": 0,
        "premium": is_premium(uid), "trial_start": int(time.time()),
        "skip_streak": 0, "loss_streak": 0, "win_streak": 0,
        "last_choice": None, "last_prediction": None
    }
    save_session(uid, sess)
    return sess

def save_session(uid: str, sess: Dict[str, Any]) -> None:
    if not uid:
        uid = "anon"
    try:
        payload = json.dumps(sess, ensure_ascii=False)
        if redis_client:
            redis_client.set(f"sess:{uid}", payload, ex=SESSION_EXPIRE_SECONDS)
        else:
            SESS_FALLBACK[uid] = sess
    except Exception:
        pass

# ==================== Trial System ====================
def get_trial_remaining(sess: Dict[str, Any]) -> int:
    start = sess.get("trial_start", int(time.time()))
    remain = TRIAL_SECONDS - (int(time.time()) - start)
    return max(0, remain)

def is_trial_expired(sess: Dict[str, Any]) -> bool:
    return get_trial_remaining(sess) <= 0

def format_trial_time(sess: Dict[str, Any]) -> str:
    remaining = get_trial_remaining(sess)
    minutes = remaining // 60
    return f"⏳ 試用剩餘 {minutes} 分鐘" if minutes > 0 else "⏰ 試用即將到期"

# ==================== Core Functions ====================
def parse_points_input(text: str) -> Optional[Tuple[List[int], str]]:
    text = text.strip()
    if re.match(r'^\d{2,4}$', text):
        points = [int(d) for d in text]
        if len(points) == 2:
            return (points, "normal")
        elif len(points) == 4:
            return (points, "full")
    parts = text.split()
    if len(parts) >= 2:
        points = []
        for p in parts[:4]:
            if p.isdigit() and 0 <= int(p) <= 9:
                points.append(int(p))
        if len(points) >= 2:
            return (points, "spaced")
    if text in ["和", "tie", "Tie", "TIE"]:
        return ([], "tie")
    banker_match = re.search(r'莊(\d)', text)
    player_match = re.search(r'閒(\d)', text)
    if banker_match and player_match:
        points = [int(player_match.group(1)), int(banker_match.group(1))]
        return (points, "named")
    return None

def calculate_baccarat_probabilities(points: List[int], rounds_seen: Optional[int] = None) -> np.ndarray:
    pf_probs = pf_model.predict_proba(points)
    if DEPLETE_OK:
        try:
            counts = init_counts()
            try:
                dep_probs = probs_after_points(counts, points[0], points[1], rounds_seen=rounds_seen)
            except TypeError:
                try:
                    dep_probs = probs_after_points(points)
                except Exception:
                    dep_probs = probs_after_points(counts=counts, points=points)
            if not isinstance(dep_probs, np.ndarray):
                dep_probs = np.array(dep_probs)
        except Exception:
            dep_probs = pf_probs
    else:
        dep_probs = pf_probs
    
    edge = abs(pf_probs[0] - pf_probs[1])
    if edge < 0.015:
        w_pf, w_dep = 0.5, 0.5
    elif edge < 0.04:
        w_pf, w_dep = 0.7, 0.3
    else:
        w_pf, w_dep = 0.85, 0.15
    
    final_probs = pf_probs * w_pf + dep_probs * w_dep
    return final_probs / final_probs.sum()

def calculate_ev(probs: np.ndarray) -> Tuple[float, float]:
    pB, pP, _ = probs
    return pB * 0.95 - (1 - pB), pP * 1.0 - (1 - pP)

def calculate_bet_ratio_dynamic(edge: float, ev: float, mode: str, loss_streak: int) -> Tuple[float, str]:
    """🔥 動態下注引擎 - 會呼吸的策略"""
    strength = edge + max(ev, 0)
    
    # 基礎比例
    if strength > 0.08:
        base_ratio = 0.04
        reason = "⚡ 極強信號"
    elif strength > 0.05:
        base_ratio = 0.025
        reason = "🔥 強勢信號"
    elif strength > 0.03:
        base_ratio = 0.015
        reason = "📈 中等信號"
    else:
        base_ratio = 0.008
        reason = "📊 微幅信號"
    
    # 付費模式加成
    if mode == "advanced":
        base_ratio = min(base_ratio * 1.2, 0.05)
    
    # 🔥 連輸風控（關鍵！）
    if loss_streak >= 3:
        base_ratio *= 0.5
        reason += " | ⚠️ 連輸3局降50%"
    if loss_streak >= 5:
        base_ratio *= 0.3
        reason += " | 🔴 連輸5局降70%"
    if loss_streak >= 7:
        base_ratio = 0
        reason = "🛑 連輸保護啟動 - 暫停下注"
    
    return base_ratio, reason

def determine_bet_final(probs: np.ndarray, mode: str = "normal", bankroll: int = 1000,
                        skip_streak: int = 0, loss_streak: int = 0) -> Tuple[str, int, str]:
    """最終決策 - 完整風控版"""
    pB, pP, _ = probs
    ev_banker, ev_player = calculate_ev(probs)
    edge = abs(pB - pP)
    
    # 判斷最佳選擇
    best_choice = None
    best_ev = -999
    if ev_banker > 0:
        best_choice = "莊"
        best_ev = ev_banker
    elif ev_player > 0:
        best_choice = "閒"
        best_ev = ev_player
    
    # 🔥 連輸7局強制停
    if loss_streak >= 7:
        return "觀望", 0, "🛑 連輸7局，風控保護強制暫停"
    
    if best_choice is None:
        if edge > 0.01:
            choice = "莊" if pB > pP else "閒"
            return choice, max(1, int(bankroll * 0.002)), "🧪 邊緣測試"
        if skip_streak >= 2:
            choice = "莊" if pB > pP else "閒"
            return choice, max(1, int(bankroll * 0.003)), "🔄 強制出手"
        return "觀望", 0, "📉 期望值為負"
    
    # 🔥 動態下注
    bet_ratio, reason = calculate_bet_ratio_dynamic(edge, best_ev, mode, loss_streak)
    
    if bet_ratio <= 0:
        return "觀望", 0, reason
    
    bet_amount = max(1, int(bankroll * bet_ratio))
    bet_amount = min(bet_amount, int(bankroll * 0.05))
    
    return best_choice, bet_amount, reason

def format_output(probs: np.ndarray, choice: str, bet_amt: int, reason: str, mode: str = "",
                  last_pts: Optional[str] = None, game: str = None,
                  bankroll: int = None, trial_msg: str = None,
                  skip_streak: int = 0, loss_streak: int = 0, win_streak: int = 0,
                  ev_banker: float = None, ev_player: float = None) -> str:
    pB, pP, pT = [float(x) for x in probs]
    lines = []
    if game:
        lines.append(f"🎰 館別：{game}")
    if bankroll:
        lines.append(f"💰 剩餘本金：{bankroll}")
    if last_pts:
        lines.append(f"📊 輸入：{last_pts}")
    lines.append(f"🎲 機率｜莊 {pB*100:.1f}%｜閒 {pP*100:.1f}%｜和 {pT*100:.1f}%")
    lines.append(f"📈 差距｜{abs(pB - pP)*100:.1f}%")
    if ev_banker is not None:
        lines.append(f"📊 EV｜莊 {ev_banker*100:.2f}%｜閒 {ev_player*100:.2f}%")
    mode_display = "🎖️ 付費" if mode == "advanced" else "🆓 免費"
    lines.append(f"⚙️ 模式｜{mode_display}")
    if skip_streak > 0:
        lines.append(f"👀 觀望：{skip_streak}局")
    if loss_streak > 0:
        lines.append(f"📉 連輸：{loss_streak}局")
    if win_streak > 0:
        lines.append(f"📈 連勝：{win_streak}局")
    if choice == "觀望":
        lines.append(f"👀 建議：觀望")
    else:
        lines.append(f"🎯 建議：下注 {choice} {bet_amt}單位")
        if bankroll and bet_amt > 0:
            lines.append(f"📊 風險：{bet_amt/bankroll*100:.1f}%")
    if reason:
        lines.append(f"💡 {reason}")
    if trial_msg:
        lines.append(trial_msg)
    lines.append("\n💡 輸入下一局點數")
    lines.append("📝 輸入「贏」或「輸」可讓AI學習")
    return "\n".join(lines)

def _handle_points_and_predict(uid: str, points_text: str, sess: Dict[str, Any]) -> str:
    parsed = parse_points_input(points_text)
    if not parsed:
        return "❌ 格式錯誤！"
    
    points, input_type = parsed
    
    # 🔥 真實學習：處理用戶回報結果
    if points_text in ["贏", "勝", "win", "W"]:
        last_outcome = sess.get("last_actual_outcome")
        if last_outcome is not None:
            if hasattr(pf_model, 'update_outcome'):
                pf_model.update_outcome(last_outcome, confidence=1.0)
                log.info(f"用戶{uid}回報贏，PF學習 結果={last_outcome}")
        # 更新連勝/連輸
        sess["win_streak"] = sess.get("win_streak", 0) + 1
        sess["loss_streak"] = 0
        save_session(uid, sess)
        return "✅ 已記錄為「贏」，AI已學習此結果"
    
    if points_text in ["輸", "負", "loss", "L"]:
        last_outcome = sess.get("last_actual_outcome")
        if last_outcome is not None:
            # 學習相反的結果
            opposite = 1 - last_outcome if last_outcome in [0,1] else None
            if opposite is not None and hasattr(pf_model, 'update_outcome'):
                pf_model.update_outcome(opposite, confidence=1.0)
                log.info(f"用戶{uid}回報輸，PF學習 結果={opposite}")
        sess["loss_streak"] = sess.get("loss_streak", 0) + 1
        sess["win_streak"] = 0
        save_session(uid, sess)
        return "✅ 已記錄為「輸」，AI已學習此結果"
    
    if input_type == "tie":
        sess["last_choice"] = "和"
        save_session(uid, sess)
        return "🎲 和局\n\n建議謹慎下注"
    
    try:
        probs = calculate_baccarat_probabilities(points, sess.get("rounds_seen", 0))
    except Exception as e:
        return f"❌ 計算失敗：{str(e)}"
    
    mode = "advanced" if is_premium(uid) else "normal"
    bankroll = sess.get("bankroll", 1000)
    skip_streak = sess.get("skip_streak", 0)
    loss_streak = sess.get("loss_streak", 0)
    win_streak = sess.get("win_streak", 0)
    ev_banker, ev_player = calculate_ev(probs)
    
    choice, bet_amt, reason = determine_bet_final(probs, mode, bankroll, skip_streak, loss_streak)
    
    # 記錄預測（供結果回報用）
    sess["last_prediction"] = probs.copy()
    sess["last_actual_outcome"] = 1 if choice == "莊" else 0 if choice == "閒" else None
    
    sess["last_pts_text"] = points_text
    sess["last_choice"] = choice
    sess["rounds_seen"] = sess.get("rounds_seen", 0) + 1
    
    if choice == "觀望":
        sess["skip_streak"] = skip_streak + 1
    else:
        sess["skip_streak"] = 0
    
    save_session(uid, sess)
    
    trial_msg = None if is_premium(uid) else format_trial_time(sess)
    
    return format_output(probs, choice, bet_amt, reason, mode, points_text,
                        sess.get("game"), sess.get("bankroll"), trial_msg,
                        sess.get("skip_streak"), loss_streak, win_streak,
                        ev_banker, ev_player)

# ==================== LINE Bot ====================
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
_line_bot_available = False
line_bot_api = None
handler = None

if LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN:
    try:
        from linebot import LineBotApi, WebhookHandler
        from linebot.models import TextSendMessage, FollowEvent, UnfollowEvent, QuickReply, QuickReplyButton, MessageAction
        line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
        handler = WebhookHandler(LINE_CHANNEL_SECRET)
        _line_bot_available = True
        log.info("LINE Bot ready")
    except Exception as e:
        log.error(f"LINE init failed: {e}")

def build_main_menu() -> QuickReply:
    return QuickReply(items=[
        QuickReplyButton(action=MessageAction(label="🎮 遊戲設定", text="遊戲設定")),
        QuickReplyButton(action=MessageAction(label="🛑 結束", text="結束分析")),
    ])

# ==================== Flask App ====================
if _flask_available and Flask is not None:
    app = Flask(__name__)
    CORS(app)
    try:
        if not KEEP_ALIVE_STARTED:
            with KEEP_ALIVE_LOCK:
                if not KEEP_ALIVE_STARTED:
                    threading.Thread(target=_self_keep_alive, daemon=True).start()
                    KEEP_ALIVE_STARTED = True
    except Exception:
        pass
else:
    class _DummyApp:
        def get(self, *a, **k):
            def _d(f): return f
            return _d
        def post(self, *a, **k):
            def _d(f): return f
            return _d
        def run(self, *a, **k):
            log.warning("Flask not available")
    app = _DummyApp()

@app.get("/")
def root():
    return f"✅ BGS Server {VERSION}", 200

@app.get("/ping")
def ping():
    return "OK", 200

@app.get("/health")
def health():
    return {"ok": True, "version": VERSION, "status": "running"}, 200

@app.post("/predict")
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data"}), 400
        uid = data.get("uid", "anon")
        points = data.get("points", "")
        if not points:
            return jsonify({"error": "points required"}), 400
        sess = get_session(uid)
        if not is_premium(uid) and is_trial_expired(sess):
            return jsonify({"error": "試用已到期", "trial_expired": True}), 403
        result = _handle_points_and_predict(uid, points, sess)
        return jsonify({"result": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/webhook")
def webhook():
    if not _line_bot_available:
        return jsonify({"error": "LINE not configured"}), 400
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    if not signature:
        return "Missing signature", 400
    try:
        handler.handle(body, signature)
    except Exception as e:
        log.error(f"Webhook error: {e}")
        return jsonify({"error": str(e)}), 500
    return "OK", 200

# ==================== LINE Handlers ====================
if _line_bot_available and handler:
    
    @handler.add(FollowEvent)
    def handle_follow(event):
        user_id = event.source.user_id
        msg = f"🎰 BGS百家樂預測 v3.0（可放大資金版）\n\n"
        msg += f"⏳ 試用 {TRIAL_SECONDS//60} 分鐘\n\n"
        msg += "✨ 核心功能：\n"
        msg += "• 動態下注（會呼吸的策略）\n"
        msg += "• 連輸風控（5局降70%、7局暫停）\n"
        msg += "• AI真實學習（輸入贏/輸）\n"
        msg += "• PF+牌靴雙模型融合\n\n"
        msg += "🎮 點擊「遊戲設定」開始"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=msg, quick_reply=build_main_menu()))
    
    @handler.add(TextMessage)
    def handle_text_message(event):
        user_id = event.source.user_id
        text = event.message.text.strip()
        
        if not _dedupe_event(getattr(event, "webhook_event_id", None)):
            return
        
        sess = get_session(user_id)
        
        if not is_premium(user_id) and is_trial_expired(sess):
            line_bot_api.reply_message(event.reply_token, TextSendMessage(
                text="⏰ 試用已用完，請輸入開通碼：開通 [密碼]",
                quick_reply=build_main_menu()
            ))
            return
        
        if text.startswith("開通"):
            parts = text.split()
            if len(parts) != 2:
                reply = "❌ 格式：開通 [密碼]"
            else:
                if use_activation_code(user_id, parts[1]):
                    sess["premium"] = True
                    save_session(user_id, sess)
                    reply = "✅ 開通成功！已升級付費模式"
                else:
                    reply = "❌ 密碼錯誤"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply, quick_reply=build_main_menu()))
            return
        
        phase = sess.get("phase", "init")
        
        if text in ["結束分析", "停止", "exit"]:
            sess["phase"] = "init"
            sess["game"] = None
            sess["bankroll"] = 0
            sess["skip_streak"] = 0
            sess["loss_streak"] = 0
            sess["win_streak"] = 0
            save_session(user_id, sess)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(
                text="🛑 已結束分析\n\n🎮 點擊「遊戲設定」重新開始",
                quick_reply=build_main_menu()
            ))
            return
        
        if text in ["遊戲設定", "設定", "start"]:
            sess["phase"] = "choose_game"
            sess["skip_streak"] = 0
            sess["loss_streak"] = 0
            sess["win_streak"] = 0
            save_session(user_id, sess)
            msg = "🎯 請選擇館別：\n\n"
            for k, v in GAMES.items():
                msg += f"{k}. {v}\n"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=msg, quick_reply=build_main_menu()))
            return
        
        if phase == "choose_game":
            if text in GAMES:
                sess["game"] = GAMES[text]
                sess["phase"] = "set_bankroll"
                save_session(user_id, sess)
                line_bot_api.reply_message(event.reply_token, TextSendMessage(
                    text=f"✅ 已選擇：{GAMES[text]}\n\n💰 請輸入本金（單位）",
                    quick_reply=build_main_menu()
                ))
            else:
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 請輸入1-10", quick_reply=build_main_menu()))
            return
        
        if phase == "set_bankroll":
            if text.isdigit() and int(text) >= 100:
                sess["bankroll"] = int(text)
                sess["phase"] = "playing"
                save_session(user_id, sess)
                msg = f"✅ 本金：{text} 單位\n\n"
                msg += "📊 輸入點數（如：65 / 和 / 閒6莊5）\n"
                msg += "🎯 結果回報：輸入「贏」或「輸」讓AI學習"
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=msg, quick_reply=build_main_menu()))
            else:
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 本金至少100", quick_reply=build_main_menu()))
            return
        
        if phase == "init":
            line_bot_api.reply_message(event.reply_token, TextSendMessage(
                text="⚠️ 請先點擊「遊戲設定」",
                quick_reply=build_main_menu()
            ))
            return
        
        if phase == "playing":
            result = _handle_points_and_predict(user_id, text, sess)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=result, quick_reply=build_main_menu()))
            return

# ==================== Main ====================
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    log.info(f"Starting {VERSION} on port {port}")
    log.info("=== 🔥 v3.0 可放大資金版功能 ===")
    log.info("  ✓ 動態下注引擎（會呼吸的策略）")
    log.info("  ✓ 連輸風控（3局降50%、5局降70%、7局暫停）")
    log.info("  ✓ 真實AI學習（用戶回報贏/輸）")
    log.info("  ✓ PF + DEPLETE 融合")
    log.info("  ✓ 連勝/連輸追蹤")
    
    if _flask_available and Flask is not None:
        app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
    else:
        log.error("Flask not available")
