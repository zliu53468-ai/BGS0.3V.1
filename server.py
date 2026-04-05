# -*- coding: utf-8 -*-
"""server.py — BGS Pure PF + EV Filter + Risk Control + Profit Taking (FINAL PROFIT VERSION - FIXED)"""
import os, sys, logging, time, re, json, threading
from typing import Optional, Dict, Any, Tuple, List
import numpy as np

# ==================== VERSION 定義 ====================
VERSION = "bgs-profit-final-v2025-11-03"

# ==================== 試用設定 ====================
TRIAL_SECONDS = int(os.getenv("TRIAL_SECONDS", "1800"))  # 30分鐘試用
PREMIUM_TRIAL_SECONDS = int(os.getenv("PREMIUM_TRIAL_SECONDS", "2592000"))  # 付費用戶30天

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
    """Keep Alive 防 Render 休眠"""
    try:
        import requests
    except Exception:
        log.warning("[KEEPALIVE] requests module not available, skip self-ping")
        return

    url = os.getenv("RENDER_EXTERNAL_URL")
    if not url:
        url = os.getenv("SELF_URL")
    if not url:
        url = os.getenv("SELF_PING_URL")

    interval = int(os.getenv("SELF_PING_INTERVAL", "120"))

    if not url:
        log.warning("[KEEPALIVE] No URL found, skip self-ping")
        return

    ping_url = url.rstrip("/") + "/ping"
    log.info(f"[KEEPALIVE] Started | URL: {ping_url} | interval: {interval}s")

    while True:
        try:
            r = requests.get(ping_url, timeout=10)
            if r.status_code < 400:
                log.info("[KEEPALIVE] self ping success")
            else:
                log.warning(f"[KEEPALIVE] ping returned status {r.status_code}")
        except Exception as e:
            log.warning("[KEEPALIVE] self ping failed: %s", e)
        time.sleep(interval)

# ==================== 館別選單 ====================
GAMES = {
    "1": "WM",
    "2": "PM", 
    "3": "DG",
    "4": "SA",
    "5": "KU",
    "6": "歐博/卡利",
    "7": "KG",
    "8": "全利",
    "9": "名人",
    "10": "MT真人"
}

# ==================== Deplete Module ====================
DEPLETE_OK = False
init_counts = None
probs_after_points = None

try:
    from deplete import init_counts, probs_after_points
    DEPLETE_OK = True
    log.info("Deplete module loaded successfully")
except Exception:
    try:
        from bgs.deplete import init_counts, probs_after_points
        DEPLETE_OK = True
        log.info("Deplete module loaded from bgs.deplete")
    except Exception:
        try:
            _cur_dir = os.path.dirname(os.path.abspath(__file__))
            if _cur_dir not in sys.path:
                sys.path.insert(0, _cur_dir)
            from deplete import init_counts, probs_after_points
            DEPLETE_OK = True
            log.info("Deplete module loaded from local path")
        except Exception as e:
            DEPLETE_OK = False
            log.warning(f"Deplete module not available: {e}")

# ==================== PF Module (Pure - No Pattern) ====================
PF_INITIALIZED = False
pf_model = None

class SmartDummyPF:
    """純PF模型 - 不做任何pattern偏移"""
    
    def __init__(self):
        self.base_probs = np.array([0.4586, 0.4462, 0.0952])
        log.info("Pure PF Dummy Model initialized")
    
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
                log.warning(f"Deplete calculation failed: {e}")
        
        return self.base_probs.copy()

try:
    try:
        from pf import SmartPF
        pf_model = SmartPF()
        PF_INITIALIZED = True
        log.info("Real PF module loaded successfully")
    except ImportError:
        pf_model = SmartDummyPF()
        PF_INITIALIZED = True
        log.info("Using Pure Dummy PF model")
except Exception as e:
    log.warning(f"PF module error: {e}")
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
redis_client: Optional["redis.Redis"] = None
if redis is not None and REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        log.info("Successfully connected to Redis.")
    except Exception as e:
        redis_client = None
        log.error(f"Failed to connect to Redis: {e}")
else:
    log.warning("Redis not configured, using in-memory session store.")

# ==================== Session Management ====================
SESS_FALLBACK: Dict[str, Dict[str, Any]] = {}
KV_FALLBACK: Dict[str, str] = {}
SESSION_EXPIRE_SECONDS = int(os.getenv("SESSION_EXPIRE_SECONDS", "1200"))
DEDUPE_TTL = 60

def _rget(k: str) -> Optional[str]:
    try:
        if redis_client:
            return redis_client.get(k)
        return KV_FALLBACK.get(k)
    except Exception as e:
        log.warning(f"[Redis] GET err: {e}")
        return None

def _rset(k: str, v: str, ex: Optional[int] = None):
    try:
        if redis_client:
            redis_client.set(k, v, ex=ex)
        else:
            KV_FALLBACK[k] = v
    except Exception as e:
        log.warning(f"[Redis] SET err: {e}")

def _rsetnx(k: str, v: str, ex: int) -> bool:
    try:
        if redis_client:
            return bool(redis_client.set(k, v, ex=ex, nx=True))
        if k in KV_FALLBACK:
            return False
        KV_FALLBACK[k] = v
        return True
    except Exception as e:
        log.warning(f"[Redis] SETNX err: {e}")
        return True

def _dedupe_event(event_id: Optional[str]) -> bool:
    if not event_id:
        return True
    key = f"dedupe:{event_id}"
    return _rsetnx(key, "1", ex=DEDUPE_TTL)

def _extract_line_event_id(event) -> Optional[str]:
    try:
        return getattr(event, "webhook_event_id", None) or getattr(event, "id", None)
    except Exception:
        return None

# ==================== Premium & Activation System ====================
def _premium_key(uid: str) -> str:
    return f"premium:{uid}"

def _premium_expire_key(uid: str) -> str:
    return f"premium_expire:{uid}"

def _blocked_key(uid: str) -> str:
    return f"blocked:{uid}"

def _block_reason_key(uid: str) -> str:
    return f"block_reason:{uid}"

def get_activation_codes() -> List[str]:
    raw = os.getenv("ACTIVATION_CODES", "")
    return [c.strip().upper() for c in raw.split(",") if c.strip()]

def use_activation_code(uid: str, code: str) -> bool:
    if code.upper() not in get_activation_codes():
        return False
    set_premium(uid, True, PREMIUM_TRIAL_SECONDS)
    return True

def is_premium(uid: str) -> bool:
    if not uid:
        return False
    return _rget(_premium_key(uid)) == "1"

def set_premium(uid: str, flag: bool = True, expire_seconds: int = PREMIUM_TRIAL_SECONDS) -> None:
    if not uid:
        return
    _rset(_premium_key(uid), "1" if flag else "0")
    if flag and expire_seconds > 0:
        _rset(_premium_expire_key(uid), str(int(time.time()) + expire_seconds), ex=expire_seconds)

def is_blocked(uid: str) -> bool:
    if not uid:
        return False
    return _rget(_blocked_key(uid)) == "1"

def get_block_reason(uid: str) -> str:
    if not uid:
        return "未知原因"
    reason = _rget(_block_reason_key(uid))
    return reason if reason else "帳號已停用"

def set_blocked(uid: str, reason: str = "試用到期") -> None:
    if not uid:
        return
    _rset(_blocked_key(uid), "1")
    _rset(_block_reason_key(uid), reason)

def unblock_user(uid: str) -> None:
    if not uid:
        return
    _rset(_blocked_key(uid), "0")
    _rset(_block_reason_key(uid), "")

# ==================== Trial System ====================
def get_trial_remaining(sess: Dict[str, Any]) -> int:
    start = sess.get("trial_start", int(time.time()))
    now = int(time.time())
    remain = TRIAL_SECONDS - (now - start)
    return max(0, remain)

def is_trial_expired(sess: Dict[str, Any]) -> bool:
    return get_trial_remaining(sess) <= 0

def format_trial_time(sess: Dict[str, Any]) -> str:
    remaining = get_trial_remaining(sess)
    minutes = remaining // 60
    if minutes > 0:
        return f"⏳ 試用剩餘 {minutes} 分鐘"
    else:
        return "⏰ 試用即將到期"

def _sess_key(uid: str) -> str:
    return f"sess:{uid}"

def get_session(uid: str) -> Dict[str, Any]:
    if not uid:
        uid = "anon"
    try:
        if redis_client:
            raw = redis_client.get(_sess_key(uid))
            if raw:
                sess = json.loads(raw)
                sess.setdefault("phase", "init")
                sess.setdefault("game", None)
                sess.setdefault("bankroll", 0)
                sess.setdefault("rounds_seen", 0)
                sess.setdefault("premium", is_premium(uid))
                sess.setdefault("trial_start", int(time.time()))
                sess.setdefault("skip_streak", 0)
                sess.setdefault("last_choice", None)
                return sess
        
        sess = SESS_FALLBACK.get(uid)
        if isinstance(sess, dict):
            sess.setdefault("phase", "init")
            sess.setdefault("game", None)
            sess.setdefault("bankroll", 0)
            sess.setdefault("rounds_seen", 0)
            sess.setdefault("premium", is_premium(uid))
            sess.setdefault("trial_start", int(time.time()))
            sess.setdefault("skip_streak", 0)
            sess.setdefault("last_choice", None)
            return sess
    except Exception as e:
        log.warning(f"get_session error: {e}")

    sess = {
        "phase": "init",
        "game": None,
        "bankroll": 0,
        "rounds_seen": 0,
        "premium": is_premium(uid),
        "trial_start": int(time.time()),
        "skip_streak": 0,
        "last_choice": None,
    }
    save_session(uid, sess)
    return sess

def save_session(uid: str, sess: Dict[str, Any]) -> None:
    if not uid:
        uid = "anon"
    try:
        payload = json.dumps(sess, ensure_ascii=False)
        if redis_client:
            redis_client.set(_sess_key(uid), payload, ex=SESSION_EXPIRE_SECONDS)
        else:
            SESS_FALLBACK[uid] = sess
    except Exception as e:
        log.warning(f"save_session error: {e}")

# ==================== Core Prediction Functions ====================
def parse_points_input(text: str) -> Optional[Tuple[List[int], str]]:
    """解析用戶輸入的點數"""
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

def calculate_baccarat_probabilities(points: List[int]) -> np.ndarray:
    """計算百家樂機率（純PF）"""
    return pf_model.predict_proba(points)

def calculate_ev(probs: np.ndarray) -> Tuple[float, float]:
    """計算期望值 EV"""
    pB, pP, pT = probs
    ev_banker = pB * 0.95 - (1 - pB)
    ev_player = pP * 1.0 - (1 - pP)
    return ev_banker, ev_player

def determine_bet_final(probs: np.ndarray, mode: str = "normal", bankroll: int = 1000, 
                        skip_streak: int = 0) -> Tuple[str, int, str]:
    """
    最終版下注決策 - 修正所有問題
    1. 不假設輸贏（移除loss_streak錯誤邏輯）
    2. 降低觀望門檻（只用EV判斷）
    3. 強制出手機制（連續觀望2局）
    4. 降低止盈門檻（連勝4局才停）
    5. 邊緣優勢測試
    """
    pB, pP, pT = probs
    ev_banker, ev_player = calculate_ev(probs)
    edge = abs(pB - pP)
    
    # 判斷最佳選擇（只看正EV）
    best_choice = None
    best_ev = -999
    
    if ev_banker > 0:
        best_choice = "莊"
        best_ev = ev_banker
    elif ev_player > 0:
        best_choice = "閒"
        best_ev = ev_player
    
    # ========== 修正1: 降低觀望門檻 ==========
    # 只要正EV就考慮出手，不再用edge < 0.02過濾
    if best_choice is None:
        # 沒有正EV，但edge還不錯 -> 邊緣優勢測試
        if edge > 0.015:
            if pB > pP:
                choice = "莊"
                bet_ratio = 0.002
                reason = "🧪 邊緣優勢測試（無正EV但差距明顯）"
            else:
                choice = "閒"
                bet_ratio = 0.002
                reason = "🧪 邊緣優勢測試（無正EV但差距明顯）"
            
            bet_amount = max(1, int(bankroll * bet_ratio))
            return choice, bet_amount, reason
        
        # ========== 修正2: 強制出手機制 ==========
        if skip_streak >= 2:
            if pB > pP:
                choice = "莊"
                reason = "🔄 連續觀望2局，強制小注出手"
            else:
                choice = "閒"
                reason = "🔄 連續觀望2局，強制小注出手"
            
            bet_amount = max(1, int(bankroll * 0.003))
            return choice, bet_amount, reason
        
        return "觀望", 0, "📉 期望值為負，等待更好機會"
    
    # ========== 正EV，計算下注金額 ==========
    if mode == "advanced":
        # 付費用戶
        if edge < 0.03:
            bet_ratio = 0.008
            reason = "📊 正EV + 微幅優勢"
        elif edge < 0.05:
            bet_ratio = 0.015
            reason = "📈 正EV + 明顯優勢"
        elif edge < 0.08:
            bet_ratio = 0.025
            reason = "🔥 正EV + 強勢優勢"
        else:
            bet_ratio = 0.04
            reason = "⚡ 正EV + 極強優勢"
    else:
        # 免費試用
        if edge < 0.03:
            bet_ratio = 0.005
            reason = "📊 正EV + 微幅優勢"
        elif edge < 0.05:
            bet_ratio = 0.01
            reason = "📈 正EV + 明顯優勢"
        else:
            bet_ratio = 0.018
            reason = "🔥 正EV + 強勢優勢"
    
    # EV加成
    if best_ev > 0.02:
        bet_ratio = min(bet_ratio * 1.3, 0.05)
        reason += " + EV顯著加成"
    
    bet_amount = max(1, int(bankroll * bet_ratio))
    bet_amount = min(bet_amount, int(bankroll * 0.05))
    
    return best_choice, bet_amount, reason

def format_output(probs: np.ndarray, choice: str, bet_amt: int, reason: str, mode: str = "", 
                  last_pts: Optional[str] = None, game: str = None, 
                  bankroll: int = None, trial_msg: str = None, 
                  skip_streak: int = 0,
                  ev_banker: float = None, ev_player: float = None) -> str:
    """格式化輸出訊息"""
    pB, pP, pT = [float(x) for x in probs]
    lines = []
    
    if game:
        lines.append(f"🎰 館別：{game}")
    if bankroll:
        lines.append(f"💰 剩餘本金：{bankroll}")
    if last_pts:
        lines.append(f"📊 輸入：{last_pts}")
    
    lines.append(f"🎲 機率｜莊 {pB*100:.1f}%｜閒 {pP*100:.1f}%｜和 {pT*100:.1f}%")
    lines.append(f"📈 差距｜莊閒 {abs(pB - pP) * 100:.1f}%")
    
    if ev_banker is not None and ev_player is not None:
        lines.append(f"📊 EV｜莊 {ev_banker*100:.2f}%｜閒 {ev_player*100:.2f}%")
    
    if mode:
        mode_display = "🎖️ 付費模式" if mode == "advanced" else "🆓 免費模式"
        lines.append(f"⚙️ 模式｜{mode_display}")
    
    if skip_streak > 0:
        lines.append(f"👀 連續觀望：{skip_streak} 局")
    
    if choice == "觀望":
        lines.append(f"👀 建議：觀望等待")
        if reason:
            lines.append(f"💡 {reason}")
    else:
        lines.append(f"🎯 建議：下注 {choice}")
        if bet_amt > 0:
            lines.append(f"💰 配注：{bet_amt} 單位")
            if bankroll and bet_amt > 0:
                risk_pct = (bet_amt / bankroll) * 100
                lines.append(f"📊 風險：{risk_pct:.1f}%")
        if reason:
            lines.append(f"💡 {reason}")
    
    if trial_msg:
        lines.append(trial_msg)
    
    lines.append("\n💡 輸入下一局點數（如：65 / 和 / 閒6莊5）")
    lines.append("📝 提示：系統不會假設輸贏，請自行記錄結果")
    
    return "\n".join(lines)

def _handle_points_and_predict(uid: str, points_text: str, sess: Dict[str, Any]) -> str:
    """處理點數輸入並預測 - 修正版：不假設輸贏"""
    parsed = parse_points_input(points_text)
    if not parsed:
        return "❌ 格式錯誤！\n請輸入：\n- 兩位點數：65\n- 四位點數：6523\n- 閒莊：閒6莊5\n- 和局：和"
    
    points, input_type = parsed
    
    if input_type == "tie":
        sess["last_choice"] = "和"
        save_session(uid, sess)
        return "🎲 和局\n\n💰 和局賠率通常為8倍\n建議謹慎下注"
    
    try:
        probs = calculate_baccarat_probabilities(points)
    except Exception as e:
        log.error(f"Probability calculation failed: {e}")
        return f"❌ 計算失敗：{str(e)}"
    
    mode = "advanced" if is_premium(uid) else "normal"
    bankroll = sess.get("bankroll", 1000)
    skip_streak = sess.get("skip_streak", 0)
    
    ev_banker, ev_player = calculate_ev(probs)
    
    # 最終決策
    choice, bet_amt, reason = determine_bet_final(probs, mode, bankroll, skip_streak)
    
    # 更新session - 重要：不假設輸贏，只更新觀望計數
    sess["last_pts_text"] = points_text
    sess["last_choice"] = choice
    sess["rounds_seen"] = sess.get("rounds_seen", 0) + 1
    
    # 只更新觀望計數，不更新勝敗
    if choice == "觀望":
        sess["skip_streak"] = skip_streak + 1
    else:
        sess["skip_streak"] = 0
        # 注意：不扣本金！不假設輸贏！
        # 讓用戶根據實際結果自行管理資金
    
    save_session(uid, sess)
    
    trial_msg = None
    if not is_premium(uid):
        trial_msg = format_trial_time(sess)
    
    return format_output(probs, choice, bet_amt, reason, mode, points_text, 
                        sess.get("game"), sess.get("bankroll"), trial_msg,
                        sess.get("skip_streak"), ev_banker, ev_player)

# ==================== LINE Bot Integration ====================
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
        log.info("LINE Bot initialized successfully")
    except Exception as e:
        log.error(f"LINE Bot initialization failed: {e}")
else:
    log.warning("LINE credentials not configured")

def build_main_menu() -> QuickReply:
    return QuickReply(
        items=[
            QuickReplyButton(action=MessageAction(label="🎮 遊戲設定", text="遊戲設定")),
            QuickReplyButton(action=MessageAction(label="🛑 結束分析", text="結束分析")),
        ]
    )

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
                    log.info("✅ KEEP ALIVE thread started")
    except Exception as e:
        log.warning(f"KEEP ALIVE boot failed: {e}")
else:
    class _DummyApp:
        def get(self, *a, **k):
            def _d(f): return f
            return _d
        def post(self, *a, **k):
            def _d(f): return f
            return _d
        def options(self, *a, **k):
            def _d(f): return f
            return _d
        def run(self, *a, **k):
            log.warning("Flask not available; cannot run HTTP server.")
    app = _DummyApp()

# ==================== Routes ====================
@app.get("/")
def root():
    ua = request.headers.get("User-Agent", "") if request else ""
    if "UptimeRobot" in ua or "bot" in ua.lower():
        return "OK", 200
    status = "OK" if PF_INITIALIZED else "BACKUP_MODE"
    return f"✅ BGS Server {status} ({VERSION})", 200

@app.get("/ping")
def ping():
    return "OK", 200

@app.get("/health")
def health():
    return {
        "ok": True,
        "ts": time.time(),
        "version": VERSION,
        "status": "running",
        "pf_initialized": PF_INITIALIZED,
        "deplete_ok": DEPLETE_OK,
        "line_bot": _line_bot_available,
        "redis": redis_client is not None,
        "trial_seconds": TRIAL_SECONDS
    }, 200

@app.post("/predict")
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        uid = data.get("uid", "anon")
        points = data.get("points", "")
        
        if not points:
            return jsonify({"error": "points required"}), 400
        
        sess = get_session(uid)
        
        if is_blocked(uid):
            return jsonify({"error": f"❌ {get_block_reason(uid)}", "blocked": True}), 403
        
        if not is_premium(uid) and is_trial_expired(sess):
            return jsonify({"error": "⏰ 試用已到期，請輸入開通碼", "trial_expired": True}), 403
        
        result = _handle_points_and_predict(uid, points, sess)
        return jsonify({"result": result}), 200
        
    except Exception as e:
        log.error(f"Predict endpoint error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.post("/webhook")
def webhook():
    if not _line_bot_available:
        return jsonify({"error": "LINE Bot not configured"}), 400
    
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    
    if not signature:
        log.warning("Missing X-Line-Signature header")
        return "Missing signature", 400
    
    try:
        handler.handle(body, signature)
    except Exception as e:
        log.error(f"Webhook handling error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    
    return "OK", 200

# ==================== LINE Event Handlers ====================
if _line_bot_available and handler:
    
    @handler.add(FollowEvent)
    def handle_follow(event):
        user_id = event.source.user_id
        welcome_msg = "🎰 歡迎使用BGS百家樂預測系統（最終盈利版）！\n\n"
        welcome_msg += f"⏳ 免費試用 {TRIAL_SECONDS//60} 分鐘\n\n"
        welcome_msg += "✨ 核心策略：\n"
        welcome_msg += "• EV期望值過濾\n"
        welcome_msg += "• 正EV才出手\n"
        welcome_msg += "• 強制出手機制\n"
        welcome_msg += "• 不假設輸贏\n\n"
        welcome_msg += "🎮 點擊下方「遊戲設定」開始使用"
        
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=welcome_msg, quick_reply=build_main_menu())
        )
    
    @handler.add(UnfollowEvent)
    def handle_unfollow(event):
        log.info(f"User {event.source.user_id} unfollowed")
    
    @handler.add(TextMessage)
    def handle_text_message(event):
        user_id = event.source.user_id
        message_text = event.message.text.strip()
        
        event_id = _extract_line_event_id(event)
        if not _dedupe_event(event_id):
            return
        
        sess = get_session(user_id)
        
        if is_blocked(user_id):
            reason = get_block_reason(user_id)
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(
                    text=f"❌ 帳號已停用\n原因：{reason}\n\n🔐 請輸入開通碼：開通 [密碼]",
                    quick_reply=build_main_menu()
                )
            )
            return
        
        if message_text.startswith("開通"):
            parts = message_text.split()
            if len(parts) != 2:
                reply_msg = "❌ 請輸入正確格式：開通 [密碼]\n\n範例：開通 VIP888"
            else:
                code = parts[1].strip().upper()
                if use_activation_code(user_id, code):
                    sess["premium"] = True
                    save_session(user_id, sess)
                    reply_msg = "✅ 開通成功！已升級付費模式 🎉\n\n🎮 點擊「遊戲設定」開始使用"
                else:
                    reply_msg = "❌ 密碼錯誤\n\n請確認後重新輸入，或聯繫客服"
            
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=reply_msg, quick_reply=build_main_menu())
            )
            return
        
        if not is_premium(user_id) and is_trial_expired(sess):
            msg = (
                "⏰ 免費試用 30 分鐘已用完\n\n"
                "🎯 想繼續使用嗎？\n\n"
                "🔐 請輸入：開通 你的專屬密碼\n"
                "👉 正確格式：開通 [密碼]\n\n"
                "📞 沒有密碼？請聯繫客服"
            )
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=msg, quick_reply=build_main_menu())
            )
            return
        
        phase = sess.get("phase", "init")
        text = message_text
        
        if text in ["結束分析", "停止", "exit"]:
            sess["phase"] = "init"
            sess["game"] = None
            sess["bankroll"] = 0
            sess["skip_streak"] = 0
            save_session(user_id, sess)
            
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(
                    text="🛑 已結束分析\n\n🎮 點擊「遊戲設定」重新開始",
                    quick_reply=build_main_menu()
                )
            )
            return
        
        if text in ["遊戲設定", "設定", "start", "重新設定"]:
            sess["phase"] = "choose_game"
            sess["skip_streak"] = 0
            save_session(user_id, sess)
            
            trial_msg = ""
            if not is_premium(user_id):
                trial_msg = f"\n\n{format_trial_time(sess)}"
            
            msg = "🎯 請選擇遊戲館別：\n\n"
            for k, v in GAMES.items():
                msg += f"{k}. {v}\n"
            msg += f"\n📝 請輸入數字（1-10）{trial_msg}"
            
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=msg, quick_reply=build_main_menu())
            )
            return
        
        if phase == "choose_game":
            if text in GAMES:
                sess["game"] = GAMES[text]
                sess["phase"] = "set_bankroll"
                save_session(user_id, sess)
                
                trial_msg = ""
                if not is_premium(user_id):
                    trial_msg = f"\n\n{format_trial_time(sess)}"
                
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(
                        text=f"✅ 已選擇：{GAMES[text]}\n\n💰 請輸入本金（單位）{trial_msg}",
                        quick_reply=build_main_menu()
                    )
                )
            else:
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text="❌ 請輸入正確數字（1-10）", quick_reply=build_main_menu())
                )
            return
        
        if phase == "set_bankroll":
            if text.isdigit() and int(text) >= 100:
                sess["bankroll"] = int(text)
                sess["phase"] = "playing"
                sess["skip_streak"] = 0
                save_session(user_id, sess)
                
                trial_msg = ""
                if not is_premium(user_id):
                    trial_msg = f"\n\n{format_trial_time(sess)}"
                
                msg = (
                    "✅ 設定完成！\n\n"
                    f"🎰 館別：{sess['game']}\n"
                    f"💰 本金：{text} 單位\n\n"
                    "📊 請輸入第一局點數\n"
                    "（例如：65 / 和 / 閒6莊5）\n\n"
                    "💡 重要：系統不假設輸贏\n"
                    "請根據實際結果自行管理資金"
                )
                msg += trial_msg
                
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text=msg, quick_reply=build_main_menu())
                )
            else:
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text="❌ 本金至少需要 100 單位\n請重新輸入", quick_reply=build_main_menu())
                )
            return
        
        if phase == "init":
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(
                    text="⚠️ 請先點擊「遊戲設定」開始使用",
                    quick_reply=build_main_menu()
                )
            )
            return
        
        if phase == "playing":
            result = _handle_points_and_predict(user_id, text, sess)
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=result, quick_reply=build_main_menu())
            )
            return

# ==================== Main Entry ====================
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    log.info(f"Starting {VERSION} on port {port}")
    log.info(f"PF Initialized: {PF_INITIALIZED} (Pure mode - no pattern)")
    log.info(f"Deplete Available: {DEPLETE_OK}")
    log.info(f"LINE Bot: {_line_bot_available}")
    log.info(f"Redis: {redis_client is not None}")
    log.info(f"Games Available: {len(GAMES)}")
    log.info(f"Trial Seconds: {TRIAL_SECONDS} ({TRIAL_SECONDS//60} minutes)")
    log.info(f"Activation Codes: {len(get_activation_codes())} codes loaded")
    
    if _flask_available and Flask is not None:
        app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
    else:
        log.error("Flask not available; cannot run HTTP server.")
