# server.py — Text-only LINE UX + Big Road PNG + Trial (30m) + Permanent Codes (30) + Kelly staking
# 特性：
# - 純文字：貼歷史 →「開始分析」→ 每打一手即回下一手預測（含表情）
# - 首次引導輸入本金；下注建議金額「顯示 本金 × % = 金額」，且 % ∈ [10%, 30%]
# - 試用 30 分鐘；輸入任一「永久開通碼」解鎖（碼不消耗；池中 30 組；ENV 未填會自動產 30 組寫入 codes.txt）
# - 大路（Big Road）建構 + /road/image 回大路 PNG；輸入「路圖」回圖
# - REST：/predict /road /road/image /health
# 相依：Flask, gunicorn, line-bot-sdk, Pillow

import os, csv, time, logging, random, string, re
from typing import List, Dict, Tuple, Optional
from collections import deque
from io import BytesIO

from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageDraw  # 需要 requirements.txt: Pillow==10.4.0

# -------------------- Flask / Logging --------------------
app = Flask(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s"
)
log = logging.getLogger("bgs")

# -------------------- Writable paths --------------------
def _is_writable(p: str) -> bool:
    try:
        os.makedirs(p, exist_ok=True)
        t = os.path.join(p, ".w")
        with open(t, "w") as f:
            f.write("ok")
        os.remove(t)
        return True
    except Exception as e:
        log.warning("not writable %s: %s", p, e)
        return False

def _base_dir() -> str:
    user = os.getenv("DATA_BASE","").strip()
    if user and _is_writable(user): return user
    if _is_writable("/tmp/bgs"): return "/tmp/bgs"
    local = os.path.join(os.getcwd(), "data")
    if _is_writable(local): return local
    return "/tmp"

BASE = _base_dir()

def _csv_path() -> str:
    custom = os.getenv("DATA_LOG_PATH","").strip()
    if custom:
        parent = os.path.dirname(custom)
        if parent and _is_writable(parent): return custom
        log.warning("DATA_LOG_PATH invalid, fallback used.")
    lg = os.path.join(BASE, "logs"); os.makedirs(lg, exist_ok=True)
    return os.path.join(lg, "rounds.csv")

CSV_PATH = _csv_path()

# -------------------- 常數 / 先驗 --------------------
CLASS_ORDER = ("B","P","T")
LAB_ZH = {"B":"莊","P":"閒","T":"和"}
THEORETICAL = {"B":0.458,"P":0.446,"T":0.096}
PAYOUT = {"B":0.95, "P":1.0, "T":8.0}
MAX_HISTORY = int(os.getenv("MAX_HISTORY","400"))

# Kelly / 配注控制（可由環境變數微調）
KELLY_SCALE    = float(os.getenv("KELLY_SCALE", "0.50"))
MIN_BET_FRAC   = float(os.getenv("MIN_BET_FRAC","0.10"))   # >=10%
MAX_BET_FRAC   = float(os.getenv("MAX_BET_FRAC","0.30"))   # <=30%
ROUND_TO       = int(os.getenv("ROUND_TO","10"))

# 試用與開通碼（永久，可重複使用）
TRIAL_MINUTES  = int(os.getenv("TRIAL_MINUTES","30"))
ACCOUNT_REGEX  = re.compile(r"^[A-Z]{5}\d{5}$")
GLOBAL_CODES   = set()
CODES_FILE     = os.path.join(BASE, "codes.txt")

# -------------------- 使用者狀態 --------------------
USER_HISTORY:     Dict[str, List[str]] = {}
USER_READY:       Dict[str, bool]      = {}
USER_TRIAL_START: Dict[str, float]     = {}
USER_ACTIVATED:   Dict[str, bool]      = {}
USER_CODE:        Dict[str, str]       = {}
USER_BANKROLL:    Dict[str, int]       = {}
_LAST_HIT:        Dict[str, float]     = {}

# -------------------- 工具 --------------------
def now_ts() -> float: return time.time()

def debounce(uid: str, key: str, window: float = 1.2) -> bool:
    k = f"{uid}:{key}"; last = _LAST_HIT.get(k, 0.0); t = now_ts()
    if t - last < window: return True
    _LAST_HIT[k] = t; return False

def ensure_user(uid: str):
    USER_HISTORY.setdefault(uid, [])
    USER_READY.setdefault(uid, False)
    USER_TRIAL_START.setdefault(uid, now_ts())
    USER_ACTIVATED.setdefault(uid, False)
    USER_CODE.setdefault(uid, "")

def trial_ok(uid: str) -> bool:
    if USER_ACTIVATED.get(uid, False): return True
    start = USER_TRIAL_START.get(uid, now_ts())
    return (now_ts() - start) / 60.0 <= TRIAL_MINUTES

def try_activate(uid: str, text: str) -> bool:
    code = (text or "").strip().upper().replace(" ", "")
    if not ACCOUNT_REGEX.fullmatch(code):
        return False
    if code in GLOBAL_CODES:
        USER_ACTIVATED[uid] = True
        USER_CODE[uid] = code
        log.info("[ACT] uid=%s activated with code=%s (permanent)", uid, code)
        return True
    return False

def _mk_code() -> str:
    return "".join(random.choices(string.ascii_uppercase, k=5)) + \
           "".join(random.choices(string.digits, k=5))

def _load_codes_from_file(path: str) -> set:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return {
                    ln.strip().upper()
                    for ln in f
                    if ACCOUNT_REGEX.fullmatch(ln.strip().upper())
                }
    except Exception as e:
        log.warning("load codes failed: %s", e)
    return set()

def _save_codes_to_file(path: str, codes: set):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for c in sorted(codes):
                f.write(c + "\n")
    except Exception as e:
        log.warning("save codes failed: %s", e)

def init_activation_codes(base_dir: str):
    """初始化『永久可重複使用』代碼池：ENV > 檔案 > 隨機產 30 組。"""
    global CODES_FILE, GLOBAL_CODES
    CODES_FILE = os.path.join(base_dir, "codes.txt")

    env_codes = os.getenv("ACTIVATION_CODES", "").strip()
    if env_codes:
        GLOBAL_CODES.clear()
        for token in env_codes.replace(" ", ",").replace("，", ",").split(","):
            t = token.strip().upper()
            if ACCOUNT_REGEX.fullmatch(t):
                GLOBAL_CODES.add(t)
        _save_codes_to_file(CODES_FILE, GLOBAL_CODES)
        log.info("[ACT] Loaded %d codes from ENV (permanent).", len(GLOBAL_CODES))
        return

    GLOBAL_CODES = _load_codes_from_file(CODES_FILE)
    if GLOBAL_CODES:
        log.info("[ACT] Loaded %d codes from file (permanent).", len(GLOBAL_CODES))
        return

    while len(GLOBAL_CODES) < 30:
        GLOBAL_CODES.add(_mk_code())
    _save_codes_to_file(CODES_FILE, GLOBAL_CODES)
    log.info("[ACT] Generated %d activation codes (permanent). See logs.", len(GLOBAL_CODES))
    for c in sorted(GLOBAL_CODES):
        log.info("[ACT-CODE] %s", c)

init_activation_codes(BASE)

# -------------------- 解析與常用 --------------------
def zh_to_bpt(ch: str) -> Optional[str]:
    if ch in ("莊","B","b"): return "B"
    if ch in ("閒","P","p"): return "P"
    if ch in ("和","T","t"): return "T"
    return None

def parse_text_seq(text: str) -> List[str]:
    """從任意文字中抓出 B/P/T/莊/閒/和 的序列（允許貼整串歷史）。"""
    res=[]
    for ch in text:
        v = zh_to_bpt(ch)
        if v: res.append(v)
    if not res:
        for tk in text.replace(",", " ").split():
            for ch in tk:
                v = zh_to_bpt(ch)
                if v: res.append(v)
    return res[-MAX_HISTORY:] if len(res)>MAX_HISTORY else res

def parse_bankroll(text: str) -> Optional[int]:
    nums = re.findall(r"\d+", text.replace(",", ""))
    if not nums: return None
    try:
        val = int(nums[0])
        return val if val > 0 else None
    except: return None

def format_money(x: float) -> str:
    return f"{int(round(x / max(1, ROUND_TO))) * max(1, ROUND_TO):,}"

# -------------------- 機率估計（逐手 + 大路） --------------------
def norm(v: List[float]) -> List[float]:
    s = sum(v); s = s if s>1e-12 else 1.0
    return [max(0.0, x)/s for x in v]

def temperature(p: List[float], tau: float) -> List[float]:
    if tau <= 1e-9: return p
    ex = [pow(max(pi,1e-12), 1.0/tau) for pi in p]
    s = sum(ex); return [e/s for e in ex]

def recent_freq(seq: List[str], win:int) -> List[float]:
    if not seq: return [1/3,1/3,1/3]
    cut = seq[-win:] if win>0 else seq
    a = float(os.getenv("LAPLACE","0.5"))
    nB=cut.count("B")+a; nP=cut.count("P")+a; nT=cut.count("T")+a
    tot=max(1,len(cut)) + 3*a
    return [nB/tot, nP/tot, nT/tot]

def exp_decay_freq(seq: List[str], gamma: Optional[float]=None) -> List[float]:
    if not seq: return [1/3,1/3,1/3]
    if gamma is None: gamma=float(os.getenv("EW_GAMMA","0.96"))
    wB=wP=wT=0.0; w=1.0
    for r in reversed(seq):
        if r=="B": wB+=w
        elif r=="P": wP+=w
        else: wT+=w
        w*=gamma
    a=float(os.getenv("LAPLACE","0.5"))
    wB+=a; wP+=a; wT+=a
    S=wB+wP+wT
    return [wB/S, wP/S, wT/S]

# 大路
def build_big_road(seq: List[str]) -> List[Dict]:
    cols=[]; cur=None; length=0; ties=0
    for r in seq:
        if r=="T":
            if cur is not None: ties+=1
            continue
        if cur is None:
            cur=r; length=1; ties=0
        elif r==cur:
            length+=1
        else:
            cols.append({"color":cur,"len":length,"ties":ties})
            cur=r; length=1; ties=0
    if cur is not None:
        cols.append({"color":cur,"len":length,"ties":ties})
    keep=int(os.getenv("ROAD_KEEP_COLS","120"))
    return cols[-keep:] if len(cols)>keep else cols

def road_tail(cols: List[Dict], k:int=1):
    if not cols or k<=0 or k>len(cols): return None,0
    c=cols[-k]; return c["color"], c["len"]

def _run_length_tail(seq: List[str], k:int=1):
    if not seq: return None,0
    blocks=deque(); cur, cnt = seq[-1],1
    for x in reversed(seq[:-1]):
        if x==cur: cnt+=1
        else:
            blocks.appendleft((cur,cnt))
            cur, cnt = x,1
    blocks.appendleft((cur,cnt))
    return blocks[-k] if 0<k<=len(blocks) else (None,0)

def pattern_boost(seq: List[str]) -> Dict[str,float]:
    m={"B":1.0,"P":1.0,"T":1.0}
    n=len(seq)
    if n<3: return m
    alt=float(os.getenv("HOOK_ALT","1.06"))
    dbl=float(os.getenv("HOOK_DBLJUMP","1.04"))
    cyc=float(os.getenv("HOOK_CYCLE","1.06"))
    th=int(os.getenv("HOOK_DRAGON_LEN","3"))
    k=float(os.getenv("HOOK_DRAGON_K","0.06"))
    a,b = seq[-1], seq[-2]
    # 交替
    if a!=b and n>=4 and seq[-3]!=seq[-2] and seq[-4]!=seq[-3]:
        if a=="P": m["B"]*=alt
        if a=="B": m["P"]*=alt
    # 雙跳
    if n>=4 and seq[-1]==seq[-2] and seq[-3]!=seq[-2] and seq[-3]==seq[-4]:
        if a=="P": m["B"]*=dbl
        if a=="B": m["P"]*=dbl
    # 1-2/2-1
    if n>=6:
        last6="".join(seq[-6
