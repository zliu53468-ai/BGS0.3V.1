# server.py — Proactive 30m trial + persistent state + text-only LINE UX + Big Road PNG
# Requirements:
#   Flask==3.0.3
#   gunicorn==21.2.0
#   line-bot-sdk==3.11.0
#   Pillow==10.4.0

import os, csv, time, logging, random, string, re, math, threading, json
from typing import List, Dict, Tuple, Optional
from collections import deque
from io import BytesIO

from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageDraw

# -------------------- Flask / Logs --------------------
app = Flask(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s"
)
log = logging.getLogger("bgs")

# -------------------- Writable base --------------------
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

def _base_prior_from_env():
    b = float(os.getenv("BASE_B", "0.458"))
    p = float(os.getenv("BASE_P", "0.446"))
    t = float(os.getenv("BASE_T", "0.096"))
    s = b + p + t
    if s <= 0:
        return {"B": 0.458, "P": 0.446, "T": 0.096}
    return {"B": b/s, "P": p/s, "T": t/s}

THEORETICAL = _base_prior_from_env()
PAYOUT = {"B":0.95, "P":1.0, "T":8.0}
MAX_HISTORY = int(os.getenv("MAX_HISTORY","400"))

# Kelly / 配注控制
KELLY_SCALE    = float(os.getenv("KELLY_SCALE", "0.50"))
MIN_BET_FRAC   = float(os.getenv("MIN_BET_FRAC","0.10"))   # >=10%
MAX_BET_FRAC   = float(os.getenv("MAX_BET_FRAC","0.30"))   # <=30%
ROUND_TO       = int(os.getenv("ROUND_TO","10"))

# 試用 & 開通碼（永久）
TRIAL_MINUTES     = int(os.getenv("TRIAL_MINUTES","30"))
TRIAL_SCAN_INTERVAL_SEC = int(os.getenv("TRIAL_SCAN_INTERVAL_SEC","30"))  # 背景掃描頻率
ACCOUNT_REGEX     = re.compile(r"^[A-Z]{5}\d{5}$")
GLOBAL_CODES      = set()
CODES_FILE        = os.path.join(BASE, "codes.txt")

# 狀態持久化
STATE_DIR         = os.path.join(BASE, "state")
PERSIST_ENABLED   = os.getenv("PERSIST_STATE","1") != "0"

# -------------------- 使用者狀態（記憶體） --------------------
USER_HISTORY:      Dict[str, List[str]] = {}
USER_READY:        Dict[str, bool]      = {}
USER_TRIAL_START:  Dict[str, float]     = {}
USER_ACTIVATED:    Dict[str, bool]      = {}
USER_CODE:         Dict[str, str]       = {}
USER_BANKROLL:     Dict[str, int]       = {}
USER_TRIAL_WARNED: Dict[str, bool]      = {}
ACTIVE_USERS:      set                  = set()

def now_ts() -> float: return time.time()

# -------------------- 狀態持久化 helpers --------------------
def _state_path(uid: str) -> str:
    return os.path.join(STATE_DIR, f"{uid}.json")

def _load_state(uid: str) -> dict:
    if not PERSIST_ENABLED: return {}
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        p = _state_path(uid)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log.warning("load state failed uid=%s: %s", uid, e)
    return {}

def _save_state(uid: str):
    if not PERSIST_ENABLED: return
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        data = {
            "trial_start": USER_TRIAL_START.get(uid),
            "activated": USER_ACTIVATED.get(uid, False),
            "code": USER_CODE.get(uid, ""),
            "warned": USER_TRIAL_WARNED.get(uid, False),
            "bankroll": USER_BANKROLL.get(uid, 0),
        }
        with open(_state_path(uid), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        log.warning("save state failed uid=%s: %s", uid, e)

def ensure_user(uid: str):
    # 先載入
    st = _load_state(uid)
    USER_HISTORY.setdefault(uid, [])
    USER_READY.setdefault(uid, False)
    USER_TRIAL_START.setdefault(uid, st.get("trial_start", now_ts()))
    USER_ACTIVATED.setdefault(uid, bool(st.get("activated", False)))
    USER_CODE.setdefault(uid, st.get("code",""))
    USER_TRIAL_WARNED.setdefault(uid, bool(st.get("warned", False)))
    USER_BANKROLL.setdefault(uid, int(st.get("bankroll", 0)))
    ACTIVE_USERS.add(uid)
    # 若首次建立 trial_start，立即落地
    if "trial_start" not in st:
        _save_state(uid)

def trial_ok(uid: str) -> bool:
    if USER_ACTIVATED.get(uid, False): return True
    start = USER_TRIAL_START.get(uid, now_ts())
    return (now_ts() - start) / 60.0 <= TRIAL_MINUTES

# -------------------- 開通碼（永久 30 組；ENV 覆寫） --------------------
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
    global CODES_FILE, GLOBAL_CODES
    CODES_FILE = os.path.join(base_dir, "codes.txt")

    env_codes = os.getenv("ACTIVATION_CODES", "").strip()
    if env_codes:
        for token in env_codes.replace(" ", ",").replace("，", ",").split(","):
            t = token.strip().upper()
            if ACCOUNT_REGEX.fullmatch(t):
                GLOBAL_CODES.add(t)
        _save_codes_to_file(CODES_FILE, GLOBAL_CODES)
        log.info("[ACT] Loaded %d codes from ENV (permanent).", len(GLOBAL_CODES))
        return

    GLOBAL_CODES = _load_codes_from_file(CODES_FILE)
    if not GLOBAL_CODES:
        while len(GLOBAL_CODES) < 30:
            GLOBAL_CODES.add(_mk_code())
        _save_codes_to_file(CODES_FILE, GLOBAL_CODES)
        log.info("[ACT] Generated %d activation codes (permanent). Stored in %s", len(GLOBAL_CODES), CODES_FILE)
        for c in sorted(GLOBAL_CODES):
            log.info("[ACT-CODE] %s", c)
    else:
        log.info("[ACT] Loaded %d codes from file (permanent).", len(GLOBAL_CODES))

def try_activate(uid: str, text: str) -> bool:
    code = (text or "").strip().upper().replace(" ", "")
    if not ACCOUNT_REGEX.fullmatch(code): return False
    if code in GLOBAL_CODES:
        USER_ACTIVATED[uid] = True
        USER_CODE[uid] = code
        USER_TRIAL_WARNED[uid] = True  # 之後不再推播試用到期
        _save_state(uid)
        log.info("[ACT] uid=%s activated with code=%s", uid, code)
        return True
    return False

init_activation_codes(BASE)

# -------------------- 解析/格式 --------------------
def zh_to_bpt(ch: str) -> Optional[str]:
    if ch in ("莊","B","b"): return "B"
    if ch in ("閒","P","p"): return "P"
    if ch in ("和","T","t"): return "T"
    return None

def parse_text_seq(text: str) -> List[str]:
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
    import re
    nums = re.findall(r"\d+", text.replace(",", ""))
    if not nums: return None
    try:
        val = int(nums[0])
        return val if val > 0 else None
    except: return None

def format_money(x: float) -> str:
    rt = max(1, ROUND_TO)
    return f"{int(round(x / rt)) * rt:,}"

# -------------------- 機率/指標 --------------------
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
    if gamma is None: gamma=float(os.getenv("EW_GAMMA","0.95"))
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

# -------------------- 大路（Big Road） --------------------
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

# -------------------- Volatility --------------------
def _entropy_bp(s: List[str]) -> float:
    a=[x for x in s if x in ("B","P")]
    if not a: return 0.0
    pB=a.count("B")/len(a); pP=1.0-pB
    ent=0.0
    for p in (pB,pP):
        if p>1e-12: ent -= p*math.log2(p)
    return ent

def _alt_rate(s: List[str]) -> float:
    a=[x for x in s if x in ("B","P")]
    if len(a)<2: return 0.0
    alt=sum(1 for i in range(1,len(a)) if a[i]!=a[i-1])
    return alt/(len(a)-1)

def _volatility_score(seq: List[str]) -> Tuple[float, dict]:
    win=int(os.getenv("VOL_WIN","8"))
    s=seq[-win:] if len(seq)>=win else seq[:]
    ent=_entropy_bp(s)
    alt=_alt_rate(s)
    vol = max(0.0, min(1.0, 0.6*ent + 0.4*alt))
    return vol, {"window":win, "entropy":ent, "alt_rate":alt, "score":vol}

def _last_k_has(items: List[str], target: str, k:int) -> bool:
    s = items[-k:] if len(items)>=k else items[:]
    return target in s

# -------------------- Signals（含 Tie-aware） --------------------
def _is_alt_dense(seq: List[str], win:int) -> bool:
    if len(seq) < 3: return False
    s = seq[-win:] if len(seq) >= win else seq[:]
    alts = 0
    for i in range(1, len(s)):
        if s[i] != s[i-1] and s[i] in ("B","P") and s[i-1] in ("B","P"):
            alts += 1
    return alts >= max(2, len(s)-1)

def _is_double_jump(seed: List[str]) -> bool:
    s = [x for x in seed if x in ("B","P")]
    n = len(s)
    if n < 3: return False
    if n >= 4 and s[-1]==s[-2] and s[-3]!=s[-2] and s[-3]==s[-4]:
        return True
    if n >= 3 and s[-1]==s[-2] and s[-3]!=s[-2]:
        return True
    return False

def _cusum_change(seq: List[str], w:int=8) -> int:
    s = [x for x in seq if x in ("B","P")]
    if len(s) < 4: return 0
    cut = s[-w:] if len(s) >= w else s[:]
    m = len(cut)//2
    pre = cut[:m]; post = cut[m:]
    def bp_score(arr): return arr.count("B") - arr.count("P")
    d = bp_score(post) - bp_score(pre)
    if abs(d) <= 0: return 0
    return 1 if d > 0 else -1

def signals(seq: List[str]) -> Tuple[Dict[str,float], Dict[str,dict]]:
    mult = {"B":1.0,"P":1.0,"T":1.0}
    dbg  = {}

    n = len(seq)
    if n < 2:
        return mult, dbg

    ALT_WIN      = int(os.getenv("ALT_WINDOW", "5"))
    ALT_GAIN     = float(os.getenv("ALT_GAIN", "1.12"))
    ALT_LEAD     = float(os.getenv("ALT_LEAD", "1.06"))
    DBL_GAIN     = float(os.getenv("DBL_GAIN", "1.10"))
    MOM_WIN      = int(os.getenv("MOMENTUM_WIN", "8"))
    MOM_BONUS    = float(os.getenv("MOMENTUM_BONUS", "0.05"))
    CHOP_WIN     = int(os.getenv("CHOP_WINDOW", "6"))
    CHOP_GAIN    = float(os.getenv("CHOP_GAIN", "1.08"))

    DR_FOLLOW_LEN   = int(os.getenv("DRAGON_FOLLOW_LEN", "3"))
    DR_FOLLOW_STEP  = float(os.getenv("DRAGON_FOLLOW_STEP", "0.03"))
    DR_BREAK_PRE    = int(os.getenv("DRAGON_BREAK_PREEMPT", "2"))
    DR_BREAK_STEP   = float(os.getenv("DRAGON_BREAK_STEP", "0.08"))
    DR_FOLLOW_W     = float(os.getenv("DRAGON_FOLLOW_W", "0.5"))
    DR_BREAK_W      = float(os.getenv("DRAGON_BREAK_W", "1.1"))

    VOL_SIG_BOOST = float(os.getenv("VOL_SIG_BOOST","0.50"))
    vol, volmeta = _volatility_score(seq)
    sig_scale = (1.0 + VOL_SIG_BOOST * vol)

    # Tie-aware
    T_NEAR_WIN      = int(os.getenv("T_NEAR_WIN","5"))
    T_NEAR_GAIN     = float(os.getenv("T_NEAR_GAIN","1.06"))
    T_CLUSTER_WIN   = int(os.getenv("T_CLUSTER_WIN","10"))
    T_CLUSTER_TH    = int(os.getenv("T_CLUSTER_TH","2"))
    T_CLUSTER_GAIN  = float(os.getenv("T_CLUSTER_GAIN","1.10"))
    T_BREAK_GAIN    = float(os.getenv("T_BREAK_RUN_GAIN","1.10"))
    T_POST_REV_GAIN = float(os.getenv("T_POST_REV_GAIN","1.08"))

    a, b = seq[-1], seq[-2]

    # ALT
    if _is_alt_dense(seq, ALT_WIN):
        opp = "B" if a=="P" else "P" if a=="B" else None
        if opp:
            g = ALT_GAIN * sig_scale
            mult[opp] *= g
            dbg["ALT"] = {"target": opp, "gain": g, "mode":"dense"}
    else:
        if a != b and a in ("B","P") and b in ("B","P"):
            opp = "B" if a=="P" else "P"
            g = ALT_LEAD * sig_scale
            mult[opp] *= g
            dbg["ALT"] = {"target": opp, "gain": g, "mode":"lead"}

    # DBL
    if _is_double_jump(seq):
        opp = "B" if a=="P" else "P" if a=="B" else None
        if opp:
            g = DBL_GAIN * sig_scale
            mult[opp] *= g
            dbg["DBL"] = {"target": opp, "gain": g}

    # MOMENTUM
    if n >= 3:
        s = [x for x in (seq[-MOM_WIN:] if n>=MOM_WIN else seq) if x in ("B","P")]
        if len(s) >= 3:
            diff = s.count("B") - s.count("P")
            if diff != 0:
                side = "B" if diff>0 else "P"
                gain = (1.0 + min(0.15, abs(diff) * MOM_BONUS)) * sig_scale
                mult[side] *= gain
                dbg["MOM"] = {"target": side, "gain": gain, "diff": diff}

    # CHOP
    if n >= 4:
        s = seq[-CHOP_WIN:] if n>=CHOP_WIN else seq
        alt_cnt = sum(1 for i in range(1,len(s)) if s[i]!=s[i-1] and s[i] in ("B","P") and s[i-1] in ("B","P"))
        if alt_cnt >= max(2, len(s)//2):
            opp = "B" if a=="P" else "P" if a=="B" else None
            if opp:
                g = CHOP_GAIN * sig_scale
                mult[opp] *= g
                dbg["CHOP"] = {"target": opp, "gain": g, "alts": alt_cnt}

    # DRAGON（跟 / 斷）
    sym, run = _run_length_tail(seq,1)
    if sym in ("B","P") and run >= 1:
        follow_gain = 1.0
        if run >= DR_FOLLOW_LEN:
            follow_gain += DR_FOLLOW_STEP * (run - (DR_FOLLOW_LEN - 1))
        break_gain = 1.0
        if run >= DR_BREAK_PRE:
            break_gain += DR_BREAK_STEP * (run - (DR_BREAK_PRE - 1))
        follow_side = sym
        break_side  = "B" if sym=="P" else "P"
        follow_gain = max(1.0, min(1.7, follow_gain)) * sig_scale
        break_gain  = max(1.0, min(1.9, break_gain))  * sig_scale
        mult[follow_side] *= pow(follow_gain, DR_FOLLOW_W)
        mult[break_side]  *= pow(break_gain,  DR_BREAK_W)
        dbg["DRAGON"] = {"run": run, "follow": {"side": follow_side, "gain": follow_gain}, "break": {"side": break_side, "gain": break_gain}}

    # TIE-aware
    if _last_k_has(seq, "T", T_NEAR_WIN):
        g = (T_NEAR_GAIN * sig_scale)
        mult["T"] *= g
        dbg["T_NEAR"] = {"win": T_NEAR_WIN, "gain": g}

    if T_CLUSTER_WIN > 0:
        s = seq[-T_CLUSTER_WIN:] if len(seq)>=T_CLUSTER_WIN else seq[:]
        tcnt = s.count("T")
        if tcnt >= T_CLUSTER_TH:
            g = (T_CLUSTER_GAIN * sig_scale)
            mult["T"] *= g
            dbg["T_CLUSTER"] = {"win": T_CLUSTER_WIN, "cnt": tcnt, "th": T_CLUSTER_TH, "gain": g}

    if n >= 3 and seq[-2] == "T" and seq[-1] in ("B","P"):
        pre_sym, pre_len = _run_length_tail(seq[:-1], 1)
        if pre_sym in ("B","P") and pre_len >= 2:
            opp = "B" if pre_sym=="P" else "P"
            g = (T_BREAK_GAIN * sig_scale)
            mult[opp] *= g
            dbg["T_BREAK_RUN"] = {"pre_sym": pre_sym, "pre_len": pre_len, "target": opp, "gain": g}

    if n >= 3 and seq[-2] == "T" and seq[-3] in ("B","P") and seq[-1] in ("B","P") and seq[-3] != seq[-1]:
        y = seq[-1]
        g = (T_POST_REV_GAIN * sig_scale)
        mult[y] *= g
        dbg["T_POST_REV"] = {"pattern": f"{seq[-3]}-T-{y}", "target": y, "gain": g}

    return mult, dbg

# -------------------- 機率估計 --------------------
def estimate_probs(seq: List[str]) -> Tuple[List[float], Dict]:
    if not seq:
        base=[THEORETICAL["B"],THEORETICAL["P"],THEORETICAL["T"]]
        return norm(base), {"base":base,"signals":{}}

    short = recent_freq(seq, int(os.getenv("WIN_SHORT","6")))
    longv = exp_decay_freq(seq, float(os.getenv("EW_GAMMA","0.95")))
    prior = [THEORETICAL["B"],THEORETICAL["P"],THEORETICAL["T"]]

    REC_W  = float(os.getenv("REC_W","0.40"))
    LONG_W = float(os.getenv("LONG_W","0.20"))
    PRIOR_W= float(os.getenv("PRIOR_W","0.10"))

    vol, volmeta = _volatility_score(seq)
    VOL_REC_BOOST  = float(os.getenv("VOL_REC_BOOST","1.20"))
    VOL_LONG_CUT   = float(os.getenv("VOL_LONG_CUT","0.80"))
    VOL_PRIOR_CUT  = float(os.getenv("VOL_PRIOR_CUT","0.80"))

    rec_w  = REC_W  * (1.0 + VOL_REC_BOOST * vol)
    long_w = LONG_W * (1.0 - VOL_LONG_CUT  * vol)
    prior_w= PRIOR_W* (1.0 - VOL_PRIOR_CUT * vol)
    rec_w  = max(0.05, min(2.5, rec_w))
    long_w = max(0.00, min(1.0, long_w))
    prior_w= max(0.00, min(0.8, prior_w))

    base = [rec_w*short[i] + long_w*longv[i] + prior_w*prior[i] for i in range(3)]
    base = norm(base)

    mult, sdbg = signals(seq)

    biasB=float(os.getenv("BIAS_B","1.0"))
    biasP=float(os.getenv("BIAS_P","1.0"))
    biasT=float(os.getenv("BIAS_T","1.0"))

    p=[base[0]*mult["B"]*biasB,
       base[1]*mult["P"]*biasP,
       base[2]*mult["T"]*biasT]

    # T floor/cap
    T_FLOOR = float(os.getenv("T_FLOOR","0.05"))
    T_CAP   = float(os.getenv("T_CAP","0.40"))
    p[2] = min(T_CAP, max(T_FLOOR, p[2]))

    p=norm(p)
    p=temperature(p, float(os.getenv("TEMP","1.02")))
    floor=float(os.getenv("EPSILON_FLOOR","0.04"))
    cap=float(os.getenv("MAX_CAP","0.92"))
    p=[min(cap, max(floor,x)) for x in p]
    p=norm(p)

    return p, {
        "volatility": volmeta,
        "short":short,"longv":longv,"prior":prior,
        "weights":{"REC_W":rec_w,"LONG_W":long_w,"PRIOR_W":prior_w},
        "signals":sdbg,"mult":{"B":mult["B"],"P":mult["P"],"T":mult["T"]}
    }

def recommend(p: List[float]) -> str:
    return CLASS_ORDER[p.index(max(p))]

# -------------------- Kelly（縮放）配注 --------------------
def kelly_fraction(p_win: float, b: float) -> float:
    f = (b*p_win - (1.0 - p_win)) / max(1e-9, b)
    return max(0.0, f)

def stake_amount(bankroll: int, rec: str, p: List[float]) -> Tuple[float, float]:
    p_win = p[CLASS_ORDER.index(rec)]
    b = PAYOUT[rec]
    f = kelly_fraction(p_win, b) * KELLY_SCALE
    f = max(MIN_BET_FRAC, min(MAX_BET_FRAC, f))
    amt = bankroll * f
    return f, amt

# -------------------- CSV --------------------
def append_round_csv(uid:str, history_before:str, label:str):
    if os.getenv("EXPORT_LOGS","1")!="1": return
    try:
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        with open(CSV_PATH,"a",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([uid, int(time.time()), history_before, label])
    except Exception as e:
        log.warning("append csv fail: %s", e)

# -------------------- Big Road PNG --------------------
def build_big_road_cols(seq: List[str]) -> List[Dict]:
    return build_big_road(seq)

def _layout_big_road_points(cols: List[dict], rows:int=6, max_cols:int=60):
    pts=[]; x=0
    for col in cols:
        color=col["color"]; L=col["len"]; t=col["ties"]
        y=0
        for k in range(L):
            if y<rows:
                yy=y; xx=x; y+=1
            else:
                yy=rows-1; x+=1; xx=x
            pts.append((xx,yy,color,(t if k==0 else 0)))
        x+=1
        if x>=max_cols: break
    return pts

def render_big_road_png(seq: List[str], cell:int=28, rows:int=6, max_cols:int=60) -> bytes:
    cols = build_big_road_cols(seq)
    pts = _layout_big_road_points(cols, rows=rows, max_cols=max_cols)
    width = (0 if not pts else (max(p[0] for p in pts)+1)) * cell
    width = max(width, cell)
    height = rows*cell
    img = Image.new("RGB",(width,height),(255,255,255))
    drw = ImageDraw.Draw(img)

    for r in range(rows+1):
        y=r*cell; drw.line([(0,y),(width,y)], fill=(230,230,230))
    for c in range(width//cell+1):
        x=c*cell; drw.line([(x,0),(x,height)], fill=(230,230,230))

    R=int(cell*0.38)
    for (cx,cy,color,ties) in pts:
        x=cx*cell+cell//2; y=cy*cell+cell//2
        outline=(220,0,0) if color=="B" else (0,102,255)
        drw.ellipse([(x-R,y-R),(x+R,y+R)], outline=outline, width=3)
        if ties>0:
            rr=int(R*0.35); tx=x+int(R*0.55); ty=y-int(R*0.55)
            for _ in range(min(3,ties)):
                drw.ellipse([(tx-rr,ty-rr),(tx+rr,ty+rr)], fill=(34,197,94))
                ty-=rr*2+2

    bio=BytesIO(); img.save(bio, format="PNG", optimize=True)
    return bio.getvalue()

# -------------------- REST API --------------------
@app.get("/")
def root(): return "ok", 200

@app.route("/health", methods=["GET", "HEAD"])
@app.route("/health/", methods=["GET", "HEAD"])
def health():
    return {"status": "healthy", "csv": CSV_PATH, "base_dir": BASE}, 200

@app.post("/predict")
def api_predict():
    data=request.get_json(silent=True) or {}
    seq=parse_text_seq(str(data.get("history","")))
    p, _dbg = estimate_probs(seq)
    rec=recommend(p)
    return jsonify({
        "history_len": len(seq),
        "probabilities": {"B":p[0], "P":p[1], "T":p[2]},
        "recommendation": rec
    }), 200

@app.post("/predict_debug")
def predict_debug():
    data = request.get_json(silent=True) or {}
    seq  = parse_text_seq(str(data.get("history", "")))
    p, detail = estimate_probs(seq)
    return jsonify({
        "len": len(seq),
        "counts": {"B": seq.count("B"), "P": seq.count("P"), "T": seq.count("T")},
        "detail": detail,
        "final": {"B":p[0], "P":p[1], "T":p[2]},
        "recommend": CLASS_ORDER[p.index(max(p))]
    }), 200

@app.post("/road")
def api_road():
    data=request.get_json(silent=True) or {}
    seq=parse_text_seq(str(data.get("history","")))
    cols=build_big_road_cols(seq)
    return jsonify({"n_cols":len(cols), "cols":cols, "tail":cols[-5:] if len(cols)>5 else cols}), 200

@app.get("/road/image")
def road_image():
    uid=request.args.get("uid")
    hist_q=request.args.get("history","")
    if uid and uid in USER_HISTORY:
        seq=USER_HISTORY.get(uid,[])
    else:
        seq=parse_text_seq(hist_q)
    png=render_big_road_png(seq)
    return send_file(BytesIO(png), mimetype="image/png", download_name="road.png")

# -------------------- LINE 啟用與事件 --------------------
LINE_TOKEN=os.getenv("LINE_CHANNEL_ACCESS_TOKEN","")
LINE_SECRET=os.getenv("LINE_CHANNEL_SECRET","")
USE_LINE=False
try:
    if LINE_TOKEN and LINE_SECRET:
        from linebot import LineBotApi, WebhookHandler
        from linebot.models import (MessageEvent, TextMessage, TextSendMessage, ImageSendMessage)
        USE_LINE=True
        line_bot_api=LineBotApi(LINE_TOKEN)
        handler=WebhookHandler(LINE_SECRET)
    else:
        line_bot_api=None; handler=None
except Exception as e:
    log.warning("LINE not ready: %s", e)
    line_bot_api=None; handler=None; USE_LINE=False

def reply_or_push(event, messages):
    try:
        line_bot_api.reply_message(event.reply_token, messages)
    except Exception:
        try:
            uid=event.source.user_id
            line_bot_api.push_message(uid, messages)
        except Exception as e2:
            log.error("send msg fail: %s", e2)

@app.post("/line-webhook")
def webhook():
    if not USE_LINE or handler is None:
        return "ok", 200
    sig=request.headers.get("X-Line-Signature","")
    body=request.get_data(as_text=True)
    try:
        handler.handle(body, sig)
    except Exception as e:
        log.error("LINE handle error: %s", e)
    return "ok", 200

# ---- 背景：試用到時主動推播（僅 LINE 啟用時） ----
def _trial_watcher():
    log.info("[TRIAL] watcher started, interval=%ss, minutes=%s", TRIAL_SCAN_INTERVAL_SEC, TRIAL_MINUTES)
    while True:
        try:
            if USE_LINE and line_bot_api is not None:
                now = now_ts()
                for uid in list(ACTIVE_USERS):
                    if not USER_ACTIVATED.get(uid, False):
                        start = USER_TRIAL_START.get(uid, now)
                        expired = (now - start) >= TRIAL_MINUTES*60
                        if expired and not USER_TRIAL_WARNED.get(uid, False):
                            try:
                                line_bot_api.push_message(uid, TextSendMessage(
                                    text="⏰ 試用時間已滿 30 分鐘。\n若要繼續使用，請加管理員 LINE：@jins888 取得開通帳號（5字母+5數字），或直接貼上開通碼解鎖。🔐"
                                ))
                                USER_TRIAL_WARNED[uid] = True
                                _save_state(uid)
                                log.info("[TRIAL] warned uid=%s", uid)
                            except Exception as e:
                                log.warning("[TRIAL] push warn fail uid=%s: %s", uid, e)
            time.sleep(TRIAL_SCAN_INTERVAL_SEC)
        except Exception as e:
            log.error("[TRIAL] watcher loop error: %s", e)
            time.sleep(TRIAL_SCAN_INTERVAL_SEC)

if USE_LINE:
    t = threading.Thread(target=_trial_watcher, daemon=True)
    t.start()

# ---- LINE 文字事件 ----
if USE_LINE and handler is not None:
    @handler.add(MessageEvent, message=TextMessage)
    def on_text(event):
        uid=event.source.user_id
        text=(event.message.text or "").strip()
        ensure_user(uid)

        # 先試開通碼（任何時刻輸入都可解鎖）
        if try_activate(uid, text):
            reply_or_push(event, TextSendMessage(text="✅ 已解鎖，歡迎繼續使用！🔓"))
            return

        # 試用時限檢查（過期就只回引導）
        if not trial_ok(uid):
            reply_or_push(event, TextSendMessage(
                text="⏳ 試用已結束。\n請加管理員 LINE：@jins888 取得開通帳號，或直接貼上你的開通帳號（5字母+5數字）解鎖。🔐"
            ))
            return

        # 初次：需要本金
        if uid not in USER_BANKROLL or USER_BANKROLL.get(uid, 0) <= 0:
            amt = parse_bankroll(text)
            if amt is None:
                reply_or_push(event, TextSendMessage(
                    text="💰 請先告訴我你的本金（例如：5000 或 本金 20000），我會用它計算配注哦！📈"
                ))
                return
            USER_BANKROLL[uid] = amt
            _save_state(uid)
            reply_or_push(event, TextSendMessage(
                text=f"👍 已設定本金：{amt:,} 元。接著貼上歷史（B/P/T 或 莊/閒/和），然後輸入「開始分析」即可！🚀"
            ))
            return

        # 快捷：大路圖
        if text in ("路圖","大路","road","Road"):
            base=os.getenv("BACKEND_URL","").rstrip("/")
            if not base:
                reply_or_push(event, TextSendMessage(text="ℹ️ 尚未設定 BACKEND_URL，因此無法顯示圖片 URL。"))
                return
            url=f"{base}/road/image?uid={uid}"
            reply_or_push(event, ImageSendMessage(original_content_url=url, preview_image_url=url))
            return

        # 控制命令
        if text == "結束分析":
            USER_HISTORY[uid]=[]
            USER_READY[uid]=False
            reply_or_push(event, TextSendMessage(text="🛑 已結束，本金設定保留。要重新開始請先貼歷史，然後輸入「開始分析」。"))
            return

        if text == "返回":
            if USER_HISTORY[uid]:
                USER_HISTORY[uid].pop()
                reply_or_push(event, TextSendMessage(text="↩️ 已返回一步。繼續輸入下一手（莊/閒/和 或 B/P/T）。"))
            else:
                reply_or_push(event, TextSendMessage(text="ℹ️ 沒有可返回的步驟。"))
            return

        seq_add = parse_text_seq(text)

        # 準備期：收歷史、等待開始
        if not USER_READY[uid]:
            if seq_add:
                cur = USER_HISTORY[uid]
                before = "".join(cur)
                cur.extend(seq_add)
                if len(cur) > MAX_HISTORY: cur[:] = cur[-MAX_HISTORY:]
                USER_HISTORY[uid] = cur
                append_round_csv(uid, before, f"+{len(seq_add)}init")
                reply_or_push(event, TextSendMessage(
                    text=f"📝 已接收歷史共 {len(seq_add)} 手，目前累計 {len(USER_HISTORY[uid])} 手。\n輸入「開始分析」即可啟動。🚦"
                ))
                return
            if text == "開始分析":
                if len(USER_HISTORY[uid]) == 0:
                    reply_or_push(event, TextSendMessage(text="📥 請先貼上你的歷史（例如：BPPBBP 或 莊閒閒莊…），再輸入「開始分析」。"))
                    return
                USER_READY[uid] = True
                reply_or_push(event, TextSendMessage(text="✅ 已開始分析。接下來每輸入一手（莊/閒/和 或 B/P/T），我會立刻回覆下一手預測 📊"))
                return

            reply_or_push(event, TextSendMessage(
                text="👋 歡迎！直接貼上你的歷史（B/P/T 或 莊/閒/和），然後輸入「開始分析」。\n也可輸入「路圖」查看當前大路圖。🗺️"
            ))
            return

        # 已開始分析：逐手回覆
        if seq_add:
            for hand in seq_add:
                before = "".join(USER_HISTORY[uid])
                USER_HISTORY[uid].append(hand)
                if len(USER_HISTORY[uid]) > MAX_HISTORY:
                    USER_HISTORY[uid] = USER_HISTORY[uid][-MAX_HISTORY:]
                append_round_csv(uid, before, hand)

            seq = USER_HISTORY[uid]
            t0=time.time()
            p, _detail = estimate_probs(seq)
            rec=recommend(p)
            dt=int((time.time()-t0)*1000)

            bankroll = USER_BANKROLL.get(uid, 0)
            frac, amt = stake_amount(bankroll, rec, p)

            def qamt(f): 
                return format_money(bankroll * f)
            quick_line = f"🧮 10%={qamt(0.10)}｜20%={qamt(0.20)}｜30%={qamt(0.30)}"

            msg = (
                f"📊 已解析 {len(seq)} 手（{dt} ms）\n"
                f"機率：莊 {p[0]:.3f}｜閒 {p[1]:.3f}｜和 {p[2]:.3f}\n"
                f"👉 下一手建議：{LAB_ZH[rec]} 🎯\n"
                f"💵 本金：{bankroll:,}\n"
                f"✅ 建議下注：{format_money(amt)} ＝ {bankroll:,} × {frac*100:.1f}%\n"
                f"{quick_line}\n"
                f"🔁 直接輸入下一手結果（莊/閒/和 或 B/P/T），我會再幫你算下一局。"
            )
            reply_or_push(event, TextSendMessage(text=msg))
            return

        # 非預期文字
        reply_or_push(event, TextSendMessage(
            text="🤔 我看不出有莊/閒/和（或 B/P/T）。\n已開始分析時，請直接輸入當前開出結果即可，例如「莊」或「P」。"
        ))

# -------------------- main --------------------
if __name__=="__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","8000")), debug=False)
