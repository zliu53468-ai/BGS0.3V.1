# -*- coding: utf-8 -*-
import os, json, re, time, threading, random, string, math
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

from flask import Flask, request, jsonify, Response

# ---- 可選：LINE Webhook（有設環境變數才啟用） -----------------
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "").strip()
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
LINE_ENABLED = bool(LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN)
if LINE_ENABLED:
    from linebot import LineBotApi, WebhookParser
    from linebot.models import MessageEvent, TextMessage, TextSendMessage
    from linebot.exceptions import InvalidSignatureError
    line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
    parser = WebhookParser(LINE_CHANNEL_SECRET)

# ---- 目錄與檔案 ------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", "/tmp/bgs-data"))
MODEL_DIR = DATA_DIR / "models"
STATE_FILE = DATA_DIR / "state.json"
SEED_FILE = DATA_DIR / "seed.txt"
for p in [DATA_DIR, MODEL_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ---- 訓練設定（預設 5 萬筆） -----------------------------------
AUTO_ROWS   = int(os.getenv("DEFAULT_ROWS", "50000"))
TIE_RATE    = float(os.getenv("DEFAULT_TIE_RATE", "0.06"))
JITTER      = float(os.getenv("DEFAULT_JITTER", "0.02"))
AUTO_RUN    = True  # 開機或新增種子就自動訓練

# ---- 嘗試用 LightGBM；沒有就退回 LogisticRegression -------------
USE_LGBM = True
try:
    import lightgbm as lgb
except Exception:
    USE_LGBM = False
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib

# ---- Flask -----------------------------------------------------
app = Flask(__name__)

# ---- 全域狀態 --------------------------------------------------
_lock = threading.Lock()
_is_training = False
_last_metrics = None
_model_path = MODEL_DIR / "bgs_model.pkl"
_thresholds = dict(B=0.5, P=0.5, T=0.5)  # 下注判斷門檻，可日後外部化

# ---- 工具 ------------------------------------------------------
BP_MAP = {"B":0, "P":1, "T":2, "莊":0, "閒":1, "和":2}
INV_MAP = {0:"B", 1:"P", 2:"T"}

TOKEN_RE = re.compile(r"[BPT莊閒和]+", re.I)

def _read_seed() -> str:
    if SEED_FILE.exists():
        return SEED_FILE.read_text(encoding="utf-8").strip()
    return ""

def _append_seed(s: str):
    s = normalize_seq(s)
    if not s: return
    prev = _read_seed()
    new = (prev + s) if prev else s
    SEED_FILE.write_text(new, encoding="utf-8")

def _save_state():
    state = {
        "is_training": _is_training,
        "model_exists": _model_path.exists(),
        "seed_records": len(_read_seed()),
        "last_metrics": _last_metrics,
        "ts": datetime.utcnow().isoformat()
    }
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state

def normalize_seq(s: str) -> str:
    s = s.strip().upper()
    s = s.replace("莊","B").replace("閒","P").replace("和","T")
    s = re.sub(r"[^BPT]", "", s)
    return s

def markov_sample(seed: str, n: int, tie_rate: float, jitter: float) -> str:
    # 簡單一階轉移機率（從種子估計），若太短則用固定分佈
    seed = normalize_seq(seed)
    if len(seed) < 10:
        base = {"B":0.46, "P":0.48, "T":tie_rate}
        # 正規化
        s = sum(base.values())
        base = {k:v/s for k,v in base.items()}
        out = []
        for _ in range(n):
            r = random.random()
            cum = 0.0
            for k in ["B","P","T"]:
                p = max(0.0, min(1.0, base[k] + random.uniform(-jitter, jitter)))
                cum += p
                if r <= cum:
                    out.append(k); break
        return "".join(out)

    # 估轉移
    trans = {"B":Counter(), "P":Counter(), "T":Counter()}
    for a,b in zip(seed[:-1], seed[1:]):
        trans[a][b] += 1
    out = [seed[-1]]
    for _ in range(n-1):
        row = trans[out[-1]]
        total = sum(row.values())
        if total == 0:  # 落入未知，退回平均
            nxt = random.choice(["B","P","T"])
        else:
            ps = []
            keys = ["B","P","T"]
            for k in keys:
                p = row[k]/total
                if k=="T":
                    p = p*(1-jitter) + tie_rate*(1-jitter)
                else:
                    p = p*(1-jitter) + (1-tie_rate)/2 * jitter
                ps.append(p)
            # normalize
            s = sum(ps)
            ps = [x/s for x in ps]
            r = random.random()
            cum=0.0
            nxt="B"
            for k,p in zip(keys, ps):
                cum += p
                if r<=cum:
                    nxt=k; break
        out.append(nxt)
    return "".join(out)

def build_supervised(seq: str, k: int = 6):
    # 以最近 k 手的 one-hot 當特徵，預測下一手
    X, y = [], []
    arr = [BP_MAP[c] for c in seq]
    for i in range(k, len(arr)):
        ctx = arr[i-k:i]
        feat = np.zeros(3*k, dtype=np.float32)
        for j,v in enumerate(ctx):
            feat[j*3 + v] = 1.0
        X.append(feat)
        y.append(arr[i])
    if not X: return np.zeros((0,3*k)), np.zeros((0,))
    return np.vstack(X), np.array(y)

def train_background(rows: int = AUTO_ROWS, tie_rate: float = TIE_RATE, jitter: float = JITTER):
    global _is_training, _last_metrics
    with _lock:
        if _is_training:
            return
        _is_training = True
        _save_state()

    try:
        seed = _read_seed()
        if not seed or len(seed) < 20:
            _last_metrics = {"error": "seed_too_short"}
            return
        synth = markov_sample(seed, max(rows, 20000), tie_rate, jitter)
        # 用 seed + synth 組合
        combo = seed + synth
        X, y = build_supervised(combo, k=6)
        if len(y) < 100:
            _last_metrics = {"error":"not_enough_samples"}
            return

        # 切 train/valid
        n = len(y)
        idx = int(n*0.85)
        Xtr, Ytr = X[:idx], y[:idx]
        Xva, Yva = X[idx:], y[idx:]

        if USE_LGBM:
            train_set = lgb.Dataset(Xtr, label=Ytr)
            valid_set = lgb.Dataset(Xva, label=Yva, reference=train_set)
            params = dict(
                objective="multiclass",
                num_class=3,
                learning_rate=0.1,
                num_leaves=31,
                min_data_in_leaf=20,
                feature_fraction=0.9,
                bagging_fraction=0.9,
                bagging_freq=1,
                verbose=-1,
            )
            model = lgb.train(
                params,
                train_set,
                valid_sets=[valid_set],
                num_boost_round=200,
                early_stopping_rounds=20,
                verbose_eval=False
            )
            pred = model.predict(Xva)
            acc = float((pred.argmax(axis=1) == Yva).mean())
            joblib.dump({"kind":"lgbm","model":model}, _model_path)
            _last_metrics = {"acc": round(acc,4), "algo":"lgbm", "n_samples": int(n)}
        else:
            clf = LogisticRegression(max_iter=400, n_jobs=None)
            clf.fit(Xtr, Ytr)
            acc = float((clf.predict(Xva) == Yva).mean())
            joblib.dump({"kind":"lr","model":clf}, _model_path)
            _last_metrics = {"acc": round(acc,4), "algo":"logreg", "n_samples": int(n)}
    except Exception as e:
        _last_metrics = {"error": f"{type(e).__name__}: {e}"}
    finally:
        with _lock:
            _is_training = False
            _save_state()

def ensure_model_async():
    # 若沒有模型或剛新增種子，就啟動背景訓練
    if _model_path.exists() or _is_training:
        return
    threading.Thread(target=train_background, daemon=True).start()

def load_model():
    if not _model_path.exists():
        return None
    try:
        return joblib.load(_model_path)
    except Exception:
        return None

def predict_next(seq_recent: str):
    m = load_model()
    if not m:
        return None
    kind = m.get("kind","")
    model = m["model"]
    seq_recent = normalize_seq(seq_recent)
    if len(seq_recent) < 6:
        return None
    X, _ = build_supervised(seq_recent, k=6)
    if X.shape[0] == 0:
        return None
    x = X[-1].reshape(1,-1)
    if kind == "lgbm":
        proba = model.predict(x)[0]
    else:
        proba = model.predict_proba(x)[0]
    # clamp
    proba = np.maximum(1e-6, np.array(proba))
    proba = proba / proba.sum()
    idx = int(proba.argmax())
    return {"B": float(proba[0]), "P": float(proba[1]), "T": float(proba[2]), "suggest": INV_MAP[idx]}

def format_advise(prob, bankroll: int = 5000):
    sug = prob["suggest"]
    b, p, t = prob["B"], prob["P"], prob["T"]
    # very simple bet sizing: 建議金額 = 本金 * (max_prob-0.5)*0.3，限制在 [0, 0.3]
    maxp = max(b,p,t)
    rate = max(0.0, min(0.3, (maxp-0.5)*0.6))
    bet = int(round(bankroll * rate))
    name = {"B":"莊", "P":"閒", "T":"和"}[sug]
    lines = []
    lines.append(f"機率：莊 {b:.3f}｜閒 {p:.3f}｜和 {t:.3f}")
    lines.append(f"👉 下一手建議：{name}")
    lines.append(f"💰 本金：{bankroll:,}")
    lines.append(f"✅ 建議下注：{bet:,}（約 {rate*100:.1f}%）")
    return "\n".join(lines)

# ---- HTTP ------------------------------------------------------
@app.get("/health")
def health():
    return "OK"

@app.get("/")
def home():
    state = _save_state()
    html = f"""
    <h3>BGS Dashboard</h3>
    <pre>{json.dumps(state, ensure_ascii=False, indent=2)}</pre>
    <form action="/predict" method="get">
      <div>即時預測：<input name="seq" placeholder="BPPBT..." /> <button type="submit">送出</button></div>
    </form>
    <form action="/ingest-seed" method="post">
      <div>追加種子：<textarea name="seed" rows="3" cols="40" placeholder="貼上歷史 B/P/T 或 莊/閒/和"></textarea></div>
      <button type="submit">追加</button>
    </form>
    """
    return Response(html, mimetype="text/html; charset=utf-8")

@app.post("/ingest-seed")
def ingest_seed():
    text = (request.form.get("seed") or request.json.get("seed","") if request.is_json else "").strip()
    s = normalize_seq(text)
    if not s:
        return jsonify({"ok": False, "msg":"no tokens"})
    _append_seed(s)
    ensure_model_async()
    return jsonify({"ok": True, "added": len(s), "seed_records": len(_read_seed())})

@app.get("/predict")
def http_predict():
    seq = request.args.get("seq","")
    seq = normalize_seq(seq)
    if not seq:
        return jsonify({"error":"empty"})
    res = predict_next(seq)
    if not res:
        return jsonify({"error":"model_not_ready"})
    return jsonify(res)

# ---- LINE Webhook（無設定就不啟用） ---------------------------
def _extract_tokens(s: str) -> str:
    m = TOKEN_RE.findall(s.upper())
    if not m: return ""
    return normalize_seq("".join(m))

def _detect_bankroll(s: str) -> int:
    # 取第一個整數視為本金
    m = re.search(r"(\d{2,})", s.replace(",",""))
    if not m: return 5000
    try:
        v = int(m.group(1))
        return max(100, min(2_000_000, v))
    except Exception:
        return 5000

if LINE_ENABLED:
    @app.post("/line-webhook")
    def line_webhook():
        signature = request.headers.get("X-Line-Signature", "")
        body = request.get_data(as_text=True)
        try:
            events = parser.parse(body, signature)
        except InvalidSignatureError:
            return "bad sig", 400

        for ev in events:
            if not isinstance(ev, MessageEvent): continue
            if not isinstance(ev.message, TextMessage): continue
            user_text = ev.message.text.strip()
            tokens = _extract_tokens(user_text)

            # 1) 有貼路單：累積 + 自動訓練（若尚未在訓練）
            if len(tokens) >= 1:
                prev_len = len(_read_seed())
                _append_seed(tokens)
                ensure_model_async()
                # 直接嘗試預測（用最近 12 手）
                recent = (_read_seed())[-12:]
                res = predict_next(recent)
                if res:
                    bankroll = _detect_bankroll(user_text)
                    reply = "已更新路單。\n" + format_advise(res, bankroll) + "\n（模型持續優化中）"
                else:
                    reply = f"已接收路單，共 {len(_read_seed())} 手。模型訓練中，完成後會套用最新數據。"
                line_bot_api.reply_message(ev.reply_token, TextSendMessage(reply))
                continue

            # 2) 沒有路單但想看建議：用目前 seed 的最近 12 手
            recent = (_read_seed())[-12:]
            res = predict_next(recent)
            if res:
                bankroll = _detect_bankroll(user_text)
                reply = format_advise(res, bankroll)
            else:
                reply = "目前尚無可用模型或歷史太少，請先貼上幾手路單即可自動開始。"
            line_bot_api.reply_message(ev.reply_token, TextSendMessage(reply))
        return "OK"

# ---- 服務啟動 --------------------------------------------------
@app.before_first_request
def _boot():
    # 啟動時若無模型就自動訓練
    if AUTO_RUN:
        ensure_model_async()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
