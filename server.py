# -*- coding: utf-8 -*-
"""
BGS server.py  (直接覆蓋可用)
- Dashboard:        GET /
- Status JSON:      GET /status.json
- Health:           GET /health
- Ingest seed:      POST /ingest-seed  { "text": "B P P T ... 或 莊 閒 和 ..." }
- Train synth:      GET  /synth-train?rows=100000&style=hybrid&tie_rate=0.06&mode=feature&jitter=0.02&with_xgb=0
- Predict:          GET  /predict?seq=BPTBPB&ensemble=light&mc=0
- LINE webhook:     POST /line-webhook   (需設定 LINE_CHANNEL_SECRET / LINE_CHANNEL_ACCESS_TOKEN)
"""
import os
import re
import json
import random
from datetime import datetime
from pathlib import Path
from threading import Thread, Lock
from typing import List, Tuple, Dict

from flask import Flask, request, jsonify, Response

# ==== 路徑與檔案 ====
ROOT = Path(os.getenv("APP_ROOT") or Path(__file__).parent.resolve())
DATA = ROOT / "data"
MODELS = ROOT / "models"
DATA.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

SEED_FILE = DATA / "seed.txt"
LGBM_FILE = MODELS / "lgbm.pkl"
LR_FILE   = MODELS / "lr.pkl"
XGB_FILE  = MODELS / "xgb.pkl"

# ==== 旗標 / 共享狀態 ====
_is_training = False
_last_metrics = None
_lock = Lock()

# ==== 依賴套件 ====
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from joblib import dump, load

# LightGBM
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

# XGBoost（可選）
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

# LINE SDK（若未設環境變數會跳過初始化）
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
line_bot = None
if LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN:
    try:
        from linebot.v3.webhook import WebhookParser
        from linebot.v3.messaging import (
            MessagingApi, Configuration, ApiClient,
            ReplyMessageRequest, TextMessage
        )
        parser = WebhookParser(LINE_CHANNEL_SECRET)
        configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
        api_client = ApiClient(configuration)
        line_bot = MessagingApi(api_client)
    except Exception as e:
        print("[WARN] LINE SDK init failed:", e)
        line_bot = None

app = Flask(__name__)

# ====== 工具 ======
ALIAS = {
    "莊": "B", "閒": "P", "和": "T",
    "b": "B", "p": "P", "t": "T"
}
VALID = {"B", "P", "T"}

def normalize_tokens(txt: str) -> List[str]:
    # 將輸入轉為 B/P/T token list
    t = re.sub(r"[,\u3000\t]+", " ", txt.strip())
    t = re.sub(r"\s+", " ", t)
    out = []
    for w in re.split(r"[ \n\r]+", t):
        w = w.strip()
        if not w:
            continue
        w_up = w.upper()
        if w_up in VALID:
            out.append(w_up)
            continue
        # 中文別名
        if w in ALIAS:
            out.append(ALIAS[w])
            continue
        # 單字串如 "BPPTB"
        if re.fullmatch(r"[BPTbpt]+", w):
            out.extend([ALIAS.get(ch, ch).upper() for ch in w])
            continue
        # 中文連續如 "莊莊閒和"
        if re.fullmatch(r"[莊閒和]+", w):
            out.extend([ALIAS[ch] for ch in w])
            continue
    return out

def save_seed_lines(tokens: List[str]) -> int:
    prev = []
    if SEED_FILE.exists():
        prev = normalize_tokens(SEED_FILE.read_text(encoding="utf-8"))
    merged = prev + tokens
    SEED_FILE.write_text(" ".join(merged), encoding="utf-8")
    return len(merged)

def seed_len() -> int:
    if not SEED_FILE.exists():
        return 0
    return len(normalize_tokens(SEED_FILE.read_text(encoding="utf-8")))

# ====== 特徵工程（簡單穩定版）======
def build_feature_row(seq: List[str], i: int) -> Tuple[List[float], str]:
    """
    以 seq[0:i] 過去視窗做特徵，標籤為 seq[i]。
    特徵：最後1/2/3手 one-hot、B/P/T 數量、近期連莊/連閒、上手是否和。
    """
    y = seq[i]
    past = seq[:i]
    last1 = past[-1] if i >= 1 else "_"
    last2 = past[-2] if i >= 2 else "_"
    last3 = past[-3] if i >= 3 else "_"

    def onehot(v):
        return [1.0 if v == "B" else 0.0, 1.0 if v == "P" else 0.0, 1.0 if v == "T" else 0.0]

    cB = past.count("B"); cP = past.count("P"); cT = past.count("T")
    # 連莊/連閒
    streakB = 0; streakP = 0
    for k in reversed(past):
        if k == "B":
            streakB += 1
            break
        elif k == "P":
            streakP += 1
            break
        else:
            break
    # 上一手是否和
    last_is_T = 1.0 if (i >= 1 and past[-1] == "T") else 0.0

    feats = []
    feats += onehot(last1)
    feats += onehot(last2)
    feats += onehot(last3)
    feats += [cB, cP, cT, streakB, streakP, last_is_T]
    return feats, y

def make_synth_dataset(tokens: List[str], rows: int = 100000,
                       style: str = "hybrid", tie_rate: float = 0.06,
                       jitter: float = 0.02, mode: str = "feature") -> Tuple[pd.DataFrame, pd.Series]:
    """
    style: 'pure' 只用馬可夫/統計產生；'hybrid' 以 seed 為底並混入亂數
    mode:  'feature' 用上面的特徵；'ngram' 以 n-gram 來當 X
    """
    if len(tokens) < 8:
        raise ValueError("seed 太短，至少 8 手以上。")

    # 先構建一條基礎序列 base_seq 長度 >= rows+8
    base_seq: List[str] = tokens[:]
    # 平衡 T 比例
    def rand_next(prev):
        # 簡易馬可夫：看最後一手，偏向交替
        if not prev:
            return random.choices(["B", "P", "T"], [0.47, 0.47, 0.06])[0]
        last = prev[-1]
        if last == "B":
            probs = [0.45, 0.49, tie_rate]
        elif last == "P":
            probs = [0.49, 0.45, tie_rate]
        else:
            probs = [0.49, 0.49, tie_rate]
        return random.choices(["B", "P", "T"], probs)[0]

    while len(base_seq) < rows + 16:
        if style == "pure":
            base_seq.append(rand_next(base_seq))
        else:
            # hybrid: 以真實 seed 為骨幹，穿插擾動
            if random.random() < 0.7:
                base_seq.append(rand_next(base_seq))
            else:
                base_seq.append(random.choice(tokens))

    # jitter：隨機把少量樣本改成 T 或互換 B/P
    if jitter > 0:
        for i in range(len(base_seq)):
            if random.random() < jitter:
                r = random.random()
                if r < tie_rate:
                    base_seq[i] = "T"
                else:
                    base_seq[i] = "B" if base_seq[i] == "P" else "P"

    # 建特徵
    X_rows = []
    y_rows = []
    if mode == "feature":
        for i in range(3, min(len(base_seq) - 1, rows + 3)):
            feats, y = build_feature_row(base_seq, i)
            X_rows.append(feats)
            y_rows.append(y)
        X = pd.DataFrame(X_rows,
                         columns=["l1B","l1P","l1T","l2B","l2P","l2T","l3B","l3P","l3T",
                                  "cB","cP","cT","streakB","streakP","lastT"])
        y = pd.Series(y_rows)
        return X, y
    else:
        # ngram：最近3手 one-hot 當 X
        for i in range(3, min(len(base_seq) - 1, rows + 3)):
            feats, y = build_feature_row(base_seq, i) # 直接沿用
            X_rows.append(feats[:9])  # 只取 last1~3 的 onehot
            y_rows.append(y)
        X = pd.DataFrame(X_rows,
                         columns=["l1B","l1P","l1T","l2B","l2P","l2T","l3B","l3P","l3T"])
        y = pd.Series(y_rows)
        return X, y

def fit_models(X: pd.DataFrame, y: pd.Series, with_xgb: int = 0) -> Dict[str, str]:
    global _last_metrics
    results = {}

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    # LGBM
    if LGBMClassifier is None:
        raise RuntimeError("lightgbm 未安裝")
    lgbm = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1
    )
    lgbm.fit(X_train, y_train)
    dump(lgbm, LGBM_FILE)
    p_val = lgbm.predict_proba(X_val)
    acc = accuracy_score(y_val, np.argmax(p_val, axis=1).astype(object).astype(str))
    results["lgbm_acc"] = f"{acc:.3f}"

    # LR
    lr = LogisticRegression(max_iter=1000, n_jobs=None, multi_class="auto")
    lr.fit(X_train, y_train)
    dump(lr, LR_FILE)
    p2 = lr.predict_proba(X_val)
    acc2 = accuracy_score(y_val, np.argmax(p2, axis=1).astype(object).astype(str))
    results["lr_acc"] = f"{acc2:.3f}"

    # XGB 可選
    if with_xgb and XGBClassifier is not None:
        xgb = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=2
        )
        xgb.fit(X_train, y_train)
        dump(xgb, XGB_FILE)
        p3 = xgb.predict_proba(X_val)
        acc3 = accuracy_score(y_val, np.argmax(p3, axis=1).astype(object).astype(str))
        results["xgb_acc"] = f"{acc3:.3f}"

    _last_metrics = {"when": datetime.utcnow().isoformat() + "Z", **results}
    return results

def model_exists() -> bool:
    return LGBM_FILE.exists() and LR_FILE.exists()

def load_all_models():
    models = {}
    if LGBM_FILE.exists(): models["lgbm"] = load(LGBM_FILE)
    if LR_FILE.exists():   models["lr"]   = load(LR_FILE)
    if XGB_FILE.exists():  models["xgb"]  = load(XGB_FILE)
    return models

def seq_to_features(seq: List[str]) -> List[float]:
    # 轉單筆特徵（對應 feature 模式）
    if len(seq) < 3:
        # 補齊空
        pad = ["_"] * (3 - len(seq)) + seq
        seq = pad[-3:]
    feats, _ = build_feature_row(seq + ["B"], len(seq))  # 假標籤無用
    return feats

def ensemble_predict(seq: List[str], mode: str = "light") -> Dict:
    models = load_all_models()
    if not models:
        raise RuntimeError("尚未訓練，請先 /synth-train")

    x = np.array([seq_to_features(seq)], dtype=float)

    probs_list = []
    names = []
    if "lgbm" in models:
        p = models["lgbm"].predict_proba(x)[0]
        probs_list.append(p); names.append("lgbm")
    if "lr" in models:
        p = models["lr"].predict_proba(x)[0]
        probs_list.append(p); names.append("lr")
    if mode == "full" and "xgb" in models:
        p = models["xgb"].predict_proba(x)[0]
        probs_list.append(p); names.append("xgb")

    if not probs_list:
        raise RuntimeError("沒有可用模型")

    p_avg = np.mean(probs_list, axis=0)
    # 對應類別順序（sklearn 會依 y 的排序決定）
    classes = getattr(models[names[0]], "classes_")
    # 轉成 B/P/T 機率
    map_idx = {c: i for i, c in enumerate(classes)}
    def pick(c):
        return float(p_avg[map_idx[c]]) if c in map_idx else 0.0
    pb = pick("B"); pp = pick("P"); pt = pick("T")
    label = ["B", "P", "T"][int(np.argmax([pb, pp, pt]))]
    return {"probabilities": {"B": pb, "P": pp, "T": pt}, "label": label}

# ====== 背景訓練 ======
def _start_training_async(tokens: List[str], rows: int, style: str,
                          tie_rate: float, mode: str, jitter: float, with_xgb: int):
    global _is_training, _last_metrics
    try:
        with _lock:
            if _is_training: 
                return
            _is_training = True
        X, y = make_synth_dataset(tokens, rows=rows, style=style,
                                  tie_rate=tie_rate, jitter=jitter, mode=mode)
        metrics = fit_models(X, y, with_xgb=with_xgb)
        _last_metrics = {"rows": len(X), **metrics}
    except Exception as e:
        _last_metrics = {"error": str(e)}
    finally:
        with _lock:
            _is_training = False

# ====== Flask Routes ======
@app.get("/health")
def health():
    return Response("OK", mimetype="text/plain; charset=utf-8")

@app.get("/status.json")
def status_json():
    return jsonify({
        "health": "/health",
        "is_training": bool(_is_training),
        "last_metrics": _last_metrics,
        "model_exists": model_exists(),
        "seed_records": seed_len(),
        "webhook": "/line-webhook"
    })

@app.get("/")
def dashboard_html():
    html = """
<!doctype html>
<html lang="zh-Hant"><meta charset="utf-8">
<title>BGS Dashboard</title>
<body style="font-family: system-ui,-apple-system,Segoe UI,Roboto,Arial; padding:20px">
<h1>BGS Dashboard</h1>
<button onclick="refresh()">重新整理狀態</button>
<span>　Webhook：/line-webhook　健康：/health</span>
<pre id="statusBox">{}</pre>

<h3>即時預測 /predict</h3>
<input id="seq" placeholder="例如：BPPTB" style="width:240px">
<button onclick="doPred()">送出</button>
<pre id="predBox"></pre>

<h3>追加種子 /ingest-seed</h3>
<textarea id="seed" placeholder="貼上真實歷史：B P P T B …" style="width:260px;height:80px"></textarea><br>
<button onclick="doSeed()">追加</button>
<pre id="seedBox"></pre>

<h3>啟動訓練 /synth-train</h3>
<label>rows <input id="rows" value="100000" style="width:100px"></label>
<label>style <select id="style"><option>hybrid</option><option>pure</option></select></label>
<label>tie_rate <input id="tie" value="0.06" style="width:60px"></label>
<label>mode <select id="mode"><option>feature</option><option>ngram</option></select></label>
<label>jitter <input id="jit" value="0.02" style="width:60px"></label>
<label>with_xgb <select id="xgb"><option value="0">0</option><option value="1">1</option></select></label>
<button onclick="doTrain()">開始訓練</button>
<pre id="trainBox"></pre>

<script>
async function refresh(){
  try{
    const r = await fetch('/status.json');
    const j = await r.json();
    document.getElementById('statusBox').textContent = JSON.stringify(j,null,2);
  }catch(e){
    document.getElementById('statusBox').textContent = '讀取失敗：'+e;
  }
}
async function doPred(){
  const s = document.getElementById('seq').value.trim();
  const r = await fetch('/predict?seq='+encodeURIComponent(s)+'&ensemble=light&mc=0');
  document.getElementById('predBox').textContent = await r.text();
}
async function doSeed(){
  const s = document.getElementById('seed').value;
  const r = await fetch('/ingest-seed', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text:s})});
  document.getElementById('seedBox').textContent = await r.text();
}
async function doTrain(){
  const q = new URLSearchParams({
    rows: document.getElementById('rows').value,
    style: document.getElementById('style').value,
    tie_rate: document.getElementById('tie').value,
    mode: document.getElementById('mode').value,
    jitter: document.getElementById('jit').value,
    with_xgb: document.getElementById('xgb').value
  }).toString();
  const r = await fetch('/synth-train?'+q);
  document.getElementById('trainBox').textContent = await r.text();
}
refresh();
</script>
</body></html>
"""
    return Response(html, mimetype="text/html; charset=utf-8")

@app.post("/ingest-seed")
def ingest_seed():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        text = payload.get("text", "")
        tokens = normalize_tokens(text)
        if not tokens:
            return jsonify({"ok": False, "err": "no tokens"})
        total = save_seed_lines(tokens)
        msg = f"📝 已接收歷史共 {len(tokens)} 手，目前累計 {total} 手。\\n輸入『開始分析』即可啟動。"
        return jsonify({"ok": True, "added": len(tokens), "total": total, "msg": msg})
    except Exception as e:
        return jsonify({"ok": False, "err": str(e)}), 400

@app.get("/synth-train")
def synth_train():
    rows = int(request.args.get("rows", os.getenv("TRAIN_ROWS", "100000")))
    style = request.args.get("style", "hybrid")
    tie_rate = float(request.args.get("tie_rate", "0.06"))
    mode = request.args.get("mode", os.getenv("TRAIN_MODE", "feature"))
    jitter = float(request.args.get("jitter", "0.02"))
    with_xgb = int(request.args.get("with_xgb", os.getenv("TRAIN_WITH_XGB", "0")))

    tokens = []
    if SEED_FILE.exists():
        tokens = normalize_tokens(SEED_FILE.read_text(encoding="utf-8"))
    if not tokens:
        return jsonify({"ok": False, "err": "請先 /ingest-seed 貼歷史"}), 400

    Thread(target=_start_training_async, args=(tokens, rows, style, tie_rate, mode, jitter, with_xgb), daemon=True).start()
    return jsonify({"ok": True, "msg": "training started", "rows": rows, "style": style, "mode": mode})

@app.get("/predict")
def predict_api():
    seq_txt = request.args.get("seq", "")
    ensemble = request.args.get("ensemble", os.getenv("PRED_ENSEMBLE", "light"))
    mc = int(request.args.get("mc", os.getenv("PRED_MC", "0")))
    tokens = normalize_tokens(seq_txt)
    if not tokens:
        return jsonify({"ok": False, "err": "seq required"}), 400
    if not model_exists():
        return jsonify({"ok": False, "err": "尚未訓練"}), 400
    # Monte Carlo 未開，直接推
    res = ensemble_predict(tokens, mode="full" if ensemble == "full" else "light")
    return jsonify({"ok": True, "source": "ensemble", **res})

# ====== LINE Webhook（簡化、穩定語法）======
def fmt_money(n: int) -> str:
    s = f"{n:,}"
    return s

def reply_text(reply_token: str, text: str):
    if not line_bot:
        return
    try:
        line_bot.reply_message(ReplyMessageRequest(
            reply_token=reply_token,
            messages=[TextMessage(text=text)]
        ))
    except Exception as e:
        print("[LINE] reply error:", e)

@app.post("/line-webhook")
def line_webhook():
    if not line_bot or not parser:
        return "LINE disabled", 200
    body = request.get_data(as_text=True)
    signature = request.headers.get("X-Line-Signature", "")
    try:
        events = parser.parse(body, signature)
    except Exception as e:
        print("[LINE] parse error:", e)
        return "NG", 400
    for ev in events:
        if ev.type != "message": 
            continue
        text_raw = getattr(ev.message, "text", "").strip()
        up = text_raw.upper()

        # 本金
        m = re.search(r"(\d{3,})", up)
        if ("本金" in text_raw) or (m and not normalize_tokens(up)):
            try:
                amt = int(m.group(1)) if m else 5000
                app.config["CAPITAL"] = amt
                reply_text(ev.reply_token, f"👍 已設定本金：{fmt_money(amt)} 元。接著貼上歷史（B/P/T 或 莊/閒/和），然後輸入「開始分析」即可。")
            except Exception:
                reply_text(ev.reply_token, "請輸入數字本金，例如 5000 或 本金 20000")
            continue

        # 開始/結束
        if up in ("開始分析", "START", "GO"):
            reply_text(ev.reply_token, "✅ 已開始分析。直接輸入下一手結果（莊／閒／和 或 B/P/T），我會再幫你算下一局。")
            continue
        if up in ("結束分析", "STOP", "END"):
            reply_text(ev.reply_token, "⛔ 已結束，本金設定保留。要重新開始請先貼歷史，然後輸入「開始分析」。")
            continue

        # 路單 or 預測
        toks = normalize_tokens(text_raw)
        if toks:
            total = save_seed_lines(toks)
            if not model_exists():
                # 若尚未有模型，提醒可訓練
                reply_text(ev.reply_token, f"📝 已接收歷史共 {len(toks)} 手，目前累計 {total} 手。\\n尚未訓練，請到後端按『開始訓練』或輸入 TRAIN。")
            else:
                # 做即時預測
                res = ensemble_predict(toks)
                pB = res["probabilities"]["B"]; pP = res["probabilities"]["P"]; pT = res["probabilities"]["T"]
                top = res["label"]
                capital = int(app.config.get("CAPITAL", 5000))
                # 建議下注比例（示例：min(0.2, top_prob*0.2)）
                top_p = {"B": pB, "P": pP, "T": pT}[top]
                bet_rate = max(0.10, min(0.20, float(top_p)*0.20))
                bet_amt = int(round(capital * bet_rate))
                text = (
                    f"🇮🇹 已解析 {len(toks)} 手 (0 ms)\\n"
                    f"機率：莊 {pB:.3f} ｜ 閒 {pP:.3f} ｜ 和 {pT:.3f}\\n"
                    f"👉 下一手建議：{'莊' if top=='B' else '閒' if top=='P' else '和'} 🎯\\n"
                    f"💰 本金：{fmt_money(capital)}\\n"
                    f"✅ 建議下注：{fmt_money(bet_amt)} = {fmt_money(capital)} × {bet_rate*100:.1f}%\\n"
                    f"🧱 10%={fmt_money(int(capital*0.10))} ｜ 20%={fmt_money(int(capital*0.20))} ｜ 30%={fmt_money(int(capital*0.30))}\\n"
                    f"📨 直接輸入下一手結果（莊／閒／和 或 B/P/T），我會再幫你算下一局。"
                )
                reply_text(ev.reply_token, text)
            continue

        # TRAIN 指令
        if up.startswith("TRAIN"):
            if not SEED_FILE.exists():
                reply_text(ev.reply_token, "請先貼歷史（B/P/T 或 莊/閒/和）再 TRAIN。")
            else:
                tokens = normalize_tokens(SEED_FILE.read_text(encoding="utf-8"))
                Thread(target=_start_training_async, args=(tokens, 100000, "hybrid", 0.06, "feature", 0.02, 0), daemon=True).start()
                reply_text(ev.reply_token, "已啟動訓練（100k, feature, hybrid）。等待約數十秒後即可開始分析。")
            continue

        # STATUS
        if up.startswith("STATUS"):
            js = {
                "is_training": bool(_is_training),
                "model_exists": model_exists(),
                "seed_records": seed_len(),
                "last_metrics": _last_metrics
            }
            reply_text(ev.reply_token, json.dumps(js, ensure_ascii=False, indent=2))
            continue

        # 其他
        reply_text(ev.reply_token, "指令：SEED: <路單>｜TRAIN <rows> [style] [tie] [mode] [jitter]｜STATUS｜PRED <路單>｜LOGIN <code>｜RESET TRIAL")
    return "OK", 200

# ====== 啟動 ======
if __name__ == "__main__":
    # 本地/預覽可直接跑；在 Render 用 gunicorn 啟動時不會進來這段
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT") or os.getenv("LISTEN_PORT") or os.getenv("RENDER_PORT") or "8000")
    app.run(host=host, port=port)
