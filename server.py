import base64
import hashlib
import hmac
import os
import time
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

from config import LINE_CHANNEL_SECRET, ENABLE_SIGNATURE_VERIFY, GAME_MAP, DEFAULT_GAME, DEFAULT_TABLE
from session_store import store
from predictor import predict
from gemini_helper import explain
from line_api import reply_messages, text_message
from message_builder import (
    game_menu_text,
    table_connecting_text,
    table_connected_text,
    ask_points_text,
    start_text,
    end_text,
    read_done_text,
    prediction_text,
    help_text,
)
from parser_utils import parse_points, looks_like_table_id
from point_db import point_db_meta
from pattern_db import pattern_db_meta

app = Flask(__name__)
CORS(app)


# 每個 UID 第一次正式輸入莊閒點數後，才開始計算試用時間。
# 預設 30 分鐘，可在 Render 環境變數設定 TRIAL_MINUTES 調整。
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "30"))
TRIAL_SECONDS = TRIAL_MINUTES * 60
TRIAL_STARTED_AT = {}


# 只針對 DG / MT真人 內建桌廳選項
# 其餘館別不要求使用者輸入桌號，會直接進入資料庫連接流程
BUILTIN_TABLES = {
    "DG": [
        # 經典百家樂
        "RB01", "RB02", "RB03", "RB04", "RB05", "RB06", "RB07",

        # 特色百家樂
        "S01", "S02", "S03", "S05", "S06", "S07",

        # 區塊百家樂
        "QC01", "QC02", "QC03", "QC05", "QC06", "QC07",
        "QD01", "QD02", "QD03", "QD05", "QD06", "QD07",
    ],
    "MT真人": [
        # MT 百家樂
        "百家樂1", "百家樂2", "百家樂3", "百家樂3A", "百家樂5",
        "百家樂6", "百家樂7", "百家樂8", "百家樂9", "百家樂10",
        "百家樂11", "百家樂12", "百家樂13", "百家樂13A",

        # MT 其他廳
        "龍虎1", "龍虎2", "牛牛1", "殷寶1",
    ],
}


def build_builtin_table_menu(game: str) -> str:
    tables = BUILTIN_TABLES.get(game, [])

    if not tables:
        return "🎯 此館別已自動設定桌廳，準備連接資料庫。"

    lines = [
        f"🎯【請選擇{game}桌廳】",
        "請直接輸入數字或桌號選擇",
        ""
    ]

    for idx, table in enumerate(tables, start=1):
        lines.append(f"{idx}. {table}")

    return "\n".join(lines)


def normalize_builtin_table_input(game: str, raw: str):
    tables = BUILTIN_TABLES.get(game, [])
    text = raw.strip().upper().replace(" ", "")

    if not tables:
        return None

    # 支援輸入數字選擇
    if text.isdigit():
        idx = int(text)
        if 1 <= idx <= len(tables):
            return tables[idx - 1]

    # 支援直接輸入 DG 桌號，例如 RB01 / QC01 / QD05 / S01
    for table in tables:
        normalized = table.upper().replace(" ", "")
        if text == normalized:
            return table

    # 支援 MT 直接輸入 1 / 百家樂1 / 龍虎1 等
    # 例如輸入「百家樂 1」會被轉成「百家樂1」
    raw_no_space = raw.strip().replace(" ", "").upper()
    for table in tables:
        if raw_no_space == table.upper().replace(" ", ""):
            return table

    return None


def welcome_join_text() -> str:
    return (
        "🎉 歡迎加入 AI 百家樂分析助手\n\n"
        "📌【使用須知】\n"
        "1️⃣ 加入後不會立刻計時。\n"
        f"2️⃣ 當你第一次正式輸入莊閒點數後，系統才會開始計算 {TRIAL_MINUTES} 分鐘試用時間。\n"
        "3️⃣ 輸入格式請用：閒點數＋莊點數，例如：65 代表閒 6、莊 5。\n\n"
        "🎯【選房提醒】\n"
        "✅ 房務必選擇非長龍房。\n"
        "✅ 優先選擇莊閒差距比例在 5 局內的房間。\n"
        "✅ 避免連續單邊過長的房間，分析穩定度會更好。\n\n"
        "🚀 準備好後請輸入「開始分析」選擇館別，或直接依照系統提示操作。"
    )


def trial_not_started_text() -> str:
    return (
        "⏱️ 試用尚未開始\n"
        f"當你第一次正式輸入莊閒點數後，才會開始計算 {TRIAL_MINUTES} 分鐘試用時間。"
    )


def trial_started_text(remaining_seconds: int) -> str:
    minutes = max(1, remaining_seconds // 60)
    return (
        "⏱️ 試用已正式開始\n"
        f"本 UID 試用時間為 {TRIAL_MINUTES} 分鐘，目前約剩 {minutes} 分鐘。"
    )


def trial_expired_text() -> str:
    return (
        "⏰ 試用時間已結束\n"
        f"本 UID 的 {TRIAL_MINUTES} 分鐘試用已到期，暫時無法再進行預測分析。\n\n"
        "如需繼續使用，請聯繫官方客服開通正式權限。"
    )


def ensure_trial_started(user_id: str):
    if user_id not in TRIAL_STARTED_AT:
        TRIAL_STARTED_AT[user_id] = time.time()
        return True
    return False


def get_trial_remaining_seconds(user_id: str):
    started_at = TRIAL_STARTED_AT.get(user_id)
    if not started_at:
        return None

    elapsed = time.time() - started_at
    return max(0, int(TRIAL_SECONDS - elapsed))


def is_trial_expired(user_id: str) -> bool:
    remaining = get_trial_remaining_seconds(user_id)
    return remaining is not None and remaining <= 0


def verify_line_signature(body: bytes, signature: str) -> bool:
    if not ENABLE_SIGNATURE_VERIFY:
        return True
    if not LINE_CHANNEL_SECRET:
        return False

    digest = hmac.new(LINE_CHANNEL_SECRET.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, signature)


@app.get("/")
def home():
    return "OK - BGS Dual 3M DB LINE BOT is running"


@app.get("/health")
def health():
    pm = point_db_meta()
    rm = pattern_db_meta()
    return jsonify({
        "ok": True,
        "service": "BGS_DUAL_3M_DB_LINE_BOT",
        "sessions": store.all_count(),
        "mode": "dual_3m_point_and_result_pattern_no_observe",
        "point_db_samples": pm.get("total_simulated_samples"),
        "pattern_db_samples": rm.get("total_simulated_samples"),
    })


@app.post("/api/predict")
def api_predict():
    data = request.get_json(force=True)
    player_point = int(data.get("player_point"))
    banker_point = int(data.get("banker_point"))
    rounds = data.get("rounds", [])
    result = predict(player_point, banker_point, rounds)
    ai_text = explain(result)
    return jsonify({**result, "ai_text": ai_text})


@app.post("/webhook")
def webhook():
    body = request.get_data()
    signature = request.headers.get("X-Line-Signature", "")

    if not verify_line_signature(body, signature):
        abort(400)

    payload = request.get_json(force=True)
    for event in payload.get("events", []):
        handle_event(event)

    return "OK"


def handle_event(event: dict):
    reply_token = event.get("replyToken")
    if not reply_token:
        return

    source = event.get("source", {})
    user_id = source.get("userId") or source.get("groupId") or source.get("roomId") or "anonymous"

    if event.get("type") in {"follow", "join", "memberJoined"}:
        reply_messages(reply_token, [text_message(welcome_join_text())])
        return

    if event.get("type") == "postback":
        data = event.get("postback", {}).get("data", "")
        return handle_text(user_id, reply_token, data)

    if event.get("type") != "message":
        return

    msg = event.get("message", {})
    if msg.get("type") != "text":
        reply_messages(reply_token, [text_message("目前只支援文字輸入點數，例如：65")])
        return

    return handle_text(user_id, reply_token, msg.get("text", "").strip())


def handle_text(user_id: str, reply_token: str, text: str):
    # 這裡仍然是依照每個 user_id 取得自己的 session
    # 不同使用者不會共用 rounds / game / table
    sess = store.get(user_id)
    raw = text.strip()

    if raw in {"help", "幫助", "說明", "指令"}:
        reply_messages(reply_token, [text_message(help_text())])
        return

    if raw in {"試用時間", "剩餘時間", "試用剩餘", "查詢試用"}:
        remaining = get_trial_remaining_seconds(user_id)
        if remaining is None:
            reply_messages(reply_token, [text_message(trial_not_started_text())])
            return
        if remaining <= 0:
            reply_messages(reply_token, [text_message(trial_expired_text())])
            return
        reply_messages(reply_token, [text_message(trial_started_text(remaining))])
        return

    if raw in {"遊戲設定", "設定遊戲", "遊戲館別", "館別設定"}:
        sess.phase = "choose_game"
        sess.active = False
        reply_messages(reply_token, [text_message(game_menu_text())])
        return

    if raw in {"開始分析", "開始", "啟動分析"}:
        sess.phase = "choose_game"
        sess.active = False
        reply_messages(reply_token, [text_message(game_menu_text())])
        return

    if sess.phase == "choose_game" and raw in GAME_MAP:
        sess.game = GAME_MAP[raw]

        # 只有 DG / MT真人 需要進入內建桌廳選擇
        if sess.game in BUILTIN_TABLES:
            sess.phase = "need_builtin_table"
            reply_messages(reply_token, [
                text_message(f"✅ 已設定遊戲類別【{sess.game}】"),
                text_message(build_builtin_table_menu(sess.game)),
            ])
            return

        # 其餘館別不用引導輸入桌廳，直接自動設定並連接
        sess.table = f"{sess.game}_AUTO"
        sess.phase = "idle"
        sess.active = True
        reply_messages(reply_token, [
            text_message(f"✅ 已設定遊戲類別【{sess.game}】"),
            text_message(table_connecting_text()),
            text_message(table_connected_text()),
            text_message(ask_points_text()),
        ])
        return

    if sess.phase == "need_builtin_table":
        selected_table = normalize_builtin_table_input(sess.game, raw)

        if selected_table:
            sess.table = selected_table
            sess.phase = "idle"
            sess.active = True
            reply_messages(reply_token, [
                text_message(table_connecting_text()),
                text_message(table_connected_text()),
                text_message(ask_points_text()),
            ])
            return

        reply_messages(reply_token, [
            text_message(
                "⚠️ 桌廳輸入錯誤\n"
                "請依照清單輸入數字或桌號。\n\n"
                f"{build_builtin_table_menu(sess.game)}"
            )
        ])
        return

    # 保留原本人工桌號輸入邏輯，避免舊流程或特殊情境不能用
    if sess.phase == "need_table" and looks_like_table_id(raw):
        sess.table = raw.upper()
        sess.phase = "idle"
        sess.active = True
        reply_messages(reply_token, [
            text_message(table_connecting_text()),
            text_message(table_connected_text()),
            text_message(ask_points_text()),
        ])
        return

    if raw in {"結束分析", "結束", "停止分析", "停止"}:
        sess.active = False
        sess.phase = "idle"
        reply_messages(reply_token, [text_message(end_text())])
        return

    if raw in {"重置", "清空", "reset"}:
        store.reset(user_id, keep_setting=True)
        reply_messages(reply_token, [text_message("♻️ 已重置本輪資料\n請直接輸入點數，例如：65")])
        return

    points = parse_points(raw)
    if points:
        if is_trial_expired(user_id):
            reply_messages(reply_token, [text_message(trial_expired_text())])
            return

        trial_first_start = ensure_trial_started(user_id)
        remaining = get_trial_remaining_seconds(user_id) or TRIAL_SECONDS

        player_point, banker_point = points
        sess.active = True

        # 先把本局結果放入臨時rounds，讓pattern能包含最新一局。
        last_result = "閒" if player_point > banker_point else ("莊" if banker_point > player_point else "和")
        temp_rounds = sess.rounds + [{
            "player_point": player_point,
            "banker_point": banker_point,
            "last_result": last_result,
        }]

        pred = predict(player_point, banker_point, temp_rounds)
        ai_text = explain(pred)

        round_data = {
            "player_point": player_point,
            "banker_point": banker_point,
            "last_result": pred["last_result"],
            "recommend": pred["recommend"],
            "player_prob": pred["player_prob"],
            "banker_prob": pred["banker_prob"],
        }

        # 只保留最近30局供莊閒pattern查詢；不是用來累計點數權重。
        # 此資料存在目前 user_id 的 session 裡，每位使用者獨立。
        sess.last_round = round_data
        sess.rounds.append(round_data)
        sess.rounds = sess.rounds[-30:]

        messages = []
        if trial_first_start:
            messages.append(text_message(trial_started_text(remaining)))

        messages.extend([
            text_message(read_done_text(player_point, banker_point)),
            text_message(prediction_text(pred, ai_text)),
        ])

        reply_messages(reply_token, messages)
        return

    reply_messages(reply_token, [
        text_message(
            "⚠️ 格式錯誤\n"
            "請直接輸入點數，例如：65\n"
            "規則：先輸入閒，再輸入莊。\n"
            "也可輸入「開始分析」重新設定館別。"
        )
    ])


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
