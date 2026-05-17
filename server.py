import base64
import hashlib
import hmac
import os
import time
from datetime import datetime, timedelta, timezone
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

# ---------- 自動化成交 / 開通碼 / 封鎖紀錄 ----------
ADMIN_LINE_URL = os.getenv("ADMIN_LINE_URL", "https://lin.ee/xYcGKN0").strip()
TEMP_TRIAL_CODE = os.getenv("TEMP_TRIAL_CODE", "aaa1688002").strip()
TEMP_TRIAL_MINUTES = int(os.getenv("TEMP_TRIAL_MINUTES", "15"))
TEMP_TRIAL_SECONDS = TEMP_TRIAL_MINUTES * 60
MONTHLY_ACTIVATION_CODE = os.getenv("MONTHLY_ACTIVATION_CODE", "aaa1688001").strip()
MONTHLY_DAYS = int(os.getenv("MONTHLY_DAYS", "30"))
PERMANENT_ACTIVATION_CODE = os.getenv("PERMANENT_ACTIVATION_CODE", "aaa1788001").strip()

# 注意：沿用目前專案架構，以下資料暫存在服務記憶體。
# Render 重啟 / 重新部署後會清空；正式長期營運建議再改 Redis / DB。
ACCESS_BY_UID = {}
BLOCKED_HISTORY_UIDS = set()


# ---------- 逆馬丁本金配注 ----------
BET_LEVEL_PCTS = [0.03, 0.07, 0.15]


def _get_attr(obj, name, default=None):
    return getattr(obj, name, default)


def _set_attr(obj, name, value):
    try:
        setattr(obj, name, value)
    except Exception:
        pass


def _normalize_side(value):
    text = str(value or "").strip()
    if "莊" in text or text.upper() in {"B", "BANKER"}:
        return "莊"
    if "閒" in text or "闲" in text or text.upper() in {"P", "PLAYER"}:
        return "閒"
    return None


def _round_bet_amount(amount: float) -> int:
    amount = float(amount or 0)
    if amount <= 0:
        return 0
    return int((amount + 50) // 100 * 100)


def _ensure_betting_fields(sess):
    if _get_attr(sess, "bankroll", None) is None:
        _set_attr(sess, "bankroll", 0)
    if _get_attr(sess, "bet_level", None) is None:
        _set_attr(sess, "bet_level", 0)
    if _get_attr(sess, "last_recommend", None) is None:
        _set_attr(sess, "last_recommend", None)
    if _get_attr(sess, "last_bet_level", None) is None:
        _set_attr(sess, "last_bet_level", 0)


def _settle_betting_level(sess, actual_side: str) -> str:
    _ensure_betting_fields(sess)
    last_recommend = _normalize_side(_get_attr(sess, "last_recommend", None))
    last_bet_level = int(_get_attr(sess, "last_bet_level", 0) or 0)

    if last_recommend not in {"莊", "閒"}:
        _set_attr(sess, "bet_level", 0)
        return "📌 尚無上一局建議紀錄，本局使用第 1 關配注。"

    if actual_side == last_recommend:
        if last_bet_level >= 2:
            _set_attr(sess, "bet_level", 0)
            return "✅ 上局建議命中，第 3 關 15% 已過，下一局自動回到第 1 關 3%。"
        _set_attr(sess, "bet_level", last_bet_level + 1)
        next_pct = BET_LEVEL_PCTS[last_bet_level + 1] * 100
        return f"✅ 上局建議命中，下一局進入第 {last_bet_level + 2} 關，本金 {next_pct:.0f}%。"

    _set_attr(sess, "bet_level", 0)
    return "❌ 上局建議未命中，下一局自動回到第 1 關，本金 3%。"


def _betting_advice_text(sess, recommend: str, settle_note: str = "") -> str:
    _ensure_betting_fields(sess)
    bankroll = int(_get_attr(sess, "bankroll", 0) or 0)
    level = int(_get_attr(sess, "bet_level", 0) or 0)
    level = max(0, min(level, len(BET_LEVEL_PCTS) - 1))
    pct = BET_LEVEL_PCTS[level]
    amount = _round_bet_amount(bankroll * pct)
    side = _normalize_side(recommend)

    lines = []
    if settle_note:
        lines.append(settle_note)
        lines.append("")

    lines.append("💰【本金配注建議】")
    lines.append(f"本金：{bankroll}")
    lines.append(f"目前關卡：第 {level + 1} 關")
    lines.append(f"配注比例：本金 {pct * 100:.0f}%")
    lines.append(f"建議下注金額：約 {amount}")

    if side:
        lines.append(f"下注方向：{side}")
    else:
        lines.append("下注方向：依系統建議判斷")

    lines.append("")
    lines.append("📌 配注規則：")
    lines.append("第 1 關 3% → 命中進第 2 關 7%")
    lines.append("第 2 關 7% → 命中進第 3 關 15%")
    lines.append("第 3 關 15% → 命中後自動回第 1 關 3%")
    lines.append("任一關未命中 → 自動回第 1 關 3%")

    return "\n".join(lines)


def ask_bankroll_text() -> str:
    return (
        "💰【請輸入本金】\n"
        "請輸入你本輪要使用的本金金額。\n\n"
        "例如：10000\n\n"
        "系統會依照逆馬丁配注給出下注金額：\n"
        "第 1 關：本金 3%\n"
        "第 2 關：本金 7%\n"
        "第 3 關：本金 15%\n\n"
        "✅ 第 3 關 15% 命中後，會自動回到第 1 關 3%。\n"
        "❌ 任一關未命中，也會自動回到第 1 關 3%。"
    )


def parse_bankroll(raw: str):
    text = str(raw or "").replace(",", "").replace("，", "").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    value = int(digits)
    if value <= 0:
        return None
    return value


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
        "🔐 如需繼續使用，請聯繫管理員官方 LINE 領取開通碼。\n"
        f"👉 管理員官方 LINE：{ADMIN_LINE_URL}"
    )


def admin_line_notice_text(title: str = "請聯繫管理員官方 LINE") -> str:
    return (
        f"{title}\n"
        "🔐 請聯繫管理員官方 LINE 領取／續用開通碼。\n"
        f"👉 管理員官方 LINE：{ADMIN_LINE_URL}"
    )


def blocked_history_notice_text() -> str:
    return admin_line_notice_text("⛔ 此 LINE UID 有封鎖／取消加入紀錄，需重新領取開通碼後才能使用。")


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


def taipei_time_text(ts: float = None) -> str:
    tz = timezone(timedelta(hours=8))
    if ts is None:
        ts = time.time()
    return datetime.fromtimestamp(ts, tz).strftime("%Y/%m/%d %H:%M:%S")


def normalize_activation_input(raw: str) -> str:
    text = (raw or "").replace("\u3000", " ").strip()
    if text.startswith("開通"):
        text = text[2:].strip()
    elif text.upper().startswith("ACTIVATE"):
        text = text[len("ACTIVATE"):].strip()
    return text.strip().strip("：:").strip()


def set_temp_trial_access(user_id: str) -> str:
    expires_at = time.time() + TEMP_TRIAL_SECONDS
    BLOCKED_HISTORY_UIDS.discard(user_id)
    ACCESS_BY_UID[user_id] = {
        "type": "temp",
        "expires_at": expires_at,
        "code_version": TEMP_TRIAL_CODE,
        "started_at": time.time(),
    }
    return (
        "✅ 臨時開通成功！\n"
        f"⏱️ 此 LINE UID 已獲得 {TEMP_TRIAL_MINUTES} 分鐘臨時試用。\n"
        f"📍 開通時間：{taipei_time_text()}（台北時間）\n"
        f"⏰ 到期時間：{taipei_time_text(expires_at)}（台北時間）\n\n"
        "請輸入「開始分析」重新開始。"
    )


def set_monthly_access(user_id: str) -> str:
    expires_at = time.time() + (MONTHLY_DAYS * 24 * 60 * 60)
    BLOCKED_HISTORY_UIDS.discard(user_id)
    ACCESS_BY_UID[user_id] = {
        "type": "monthly",
        "expires_at": expires_at,
        "code_version": MONTHLY_ACTIVATION_CODE,
        "started_at": time.time(),
    }
    return (
        "✅ 月租開通成功！\n"
        f"👑 此 LINE UID 已認定為月租客戶，有效期限 {MONTHLY_DAYS} 日。\n"
        f"📍 開通時間：{taipei_time_text()}（台北時間）\n"
        f"⏰ 到期時間：{taipei_time_text(expires_at)}（台北時間）\n\n"
        "請輸入「開始分析」重新開始。"
    )


def set_permanent_access(user_id: str) -> str:
    BLOCKED_HISTORY_UIDS.discard(user_id)
    ACCESS_BY_UID[user_id] = {
        "type": "permanent",
        "expires_at": None,
        "code_version": PERMANENT_ACTIVATION_CODE,
        "started_at": time.time(),
    }
    return (
        "✅ 永久開通成功！\n"
        "💎 此 LINE UID 已認定為永久客戶，不受試用時間限制。\n"
        f"📍 開通時間：{taipei_time_text()}（台北時間）\n\n"
        "請輸入「開始分析」重新開始。"
    )


def handle_activation_code(user_id: str, raw: str):
    code = normalize_activation_input(raw)

    if TEMP_TRIAL_CODE and code == TEMP_TRIAL_CODE:
        return set_temp_trial_access(user_id)

    if MONTHLY_ACTIVATION_CODE and code == MONTHLY_ACTIVATION_CODE:
        return set_monthly_access(user_id)

    if PERMANENT_ACTIVATION_CODE and code == PERMANENT_ACTIVATION_CODE:
        return set_permanent_access(user_id)

    return None


def get_access_status(user_id: str):
    info = ACCESS_BY_UID.get(user_id)
    if not info:
        return None, None

    access_type = info.get("type")
    code_version = info.get("code_version", "")

    # 若 Render 環境變數已換開通碼，舊 UID 權限自動失效。
    if access_type == "temp" and code_version != TEMP_TRIAL_CODE:
        ACCESS_BY_UID.pop(user_id, None)
        return None, admin_line_notice_text("⛔ 臨時開通碼已更新，此 UID 需要重新領取開通碼。")

    if access_type == "monthly" and code_version != MONTHLY_ACTIVATION_CODE:
        ACCESS_BY_UID.pop(user_id, None)
        return None, admin_line_notice_text("⛔ 月租開通碼已更新，此 UID 需要重新領取開通碼。")

    if access_type == "permanent" and code_version != PERMANENT_ACTIVATION_CODE:
        ACCESS_BY_UID.pop(user_id, None)
        return None, admin_line_notice_text("⛔ 永久開通碼已更新，此 UID 需要重新領取開通碼。")

    expires_at = info.get("expires_at")
    if expires_at is not None and time.time() >= float(expires_at):
        ACCESS_BY_UID.pop(user_id, None)
        if access_type == "temp":
            return None, admin_line_notice_text("⏰ 15 分鐘臨時試用已到期。")
        if access_type == "monthly":
            return None, admin_line_notice_text("⏰ 月租 30 日權限已到期，如需繼續租用請聯繫管理員官方 LINE。")
        return None, admin_line_notice_text("⏰ 權限已到期。")

    return info, None


def has_active_access(user_id: str) -> bool:
    info, _ = get_access_status(user_id)
    return bool(info)


def access_status_text(user_id: str) -> str:
    info, expired_msg = get_access_status(user_id)
    if expired_msg:
        return expired_msg

    if user_id in BLOCKED_HISTORY_UIDS and not info:
        return blocked_history_notice_text()

    if not info:
        remaining = get_trial_remaining_seconds(user_id)
        if remaining is None:
            return trial_not_started_text()
        if remaining <= 0:
            return trial_expired_text()
        return trial_started_text(remaining)

    access_type = info.get("type")
    expires_at = info.get("expires_at")

    if access_type == "permanent":
        return (
            "💎 權限狀態：永久會員\n"
            "✅ 此 LINE UID 不受試用時間限制。"
        )

    if access_type == "monthly":
        left_seconds = max(0, int(float(expires_at) - time.time()))
        left_days = max(1, left_seconds // 86400)
        return (
            "👑 權限狀態：月租會員\n"
            f"⏳ 約剩 {left_days} 日\n"
            f"⏰ 到期時間：{taipei_time_text(float(expires_at))}（台北時間）"
        )

    if access_type == "temp":
        left_seconds = max(0, int(float(expires_at) - time.time()))
        left_minutes = max(1, left_seconds // 60)
        return (
            "⏱️ 權限狀態：臨時試用\n"
            f"⏳ 約剩 {left_minutes} 分鐘\n"
            f"⏰ 到期時間：{taipei_time_text(float(expires_at))}（台北時間）"
        )

    return "✅ 權限正常"


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
        "version": "server-access-betting-patch-v3",
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
    source = event.get("source", {})
    user_id = source.get("userId") or source.get("groupId") or source.get("roomId") or "anonymous"

    if event.get("type") in {"unfollow", "leave", "memberLeft"}:
        BLOCKED_HISTORY_UIDS.add(user_id)
        ACCESS_BY_UID.pop(user_id, None)
        return

    reply_token = event.get("replyToken")
    if not reply_token:
        return

    if event.get("type") in {"follow", "join", "memberJoined"}:
        if user_id in BLOCKED_HISTORY_UIDS and not has_active_access(user_id):
            reply_messages(reply_token, [text_message(blocked_history_notice_text())])
            return
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
    _ensure_betting_fields(sess)
    raw = text.strip()

    activation_msg = handle_activation_code(user_id, raw)
    if activation_msg:
        reply_messages(reply_token, [text_message(activation_msg)])
        return

    access_info, access_expired_msg = get_access_status(user_id)
    if access_expired_msg:
        reply_messages(reply_token, [text_message(access_expired_msg)])
        return

    if user_id in BLOCKED_HISTORY_UIDS and not access_info:
        reply_messages(reply_token, [text_message(blocked_history_notice_text())])
        return

    if raw in {"help", "幫助", "說明", "指令"}:
        reply_messages(reply_token, [text_message(help_text())])
        return

    if raw in {"試用時間", "剩餘時間", "試用剩餘", "查詢試用", "權限", "權限查詢", "會員狀態"}:
        reply_messages(reply_token, [text_message(access_status_text(user_id))])
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
        sess.phase = "need_bankroll"
        sess.active = False
        reply_messages(reply_token, [
            text_message(f"✅ 已設定遊戲類別【{sess.game}】"),
            text_message(table_connecting_text()),
            text_message(table_connected_text()),
            text_message(ask_bankroll_text()),
        ])
        return

    if sess.phase == "need_builtin_table":
        selected_table = normalize_builtin_table_input(sess.game, raw)

        if selected_table:
            sess.table = selected_table
            sess.phase = "need_bankroll"
            sess.active = False
            reply_messages(reply_token, [
                text_message(table_connecting_text()),
                text_message(table_connected_text()),
                text_message(ask_bankroll_text()),
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
        sess.phase = "need_bankroll"
        sess.active = False
        reply_messages(reply_token, [
            text_message(table_connecting_text()),
            text_message(table_connected_text()),
            text_message(ask_bankroll_text()),
        ])
        return

    if sess.phase == "need_bankroll":
        bankroll = parse_bankroll(raw)
        if bankroll is None:
            reply_messages(reply_token, [
                text_message(
                    "⚠️ 本金格式錯誤\n"
                    "請直接輸入數字，例如：10000"
                )
            ])
            return

        sess.bankroll = bankroll
        sess.bet_level = 0
        sess.last_recommend = None
        sess.last_bet_level = 0
        sess.phase = "idle"
        sess.active = True
        reply_messages(reply_token, [
            text_message(
                f"✅ 本金設定完成：{bankroll}\n"
                "🎯 逆馬丁配注已啟動\n"
                "第 1 關：本金 3%\n"
                "第 2 關：本金 7%\n"
                "第 3 關：本金 15%\n\n"
                "✅ 第 3 關命中後會自動回到第 1 關 3%。\n"
                "❌ 任一關未命中也會自動回到第 1 關 3%。\n\n"
                "📝 請輸入上一局莊閒點數\n"
                "例：89　先輸入閒再輸入莊"
            )
        ])
        return

    if raw in {"結束分析", "結束", "停止分析", "停止"}:
        sess.active = False
        sess.phase = "idle"
        reply_messages(reply_token, [text_message(end_text())])
        return

    if raw in {"重置", "清空", "reset"}:
        store.reset(user_id, keep_setting=True)
        new_sess = store.get(user_id)
        _ensure_betting_fields(new_sess)
        reply_messages(reply_token, [text_message("♻️ 已重置本輪資料\n請輸入「開始分析」重新設定館別與本金。")])
        return

    points = parse_points(raw)
    if points:
        if not access_info:
            if is_trial_expired(user_id):
                reply_messages(reply_token, [text_message(trial_expired_text())])
                return

            trial_first_start = ensure_trial_started(user_id)
            remaining = get_trial_remaining_seconds(user_id) or TRIAL_SECONDS
        else:
            trial_first_start = False
            remaining = TRIAL_SECONDS

        if not int(_get_attr(sess, "bankroll", 0) or 0):
            sess.phase = "need_bankroll"
            reply_messages(reply_token, [text_message(ask_bankroll_text())])
            return

        player_point, banker_point = points
        sess.active = True

        # 先依照用戶輸入的實際結果，結算上一局建議是否命中，再決定下一局配注關卡。
        actual_side = "閒" if player_point > banker_point else ("莊" if banker_point > player_point else "和")
        settle_note = _settle_betting_level(sess, actual_side)

        # 先把本局結果放入臨時rounds，讓pattern能包含最新一局。
        last_result = actual_side
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

        current_level = int(_get_attr(sess, "bet_level", 0) or 0)
        sess.last_recommend = pred.get("recommend")
        sess.last_bet_level = current_level

        messages = []
        if trial_first_start:
            messages.append(text_message(trial_started_text(remaining)))

        messages.extend([
            text_message(read_done_text(player_point, banker_point)),
            text_message(prediction_text(pred, ai_text)),
            text_message(_betting_advice_text(sess, pred.get("recommend"), settle_note)),
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
