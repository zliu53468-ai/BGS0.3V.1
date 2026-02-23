# -*- coding: utf-8 -*-
"""server.py — BGS Independent + Stage Overrides + FULL LINE Flow + Compatibility + Cumulative History + Game Setup + PF Fixed (2025-11-03+lock-final-production) """
import os, sys, logging, time, re, json, threading
from typing import Optional, Dict, Any, Tuple, List
import numpy as np

VERSION = "lock-final-production"  # 修正②：明確定義 VERSION，避免 NameError

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

# PATTERN & GRU patch（保持原樣，略過貼出，請保留你原本的這段）

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-server")
np.seterr(all="ignore")

# DEPLETE_OK, Flask, Redis, session 相關函數（保持原樣，略過貼出）

GAMES = {"1": "WM", "2": "PM", "3": "DG", "4": "SA", "5": "KU", "6": "歐博/卡利", "7": "KG", "8": "全利", "9": "名人", "10": "MT真人"}

def flex_history_card():
    from linebot.models import FlexSendMessage
    return FlexSendMessage(
        alt_text="請開始輸入歷史數據",
        contents={
            "type": "bubble",
            "body": {
                "type": "box",
                "layout": "vertical",
                "spacing": "md",
                "contents": [
                    {"type": "text", "text": "🤖 請開始輸入歷史數據", "weight": "bold", "size": "lg"},
                    {"type": "text", "text": "先輸入莊/閒/和；按「開始分析」才會給出下注建議。", "wrap": True, "size": "sm"},
                    {"type": "box", "layout": "horizontal", "spacing": "sm", "contents": [
                        {"type": "button", "style": "primary", "color": "#00C300", "action": {"type": "message", "label": "莊", "text": "B"}},
                        {"type": "button", "style": "primary", "color": "#00C300", "action": {"type": "message", "label": "閒", "text": "P"}},
                        {"type": "button", "style": "primary", "color": "#00C300", "action": {"type": "message", "label": "和", "text": "T"}}
                    ]},
                    {"type": "box", "layout": "horizontal", "spacing": "sm", "contents": [
                        {"type": "button", "style": "secondary", "action": {"type": "message", "label": "開始分析", "text": "開始"}},
                        {"type": "button", "style": "secondary", "action": {"type": "message", "label": "結束分析", "text": "RESET"}},
                        {"type": "button", "style": "secondary", "action": {"type": "message", "label": "遊戲設定", "text": "遊戲設定"}}
                    ]}
                ]
            }
        }
    )

def flex_history_card_with_history(hist: str = ""):
    from linebot.models import FlexSendMessage
    show_text = " ".join(list(hist)) if hist else "（尚未輸入）"
    return FlexSendMessage(
        alt_text="輸入歷史中...",
        contents={
            "type": "bubble",
            "body": {
                "type": "box",
                "layout": "vertical",
                "spacing": "md",
                "contents": [
                    {"type": "text", "text": "🤖 正在輸入歷史數據", "weight": "bold", "size": "lg"},
                    {"type": "text", "text": f"目前歷史：{show_text}", "wrap": True, "size": "md", "color": "#333333"},
                    {"type": "text", "text": "繼續點擊莊/閒/和，或按「開始分析」開始預測", "wrap": True, "size": "sm", "color": "#666666"},
                    {"type": "box", "layout": "horizontal", "spacing": "sm", "contents": [
                        {"type": "button", "style": "primary", "color": "#00C300", "action": {"type": "message", "label": "莊", "text": "B"}},
                        {"type": "button", "style": "primary", "color": "#00C300", "action": {"type": "message", "label": "閒", "text": "P"}},
                        {"type": "button", "style": "primary", "color": "#00C300", "action": {"type": "message", "label": "和", "text": "T"}}
                    ]},
                    {"type": "box", "layout": "horizontal", "spacing": "sm", "contents": [
                        {"type": "button", "style": "secondary", "action": {"type": "message", "label": "開始分析", "text": "開始"}},
                        {"type": "button", "style": "secondary", "action": {"type": "message", "label": "結束分析", "text": "RESET"}},
                        {"type": "button", "style": "secondary", "action": {"type": "message", "label": "遊戲設定", "text": "遊戲設定"}}
                    ]}
                ]
            }
        }
    )

# quick_ 相關、_handle_points_and_predict 等函數保持原樣（請保留你原本的）

try:
    from linebot import LineBotApi, WebhookHandler
    from linebot.models import MessageEvent, TextMessage, FollowEvent, UnfollowEvent, FlexSendMessage, TextSendMessage
    if LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN:
        line_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
        line_handler = WebhookHandler(LINE_CHANNEL_SECRET)

        # on_unfollow 保持原樣（略）

        @line_handler.add(FollowEvent)
        def on_follow(event):
            if not _dedupe_event(_extract_line_event_id(event)): return
            uid = event.source.user_id
            if (not is_premium(uid)) and is_trial_blocked(uid):
                sess = get_session(uid)
                guard_msg = trial_persist_guard(uid)
                msg = guard_msg if guard_msg else f"⛔ 試用已到期\n🔐 請輸入：開通 你的密碼\n👉 正確格式：開通 [密碼]\n📞 沒有密碼？請聯繫：{ADMIN_CONTACT}"
                line_api.reply_message(event.reply_token, FlexSendMessage(alt_text="試用已到期", contents={"type": "bubble", "body": {"type": "box", "layout": "vertical", "contents": [{"type": "text", "text": msg, "wrap": True}]}}))
                save_session(uid, sess)
                return
            now = int(time.time())
            ft_key = _trial_key(uid, "first_ts")
            ex_key = _trial_key(uid, "expired")
            first_ts = _rget(ft_key)
            if not first_ts:
                _rset(ft_key, str(now))
                _rset(ex_key, "0")
                first_ts = str(now)
            else:
                try:
                    first = int(first_ts)
                    used_min = (now - first) // 60
                    if _rget(ex_key) == "1" and used_min < TRIAL_MINUTES:
                        _rset(ex_key, "0")
                except:
                    _rset(ft_key, str(now))
                    _rset(ex_key, "0")
                    first_ts = str(now)
            guard_msg = trial_persist_guard(uid)
            sess = get_session(uid)
            sess["phase"] = "choose_game"
            sess["history_input"] = ""
            try: sess["trial_start"] = int(first_ts) if first_ts else int(time.time())
            except: pass
            if sess.get("premium", False) or is_premium(uid):
                msg = "👋 歡迎回來，已是永久開通用戶。\n請選擇遊戲館別"
            else:
                if guard_msg:
                    msg = guard_msg
                else:
                    try:
                        ft = int(first_ts) if first_ts else int(time.time())
                        used_min = max(0, (int(time.time()) - ft) // 60)
                        left = max(0, TRIAL_MINUTES - used_min)
                    except: left = TRIAL_MINUTES
                    msg = f"👋 歡迎！你有 {left} 分鐘免費試用。\n請選擇遊戲館別"
            line_api.reply_message(
                event.reply_token,
                TextSendMessage(text=
                    "🎰 請選擇遊戲館別\n"
                    "1. WM\n"
                    "2. PM\n"
                    "3. DG\n"
                    "4. SA\n"
                    "5. KU\n"
                    "6. 歐博/卡利\n"
                    "7. KG\n"
                    "8. 全利\n"
                    "9. 名人\n"
                    "10. MT真人\n"
                    "「請直接輸入數字選擇」\n"
                    f"⏳ 試用剩餘 {left} 分鐘（共 {TRIAL_MINUTES} 分鐘）"
                )
            )
            save_session(uid, sess)

        @line_handler.add(MessageEvent, message=TextMessage)
        def on_text(event):
            if not _dedupe_event(_extract_line_event_id(event)): return
            uid = event.source.user_id
            raw = (event.message.text or "")
            text = re.sub(r"\s+", " ", raw.replace("\u3000", " ").strip())
            sess = get_session(uid)
            up = text.upper()

            # 開通邏輯（保持原樣，略）

            guard = trial_persist_guard(uid)
            if guard and not sess.get("premium", False):
                _reply(line_api, event.reply_token, guard)
                save_session(uid, sess)
                return

            # 遊戲館選擇
            if sess.get("phase") == "choose_game":
                if text in GAMES:
                    game = GAMES[text]
                    sess["game"] = game
                    sess["phase"] = "input_bankroll"
                    save_session(uid, sess)
                    _reply(line_api, event.reply_token, f"🎰 已選擇：{game}，請輸入初始籌碼（金額）")
                    return
                else:
                    _reply(line_api, event.reply_token, "⚠️ 請輸入上方數字選擇館別")
                    return

            # 輸入籌碼
            if sess.get("phase") == "input_bankroll":
                if text.isdigit():
                    bankroll = int(text)
                    if bankroll <= 0:
                        _reply(line_api, event.reply_token, "⚠️ 請輸入正整數金額")
                        return
                    sess["bankroll"] = bankroll
                    sess["phase"] = "await_history"
                    sess["history_input"] = ""
                    save_session(uid, sess)
                    _reply(
                        line_api,
                        event.reply_token,
                        f"""✅ 設定完成！館別：{sess.get("game")}
初始籌碼：{bankroll}。

📌 連續模式：現在輸入第一局點數
（例：閒6莊5 / 65 / 和）"""
                    )
                    line_api.reply_message(event.reply_token, flex_history_card())
                    return
                else:
                    _reply(line_api, event.reply_token, "⚠️ 請輸入數字金額")
                    return

            # 遊戲設定按鈕
            if up == "遊戲設定":
                sess["phase"] = "choose_game"
                save_session(uid, sess)
                line_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text=
                        "🎰 請選擇遊戲館別\n"
                        "1. WM\n"
                        "2. PM\n"
                        "3. DG\n"
                        "4. SA\n"
                        "5. KU\n"
                        "6. 歐博/卡利\n"
                        "7. KG\n"
                        "8. 全利\n"
                        "9. 名人\n"
                        "10. MT真人\n"
                        "「請直接輸入數字選擇」"
                    )
                )
                return

            # RESET / 結束分析
            if up in ("結束分析", "清空", "RESET"):
                premium = sess.get("premium", False) or is_premium(uid)
                start_ts = sess.get("trial_start", int(time.time()))
                sess = {
                    "phase": "choose_game",
                    "bankroll": 0,
                    "rounds_seen": 0,
                    "last_pts_text": None,
                    "premium": premium,
                    "trial_start": start_ts,
                    "last_card": None,
                    "last_card_ts": None,
                    "pending": False,
                    "pending_seq": 0,
                    "history_input": "",
                    "game": None,
                }
                try:
                    reset_pf_for_uid(uid)
                    reset_pattern_for_uid(uid)
                    reset_gru_for_uid(uid)
                except: pass
                line_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text=
                        "🎰 請選擇遊戲館別\n"
                        "1. WM\n"
                        "2. PM\n"
                        "3. DG\n"
                        "4. SA\n"
                        "5. KU\n"
                        "6. 歐博/卡利\n"
                        "7. KG\n"
                        "8. 全利\n"
                        "9. 名人\n"
                        "10. MT真人\n"
                        "「請直接輸入數字選擇」"
                    )
                )
                save_session(uid, sess)
                return

            # 必修修正①：完整的 await_history 累積輸入區塊
            if sess.get("phase") == "await_history":
                # 按鈕累積模式
                if up in ("B", "P", "T", "莊", "閒", "和"):
                    if up in ("B", "莊"):
                        c = "B"
                    elif up in ("P", "閒"):
                        c = "P"
                    else:
                        c = "T"
                    hist = sess.get("history_input", "")
                    hist = (hist + c)[-80:]  # 建議修正：限制最大 80 筆
                    sess["history_input"] = hist
                    save_session(uid, sess)
                    line_api.reply_message(event.reply_token, flex_history_card_with_history(hist))
                    return

                # 一次輸入整串歷史（相容舊行為）
                if re.fullmatch(r"[BPTHbpht]{1,80}", text):
                    hist = text.upper()
                    sess["history_input"] = hist
                    save_session(uid, sess)
                    line_api.reply_message(event.reply_token, flex_history_card_with_history(hist))
                    return

                # 開始分析
                if up == "開始":
                    hist = sess.get("history_input", "")
                    if len(hist) == 0:
                        _reply(line_api, event.reply_token, "請先輸入歷史數據")
                        return
                    seq = []
                    for c in hist:
                        if c == "B":
                            seq.append(0)
                        elif c == "P":
                            seq.append(1)
                        else:
                            seq.append(2)
                    try:
                        reset_pf_for_uid(uid)
                        reset_pattern_for_uid(uid)
                        reset_gru_for_uid(uid)
                    except:
                        pass
                    try:
                        pat = get_pattern_for_uid(uid)
                        if pat:
                            pat.load_history(seq)
                        gru = get_gru_for_uid(uid)
                        if gru:
                            gru.load_history(seq)
                    except:
                        pass
                    sess["rounds_seen"] = len(seq)
                    sess["phase"] = "await_pts"
                    save_session(uid, sess)
                    _reply(line_api, event.reply_token, "已開始分析\n請輸入第一局結果\n例如65", quick_predict())
                    return

                # 其他輸入顯示帶歷史的卡片
                line_api.reply_message(event.reply_token, flex_history_card_with_history(sess.get("history_input", "")))
                return

            # await_pts 階段點數處理（已完整）
            pts = parse_last_hand_points(text)
            if pts and sess.get("bankroll", 0) >= 0 and sess.get("phase") == "await_pts":
                p_pts, b_pts = pts
                if p_pts == b_pts:
                    sess["last_pts_text"] = "上局結果: 和局"
                else:
                    sess["last_pts_text"] = f"上局結果: 閒 {p_pts} 莊 {b_pts}"
                probs, choice, bet_amt, reason = _handle_points_and_predict(uid, sess, p_pts, b_pts)
                msg = format_output_card(probs, choice, sess.get("last_pts_text"), bet_amt, cont=True)
                sess["last_card"] = msg
                sess["last_card_ts"] = int(time.time())
                sess["pending"] = False
                save_session(uid, sess)
                _reply(line_api, event.reply_token, msg, quick_predict())
                return

            # 預設回應
            _reply(line_api, event.reply_token, "指令無法辨識。\n請輸入莊/閒/和 或點擊按鈕", quick_predict() if sess.get("phase") == "await_pts" else None)

except Exception as e:
    log.warning("LINE not fully configured: %s", e)

# webhook, health, predict 等保持原樣（略）

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    log.info("Starting %s on port %s", VERSION, port)
    app.run(host="0.0.0.0", port=port, debug=False)
