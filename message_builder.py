from typing import Dict, Any


def game_menu_text() -> str:
    return (
        "🎮【請選擇遊戲館別】\n"
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


def table_connecting_text() -> str:
    return "🔌 連接數據庫中.."


def table_connected_text() -> str:
    return "✅ 連接數據庫完成\n🎯 桌號已設定完成"


def ask_points_text() -> str:
    return "📥 請輸入上局閒莊點數\n例:89  先輸入閒再輸入莊"


def start_text(game: str, table: str) -> str:
    return (
        "🚀 開始分析\n"
        f"🎮 館別：{game}\n"
        f"🎯 桌號：{table}\n"
        "請直接輸入點數，例如：65\n"
        "規則：先閒點，再莊點。"
    )


def end_text() -> str:
    return "🛑 結束分析\n本輪資料已停止讀取。"


def read_done_text(player_point: int, banker_point: int) -> str:
    return (
        f"✅ 讀取完成\n"
        f"📊 上局結果：閒 {player_point} 莊 {banker_point}\n"
        f"⚡ 開始分析下局...."
    )


def _fmt_pct(value: Any, decimals: int = 2) -> str:
    try:
        return f"{float(value):.{decimals}f}%"
    except Exception:
        return "0.00%"


def _scenario_tw(s: str) -> str:
    s = str(s or "UNKNOWN").upper()
    mapping = {
        "NONE_DRAW": "雙方未補",
        "NO_DRAW": "雙方未補",
        "PLAYER_DRAW": "閒補一張",
        "BANKER_DRAW": "莊補一張",
        "BOTH_DRAW": "莊閒皆補",
        "UNKNOWN": "未判定",
    }
    return mapping.get(s, s)


def _debug_block(pred: Dict[str, Any]) -> str:
    point_sample = int(pred.get("point_sample_size", 0) or 0)
    combo_sample = int(pred.get("combo_sample_size", pred.get("pattern_sample_size", 0)) or 0)
    comp_sample = int(pred.get("composition_mc_sample_size", 0) or 0)
    top_scenario = _scenario_tw(pred.get("composition_top_scenario", pred.get("combo_top_scenario", "UNKNOWN")))

    text = (
        "\n\n🤖 條件資料庫比對完成："
        f"\n點數樣本 {point_sample} + 條件樣本 {combo_sample}"
        f"\n補牌情境：{top_scenario}"
        f"\n補牌MC樣本：{comp_sample}"
    )

    road_sample = int(pred.get("road_profile_sample_size", 0) or 0)
    if road_sample > 0 or pred.get("road_profile_available"):
        road_name = pred.get("road_profile_top_zh", "中性路段")
        road_raw = pred.get("raw_layers", {}) if isinstance(pred.get("raw_layers", {}), dict) else {}
        rb = road_raw.get("road_profile_banker_prob")
        rp = road_raw.get("road_profile_player_prob")
        try:
            rb_txt = _fmt_pct(float(rb) * 100)
            rp_txt = _fmt_pct(float(rp) * 100)
            road_prob_txt = f"｜莊 {rb_txt} / 閒 {rp_txt}"
        except Exception:
            road_prob_txt = ""
        text += (
            f"\n牌路資料庫：{road_name}"
            f"\n牌路樣本：{road_sample}{road_prob_txt}"
            f"\n牌路模式：無記憶比對，不延續用戶紀錄"
        )

    mc = pred.get("monte_carlo", {})
    if isinstance(mc, dict) and mc.get("mc_enabled"):
        text += (
            f"\nMC模擬：莊 {_fmt_pct(mc.get('mc_banker_rate', 0))} / "
            f"閒 {_fmt_pct(mc.get('mc_player_rate', 0))}"
        )

    return text


def prediction_text(pred: Dict[str, Any], ai_text: str = "") -> str:
    entry_allowed = bool(pred.get("entry_allowed", True))
    weak_reason = pred.get("weak_reason", "")

    player_prob = float(pred.get("player_prob", 0))
    banker_prob = float(pred.get("banker_prob", 0))
    gap = float(pred.get("confidence_gap", 0))

    if not entry_allowed:
        text = (
            "⚠️【建議觀察】\n"
            f"閒：{player_prob:.2f}%\n"
            f"莊：{banker_prob:.2f}%\n"
            f"差距：{gap:.2f}%\n"
            "🎯 本次建議：觀察一局"
        )
        if weak_reason:
            text += f"\n原因：{weak_reason}"
        text += _debug_block(pred)
        if ai_text:
            text += f"\n\n{ai_text}"
        return text

    text = (
        "🔮【預測結果】\n"
        f"閒：{player_prob:.2f}%\n"
        f"莊：{banker_prob:.2f}%\n"
        f"差距：{gap:.2f}%\n"
        f"🎯 本次預測結果：{pred.get('recommend', '-')}"
    )
    text += _debug_block(pred)
    if ai_text:
        text += f"\n\n{ai_text}"
    return text


def combined_prediction_text(pred: Dict[str, Any], ai_text: str = "", bet_text: str = "") -> str:
    text = prediction_text(pred, ai_text)
    if bet_text:
        text += f"\n\n{bet_text}"
    return text


def observe_bet_text(reason: str = "") -> str:
    text = (
        "💰【本金配注建議】\n"
        "本局建議觀望，不建議下注。"
    )
    if reason:
        text += f"\n原因：{reason}"
    return text


def help_text() -> str:
    return (
        "⚙️ 指令說明\n"
        "遊戲設定：選擇館別與桌號\n"
        "開始分析：啟動分析模式\n"
        "結束分析：停止分析模式\n"
        "65：直接輸入點數，先閒再莊\n"
        "重置：清空本輪狀態"
    )
