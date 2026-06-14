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
    return f"✅ 讀取完成\n📊 上局結果: 閒 {player_point} 莊 {banker_point}\n⚡ 開始分析下局...."


def _fmt_bool(v: Any) -> str:
    return "有" if bool(v) else "無"


def prediction_text(pred: Dict[str, Any], ai_text: str = "") -> str:
    entry_allowed = bool(pred.get("entry_allowed", True))
    weak_reason = pred.get("weak_reason", "")

    title = "🔮【預測結果】" if entry_allowed else "⚠️【建議觀察】"
    recommend = pred.get("recommend", "-") if entry_allowed else "觀察一局"

    text = (
        f"{title}\n"
        f"閒：{pred.get('player_prob', 0):.2f}%\n"
        f"莊：{pred.get('banker_prob', 0):.2f}%\n"
        f"差距：{pred.get('confidence_gap', 0):.2f}%\n"
        f"🎯 本次建議：{recommend}"
    )

    if not entry_allowed and weak_reason:
        text += f"\n原因：{weak_reason}"

    # 顯示核心資料來源，方便你確認是否真的吃到 combo / point / pattern。
    text += (
        "\n\n📊【資料層狀態】"
        f"\n點數庫：{_fmt_bool(pred.get('point_available'))} / sample {pred.get('point_sample_size', 0)}"
        f"\n複合庫：{_fmt_bool(pred.get('combo_available'))} / sample {pred.get('combo_sample_size', 0)}"
        f"\n規律庫：{_fmt_bool(pred.get('pattern_available'))} / sample {pred.get('pattern_sample_size', 0)}"
    )

    combo_key = pred.get("combo_feature_key")
    if combo_key:
        text += f"\n複合特徵：{combo_key}"

    pattern_key = pred.get("pattern_feature_key")
    if pattern_key:
        text += f"\n規律特徵：{pattern_key}"

    mc_enabled = pred.get("mc_enabled")
    if mc_enabled:
        text += (
            "\n\n🧪【蒙卡穩定度】"
            f"\n莊：{pred.get('mc_banker_rate', 0):.2f}%"
            f"\n閒：{pred.get('mc_player_rate', 0):.2f}%"
            f"\n方向：{pred.get('mc_recommend', '-')}"
            f"\n差距：{pred.get('mc_gap', 0):.2f}%"
        )

    if ai_text:
        text += f"\n\n{ai_text}"

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
