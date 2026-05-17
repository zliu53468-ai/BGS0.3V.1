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

def prediction_text(pred: Dict[str, Any], ai_text: str = "") -> str:
    text = (
        "🔮【預測結果】\n"
        f"閒：{pred['player_prob']:.2f}%\n"
        f"莊：{pred['banker_prob']:.2f}%\n"
        f"🎯 本次預測結果：{pred['recommend']}"
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
