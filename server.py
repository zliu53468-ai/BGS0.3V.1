# === Flex Menu (3x2 彩色按鈕) ===
def flex_menu():
    from linebot.models import FlexSendMessage  # 動態 import，避免沒裝 SDK 時報錯
    bubble = {
      "type": "bubble",
      "size": "mega",
      "body": {
        "type": "box",
        "layout": "vertical",
        "spacing": "lg",
        "contents": [
          # 標題
          {
            "type": "box",
            "layout": "horizontal",
            "contents": [
              {"type": "text", "text": "🤖 請開始輸入歷史數據", "weight": "bold", "size": "md"}
            ]
          },
          # 說明
          {
            "type": "text",
            "text": "先輸入莊/閒/和；按「開始分析」後才會給出下注建議。",
            "wrap": True,
            "size": "sm",
            "color": "#6B7280"
          },
          # 第一列：莊/閒/和
          {
            "type": "box",
            "layout": "horizontal",
            "spacing": "sm",
            "contents": [
              {
                "type": "button",
                "style": "primary",
                "height": "sm",
                "color": "#EF4444",  # 紅
                "action": {"type": "postback", "label": "莊", "data": "B"}
              },
              {
                "type": "button",
                "style": "primary",
                "height": "sm",
                "color": "#3B82F6",  # 藍
                "action": {"type": "postback", "label": "閒", "data": "P"}
              },
              {
                "type": "button",
                "style": "primary",
                "height": "sm",
                "color": "#22C55E",  # 綠
                "action": {"type": "postback", "label": "和", "data": "T"}
              }
            ]
          },
          # 第二列：開始 / 結束 / 返回（灰）
          {
            "type": "box",
            "layout": "horizontal",
            "spacing": "sm",
            "contents": [
              {
                "type": "button",
                "style": "secondary",
                "height": "sm",
                "color": "#E5E7EB",  # 淺灰底
                "action": {"type": "postback", "label": "開始...", "data": "START"}
              },
              {
                "type": "button",
                "style": "secondary",
                "height": "sm",
                "color": "#E5E7EB",
                "action": {"type": "postback", "label": "結束...", "data": "END"}
              },
              {
                "type": "button",
                "style": "secondary",
                "height": "sm",
                "color": "#E5E7EB",
                "action": {"type": "postback", "label": "返回", "data": "UNDO"}
              }
            ]
          }
        ]
      }
    }
    return FlexSendMessage(alt_text="請開始輸入歷史數據", contents=bubble)
