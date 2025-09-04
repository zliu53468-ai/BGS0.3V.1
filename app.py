import os, io, time, math, logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

from flask import Flask, request, jsonify, abort
from PIL import Image
import numpy as np
import cv2

# ===== LINE SDK =====
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, ImageMessage, TextSendMessage, FollowEvent
)

# ===== OCR =====
try:
    import pytesseract
    from pytesseract import Output
except ImportError:
    pytesseract = None

app = Flask(__name__)

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bgs-bot")

# ---------- ENV ----------
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN) if LINE_CHANNEL_ACCESS_TOKEN else None
line_handler = WebhookHandler(LINE_CHANNEL_SECRET) if LINE_CHANNEL_SECRET else None

DEBUG_VISION = os.getenv("DEBUG_VISION", "0") == "1"

# ---------- User session ----------
user_mode: Dict[str, bool] = {}   # user_id -> True/False

# =========================================================
# 6×6表格識別
# =========================================================
def detect_table_structure(image):
    """檢測6×6表格結構"""
    # 轉換為灰度圖
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用自適應閾值進行二值化
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # 檢測水平和垂直線條
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    
    # 應用形態學操作
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    
    # 合併線條
    table_mask = cv2.add(horizontal_lines, vertical_lines)
    
    # 查找輪廓
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 找到最大的矩形輪廓（表格）
    table_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            table_contour = contour
    
    return table_contour

def extract_table_cells(image, table_contour):
    """提取表格中的單元格"""
    # 獲取表格邊界
    x, y, w, h = cv2.boundingRect(table_contour)
    table_roi = image[y:y+h, x:x+w]
    
    # 轉換為灰度圖
    gray = cv2.cvtColor(table_roi, cv2.COLOR_BGR2GRAY)
    
    # 使用自適應閾值進行二值化
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # 檢測水平和垂直線條
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    
    # 合併線條
    lines = cv2.add(horizontal_lines, vertical_lines)
    
    # 查找線條的交點以確定單元格
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 提取單元格
    cells = []
    for contour in contours:
        x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
        if w_c > 20 and h_c > 20:  # 過濾太小的區域
            cells.append((x + x_c, y + y_c, w_c, h_c))
    
    return cells

def recognize_table_content(image, cells):
    """識別表格內容"""
    results = []
    
    # 按位置排序單元格（從左到右，從上到下）
    cells.sort(key=lambda cell: (cell[1] // 50, cell[0] // 50))
    
    for i, (x, y, w, h) in enumerate(cells):
        # 提取單元格區域
        cell_roi = image[y:y+h, x:x+w]
        
        # 預處理以優化OCR
        gray = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2GRAY)
        processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
        
        # 設置Tesseract參數（支持簡體和繁體中文）
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=庄闲和莊閒閑'
        
        # 進行OCR識別
        text = pytesseract.image_to_string(processed, config=custom_config, lang='chi_sim+chi_tra')
        text = text.strip()
        
        # 映射到對應的代碼
        if text in ['莊', '庄']:
            results.append('B')
        elif text in ['閒', '閑', '闲']:
            results.append('P')
        elif text == '和':
            results.append('T')
        else:
            results.append('?')  # 無法識別
            
        logger.info(f"單元格 {i+1}: 位置({x},{y}), 識別結果: '{text}' -> {results[-1]}")
    
    return results

def extract_6x6_table(img_bytes: bytes) -> List[str]:
    """專門用於識別6×6表格的函數"""
    try:
        if pytesseract is None:
            logger.error("pytesseract is not installed")
            return []
        
        # 圖像預處理
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 檢測表格結構
        table_contour = detect_table_structure(img)
        if table_contour is None:
            logger.warning("未檢測到表格結構")
            return []
        
        # 提取表格單元格
        cells = extract_table_cells(img, table_contour)
        if len(cells) != 36:  # 6x6=36個單元格
            logger.warning(f"檢測到 {len(cells)} 個單元格，預期為36個")
            # 仍然繼續處理，但記錄警告
        
        # 識別表格內容
        sequence = recognize_table_content(img, cells)
        
        logger.info(f"表格識別完成: {len(sequence)}個單元格")
        return sequence
        
    except Exception as e:
        logger.error(f"表格識別錯誤: {e}")
        return []

# =========================================================
# 特徵工程 / 預測（保持不變）
# =========================================================
IDX = {"B":0,"P":1,"T":2}

def _streak_tail(seq: List[str]) -> int:
    if not seq: return 0
    t, c = seq[-1], 1
    for i in range(len(seq)-2, -1, -1):
        if seq[i]==t: c+=1
        else: break
    return c

def _transitions(seq: List[str]) -> np.ndarray:
    m = np.zeros((3,3), dtype=np.float32)
    for a,b in zip(seq, seq[1:]):
        if a in IDX and b in IDX:
            m[IDX[a], IDX[b]] += 1
    if m.sum()>0: m = m/(m.sum()+1e-6)
    return m.flatten()

def _ratio_lastN(seq: List[str], N: int) -> Tuple[float,float,float]:
    s = seq[-N:] if len(seq)>=N else seq
    if not s: return (0.33,0.33,0.34)
    n = len(s); return (s.count("B")/n, s.count("P")/n, s.count("T")/n)

def build_features(seq: List[str]) -> np.ndarray:
    n = len(seq)
    pb, pp, pt = _ratio_lastN(seq, n)
    b10,p10,t10 = _ratio_lastN(seq,10)
    b20,p20,t20 = _ratio_lastN(seq,20)
    streak = _streak_tail(seq)
    last = np.zeros(3); last[IDX[seq[-1]]] = 1.0
    trans = _transitions(seq)
    entropy = 0.0
    for v in [pb,pp,pt]:
        if v>1e-9: entropy -= v*math.log(v+1e-9)
    feat = np.array([n,pb,pp,pt,b10,p10,t10,b20,p20,t20,streak,entropy,*last,*trans], dtype=np.float32).reshape(1,-1)
    return feat

def predict_probs_from_seq_rule(seq: List[str]) -> Dict[str,float]:
    n=len(seq)
    if n==0: return {"banker":0.33,"player":0.33,"tie":0.34}
    pb = seq.count("B")/n
    pp = seq.count("P")/n
    pt = max(0.02, seq.count("T")/n*0.6)

    tail=1
    for i in range(n-2,-1,-1):
        if seq[i]==seq[-1]: tail+=1
        else: break
    if seq[-1] in {"B","P"}:
        boost = min(0.08, 0.025*(tail-1))
        if seq[-1]=="B": pb+=boost
        else: pp+=boost

    s=pb+pp+pt
    if s<=0: return {"banker":0.34,"player":0.34,"tie":0.32}
    return {"banker":round(pb/s,4),"player":round(pp/s,4),"tie":round(pt/s,4)}

def betting_plan(pb: float, pp: float, oscillating: bool, alt_streak: int=0) -> Dict[str, Any]:
    diff = abs(pb-pp)
    side = "banker" if pb >= pp else "player"
    side_prob = max(pb, pp)

    ALT_STRICT = int(os.getenv("ALT_STRICT_STREAK","5"))
    if oscillating and alt_streak >= ALT_STRICT:
        return {"side": side, "percent": 0.0, "side_prob": side_prob, "note": "單跳震盪期觀望"}

    if oscillating:
        if diff < 0.12: return {"side": side, "percent": 0.0, "side_prob": side_prob, "note": "震盪期風險高"}
        if diff < 0.18: pct = 0.02
        elif diff < 0.24: pct = 0.04
        else: pct = 0.08
        return {"side": side, "percent": pct, "side_prob": side_prob, "note": "震盪期降倉"}

    if diff < 0.05:
        return {"side": side, "percent": 0.0, "side_prob": side_prob, "note": "差距不足 5%"}
    if diff < 0.08: pct = 0.02
    elif diff < 0.12: pct = 0.04
    elif diff < 0.18: pct = 0.08
    else: pct = 0.12
    return {"side": side, "percent": pct, "side_prob": side_prob}

def render_reply(seq: List[str], probs: Dict[str,float], by_model: bool, info: Dict[str,Any] | None=None) -> str:
    b, p, t = probs["banker"], probs["player"], probs["tie"]
    oscillating = bool(info.get("oscillating")) if info else False
    alt_streak = int(info.get("alt_streak", 0)) if info else 0
    plan = betting_plan(b, p, oscillating, alt_streak)
    
    side = "莊" if plan["side"] == "banker" else "閒"
    win_rate = plan["side_prob"] * 100
    
    reply = f"推薦預測：{side}（勝率{win_rate:.1f}%）\n\n"
    reply += f"解析路數：{len(seq)}手\n"
    reply += f"莊勝率：{b*100:.1f}% | 閒勝率：{p*100:.1f}% | 和局率：{t*100:.1f}%\n"
    
    if plan["percent"] > 0:
        reply += f"建議下注：{plan['percent']*100:.0f}%資金於{side}"
    else:
        reply += "建議：觀望不下注"
    
    if info and info.get("oscillating"):
        reply += f"\n當前牌路震盪中（交替率：{info.get('alt_rate', 0):.2f}）"
    
    return reply

# =========================================================
# API（可自測）
# =========================================================
@app.route("/")
def index():
    return "BGS AI 助手正在運行 ✅ /line-webhook 已就緒", 200

@app.route("/health")
def health():
    return jsonify({
        "status":"ok",
        "ts":int(time.time()),
        "ocr_available": pytesseract is not None
    })

# =========================================================
# LINE Webhook
# =========================================================
@app.route("/line-webhook", methods=['POST'])
def line_webhook():
    if not (line_bot_api and line_handler):
        logger.error("LINE creds missing: ACCESS_TOKEN or SECRET not set")
        abort(403)
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    logger.info(f"/line-webhook called, sig_len={len(signature)}, body_len={len(body)}")
    try:
        line_handler.handle(body, signature)
    except InvalidSignatureError as e:
        logger.exception(f"InvalidSignatureError: {e}. "
                         f"==> 通常是 LINE_CHANNEL_SECRET 不對 或 用錯 Channel 的 Secret/Token")
        return "Invalid signature", 200
    except Exception as e:
        logger.exception(f"Unhandled error while handling webhook: {e}")
        return "Error", 200
    return "OK"

if line_handler and line_bot_api:

    @line_handler.add(FollowEvent)
    def on_follow(event: FollowEvent):
        welcome = (
            "歡迎加入BGS AI 助手 🎉\n\n"
            "輸入「開始分析」後，上傳6×6表格截圖，我會使用表格識別技術自動辨識並回傳建議下注。"
        )
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=welcome))

    @line_handler.add(MessageEvent, message=TextMessage)
    def on_text(event: MessageEvent):
        uid = getattr(event.source, "user_id", "unknown")
        txt = (event.message.text or "").strip()
        if txt in {"開始分析", "開始", "START", "分析"}:
            user_mode[uid] = True
            if pytesseract is None:
                msg = "系統錯誤：OCR功能未啟用，請聯繫管理員安裝Tesseract OCR"
            else:
                msg = "已進入分析模式 ✅\n請上傳6×6表格截圖：我會使用表格識別技術自動辨識並回覆預測建議"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=msg))
            return
        line_bot_api.reply_message(event.reply_token, TextSendMessage(
            text="請先輸入「開始分析」，再上傳表格截圖。"
        ))

    @line_handler.add(MessageEvent, message=ImageMessage)
    def on_image(event: MessageEvent):
        uid = getattr(event.source, "user_id", "unknown")
        if not user_mode.get(uid):
            line_bot_api.reply_message(event.reply_token, TextSendMessage(
                text="尚未啟用分析模式。\n請先輸入「開始分析」，再上傳表格截圖。"
            ))
            return
            
        if pytesseract is None:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(
                text="系統錯誤：OCR功能未啟用，請聯繫管理員安裝Tesseract OCR"
            ))
            return

        content = line_bot_api.get_message_content(event.message.id)
        img_bytes = b"".join(chunk for chunk in content.iter_content())
        seq = extract_6x6_table(img_bytes)
        
        if not seq or len(seq) < 12:  # 至少需要識別12個單元格
            tip = f"表格識別結果不理想 😥 只識別到 {len(seq)} 個單元格\n請確保截圖清晰，表格完整可見。"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=tip))
            return

        probs = predict_probs_from_seq_rule(seq)
        msg = render_reply(seq, probs, False, {})
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=msg))

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
