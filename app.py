import os, io, time, math, logging, re
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
    MessageEvent,
    TextMessage,
    ImageMessage,
    TextSendMessage,
    FollowEvent,
    PostbackEvent,
    PostbackAction,
    FlexSendMessage,
    BubbleContainer,
    BoxComponent,
    ButtonComponent,
    TextComponent,
)

# ===== OCR =====
try:
    import pytesseract
    from pytesseract import Output
except ImportError:
    pytesseract = None

# ===== 机器学习库 =====
ML_AVAILABLE = False
DEEP_LEARNING_AVAILABLE = False
rnn_model = None
label_encoder = None

try:
    from sklearn.preprocessing import LabelEncoder
    ML_AVAILABLE = True
    
    # 尝试导入 TensorFlow
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential, load_model
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.utils import to_categorical
        DEEP_LEARNING_AVAILABLE = True
    except ImportError as e:
        # 如果 TensorFlow 不可用，尝试使用 ONNX Runtime 作为备选
        try:
            import onnxruntime as ort
            DEEP_LEARNING_AVAILABLE = True
        except ImportError:
            pass
            
except ImportError:
    pass

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
# 手動輸入模式狀態：記錄各用戶是否處於手動輸入歷史數據模式
manual_mode: Dict[str, bool] = {}
# 存儲用戶手動輸入的歷史結果序列
user_history_seq: Dict[str, List[str]] = {}

# ---------- RNN模型相关 ----------
def init_rnn_model():
    """初始化RNN模型"""
    global rnn_model, label_encoder
    
    if not DEEP_LEARNING_AVAILABLE:
        logger.warning("深度学习库不可用，无法初始化RNN模型")
        return
    
    try:
        # 创建标签编码器
        label_encoder = LabelEncoder()
        label_encoder.fit(['B', 'P', 'T'])
        
        # 创建简单的RNN模型
        rnn_model = Sequential([
            LSTM(32, input_shape=(10, 3)),  # 减少层大小以节省内存
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax')
        ])
        
        rnn_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("RNN模型初始化成功")
        
        # 尝试加载预训练模型
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'rnn_model.h5')
            if os.path.exists(model_path):
                rnn_model = load_model(model_path)
                logger.info("预训练RNN模型加载成功")
        except Exception as e:
            logger.warning(f"预训练模型加载失败: {e}")
            
    except Exception as e:
        logger.error(f"RNN模型初始化失败: {e}")
        rnn_model = None

# 初始化RNN模型
init_rnn_model()

def make_baccarat_buttons(prompt_text: str = "請選擇莊、閒或和：",
                          title_text: str = "下注選擇") -> FlexSendMessage:
    """
    產生一個包含莊、閒、和按鈕的 Flex 訊息。每個按鈕使用不同顏色以便識別：
    - 莊（紅色）
    - 閒（藍色）
    - 和（綠色）

    參數：
        prompt_text: 按鈕上方提示文字，預設為「請選擇莊、閒或和：」
        title_text: Flex 氣泡的標題文字，預設為「下注選擇」

    回傳值：
        FlexSendMessage，可用於 push 或 reply。
    """
    # 建立按鈕：使用 primary 風格並指定顏色
    buttons = [
        ButtonComponent(
            action=PostbackAction(label="莊", data="choice=banker"),
            style="primary",
            color="#E53935",  # 紅色
            height="sm",
            flex=1,
        ),
        ButtonComponent(
            action=PostbackAction(label="閒", data="choice=player"),
            style="primary",
            color="#1E88E5",  # 藍色
            height="sm",
            flex=1,
        ),
        ButtonComponent(
            action=PostbackAction(label="和", data="choice=tie"),
            style="primary",
            color="#43A047",  # 綠色
            height="sm",
            flex=1,
        ),
    ]

    bubble = BubbleContainer(
        size="mega",
        header=BoxComponent(
            layout="vertical",
            contents=[
                TextComponent(text=title_text, weight="bold", size="lg", align="center")
            ],
        ),
        body=BoxComponent(
            layout="vertical",
            contents=[
                TextComponent(text=prompt_text, size="md")
            ],
        ),
        footer=BoxComponent(
            layout="horizontal",
            spacing="sm",
            contents=buttons,
        ),
    )
    return FlexSendMessage(alt_text=title_text, contents=bubble)

def send_manual_prompt(uid: str, reply_token: str | None = None) -> None:
    """
    發送手動輸入歷史數據的提示訊息與按鈕。
    此函式會將使用者的 manual_mode 設為 True，並清空其歷史序列。

    參數：
        uid: 使用者 ID
        reply_token: 若提供，使用 reply_message 回覆；否則使用 push_message
    """
    # 啟用手動模式並重置歷史序列
    manual_mode[uid] = True
    user_history_seq[uid] = []

    # 建立提示氣泡
    flex_msg = make_baccarat_buttons(
        prompt_text="請點擊下方按鈕依序輸入過往莊/閒/和結果：",
        title_text="🤖請開始輸入歷史數據"
    )

    try:
        if reply_token:
            # 使用 reply_message 回覆
            line_bot_api.reply_message(reply_token, [flex_msg])
        else:
            # 使用 push_message 發送
            line_bot_api.push_message(uid, flex_msg)
    except Exception as e:
        logger.exception(f"發送手動輸入提示時發生錯誤: {e}")

# =========================================================
# 博彩游戏结果识别 - 改进版本
# =========================================================
def extract_gaming_result(img_bytes: bytes) -> List[str]:
    """专门用于识别博彩游戏结果的函数 - 改进版本"""
    try:
        if pytesseract is None:
            logger.error("pytesseract is not installed")
            return []
        
        # 图像预处理
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)
        
        # 转换为OpenCV格式
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # 调整图像大小以提高OCR精度
        height, width = img_cv.shape[:2]
        scale = 2.0  # 放大倍数
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_img = cv2.resize(img_cv, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # 转换为灰度图
        gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        
        # 增强对比度
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        
        # 使用自适应阈值进行二值化
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # 尝试多种PSM模式
        results = []
        for psm in [6, 7, 8, 13]:
            custom_config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=庄闲和莊閒閑田0123456789对总'
            text = pytesseract.image_to_string(binary, config=custom_config, lang='chi_sim+chi_tra')
            if text.strip():
                results.append(text)
                logger.info(f"PSM {psm} 识别结果: {text}")
        
        # 选择最可能的结果
        best_text = ""
        for text in results:
            if "最新好路" in text or "好路" in text:
                best_text = text
                break
            if any(keyword in text for keyword in ["田", "莊", "庄", "閒", "閑", "闲", "和"]):
                best_text = text
                break
        
        if not best_text and results:
            best_text = results[0]
        
        logger.info(f"最终OCR识别结果: {best_text}")
        
        # 解析结果
        return parse_gaming_text(best_text)
        
    except Exception as e:
        logger.error(f"博彩结果识别错误: {e}")
        return []

def parse_gaming_text(text: str) -> List[str]:
    """解析博彩游戏结果文本 - 改进版本"""
    # 查找"最新好路"或类似关键词
    lines = text.split('\n')
    gaming_line = None
    
    # 首先查找包含"最新好路"的行
    for line in lines:
        if '最新好路' in line:
            gaming_line = line
            break
    
    # 如果没有找到，查找包含数字和庄闲字符的行
    if not gaming_line:
        for line in lines:
            if (any(char in line for char in ['庄', '閒', '閑', '闲', '和', '田']) and 
                any(char.isdigit() for char in line)):
                gaming_line = line
                break
    
    if not gaming_line:
        logger.warning("未找到游戏结果行")
        return []
    
    logger.info(f"找到游戏结果行: {gaming_line}")
    
    # 提取庄、闲、和的数量
    banker_count = 0
    player_count = 0
    tie_count = 0
    
    # 使用正则表达式提取数字 - 更灵活的模式
    patterns = [
        r'(田|莊|庄)[^\d]*(\d+)',    # 田 15 或 庄15
        r'(閒|閑|闲)[^\d]*(\d+)',    # 閒 17 或 闲17
        r'(和)[^\d]*(\d+)',          # 和 1
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, gaming_line)
        for match in matches:
            count = int(match[1])
            if match[0] in ['田', '莊', '庄']:
                banker_count = count
            elif match[0] in ['閒', '閑', '闲']:
                player_count = count
            elif match[0] == '和':
                tie_count = count
    
    # 如果正则没有匹配到，尝试更简单的方法
    if banker_count == 0 and player_count == 0 and tie_count == 0:
        parts = re.split(r'\s+', gaming_line)
        for i, part in enumerate(parts):
            if part in ['田', '莊', '庄'] and i+1 < len(parts) and parts[i+1].isdigit():
                banker_count = int(parts[i+1])
            elif part in ['閒', '閑', '闲'] and i+1 < len(parts) and parts[i+1].isdigit():
                player_count = int(parts[i+1])
            elif part == '和' and i+1 < len(parts) and parts[i+1].isdigit():
                tie_count = int(parts[i+1])
    
    logger.info(f"解析结果: 庄={banker_count}, 闲={player_count}, 和={tie_count}")
    
    # 生成序列 (假设最后几局的结果)
    sequence = []
    
    # 添加庄的结果
    for _ in range(banker_count):
        sequence.append('B')
    
    # 添加闲的结果
    for _ in range(player_count):
        sequence.append('P')
    
    # 添加和的结果
    for _ in range(tie_count):
        sequence.append('T')
    
    # 只返回最后36个结果（假设这是最近的结果）
    return sequence[-36:] if len(sequence) > 36 else sequence

# =========================================================
# 特征工程 / 预测
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

def prepare_rnn_data(seq: List[str], seq_length=10):
    """准备RNN输入数据"""
    if not seq or len(seq) < seq_length:
        return None
    
    # 将序列转换为one-hot编码
    encoded = label_encoder.transform(seq)
    one_hot = to_categorical(encoded, num_classes=3)
    
    # 创建滑动窗口数据
    X = []
    for i in range(len(one_hot) - seq_length):
        X.append(one_hot[i:i+seq_length])
    
    return np.array(X)

def predict_with_rnn(seq: List[str]) -> Dict[str, float]:
    """使用RNN模型进行预测"""
    if not DEEP_LEARNING_AVAILABLE or rnn_model is None:
        logger.warning("RNN模型不可用，使用规则预测")
        return predict_probs_from_seq_rule(seq)
    
    # 准备数据
    X = prepare_rnn_data(seq)
    if X is None or len(X) == 0:
        logger.warning("数据不足，使用规则预测")
        return predict_probs_from_seq_rule(seq)
    
    # 使用RNN预测
    predictions = rnn_model.predict(X[-1:])  # 只预测最后一个窗口
    probs = predictions[0]
    
    return {
        "banker": float(probs[0]),
        "player": float(probs[1]),
        "tie": float(probs[2])
    }

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

def predict_probs_with_tie_adjustment(seq: List[str]) -> Dict[str,float]:
    """考虑和局变数的预测函数"""
    n=len(seq)
    if n==0: return {"banker":0.33,"player":0.33,"tie":0.34}
    
    # 计算基本概率
    pb = seq.count("B")/n
    pp = seq.count("P")/n
    pt = seq.count("T")/n
    
    # 考虑和局的影响 - 和局后通常会有趋势延续
    tie_indices = [i for i, result in enumerate(seq) if result == "T"]
    post_tie_results = []
    
    for idx in tie_indices:
        if idx + 1 < len(seq):
            post_tie_results.append(seq[idx + 1])
    
    if post_tie_results:
        post_tie_b = post_tie_results.count("B") / len(post_tie_results)
        post_tie_p = post_tie_results.count("P") / len(post_tie_results)
        
        # 调整概率，考虑和局后的趋势
        pb = (pb * 0.7) + (post_tie_b * 0.3)
        pp = (pp * 0.7) + (post_tie_p * 0.3)
    
    # 考虑连庄/连闲的影响
    tail=1
    for i in range(n-2,-1,-1):
        if seq[i]==seq[-1]: tail+=1
        else: break
    
    if seq[-1] in {"B","P"}:
        boost = min(0.10, 0.03*(tail-1))
        if seq[-1]=="B": pb+=boost
        else: pp+=boost
    
    # 确保概率合理
    pt = max(0.02, min(0.15, pt))  # 和局概率限制在2%-15%之间
    
    # 归一化
    s=pb+pp+pt
    if s<=0: return {"banker":0.34,"player":0.34,"tie":0.32}
    return {"banker":round(pb/s,4),"player":round(pp/s,4),"tie":round(pt/s,4)}

def ensemble_prediction(seq: List[str]) -> Dict[str, float]:
    """集成预测：结合规则和RNN模型"""
    # 获取规则预测
    rule_probs = predict_probs_with_tie_adjustment(seq)
    
    # 获取RNN预测
    rnn_probs = predict_with_rnn(seq)
    
    # 获取权重配置
    weights_str = os.getenv("ENSEMBLE_WEIGHTS", "rule:0.7,rnn:0.3")
    weights = {}
    for part in weights_str.split(','):
        name, weight = part.split(':')
        weights[name] = float(weight)
    
    # 计算加权平均
    ensemble_probs = {
        "banker": weights.get("rule", 0.7) * rule_probs["banker"] + weights.get("rnn", 0.3) * rnn_probs["banker"],
        "player": weights.get("rule", 0.7) * rule_probs["player"] + weights.get("rnn", 0.3) * rnn_probs["player"],
        "tie": weights.get("rule", 0.7) * rule_probs["tie"] + weights.get("rnn", 0.3) * rnn_probs["tie"]
    }
    
    # 归一化
    total = sum(ensemble_probs.values())
    for key in ensemble_probs:
        ensemble_probs[key] /= total
    
    return ensemble_probs

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
    
    # 將輸出文字轉換為繁體中文
    side = "莊" if plan["side"] == "banker" else "閒"
    win_rate = plan["side_prob"] * 100

    reply = f"推薦預測：{side}（勝率{win_rate:.1f}%）\n\n"
    reply += f"解析路數：{len(seq)}手\n"
    reply += f"莊勝率：{b*100:.1f}% | 閒勝率：{p*100:.1f}% | 和局率：{t*100:.1f}%\n"

    if by_model:
        reply += "預測方法：RNN深度學習模型\n"
    else:
        reply += "預測方法：規則引擎\n"

    if plan["percent"] > 0:
        reply += f"建議下注：{plan['percent']*100:.0f}%資金於{side}"
    else:
        reply += "建議：觀望不下注"

    if info and info.get("oscillating"):
        reply += f"\n當前牌路震盪中（交替率：{info.get('alt_rate', 0):.2f}）"

    return reply

# =========================================================
# 调试API
# =========================================================
@app.route("/debug-ocr", methods=['POST'])
def debug_ocr():
    """调试OCR识别"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        img_bytes = file.read()
        
        # 图像预处理
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)
        
        # 转换为OpenCV格式
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # 调整图像大小以提高OCR精度
        height, width = img_cv.shape[:2]
        scale = 2.0  # 放大倍数
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_img = cv2.resize(img_cv, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # 转换为灰度图
        gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        
        # 增强对比度
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        
        # 使用自适应阈值进行二值化
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # 尝试多种PSM模式
        results = {}
        for psm in [6, 7, 8, 13]:
            custom_config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=庄闲和莊閒閑田0123456789对总'
            text = pytesseract.image_to_string(binary, config=custom_config, lang='chi_sim+chi_tra')
            results[f"psm_{psm}"] = text
        
        # 解析结果
        parsed_results = {}
        for psm, text in results.items():
            parsed_results[psm] = parse_gaming_text(text)
        
        return jsonify({
            "original_results": results,
            "parsed_results": parsed_results
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================================================
# API
# =========================================================
@app.route("/")
def index():
    # 返回健康提示，使用繁體中文
    return "BGS AI 助手正在運行 ✅ /line-webhook 已就緒", 200

@app.route("/health")
def health():
    return jsonify({
        "status":"ok",
        "ts":int(time.time()),
        "ocr_available": pytesseract is not None,
        "ml_available": ML_AVAILABLE,
        "rnn_available": rnn_model is not None
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
        # 中文提示使用繁體字
        logger.exception(
            f"InvalidSignatureError: {e}. "
            f"==> 通常是 LINE_CHANNEL_SECRET 不對 或 用錯 Channel 的 Secret/Token"
        )
        return "Invalid signature", 200
    except Exception as e:
        logger.exception(f"Unhandled error while handling webhook: {e}")
        return "Error", 200
    return "OK"

if line_handler and line_bot_api:

    @line_handler.add(FollowEvent)
    def on_follow(event: FollowEvent):
        """處理使用者加入好友的事件。顯示歡迎訊息並引導輸入歷史數據。"""
        uid = getattr(event.source, "user_id", "unknown")
        # 歡迎文字
        welcome = (
            "歡迎加入BGS AI 助手 🎉\n\n"
            "請先依序輸入過往莊/閒/和結果，我會在收到一定數量後開始給出下注建議。\n"
            "如果要上傳截圖進行分析，可輸入「開始分析」。\n"
            "完成本輪分析後，可輸入「結束分析」重置資料並重新開始。"
        )
        # 啟動手動模式並清空歷史
        manual_mode[uid] = True
        user_history_seq[uid] = []
        # 建立輸入按鈕
        flex_msg = make_baccarat_buttons(
            prompt_text="請點擊下方按鈕依序輸入過往莊/閒/和結果：",
            title_text="🤖請開始輸入歷史數據"
        )
        # 同時回覆歡迎文字與 Flex 訊息
        line_bot_api.reply_message(event.reply_token, [TextSendMessage(text=welcome), flex_msg])

    @line_handler.add(MessageEvent, message=TextMessage)
    def on_text(event: MessageEvent):
        uid = getattr(event.source, "user_id", "unknown")
        txt = (event.message.text or "").strip()
        # 若使用者輸入結束分析 / 結束分析
        if txt in {"結束分析", "结束分析"}:
            # 如果處於手動輸入模式，清空資料並重新提示
            if manual_mode.get(uid):
                # 清除歷史數據
                manual_mode[uid] = False
                user_history_seq[uid] = []
                # 發送提示
                msg = TextSendMessage(text="已結束本輪分析。所有歷史數據已刪除。請使用下方按鈕重新輸入歷史數據。")
                flex_msg = make_baccarat_buttons(
                    prompt_text="請點擊下方按鈕依序輸入過往莊/閒/和結果：",
                    title_text="🤖請開始輸入歷史數據"
                )
                # 重啟手動模式
                manual_mode[uid] = True
                user_history_seq[uid] = []
                line_bot_api.reply_message(event.reply_token, [msg, flex_msg])
                return
            else:
                # 非手動模式僅提示
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text="當前未處於分析模式。若要重新開始，請輸入「開始分析」或使用按鈕輸入歷史數據。")
                )
                return

        # 支援簡繁體關鍵字："開始分析"、"开始分析"、"開始"、"开始"、"START"、"分析"
        if txt in {"開始分析", "开始分析", "開始", "开始", "START", "分析"}:
            # 啟動截圖分析模式
            user_mode[uid] = True
            # 關閉手動輸入模式
            manual_mode[uid] = False
            if pytesseract is None:
                msg = "系統錯誤：OCR功能未啟用，請聯繫管理員安裝Tesseract OCR"
            else:
                msg = "已進入截圖分析模式 ✅\n請上傳博彩遊戲截圖：我會使用OCR技術自動辨識並回覆預測建議"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=msg))
            return

        # 其他文字處理
        if manual_mode.get(uid):
            # 手動模式下，提示使用按鈕
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(
                    text="請使用下方按鈕輸入莊/閒/和，或輸入「結束分析」結束本輪分析。"
                ),
            )
            return
        # 如果不是手動模式，提示開始或上傳
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(
                text="請輸入「開始分析」進入截圖模式，或使用按鈕輸入歷史數據開始分析。"
            ),
        )

    @line_handler.add(MessageEvent, message=ImageMessage)
    def on_image(event: MessageEvent):
        uid = getattr(event.source, "user_id", "unknown")
        # 若目前處於手動模式，提醒使用按鈕
        if manual_mode.get(uid):
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(
                    text="您目前處於手動輸入模式，請使用下方按鈕輸入莊/閒/和，或輸入「結束分析」結束本輪分析。"
                ),
            )
            return

        if not user_mode.get(uid):
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(
                    text="尚未啟用截圖分析模式。\n請輸入「開始分析」後再上傳遊戲截圖。"
                ),
            )
            return
            
        if pytesseract is None:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(
                    text="系統錯誤：OCR功能未啟用，請聯繫管理員安裝Tesseract OCR"
                ),
            )
            return

        content = line_bot_api.get_message_content(event.message.id)
        img_bytes = b"".join(chunk for chunk in content.iter_content())
        seq = extract_gaming_result(img_bytes)
        
        if not seq or len(seq) < 5:  # 至少需要識別5個結果
            tip = (
                f"識別結果不理想 😥 只識別到 {len(seq)} 個結果\n"
                f"請確保截圖清晰，包含「最新好路」區域。"
            )
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=tip))
            return

        # 使用集成預測（結合規則和 RNN）
        probs = ensemble_prediction(seq)
        # 判斷是否使用 RNN 模型
        using_rnn = DEEP_LEARNING_AVAILABLE and rnn_model is not None and len(seq) >= 10
        # 生成回覆訊息（繁體中文）
        msg = render_reply(seq, probs, using_rnn, {})
        # 回覆分析結果文字
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=msg))
        # 之後推送下注按鈕給用戶（預設標題及提示）
        try:
            flex_msg = make_baccarat_buttons()
            line_bot_api.push_message(uid, flex_msg)
        except Exception as e:
            logger.exception(f"無法發送下注按鈕: {e}")

    @line_handler.add(PostbackEvent)
    def handle_postback(event: PostbackEvent):
        """
        監聽使用者按下 Flex 按鈕後送出的回傳事件。

        按鈕的 data 格式為 'choice=banker'、'choice=player' 或 'choice=tie'。根據這些鍵值
        轉換為相應的單字序列，並使用簡單的規則引擎進行一次預測回覆。
        """
        try:
            uid = getattr(event.source, "user_id", "unknown")
            data = event.postback.data or ""
            # 將回傳資料解析為字典
            params = {}
            for part in data.split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = v
            choice = params.get("choice")
            # 映射選擇為序列字元
            choice_map = {"banker": "B", "player": "P", "tie": "T"}
            if choice and choice in choice_map:
                seq_char = choice_map[choice]
                # 如果處於手動輸入模式，累計序列並回覆預測
                if manual_mode.get(uid):
                    # 初始化列表
                    history = user_history_seq.get(uid, [])
                    history.append(seq_char)
                    user_history_seq[uid] = history
                    # 使用集成預測
                    probs = ensemble_prediction(history)
                    # 是否使用RNN
                    using_rnn = DEEP_LEARNING_AVAILABLE and rnn_model is not None and len(history) >= 10
                    msg = render_reply(history, probs, using_rnn, {})
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=msg))
                else:
                    # 非手動模式，單步預測
                    seq = [seq_char]
                    # 使用規則引擎進行單步預測
                    probs = predict_probs_with_tie_adjustment(seq)
                    msg = render_reply(seq, probs, False, {})
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=msg))
            else:
                # 未知或缺失的回傳資料
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text="收到未知操作，請重新選擇。"),
                )
        except Exception as e:
            logger.exception(f"處理回傳事件時發生錯誤: {e}")
            try:
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text="處理您的操作時發生錯誤，請稍後再試。"),
                )
            except Exception:
                pass

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
