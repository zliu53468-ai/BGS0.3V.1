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
    MessageEvent, TextMessage, ImageMessage, TextSendMessage, FollowEvent
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

# =========================================================
# 博彩游戏结果识别
# =========================================================
def extract_gaming_result(img_bytes: bytes) -> List[str]:
    """专门用于识别博彩游戏结果的函数"""
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
        
        # 使用自适应阈值进行二值化
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # 设置Tesseract参数（支持简体中文和繁体中文）
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=庄闲和莊閒閑田0123456789对总'
        
        # 进行OCR识别
        text = pytesseract.image_to_string(binary, config=custom_config, lang='chi_sim+chi_tra')
        
        logger.info(f"OCR识别结果: {text}")
        
        # 解析结果
        return parse_gaming_text(text)
        
    except Exception as e:
        logger.error(f"博彩结果识别错误: {e}")
        return []

def parse_gaming_text(text: str) -> List[str]:
    """解析博彩游戏结果文本"""
    # 查找"最新好路"或类似关键词
    lines = text.split('\n')
    gaming_line = None
    
    for line in lines:
        if '最新好路' in line or '好路' in line:
            gaming_line = line
            break
    
    if not gaming_line:
        # 如果没有找到关键词，尝试查找包含数字和庄闲字符的行
        for line in lines:
            if any(char in line for char in ['庄', '閒', '閑', '闲', '和', '田']) and any(char.isdigit() for char in line):
                gaming_line = line
                break
    
    if not gaming_line:
        return []
    
    logger.info(f"找到游戏结果行: {gaming_line}")
    
    # 提取庄、闲、和的数量
    banker_count = 0
    player_count = 0
    tie_count = 0
    
    # 使用正则表达式提取数字
    patterns = [
        r'田\s*(\d+)',    # 田 23
        r'莊\s*(\d+)',    # 莊 23
        r'庄\s*(\d+)',    # 庄 23
        r'閒\s*(\d+)',    # 閒 15
        r'閑\s*(\d+)',    # 閑 15
        r'闲\s*(\d+)',    # 闲 15
        r'和\s*(\d+)',    # 和 3
    ]
    
    for pattern in patterns:
        match = re.search(pattern, gaming_line)
        if match:
            count = int(match.group(1))
            if '田' in pattern or '莊' in pattern or '庄' in pattern:
                banker_count = count
            elif '閒' in pattern or '閑' in pattern or '闲' in pattern:
                player_count = count
            elif '和' in pattern:
                tie_count = count
    
    # 如果正则没有匹配到，尝试更简单的方法
    if banker_count == 0 and player_count == 0 and tie_count == 0:
        parts = gaming_line.split()
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
        return {"side": side, "percent": 0.0, "side_prob": side_prob, "note": "单跳震荡期观望"}

    if oscillating:
        if diff < 0.12: return {"side": side, "percent": 0.0, "side_prob": side_prob, "note": "震荡期风险高"}
        if diff < 0.18: pct = 0.02
        elif diff < 0.24: pct = 0.04
        else: pct = 0.08
        return {"side": side, "percent": pct, "side_prob": side_prob, "note": "震荡期降仓"}

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
    
    side = "庄" if plan["side"] == "banker" else "闲"
    win_rate = plan["side_prob"] * 100
    
    reply = f"推荐预测：{side}（胜率{win_rate:.1f}%）\n\n"
    reply += f"解析路数：{len(seq)}手\n"
    reply += f"庄胜率：{b*100:.1f}% | 闲胜率：{p*100:.1f}% | 和局率：{t*100:.1f}%\n"
    
    if by_model:
        reply += "预测方法：RNN深度学习模型\n"
    else:
        reply += "预测方法：规则引擎\n"
    
    if plan["percent"] > 0:
        reply += f"建议下注：{plan['percent']*100:.0f}%资金于{side}"
    else:
        reply += "建议：观望不下注"
    
    if info and info.get("oscillating"):
        reply += f"\n当前牌路震荡中（交替率：{info.get('alt_rate', 0):.2f}）"
    
    return reply

# =========================================================
# API
# =========================================================
@app.route("/")
def index():
    return "BGS AI 助手正在运行 ✅ /line-webhook 已就绪", 200

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
        logger.exception(f"InvalidSignatureError: {e}. "
                         f"==> 通常是 LINE_CHANNEL_SECRET 不对 或 用错 Channel 的 Secret/Token")
        return "Invalid signature", 200
    except Exception as e:
        logger.exception(f"Unhandled error while handling webhook: {e}")
        return "Error", 200
    return "OK"

if line_handler and line_bot_api:

    @line_handler.add(FollowEvent)
    def on_follow(event: FollowEvent):
        welcome = (
            "欢迎加入BGS AI 助手 🎉\n\n"
            "输入「开始分析」后，上传博彩游戏截图，我会使用OCR技术自动辨识并回传建议下注。"
        )
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=welcome))

    @line_handler.add(MessageEvent, message=TextMessage)
    def on_text(event: MessageEvent):
        uid = getattr(event.source, "user_id", "unknown")
        txt = (event.message.text or "").strip()
        if txt in {"开始分析", "开始", "START", "分析"}:
            user_mode[uid] = True
            if pytesseract is None:
                msg = "系统错误：OCR功能未启用，请联系管理员安装Tesseract OCR"
            else:
                msg = "已进入分析模式 ✅\n请上传博彩游戏截图：我会使用OCR技术自动辨识并回复预测建议"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=msg))
            return
        line_bot_api.reply_message(event.reply_token, TextSendMessage(
            text="请先输入「开始分析」，再上传游戏截图。"
        ))

    @line_handler.add(MessageEvent, message=ImageMessage)
    def on_image(event: MessageEvent):
        uid = getattr(event.source, "user_id", "unknown")
        if not user_mode.get(uid):
            line_bot_api.reply_message(event.reply_token, TextSendMessage(
                text="尚未启用分析模式。\n请先输入「开始分析」，再上传游戏截图。"
            ))
            return
            
        if pytesseract is None:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(
                text="系统错误：OCR功能未启用，请联系管理员安装Tesseract OCR"
            ))
            return

        content = line_bot_api.get_message_content(event.message.id)
        img_bytes = b"".join(chunk for chunk in content.iter_content())
        seq = extract_gaming_result(img_bytes)
        
        if not seq or len(seq) < 5:  # 至少需要识别5个结果
            tip = f"识别结果不理想 😥 只识别到 {len(seq)} 个结果\n请确保截图清晰，包含'最新好路'区域。"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=tip))
            return

        # 使用集成预测（结合规则和RNN）
        probs = ensemble_prediction(seq)
        # 检查是否使用了RNN模型
        using_rnn = DEEP_LEARNING_AVAILABLE and rnn_model is not None and len(seq) >= 10
        msg = render_reply(seq, probs, using_rnn, {})
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=msg))

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
