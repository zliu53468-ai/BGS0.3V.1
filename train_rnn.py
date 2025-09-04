import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import os

# 创建示例训练数据（实际应用中应该使用真实历史数据）
def create_sample_data():
    # 这是一个简单的示例，实际应该使用真实的历史数据
    sequences = []
    
    # 创建一些示例序列
    for i in range(1000):
        # 随机生成序列，但有一定模式
        seq_length = np.random.randint(20, 50)
        seq = []
        
        # 添加一些模式
        current = np.random.choice(['B', 'P'])
        streak = 0
        max_streak = np.random.randint(3, 8)
        
        for j in range(seq_length):
            if streak >= max_streak or np.random.random() < 0.2:
                current = 'P' if current == 'B' else 'B'
                streak = 0
                max_streak = np.random.randint(3, 8)
            
            seq.append(current)
            streak += 1
            
            # 随机添加和局
            if np.random.random() < 0.05:
                seq.append('T')
        
        sequences.append(seq)
    
    return sequences

def prepare_data(sequences, seq_length=10):
    """准备训练数据"""
    X = []
    y = []
    
    label_encoder = LabelEncoder()
    label_encoder.fit(['B', 'P', 'T'])
    
    for seq in sequences:
        if len(seq) < seq_length + 1:
            continue
            
        # 将序列转换为one-hot编码
        encoded = label_encoder.transform(seq)
        one_hot = to_categorical(encoded, num_classes=3)
        
        # 创建滑动窗口数据
        for i in range(len(one_hot) - seq_length):
            X.append(one_hot[i:i+seq_length])
            y.append(one_hot[i+seq_length])
    
    return np.array(X), np.array(y), label_encoder

def train_rnn_model():
    """训练RNN模型"""
    print("生成训练数据...")
    sequences = create_sample_data()
    
    print("准备训练数据...")
    X, y, label_encoder = prepare_data(sequences)
    
    print(f"训练数据形状: X={X.shape}, y={y.shape}")
    
    # 创建模型
    model = Sequential([
        LSTM(32, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("开始训练模型...")
    history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
    
    # 保存模型
    os.makedirs('models', exist_ok=True)
    model.save('models/rnn_model.h5')
    
    print("模型训练完成并已保存")
    
    # 保存标签编码器
    np.save('models/label_encoder_classes.npy', label_encoder.classes_)
    
    return model, label_encoder

if __name__ == "__main__":
    train_rnn_model()
