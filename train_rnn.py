import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

np.random.seed(42)

# 创建示例训练数据（实际应用中应该使用真实历史数据）
def create_sample_data(num_sequences: int = 5000) -> list:
    """生成帶有簡單規律的隨機序列。

    Args:
        num_sequences: 要產生的序列數量。
    Returns:
        list: 由 'B'、'P'、'T' 組成的序列列表。
    """
    sequences = []

    # 創建一些示例序列
    for _ in range(num_sequences):
        # 隨機生成序列，但帶有一定模式
        seq_length = np.random.randint(20, 50)
        seq: list = []

        current = np.random.choice(['B', 'P'])
        streak = 0
        max_streak = np.random.randint(3, 8)

        for _ in range(seq_length):
            if streak >= max_streak or np.random.random() < 0.2:
                current = 'P' if current == 'B' else 'B'
                streak = 0
                max_streak = np.random.randint(3, 8)

            seq.append(current)
            streak += 1

            # 隨機添加和局
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
    """训练改进的RNN模型"""
    print("生成训练数据...")
    sequences = create_sample_data()

    print("准备训练数据...")
    X, y, label_encoder = prepare_data(sequences)

    print(f"训练数据形状: X={X.shape}, y={y.shape}")

    # 创建更深的模型
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    os.makedirs('models', exist_ok=True)
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
        ModelCheckpoint('models/best_rnn_model.h5', save_best_only=True, monitor='val_accuracy')
    ]

    print("开始训练模型...")
    model.fit(
        X,
        y,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1,
    )

    # 保存最终模型
    model.save('models/rnn_model.h5')

    print("模型训练完成并已保存")

    # 保存标签编码器
    np.save('models/label_encoder_classes.npy', label_encoder.classes_)

    return model, label_encoder

if __name__ == "__main__":
    train_rnn_model()
