# -*- coding: utf-8 -*-
"""
pfilter.py — 獨立預測簡化版本
專注於點數特徵分析，不依賴長期歷史
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

def _env_float(k: str, default: str) -> float:
    try: return float(os.getenv(k, default))
    except Exception: return float(default)

def _env_int(k: str, default: str) -> int:
    try: return int(os.getenv(k, default))
    except Exception: return int(default)

# 基準概率（考慮抽水）
BASE_B = _env_float("PRIOR_B", "0.458")
BASE_P = _env_float("PRIOR_P", "0.446") 
BASE_T = _env_float("PRIOR_T", "0.096")

EPS = 1e-9

@dataclass
class PointAnalyzer:
    """點數分析器 - 獨立預測核心"""
    
    def __init__(self):
        self.recent_points = []
        self.max_history = 3  # 只保留最近3局
        
    def add_points(self, p_pts: int, b_pts: int):
        """添加點數記錄"""
        point_data = {
            'p_pts': p_pts, 'b_pts': b_pts,
            'diff': abs(p_pts - b_pts), 
            'total': p_pts + b_pts,
            'natural': p_pts >= 8 or b_pts >= 8
        }
        self.recent_points.append(point_data)
        if len(self.recent_points) > self.max_history:
            self.recent_points.pop(0)
    
    def analyze_pattern(self) -> dict:
        """分析點數模式"""
        if not self.recent_points:
            return {'trend': 'neutral', 'volatility': 'medium'}
            
        latest = self.recent_points[-1]
        diff = latest['diff']
        total = latest['total']
        
        # 分析趨勢
        if diff >= 6:
            trend = 'strong'
        elif diff >= 3:
            trend = 'moderate' 
        else:
            trend = 'weak'
            
        # 分析波動性
        if len(self.recent_points) >= 2:
            prev_diff = self.recent_points[-2]['diff']
            volatility = 'high' if abs(diff - prev_diff) >= 3 else 'medium'
        else:
            volatility = 'medium'
            
        return {'trend': trend, 'volatility': volatility, 'natural': latest['natural']}
    
    def predict_probs(self) -> np.ndarray:
        """基於點數模式預測概率"""
        pattern = self.analyze_pattern()
        
        if pattern['trend'] == 'strong':
            # 強趨勢 - 傾向延續
            if self.recent_points[-1]['p_pts'] > self.recent_points[-1]['b_pts']:
                return np.array([0.35, 0.60, 0.05], dtype=np.float32)  # 閒強
            else:
                return np.array([0.60, 0.35, 0.05], dtype=np.float32)  # 莊強
                
        elif pattern['trend'] == 'moderate':
            # 中等趨勢 - 輕微傾向
            if self.recent_points[-1]['p_pts'] > self.recent_points[-1]['b_pts']:
                return np.array([0.42, 0.53, 0.05], dtype=np.float32)
            else:
                return np.array([0.53, 0.42, 0.05], dtype=np.float32)
                
        else:
            # 弱趨勢或無趨勢 - 回歸基準
            if pattern['natural']:
                # 有自然牌時和局機率降低
                return np.array([0.47, 0.48, 0.05], dtype=np.float32)
            else:
                # 常規情況
                return np.array([BASE_B, BASE_P, BASE_T], dtype=np.float32)

@dataclass 
class OutcomePF:
    """獨立預測版本 - 簡化接口"""
    decks: int = _env_int("DECKS", "6")
    
    def __post_init__(self):
        self.analyzer = PointAnalyzer()
        self.last_prediction = np.array([BASE_B, BASE_P, BASE_T], dtype=np.float32)
    
    def update_point_history(self, p_pts: int, b_pts: int):
        """更新點數歷史"""
        self.analyzer.add_points(p_pts, b_pts)
    
    def update_outcome(self, outcome: int):
        """獨立預測版本中，結果只用於驗證，不影響預測"""
        pass  # 獨立預測不依賴歷史結果
    
    def predict(self, sims_per_particle: int = 1) -> np.ndarray:
        """獨立預測"""
        if not self.analyzer.recent_points:
            # 無數據時返回基準概率
            return self.last_prediction
            
        self.last_prediction = self.analyzer.predict_probs()
        return self.last_prediction
