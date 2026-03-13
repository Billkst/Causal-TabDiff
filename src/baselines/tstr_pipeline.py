"""
通用 TSTR (Train on Synthetic, Test on Real) Pipeline.
支持 STaSy, TabSyn, TabDiff, TSDiff 原版。
"""
import torch
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import os
import sys
sys.path.insert(0, 'src')
from data.landmark_adapter import adapt_landmark_to_generative


class TSTRPipeline:
    """通用 TSTR 评估流程"""
    
    def __init__(self, generative_model, downstream_classifier='xgboost'):
        """
        Args:
            generative_model: 生成模型实例，需实现 fit() 和 sample() 方法
            downstream_classifier: 下游分类器类型，默认 'xgboost'
        """
        self.gen_model = generative_model
        self.classifier_type = downstream_classifier
        self.classifier = None
    
    def train_generative_model(self, train_loader, epochs, device):
        """在真实训练集上训练生成模型"""
        print(f"[TSTR] 训练生成模型...")
        adapted_loader = []
        for batch in train_loader:
            adapted_loader.append(adapt_landmark_to_generative(batch, device))
        self.gen_model.fit(adapted_loader, epochs, device)
        print(f"[TSTR] 生成模型训练完成")
    
    def generate_synthetic_data(self, n_samples, device):
        """生成合成训练数据 - 联合生成 (X, Y)"""
        print(f"[TSTR] 生成 {n_samples} 个合成样本...")
        # 所有 generative baselines 都返回 (X_cf, Y_cf)
        X_synthetic, Y_synthetic = self.gen_model.sample(n_samples, alpha_target=None, device=device)
        print(f"[TSTR] 合成数据生成完成，X shape={X_synthetic.shape}, Y shape={Y_synthetic.shape}")
        return X_synthetic, Y_synthetic
    
    def train_downstream_classifier(self, X_synthetic, Y_synthetic):
        """在合成数据上训练下游分类器"""
        print(f"[TSTR] 训练下游分类器 ({self.classifier_type})...")
        
        if self.classifier_type == 'xgboost':
            from xgboost import XGBClassifier
            self.classifier = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        else:
            raise ValueError(f"不支持的分类器类型: {self.classifier_type}")
        
        # Flatten features
        if len(X_synthetic.shape) == 3:
            X_flat = X_synthetic.reshape(X_synthetic.shape[0], -1)
        else:
            X_flat = X_synthetic
        
        y_flat = Y_synthetic.flatten() if len(Y_synthetic.shape) > 1 else Y_synthetic
        
        print(f"[TSTR] 训练数据: X shape={X_flat.shape}, Y shape={y_flat.shape}, Y positive rate={y_flat.mean():.4f}")
        self.classifier.fit(X_flat, y_flat)
        print(f"[TSTR] 下游分类器训练完成")
    
    def predict(self, X_real):
        """在真实数据上预测"""
        if self.classifier is None:
            raise RuntimeError("分类器未训练，请先调用 train_downstream_classifier()")
        
        X_flat = X_real.reshape(X_real.shape[0], -1)
        y_pred_proba = self.classifier.predict_proba(X_flat)[:, 1]
        return y_pred_proba
    
    def save(self, save_path):
        """保存 TSTR pipeline"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'classifier_type': self.classifier_type
            }, f)
        print(f"[TSTR] Pipeline 已保存到 {save_path}")
    
    def load(self, load_path):
        """加载 TSTR pipeline"""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        self.classifier = data['classifier']
        self.classifier_type = data['classifier_type']
        print(f"[TSTR] Pipeline 已从 {load_path} 加载")


def extract_features_and_labels(dataloader, device):
    """从 dataloader 提取特征和标签"""
    X_list, y_list = [], []
    for batch in dataloader:
        x = batch['x'].to(device)  # (batch, seq_len, features)
        y = batch['y_2year'].to(device)  # (batch, 1)
        X_list.append(x.cpu().numpy())
        y_list.append(y.cpu().numpy())
    
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0).flatten()
    return X, y
