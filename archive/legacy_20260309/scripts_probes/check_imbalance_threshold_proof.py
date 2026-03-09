import json, numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score

def check():
    print("========== 弱信号下极度不平衡数据的 XGBoost 阈值探测 ==========")
    
    # 1. 模拟与目前 Noleak 实验数据分布完全相同的场景
    # 我们知道 nlst 阳性率在 2% 左右。特征信号十分微弱（AUC ≈ 0.48-0.5）
    np.random.seed(42)
    n_samples = 5000
    pos_rate = 0.02
    
    # 生成真实的 Y分布
    y_true = np.random.binomial(1, pos_rate, size=n_samples)
    print(f"数据总数: {n_samples}, 真实阳性数量: {y_true.sum()}, 阳性比例: {y_true.mean():.4f}")
    
    # 模拟合成数据 (TabDiff产生的X)：因为无强预测信号，我们以近似随机的弱相关特征代替 
    X_train = np.random.randn(n_samples, 4) 
    y_train = np.random.binomial(1, pos_rate, size=n_samples)
    
    X_test = np.random.randn(n_samples, 4)
    y_test = y_true # 测试集的Y
    
    # 我们在训练集中强行加入一丢丢弱关联，让AUC达到大概0.5稍微浮动
    X_train[:, 0] += y_train * 0.02
    X_test[:, 0] += y_test * 0.02
    
    print("\n--- 开始训练 XGBoost (模拟 TSTR 流程) ---")
    model = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=1, n_estimators=50)
    model.fit(X_train, y_train)
    
    # 预测概率
    probs = model.predict_proba(X_test)[:, 1]
    
    print(f"预测的全部概率统计:")
    print(f"  Max  预测概率: {probs.max():.6f}")
    print(f"  Min  预测概率: {probs.min():.6f}")
    print(f"  Mean 预测概率: {probs.mean():.6f}")
    
    # 默认 0.5 阈值预测
    preds_default = model.predict(X_test)
    f1_default = f1_score(y_test, preds_default)
    auc_score = roc_auc_score(y_test, probs)
    
    print("\n--- 默认阈值 (0.5) 结果 ---")
    print(f"在 predict() > 0.5 的阈值下被判为阳性的样本数: {preds_default.sum()}")
    print(f"AUC: {auc_score:.4f}")
    print(f"F1 Score: {f1_default:.4f}")
    
    # 我们利用 Precision-Recall curve 找到能使 F1 > 0 的最低/最佳阈值
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-15)
    best_idx = np.argmax(f1s)
    
    print(f"\n--- PR-Curve 最佳阈值扫描结果 ---")
    print(f"使 F1 最大的最佳概率阈值: {thresholds[best_idx] if best_idx < len(thresholds) else 'NaN':.6f}")
    print(f"该阈值下的最佳 F1 Score: {f1s[best_idx]:.4f}")
    print("\n结论：当模型的最高预测概率远远达不到 0.5 时，默认的 predict() 一定会输出全0，导致 F1 严丝合缝等于 0。")

if __name__ == '__main__':
    check()
