import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score, 
    recall_score, accuracy_score, confusion_matrix, brier_score_loss,
    matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression


def compute_ranking_metrics(y_true, y_pred_proba):
    y_true = np.asarray(y_true).flatten()
    y_pred_proba = np.asarray(y_pred_proba).flatten()
    
    if len(np.unique(y_true)) < 2:
        return {'auroc': np.nan, 'auprc': np.nan}
    
    auroc = roc_auc_score(y_true, y_pred_proba)
    auprc = average_precision_score(y_true, y_pred_proba)
    
    return {'auroc': auroc, 'auprc': auprc}


def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    y_true = np.asarray(y_true).flatten()
    y_pred_proba = np.asarray(y_pred_proba).flatten()

    if metric == 'f1':
        from sklearn.metrics import precision_recall_curve

        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
        best_idx = np.argmax(f1_scores)
        return float(thresholds[best_idx]), float(f1_scores[best_idx])
    elif metric == 'balanced_acc':
        thresholds = np.unique(y_pred_proba)
        if len(thresholds) > 1000:
            thresholds = np.linspace(y_pred_proba.min(), y_pred_proba.max(), 1000)

        best_score = -1
        best_thresh = 0.5
        for thresh in thresholds:
            y_pred_binary = (y_pred_proba >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = (sensitivity + specificity) / 2
            if score > best_score:
                best_score = score
                best_thresh = thresh

        return best_thresh, best_score
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_threshold_metrics(y_true, y_pred_binary):
    y_true = np.asarray(y_true).flatten()
    y_pred_binary = np.asarray(y_pred_binary).flatten()
    
    cm = confusion_matrix(y_true, y_pred_binary)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    balanced_acc = (recall + specificity) / 2
    
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred_binary)
    
    return {
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'npv': npv,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1': f1,
        'mcc': mcc,
        'confusion_matrix': cm
    }


def compute_calibration_metrics(y_true, y_pred_proba):
    """计算校准指标: Brier Score + E/O Ratio.

    E/O ratio = mean(predicted) / mean(observed).
    完美校准 = 1.0; >1 高估风险; <1 低估风险.
    替代了原先不稳定的 calibration_intercept / calibration_slope.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred_proba = np.asarray(y_pred_proba).flatten()

    brier = brier_score_loss(y_true, y_pred_proba)

    mean_pred = float(np.mean(y_pred_proba))
    mean_obs = float(np.mean(y_true))
    eo_ratio = mean_pred / mean_obs if mean_obs > 0 else float('nan')

    return {
        'brier_score': brier,
        'eo_ratio': eo_ratio,
    }


def platt_calibrate(val_y_true, val_y_pred, test_y_pred):
    """Platt Scaling: 在验证集 logits 上拟合 LogisticRegression，校准测试集概率.

    排序不变 (AUROC/AUPRC 保持一致)，只改变概率值.

    Args:
        val_y_true: 验证集真实标签 (0/1)
        val_y_pred: 验证集预测概率 [0, 1]
        test_y_pred: 测试集预测概率 [0, 1]

    Returns:
        test_pred_calibrated: 校准后的测试集概率
    """
    val_y_true = np.asarray(val_y_true).flatten()
    val_y_pred = np.asarray(val_y_pred).flatten()
    test_y_pred = np.asarray(test_y_pred).flatten()

    eps = 1e-7
    val_logits = np.log(
        np.clip(val_y_pred, eps, 1 - eps) / (1 - np.clip(val_y_pred, eps, 1 - eps))
    )
    test_logits = np.log(
        np.clip(test_y_pred, eps, 1 - eps) / (1 - np.clip(test_y_pred, eps, 1 - eps))
    )

    lr = LogisticRegression(max_iter=1000, solver='lbfgs')
    lr.fit(val_logits.reshape(-1, 1), val_y_true)
    test_calibrated = lr.predict_proba(test_logits.reshape(-1, 1))[:, 1]
    return test_calibrated


def compute_all_metrics(y_true, y_pred_proba, threshold=None, val_y_true=None, val_y_pred_proba=None):
    if threshold is None:
        if val_y_true is not None and val_y_pred_proba is not None:
            threshold, _ = find_optimal_threshold(val_y_true, val_y_pred_proba, metric='f1')
        else:
            threshold = 0.5

    y_pred_binary = (y_pred_proba >= threshold).astype(int)

    ranking = compute_ranking_metrics(y_true, y_pred_proba)
    threshold_metrics = compute_threshold_metrics(y_true, y_pred_binary)
    calibration = compute_calibration_metrics(y_true, y_pred_proba)

    return {
        **ranking,
        **threshold_metrics,
        **calibration,
        'threshold': threshold
    }
