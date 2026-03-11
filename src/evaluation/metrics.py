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
    
    thresholds = np.linspace(0.01, 0.99, 99)
    best_score = -1
    best_thresh = 0.5
    
    for thresh in thresholds:
        y_pred_binary = (y_pred_proba >= thresh).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred_binary, zero_division=0)
        elif metric == 'balanced_acc':
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = (sensitivity + specificity) / 2
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_thresh = thresh
    
    return best_thresh, best_score


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
    y_true = np.asarray(y_true).flatten()
    y_pred_proba = np.asarray(y_pred_proba).flatten()
    
    brier = brier_score_loss(y_true, y_pred_proba)
    
    lr = LogisticRegression(max_iter=1000)
    lr.fit(y_pred_proba.reshape(-1, 1), y_true)
    
    intercept = lr.intercept_[0]
    slope = lr.coef_[0][0]
    
    return {
        'brier_score': brier,
        'calibration_intercept': intercept,
        'calibration_slope': slope
    }


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
