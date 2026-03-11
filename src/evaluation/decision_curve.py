import numpy as np
import matplotlib.pyplot as plt
import os


def compute_net_benefit(y_true, y_pred_proba, threshold):
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    tp = np.sum((y_true == 1) & (y_pred_binary == 1))
    fp = np.sum((y_true == 0) & (y_pred_binary == 1))
    n = len(y_true)
    
    net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
    return net_benefit


def compute_decision_curve(y_true, y_pred_proba, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    
    net_benefits = []
    for thresh in thresholds:
        nb = compute_net_benefit(y_true, y_pred_proba, thresh)
        net_benefits.append(nb)
    
    treat_all = [(np.sum(y_true) / len(y_true)) - (1 - np.sum(y_true) / len(y_true)) * (t / (1 - t)) 
                 for t in thresholds]
    treat_none = [0.0] * len(thresholds)
    
    return thresholds, net_benefits, treat_all, treat_none


def plot_decision_curve(y_true, y_pred_proba, save_path=None, model_name='Model'):
    thresholds, net_benefits, treat_all, treat_none = compute_decision_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, net_benefits, linewidth=2, label=model_name)
    plt.plot(thresholds, treat_all, '--', linewidth=1.5, label='Treat All')
    plt.plot(thresholds, treat_none, ':', linewidth=1.5, label='Treat None')
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.title('Decision Curve Analysis')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim([0, 1])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
