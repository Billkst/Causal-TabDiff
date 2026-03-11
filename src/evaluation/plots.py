import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve
from .decision_curve import plot_decision_curve
import os


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linewidth=2, label='Model')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_pr_curve(y_true, y_pred_proba, save_path=None):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, linewidth=2, label='Model')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(y_true, y_pred_binary, save_path=None, normalize=False):
    cm = confusion_matrix(y_true, y_pred_binary)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['Negative', 'Positive'],
           yticklabels=['Negative', 'Positive'],
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_calibration_curve(y_true, y_pred_proba, n_bins=10, save_path=None):
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy='uniform'
    )
    
    plt.figure(figsize=(6, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', linewidth=2, label='Model')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_all_plots(y_true, y_pred_proba, y_pred_binary, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    plot_roc_curve(y_true, y_pred_proba, os.path.join(output_dir, 'roc_curve.png'))
    plot_pr_curve(y_true, y_pred_proba, os.path.join(output_dir, 'pr_curve.png'))
    plot_confusion_matrix(y_true, y_pred_binary, os.path.join(output_dir, 'confusion_matrix.png'))
    plot_confusion_matrix(y_true, y_pred_binary, os.path.join(output_dir, 'confusion_matrix_normalized.png'), normalize=True)
    plot_calibration_curve(y_true, y_pred_proba, save_path=os.path.join(output_dir, 'calibration_plot.png'))
    plot_decision_curve(y_true, y_pred_proba, save_path=os.path.join(output_dir, 'decision_curve.png'))
