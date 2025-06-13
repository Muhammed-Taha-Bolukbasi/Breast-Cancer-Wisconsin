import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve


def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_score, pos_label=1, ax=None, title="ROC Curve"):
    """
    y_score: probability estimates of the positive class
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def plot_precision_recall_curve(
    y_true, y_score, pos_label=1, ax=None, title="Precision-Recall Curve"
):
    precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=pos_label)
    avg_prec = average_precision_score(y_true, y_score, pos_label=pos_label)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.plot(recall, precision, color="b", lw=2, label=f"AP = {avg_prec:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    plt.tight_layout()
    return fig


def plot_calibration_curve(y_true, y_prob, n_bins=10, ax=None):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.plot(prob_pred, prob_true, marker="o", label="Calibration curve")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    plt.tight_layout()
    return fig
