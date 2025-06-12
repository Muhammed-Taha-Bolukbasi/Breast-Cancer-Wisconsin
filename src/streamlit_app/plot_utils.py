import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    return fig

def plot_metric_bar(metrics_dict, title="Model Metrics"):
    """
    Plot a bar chart for metrics such as accuracy, precision, recall, f1-score.
    metrics_dict: dict, e.g. {"accuracy": 0.95, "precision": 0.93, ...}
    """
    fig, ax = plt.subplots()
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    bars = ax.bar(names, values, color=sns.color_palette("Blues", len(names)))
    ax.set_ylim(0, 1)
    ax.bar_label(bars, fmt='%.2f')
    ax.set_ylabel('Score')
    ax.set_title(title)
    return fig

def plot_metric_radar(metrics_dict, title="Model Metrics Radar"):
    """
    Plot a radar/spider chart for metrics such as accuracy, precision, recall, f1-score.
    metrics_dict: dict, e.g. {"accuracy": 0.95, "precision": 0.93, ...}
    """
    labels = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    values += values[:1]  # close the loop
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1, endpoint=True)
    fig, ax = plt.subplots(subplot_kw={"polar": True})
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    return fig
