import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
   
)

def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


