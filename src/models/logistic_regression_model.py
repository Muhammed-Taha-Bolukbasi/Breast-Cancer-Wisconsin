from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.data_loader.data_loader import DataLoader
from src.data_preprocessing.data_preprocessor import DataPreprocessor
import pandas as pd
import numpy as np

from typing import Literal, Optional

class LogisticRegressionModel(BaseEstimator, ClassifierMixin):
    def __init__(self, penalty: Optional[Literal['l1', 'l2', 'elasticnet']] = 'l2', C: float = 1.0, solver: Literal['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'] = 'lbfgs', **kwargs):
        self.model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=1000, **kwargs)

    def fit(self, X, y=None):
        if y is None and isinstance(X, pd.DataFrame):
            if "Target_Label" in X.columns:
                y = X["Target_Label"]
                X = X.drop(columns=["Target_Label"])
        if y is None:
            raise ValueError("Target variable y must be provided or present as 'Target_Label' in X.")
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        self.model.set_params(**params)
        return self

if __name__ == "__main__":
    dataloader = DataLoader()
    csv_path = os.path.join(project_root, "data", "breast_cancer.csv")
    df, df_target = dataloader.load_data(csv_path)
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.fit_transform(df, df_target)
    X_train, X_test, y_train, y_test = train_test_split(df_processed.drop(columns=["Target_Label"]), df_processed["Target_Label"], test_size=0.2, random_state=42)
    model = LogisticRegressionModel(penalty='l1', solver='liblinear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
