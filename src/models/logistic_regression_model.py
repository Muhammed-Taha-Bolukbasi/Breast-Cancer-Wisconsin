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
        if y is None:
            if not (isinstance(X, pd.DataFrame) and "Target_Label" in X.columns):
                raise ValueError("y must not be None. Provide y or ensure 'Target_Label' exists in X.")
            y = X["Target_Label"]
            X = X.drop(columns=["Target_Label"])
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

    def save_model(self):
        """
        Save the trained Logistic Regression model to the given file path using joblib.
        """
        import yaml
        import joblib
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        with open(os.path.join(project_root, "conf.yaml"), 'r') as file:
            config = yaml.safe_load(file)
            model_save_name = config.get("model_save_name", "logistic_regression_model.pkl")
        path = os.path.join(project_root, "models_saves", "logistic_regression")
        abs_model_path = os.path.join(path, model_save_name)
        joblib.dump(self.model, abs_model_path)
        return abs_model_path
