from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
import os
import sys
# Add project root directory to sys.path so that modules in src can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.data_loader.data_loader import DataLoader
from src.data_preprocessing.data_preprocessor import DataPreprocessor
import pandas as pd
import numpy as np

# Ensure that the DataLoader and DataPreprocessor are correctly imported
class XGBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **kwargs)        


    def fit(self, X, y=None):
        """
        Fit the XGBoost model. If y is None, tries to use 'Target_Label' column from X.
        If 'Target_Label' exists in X, it is used as target and dropped from features.
        """
        if y is None and isinstance(X, (pd.DataFrame, )):
            if "Target_Label" in X.columns:
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
    

if __name__ == "__main__":
    # Example usage
    dataloader = DataLoader()
    csv_path = os.path.join(project_root, "data", "breast_cancer.csv")
    df, df_target = dataloader.load_data(csv_path)
    
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.fit_transform(df, df_target)

    X_train, X_test, y_train, y_test = train_test_split(df_processed.drop(columns=["Target_Label"]), df_processed["Target_Label"], test_size=0.3, random_state=42)

    model = XGBoost(n_estimators=100, max_depth=3, learning_rate=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Calculate and print accuracy
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.8f}")
