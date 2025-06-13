from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import os
import sys
import yaml
import joblib
import pandas as pd
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.data_preprocessing.data_preprocessor import DataPreprocessorPipeline
from src.data_loader.data_loader import DataLoader

from sklearn.base import BaseEstimator, ClassifierMixin


class XGBoostPipeline(BaseEstimator, ClassifierMixin):
    XGBOOST_ALLOWED_PARAMS = {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "random_state",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "gamma",
        "reg_alpha",
        "reg_lambda",
        "scale_pos_weight",
        "objective",
        "booster",
        "tree_method",
        "eval_metric",
        "use_label_encoder",
    }

    def __init__(self, **kwargs):
        self.model_params = {
            k: v for k, v in kwargs.items() if k in self.XGBOOST_ALLOWED_PARAMS
        }
        self.pipeline = None
        with open(os.path.join(project_root, "conf.yaml"), "r") as file:
            self.config = yaml.safe_load(file)

    def fit(self, X, y):
        preprocessor = DataPreprocessorPipeline().build_pipeline(
            X, feature_extraction=self.config["feature_extraction"]
        )
        self.pipeline = Pipeline(
            [
                ("preprocessing", preprocessor),
                ("model", XGBClassifier(eval_metric="logloss", **self.model_params)),
            ]
        )
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        if self.pipeline is None:
            raise RuntimeError(
                "Pipeline is not fitted yet. Call 'fit' before 'predict'."
            )
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        if self.pipeline is None:
            raise RuntimeError(
                "Pipeline is not fitted yet. Call 'fit' before 'predict_proba'."
            )
        return self.pipeline.predict_proba(X)

    def get_params(self, deep=True):
        if self.pipeline is not None:
            return self.pipeline.get_params(deep=deep)
        return self.model_params

    def set_params(self, **params):
        self.model_params.update(params)
        if self.pipeline is not None:
            self.pipeline.set_params(**params)
        return self

    def save_model(self):
        if self.pipeline is None:
            raise RuntimeError(
                "Pipeline is not fitted yet. Call 'fit' before 'save_model'."
            )
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        with open(os.path.join(project_root, "conf.yaml"), "r") as file:
            config = yaml.safe_load(file)
            model_save_name = config.get("model_save_name", "xgboost_model.pkl")
        path = os.path.join(project_root, "model_saves", "xgboost")
        os.makedirs(path, exist_ok=True)
        abs_model_path = os.path.join(path, model_save_name)
        joblib.dump(self.pipeline, abs_model_path)
        return abs_model_path


