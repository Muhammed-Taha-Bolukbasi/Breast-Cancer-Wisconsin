from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.data_preprocessing.data_preprocessor import DataPreprocessorPipeline
from sklearn.base import BaseEstimator, ClassifierMixin
import os
import sys
import yaml
import joblib

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)


class RandomForestPipeline(BaseEstimator, ClassifierMixin):
    RF_ALLOWED_PARAMS = {
        "n_estimators",
        "max_depth",
        "random_state",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "bootstrap",
        "criterion",
        "class_weight",
    }

    def __init__(self, **kwargs):
        self.model_params = {
            k: v for k, v in kwargs.items() if k in self.RF_ALLOWED_PARAMS
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
                ("model", RandomForestClassifier(**self.model_params)),
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
        return self.model_params.copy()

    def set_params(self, **params):
        for k, v in params.items():
            if k in self.RF_ALLOWED_PARAMS:
                self.model_params[k] = v
        return self

    def save_model(self):
        if self.pipeline is None:
            raise RuntimeError(
                "Pipeline is not fitted yet. Call 'fit' before 'save_model'."
            )
        with open(os.path.join(project_root, "conf.yaml"), "r") as file:
            config = yaml.safe_load(file)
            model_save_name = config.get("model_save_name", "random_forest_model.pkl")
        path = os.path.join(project_root, "model_saves", "random_forest")
        os.makedirs(path, exist_ok=True)
        abs_model_path = os.path.join(path, model_save_name)
        joblib.dump(self.pipeline, abs_model_path)
        return abs_model_path
