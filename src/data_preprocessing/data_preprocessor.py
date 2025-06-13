import pandas as pd
import numpy as np
import yaml
import sys
import os

# Add project root directory to sys.path so that modules in src can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)
from src.data_loader.data_loader import DataLoader
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures


class DropUnnecessaryColumns(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer to drop columns that are all NaN or specified by the user.
    """

    def fit(self, X, y=None):
        # Find columns that are all NaN
        empty_cols = X.columns[X.isnull().all()].tolist()
        # If there is an 'id' column, add it to the list
        id_cols = [col for col in X.columns if col.lower() in ["id"]]
        # List of columns to drop
        self.columns_to_drop_ = empty_cols + id_cols
        # Columns to keep (used for get_feature_names_out)
        self.columns_to_keep_ = [
            col for col in X.columns if col not in self.columns_to_drop_
        ]
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop_, errors="ignore")

    def get_feature_names_out(self, input_features=None):
        # If input_features is provided, filter based on columns to keep
        if input_features is not None:
            return np.array(
                [col for col in input_features if col in self.columns_to_keep_]
            )
        # Else return stored columns to keep
        return np.array(self.columns_to_keep_)


class DataPreprocessorPipeline(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible class for automatic data preprocessing.
    Handles missing value imputation, categorical encoding, and feature scaling using a pipeline.
    """

    def __init__(self):
        self.config = None
        self.numerical_cols = None
        self.categorical_cols = None
        self.pipeline = None
        with open(os.path.join(project_root, "conf.yaml"), "r") as file:
            self.config = yaml.safe_load(file)

    def build_pipeline(self, X_train: pd.DataFrame, feature_extraction: bool = True):

        self.numerical_cols = X_train.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        self.categorical_cols = X_train.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        feature_extraction_flag = (
            self.config.get("feature_extraction", True)
            if feature_extraction is None
            else feature_extraction
        )

        num_steps = [
            ("drop_empty", DropUnnecessaryColumns()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
        if feature_extraction_flag:
            num_steps.append(
                ("feature_extraction", PolynomialFeatures(degree=2, include_bias=False))
            )
            num_steps.append(
                (
                    "feature_selection",
                    SelectKBest(
                        score_func=f_classif, 
                        k=self.config["selectkbest"] if self.config and "selectkbest" in self.config else 100
                    ),
                )
            )  # type: ignore

        num_pipeline = Pipeline(num_steps)
        cat_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("num", num_pipeline, self.numerical_cols),
                ("cat", cat_pipeline, self.categorical_cols),
            ]
        )

        return preprocessor
