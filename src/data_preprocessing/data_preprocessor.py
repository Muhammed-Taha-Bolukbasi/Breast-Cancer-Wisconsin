import pandas as pd
import numpy as np
import sys
import os

# Add project root directory to sys.path so that modules in src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data_loader.data_loader import DataLoader
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class DropUnnecessaryColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Find columns that are all NaN
        empty_cols = X.columns[X.isnull().all()].tolist()
        
        # If there is an 'id' column, add it to the list
        id_cols = [col for col in X.columns if col.lower() in ['id']]

        # List of columns to drop
        self.columns_to_drop_ = empty_cols + id_cols
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop_, errors='ignore')

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible class for automatic data preprocessing.
    Handles missing value imputation, categorical encoding, and feature scaling using a pipeline.
    """

    def __init__(self):
        # Pipeline and column lists are initialized as None
        self.pipeline = None
        self.categorical_cols = None
        self.numerical_cols = None

    def fit(self, X, y=None):
        # Automatically detect numerical and categorical columns
        self.numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # For numerical features: impute missing values with median, then scale with MinMaxScaler
        num_pipeline = Pipeline([
            ('drop_empty', DropUnnecessaryColumns()),
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ])

        # For categorical features: impute missing values with mode, then encode with OneHotEncoder
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Combine numerical and categorical pipelines
        self.pipeline = ColumnTransformer([
            ('num', num_pipeline, self.numerical_cols),
            ('cat', cat_pipeline, self.categorical_cols)
        ])
        # Fit the pipeline to the data
        self.pipeline.fit(X)
        return self

    def transform(self, X):
        # Raise an error if transform is called before fit
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not fitted yet. Call 'fit' before 'transform'.")
        # Transform the data using the pipeline
        return self.pipeline.transform(X)

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        # Perform both fit and transform, return the result as a numpy array
        return np.asarray(self.fit(X, y).transform(X))

