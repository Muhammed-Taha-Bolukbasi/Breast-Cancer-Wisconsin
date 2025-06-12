import pandas as pd
import numpy as np
import yaml
import sys
import os
from typing import Union, Tuple, Optional, Any

# Add project root directory to sys.path so that modules in src can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.data_loader.data_loader import DataLoader
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PolynomialFeatures


class DropUnnecessaryColumns(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer to drop columns that are all NaN or specified by the user.
    """
    def fit(self, X, y=None):
        # Find columns that are all NaN
        empty_cols = X.columns[X.isnull().all()].tolist()
        # If there is an 'id' column, add it to the list
        id_cols = [col for col in X.columns if col.lower() in ['id']]
        # List of columns to drop
        self.columns_to_drop_ = empty_cols + id_cols
        # Columns to keep (used for get_feature_names_out)
        self.columns_to_keep_ = [col for col in X.columns if col not in self.columns_to_drop_]
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop_, errors='ignore')

    def get_feature_names_out(self, input_features=None):
        # If input_features is provided, filter based on columns to keep
        if input_features is not None:
            return np.array([col for col in input_features if col in self.columns_to_keep_])
        # Else return stored columns to keep
        return np.array(self.columns_to_keep_)

class LabelEncoderTransformer:
    """
    If the target column (y) is categorical, applies label encoding; otherwise, leaves it unchanged.
    X (features) can be a numpy array or DataFrame. Returns a DataFrame with X and the (possibly encoded) y column merged.
    And provides mappings for original labels to encoded values.
    Usage:
        le_transformer = LabelEncoderTransformer()
        df_merged = le_transformer.encode_and_merge(X, y)

    Returns:
        df_merged: DataFrame with features and encoded target label.        
    """
    def __init__(self, feature_names=None):
        self.label_encoder = None
        self.was_encoded = False
        self.label_mapping = None  # Original label -> encoded value
        self.inverse_label_mapping = None  # Encoded value -> original label

    def encode_and_merge(
        self, X, y: pd.Series, return_mapping: bool = False
    ) -> Any:
        """
        If the target column (y) is categorical, applies label encoding; otherwise, leaves it unchanged.
        X (features) can be a numpy array or DataFrame. Returns a DataFrame with X and the (possibly encoded) y column merged.
        If return_mapping=True, returns (df_merged, inverse_label_mapping) tuple.
        Returns:
            df_merged: DataFrame with features and encoded target label.
            inverse_label_mapping: (Optional) dict, encoded value -> original label
        """
        # If X is a numpy array, convert to DataFrame
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, index=y.index)
        else:
            X_df = X.copy()
        # If y is a DataFrame, convert to Series
        if isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                y = y[y.columns[0]]
            else:
                raise ValueError("y DataFrame has more than one column!")
        # If y is categorical, encode it
        if y.dtype == 'object' or str(y.dtype).startswith('category'):
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y.astype(str))
            self.was_encoded = True
            y_new = pd.Series(np.asarray(y_encoded), name='Target_Label', index=y.index)
            # Label mapping'i oluştur
            classes = self.label_encoder.classes_
            encoded = np.array(self.label_encoder.transform(classes)).tolist()
            self.label_mapping = dict(zip(classes, encoded))
            self.inverse_label_mapping = dict(zip(encoded, classes))
        else:
            y_new = y.copy()
            y_new.name = 'Target_Label'  # Always set name to Target_Label
            self.was_encoded = False
            self.label_mapping = None
            self.inverse_label_mapping = None
        # Check length consistency
        if len(X_df) != len(y_new):
            raise ValueError("Length mismatch between X and y!")
        # Merge X and y
        df_merged: pd.DataFrame = X_df.copy()
        df_merged[y_new.name] = y_new
        if return_mapping:
            return df_merged, self.inverse_label_mapping
        return df_merged

class DataPreprocessorPipeline(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible class for automatic data preprocessing.
    Handles missing value imputation, categorical encoding, and feature scaling using a pipeline.
    """
    def __init__(self):
        # Pipeline and column lists are initialized as None
        self.pipeline = None
        self.categorical_cols = None
        self.numerical_cols = None
        self.all_cols = None
        self.target = None

        with open(os.path.join(project_root, 'conf.yaml'), 'r') as file:
            self.config = yaml.safe_load(file)

    def fit(self, X, y=None, feature_extraction: bool = True):
        # Automatically detect numerical and categorical columns
        self.numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        # Feature extraction flag from config if not explicitly passed
        feature_extraction_flag = self.config.get('feature_extraction', True) if feature_extraction is None else feature_extraction
        # Build numerical pipeline conditionally
        num_steps = [
            ('drop_empty', DropUnnecessaryColumns()),
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler()),
        ]
        if feature_extraction_flag:
            num_steps.append(('feature_extraction', PolynomialFeatures(degree=2, include_bias=False)))
            num_steps.append(('feature_selection', SelectKBest(score_func=f_classif, k=self.config.get('selectkbest', 100))))
        num_pipeline = Pipeline(num_steps)
        # For categorical features: impute missing values with mode, then encode with OneHotEncoder
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ])
        # Combine numerical and categorical pipelines
        self.pipeline = ColumnTransformer([
            ('num', num_pipeline, self.numerical_cols),
            ('cat', cat_pipeline, self.categorical_cols),
        ])
        # Fit the pipeline to the data
        self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        # Raise an error if transform is called before fit
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not fitted yet. Call 'fit' before 'transform'.")
        # Transform the data using the pipeline
        return self.pipeline.transform(X)

    def fit_transform(self, X, y=None, feature_extraction: bool = True,  **fit_params) -> np.ndarray:
        # Perform both fit and transform, return the result as a numpy array
        return np.asarray(self.fit(X, y, feature_extraction=feature_extraction).transform(X))
    

class DataPreprocessor:
    """
    Combines feature preprocessing and target label encoding into a single step.
    Usage:
        dp = DataPreprocessor()
        df_processed = dp.fit_transform(df, df_target)
    """
    def __init__(self):
        self.pipeline = DataPreprocessorPipeline()
        self.feature_names = None
        self.label_encoder = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, return_mapping: bool = False, feature_extraction: bool = True) -> Any:
        """
        Fit the preprocessing pipeline to the features and target, then transform them.
        args:
            X (pd.DataFrame): Features to preprocess.
            y (pd.Series): Target labels to encode.
            return_mapping (bool): If True, returns a tuple of (processed DataFrame, label mapping).
        returns:
            df (pd.DataFrame): Processed features with encoded target label.
            mapping (Optional[dict]): Mapping of original labels to encoded values if return_mapping is True.
        """
        # Fit and transform features
        X_transformed = self.pipeline.fit_transform(X, y, feature_extraction=feature_extraction)
        feature_names = self.pipeline.pipeline.get_feature_names_out() # type: ignore
        X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
        # Encode and merge target
        self.label_encoder = LabelEncoderTransformer()
        if return_mapping:
            result = self.label_encoder.encode_and_merge(X_transformed_df, y, return_mapping=True)
            df = result[0] if isinstance(result, tuple) else result
            mapping = result[1] if isinstance(result, tuple) and len(result) > 1 else None
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            return df, mapping
        else:
            df = self.label_encoder.encode_and_merge(X_transformed_df, y, return_mapping=False)
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            return df
        
