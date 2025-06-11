import pandas as pd
import numpy as np
import yaml
import sys
import os

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
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop_, errors='ignore')

class LabelEncoderTransformer:
    """
    If the target column (y) is categorical, applies label encoding; otherwise, leaves it unchanged.
    X (features) can be a numpy array or DataFrame. Returns a DataFrame with X and the (possibly encoded) y column merged.
    """
    def __init__(self, feature_names=None):
        self.label_encoder = None
        self.was_encoded = False
        

    def encode_and_merge(self, X, y: pd.Series) -> pd.DataFrame:
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
        else:
            y_new = y
            self.was_encoded = False
        # Check length consistency
        if len(X_df) != len(y_new):
            raise ValueError("Length mismatch between X and y!")
        # Merge X and y
        df_merged = X_df.copy()
        df_merged[y_new.name] = y_new
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

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        # Fit and transform features
        X_transformed = self.pipeline.fit_transform(X)
        
        # Encode and merge target
        self.label_encoder = LabelEncoderTransformer()
        df_merged = self.label_encoder.encode_and_merge(X_transformed, y)
        return df_merged

if __name__ == "__main__":
    # Example usage
    dataloader = DataLoader()
    csv_path = os.path.join(project_root, "data", "breast_cancer.csv")
    df, df_target = dataloader.load_data(csv_path)
    
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.fit_transform(df, df_target)
    print(df_processed.tail(5))



