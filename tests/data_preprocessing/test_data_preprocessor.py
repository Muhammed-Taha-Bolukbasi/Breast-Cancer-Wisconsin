import pytest
import os       
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data_preprocessing.data_preprocessor import DataPreprocessor

from typing import Optional

def split_features_target(df: pd.DataFrame, target_col: Optional[str] = None):
    # If no target_col specified, use the last column
    if target_col is None:
        target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

@pytest.fixture
def preprocessor():
    return DataPreprocessor()

@pytest.fixture
def preprocessor_test_df(test_datasets_path):
    csv_path = test_datasets_path / "preprocessor_test.csv"
    return pd.read_csv(csv_path)

def test_fit_transform_output_shape(preprocessor_test_df, preprocessor):
    X, y = split_features_target(preprocessor_test_df)
    result = preprocessor.fit_transform(X, y)
    # The output should be a DataFrame
    assert isinstance(result, pd.DataFrame)
    # The number of rows should be the same as the input
    assert result.shape[0] == preprocessor_test_df.shape[0]
    # The target column should be present
    assert 'Target_Label' in result.columns or y.name in result.columns

def test_no_missing_after_transform(preprocessor_test_df, preprocessor):
    X, y = split_features_target(preprocessor_test_df)
    result = preprocessor.fit_transform(X, y)
    # There should be no missing values in the output
    assert result.isnull().sum().sum() == 0

def test_categorical_encoding(preprocessor_test_df, preprocessor):
    X, y = split_features_target(preprocessor_test_df)
    result = preprocessor.fit_transform(X, y)
    # The number of columns should increase due to one-hot encoding (if categorical columns exist)
    assert result.shape[1] >= preprocessor_test_df.shape[1]

def test_transform_without_fit_raises(preprocessor_test_df, preprocessor):
    X, y = split_features_target(preprocessor_test_df)
    # Calling transform before fit should raise a RuntimeError
    with pytest.raises(RuntimeError, match="Pipeline is not fitted yet. Call 'fit' before 'transform'."):
        preprocessor.pipeline.transform(X)

def test_label_encoder_transformer_encodes_categorical_target(preprocessor_test_df):
    from src.data_preprocessing.data_preprocessor import LabelEncoderTransformer
    # Use a categorical y (force to string/object type for test)
    X, y = split_features_target(preprocessor_test_df)
    y_cat = y.astype(str)
    encoder = LabelEncoderTransformer()
    df_merged = encoder.encode_and_merge(X, y_cat)
    # Check that the new column is named 'Target_Label'
    assert 'Target_Label' in df_merged.columns
    # Check that the values are integer encoded (0, 1, ...)
    assert pd.api.types.is_integer_dtype(df_merged['Target_Label'])
    # Check that the number of unique values matches the number of unique categories
    assert df_merged['Target_Label'].nunique() == y_cat.nunique()

def test_label_encoder_raises_on_multi_column_dataframe(preprocessor_test_df):
    from src.data_preprocessing.data_preprocessor import LabelEncoderTransformer
    X, y = split_features_target(preprocessor_test_df)
    # Create a DataFrame with two columns for y
    y_df = pd.concat([y, y], axis=1)
    encoder = LabelEncoderTransformer()
    with pytest.raises(ValueError, match="y DataFrame has more than one column!"):
        encoder.encode_and_merge(X, y_df) # type: ignore

def test_label_encoder_accepts_single_column_dataframe(preprocessor_test_df):
    from src.data_preprocessing.data_preprocessor import LabelEncoderTransformer
    X, y = split_features_target(preprocessor_test_df)
    y_df = y.to_frame()
    encoder = LabelEncoderTransformer()
    df_merged = encoder.encode_and_merge(X, y_df)  # type: ignore
    assert 'Target_Label' in df_merged.columns or y.name in df_merged.columns

def test_label_encoder_numeric_target_not_encoded(preprocessor_test_df):
    from src.data_preprocessing.data_preprocessor import LabelEncoderTransformer
    X, y = split_features_target(preprocessor_test_df)
    y_num = pd.Series(np.arange(len(y)), name='numeric_target', index=y.index)
    encoder = LabelEncoderTransformer()
    df_merged = encoder.encode_and_merge(X, y_num)
    assert 'Target_Label' not in df_merged.columns
    assert 'numeric_target' in df_merged.columns
    assert pd.api.types.is_integer_dtype(df_merged['numeric_target'])
    assert (df_merged['numeric_target'] == np.arange(len(y))).all()

def test_label_encoder_length_mismatch_raises(preprocessor_test_df):
    from src.data_preprocessing.data_preprocessor import LabelEncoderTransformer
    X, y = split_features_target(preprocessor_test_df)
    y_short = y.iloc[:-1]
    encoder = LabelEncoderTransformer()
    with pytest.raises(ValueError, match="Length mismatch between X and y!"):
        encoder.encode_and_merge(X, y_short)




