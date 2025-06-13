import pytest
import os       
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data_preprocessing.data_preprocessor import DataPreprocessorPipeline


@pytest.fixture
def preprocessor():
    return DataPreprocessorPipeline()

@pytest.fixture
def preprocessor_test_df(test_datasets_path):
    csv_path = test_datasets_path / "preprocessor_test.csv"
    return pd.read_csv(csv_path)

def test_build_pipeline_and_transform(tmp_path):
    """Test DataPreprocessorPipeline builds and transforms data correctly."""
    # Create a small DataFrame with missing values and categorical columns
    df = pd.DataFrame({
        'num1': [1, 2, 3, 4],
        'cat1': ['a', 'b', 'a', None],
        'num2': [10, None, 30, 40],
        'cat2': ['x', 'y', 'z', 'w']
    })
    preprocessor = DataPreprocessorPipeline()
    pipeline = preprocessor.build_pipeline(df, feature_extraction=False)
    X_processed = pipeline.fit_transform(df)
    # Should return a numpy array
    assert isinstance(X_processed, np.ndarray)
    # Should have no NaNs after transformation
    assert not np.isnan(X_processed).any()
    # Should have correct number of rows
    assert X_processed.shape[0] == df.shape[0]

def test_pipeline_feature_extraction(tmp_path):
    """Test DataPreprocessorPipeline with feature extraction enabled."""
    df = pd.DataFrame({
        'num1': [1, 2, 3, 4],
        'cat1': ['a', 'b', 'a', None],
        'num2': [10, 20, 30, 40],
        'cat2': ['x', 'y', 'z', 'w']
    })
    y = [0, 1, 0, 1]  # Dummy target for SelectKBest
    preprocessor = DataPreprocessorPipeline()
    pipeline = preprocessor.build_pipeline(df, feature_extraction=True)
    X_processed = pipeline.fit_transform(df, y)
    # Should return a numpy array
    assert isinstance(X_processed, np.ndarray)
    # Should have correct number of rows
    assert X_processed.shape[0] == df.shape[0]
    # Should have more features than without feature extraction
    pipeline_no_feat = preprocessor.build_pipeline(df, feature_extraction=False)
    X_no_feat = pipeline_no_feat.fit_transform(df)
    assert X_processed.shape[1] > X_no_feat.shape[1]

def test_drop_unnecessary_columns():
    """Test DropUnnecessaryColumns drops all-NaN and 'id' columns."""
    from src.data_preprocessing.data_preprocessor import DropUnnecessaryColumns
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'a': [1, 2, 3],
        'b': [np.nan, np.nan, np.nan],
        'c': [4, 5, 6]
    })
    dropper = DropUnnecessaryColumns()
    dropper.fit(df)
    df_dropped = dropper.transform(df)
    assert 'id' not in df_dropped.columns
    assert 'b' not in df_dropped.columns
    assert 'a' in df_dropped.columns
    assert 'c' in df_dropped.columns

def test_drop_unnecessary_columns_get_feature_names_out():
    """Test get_feature_names_out returns correct columns with and without input_features."""
    from src.data_preprocessing.data_preprocessor import DropUnnecessaryColumns
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'a': [1, 2, 3],
        'b': [np.nan, np.nan, np.nan],
        'c': [4, 5, 6]
    })
    dropper = DropUnnecessaryColumns()
    dropper.fit(df)
    # Without input_features
    out1 = dropper.get_feature_names_out()
    assert set(out1) == {'a', 'c'}
    # With input_features (simulate a subset)
    out2 = dropper.get_feature_names_out(['a', 'b', 'c'])
    assert set(out2) == {'a', 'c'}
    # With input_features that are not all in columns_to_keep_
    out3 = dropper.get_feature_names_out(['a', 'b'])
    assert set(out3) == {'a'}






