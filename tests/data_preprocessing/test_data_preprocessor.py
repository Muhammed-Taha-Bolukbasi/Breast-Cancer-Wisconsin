import pytest
import os       
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data_preprocessing.data_preprocessor import DataPreprocessor

@pytest.fixture
def data_loader():
    """Fixture to create a DataLoader instance."""
    return DataPreprocessor()

@pytest.fixture
def preprocessor_test_df(test_datasets_path):
    csv_path = test_datasets_path / "preprocessor_test.csv"
    return pd.read_csv(csv_path)

def test_fit_transform_output_shape(preprocessor_test_df, data_loader):
    result = data_loader.fit_transform(preprocessor_test_df)
    # The output should be a numpy array
    assert isinstance(result, np.ndarray)
    # The number of rows should be the same as the input
    assert result.shape[0] == preprocessor_test_df.shape[0]

def test_no_missing_after_transform(preprocessor_test_df, data_loader):
    result = data_loader.fit_transform(preprocessor_test_df)
    # There should be no missing values in the output
    assert np.isnan(result).sum() == 0

def test_categorical_encoding(preprocessor_test_df, data_loader):
    result = data_loader.fit_transform(preprocessor_test_df)
    # The number of columns should increase due to one-hot encoding
    assert result.shape[1] > preprocessor_test_df.shape[1]

def test_transform_without_fit_raises(preprocessor_test_df, data_loader):
    # Calling transform before fit should raise a RuntimeError
    with pytest.raises(RuntimeError, match="Pipeline is not fitted yet. Call 'fit' before 'transform'."):
        data_loader.transform(preprocessor_test_df)




