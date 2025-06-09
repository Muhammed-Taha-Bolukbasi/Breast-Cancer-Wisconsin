import os
import sys
import pytest
from pathlib import Path  
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@pytest.fixture
def test_datasets_path():
    """Fixture to provide the path to the test datasets."""
    return Path(project_root) / "tests" / "test_datasets"

@pytest.fixture
def empty_dataset(test_datasets_path):
    """Fixture to provide an empty dataset path."""
    return pd.read_csv(test_datasets_path / "test_empty.csv")

@pytest.fixture
def load_csv_data(test_datasets_path):
    """Fixture to load CSV data for testing."""
    csv_file_path = test_datasets_path / "test.csv"
    if not csv_file_path.exists():
        raise FileNotFoundError(f"Test CSV file not found: {csv_file_path}")
    return pd.read_csv(csv_file_path)


