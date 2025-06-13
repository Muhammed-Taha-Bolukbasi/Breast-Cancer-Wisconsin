import pytest
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data_loader.data_loader import DataLoader

@pytest.fixture
def data_loader():
    return DataLoader(target_col="col_0")

def test_load_data_valid_csv(tmp_path):
    df = pd.DataFrame({
        'col_0': ['A', 'B', 'A', 'B', 'A', 'B'],
        'col_1': [1, 2, 3, 4, 5, 6],
        'col_2': [10, 20, 30, 40, 50, 60]
    })
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)
    loader = DataLoader(target_col="col_0")
    X_train, X_test, y_train, y_test, label_map = loader.load_data(csv_path)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, (pd.Series, np.ndarray))
    assert isinstance(y_test, (pd.Series, np.ndarray))
    assert 'col_0' not in X_train.columns
    assert 'col_0' not in X_test.columns
    assert isinstance(label_map, dict)
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)

def test_load_data_raises_for_missing_file():
    loader = DataLoader(target_col="col_0")
    with pytest.raises(FileNotFoundError):
        loader.load_data("not_a_real_file.csv")

def test_load_data_raises_for_empty_file(tmp_path):
    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("")
    loader = DataLoader(target_col="col_0")
    with pytest.raises(ValueError):
        loader.load_data(empty_path)

def test_label_map_none_for_non_categorical(tmp_path):
    df = pd.DataFrame({'col_0': [1,2,3,4], 'col_1': [5,6,7,8]})
    csv_path = tmp_path / "test_numeric.csv"
    df.to_csv(csv_path, index=False)
    loader = DataLoader(target_col="col_0")
    X_train, X_test, y_train, y_test, label_map = loader.load_data(csv_path)
    assert label_map is None

def test_invalid_file_path_type():
    loader = DataLoader(target_col="col_0")
    # Pass a list (invalid type) to trigger ValueError
    with pytest.raises(ValueError):
        loader.load_data(["not_a_path"])    # type: ignore

def test_unsupported_extension(tmp_path):
    loader = DataLoader(target_col="col_0")
    fake_file = tmp_path / "data.unsupported"
    fake_file.write_text("dummy")
    with pytest.raises(ValueError):
        loader.load_data(fake_file)

def test_target_col_none_uses_config(tmp_path, monkeypatch):
    # Write a config file with a known target_col
    config_path = tmp_path / "conf.yaml"
    config_path.write_text("target_col: col_0\n")
    # Patch os.path.join to return our config path for conf.yaml
    orig_join = os.path.join
    def fake_join(*a):
        if a[-1] == "conf.yaml":
            return str(config_path)
        return orig_join(*a)
    monkeypatch.setattr(os.path, "join", fake_join)
    df = pd.DataFrame({'col_0': [1,2,3], 'col_1': [4,5,6]})
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)
    loader = DataLoader()
    X_train, X_test, y_train, y_test, label_map = loader.load_data(csv_path)
    assert loader.target_col == "col_0"

def test_empty_data_file_raises_value_error(tmp_path):
    """Test that loading a CSV with only headers (no data) raises ValueError for empty data file."""
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("col_0,col_1\n")  # Only header, no data
    loader = DataLoader(target_col="col_0")
    with pytest.raises(ValueError) as excinfo:
        loader.load_data(empty_csv)
    assert "empty" in str(excinfo.value).lower() or "no data" in str(excinfo.value).lower()

