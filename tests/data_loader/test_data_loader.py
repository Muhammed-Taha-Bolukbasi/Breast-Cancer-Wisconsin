import pytest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data_loader.data_loader import DataLoader
import re
import pandas as pd

@pytest.fixture
def data_loader():
    """Fixture to create a DataLoader instance."""
    return DataLoader()

# Expected value test
def test_load_data_with_valid_csv(test_datasets_path, load_csv_data):
    """Test loading data from a valid CSV file."""
    file_path = test_datasets_path / "test.csv"
    # Use a target_col that exists in the test.csv, e.g., 'col_0'
    data_loader = DataLoader(target_col="col_0")
    data, data_target = data_loader.load_data(file_path)
    assert isinstance(data, pd.DataFrame)
    assert isinstance(data_target, pd.Series)

# Method Tests
@pytest.mark.parametrize("extension, file_name",  [
        (".csv", "data.csv"),         
        (".csv", "data_with.parquet.csv"),
        (".csv", "data_with.csv")        
])    
def test_check_if_file_extension_returns_correct_result(data_loader, extension, file_name):
    assert data_loader._check_if_file_extension_suported(file_name) == extension


@pytest.mark.parametrize("extension", [ 
    ".txt",
    ".xlsx",
    ".json",
    ".xml",
    ".html",
    ".docx",
    ".parquet"
])
def test_check_if_file_extension_raises_error_for_unsupported_extension(data_loader, extension):
    with pytest.raises(ValueError, match=re.escape(f"Error loading data: File extension {extension} is not supported, expected one of ['.csv'].")):
        data_loader._check_if_file_extension_suported(f"data{extension}")

def test_check_if_filename_is_empty(data_loader):
    with pytest.raises(ValueError):
        data_loader._check_if_file_extension_suported("")

def test_check_if_file_isnot_existent(data_loader):
    with pytest.raises(FileNotFoundError):
        data_loader._validate_file_path("non_existent_file.csv")

def test_check_if_file_is_empty(data_loader, test_datasets_path):
    file_path = test_datasets_path / "empty.csv"
    with pytest.raises(ValueError, match=re.escape("Error loading data: File is empty.")):
        data_loader.load_data(file_path)

def test_dataloader_uses_provided_target_col():
    """Test that DataLoader uses the provided target_col argument directly."""
    loader = DataLoader(target_col="my_target")
    assert loader.target_col == "my_target"


def test_dataloader_reads_target_col_from_conf_yaml(monkeypatch, tmp_path):
    """Test that DataLoader reads target_col from conf.yaml if not provided."""
    import yaml
    # Create a temporary conf.yaml
    conf_path = tmp_path / "conf.yaml"
    conf_content = {"target_col": "yaml_target"}
    conf_path.write_text(yaml.dump(conf_content))

    # Monkeypatch project_root to tmp_path
    monkeypatch.setattr("src.data_loader.data_loader.project_root", str(tmp_path))

    loader = DataLoader(target_col=None)
    assert loader.target_col == "yaml_target"


def test_dataloader_raises_if_no_target_col_in_conf(monkeypatch, tmp_path):
    """Test that DataLoader sets target_col to None if not in conf.yaml and not provided."""
    import yaml
    conf_path = tmp_path / "conf.yaml"
    conf_content = {"not_target_col": "something_else"}
    conf_path.write_text(yaml.dump(conf_content))
    monkeypatch.setattr("src.data_loader.data_loader.project_root", str(tmp_path))

    loader = DataLoader(target_col=None)
    assert loader.target_col is None

def test_validate_file_path_invalid_type(data_loader):
    """Test that ValueError is raised if file_path is not str or Path."""
    from src.data_loader.error_messages import DataReadingErrorMessages
    for invalid in [123, 3.14, [], {}, None, True]:
        with pytest.raises(ValueError, match=DataReadingErrorMessages.INVALID_FILE_PATH_TYPE.value.format(type=type(invalid).__name__)):
            data_loader._validate_file_path(invalid)

