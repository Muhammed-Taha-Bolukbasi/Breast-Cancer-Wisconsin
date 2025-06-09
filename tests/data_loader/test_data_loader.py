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
def test_load_data_with_valid_csv(data_loader, test_datasets_path, load_csv_data):
    """Test loading data from a valid CSV file."""
    file_path = test_datasets_path / "test.csv"
    data = data_loader.load_data(file_path)
    assert isinstance(data, pd.DataFrame)
    pd.testing.assert_frame_equal(data, load_csv_data)


@pytest.mark.parametrize(
    "file_name",
    [
        pytest.param("empty.csv", id="empty_csv"),
    ],
)
def test_load_data_with_empty_file(
    data_loader, test_datasets_path, file_name
):
    file_path = test_datasets_path / file_name
    
    with pytest.raises(ValueError, match="Error loading data: File is empty."):
        data_loader.load_data(file_path)

#Method Tests
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

def test_check_if_file_path_is_invalid(data_loader):
    # int tipinde dosya yolu verilirse ValueError beklenir
    with pytest.raises(ValueError):
        data_loader.load_data(12345)





