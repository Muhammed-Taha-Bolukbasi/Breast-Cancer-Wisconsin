import pandas as pd
import os
import sys
from pathlib import Path
import yaml
from typing import Tuple, Union
from typing import Optional, Union, Any
from sklearn.preprocessing import LabelEncoder

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)
from src.data_loader.error_messages import (
    DataReadingErrorMessages as EM,
    SUPPORTED_FILE_EXTENSIONS,
)
from sklearn.model_selection import train_test_split

data_reader_functions = {".csv": pd.read_csv}


class DataLoader:
    """A class to load data from various file formats."""

    def __init__(self, target_col=None):
        """
        Initializes the DataLoader class.
        """
        # Always load config so self.config is available
        with open(os.path.join(project_root, "conf.yaml"), "r") as f:
            self.config = yaml.safe_load(f)
        if target_col is None:
            self.target_col = self.config.get("target_col")
        else:
            self.target_col = target_col

    def load_data(self, file_path: Union[str, Path]) -> Any:
        """
        Loads data from a  file.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the data from the  file.

        Raises:
            ValueError: If the file is empty, or if the file extension is not supported.
            FileNotFoundError: If the file does not exist.
            TypeError: If the file path is not a string or Path object.
        """

        self._validate_file_path(file_path)

        ext = self._check_if_file_extension_suported(file_path)

        reader_func = data_reader_functions.get(ext)
        data: pd.DataFrame = reader_func(file_path)  # type: ignore
        if data.empty:
            raise ValueError(EM.EMPTY_DATA_FILE.value)

        y_enc, le = label_encode_series(
            data[self.target_col]
        )  # Encode target column if categorical

        X_train, X_test, y_train, y_test = train_test_split(
            data.drop(columns=[self.target_col]),
            y_enc,
            test_size=self.config.get("test_size", 0.2),  # Default test size is 20%
            random_state=42,
        )

        # Return a label_map (dict) instead of the LabelEncoder object
        if le is not None:
            label_map = {i: label for i, label in enumerate(le.classes_)}
        else:
            label_map = None

        return X_train, X_test, y_train, y_test, label_map

    def _validate_file_path(self, file_path: Union[str, Path]) -> None:
        """
        Validates the file path.

        Args:
            file_path (Union[str, Path]): The file path to validate.

        Raises:
            ValueError: If the file path is not a string or Path object.
        """
        if not isinstance(file_path, (str, Path)):
            raise ValueError(
                EM.INVALID_FILE_PATH_TYPE.value.format(type=type(file_path).__name__)
            )

        if not os.path.exists(file_path):
            raise FileNotFoundError(EM.FILE_NOT_FOUND.value.format(file_path=file_path))

    def _check_if_file_extension_suported(self, file_path: Union[str, Path]) -> str:
        """
        Checks if the file extension is supported.

        Args:
            file_path (Union[str, Path]): The file path to check.

        Raises:
            ValueError: If the file extension is not supported.
        """
        ext = Path(file_path).suffix
        if ext not in SUPPORTED_FILE_EXTENSIONS:
            raise ValueError(
                EM.EXT_NOT_SUPPORTED.value.format(
                    ext=ext, supported_extensions=SUPPORTED_FILE_EXTENSIONS
                )
            )

        return ext


def label_encode_series(y):
    """
    If the input series is categorical, encode it with LabelEncoder.
    Returns: encoded array, label_encoder (or None if not encoded)
    """
    if y.dtype == "object" or str(y.dtype).startswith("category"):
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        return y_enc, le
    else:
        return y, None

