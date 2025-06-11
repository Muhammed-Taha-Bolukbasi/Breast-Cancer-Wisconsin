import pandas as pd
import os
from pathlib import Path
import yaml
from typing import Tuple, Union
from typing import Optional, Union
from .error_messages import DataReadingErrorMessages as EM, SUPPORTED_FILE_EXTENSIONS
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

data_reader_functions = {
    ".csv": pd.read_csv
}

class DataLoader:
    """A class to load data from various file formats."""

    def __init__(self, target_col = None):
        """
        Initializes the DataLoader class.
        """
        if target_col is None:
            with open(os.path.join(project_root, "conf.yaml"), "r") as f:
                config = yaml.safe_load(f)
            self.target_col = config.get("target_col")
        else:
            self.target_col = target_col

    def load_data(self, file_path: Union[str, Path]) -> Tuple[pd.DataFrame, pd.Series]:
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

        X, y = self._split_dataframe(data)
        return X, y

    def _split_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Splits the DataFrame into features and target.

        Args:
            df (pd.DataFrame): The DataFrame to split.

        Returns:
            tuple[pd.DataFrame, pd.Series]: A tuple containing the features DataFrame and the target Series.
        """
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        return X, y


    def _validate_file_path(self, file_path: Union[str, Path]) -> None:
        """
        Validates the file path.

        Args:
            file_path (Union[str, Path]): The file path to validate.

        Raises:
            ValueError: If the file path is not a string or Path object.
        """
        if not isinstance(file_path, (str, Path)):
            raise ValueError(EM.INVALID_FILE_PATH_TYPE.value.format(type=type(file_path).__name__))
        
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
            raise ValueError(EM.EXT_NOT_SUPPORTED.value.format(ext=ext, supported_extensions=SUPPORTED_FILE_EXTENSIONS))
        
        return ext