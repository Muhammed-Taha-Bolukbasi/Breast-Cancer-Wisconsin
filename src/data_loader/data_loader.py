import pandas as pd
import os
from pathlib import Path
import logging
from typing import Optional, Union
from .error_messages import DataReadingErrorMessages as EM, SUPPORTED_FILE_EXTENSIONS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

data_reader_functions = {
    ".csv": pd.read_csv
}

class DataLoader:
    """A class to load data from various file formats."""

    def load_data(self, file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
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
            logger.error(EM.EMPTY_DATA_FILE.value)
            raise ValueError(EM.EMPTY_DATA_FILE.value)

        return data
    

    def _validate_file_path(self, file_path: Union[str, Path]) -> None:
        """
        Validates the file path.

        Args:
            file_path (Union[str, Path]): The file path to validate.

        Raises:
            ValueError: If the file path is not a string or Path object.
        """
        if not isinstance(file_path, (str, Path)):
            logger.error(EM.INVALID_FILE_PATH_TYPE.value.format(type=type(file_path).__name__))
            raise ValueError(EM.INVALID_FILE_PATH_TYPE.value.format(type=type(file_path).__name__))
        
        if not os.path.exists(file_path):
            logger.error(EM.FILE_NOT_FOUND.value.format(file_path=file_path))
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
            logger.error(EM.EXT_NOT_SUPPORTED.value.format(ext=ext, supported_extensions=SUPPORTED_FILE_EXTENSIONS))
            raise ValueError(EM.EXT_NOT_SUPPORTED.value.format(ext=ext, supported_extensions=SUPPORTED_FILE_EXTENSIONS))
        
        return ext