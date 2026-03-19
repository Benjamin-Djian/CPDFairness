from abc import ABC
from pathlib import Path

import pandas as pd

from src.utils.env import ADULT_DATA_PATH, LAW_DATA_PATH, GERMAN_DATA_PATH, ADULT_TARGET, GERMAN_TARGET, LAW_TARGET


class DataReader(ABC):
    """Base class for reading CSV datasets with target and sensitive attribute columns."""
    def __init__(self, data_path: Path, index_column: str, target_name: str, sens_attr_name: str):
        self.data_path = data_path
        self.index_column = index_column
        self.target_name = target_name
        self.sens_attr_name = sens_attr_name

    def read_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path, index_col=self.index_column)
        if self.target_name not in df.columns:
            raise ValueError(f"ERROR DataReader: target {self.target_name} not in dataframe")
        if self.sens_attr_name not in df.columns:
            raise ValueError(f"ERROR DataReader: Sensitive attribute {self.sens_attr_name} not in dataframe")
        return df


class AdultDataReader(DataReader):
    """DataReader for Adult dataset with income prediction task."""

    def __init__(self, sens_attr_name: str):
        super().__init__(data_path=ADULT_DATA_PATH,
                         index_column='inputId',
                         target_name=ADULT_TARGET,
                         sens_attr_name=sens_attr_name)


class GermanDataReader(DataReader):
    """DataReader for German Credit dataset with credit risk prediction task."""

    def __init__(self, sens_attr_name: str):
        super().__init__(data_path=GERMAN_DATA_PATH,
                         index_column='inputId',
                         target_name=GERMAN_TARGET,
                         sens_attr_name=sens_attr_name)


class LawDataReader(DataReader):
    """DataReader for Law School dataset with bar passage prediction task."""

    def __init__(self, sens_attr_name: str):
        super().__init__(data_path=LAW_DATA_PATH,
                         index_column='inputId',
                         target_name=LAW_TARGET,
                         sens_attr_name=sens_attr_name)
