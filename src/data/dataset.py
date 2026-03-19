from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class IndexDataset(Dataset):
    """PyTorch Dataset wrapper around a DataFrame with target and a sensitive attribute."""

    def __init__(self, df: pd.DataFrame, sens_attr_name: str, target_name: str):
        self.df = df
        self.check_binary_feature(sens_attr_name)
        self.check_binary_feature(target_name)
        self.sens_attr_name = sens_attr_name
        self.target_name = target_name

        self.features = df.drop(columns=[target_name]).values
        self.target = df[target_name].values

    @property
    def feature_columns(self):
        return list(self.df.drop(columns=[self.target_name]).columns)

    @property
    def index(self):
        return self.df.index

    def copy(self):
        return IndexDataset(df=self.df.copy(), sens_attr_name=self.sens_attr_name, target_name=self.target_name)

    def create_from_df(self, df: pd.DataFrame):
        return IndexDataset(df=df, sens_attr_name=self.sens_attr_name, target_name=self.target_name)

    def to_csv(self, save_path: Path):
        self.df.to_csv(save_path, index=True, index_label='inputId')

    def check_binary_feature(self, col_name: str):
        if col_name not in self.df.columns:
            raise ValueError(f"{col_name} column must be in dataset columns")
        if not self.df[col_name].isin([0, 1]).all():
            raise ValueError(f"{col_name} feature must be binary and takes values in 0 or 1")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.target[idx], dtype=torch.float32)

        return idx, x, y

    def get_index_col(self, col: str) -> int:
        feature_columns = self.feature_columns
        try:
            return feature_columns.index(col)
        except ValueError:
            raise ValueError(f"Column '{col}' not found in dataset columns: {feature_columns}")
