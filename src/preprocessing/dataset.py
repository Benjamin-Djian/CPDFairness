import pandas as pd
import torch
from torch.utils.data import Dataset


class IndexDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df
        self.target_column = target_column

        self.features = df.drop(columns=[target_column]).values
        self.target = df[target_column].values

    @property
    def feature_columns(self):
        return list(self.df.drop(columns=[self.target_column]).columns)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.target[idx], dtype=torch.float32)

        return idx, x, y

    def get_feature_columns(self) -> list[str]:
        return self.feature_columns

    def get_index_col(self, col: str) -> int:
        feature_columns = self.get_feature_columns()
        try:
            return feature_columns.index(col)
        except ValueError:
            raise ValueError(f"Column '{col}' not found in dataset columns: {feature_columns}")
