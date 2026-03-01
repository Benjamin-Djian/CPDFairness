import pandas as pd
import torch
from torch.utils.data import Dataset


class IndexDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df
        self.target_column = target_column

        self.features = df.drop(columns=[target_column]).values
        self.target = df[target_column].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.target[idx], dtype=torch.float32)

        return idx, x, y
