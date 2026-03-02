from abc import ABC

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.preprocessing.dataset import IndexDataset
from src.preprocessing.prepro_operations import PreprocessingOperation, MakeCategorical, Scale, ToFloat
from src.utils.env import ADULT_DATA_PATH, LAW_DATA_PATH, GERMAN_DATA_PATH, ADULT_TARGET, GERMAN_TARGET, LAW_TARGET
from src.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(name=__name__)


class Preprocessing(ABC):
    def __init__(self, df: pd.DataFrame, operations: list[PreprocessingOperation],
                 target_column: str,
                 sens_attr_column: str):
        self.df = df
        if not operations:
            logger.warning(f'Defined a empty preprocessing pipeline')
        self.operations = operations
        self.target_column = target_column
        self.sens_attr_column = sens_attr_column

    def run(self):
        if not self.operations:
            raise ValueError("Preprocessing operations are not defined yet")
        for op in self.operations:
            self.df = op.run(self.df)

    def generate_dataset(self, new_df: pd.DataFrame) -> IndexDataset:
        dataset = IndexDataset(new_df, target_column=self.target_column, sens_attr_column=self.sens_attr_column)
        return dataset

    def generate_loaders(self,
                         prop_train: float,
                         prop_valid: float,
                         batch_size: int, seed) -> tuple[DataLoader, DataLoader, DataLoader]:

        train_df, val_test_df = train_test_split(self.df, train_size=prop_train, random_state=seed, shuffle=True)
        val_df, test_df = train_test_split(val_test_df, train_size=prop_valid, random_state=seed, shuffle=True)

        train_dataset = self.generate_dataset(train_df)
        val_dataset = self.generate_dataset(val_df)
        test_dataset = self.generate_dataset(test_df)

        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        return train_loader, val_loader, test_loader


class AdultPreprocessing(Preprocessing):
    def __init__(self, sens_attr_column: str):
        operations = [MakeCategorical(lb=3, ub=5), Scale(), ToFloat()]
        df = pd.read_csv(ADULT_DATA_PATH, index_col='inputId')
        super().__init__(df, operations, ADULT_TARGET, sens_attr_column)


class GermanCreditPreprocessing(Preprocessing):
    def __init__(self, sens_attr_column: str):
        operations = [MakeCategorical(lb=3, ub=5), Scale(), ToFloat()]
        df = pd.read_csv(GERMAN_DATA_PATH, index_col='inputId')
        super().__init__(df, operations, GERMAN_TARGET, sens_attr_column)


class LawSchoolPreprocessing(Preprocessing):
    def __init__(self, sens_attr_column: str):
        operations = [MakeCategorical(lb=3, ub=5), Scale(), ToFloat()]
        df = pd.read_csv(LAW_DATA_PATH, index_col='inputId')
        super().__init__(df, operations, LAW_TARGET, sens_attr_column)
