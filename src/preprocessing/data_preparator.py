from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader

from src.data.data_reader import DataReader, AdultDataReader, LawDataReader, GermanDataReader
from src.data.dataset import IndexDataset
from src.preprocessing.prepro_operations import PreprocessingOperation, MakeCategorical, Scale, ToFloat, TargetAtTheEnd
from src.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(name=__name__)


class DataPreparator:
    def __init__(self, data_reader: DataReader, ops: list[PreprocessingOperation]):
        self.data_reader = data_reader
        self.df = data_reader.read_data()
        self.pipeline: Pipeline = Pipeline([(op.__class__.__name__, op) for op in ops])

    def split_sets(self, prop_train: float, prop_valid: float, seed) -> tuple[IndexDataset, IndexDataset, IndexDataset]:
        train_df, val_test_df = train_test_split(self.df, train_size=prop_train, random_state=seed, shuffle=True)
        val_df, test_df = train_test_split(val_test_df, train_size=prop_valid, random_state=seed, shuffle=True)
        train_dataset = self.to_index_dataframe(train_df)
        val_dataset = self.to_index_dataframe(val_df)
        test_dataset = self.to_index_dataframe(test_df)
        return train_dataset, val_dataset, test_dataset

    def to_index_dataframe(self, df: pd.DataFrame) -> IndexDataset:
        dataset = IndexDataset(df,
                               target_name=self.data_reader.target_name,
                               sens_attr_name=self.data_reader.sens_attr_name)
        return dataset

    @staticmethod
    def to_dataloaders(index_data: IndexDataset, batch_size: int, seed):
        torch.manual_seed(seed)
        return DataLoader(index_data, batch_size=batch_size, shuffle=True)

    def run(self, prop_train: float, prop_valid: float, batch_size: int, seed, save_dir: Path = None):
        train_dataset, val_dataset, test_dataset = self.split_sets(prop_train, prop_valid, seed)

        self.pipeline.fit(train_dataset)

        preprocessed_train = self.pipeline.transform(train_dataset)
        preprocessed_val = self.pipeline.transform(val_dataset)
        preprocessed_test = self.pipeline.transform(test_dataset)

        if save_dir:
            preprocessed_train.to_csv(save_dir / "train.csv")
            preprocessed_val.to_csv(save_dir / "val.csv")
            preprocessed_test.to_csv(save_dir / "test.csv")

        train_loader = self.to_dataloaders(preprocessed_train, batch_size, seed)
        val_loader = self.to_dataloaders(preprocessed_val, batch_size, seed)
        test_loader = self.to_dataloaders(preprocessed_test, batch_size, seed)

        return train_loader, val_loader, test_loader


class AdultDataPreparator(DataPreparator):
    def __init__(self, sens_attr_name):
        data_reader = AdultDataReader(sens_attr_name)
        operations = [MakeCategorical(lb=3, ub=5), Scale(), ToFloat(), TargetAtTheEnd()]
        super().__init__(data_reader, ops=operations)


class GermanDataPreparator(DataPreparator):
    def __init__(self, sens_attr_name):
        data_reader = GermanDataReader(sens_attr_name)
        operations = [MakeCategorical(lb=3, ub=5), Scale(), ToFloat(), TargetAtTheEnd()]
        super().__init__(data_reader, ops=operations)


class LawDataPreparator(DataPreparator):
    def __init__(self, sens_attr_name):
        data_reader = LawDataReader(sens_attr_name)
        operations = [MakeCategorical(lb=3, ub=5), Scale(), ToFloat(), TargetAtTheEnd()]
        super().__init__(data_reader, ops=operations)
