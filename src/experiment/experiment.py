from abc import abstractmethod, ABC
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from torch.nn import NLLLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.data.dataset import IndexDataset
from src.likelihood.activation_extractor import ActivationExtractor, ActivationFilter, ClassificationFilter, \
    FeatureFilter, PredictionFilter
from src.likelihood.histograms import HistogramConstructor, Histogram
from src.likelihood.likelihood import LikelihoodCalculator, LikelihoodScore
from src.model.binary_classificator import BinaryClassificator
from src.model.trainer import Trainer
from src.preprocessing.data_preparator import DataPreparator, AdultDataPreparator, GermanDataPreparator, \
    LawDataPreparator
from src.utils.env import REQUIRED_CONFIG_KEYS, HIST_SAVE_PATH_0, HIST_SAVE_PATH_1, LH_G0_H0, LH_G0_H1, LH_G1_H0, \
    LH_G1_H1
from src.utils.file_writer import LikelihoodWriter, HistWriter
from src.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(name=__name__)


class Experiment(ABC):
    def __init__(self, config_path: Path):
        self.config = self.load_config(config_path)
        self.validate_config()

    @staticmethod
    def load_config(config_file: Path) -> dict[str, Any]:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def validate_config(self) -> None:
        for section, keys in REQUIRED_CONFIG_KEYS.items():
            if section not in self.config:
                raise ValueError(f"Missing required section: '{section}'")
            for key in keys:
                if key not in self.config[section]:
                    raise ValueError(f"Missing required key: '{section}.{key}'")

        for section, keys in self.config.items():
            if section in REQUIRED_CONFIG_KEYS:
                for key in keys:
                    if key not in REQUIRED_CONFIG_KEYS[section]:
                        logger.warning(
                            f"Unknown key '{key}' in section '{section}' - this key is not required and will be ignored")
            else:
                logger.warning(
                    f"Unknown section '{section}' this section is not required and will be ignored")

    def get_preparator(self) -> DataPreparator:
        if self.config["data"]["name"] == "adult":
            preparator = AdultDataPreparator(self.config["data"]["sens_attr"])
        elif self.config["data"]["name"] == "german":
            preparator = GermanDataPreparator(self.config["data"]["sens_attr"])
        elif self.config["data"]["name"] == "law":
            preparator = LawDataPreparator(self.config["data"]["sens_attr"])
        else:
            raise ValueError(f"Unknown dataset name : {self.config["data"]["name"]}")
        return preparator

    @staticmethod
    def compute_binary_class_weights(loader: DataLoader) -> torch.Tensor:
        all_labels = []

        for _, _, labels in loader:
            all_labels.append(labels)

        all_labels = torch.cat(all_labels)
        class_counts = torch.bincount(all_labels.long())
        return class_counts.float()

    def define_model(self) -> BinaryClassificator:
        return BinaryClassificator(input_dim=self.config["model"]["input_dim"],
                                   hidden_dims=self.config["model"]["hidden_dims"],
                                   negative_slope=self.config["model"]["neg_slope"],
                                   dropout=self.config["model"]["dropout"],
                                   seed=self.config["experiment"]["seed"])

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader) -> BinaryClassificator:
        model = self.define_model()

        c_weight = None
        if self.config["training"]["class_weight"]:
            c_weight = self.compute_binary_class_weights(train_loader)
            logger.info(f"Balancing loss function with class weights {c_weight.tolist()}")

        loss_fctn = NLLLoss(c_weight)
        optimizer = Adam(params=model.parameters(), lr=self.config["training"]["learning_rate"])
        trainer = Trainer(model=model, loss_fctn=loss_fctn, optimizer=optimizer)

        history = trainer.fit(train_loader, val_loader, epochs=self.config["training"]["epochs"])
        logger.info(f"Model trained with val_accuracy of {history.val_acc[-1] * 100:.3f}%")
        logger.debug(history)
        return model

    @staticmethod
    def get_filter_hist(train_loader: DataLoader) -> tuple[list[ActivationFilter], list[ActivationFilter]]:
        train_dataset = cast(IndexDataset, train_loader.dataset)  # Casting to avoid type checks
        filters_correct_group_0 = [ClassificationFilter(keep_correct=True),
                                   PredictionFilter(column_name=train_dataset.target_name,
                                                    value=0,
                                                    dataset=train_dataset)]
        filters_correct_group_1 = [ClassificationFilter(keep_correct=True),
                                   PredictionFilter(column_name=train_dataset.target_name,
                                                    value=1,
                                                    dataset=train_dataset)]
        return filters_correct_group_0, filters_correct_group_1

    @staticmethod
    def _get_histograms(model: BinaryClassificator,
                        train_loader: DataLoader,
                        filters: list[ActivationFilter] | None = None) -> list[Histogram]:
        extractor = ActivationExtractor(model)
        histograms = []
        for node_id in range(model.last_hidden_dim):
            hist_construct = HistogramConstructor(node_id, extractor)
            hist = hist_construct.compute_hist(train_loader, filters=filters)
            histograms.append(hist)
        return histograms

    @staticmethod
    def save_histograms(save_dir: Path, hist_g0: list[Histogram], hist_g1: list[Histogram]) -> None:
        writer = HistWriter()
        writer.write(path=save_dir / HIST_SAVE_PATH_0, content=hist_g0)
        writer.write(path=save_dir / HIST_SAVE_PATH_1, content=hist_g1)

    def get_filter_likelihood(self, test_loader: DataLoader) -> tuple[list[FeatureFilter], list[FeatureFilter]]:
        test_dataset = cast(IndexDataset, test_loader.dataset)
        filters_group_0 = [FeatureFilter(column_name=self.config["data"]["sens_attr"],
                                         value=0,
                                         dataset=test_dataset)]
        filters_group_1 = [FeatureFilter(column_name=self.config["data"]["sens_attr"],
                                         value=1,
                                         dataset=test_dataset)]

        return filters_group_0, filters_group_1

    @staticmethod
    def _get_likelihood(model: BinaryClassificator,
                        test_loader: DataLoader,
                        histograms: list[Histogram],
                        filters: list[ActivationFilter] | None = None) -> list[LikelihoodScore]:

        extractor = ActivationExtractor(model)
        calculator = LikelihoodCalculator(extractor)
        likelihoods = calculator.compute_likelihood(test_loader, histograms, filters=filters)
        return likelihoods

    @staticmethod
    def save_likelihoods(save_dir: Path,
                         likelihoods_g0_h0: list[LikelihoodScore],
                         likelihoods_g0_h1: list[LikelihoodScore],
                         likelihoods_g1_h0: list[LikelihoodScore],
                         likelihoods_g1_h1: list[LikelihoodScore]) -> None:
        writer = LikelihoodWriter()
        writer.write(path=save_dir / LH_G0_H0, content=likelihoods_g0_h0)
        writer.write(path=save_dir / LH_G0_H1, content=likelihoods_g0_h1)
        writer.write(path=save_dir / LH_G1_H0, content=likelihoods_g1_h0)
        writer.write(path=save_dir / LH_G1_H1, content=likelihoods_g1_h1)

    @abstractmethod
    def run(self, save_dir: Path):
        pass
