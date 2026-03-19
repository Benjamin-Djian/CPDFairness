from abc import abstractmethod, ABC
from pathlib import Path
from typing import cast

import torch
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
from src.utils.config import load_config, Config
from src.utils.env import HIST_SAVE_PATH_0, HIST_SAVE_PATH_1, LH_G0_H0, LH_G0_H1, LH_G1_H0, \
    LH_G1_H1
from src.utils.file_writer import LikelihoodWriter, HistWriter
from src.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(name=__name__)


class Experiment(ABC):
    """Base class for CPD fairness experiments with model training and likelihood computation."""

    def __init__(self, config_path: Path):
        self.config: Config = load_config(config_path)
        self.data_preparator = self.config.build_preparator()
        self.model = BinaryClassificator(input_dim=self.config.model.input_dim,
                                         hidden_dims=self.config.model.layers.hidden_dims,
                                         dropout=self.config.model.layers.dropout,
                                         activation_fctn=self.config.model.activation.build(),
                                         seed=self.config.experiment.seed)

    @staticmethod
    def compute_binary_class_weights(loader: DataLoader) -> torch.Tensor:
        all_labels = []

        for _, _, labels in loader:
            all_labels.append(labels)

        all_labels = torch.cat(all_labels)
        class_counts = torch.bincount(all_labels.long())
        return class_counts.float()

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader) -> BinaryClassificator:
        model = self.model

        c_weight = None
        if self.config.training.use_class_weight:
            c_weight = self.compute_binary_class_weights(train_loader)
            logger.info(f"Balancing loss function with class weights {c_weight.tolist()}")

        loss_fctn = NLLLoss(c_weight)
        optimizer = Adam(params=model.parameters(), lr=self.config.training.learning_rate)
        trainer = Trainer(model=model, loss_fctn=loss_fctn, optimizer=optimizer)

        history = trainer.fit(train_loader, val_loader, epochs=self.config.training.epochs)
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
        filters_group_0 = [FeatureFilter(column_name=self.config.data.sens_attr,
                                         value=0,
                                         dataset=test_dataset)]
        filters_group_1 = [FeatureFilter(column_name=self.config.data.sens_attr,
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
