from pathlib import Path
from typing import Any

import yaml
from torch.utils.data import DataLoader

from src.likelihood.activation_extractor import ActivationExtractor, ActivationFilter, ClassificationFilter, \
    FeatureFilter
from src.likelihood.histograms import HistogramConstructor, Histogram
from src.likelihood.likelihood import LikelihoodCalculator, LikelihoodScore
from src.model.binary_classificator import BinaryClassificator
from src.model.trainer import Trainer
from src.preprocessing.preprocessing import AdultPreprocessing, GermanCreditPreprocessing, LawSchoolPreprocessing, \
    Preprocessing
from src.utils.env import REQUIRED_CONFIG_KEYS
from src.utils.file_writer import LikelihoodWriter
from src.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(name=__name__)


class Experiment:
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

    def get_prepro(self) -> Preprocessing:
        if self.config["data"]["name"] == "adult":
            prepro = AdultPreprocessing(self.config["data"]["sens_attr"])
        elif self.config["data"]["name"] == "german":
            prepro = GermanCreditPreprocessing(self.config["data"]["sens_attr"])
        elif self.config["data"]["name"] == "law":
            prepro = LawSchoolPreprocessing(self.config["data"]["sens_attr"])
        else:
            raise ValueError(f"Unknown dataset name : {self.config["data"]["name"]}")
        return prepro

    def preprocess_data(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        prepro = self.get_prepro()
        prepro.run()

        train_loader, val_loader, test_loader = prepro.generate_loaders(prop_train=self.config["data"]["prop_train"],
                                                                        prop_valid=self.config["data"]["prop_valid"],
                                                                        seed=self.config["experiment"]["seed"],
                                                                        batch_size=self.config["data"]["batch_size"])

        return train_loader, val_loader, test_loader

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader) -> BinaryClassificator:
        model = BinaryClassificator(input_dim=self.config["model"]["input_dim"],
                                    hidden_dims=self.config["model"]["hidden_dims"],
                                    negative_slope=self.config["model"]["neg_slope"],
                                    dropout=self.config["model"]["dropout"],
                                    seed=self.config["experiment"]["seed"])

        trainer = Trainer(model=model, learning_rate=self.config["training"]["learning_rate"])

        history = trainer.fit(train_loader, val_loader, epochs=self.config["training"]["epochs"])
        logger.info(f"Model trained with val_accuracy of {history.val_acc[-1]*100:.3f}%")
        logger.debug(history)
        return model

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

    def get_hist_by_sens_group(self,
                               model: BinaryClassificator,
                               train_loader: DataLoader) -> tuple[list[Histogram], list[Histogram]]:

        filters_group_0 = [ClassificationFilter(keep_correct=True),
                           FeatureFilter(column_name=self.config["data"]["sens_attr"],
                                         value=0,
                                         dataset=train_loader.dataset)]
        filters_group_1 = [ClassificationFilter(keep_correct=True),
                           FeatureFilter(column_name=self.config["data"]["sens_attr"],
                                         value=1,
                                         dataset=train_loader.dataset)]

        histograms_group_0 = self._get_histograms(model, train_loader, filters_group_0)
        histograms_group_1 = self._get_histograms(model, train_loader, filters_group_1)
        return histograms_group_0, histograms_group_1

    @staticmethod
    def _get_likelihood(model: BinaryClassificator,
                        test_loader: DataLoader,
                        histograms: list[Histogram],
                        filters: list[ActivationFilter] | None = None) -> list[LikelihoodScore]:

        extractor = ActivationExtractor(model)
        calculator = LikelihoodCalculator(extractor)
        likelihoods = calculator.compute_likelihood(test_loader, histograms, filters=filters)
        return likelihoods

    def get_likelihood_by_sens_group(
            self,
            model: BinaryClassificator,
            test_loader: DataLoader,
            histograms_group_0: list[Histogram],
            histograms_group_1: list[Histogram]) -> tuple[
        list[LikelihoodScore],
        list[LikelihoodScore],
        list[LikelihoodScore],
        list[LikelihoodScore]]:

        filters_group_0 = [FeatureFilter(column_name=self.config["data"]["sens_attr"],
                                         value=0,
                                         dataset=test_loader.dataset)]
        filters_group_1 = [FeatureFilter(column_name=self.config["data"]["sens_attr"],
                                         value=1,
                                         dataset=test_loader.dataset)]

        likelihoods_g0_h0 = self._get_likelihood(model, test_loader, histograms_group_0, filters_group_0)
        likelihoods_g0_h1 = self._get_likelihood(model, test_loader, histograms_group_1, filters_group_0)
        likelihoods_g1_h0 = self._get_likelihood(model, test_loader, histograms_group_0, filters_group_1)
        likelihoods_g1_h1 = self._get_likelihood(model, test_loader, histograms_group_1, filters_group_1)

        return likelihoods_g0_h0, likelihoods_g0_h1, likelihoods_g1_h0, likelihoods_g1_h1

    @staticmethod
    def save_likelihoods(save_dir: Path,
                         likelihoods_g0_h0: list[LikelihoodScore],
                         likelihoods_g0_h1: list[LikelihoodScore],
                         likelihoods_g1_h0: list[LikelihoodScore],
                         likelihoods_g1_h1: list[LikelihoodScore]):
        writer = LikelihoodWriter()
        writer.write(path=save_dir / "likelihoods_g0_h0.csv", content=likelihoods_g0_h0)
        writer.write(path=save_dir / "likelihoods_g0_h1.csv", content=likelihoods_g0_h1)
        writer.write(path=save_dir / "likelihoods_g1_h0.csv", content=likelihoods_g1_h0)
        writer.write(path=save_dir / "likelihoods_g1_h1.csv", content=likelihoods_g1_h1)

    def run(self, save_path: Path):
        logger.info("===== Running Experiment =====")
        logger.info("Preprocessing data")
        train_loader, val_loader, test_loader = self.preprocess_data()

        logger.info("Training model")
        model = self.train_model(train_loader, val_loader)

        logger.info("Constructing histograms")
        histograms_g0, histograms_g1 = self.get_hist_by_sens_group(model, train_loader)

        logger.info("Computing likelihood")
        likelihoods_groups = self.get_likelihood_by_sens_group(model, test_loader, histograms_g0, histograms_g1)
        likelihoods_g0_h0, likelihoods_g0_h1, likelihoods_g1_h0, likelihoods_g1_h1 = likelihoods_groups
        self.save_likelihoods(save_path, likelihoods_g0_h0, likelihoods_g0_h1, likelihoods_g1_h0, likelihoods_g1_h1)
