from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from src.likelihood.activation_extractor import ActivationExtractor, ClassificationFilter
from src.likelihood.histograms import HistogramConstructor, Histogram
from src.likelihood.likelihood import LikelihoodCalculator, LikelihoodScore
from src.model.classificator import Classificator
from src.model.trainer import Trainer
from src.preprocessing.preprocessing import AdultPreprocessing, GermanCreditPreprocessing, LawSchoolPreprocessing, \
    Preprocessing


class Experiment:
    def __init__(self, config_file: Path):
        self.config = self.load_config(config_file)

    @staticmethod
    def load_config(config_file: Path):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def get_prepro(self) -> Preprocessing:
        if self.config["data"]["name"] == "adult":
            prepro = AdultPreprocessing()
        elif self.config["data"]["name"] == "german":
            prepro = GermanCreditPreprocessing()
        elif self.config["data"]["name"] == "law":
            prepro = LawSchoolPreprocessing()
        else:
            raise ValueError(f"Unknown dataset name : {self.config["data"]["name"]}")
        return prepro

    def preprocess_data(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        prepro = self.get_prepro()
        prepro.run()

        train_loader, val_loader, test_loader = prepro.generate_loaders(prop_train=self.config["data"]["prop_train"],
                                                                        prop_valid=self.config["data"]["prop_valid"],
                                                                        seed=self.config["seed"],
                                                                        batch_size=self.config["data"]["batch_size"])

        return train_loader, val_loader, test_loader

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader) -> Classificator:
        model = Classificator(input_dim=self.config["model"]["input_dim"],
                              hidden_dims=self.config["model"]["hidden_dims"],
                              num_classes=self.config["model"]["num_classes"],
                              negative_slope=self.config["model"]["neg_slope"],
                              dropout=self.config["model"]["dropout"],
                              seed=self.config["seed"])

        trainer = Trainer(model=model, optimizer=..., criterion=...)

        trainer.fit(train_loader, val_loader, epochs=self.config["model"]["epochs"])
        return model

    @staticmethod
    def get_histograms(model: Classificator,
                       train_loader: DataLoader) -> list[Histogram]:
        extractor = ActivationExtractor(model)
        histograms = []
        for node_id in range(model.last_hidden_dim):
            hist_construct = HistogramConstructor(node_id, extractor)
            hist = hist_construct.get_hist(train_loader, filters=[ClassificationFilter(keep_correct=True)])
            histograms.append(hist)
        return histograms

    @staticmethod
    def compute_likelihood(model: Classificator,
                           test_loader: DataLoader,
                           histograms: list[Histogram]) -> list[LikelihoodScore]:

        extractor = ActivationExtractor(model)
        calculator = LikelihoodCalculator(extractor)
        likelihoods = calculator.get_likelihood(test_loader, histograms)
        return likelihoods

    def run(self):
        train_loader, val_loader, test_loader = self.preprocess_data()
        model = self.train_model(train_loader, val_loader)
        histograms = self.get_histograms(model, train_loader)
        likelihoods = self.compute_likelihood(model, test_loader, histograms)
        return likelihoods
