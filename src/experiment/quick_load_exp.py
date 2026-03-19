from pathlib import Path

import torch

from src.experiment.experiment import Experiment
from src.utils.env import MODEL_SAVE_PATH, HIST_SAVE_PATH_0, HIST_SAVE_PATH_1
from src.utils.file_reader import HistReader
from src.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(name=__name__)


class LoadModelExperiment(Experiment):

    def run(self, save_dir: Path):
        logger.info("===== Running Experiment =====")
        logger.info("Preprocessing data")
        save_data_dir = save_dir if self.config.experiment.save.dataset else None
        train_loader, val_loader, test_loader = self.data_preparator.run(prop_train=self.config.data.train_split,
                                                                         prop_valid=self.config.data.valid_split,
                                                                         batch_size=self.config.data.batch_size,
                                                                         seed=self.config.experiment.seed,
                                                                         save_dir=save_data_dir)

        logger.info("Loading model")
        model = self.model
        model.load_state_dict(torch.load(save_dir / MODEL_SAVE_PATH))

        logger.info("Constructing histograms")
        filter_correct_g0, filter_correct_g1 = self.get_filter_hist(train_loader)
        histograms_g0 = self._get_histograms(model, train_loader, filter_correct_g0)
        histograms_g1 = self._get_histograms(model, train_loader, filter_correct_g1)
        if self.config.experiment.save.histogram:
            self.save_histograms(save_dir, histograms_g0, histograms_g1)

        logger.info("Computing likelihood")
        filter_g0, filter_g1 = self.get_filter_likelihood(test_loader)
        likelihoods_g0_h0 = self._get_likelihood(model, test_loader, histograms_g0, filter_g0)
        likelihoods_g0_h1 = self._get_likelihood(model, test_loader, histograms_g1, filter_g0)
        likelihoods_g1_h0 = self._get_likelihood(model, test_loader, histograms_g0, filter_g1)
        likelihoods_g1_h1 = self._get_likelihood(model, test_loader, histograms_g1, filter_g1)
        if self.config.experiment.save.likelihood:
            self.save_likelihoods(save_dir, likelihoods_g0_h0, likelihoods_g0_h1, likelihoods_g1_h0, likelihoods_g1_h1)
        logger.info("End of the experiment")


class LoadModelHistExperiment(Experiment):
    def run(self, save_dir: Path):
        logger.info("===== Running Experiment =====")
        logger.info("Preprocessing data")
        save_data_dir = save_dir if self.config.experiment.save.dataset else None
        train_loader, val_loader, test_loader = self.data_preparator.run(prop_train=self.config.data.train_split,
                                                                         prop_valid=self.config.data.valid_split,
                                                                         batch_size=self.config.data.batch_size,
                                                                         seed=self.config.experiment.seed,
                                                                         save_dir=save_data_dir)

        logger.info("Loading model")
        model = self.model
        model.load_state_dict(torch.load(save_dir / MODEL_SAVE_PATH))

        logger.info("Loading histograms")
        hist_reader = HistReader()
        histograms_g0 = hist_reader.parse_file(save_dir / HIST_SAVE_PATH_0)
        histograms_g1 = hist_reader.parse_file(save_dir / HIST_SAVE_PATH_1)

        logger.info("Computing likelihood")
        filter_g0, filter_g1 = self.get_filter_likelihood(test_loader)
        likelihoods_g0_h0 = self._get_likelihood(model, test_loader, histograms_g0, filter_g0)
        likelihoods_g0_h1 = self._get_likelihood(model, test_loader, histograms_g1, filter_g0)
        likelihoods_g1_h0 = self._get_likelihood(model, test_loader, histograms_g0, filter_g1)
        likelihoods_g1_h1 = self._get_likelihood(model, test_loader, histograms_g1, filter_g1)
        if self.config.experiment.save.likelihood:
            self.save_likelihoods(save_dir, likelihoods_g0_h0, likelihoods_g0_h1, likelihoods_g1_h0, likelihoods_g1_h1)
        logger.info("End of the experiment")
