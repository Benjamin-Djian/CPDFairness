from pathlib import Path

from src.experiment.experiment import Experiment
from src.utils.file_reader import LikelihoodReader
from src.utils.logger import LoggerFactory
from src.visualisation.visualisation import DualCPDFractionPop, PlotCPDDistr

logger = LoggerFactory.get_logger(name=__name__)


class BaseExperiment(Experiment):

    def run(self, save_dir: Path):
        logger.info("===== Running Experiment =====")
        logger.info("Preprocessing data")
        train_loader, val_loader, test_loader = self.preprocess_data()

        logger.info("Training model")
        model = self.train_model(train_loader, val_loader)

        logger.info("Constructing histograms")
        filter_correct_g0, filter_correct_g1 = self.get_filter_hist(train_loader)
        histograms_g0 = self._get_histograms(model, train_loader, filter_correct_g0)
        histograms_g1 = self._get_histograms(model, train_loader, filter_correct_g1)
        if self.config["experiment"]["save_hist"]:
            self.save_histograms(save_dir, histograms_g0, histograms_g1)

        logger.info("Computing likelihood")
        filter_g0, filter_g1 = self.get_filter_likelihood(test_loader)
        likelihoods_g0_h0 = self._get_likelihood(model, test_loader, histograms_g0, filter_g0)
        likelihoods_g0_h1 = self._get_likelihood(model, test_loader, histograms_g1, filter_g0)
        likelihoods_g1_h0 = self._get_likelihood(model, test_loader, histograms_g0, filter_g1)
        likelihoods_g1_h1 = self._get_likelihood(model, test_loader, histograms_g1, filter_g1)
        if self.config["experiment"]["save_likelihood"]:
            self.save_likelihoods(save_dir, likelihoods_g0_h0, likelihoods_g0_h1, likelihoods_g1_h0, likelihoods_g1_h1)
        logger.info("End of the experiment")


class PlotExperiment(Experiment):
    """Read saved likelihoods files and plot CPD distributions"""

    def run(self, save_dir: Path):
        reader = LikelihoodReader()

        graph = DualCPDFractionPop(
            serie_plain_left=list(reader.parse_file(path=save_dir / "likelihoods_g0_h0.csv")),
            serie_plain_right=list(reader.parse_file(path=save_dir / "likelihoods_g0_h1.csv")),
            serie_dashed_left=list(reader.parse_file(path=save_dir / "likelihoods_g1_h0.csv")),
            serie_dashed_right=list(reader.parse_file(path=save_dir / "likelihoods_g1_h1.csv")),
            label_ax_left="Hist 0",
            label_ax_right="Hist 1",
            label_dashed="Group 1",
            label_plain="Group 0")
        graph.plot()
        graph.save(path=save_dir / "plot.png")

        graph_hist_h0 = PlotCPDDistr(serie_1=list(reader.parse_file(path=save_dir / "likelihoods_g0_h0.csv")),
                                     serie_2=list(reader.parse_file(path=save_dir / "likelihoods_g1_h0.csv")),
                                     legend_1="likelihoods_g0_h0", legend_2="likelihoods_g1_h0", nbr_bins=50)

        graph_hist_h0.plot()
        graph_hist_h0.save(path=save_dir / "g0.png")

        graph_hist_h1 = PlotCPDDistr(serie_1=list(reader.parse_file(path=save_dir / "likelihoods_g0_h1.csv")),
                                     serie_2=list(reader.parse_file(path=save_dir / "likelihoods_g1_h1.csv")),
                                     legend_1="likelihoods_g0_h1", legend_2="likelihoods_g1_h1", nbr_bins=50)

        graph_hist_h1.plot()
        graph_hist_h1.save(path=save_dir / "g1.png")
