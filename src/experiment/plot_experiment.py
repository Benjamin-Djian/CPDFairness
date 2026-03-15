from pathlib import Path

from src.experiment.experiment import Experiment
from src.utils.env import LH_G0_H0, LH_G0_H1, LH_G1_H0, LH_G1_H1
from src.utils.file_reader import LikelihoodReader
from src.visualisation.visualisation import DualCPDFractionPop, PlotCPDDistr


class PlotExperiment(Experiment):
    """Read saved likelihoods files and plot CPD distributions"""

    def run(self, save_dir: Path):
        reader = LikelihoodReader()

        graph = DualCPDFractionPop(
            serie_plain_left=list(reader.parse_file(path=save_dir / LH_G0_H0)),
            serie_plain_right=list(reader.parse_file(path=save_dir / LH_G0_H1)),
            serie_dashed_left=list(reader.parse_file(path=save_dir / LH_G1_H0)),
            serie_dashed_right=list(reader.parse_file(path=save_dir / LH_G1_H1)),
            label_ax_left="Hist 0",
            label_ax_right="Hist 1",
            label_dashed="Group 1",
            label_plain="Group 0")
        graph.plot()
        graph.save(path=save_dir / "plot.png")

        graph_hist_h0 = PlotCPDDistr(serie_1=list(reader.parse_file(path=save_dir / LH_G0_H0)),
                                     serie_2=list(reader.parse_file(path=save_dir / LH_G1_H0)),
                                     legend_1=LH_G0_H0, legend_2=LH_G1_H0, nbr_bins=50)

        graph_hist_h0.plot()
        graph_hist_h0.save(path=save_dir / "g0.png")

        graph_hist_h1 = PlotCPDDistr(serie_1=list(reader.parse_file(path=save_dir / LH_G0_H1)),
                                     serie_2=list(reader.parse_file(path=save_dir / LH_G1_H1)),
                                     legend_1=LH_G0_H1, legend_2=LH_G1_H1, nbr_bins=50)

        graph_hist_h1.plot()
        graph_hist_h1.save(path=save_dir / "g1.png")
