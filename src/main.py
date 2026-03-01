import argparse
from pathlib import Path

from src.experiment.experiment import Experiment
from src.visualisation.visualisation import PlotCPDFractionPop

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str, help="Path to exp config")
    args = parser.parse_args()
    config_path = Path(args.config)
    exp = Experiment(config_path)

    a, b, c, d = exp.run()
    graph_h0 = PlotCPDFractionPop(serie_1=a, serie_2=c, size=(7, 7),
                                  title="Group 0 & 1 from H0",
                                  label_1="Group 0",
                                  label_2="Group 1")

    graph_h1 = PlotCPDFractionPop(serie_1=b, serie_2=d, size=(7, 7),
                                  title="Group 0 & 1 from H1",
                                  label_1="Group 0",
                                  label_2="Group 1")

    graph_h0.plot()

    graph_h1.plot()