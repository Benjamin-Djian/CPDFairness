import argparse
from pathlib import Path

from src.experiment.base_experiment import BaseExperiment, PlotExperiment
from src.utils.logger import LoggerFactory


def main(config_file: Path):
    experiment_dir = config_file.parent

    base_exp = BaseExperiment(config_file)
    base_exp.run(experiment_dir)

    plot_exp = PlotExperiment(config_file)
    plot_exp.run(experiment_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, nargs='+', help="List of path to exp config")
    args = parser.parse_args()
    config_paths = [Path(c) for c in args.config]
    for c_path in config_paths:
        exp_dir = c_path.resolve().parent
        log_file = exp_dir / "info.log"
        LoggerFactory.reset()
        LoggerFactory(log_file=log_file)
        logger = LoggerFactory.get_logger(__name__)
        logger.info(f"Starting experiment in {exp_dir}")
        main(exp_dir / "config.yaml")
