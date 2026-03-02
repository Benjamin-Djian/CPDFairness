import argparse
from pathlib import Path

from src.experiment.experiment import Experiment
from src.utils.logger import LoggerFactory


def main(config_file: Path):
    experiment_dir = config_file.parent
    log_file = experiment_dir / "info.log"
    LoggerFactory(log_file=log_file)
    logger = LoggerFactory.get_logger(__name__)
    logger.info(f"Starting experiment in {experiment_dir}")
    exp = Experiment(config_path)
    exp.run(config_path.parent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str, help="Path to exp config")
    args = parser.parse_args()
    config_path = Path(args.config)
    main(config_path)
