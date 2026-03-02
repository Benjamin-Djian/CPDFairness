import csv
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Iterable

import src.utils.env as e
from src.likelihood.activation_extractor import ActivationGetter
from src.likelihood.histograms import Histogram, UniBinHistogram, MultiBinsHistogram
from src.likelihood.likelihood import LikelihoodScore
from src.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(name=__name__)


class FileWriter(ABC):
    def __init__(self, header: dict[str, type], sep: str = ","):
        self.header = header
        self.sep = sep

    @staticmethod
    def check_path_exists(path: Path):
        if path.exists():
            logger.warning(f"FileWriter: overwriting path {path}")

    @abstractmethod
    def make_iterable(self, elem) -> Iterable[Iterable[str]]:
        pass

    def write(self, path: Path, content: Iterable):
        with open(path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(self.header.keys())

            for elem in content:
                csvwriter.writerows(self.make_iterable(elem))


class ActivationWriter(FileWriter):
    def __init__(self, sep: str = ','):
        super().__init__(header=e.CONTRIBS_HEADER, sep=sep)

    def make_iterable(self, activation_getter: ActivationGetter) -> Iterable[Iterable[str]]:
        to_write = []
        for index, activ_index in activation_getter.iterate_indexes():
            for node_id in range(activ_index.shape[0]):
                to_write.append([str(index),
                                 str(node_id),
                                 f'{activ_index[node_id]:.{e.EPSILON_PREC}f}'])
        return to_write


class HistWriter(FileWriter):
    def __init__(self, sep: str = ','):
        super().__init__(header=e.HIST_HEADER, sep=sep)

    def make_iterable(self, hist: Histogram) -> Iterable[Iterable[str]]:
        if isinstance(hist, UniBinHistogram):
            return [[str(hist.node_id),
                     "0",
                     f'{hist.lower_bound:.{e.EPSILON_PREC}f}',
                     f'{hist.lower_bound:.{e.EPSILON_PREC}f}',
                     str(hist.freq[0])]]
        elif isinstance(hist, MultiBinsHistogram):
            to_write = []
            prev_bin = hist.bins[0]
            for i, current_bin in enumerate(hist.bins[1:]):
                to_write.append([str(hist.node_id),
                                 str(i),
                                 f'{prev_bin:.{e.EPSILON_PREC}f}',
                                 f'{current_bin:.{e.EPSILON_PREC}f}',
                                 str(hist.freq[i])])

                prev_bin = current_bin

            return to_write

        else:
            raise ValueError("Got invalid type of histogram")


class LikelihoodWriter(FileWriter):
    def __init__(self, sep: str = ','):
        super().__init__(header=e.LH_HEADER, sep=sep)

    def make_iterable(self, likelihood: LikelihoodScore) -> Iterable[Iterable[str]]:
        return [[str(likelihood.input_id), f'{likelihood.score:.{e.EPSILON_PREC}f}']]
