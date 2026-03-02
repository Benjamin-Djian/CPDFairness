import csv
from abc import abstractmethod, ABC
from collections import OrderedDict
from itertools import groupby
from pathlib import Path

import numpy as np
import torch

import src.utils.env as e
from src.likelihood.activation_extractor import ActivationGetter
from src.likelihood.histograms import Histogram, UniBinHistogram, MultiBinsHistogram
from src.likelihood.likelihood import LikelihoodScore
from src.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(name=__name__)


class FileReader(ABC):
    def __init__(self, header: OrderedDict[str, type], sep: str = ","):
        self.header = header
        self.sep = sep

    @staticmethod
    def check_path_exists(path: Path):
        if not path.exists():
            raise ValueError(f'ERROR file_reader: Path {path} does not exist')

    @abstractmethod
    def parse_file(self, path: Path):
        pass

    def iterate_rows(self, path: Path):
        self.check_path_exists(path)

        with open(path, "r") as file:
            reader = csv.reader(file, delimiter=self.sep)

            try:
                file_header = next(reader)
            except StopIteration:
                logger.warning(f'Empty file at {path}')
                return

            if file_header != list(self.header.keys()):
                raise ValueError(
                    f'ERROR file_reader: Header of file is not the one expected. Got {file_header} instead of {self.header}')

            for no_row, row in enumerate(reader):
                if len(row) != len(self.header):
                    raise ValueError(
                        f'ERROR file_reader: invalid signature body line length {len(row)} at line {no_row} ({len(self.header)} expected)')

                for index, (_, col_type_func) in enumerate(self.header.items()):
                    row[index] = col_type_func(row[index])
                yield row


class ContribsReader(FileReader):
    def __init__(self, sep: str = ','):
        super().__init__(header=e.CONTRIBS_HEADER, sep=sep)

    def parse_file(self, path: Path) -> ActivationGetter:
        rows = self.iterate_rows(path)

        indexes = []
        activations = []

        for key, group in groupby(rows, key=lambda r: r[e.CONTRIB_INPUT_POS]):
            indexes.append(key)
            activations.append([row[e.CONTRIB_VALUE_POS] for row in group])

        if not indexes:
            raise ValueError(f"Empty contributions file at {path}")

        return ActivationGetter(
            indexes=torch.tensor(indexes),
            activations=torch.tensor(activations),
        )


class HistReader(FileReader):
    def __init__(self, sep: str = ','):
        super().__init__(header=e.HIST_HEADER, sep=sep)

    def parse_file(self, path: Path) -> list[Histogram]:
        rows = self.iterate_rows(path)

        histograms = []

        for node_id, group in groupby(rows, key=lambda r: r[e.HIST_NODE_POS]):
            bins = []
            freq = []
            last_row = None

            for row in group:
                last_row = row
                lb = row[e.HIST_LB_POS]
                bins.append(lb)
                freq.append(row[e.HIST_FREQ_POS])

            bins.append(last_row[e.HIST_UB_POS])
            if len(freq) == 1:
                histo = UniBinHistogram(node_id=int(node_id),
                                        bins=np.array(bins, dtype=float),
                                        freq=np.array(freq, dtype=float))
            else:
                histo = MultiBinsHistogram(node_id=int(node_id),
                                           bins=np.array(bins, dtype=float),
                                           freq=np.array(freq, dtype=float))

            histograms.append(histo)

        if not histograms:
            raise ValueError(f"Empty histogram file at {path}")

        return histograms


class LikelihoodReader(FileReader):
    def __init__(self, sep: str = ','):
        super().__init__(header=e.LH_HEADER, sep=sep)

    def parse_file(self, path: Path) -> list[LikelihoodScore]:
        res = []
        for row in self.iterate_rows(path):
            res.append(LikelihoodScore(input_id=row[e.LH_INPUT_ID_POS],
                                       score=row[e.LH_SCORE_POS]))
        return res
