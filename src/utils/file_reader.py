import csv
from pathlib import Path

import src.utils.env as e
from src.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(name=__name__)


class FileReader:
    def __init__(self, header: dict[str, type], sep: str = ","):
        self.header = header
        self.sep = sep

    @staticmethod
    def check_path_exists(path: Path):
        if not path.exists():
            raise ValueError(f'ERROR file_reader: Path {path} does not exist')

    def iterate_rows(self, path):
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

                for col_name, col_type_func in self.header.items():
                    row[col_name] = col_type_func(row[col_name])
                yield row


class ContribsReader(FileReader):
    def __init__(self, sep: str = ','):
        super().__init__(header=e.CONTRIBS_HEADER, sep=sep)


class HistReader(FileReader):
    def __init__(self, sep: str = ','):
        super().__init__(header=e.HIST_HEADER, sep=sep)


class LikelihoodReader(FileReader):
    def __init__(self, sep: str = ','):
        super().__init__(header=e.LH_HEADER, sep=sep)
