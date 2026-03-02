import csv
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch

from src.likelihood.activation_extractor import ActivationGetter
from src.utils.file_reader import (
    FileReader,
    ContribsReader,
    HistReader,
    LikelihoodReader,
)


class ConcreteFileReader(FileReader):
    """Concrete implementation of FileReader for testing."""

    def parse_file(self, path: Path):
        pass


class TestFileReader:
    """Tests for FileReader base class."""

    def test_file_reader_init(self):
        """Test FileReader initialization."""
        header: OrderedDict[str, type] = OrderedDict([("col1", int), ("col2", str)])
        reader = ConcreteFileReader(header)
        assert reader.header == header
        assert reader.sep == ","

    def test_file_reader_custom_separator(self):
        """Test FileReader with custom separator."""
        header: OrderedDict[str, type] = OrderedDict([("col1", int)])
        reader = ConcreteFileReader(header, sep=";")
        assert reader.sep == ";"

    def test_check_path_exists_raises_on_missing_path(self):
        """Test check_path_exists raises ValueError when path doesn't exist."""
        reader: ConcreteFileReader = ConcreteFileReader(OrderedDict())
        with pytest.raises(ValueError, match="does not exist"):
            reader.check_path_exists(Path("nonexistent.csv"))

    def test_check_path_exists_does_not_raise_on_existing_path(self, tmp_path):
        """Test check_path_exists does not raise when path exists."""
        existing_file = tmp_path / "test.csv"
        existing_file.touch()

        reader: ConcreteFileReader = ConcreteFileReader(OrderedDict())
        reader.check_path_exists(existing_file)


class TestContribsReader:
    """Tests for ContribsReader class."""

    def test_hist_reader_init(self):
        """Test HistReader initialization."""
        reader = ContribsReader()
        assert reader.sep == ","

    def test_parse_file_returns_activation_getter(self, tmp_path):
        """Test parse_file returns ActivationGetter."""
        reader = ContribsReader()

        csv_file = tmp_path / "contribs.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["inputId", "layerId", "nodeId", "nodeContrib"])
            writer.writerow([0, 0, 0, "0.1"])
            writer.writerow([0, 0, 1, "0.2"])
            writer.writerow([0, 0, 2, "0.3"])
            writer.writerow([1, 0, 0, "0.4"])
            writer.writerow([1, 0, 1, "0.5"])
            writer.writerow([1, 0, 2, "0.6"])

        result = reader.parse_file(csv_file)
        assert isinstance(result, ActivationGetter)
        assert torch.equal(result.indexes, torch.Tensor([0, 1]))
        assert torch.equal(result.activations, torch.Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))

    def test_parse_file_empty_raises_error(self, tmp_path):
        """Test parse_file raises error on empty file."""
        reader = ContribsReader()

        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("inputId,layerId,nodeId,nodeContrib\n")

        with pytest.raises(ValueError, match="Empty contributions file"):
            reader.parse_file(csv_file)


class TestHistReader:
    """Tests for HistReader class."""

    def test_hist_reader_init(self):
        """Test HistReader initialization."""
        reader = HistReader()
        assert reader.sep == ","

    def test_parse_file_returns_list_of_histograms(self, tmp_path):
        """Test parse_file returns list of Histogram objects."""
        from src.likelihood.histograms import Histogram
        reader = HistReader()

        csv_file = tmp_path / "hist.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["nodeId", "binId", "sigmaInterval_lb", "sigmaInterval_ub", "sigmaFreq"])
            writer.writerow([0, 0, "0.0", "0.5", "10"])
            writer.writerow([0, 1, "0.5", "1.0", "20"])
            writer.writerow([1, 0, "0.0", "1.0", "15"])

        result = reader.parse_file(csv_file)
        assert len(result) == 2
        assert all(isinstance(h, Histogram) for h in result)
        assert result[0].node_id == 0
        assert np.array_equal(result[0].bins, np.array([0., 0.5, 1.0]))
        assert np.array_equal(result[0].freq, np.array([10, 20]))
        assert result[1].node_id == 1
        assert np.array_equal(result[1].bins, np.array([0., 1.0]))
        assert np.array_equal(result[1].freq, np.array([15]))

    def test_parse_file_unibin_histogram(self, tmp_path):
        """Test parse_file creates UniBinHistogram for single bin."""
        from src.likelihood.histograms import UniBinHistogram
        reader = HistReader()

        csv_file = tmp_path / "hist.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["nodeId", "binId", "sigmaInterval_lb", "sigmaInterval_ub", "sigmaFreq"])
            writer.writerow([0, 0, "0.0", "0.0", "10"])

        result = reader.parse_file(csv_file)
        assert len(result) == 1
        assert isinstance(result[0], UniBinHistogram)
        assert result[0].node_id == 0
        assert np.array_equal(result[0].bins, np.array([0., 0.]))
        assert np.array_equal(result[0].freq, np.array([10]))

    def test_parse_file_empty_raises_error(self, tmp_path):
        """Test parse_file raises error on empty file."""
        reader = HistReader()

        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("nodeId,binId,sigmaInterval_lb,sigmaInterval_ub,sigmaFreq\n")

        with pytest.raises(ValueError, match="Empty histogram file"):
            reader.parse_file(csv_file)


class TestLikelihoodReader:
    """Tests for LikelihoodReader class."""

    def test_likelihood_reader_init(self):
        """Test LikelihoodReader initialization."""
        reader = LikelihoodReader()
        assert reader.sep == ","

    def test_parse_file_returns_list_of_likelihood_scores(self, tmp_path):
        """Test parse_file returns list of LikelihoodScore objects."""
        from src.likelihood.likelihood import LikelihoodScore
        reader = LikelihoodReader()

        csv_file = tmp_path / "lh.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["inputId", "Score"])
            writer.writerow([0, "0.95"])
            writer.writerow([1, "0.87"])
            writer.writerow([2, "0.42"])

        result = reader.parse_file(csv_file)
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(l, LikelihoodScore) for l in result)
        assert result[0].input_id == 0
        assert result[0].score == 0.95
        assert result[1].input_id == 1
        assert result[1].score == 0.87
        assert result[2].input_id == 2
        assert result[2].score == 0.42


class TestIterateRows:
    """Tests for iterate_rows method."""

    def test_iterate_rows_valid_file(self, tmp_path):
        """Test iterate_rows yields correctly parsed rows."""
        header = OrderedDict([("col1", int), ("col2", float)])
        reader = ConcreteFileReader(header)

        csv_file = tmp_path / "test.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["col1", "col2"])
            writer.writerow(["1", "1.5"])
            writer.writerow(["2", "2.5"])

        rows = list(reader.iterate_rows(csv_file))
        assert len(rows) == 2
        assert rows[0] == [1, 1.5]
        assert rows[1] == [2, 2.5]

    def test_iterate_rows_empty_file_warns(self, tmp_path):
        """Test iterate_rows warns on empty file."""
        header= OrderedDict([("col1", int)])
        reader = ConcreteFileReader(header)

        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")

        rows = list(reader.iterate_rows(csv_file))
        assert len(rows) == 0

    def test_iterate_rows_invalid_header_raises(self, tmp_path):
        """Test iterate_rows raises on mismatched header."""
        header= OrderedDict([("col1", int), ("col2", str)])
        reader = ConcreteFileReader(header)

        csv_file = tmp_path / "test.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["wrong", "header"])

        with pytest.raises(ValueError, match="Header of file is not the one expected"):
            list(reader.iterate_rows(csv_file))

    def test_iterate_rows_invalid_row_length_raises(self, tmp_path):
        """Test iterate_rows raises on invalid row length."""
        header= OrderedDict([("col1", int), ("col2", str)])
        reader = ConcreteFileReader(header)

        csv_file = tmp_path / "test.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["col1", "col2"])
            writer.writerow(["1", "2", "3"])

        with pytest.raises(ValueError, match="invalid signature body line length"):
            list(reader.iterate_rows(csv_file))
