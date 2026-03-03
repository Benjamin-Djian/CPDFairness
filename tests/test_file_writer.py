from unittest.mock import MagicMock

import pytest

import src.utils.env as e
from src.utils.file_writer import FileWriter, ActivationWriter, HistWriter, LikelihoodWriter


class ConcreteFileWriter(FileWriter):
    """Concrete implementation of FileWriter for testing."""

    def make_iterable(self, elem) -> list[list[str]]:
        return [[str(elem)]]


class TestFileWriter:
    """Tests for FileWriter base class."""

    def test_file_writer_init(self):
        """Test FileWriter initialization."""
        header = {"col1": int, "col2": str}
        writer = ConcreteFileWriter(header)
        assert writer.header == header
        assert writer.sep == ","

    def test_file_writer_custom_separator(self):
        """Test FileWriter with custom separator."""
        header = {"col1": int}
        writer = ConcreteFileWriter(header, sep=";")
        assert writer.sep == ";"

    def test_check_path_exists_does_not_raise_on_existing_path(self, tmp_path):
        """Test check_path_exists does not raise when file exists."""
        existing_file = tmp_path / "test.csv"
        existing_file.touch()

        writer = ConcreteFileWriter({})
        writer.check_path_exists(existing_file)


class TestActivationWriter:
    """Tests for ActivationWriter class."""

    def test_activation_writer_init(self):
        """Test ActivationWriter initialization."""
        writer = ActivationWriter()
        assert writer.sep == ","

    def test_make_iterable_returns_correct_format(self):
        """Test make_iterable returns correct format."""
        writer = ActivationWriter()

        mock_getter = MagicMock()
        mock_activ_index = MagicMock()
        type(mock_activ_index).__getitem__ = lambda x, i: 0.1
        mock_activ_index.shape = MagicMock()
        mock_activ_index.shape.__getitem__ = lambda x, i: 2
        mock_getter.iterate_indexes.return_value = [(0, mock_activ_index)]

        result = list(writer.make_iterable(mock_getter))
        assert len(result) > 0
        assert all(isinstance(row, list) for row in result)


class TestHistWriter:
    """Tests for HistWriter class."""

    def test_hist_writer_init(self):
        """Test HistWriter initialization."""
        writer = HistWriter()
        assert writer.sep == ","

    def test_make_iterable_unibin_histogram(self):
        """Test make_iterable with UniBinHistogram."""
        import numpy as np
        from src.likelihood.histograms import UniBinHistogram
        writer = HistWriter()

        hist = UniBinHistogram(node_id=1, bins=np.array([0.0, 0.0]), freq=np.array([10]))

        result = list(writer.make_iterable(hist))
        assert result == [['1', '0', f'{0.0:.{e.EPSILON_PREC}f}', f'{0.0:.{e.EPSILON_PREC}f}', '10']]

    def test_make_iterable_multibins_histogram(self):
        """Test make_iterable with MultiBinsHistogram."""
        import numpy as np
        from src.likelihood.histograms import MultiBinsHistogram
        writer = HistWriter()

        hist = MultiBinsHistogram(node_id=1, bins=np.array([0.0, 0.5, 1.0]), freq=np.array([5, 6]))

        result = list(writer.make_iterable(hist))
        assert result == [['1', '0', f'{0.0:.{e.EPSILON_PREC}f}', f'{0.5:.{e.EPSILON_PREC}f}', '5'],
                          ['1', '1', f'{0.5:.{e.EPSILON_PREC}f}', f'{1.0:.{e.EPSILON_PREC}f}', '6']]

    def test_make_iterable_invalid_type_raises_error(self):
        """Test make_iterable raises error for invalid histogram type."""
        writer = HistWriter()

        mock_hist = MagicMock()
        type(mock_hist).__name__ = "InvalidType"

        with pytest.raises(ValueError, match="invalid type"):
            writer.make_iterable(mock_hist)


class TestLikelihoodWriter:
    """Tests for LikelihoodWriter class."""

    def test_likelihood_writer_init(self):
        """Test LikelihoodWriter initialization."""
        writer = LikelihoodWriter()
        assert writer.sep == ","

    def test_make_iterable_returns_correct_format(self):
        """Test make_iterable returns correct format."""
        writer = LikelihoodWriter()

        mock_likelihood = MagicMock()
        mock_likelihood.input_id = 42
        mock_likelihood.score = 0.95

        result = list(writer.make_iterable(mock_likelihood))
        assert len(result) == 1
        assert result == [["42", f"{0.95:.{e.EPSILON_PREC}f}"]]
