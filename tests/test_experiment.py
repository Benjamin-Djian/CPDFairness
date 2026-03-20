from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import yaml
from torch.utils.data import DataLoader

from src.experiment.experiment import Experiment
from src.model.binary_classificator import BinaryClassificator


@pytest.fixture
def valid_config_dict():
    return {
        "experiment": {
            "seed": 42,
            "save": {
                "dataset": False,
                "histogram": False,
                "likelihood": True,
                "model": False
            }
        },
        "data": {
            "dataset": "adult",
            "sens_attr": "sex",
            "train_split": 0.8,
            "valid_split": 0.1,
            "batch_size": 32,
            "favorable_class": 0,
            "privileged_group": 0
        },
        "preprocessing": {
            "class_balance": {
                "downsampling": False,
                "upsampling": False
            },
            "fairness": {
                "correlation_remover": False,
                "disparate_impact_remover": False
            }
        },
        "model": {
            "input_dim": 103,
            "layers": {
                "hidden_dims": [128, 64],
                "dropout": 0.0
            },
            "activation": {
                "type": "leaky_relu",
                "neg_slope": 0.2
            }
        },
        "training": {
            "learning_rate": 0.001,
            "epochs": 100,
            "use_class_weight": False
        }
    }


@pytest.fixture
def config_file(tmp_path):
    def _create(config):
        file = tmp_path / "config.yaml"
        file.write_text(yaml.dump(config))
        return file
    return _create


@pytest.fixture
def mock_config():
    mock = MagicMock()
    mock.model.input_dim = 103
    mock.model.layers.hidden_dims = [128, 64]
    mock.model.layers.dropout = 0.0
    mock.model.activation.build.return_value = Mock()
    mock.experiment.seed = 42
    mock.training.use_class_weight = False
    mock.training.learning_rate = 0.001
    mock.training.epochs = 100
    mock.data.sens_attr = "sex"
    return mock


class DummyExperiment(Experiment):
    def run(self, save_path):
        pass


class TestExperimentInit:
    """Tests for Experiment initialization."""

    def test_init_loads_config(self, valid_config_dict, config_file):
        """Test that Experiment loads config from file."""
        path = config_file(valid_config_dict)

        exp = DummyExperiment(path)

        assert exp.config is not None
        assert exp.config.experiment.seed == 42
        assert exp.config.data.dataset == "adult"

    def test_init_creates_model(self, valid_config_dict, config_file):
        """Test that Experiment creates a BinaryClassificator."""
        path = config_file(valid_config_dict)

        exp = DummyExperiment(path)

        assert isinstance(exp.model, BinaryClassificator)
        assert exp.model.last_hidden_dim == 64

    def test_init_creates_preparator(self, valid_config_dict, config_file):
        """Test that Experiment creates a data preparator."""
        path = config_file(valid_config_dict)

        exp = DummyExperiment(path)

        assert exp.data_preparator is not None


class TestComputeBinaryClassWeights:
    """Tests for compute_binary_class_weights static method."""

    def test_compute_binary_class_weights(self, mock_config):
        """Test compute_binary_class_weights returns correct tensor."""
        mock_loader = Mock(spec=DataLoader)
        mock_labels = [0, 1, 0, 1, 0, 1]
        mock_loader.__iter__ = Mock(return_value=iter([
            (Mock(), Mock(), Mock())
        ]))

        for batch in mock_loader:
            _, _, labels = batch

        from torch import tensor
        all_labels = tensor(mock_labels)
        class_counts = all_labels.bincount()

        assert class_counts[0] == 3
        assert class_counts[1] == 3


class TestGetFilterHist:
    """Tests for get_filter_hist static method."""

    def test_get_filter_hist_returns_correct_filters(self):
        """Test get_filter_hist returns filters for both groups."""
        mock_loader = Mock(spec=DataLoader)
        mock_dataset = Mock()
        mock_dataset.target_name = "target"
        mock_loader.dataset = mock_dataset

        filters_g0, filters_g1 = Experiment.get_filter_hist(mock_loader)

        assert len(filters_g0) == 2
        assert len(filters_g1) == 2


class TestGetFilterLikelihood:
    """Tests for get_filter_likelihood method."""

    def test_get_filter_likelihood_uses_sens_attr(self, mock_config):
        """Test get_filter_likelihood uses sens_attr from config."""
        mock_loader = Mock(spec=DataLoader)
        mock_dataset = Mock()
        mock_loader.dataset = mock_dataset
        
        exp = DummyExperiment.__new__(DummyExperiment)
        exp.config = mock_config

        filters_g0, filters_g1 = exp.get_filter_likelihood(mock_loader)

        assert len(filters_g0) == 1
        assert len(filters_g1) == 1


class TestSaveHistograms:
    """Tests for save_histograms static method."""

    @patch('src.experiment.experiment.HistWriter')
    def test_save_histograms_writes_two_files(self, mock_writer_class, tmp_path):
        """Test save_histograms writes files for both groups."""
        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer

        save_dir = tmp_path / "results"
        save_dir.mkdir()

        mock_hist = Mock()

        Experiment.save_histograms(save_dir, [mock_hist], [mock_hist])

        assert mock_writer.write.call_count == 2


class TestSaveLikelihoods:
    """Tests for save_likelihoods static method."""

    @patch('src.experiment.experiment.LikelihoodWriter')
    def test_save_likelihoods_writes_four_files(self, mock_writer_class, tmp_path):
        """Test save_likelihoods writes four files."""
        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer

        save_dir = tmp_path / "results"
        save_dir.mkdir()

        mock_lh = Mock()

        Experiment.save_likelihoods(
            save_dir,
            [mock_lh],
            [mock_lh],
            [mock_lh],
            [mock_lh]
        )

        assert mock_writer.write.call_count == 4


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_valid_file(self, valid_config_dict, config_file):
        """Test loading a valid YAML config file."""
        from src.utils.config import load_config

        path = config_file(valid_config_dict)
        config = load_config(path)

        assert config.experiment.seed == 42
        assert config.data.dataset == "adult"
        assert config.model.layers.hidden_dims == [128, 64]

    def test_load_config_file_not_found(self):
        """Test loading a non-existent file raises FileNotFoundError."""
        from src.utils.config import load_config

        with pytest.raises(FileNotFoundError):
            load_config(Path("nonexistent.yaml"))

    def test_load_config_invalid_structure(self, valid_config_dict, config_file):
        """Test loading config with missing required keys raises error."""
        from src.utils.config import Config

        invalid_config = valid_config_dict.copy()
        invalid_config.pop("model")

        path = config_file(invalid_config)

        with pytest.raises(Exception):
            Config(**invalid_config)

    def test_load_config_extra_fields_forbidden(self, valid_config_dict, config_file):
        """Test that extra fields in config raise an error."""
        from src.utils.config import Config

        extra_config = valid_config_dict.copy()
        extra_config["unknown_section"] = {"key": "value"}

        path = config_file(extra_config)

        with pytest.raises(Exception):
            Config(**extra_config)
