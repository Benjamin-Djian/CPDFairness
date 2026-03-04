from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from torch.utils.data import DataLoader

from src.experiment.experiment import Experiment


@pytest.fixture
def valid_config_file(tmp_path):
    """Create a valid config file for testing."""
    config_content = """
experiment:
  seed: 42
  save_hist: false
  save_likelihood: true

data:
  name: adult
  sens_attr: sex
  prop_train: 0.8
  prop_valid: 0.1
  batch_size: 32

model:
  input_dim: 103
  hidden_dims:
    - 128
    - 64
  neg_slope: 0.2
  dropout: 0.0

training:
  learning_rate: 0.001
  epochs: 100
  class_weight: false
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def config_missing_section(tmp_path):
    """Create a config with missing section."""
    config_content = """
experiment:
  seed: 42
  save_hist: false
  save_likelihood: true

data:
  name: adult
  sens_attr: sex
  prop_train: 0.8
  prop_valid: 0.1
  batch_size: 32
"""
    config_file = tmp_path / "config_missing.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def config_missing_key(tmp_path):
    """Create a config with missing key."""
    config_content = """
experiment:
  seed: 42

data:
  name: adult
  sens_attr: sex

model:
  input_dim: 103
  hidden_dims:
    - 128

training:
  optimizer: Adam
  criterion: NLL
  learning_rate: 0.001
"""
    config_file = tmp_path / "config_missing_key.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def config_unknown_key(tmp_path):
    """Create a config with unknown key in known section."""
    config_content = """
experiment:
  seed: 42
  save_hist: false
  save_likelihood: true
  unknown_key: "this is unknown"

data:
  name: adult
  sens_attr: sex
  prop_train: 0.8
  prop_valid: 0.1
  batch_size: 32

model:
  input_dim: 103
  hidden_dims:
    - 128
    - 64
  neg_slope: 0.2
  dropout: 0.0

training:
  learning_rate: 0.001
  epochs: 100
  class_weight: false
"""
    config_file = tmp_path / "config_unknown_key.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def config_unknown_section(tmp_path):
    """Create a config with unknown section."""
    config_content = """
experiment:
  seed: 42
  save_hist: false
  save_likelihood: true

data:
  name: adult
  sens_attr: sex
  prop_train: 0.8
  prop_valid: 0.1
  batch_size: 32

model:
  input_dim: 103
  hidden_dims:
    - 128
    - 64
  neg_slope: 0.2
  dropout: 0.0

training:
  learning_rate: 0.001
  epochs: 100
  class_weight: false

logging:
  level: DEBUG
  format: "%(levelname)s: %(message)s"
"""
    config_file = tmp_path / "config_unknown_section.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def mock_preprocessing():
    """Mock Preprocessing object."""
    mock = Mock()
    mock.run = Mock()
    mock.generate_loaders = Mock(return_value=(
        Mock(spec=DataLoader),
        Mock(spec=DataLoader),
        Mock(spec=DataLoader)
    ))
    return mock


class TestLoadConfig:
    """Tests for load_config static method."""

    def test_load_config_valid_file(self, valid_config_file):
        """Test loading a valid YAML config file."""
        config = Experiment.load_config(valid_config_file)

        assert isinstance(config, dict)
        assert "experiment" in config
        assert config["experiment"]["seed"] == 42
        assert config["data"]["name"] == "adult"

    def test_load_config_file_not_found(self):
        """Test loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            Experiment.load_config(Path("nonexistent.yaml"))


class TestValidateConfig:
    """Tests for validate_config method."""

    def test_validate_config_valid(self, valid_config_file):
        """Test validation passes for valid config."""
        exp = Experiment.__new__(Experiment)
        exp.config = Experiment.load_config(valid_config_file)

        exp.validate_config()

    def test_validate_config_missing_section(self, config_missing_section):
        """Test validation fails when section is missing."""
        exp = Experiment.__new__(Experiment)
        exp.config = Experiment.load_config(config_missing_section)

        with pytest.raises(ValueError, match="Missing required section"):
            exp.validate_config()

    def test_validate_config_missing_key(self, config_missing_key):
        """Test validation fails when key is missing."""
        exp = Experiment.__new__(Experiment)
        exp.config = Experiment.load_config(config_missing_key)

        with pytest.raises(ValueError, match="Missing required key"):
            exp.validate_config()

    def test_validate_config_unknown_key_in_known_section(self, config_unknown_key, caplog):
        """Test validation logs warning for unknown key in known section."""
        exp = Experiment.__new__(Experiment)
        exp.config = Experiment.load_config(config_unknown_key)

        with caplog.at_level("WARNING"):
            exp.validate_config()

        assert any("Unknown key 'unknown_key'" in record.message for record in caplog.records)

    def test_validate_config_unknown_section(self, config_unknown_section, caplog):
        """Test validation logs warning for unknown section."""
        exp = Experiment.__new__(Experiment)
        exp.config = Experiment.load_config(config_unknown_section)

        with caplog.at_level("WARNING"):
            exp.validate_config()

        assert any("Unknown section 'logging'" in record.message for record in caplog.records)

    def test_validate_config_no_warning_for_valid(self, valid_config_file, caplog):
        """Test validation does not log warning for valid config."""
        exp = Experiment.__new__(Experiment)
        exp.config = Experiment.load_config(valid_config_file)

        with caplog.at_level("WARNING"):
            exp.validate_config()

        assert not any("Unknown key" in record.message for record in caplog.records)
        assert not any("Unknown section" in record.message for record in caplog.records)


class TestGetPrepro:
    """Tests for get_prepro method."""

    @patch('src.experiment.experiment.AdultPreprocessing')
    @patch('src.experiment.experiment.GermanCreditPreprocessing')
    @patch('src.experiment.experiment.LawSchoolPreprocessing')
    def test_get_prepro_all_datasets(self, mock_law, mock_german, mock_adult):
        """Test get_prepro returns correct preprocessing for each dataset."""
        mock_adult.return_value = Mock()
        mock_german.return_value = Mock()
        mock_law.return_value = Mock()

        exp = Experiment.__new__(Experiment)
        exp.config = {
            "data": {"name": "adult", "sens_attr": "sex"},
            "model": {},
            "training": {},
            "experiment": {}
        }
        assert exp.get_prepro() == mock_adult.return_value

        exp.config["data"]["name"] = "german"
        assert exp.get_prepro() == mock_german.return_value

        exp.config["data"]["name"] = "law"
        assert exp.get_prepro() == mock_law.return_value

    def test_get_prepro_unknown_dataset(self, valid_config_file):
        """Test get_prepro raises error for unknown dataset."""
        exp = Experiment.__new__(Experiment)
        exp.config = Experiment.load_config(valid_config_file)
        exp.config["data"]["name"] = "unknown"

        with pytest.raises(ValueError, match="Unknown dataset name"):
            exp.get_prepro()


class TestSaveLikelihoods:
    """Tests for save_likelihoods static method."""

    @patch('src.experiment.experiment.LikelihoodWriter')
    def test_save_likelihoods(self, mock_writer_class, tmp_path):
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
