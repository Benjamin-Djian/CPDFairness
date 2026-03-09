from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from torch.utils.data import DataLoader

from src.experiment.experiment import Experiment


@pytest.fixture
def base_config():
    return {
        "experiment": {
            "seed": 42,
            "save_hist": False,
            "save_likelihood": True
        },
        "data": {
            "name": "adult",
            "sens_attr": "sex",
            "prop_train": 0.8,
            "prop_valid": 0.1,
            "batch_size": 32
        },
        "model": {
            "input_dim": 103,
            "hidden_dims": [128, 64],
            "neg_slope": 0.2,
            "dropout": 0.0
        },
        "training": {
            "learning_rate": 0.001,
            "epochs": 100,
            "class_weight": False
        }
    }


@pytest.fixture
def config_file(tmp_path):
    def _create(config):
        file = tmp_path / "config.yaml"
        import yaml
        file.write_text(yaml.dump(config))
        return file

    return _create


@pytest.fixture
def config_missing_key(base_config):
    cfg = base_config.copy()
    del cfg["model"]["input_dim"]
    return cfg


@pytest.fixture
def config_missing_section(base_config):
    cfg = base_config.copy()
    del cfg["model"]
    return cfg


@pytest.fixture
def config_unknown_section(base_config):
    cfg = base_config.copy()
    cfg["logging"] = {"level": "DEBUG"}
    return cfg


@pytest.fixture
def config_unknown_key(base_config):
    cfg = base_config.copy()
    cfg["experiment"]["unknown_key"] = "test"
    return cfg


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


class DummyExperiment(Experiment):
    def run(self, save_path):
        pass


@pytest.fixture
def experiment(base_config, config_file):
    path = config_file(base_config)
    return DummyExperiment(path)


class TestLoadConfig:
    """Tests for load_config static method."""

    def test_load_config_valid_file(self, base_config, config_file):
        """Test loading a valid YAML config file."""
        path = config_file(base_config)
        config = Experiment.load_config(path)

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

    def test_validate_config_valid(self, experiment):
        experiment.validate_config()

    def test_validate_config_missing_section(self, config_missing_section, config_file):
        exp = DummyExperiment.__new__(DummyExperiment)
        exp.config = config_missing_section

        with pytest.raises(ValueError, match="Missing required section"):
            exp.validate_config()

    def test_validate_config_missing_key(self, config_missing_key):
        """Test validation fails when key is missing."""
        exp = DummyExperiment.__new__(DummyExperiment)
        exp.config = config_missing_key

        with pytest.raises(ValueError, match="Missing required key"):
            exp.validate_config()

    def test_validate_config_unknown_key_in_known_section(self, config_unknown_key, caplog):
        """Test validation logs warning for unknown key in known section."""
        exp = DummyExperiment.__new__(DummyExperiment)
        exp.config = config_unknown_key

        with caplog.at_level("WARNING"):
            exp.validate_config()

        assert any("Unknown key 'unknown_key'" in record.message for record in caplog.records)

    def test_validate_config_unknown_section(self, config_unknown_section, caplog):
        exp = DummyExperiment.__new__(DummyExperiment)
        exp.config = config_unknown_section

        with caplog.at_level("WARNING"):
            exp.validate_config()

        assert "Unknown section 'logging'" in caplog.text

    def test_validate_config_no_warning_for_valid(self, base_config, caplog):
        """Test validation does not log warning for valid config."""
        exp = DummyExperiment.__new__(DummyExperiment)
        exp.config = base_config

        with caplog.at_level("WARNING"):
            exp.validate_config()

        assert not any("Unknown key" in record.message for record in caplog.records)
        assert not any("Unknown section" in record.message for record in caplog.records)


class TestGetPrepro:
    """Tests for get_prepro method."""

    @pytest.mark.parametrize(
        "dataset, sens_attr",
        [
            ("adult", "sex"),
            ("german", "Personal_status_and_sex"),
            ("law", "male"),
        ]
    )
    @patch("src.experiment.experiment.AdultPreprocessing")
    @patch("src.experiment.experiment.GermanCreditPreprocessing")
    @patch("src.experiment.experiment.LawSchoolPreprocessing")
    def test_get_prepro_all_datasets(
            self,
            mock_law,
            mock_german,
            mock_adult,
            dataset,
            sens_attr
    ):
        """Test get_prepro returns correct preprocessing for each dataset."""

        exp = DummyExperiment.__new__(DummyExperiment)
        exp.config = {
            "data": {"name": dataset, "sens_attr": sens_attr},
            "model": {},
            "training": {},
            "experiment": {}
        }

        prepro_obj = exp.get_prepro()

        if dataset == "adult":
            mock_adult.assert_called_once_with(sens_attr)
            assert prepro_obj == mock_adult.return_value

        elif dataset == "german":
            mock_german.assert_called_once_with(sens_attr)
            assert prepro_obj == mock_german.return_value

        elif dataset == "law":
            mock_law.assert_called_once_with(sens_attr)
            assert prepro_obj == mock_law.return_value

    def test_get_prepro_unknown_dataset(self, base_config):
        """Test get_prepro raises error for unknown dataset."""
        exp = DummyExperiment.__new__(DummyExperiment)
        exp.config = base_config
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
