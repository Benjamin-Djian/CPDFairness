from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from torch.utils.data import DataLoader

from src.data.data_reader import DataReader
from src.data.dataset import IndexDataset
from src.preprocessing.data_preparator import (
    AdultDataPreparator,
    DataPreparator,
    GermanDataPreparator,
    LawDataPreparator,
)
from src.preprocessing.prepro_operations import (
    DownSampling,
    MakeCategorical,
    Scale,
    ToFloat,
    TargetAtTheEnd,
)


class TestDataPreparator:
    """Tests for DataPreparator class."""

    @pytest.fixture
    def mock_data_reader(self):
        reader = MagicMock(spec=DataReader)
        reader.target_name = "target"
        reader.sens_attr_name = "sens_attr"
        reader.read_data.return_value = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [0.5, 0.6, 0.7, 0.8, 0.9],
            "target": [0, 1, 0, 1, 0],
            "sens_attr": [0, 1, 0, 1, 0]
        })
        return reader

    @pytest.fixture
    def operations(self):
        return [
            MakeCategorical(lb=3, ub=5),
            Scale(),
            ToFloat(),
            TargetAtTheEnd()
        ]

    def test_init(self, mock_data_reader, operations):
        """Test DataPreparator initialization."""
        preparator = DataPreparator(mock_data_reader, operations)

        assert preparator.data_reader == mock_data_reader
        assert preparator.df is not None
        assert preparator.pipeline is not None

    def test_init_reads_data(self, mock_data_reader, operations):
        """Test DataPreparator reads data on init."""
        DataPreparator(mock_data_reader, operations)

        mock_data_reader.read_data.assert_called_once()

    def test_split_sets_returns_three_datasets(self, mock_data_reader, operations):
        """Test split_sets returns train, val, test datasets."""
        preparator = DataPreparator(mock_data_reader, operations)

        train, val, test = preparator.split_sets(prop_train=0.6, prop_valid=0.5, seed=42)

        assert isinstance(train, IndexDataset)
        assert isinstance(val, IndexDataset)
        assert isinstance(test, IndexDataset)
        assert len(train) + len(val) + len(test) == 5

    def test_split_sets_proportions(self, mock_data_reader, operations):
        """Test split_sets respects proportions."""
        preparator = DataPreparator(mock_data_reader, operations)

        train, val, test = preparator.split_sets(prop_train=0.6, prop_valid=0.5, seed=42)

        total = len(train) + len(val) + len(test)
        assert len(train) / total == pytest.approx(0.6, rel=0.1)
        assert (len(val) + len(test)) / total == pytest.approx(0.4, rel=0.1)

    def test_split_sets_reproducible_with_seed(self, mock_data_reader, operations):
        """Test split_sets is reproducible with same seed."""
        preparator = DataPreparator(mock_data_reader, operations)

        train1, val1, test1 = preparator.split_sets(prop_train=0.6, prop_valid=0.5, seed=42)
        train2, val2, test2 = preparator.split_sets(prop_train=0.6, prop_valid=0.5, seed=42)

        assert len(train1) == len(train2)
        assert len(val1) == len(val2)
        assert len(test1) == len(test2)

    def test_to_index_dataframe(self, mock_data_reader, operations):
        """Test to_index_dataframe creates IndexDataset."""
        preparator = DataPreparator(mock_data_reader, operations)
        df = pd.DataFrame({
            "feature1": [1.0, 2.0],
            "target": [0, 1],
            "sens_attr": [0, 1]
        })

        dataset = preparator.to_index_dataframe(df)

        assert isinstance(dataset, IndexDataset)
        assert dataset.target_name == "target"
        assert dataset.sens_attr_name == "sens_attr"

    def test_to_dataloaders(self, mock_data_reader, operations):
        """Test to_dataloaders creates DataLoader."""
        preparator = DataPreparator(mock_data_reader, operations)
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0],
            "target": [0, 1, 0, 1],
            "sens_attr": [0, 1, 0, 1]
        })
        dataset = IndexDataset(df=df, sens_attr_name="sens_attr", target_name="target")

        loader = preparator.to_dataloaders(dataset, batch_size=2, seed=42)

        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 2

    def test_run_returns_three_loaders(self, mock_data_reader, operations):
        """Test run returns train, val, test loaders."""
        preparator = DataPreparator(mock_data_reader, operations)

        train_loader, val_loader, test_loader = preparator.run(
            prop_train=0.6,
            prop_valid=0.5,
            batch_size=2,
            seed=42
        )

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

    def test_run_fits_pipeline(self, mock_data_reader, operations):
        """Test run fits pipeline on training data."""
        preparator = DataPreparator(mock_data_reader, operations)

        preparator.run(prop_train=0.6, prop_valid=0.5, batch_size=2, seed=42)

        assert preparator.pipeline.named_steps is not None

    def test_run_uses_downsampling_in_pipeline(self):
        """Test run with downsampling operation."""
        reader = MagicMock(spec=DataReader)
        reader.target_name = "target"
        reader.sens_attr_name = "sens_attr"
        reader.read_data.return_value = pd.DataFrame({
            "feature1": [1.0] * 20 + [2.0] * 10,
            "target": [0] * 20 + [1] * 10,
            "sens_attr": [0, 1] * 10 + [0, 1] * 5
        })

        operations = [DownSampling(seed=42)]
        preparator = DataPreparator(reader, operations)

        train_loader, val_loader, test_loader = preparator.run(
            prop_train=0.6,
            prop_valid=0.5,
            batch_size=2,
            seed=42
        )

        assert isinstance(train_loader, DataLoader)


class TestAdultDataPreparator:
    """Tests for AdultDataPreparator class."""

    def test_init(self):
        """Test AdultDataPreparator initialization."""
        with patch("src.preprocessing.data_preparator.AdultDataReader") as mock_reader:
            mock_instance = MagicMock()
            mock_instance.target_name = "income"
            mock_instance.sens_attr_name = "sex"
            mock_reader.return_value = mock_instance

            preparator = AdultDataPreparator(sens_attr_name="sex")

            assert preparator.data_reader is not None

    def test_has_correct_operations(self):
        """Test AdultDataPreparator has correct preprocessing operations."""
        with patch("src.preprocessing.data_preparator.AdultDataReader") as mock_reader:
            mock_instance = MagicMock()
            mock_instance.read_data.return_value = pd.DataFrame({
                "feature1": [1.0, 2.0],
                "target": [0, 1],
                "sens_attr": [0, 1]
            })
            mock_reader.return_value = mock_instance

            preparator = AdultDataPreparator(sens_attr_name="sex")

            assert len(preparator.pipeline.steps) == 4


class TestGermanDataPreparator:
    """Tests for GermanDataPreparator class."""

    def test_init(self):
        """Test GermanDataPreparator initialization."""
        with patch("src.preprocessing.data_preparator.GermanDataReader") as mock_reader:
            mock_instance = MagicMock()
            mock_instance.target_name = "good_3_bad_2_customer"
            mock_instance.sens_attr_name = "sex"
            mock_reader.return_value = mock_instance

            preparator = GermanDataPreparator(sens_attr_name="sex")

            assert preparator.data_reader is not None

    def test_has_correct_operations(self):
        """Test GermanDataPreparator has correct preprocessing operations."""
        with patch("src.preprocessing.data_preparator.GermanDataReader") as mock_reader:
            mock_instance = MagicMock()
            mock_instance.read_data.return_value = pd.DataFrame({
                "feature1": [1.0, 2.0],
                "target": [0, 1],
                "sens_attr": [0, 1]
            })
            mock_reader.return_value = mock_instance

            preparator = GermanDataPreparator(sens_attr_name="sex")

            assert len(preparator.pipeline.steps) == 4


class TestLawDataPreparator:
    """Tests for LawDataPreparator class."""

    def test_init(self):
        """Test LawDataPreparator initialization."""
        with patch("src.preprocessing.data_preparator.LawDataReader") as mock_reader:
            mock_instance = MagicMock()
            mock_instance.target_name = "pass_bar"
            mock_instance.sens_attr_name = "sex"
            mock_reader.return_value = mock_instance

            preparator = LawDataPreparator(sens_attr_name="sex")

            assert preparator.data_reader is not None

    def test_has_correct_operations(self):
        """Test LawDataPreparator has correct preprocessing operations."""
        with patch("src.preprocessing.data_preparator.LawDataReader") as mock_reader:
            mock_instance = MagicMock()
            mock_instance.read_data.return_value = pd.DataFrame({
                "feature1": [1.0, 2.0],
                "target": [0, 1],
                "sens_attr": [0, 1]
            })
            mock_reader.return_value = mock_instance

            preparator = LawDataPreparator(sens_attr_name="sex")

            assert len(preparator.pipeline.steps) == 4
