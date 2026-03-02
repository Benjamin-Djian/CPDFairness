from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.preprocessing.preprocessing import (
    Preprocessing,
    AdultPreprocessing,
    GermanCreditPreprocessing,
    LawSchoolPreprocessing,
)
from src.preprocessing.prepro_operations import (
    PreprocessingOperation,
    MakeCategorical,
    Scale,
    ToFloat,
    TargetAtTheEnd,
)


class TestPreprocessing:
    """Tests for Preprocessing base class."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "feature1": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "feature2": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
            "sens_attr": [0, 0, 1, 1, 0, 0, 1, 1]
        })

    def test_init(self, sample_df):
        """Test Preprocessing initialization."""
        operations: list[PreprocessingOperation] = [Scale()]
        prepro = Preprocessing(sample_df, operations, "target", "sens_attr")
        
        assert prepro.df is not None
        assert prepro.target_column == "target"
        assert prepro.sens_attr_column == "sens_attr"
        assert len(prepro.operations) == 1

    def test_run_raises_if_no_operations(self, sample_df):
        """Test run raises ValueError if no operations defined."""
        prepro = Preprocessing(sample_df, [], "target", "sens_attr")
        with pytest.raises(ValueError, match="operations are not defined"):
            prepro.run()

    def test_generate_dataset_returns_index_dataset(self, sample_df):
        """Test generate_dataset returns IndexDataset."""
        operations: list[PreprocessingOperation] = [ToFloat()]
        prepro = Preprocessing(sample_df, operations, "target", "sens_attr")
        
        dataset = prepro.generate_dataset(sample_df)
        
        assert dataset is not None
        assert dataset.target_column == "target"
        assert dataset.sens_attr_column == "sens_attr"

    def test_generate_loaders_returns_three_loaders(self, sample_df):
        """Test generate_loaders returns train, val, test loaders."""
        operations: list[PreprocessingOperation] = [ToFloat()]
        prepro = Preprocessing(sample_df, operations, "target", "sens_attr")
        
        sample_df_clean = sample_df.copy()
        sample_df_clean["target"] = sample_df_clean["target"].astype(float)
        sample_df_clean["sens_attr"] = sample_df_clean["sens_attr"].astype(float)
        prepro.df = sample_df_clean
        
        train_loader, val_loader, test_loader = prepro.generate_loaders(
            prop_train=0.5,
            prop_valid=0.25,
            batch_size=1,
            seed=42
        )
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)


class TestAdultPreprocessing:
    """Tests for AdultPreprocessing class."""

    @patch("src.preprocessing.preprocessing.pd")
    @patch("src.preprocessing.preprocessing.ADULT_DATA_PATH", "data/adult.csv")
    def test_init(self, mock_pd):
        """Test AdultPreprocessing initialization."""
        mock_df = MagicMock()
        mock_pd.read_csv.return_value = mock_df
        
        prepro = AdultPreprocessing("sex")
        
        assert prepro.target_column is not None
        assert prepro.sens_attr_column == "sex"

    @patch("src.preprocessing.preprocessing.pd")
    @patch("src.preprocessing.preprocessing.ADULT_DATA_PATH", "data/adult.csv")
    @patch("src.preprocessing.preprocessing.ADULT_TARGET", "income")
    def test_operations(self, mock_pd):
        """Test AdultPreprocessing has correct operations."""
        mock_df = MagicMock()
        mock_pd.read_csv.return_value = mock_df
        
        prepro = AdultPreprocessing("sex")
        
        assert len(prepro.operations) == 4
        assert isinstance(prepro.operations[0], MakeCategorical)
        assert isinstance(prepro.operations[1], Scale)
        assert isinstance(prepro.operations[2], ToFloat)
        assert isinstance(prepro.operations[3], TargetAtTheEnd)


class TestGermanCreditPreprocessing:
    """Tests for GermanCreditPreprocessing class."""

    @patch("src.preprocessing.preprocessing.pd")
    @patch("src.preprocessing.preprocessing.GERMAN_DATA_PATH", "data/german.csv")
    def test_init(self, mock_pd):
        """Test GermanCreditPreprocessing initialization."""
        mock_df = MagicMock()
        mock_pd.read_csv.return_value = mock_df
        
        prepro = GermanCreditPreprocessing("sex")
        
        assert prepro.sens_attr_column == "sex"


class TestLawSchoolPreprocessing:
    """Tests for LawSchoolPreprocessing class."""

    @patch("src.preprocessing.preprocessing.pd")
    @patch("src.preprocessing.preprocessing.LAW_DATA_PATH", "data/law.csv")
    def test_init(self, mock_pd):
        """Test LawSchoolPreprocessing initialization."""
        mock_df = MagicMock()
        mock_pd.read_csv.return_value = mock_df
        
        prepro = LawSchoolPreprocessing("sex")
        
        assert prepro.sens_attr_column == "sex"
