import pandas as pd
import pytest
import torch

from src.data.dataset import IndexDataset


class TestIndexDataset:
    """Tests for IndexDataset class."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "feature1": [1.0, 0.0, 1.0, 0.0],
            "feature2": [0.5, 0.5, 0.5, 0.5],
            "target": [0, 1, 0, 1],
            "sens_attr": [0, 0, 1, 1]
        })

    def test_init_with_valid_data(self, sample_df):
        """Test IndexDataset initialization with valid data."""
        dataset = IndexDataset(
            df=sample_df,
            sens_attr_name="sens_attr",
            target_name="target"
        )

        assert dataset is not None
        assert dataset.sens_attr_name == "sens_attr"
        assert dataset.target_name == "target"
        assert (dataset.df == sample_df).all().all()

    def test_init_extracts_features(self, sample_df):
        """Test that features are correctly extracted from DataFrame."""
        dataset = IndexDataset(
            df=sample_df,
            sens_attr_name="sens_attr",
            target_name="target"
        )

        expected_features = sample_df.drop(columns=["target"]).values
        assert (dataset.features == expected_features).all()

    def test_init_extracts_target(self, sample_df):
        """Test that target is correctly extracted from DataFrame."""
        dataset = IndexDataset(
            df=sample_df,
            sens_attr_name="sens_attr",
            target_name="target"
        )

        expected_target = sample_df["target"].values
        assert (dataset.target == expected_target).all()

    def test_init_raises_on_missing_sens_attr(self, sample_df):
        """Test initialization raises error if sens_attr column is missing."""
        sample_df = sample_df.drop(columns=["sens_attr"])

        with pytest.raises(ValueError, match="must be in dataset"):
            IndexDataset(df=sample_df, sens_attr_name="sens_attr", target_name="target")

    def test_init_raises_on_missing_target(self, sample_df):
        """Test initialization raises error if target column is missing."""
        sample_df = sample_df.drop(columns=["target"])

        with pytest.raises(ValueError, match="must be in dataset"):
            IndexDataset(df=sample_df, sens_attr_name="sens_attr", target_name="target")

    def test_init_raises_on_non_binary_sens_attr(self, sample_df):
        """Test initialization raises error if sens_attr is not binary."""
        sample_df["sens_attr"] = [0, 2, 1, 1]

        with pytest.raises(ValueError, match="must be binary"):
            IndexDataset(df=sample_df, sens_attr_name="sens_attr", target_name="target")

    def test_init_raises_on_non_binary_target(self, sample_df):
        """Test initialization raises error if target is not binary."""
        sample_df["target"] = [0, 1, 2, 1]

        with pytest.raises(ValueError, match="must be binary"):
            IndexDataset(df=sample_df, sens_attr_name="sens_attr", target_name="target")

    def test_len_returns_correct_length(self, sample_df):
        """Test __len__ returns correct length."""
        dataset = IndexDataset(
            df=sample_df,
            sens_attr_name="sens_attr",
            target_name="target"
        )

        assert len(dataset) == 4

    def test_getitem_returns_tuple(self, sample_df):
        """Test __getitem__ returns correct format."""
        dataset = IndexDataset(
            df=sample_df,
            sens_attr_name="sens_attr",
            target_name="target"
        )

        idx, x, y = dataset[0]

        assert idx == 0
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape[0] == 3

    def test_getitem_returns_correct_values(self, sample_df):
        """Test __getitem__ returns correct values."""
        dataset = IndexDataset(
            df=sample_df,
            sens_attr_name="sens_attr",
            target_name="target"
        )

        idx, x, y = dataset[0]

        assert idx == 0
        assert torch.allclose(x, torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32))
        assert y.item() == 0.0

    def test_getitem_with_custom_index(self):
        """Test __getitem__ works with custom DataFrame index."""
        df = pd.DataFrame(
            {"feature1": [1.0, 0.0], "target": [0, 1], "sens_attr": [0, 1]},
            index=[10, 20]
        )

        dataset = IndexDataset(
            df=df,
            sens_attr_name="sens_attr",
            target_name="target"
        )

        assert len(dataset) == 2
        idx, x, y = dataset[1]
        assert idx == 1

    def test_feature_columns_property(self, sample_df):
        """Test feature_columns property returns correct columns."""
        dataset = IndexDataset(
            df=sample_df,
            sens_attr_name="sens_attr",
            target_name="target"
        )

        feature_cols = dataset.feature_columns

        assert "feature1" in feature_cols
        assert "feature2" in feature_cols
        assert "sens_attr" in feature_cols
        assert "target" not in feature_cols

    def test_index_property(self, sample_df):
        """Test index property returns correct DataFrame index."""
        dataset = IndexDataset(
            df=sample_df,
            sens_attr_name="sens_attr",
            target_name="target"
        )

        assert (dataset.index == sample_df.index).all()

    def test_index_property_with_custom_index(self):
        """Test index property works with custom DataFrame index."""
        df = pd.DataFrame(
            {"feature1": [1.0, 0.0], "target": [0, 1], "sens_attr": [0, 1]},
            index=[100, 200]
        )

        dataset = IndexDataset(
            df=df,
            sens_attr_name="sens_attr",
            target_name="target"
        )

        assert list(dataset.index) == [100, 200]

    def test_copy_creates_new_instance(self, sample_df):
        """Test copy creates a new IndexDataset with copied DataFrame."""
        dataset = IndexDataset(
            df=sample_df,
            sens_attr_name="sens_attr",
            target_name="target"
        )

        copied = dataset.copy()

        assert isinstance(copied, IndexDataset)
        assert copied is not dataset
        assert (copied.df == dataset.df).all().all()
        assert copied.sens_attr_name == dataset.sens_attr_name
        assert copied.target_name == dataset.target_name

    def test_copy_is_independent(self, sample_df):
        """Test that copy is independent of original."""
        dataset = IndexDataset(
            df=sample_df,
            sens_attr_name="sens_attr",
            target_name="target"
        )

        copied = dataset.copy()
        dataset.df.iloc[0, 0] = 999.0

        assert copied.df.iloc[0, 0] == 1.0

    def test_create_from_df(self, sample_df):
        """Test create_from_df creates new IndexDataset with new DataFrame."""
        dataset = IndexDataset(
            df=sample_df,
            sens_attr_name="sens_attr",
            target_name="target"
        )

        new_df = pd.DataFrame({
            "feature1": [2.0, 3.0],
            "sens_attr": [0, 1],
            "target": [1, 0]
        })

        new_dataset = dataset.create_from_df(new_df)

        assert isinstance(new_dataset, IndexDataset)
        assert (new_dataset.df == new_df).all().all()
        assert new_dataset.sens_attr_name == dataset.sens_attr_name
        assert new_dataset.target_name == dataset.target_name

    def test_to_csv(self, sample_df, tmp_path):
        """Test to_csv saves DataFrame correctly."""
        dataset = IndexDataset(
            df=sample_df,
            sens_attr_name="sens_attr",
            target_name="target"
        )

        save_path = tmp_path / "test_output.csv"
        dataset.to_csv(save_path)

        assert save_path.exists()

        loaded_df = pd.read_csv(save_path, index_col="inputId")
        assert (loaded_df == sample_df).all().all()

    def test_get_index_col(self, sample_df):
        """Test get_index_col returns correct column index."""
        dataset = IndexDataset(
            df=sample_df,
            sens_attr_name="sens_attr",
            target_name="target"
        )

        assert dataset.get_index_col("feature1") == 0
        assert dataset.get_index_col("feature2") == 1
        assert dataset.get_index_col("sens_attr") == 2

    def test_get_index_col_raises_on_missing_column(self, sample_df):
        """Test get_index_col raises error for missing column."""
        dataset = IndexDataset(
            df=sample_df,
            sens_attr_name="sens_attr",
            target_name="target"
        )

        with pytest.raises(ValueError, match="not found"):
            dataset.get_index_col("nonexistent_column")
