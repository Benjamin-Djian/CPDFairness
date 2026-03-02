import pandas as pd
import pytest
import torch

from src.preprocessing.dataset import IndexDataset
from src.preprocessing.prepro_operations import (
    MakeCategorical,
    Scale,
    ToFloat,
    TargetAtTheEnd,
)


class TestMakeCategorical:
    """Tests for MakeCategorical operation."""

    def test_init(self):
        """Test MakeCategorical initialization."""
        op = MakeCategorical(lb=3, ub=5)
        assert op.lb == 3
        assert op.ub == 5

    def test_run_converts_categorical_columns(self):
        """Test run converts string columns to dummies."""
        op = MakeCategorical(lb=3, ub=10)
        df = pd.DataFrame({
            "col1": ["a", "b", "a", "c"],
            "col2": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
            "sens_attr": [0, 1, 0, 1]
        })

        result = op.run(df, "target", "sens_attr")
        expected = pd.DataFrame({
            "target": [0, 1, 0, 1],
            "sens_attr": [0, 1, 0, 1],
            "col1=a": [1, 0, 1, 0],
            "col1=b": [0, 1, 0, 0],
            "col1=c": [0, 0, 0, 1],
            "col2=1": [1, 0, 0, 0],
            "col2=2": [0, 1, 0, 0],
            "col2=3": [0, 0, 1, 0],
            "col2=4": [0, 0, 0, 1],
        })
        assert result.equals(expected)


class TestScale:
    """Tests for Scale operation."""

    def test_run_scales_numeric_columns(self):
        """Test run scales numeric columns to 0-1 range."""
        op = Scale()
        df = pd.DataFrame({
            "col1": [0, 50, 100],
            "col2": [10, 20, 30],
            "target": [0, 1, 0],
            "sens_attr": [0, 3, 0]
        })
        target_name = "target"
        sens_attr_name = "sens_attr"
        result = op.run(df, target_name, sens_attr_name)

        scaled_columns = [c for c in result.columns if c not in [target_name, sens_attr_name]]
        assert all([(result[c].min() >= 0) and (result[c].min() >= 0) for c in scaled_columns])

    def test_run_excludes_target_and_sens_attr(self):
        """Test target and sens_attr are not scaled."""
        op = Scale()
        df = pd.DataFrame({
            "col1": [0, 50, 100],
            "target": [0, 3, 100],
            "sens_attr": [2, 1, 0]
        })

        result = op.run(df, "target", "sens_attr")

        assert result["target"].tolist() == [0, 3, 100]
        assert result["sens_attr"].tolist() == [2, 1, 0]


class TestToFloat:
    """Tests for ToFloat operation."""

    def test_run_converts_to_float(self):
        """Test run converts all columns to float."""
        op = ToFloat()
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [1.5, 2.5, 3.5]
        })

        result = op.run(df, "target", "sens_attr")

        assert result["col1"].dtype == float
        assert result["col2"].dtype == float


class TestTargetAtTheEnd:
    """Tests for TargetAtTheEnd operation."""

    def test_run_moves_target_to_end(self):
        """Test run moves target column to the end."""
        op = TargetAtTheEnd()
        df = pd.DataFrame({
            "target": [0, 1, 0],
            "col1": [1, 2, 3],
            "col2": [4, 5, 6]
        })

        result = op.run(df, "target", "sens_attr")

        assert list(result.columns)[-1] == "target"


class TestIndexDataset:
    """Tests for IndexDataset class."""

    def test_init_with_valid_data(self):
        """Test IndexDataset initialization with valid data."""
        df = pd.DataFrame({
            "feature1": [1.0, 0.0, 1.0, 0.0],
            "feature2": [0.0, 1.0, 0.0, 1.0],
            "target": [0, 1, 0, 1],
            "sens_attr": [0, 0, 1, 1]
        })

        dataset = IndexDataset(df, sens_attr_column="sens_attr", target_column="target")

        assert dataset is not None
        assert dataset.df.equals(df)
        assert dataset.sens_attr_column == "sens_attr"
        assert dataset.target_column == "target"

    def test_init_raises_on_missing_column(self):
        """Test initialization raises error if column is missing."""
        df = pd.DataFrame({
            "feature1": [1.0, 0.0],
            "target": [0, 1],
        })

        with pytest.raises(ValueError, match="must be in dataset"):
            IndexDataset(df, sens_attr_column="missing", target_column="target")

    def test_init_raises_on_non_binary_sens_attr(self):
        """Test initialization raises error if sens_attr is not binary."""
        df = pd.DataFrame({
            "feature1": [1.0, 0.0],
            "target": [0, 1],
            "sens_attr": [0, 2]
        })

        with pytest.raises(ValueError, match="must be binary"):
            IndexDataset(df, sens_attr_column="sens_attr", target_column="target")

    def test_len(self):
        """Test __len__ returns correct length."""
        df = pd.DataFrame({
            "feature1": [1.0, 0.0, 1.0, 0.0, 1.0],
            "target": [0, 1, 0, 1, 0],
            "sens_attr": [0, 1, 0, 1, 0]
        })

        dataset = IndexDataset(df, sens_attr_column="sens_attr", target_column="target")
        assert len(dataset) == 5

    def test_getitem(self):
        """Test __getitem__ returns correct format."""
        df = pd.DataFrame({
            "feature1": [1.0, 0.0],
            "target": [0, 1],
            "sens_attr": [1, 1]
        })

        dataset = IndexDataset(df, sens_attr_column="sens_attr", target_column="target")
        idx, x, y = dataset[0]

        assert idx == 0
        assert torch.equal(x, torch.Tensor([1.0, 1]))
        assert y.item() == 0.0

    def test_feature_columns(self):
        """Test feature_columns property."""
        df = pd.DataFrame({
            "feature1": [1.0, 0.0],
            "feature2": [0.5, 0.5],
            "target": [0, 1],
            "sens_attr": [0, 1]
        })

        dataset = IndexDataset(df, sens_attr_column="sens_attr", target_column="target")
        feature_cols = dataset.feature_columns

        assert feature_cols == ["feature1", "feature2", "sens_attr"]
        df.drop(columns=["feature1"], inplace=True)
        feature_cols = dataset.feature_columns
        assert feature_cols == ["feature2", "sens_attr"]

    def test_get_index_col(self):
        """Test get_index_col returns correct index."""
        df = pd.DataFrame({
            "feature1": [1.0, 0.0],
            "feature2": [0.5, 0.5],
            "target": [0, 1],
            "sens_attr": [0, 1]
        })

        dataset = IndexDataset(df, sens_attr_column="sens_attr", target_column="target")
        idx = dataset.get_index_col("feature2")
        assert idx == 1

    def test_get_index_col_raises_on_missing(self):
        """Test get_index_col raises error for missing column."""
        df = pd.DataFrame({
            "feature1": [1.0, 0.0],
            "target": [0, 1],
            "sens_attr": [0, 1]
        })

        dataset = IndexDataset(df, sens_attr_column="sens_attr", target_column="target")

        with pytest.raises(ValueError, match="not found"):
            dataset.get_index_col("nonexistent")
