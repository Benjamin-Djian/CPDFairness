import numpy as np
import pandas as pd
import pytest

from src.data.dataset import IndexDataset
from src.preprocessing.prepro_operations import (
    DownSampling,
    MakeCategorical,
    Scale,
    TargetAtTheEnd,
    ToFloat,
    UpSampling,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "feature_num": [0.0, 50.0, 100.0],
        "feature_cat": ["a", "b", "c"],
        "target": [0, 1, 0],
        "sens_attr": [0, 1, 0]
    })


@pytest.fixture
def index_dataset(sample_df):
    return IndexDataset(
        df=sample_df,
        sens_attr_name="sens_attr",
        target_name="target"
    )


class TestMakeCategorical:
    """Tests for MakeCategorical preprocessing operation."""

    def test_init(self):
        """Test MakeCategorical initialization."""
        op = MakeCategorical(lb=3, ub=5)
        assert op.lb == 3
        assert op.ub == 5

    def test_fit_learns_categorical_columns(self, index_dataset):
        """Test fit learns categorical columns from dataset."""
        op = MakeCategorical(lb=3, ub=5)
        op.fit(index_dataset)

        assert "feature_cat" in op.categorical_columns_
        assert "feature_num" not in op.categorical_columns_
        assert "target" not in op.categorical_columns_
        assert "sens_attr" not in op.categorical_columns_

    def test_fit_excludes_target_and_sens_attr(self, index_dataset):
        """Test fit excludes target and sens_attr from categorical columns."""
        op = MakeCategorical(lb=1, ub=10)
        op.fit(index_dataset)

        assert "target" not in op.categorical_columns_
        assert "sens_attr" not in op.categorical_columns_

    def test_fit_with_no_categorical_columns(self):
        """Test fit with only numeric columns."""
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [0.0, 1.0, 2.0],
            "target": [0, 1, 0],
            "sens_attr": [0, 1, 0]
        })

    def test_transform_one_hot_encodes_categorical(self, index_dataset):
        """Test transform one-hot encodes categorical columns."""
        op = MakeCategorical(lb=3, ub=5)
        op.fit(index_dataset)

        result = op.transform(index_dataset)

        assert "feature_cat" not in result.df.columns
        assert "feature_cat_a" in result.df.columns
        assert "feature_cat_b" in result.df.columns
        assert "feature_cat_c" in result.df.columns
        assert "feature_num" in result.df.columns

    def test_transform_preserves_original(self, index_dataset):
        """Test transform does not modify original dataset."""
        op = MakeCategorical(lb=3, ub=5)
        op.fit(index_dataset)

        original_columns = list(index_dataset.df.columns)
        op.transform(index_dataset)

        assert list(index_dataset.df.columns) == original_columns

    def test_transform_with_no_categorical(self):
        """Test transform when no categorical columns exist."""
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [0.0, 1.0, 2.0],
            "target": [0, 1, 0],
            "sens_attr": [0, 1, 0]
        })
        dataset = IndexDataset(df=df, sens_attr_name="sens_attr", target_name="target")
        op = MakeCategorical(lb=3, ub=5)
        op.fit(dataset)

        result = op.transform(dataset)

        assert list(result.df.columns) == list(dataset.df.columns)

    def test_fit_transform_pipeline(self, index_dataset):
        """Test fit_transform works correctly."""
        op = MakeCategorical(lb=3, ub=5)

        result = op.fit_transform(index_dataset)

        assert "feature_cat_a" in result.df.columns


class TestScale:
    """Tests for Scale preprocessing operation."""

    def test_init(self):
        """Test Scale initialization."""
        op = Scale()
        assert op.scaler_ is None
        assert op.numeric_columns_ is None

    def test_fit_learns_numeric_columns(self, index_dataset):
        """Test fit learns numeric columns from dataset."""
        op = Scale()
        op.fit(index_dataset)

        assert "feature_num" in op.numeric_columns_
        assert "target" not in op.numeric_columns_
        assert "sens_attr" not in op.numeric_columns_

    def test_fit_excludes_target_and_sens_attr(self, index_dataset):
        """Test fit excludes target and sens_attr from numeric columns."""
        op = Scale()
        op.fit(index_dataset)

        assert "target" not in op.numeric_columns_
        assert "sens_attr" not in op.numeric_columns_

    def test_transform_scales_values(self, index_dataset):
        """Test transform scales numeric columns to 0-1 range."""
        op = Scale()
        op.fit(index_dataset)

        result = op.transform(index_dataset)

        scaled_values = result.df["feature_num"].values
        assert np.all(scaled_values >= 0)
        assert np.all(scaled_values <= 1)

    def test_transform_preserves_non_numeric(self, index_dataset):
        """Test transform does not scale non-numeric columns."""
        op = Scale()
        op.fit(index_dataset)

        result = op.transform(index_dataset)

        assert "feature_cat" in result.df.columns

    def test_transform_preserves_original(self, index_dataset):
        """Test transform does not modify original dataset."""
        op = Scale()
        op.fit(index_dataset)

        original_values = index_dataset.df["feature_num"].copy()
        op.transform(index_dataset)

        assert np.allclose(index_dataset.df["feature_num"].values, original_values.values)


class TestToFloat:
    """Tests for ToFloat preprocessing operation."""

    def test_init(self):
        """Test ToFloat initialization."""
        op = ToFloat()
        assert op.fitted_ is None

    def test_fit_sets_fitted_flag(self):
        """Test fit sets fitted flag."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "target": [0, 1, 0],
            "sens_attr": [0, 1, 0]
        })
        dataset = IndexDataset(df=df, sens_attr_name="sens_attr", target_name="target")
        op = ToFloat()
        op.fit(dataset)

        assert op.fitted_ is True

    def test_transform_converts_to_float(self):
        """Test transform converts all columns to float."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4.0, 5.0, 6.0],
            "target": [0, 1, 0],
            "sens_attr": [0, 1, 0]
        })
        dataset = IndexDataset(df=df, sens_attr_name="sens_attr", target_name="target")
        op = ToFloat()
        op.fit(dataset)

        result = op.transform(dataset)

        for col in result.df.columns:
            assert result.df[col].dtype == float

    def test_transform_preserves_values(self):
        """Test transform preserves values during conversion."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "target": [0, 1, 0],
            "sens_attr": [0, 1, 0]
        })
        dataset = IndexDataset(df=df, sens_attr_name="sens_attr", target_name="target")
        op = ToFloat()
        op.fit(dataset)

        result = op.transform(dataset)

        assert result.df["feature1"].tolist() == [1.0, 2.0, 3.0]


class TestTargetAtTheEnd:
    """Tests for TargetAtTheEnd preprocessing operation."""

    def test_init(self):
        """Test TargetAtTheEnd initialization."""
        op = TargetAtTheEnd()
        assert op.fitted_ is None

    def test_fit_sets_fitted_flag(self, index_dataset):
        """Test fit sets fitted flag."""
        op = TargetAtTheEnd()
        op.fit(index_dataset)

        assert op.fitted_ is True

    def test_transform_moves_target_to_end(self, index_dataset):
        """Test transform moves target column to the end."""
        op = TargetAtTheEnd()
        op.fit(index_dataset)

        result = op.transform(index_dataset)

        columns = list(result.df.columns)
        assert columns[-1] == "target"

    def test_transform_preserves_other_columns(self, index_dataset):
        """Test transform preserves all other columns."""
        op = TargetAtTheEnd()
        op.fit(index_dataset)

        result = op.transform(index_dataset)

        assert "feature_num" in result.df.columns
        assert "feature_cat" in result.df.columns
        assert "sens_attr" in result.df.columns
        assert len(result.df.columns) == len(index_dataset.df.columns)


class TestDownSampling:
    """Tests for DownSampling preprocessing operation."""

    def test_init(self):
        """Test DownSampling initialization."""
        op = DownSampling(seed=42)
        assert op.rus_ is not None

    def test_fit_is_noop(self, index_dataset):
        """Test fit does nothing (stateless)."""
        op = DownSampling(seed=42)
        result = op.fit(index_dataset)

        assert result is op

    def test_transform_reduces_majority_class(self):
        """Test transform reduces majority class samples."""
        df = pd.DataFrame({
            "feature1": [1.0] * 10 + [2.0] * 5,
            "target": [0] * 10 + [1] * 5,
            "sens_attr": [0, 1] * 5 + [0, 1, 0, 1, 0]
        })
        dataset = IndexDataset(df=df, sens_attr_name="sens_attr", target_name="target")
        op = DownSampling(seed=42)
        op.fit(dataset)

        result = op.transform(dataset)

        class_counts = result.df["target"].value_counts()
        assert class_counts[0] == class_counts[1]
        assert len(result.df) < len(dataset.df)
        assert len(result.df) == 10

    def test_transform_returns_index_dataset(self, index_dataset):
        """Test transform returns IndexDataset."""
        op = DownSampling(seed=42)
        op.fit(index_dataset)

        result = op.transform(index_dataset)

        assert isinstance(result, IndexDataset)
        assert result.sens_attr_name == index_dataset.sens_attr_name
        assert result.target_name == index_dataset.target_name


class TestUpSampling:
    """Tests for UpSampling preprocessing operation."""

    def test_init(self):
        """Test UpSampling initialization."""
        op = UpSampling(seed=42)
        assert op.ros_ is not None

    def test_fit_is_noop(self, index_dataset):
        """Test fit does nothing (stateless)."""
        op = UpSampling(seed=42)
        result = op.fit(index_dataset)

        assert result is op

    def test_transform_increases_minority_class(self):
        """Test transform increases minority class samples."""
        df = pd.DataFrame({
            "feature1": [1.0] * 10 + [2.0] * 5,
            "target": [0] * 10 + [1] * 5,
            "sens_attr": [0, 1] * 5 + [0, 1, 0, 1, 0]
        })
        dataset = IndexDataset(df=df, sens_attr_name="sens_attr", target_name="target")
        op = UpSampling(seed=42)
        op.fit(dataset)

        result = op.transform(dataset)

        class_counts = result.df["target"].value_counts()
        assert class_counts[0] == class_counts[1]
        assert len(result.df) > len(dataset.df)
        assert len(result.df) == 20

    def test_transform_returns_index_dataset(self, index_dataset):
        """Test transform returns IndexDataset."""
        op = UpSampling(seed=42)
        op.fit(index_dataset)

        result = op.transform(index_dataset)

        assert isinstance(result, IndexDataset)
        assert result.sens_attr_name == index_dataset.sens_attr_name
        assert result.target_name == index_dataset.target_name


class TestPipelineIntegration:
    """Tests for sklearn Pipeline integration."""

    def test_pipeline_make_categorical_then_scale(self, index_dataset):
        """Test pipeline with MakeCategorical then Scale."""
        pipeline = [
            MakeCategorical(lb=3, ub=5),
            Scale()
        ]

        from sklearn.pipeline import Pipeline as SklearnPipeline
        sklearn_pipe = SklearnPipeline([(f"step_{i}", op) for i, op in enumerate(pipeline)])

        sklearn_pipe.fit(index_dataset)
        result = sklearn_pipe.transform(index_dataset)

        assert isinstance(result, IndexDataset)
        assert "feature_cat_a" in result.df.columns
        assert "feature_num" in result.df.columns

    def test_pipeline_full_preprocessing(self, index_dataset):
        """Test full preprocessing pipeline."""
        pipeline = [
            MakeCategorical(lb=3, ub=5),
            ToFloat(),
            Scale(),
            TargetAtTheEnd()
        ]

        from sklearn.pipeline import Pipeline as SklearnPipeline
        sklearn_pipe = SklearnPipeline([(f"step_{i}", op) for i, op in enumerate(pipeline)])

        sklearn_pipe.fit(index_dataset)
        result = sklearn_pipe.transform(index_dataset)

        assert isinstance(result, IndexDataset)
        assert list(result.df.columns)[-1] == "target"
        assert result.df["feature_num"].dtype == float
