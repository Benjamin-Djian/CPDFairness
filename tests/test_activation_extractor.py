import pandas as pd
import pytest
import torch

from src.likelihood.activation_extractor import (
    ClassificationFilter,
    FeatureFilter,
    FilterChain,
    ActivationGetter,
    ActivationExtractor,
)
from src.model.binary_classificator import BinaryClassificator
from src.preprocessing.dataset import IndexDataset


@pytest.fixture
def sample_dataset():
    df = pd.DataFrame({
        "feature1": [1.0, 0.0, 1.0, 0.0],
        "feature2": [0.5, 0.5, 0.5, 0.5],
        "target": [0, 1, 0, 1],
        "sens_attr": [0, 0, 1, 1]
    })
    return IndexDataset(df, sens_attr_column="sens_attr", target_column="target")


class TestClassificationFilter:
    """Tests for ClassificationFilter class."""

    def test_init(self):
        """Test ClassificationFilter initialization."""
        f = ClassificationFilter(keep_correct=True)
        assert f.keep_correct is True

    def test_get_mask_keep_correct(self):
        """Test get_mask keeps correct predictions."""
        f = ClassificationFilter(keep_correct=True)

        inputs = torch.randn(3, 5)
        targets = torch.tensor([0, 1, 0])
        predictions = torch.tensor([0, 1, 1])
        activations = torch.randn(3, 10)

        mask = f.get_mask(inputs, targets, predictions, activations)

        assert mask.tolist() == [True, True, False]

    def test_get_mask_keep_incorrect(self):
        """Test get_mask keeps incorrect predictions."""
        f = ClassificationFilter(keep_correct=False)

        inputs = torch.randn(3, 5)
        targets = torch.tensor([0, 1, 0])
        predictions = torch.tensor([0, 1, 1])
        activations = torch.randn(3, 10)

        mask = f.get_mask(inputs, targets, predictions, activations)

        assert mask.tolist() == [False, False, True]


class TestFeatureFilter:
    """Tests for FeatureFilter class."""

    def test_init(self, sample_dataset):
        """Test FeatureFilter initialization."""
        f = FeatureFilter(column_name="sens_attr", value=1, dataset=sample_dataset)
        assert f.column_name == "sens_attr"
        assert f.value == 1

    def test_get_mask(self, sample_dataset):
        """Test get_mask filters by feature value."""
        f = FeatureFilter(column_name="sens_attr", value=1, dataset=sample_dataset)

        inputs = torch.tensor([
            [1.0, 0.5, 0.0],
            [0.0, 0.5, 0.0],
            [1.0, 0.5, 1.0],
            [0.0, 0.5, 1.0]
        ])
        targets = torch.tensor([0, 1, 0, 1])
        predictions = torch.tensor([0, 1, 0, 1])
        activations = torch.randn(4, 10)

        mask = f.get_mask(inputs, targets, predictions, activations)

        assert mask.tolist() == [False, False, True, True]


class TestFilterChain:
    """Tests for FilterChain class."""

    def test_init(self):
        """Test FilterChain initialization."""
        filters = [ClassificationFilter(keep_correct=True)]
        chain = FilterChain(filters)
        assert len(chain.filters) == 1

    def test_apply_multiple_filters(self, sample_dataset):
        """Test apply with multiple filters (AND logic)."""
        f1 = ClassificationFilter(keep_correct=True)
        f2 = FeatureFilter(column_name="feature1", value=1.0, dataset=sample_dataset)
        filters = [f1, f2]

        chain = FilterChain(filters)

        indexes = torch.tensor([0, 1, 2, 3])
        inputs = torch.tensor([
            [1.0, 0.5, 0.0, 1.0],
            [0.0, 0.5, 0.0, 1.0],
            [1.0, 0.5, 1.0, 0.5],
            [0.0, 0.5, 1.0, 0.0]
        ])
        targets = torch.tensor([0, 1, 0, 1])
        predictions = torch.tensor([0, 1, 1, 0])
        activations = torch.randn(4, 10)

        idx, act = chain.apply(indexes, inputs, targets, predictions, activations)
        assert torch.equal(idx, torch.Tensor([0]))
        assert torch.equal(act[0], activations[0])


class TestActivationGetter:
    """Tests for ActivationGetter class."""

    def test_init(self):
        """Test ActivationGetter initialization."""
        indexes = torch.tensor([0, 1, 2])
        activations = torch.randn(3, 10)

        getter = ActivationGetter(indexes, activations)

        assert getter.activations.shape == (3, 10)
        assert getter.indexes.tolist() == [0, 1, 2]

    def test_get_by_index(self):
        """Test get_by_index method."""
        indexes = torch.tensor([0, 1, 2])
        activations = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])

        getter = ActivationGetter(indexes, activations)
        result = getter.get_by_index(1)

        assert result.tolist() == [3.0, 4.0]

    def test_get_by_node(self):
        """Test get_by_node method."""
        indexes = torch.tensor([0, 1])
        activations = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])

        getter = ActivationGetter(indexes, activations)
        result = getter.get_by_node(2)

        assert result.tolist() == [3.0, 6.0]

    def test_num_inputs(self):
        """Test num_inputs method."""
        indexes = torch.tensor([0, 1, 2])
        activations = torch.randn(3, 10)

        getter = ActivationGetter(indexes, activations)

        assert getter.num_inputs() == 3

    def test_num_nodes(self):
        """Test num_nodes method."""
        indexes = torch.tensor([0, 1])
        activations = torch.randn(2, 10)

        getter = ActivationGetter(indexes, activations)

        assert getter.num_nodes() == 10

    def test_iterate_indexes(self):
        """Test iterate_indexes method."""
        indexes = torch.tensor([0, 1])
        activations = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0]
        ])

        getter = ActivationGetter(indexes, activations)
        results = list(getter.iterate_indexes())

        assert len(results) == 2
        assert results[0][0] == 0
        assert torch.equal(results[0][1], torch.Tensor([1., 2.]))
        assert results[1][0] == 1
        assert torch.equal(results[1][1], torch.Tensor([3., 4.]))


class TestActivationExtractor:
    """Tests for ActivationExtractor class."""

    def test_init(self):
        """Test ActivationExtractor initialization."""
        model = BinaryClassificator(input_dim=10, hidden_dims=[5], seed=42)
        extractor = ActivationExtractor(model)

        assert extractor.classificator is model
