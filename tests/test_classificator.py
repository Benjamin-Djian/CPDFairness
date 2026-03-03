import pytest
import torch
from src.model.binary_classificator import BinaryClassificator


class TestClassificator:
    """Tests for Classificator class."""

    def test_init_multiple_hidden_layers(self):
        """Test initialization with multiple hidden layers."""
        model = BinaryClassificator(
            input_dim=10,
            hidden_dims=[8, 6, 4],
            seed=42
        )

        assert model.last_hidden_dim == 4

    def test_init_no_hidden_layers_raises(self):
        """Test initialization raises error with no hidden layers."""
        with pytest.raises(ValueError, match="at least one hidden layer"):
            BinaryClassificator(input_dim=10, hidden_dims=[])

    def test_forward_returns_tuple(self):
        """Test forward returns tuple of logits and hidden activations."""
        model = BinaryClassificator(
            input_dim=10,
            hidden_dims=[5],
            seed=42
        )

        x = torch.randn(4, 10)
        logits, hidden = model(x)

        assert logits.shape == (4, 2)
        assert hidden.shape == (4, 5)

    def test_forward_output_is_log_probs(self):
        """Test forward output are log probabilities."""
        model = BinaryClassificator(
            input_dim=10,
            hidden_dims=[5],
            seed=42
        )

        x = torch.randn(4, 10)
        logits, _ = model(x)

        assert torch.all(logits <= 0)

    def test_model_has_seq_attribute(self):
        """Test model has seq attribute for sequential layers."""
        model = BinaryClassificator(input_dim=10, hidden_dims=[5], seed=42)

        assert hasattr(model, 'seq')
        assert hasattr(model, 'output')
