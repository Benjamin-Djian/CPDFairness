from src.model.binary_classificator import BinaryClassificator

from src.likelihood.activation_extractor import ActivationExtractor
from src.likelihood.likelihood import (
    LikelihoodScore,
    LikelihoodCalculator,
)


class TestLikelihoodScore:
    """Tests for LikelihoodScore class."""

    def test_init(self):
        """Test LikelihoodScore initialization."""
        score = LikelihoodScore(input_id=42, score=0.95)

        assert score.input_id == 42
        assert score.score == 0.95


class TestLikelihoodCalculator:
    """Tests for LikelihoodCalculator class."""

    def test_init(self):
        """Test LikelihoodCalculator initialization."""
        model = BinaryClassificator(input_dim=10, hidden_dims=[5], seed=42)
        extractor = ActivationExtractor(model)

        calc = LikelihoodCalculator(extractor)

        assert calc.extractor is extractor
