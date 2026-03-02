import numpy as np
import pytest
import torch

import src.utils.env as e
from src.likelihood.activation_extractor import ActivationExtractor
from src.likelihood.histograms import (
    UniBinHistogram,
    MultiBinsHistogram,
    HistogramConstructor,
)
from src.model.classificator import Classificator


class TestUniBinHistogram:
    """Tests for UniBinHistogram class."""

    def test_init_valid(self):
        """Test UniBinHistogram initialization."""
        bins = np.array([0.0, 0.0])
        freq = np.array([10.0])

        hist = UniBinHistogram(node_id=1, bins=bins, freq=freq)

        assert hist.node_id == 1

    def test_init_too_many_bins_raises(self):
        """Test UniBinHistogram raises with too many bins."""
        bins = np.array([0.0, 0.5, 1.0])
        freq = np.array([5, 10])

        with pytest.raises(ValueError, match="only one bin"):
            UniBinHistogram(node_id=1, bins=bins, freq=freq)

    def test_init_too_many_freq_raises(self):
        """Test UniBinHistogram raises with too many freq values."""
        bins = np.array([0.0, 0.0])
        freq = np.array([5, 10])

        with pytest.raises(ValueError, match="only one bin"):
            UniBinHistogram(node_id=1, bins=bins, freq=freq)

    def test_compute_hist_prob_at_lower_bound(self):
        """Test compute_hist_prob returns 1.0 at lower bound."""
        bins = np.array([0.0, 0.0])
        freq = np.array([10.0])

        hist = UniBinHistogram(node_id=1, bins=bins, freq=freq)

        prob = hist.compute_hist_prob(0.0)

        assert prob == 1.0

    def test_compute_hist_prob_away_from_lower_bound(self):
        """Test compute_hist_prob returns low prob away from lower bound."""
        bins = np.array([0.0, 0.0])
        freq = np.array([10.0])

        hist = UniBinHistogram(node_id=1, bins=bins, freq=freq)

        prob = hist.compute_hist_prob(0.5)

        assert prob < 1.0


class TestMultiBinsHistogram:
    """Tests for MultiBinsHistogram class."""

    def test_init_valid(self):
        """Test MultiBinsHistogram initialization."""
        bins = np.array([0.0, 0.5, 1.0])
        freq = np.array([5.0, 10.0])

        hist = MultiBinsHistogram(node_id=1, bins=bins, freq=freq)

        assert hist.node_id == 1
        assert np.array_equal(hist.bins, bins)
        assert np.array_equal(hist.freq, freq)

    def test_compute_hist_prob(self):
        """Test compute_hist_prob returns probability."""
        bins = np.array([0.0, 0.5, 1.0])
        freq = np.array([5.0, 20.0])

        hist = MultiBinsHistogram(node_id=1, bins=bins, freq=freq)

        prob = hist.compute_hist_prob(0.25)

        assert prob == 0.2

    def test_compute_hist_prob_at_lower_bound(self):
        """Test compute_hist_prob at lower bound."""
        bins = np.array([0.0, 0.5, 1.0])
        freq = np.array([5.0, 20.0])

        hist = MultiBinsHistogram(node_id=1, bins=bins, freq=freq)

        prob = hist.compute_hist_prob(0.0)

        assert prob == 0.2


class TestHistogramConstructor:
    """Tests for HistogramConstructor class."""

    def test_init(self):
        """Test HistogramConstructor initialization."""
        model = Classificator(input_dim=10, hidden_dims=[5], num_classes=2, seed=42)
        extractor = ActivationExtractor(model)

        constructor = HistogramConstructor(node_id=0, act_extract=extractor)

        assert constructor.node_id == 0
        assert constructor.extractor is extractor

    def test_check_null_std_negative_raises(self):
        """Test check_null_std raises on negative std."""
        model = Classificator(input_dim=10, hidden_dims=[5], num_classes=2, seed=42)
        extractor = ActivationExtractor(model)

        constructor = HistogramConstructor(node_id=0, act_extract=extractor)

        with pytest.raises(ValueError, match="negative"):
            constructor.check_null_std(-1.0)

    def test_check_null_std_positive_ok(self):
        """Test check_null_std passes on positive std."""
        model = Classificator(input_dim=10, hidden_dims=[5], num_classes=2, seed=42)
        extractor = ActivationExtractor(model)

        constructor = HistogramConstructor(node_id=0, act_extract=extractor)

        constructor.check_null_std(1.0)

    def test_construct_single_bins(self):
        """Test construct_single_bins creates UniBinHistogram."""
        model = Classificator(input_dim=10, hidden_dims=[5], num_classes=2, seed=42)
        extractor = ActivationExtractor(model)

        constructor = HistogramConstructor(node_id=0, act_extract=extractor)

        activations = torch.tensor([0.5, 0.5, 0.5])

        hist = constructor.construct_single_bins(activations)

        assert isinstance(hist, UniBinHistogram)
        assert hist.node_id == 0
        assert np.array_equal(hist.bins, np.array([0.5, 0.5]))
        assert np.array_equal(hist.freq, np.array([3]))
        assert hist.lower_bound == 0.5

    def test_construct_mult_bins(self):
        """Test construct_mult_bins creates MultiBinsHistogram."""
        model = Classificator(input_dim=10, hidden_dims=[5], num_classes=2, seed=42)
        extractor = ActivationExtractor(model)

        constructor = HistogramConstructor(node_id=0, act_extract=extractor)

        activations = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])

        hist = constructor.construct_mult_bins(activations, std_dev=0.25)

        assert isinstance(hist, MultiBinsHistogram)
        assert hist.node_id == 0
        assert np.allclose(hist.bins, np.array([0.1, 0.35, 0.6, 0.85, 1.1, 1.35]), atol=e.LOW_SMOOTHED_PROB)
        assert np.array_equal(hist.freq, np.array([2, 1, 1, 1, 0]))
        assert np.allclose(hist.lower_bound, 0.1, atol=e.LOW_SMOOTHED_PROB)
