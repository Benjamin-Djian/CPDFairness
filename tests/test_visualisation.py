from src.likelihood.likelihood import LikelihoodScore
from src.visualisation.visualisation import (
    PlotCPDDistr,
    DualCPDFractionPop,
)


class TestPlotCPDDistr:
    """Tests for PlotCPDDistr class."""

    def test_extracts_scores(self):
        """Test PlotCPDDistr extracts scores from LikelihoodScore."""
        scores_1 = [LikelihoodScore(0, 0.5), LikelihoodScore(1, 0.6)]
        scores_2 = [LikelihoodScore(0, 0.3)]

        viz = PlotCPDDistr(
            serie_1=scores_1,
            serie_2=scores_2,
            nbr_bins=10,
            legend_1="G1",
            legend_2="G2"
        )

        assert viz.serie_1 == [0.5, 0.6]
        assert viz.serie_2 == [0.3]


class TestDualCPDFractionPop:
    """Tests for DualCPDFractionPop class."""

    def test_serie_attributes_set(self):
        """Test series attributes are set correctly."""
        scores = [LikelihoodScore(i, v) for i, v in enumerate([0.1, 0.3, 0.5, 0.7, 0.9])]

        viz = DualCPDFractionPop(
            serie_plain_left=scores,
            serie_plain_right=scores,
            serie_dashed_left=scores,
            serie_dashed_right=scores,
            title="Test",
            label_ax_left="Left",
            label_ax_right="Right",
            label_plain="Plain",
            label_dashed="Dashed"
        )

        assert len(viz.serie_plain_left) == 5
        assert len(viz.serie_plain_right) == 5

    def test_compute_dual_cpd_fraction_returns_tuple(self):
        """Test _compute_dual_cpd_fraction returns correct tuple structure."""
        scores = [LikelihoodScore(i, v) for i, v in enumerate([0.1, 0.3, 0.5, 0.7, 0.9])]

        viz = DualCPDFractionPop(
            serie_plain_left=scores,
            serie_plain_right=scores,
            serie_dashed_left=scores,
            serie_dashed_right=scores,
            title="Test",
            label_ax_left="Left",
            label_ax_right="Right",
            label_plain="Plain",
            label_dashed="Dashed"
        )

        result = viz._compute_dual_cpd_fraction()

        assert isinstance(result, tuple)
        assert len(result) == 4

        for series in result:
            assert isinstance(series, tuple)
            assert len(series) == 2

    def test_compute_with_single_value(self):
        """Test _compute_dual_cpd_fraction with single value."""
        scores = [LikelihoodScore(0, 0.5)]

        viz = DualCPDFractionPop(
            serie_plain_left=scores,
            serie_plain_right=scores,
            serie_dashed_left=scores,
            serie_dashed_right=scores,
            title="Test",
            label_ax_left="Left",
            label_ax_right="Right",
            label_plain="Plain",
            label_dashed="Dashed"
        )

        result = viz._compute_dual_cpd_fraction()

        assert len(result) == 4
