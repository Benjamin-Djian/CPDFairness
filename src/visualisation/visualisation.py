from abc import abstractmethod, ABC
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import src.utils.env as e
from src.likelihood.likelihood import LikelihoodScore


class Visualisation(ABC):
    def __init__(self, title: str | None, x_label: str | None, y_label: str | None):
        self.figure: Figure
        self.ax: Axes
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self._create_figure(e.DEFAULT_FIGURE_SIZE)
        self._configures_axes()

    def _create_figure(self, size: tuple[float, float]) -> None:
        self.figure, self.ax = plt.subplots(figsize=size)

    def _configures_axes(self) -> None:
        if self.title:
            self.ax.set_title(self.title)
        if self.x_label:
            self.ax.set_xlabel(self.x_label)
        if self.y_label:
            self.ax.set_ylabel(self.y_label)

    @abstractmethod
    def plot(self):
        pass

    def show(self) -> None:
        self.figure.tight_layout()
        plt.show()

    def save(self, path: Path, dpi: int = e.DEFAULT_DPI) -> None:
        self.figure.tight_layout()
        self.figure.savefig(path, dpi=dpi)


class DualAxisVisualisation(Visualisation, ABC):
    """Abstract class for visualizations with dual Y-axis."""

    def __init__(
            self,
            title: str | None,
            x_label: str | None,
            y_label_left: str | None,
            y_label_right: str | None):
        self.y_label_left = y_label_left
        self.y_label_right = y_label_right
        self.ax_right = None
        super().__init__(title=title, x_label=x_label, y_label=y_label_left)

    def _create_figure(self, size: tuple[float, float]) -> None:
        self.figure, self.ax = plt.subplots(figsize=size)
        self.ax_right = self.ax.twinx()

    def _configures_axes(self) -> None:
        super()._configures_axes()
        if self.y_label_right:
            self.ax_right.set_ylabel(self.y_label_right)


class PlotCPDDistr(Visualisation):
    def __init__(self,
                 serie_1: list[LikelihoodScore],
                 serie_2: list[LikelihoodScore],
                 legend_1: str,
                 legend_2: str,
                 nbr_bins: int = 10):
        super().__init__(title="",
                         x_label="Activation Levels",
                         y_label="Percentage of input")
        self.legend_1 = legend_1
        self.legend_2 = legend_2
        self.serie_1 = [lh.score for lh in serie_1]
        self.serie_2 = [lh.score for lh in serie_2]
        self.nbr_bins = nbr_bins

    def plot(self):
        weights_serie_1 = np.ones(len(self.serie_1)) / len(self.serie_1)
        weights_serie_2 = np.ones(len(self.serie_2)) / len(self.serie_2)
        self.ax.hist(self.serie_1, bins=self.nbr_bins, weights=weights_serie_1, label=self.legend_1, color='blue',
                     alpha=0.7)
        self.ax.hist(self.serie_2, bins=self.nbr_bins, weights=weights_serie_2, label=self.legend_2, color='red',
                     alpha=0.7)

        self.ax.grid()
        self.ax.legend()


class PlotCPDFractionPop(Visualisation):
    def __init__(self,
                 serie_1: list[LikelihoodScore],
                 serie_2: list[LikelihoodScore],
                 resolution: int,
                 legend_1: str,
                 legend_2: str):
        super().__init__(title="",
                         x_label="Normalized CPD",
                         y_label="Fraction of respective population")
        self.resolution = resolution
        self.legend_1 = legend_1
        self.legend_2 = legend_2
        self.serie_1 = [lh.score for lh in serie_1]
        self.serie_2 = [lh.score for lh in serie_2]

    def plot(self):
        self.serie_1.sort(reverse=True)
        self.serie_2.sort(reverse=True)

        frac_pop_1 = []
        frac_pop_2 = []

        for cpd_value in np.linspace(min(self.serie_1), max(self.serie_1), self.resolution):
            smaller_than = [elem for elem in self.serie_1 if elem > cpd_value]
            frac_pop_1.append(100 * len(smaller_than) / len(self.serie_1))

        for cpd_value in np.linspace(min(self.serie_2), max(self.serie_2), self.resolution):
            smaller_than = [elem for elem in self.serie_2 if elem > cpd_value]
            frac_pop_2.append(100 * len(smaller_than) / len(self.serie_2))

        self.ax.plot(frac_pop_1, label=self.legend_1, color='blue')
        self.ax.plot(frac_pop_2, label=self.legend_2, color='red')

        self.ax.grid()
        self.ax.legend()


class DualCPDFractionPop(DualAxisVisualisation):
    def __init__(
            self,
            serie_plain_left: list[LikelihoodScore],
            serie_plain_right: list[LikelihoodScore],
            serie_dashed_left: list[LikelihoodScore],
            serie_dashed_right: list[LikelihoodScore],
            title: str = "",
            label_ax_left: str = "",
            label_ax_right: str = "",
            label_plain: str = "",
            label_dashed: str = "",
    ):
        super().__init__(
            title=title,
            x_label="Fraction of respective population",
            y_label_left=label_ax_left,
            y_label_right=label_ax_right,
        )
        self.serie_plain_left = [lh.score for lh in serie_plain_left]
        self.serie_plain_left.sort(reverse=False)
        self.serie_plain_right = [lh.score for lh in serie_plain_right]
        self.serie_plain_right.sort(reverse=False)
        self.serie_dashed_left = [lh.score for lh in serie_dashed_left]
        self.serie_dashed_left.sort(reverse=False)
        self.serie_dashed_right = [lh.score for lh in serie_dashed_right]
        self.serie_dashed_right.sort(reverse=False)

        self.label_plain = label_plain
        self.label_dashed = label_dashed

    def _compute_dual_cpd_fraction(self):
        def compute_series(data: list[float], global_max: float):
            if not data:
                return np.array([]), np.array([])

            data = np.sort(np.array(data))
            n = len(data)

            min_val = data[0]
            denom = global_max - min_val

            if denom == 0:
                return np.linspace(0, 100, n), np.zeros(n)

            x = np.linspace(0, 100, n)

            y = (data - min_val) / denom

            return x, y

        max_left = max(self.serie_plain_left + self.serie_dashed_left)
        max_right = max(self.serie_plain_right + self.serie_dashed_right)

        x_plain_left, y_plain_left = compute_series(self.serie_plain_left, max_left)
        x_dashed_left, y_dashed_left = compute_series(self.serie_dashed_left, max_left)

        x_plain_right, y_plain_right = compute_series(self.serie_plain_right, max_right)
        x_dashed_right, y_dashed_right = compute_series(self.serie_dashed_right, max_right)

        return (
            (x_plain_left, y_plain_left),
            (x_dashed_left, y_dashed_left),
            (x_plain_right, y_plain_right),
            (x_dashed_right, y_dashed_right),
        )

    def plot(self):
        (x_pl, y_pl), (x_dl, y_dl), (x_pr, y_pr), (x_dr, y_dr) = self._compute_dual_cpd_fraction()

        self.ax.plot(x_pl, y_pl, color='red')
        self.ax.plot(x_dl, y_dl, color='red', linestyle='dashed')

        self.ax_right.plot(x_pr, y_pr, color='green')
        self.ax_right.plot(x_dr, y_dr, color='green', linestyle='dashed')

        self.ax.set_ylim(0, 1.1)
        self.ax_right.set_ylim(0, max(np.max(y_pr), np.max(y_dr)) * 1.1)

        self.ax.spines['left'].set_color('red')
        self.ax.spines['right'].set_color('green')
        self.ax.spines[['left', 'right']].set_linewidth(2)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='grey', lw=1, label=self.label_plain),
            Line2D([0], [0], color='grey', lw=1, linestyle='dashed', label=self.label_dashed),
        ]
        self.ax.legend(handles=legend_elements, loc='best')
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label_left)
        self.ax_right.set_ylabel(self.y_label_right)
        self.ax.grid()
