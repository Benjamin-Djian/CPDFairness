from abc import abstractmethod, ABC

import numpy as np
from matplotlib import pyplot as plt

from src.likelihood.likelihood import LikelihoodScore


class Visualisation(ABC):
    def __init__(self, size: tuple[float, float], title: str | None, x_label: str | None, y_label: str | None):
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(*size)
        self.ax.set_title(title, wrap=True)

        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)

    @abstractmethod
    def plot(self):
        pass


class LikelihoodVisualisation(Visualisation):
    def __init__(self,
                 serie_1: list[LikelihoodScore],
                 serie_2: list[LikelihoodScore],
                 size: tuple[float, float],
                 title: str | None,
                 x_label: str | None,
                 y_label: str | None):
        super().__init__(size, title, x_label, y_label)
        self.serie_1 = serie_1
        self.serie_2 = serie_2

    def plot(self):
        pass


class PlotCPDFractionPop(LikelihoodVisualisation):
    def __init__(self,
                 serie_1: list[LikelihoodScore],
                 serie_2: list[LikelihoodScore],
                 size: tuple[float, float],
                 title: str | None,
                 resolution: int = 100,
                 label_1: str | None = None,
                 label_2: str | None = None):
        super().__init__(serie_1, serie_2, size, title, "Normalized CPD", "Fraction of respective population")
        self.resolution = resolution
        self.label_1 = label_1
        self.label_2 = label_2

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

        self.ax.plot(frac_pop_1, label=self.label_1, color='blue')
        self.ax.plot(frac_pop_2, label=self.label_1, color='red')

        plt.grid()
        plt.legend()
        plt.show()
