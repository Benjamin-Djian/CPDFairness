from abc import ABC, abstractmethod
from statistics import pstdev

import numpy as np
import torch
from torch.utils.data import DataLoader

import src.utils.env as e
from src.likelihood.activation_extractor import (
    ActivationExtractor,
    ActivationFilter,
    ClassificationFilter,
    FeatureFilter,
)
from src.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(name=__name__)


class Histogram(ABC):
    def __init__(self, node_id: int, bins: np.ndarray, freq: np.ndarray):
        self.node_id = node_id
        if len(bins) < 2:
            raise ValueError("An Histogram needs at least one bins to be defined.")
        if len(bins) - 1 != len(freq):
            raise ValueError("Inconsistent numbers of bins and frequencies")
        if np.sum(freq) <= 0:
            raise ValueError("Frequency can not be zero")
        self.bins = bins
        self.freq = freq
        self.lower_bound = min(bins)
        self.step = bins[1] - bins[0]

    @abstractmethod
    def compute_hist_prob(self, activation: float):
        pass


class UniBinHistogram(Histogram):
    def compute_hist_prob(self, activation: float) -> float:
        if abs(activation - self.lower_bound) > e.EPSILON:
            return e.LOW_SMOOTHED_PROB
        else:
            return 1.


class MultiBinsHistogram(Histogram):
    def compute_hist_prob(self, activation: float) -> float:
        if (activation - self.lower_bound) < e.EPSILON:
            return e.LOW_SMOOTHED_PROB
        else:
            cur_bin_id = ((activation - self.lower_bound) / self.step)
            cur_bin_id = int(round(cur_bin_id, e.EPSILON_PREC))
            if cur_bin_id < 0:
                raise ValueError(f'ERROR compute_hist_prob : cur_bin_id {cur_bin_id} is undefined')

            return self.freq[cur_bin_id] / np.sum(self.freq)


class HistogramConstructor:
    def __init__(self, node_id, act_extract: ActivationExtractor):
        self.node_id = node_id
        self.extractor = act_extract

    def check_null_std(self, std_dev: float):
        if std_dev < -e.EPSILON:
            raise ValueError(
                f'ERROR construct_hist: Invalid distribution for node {self.node_id}.'
                f' The standard deviation is negative : {std_dev}'
            )

    def check_correct_bins(self, bins: np.ndarray):
        step = bins[1] - bins[0]

        if len(bins) < 2:
            raise ValueError(f'ERROR check_correct_bins: too few bins for node {self.node_id}')
        if step < e.EPSILON:
            raise ValueError(f'ERROR check_correct_bins: step is null for node {self.node_id}')

        nbr_bins = len(bins[1:])
        if nbr_bins > e.POS_UNDEF_INT:
            raise ValueError(
                f'ERROR check_correct_bins: There are {nbr_bins}, which is too high for the current precision. '
                f'Hist file would be incorrect')

    def construct_single_bins(self, activations: torch.Tensor) -> UniBinHistogram:
        min_act = min(activations)
        return UniBinHistogram(node_id=self.node_id,
                               bins=np.array([min_act, min_act]),
                               freq=np.array([len(activations)]))

    def construct_mult_bins(self, activations: torch.Tensor, std_dev: float) -> MultiBinsHistogram:
        min_act = torch.min(activations).item()
        max_act = torch.max(activations).item()
        hist, bins = np.histogram(activations, bins=np.arange(min_act,
                                                              max_act + 2 * std_dev,
                                                              std_dev - e.LOW_SMOOTHED_PROB))
        bins = np.round(bins, e.EPSILON_PREC)
        self.check_correct_bins(bins)

        return MultiBinsHistogram(node_id=self.node_id, bins=bins, freq=hist)

    def get_hist(
            self,
            data_loader: DataLoader,
            filters: list[ActivationFilter] | None = None) -> Histogram:

        act_getter = self.extractor.extract(data_loader, filters=filters)
        activations = act_getter.get_by_node(self.node_id)

        sigma_act = pstdev(activations)
        sigma_act = round(sigma_act, e.EPSILON_PREC) if sigma_act >= e.EPSILON else 0.0

        self.check_null_std(sigma_act)

        if -e.EPSILON <= sigma_act < e.EPSILON:
            histogram = self.construct_single_bins(activations)

        else:
            histogram = self.construct_mult_bins(activations, sigma_act)

        return histogram
