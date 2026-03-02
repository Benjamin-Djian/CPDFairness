import math

from torch.utils.data import DataLoader

from src.likelihood.activation_extractor import ActivationExtractor, ActivationFilter
from src.likelihood.histograms import Histogram
from src.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(name=__name__)

nk_type = tuple[int, int]
table_type = dict[nk_type, float]
prob_table_type = dict[nk_type, dict[int, float]]


class LikelihoodScore:
    def __init__(self, input_id: int, score: float):
        self.input_id = input_id
        self.score = score

    def __repr__(self):
        return f"{self.input_id} {self.score}"


class LikelihoodCalculator:
    def __init__(self, extractor: ActivationExtractor):
        self.extractor = extractor

    def compute_likelihood(self,
                           data_loader: DataLoader,
                           histograms: list[Histogram],
                           filters: list[ActivationFilter] | None = None) -> list[LikelihoodScore]:
        likelihoods = []
        act_getter = self.extractor.extract(data_loader, filters=filters)
        for index, activation in act_getter.iterate_indexes():
            likelihood = 0.
            for hist in histograms:
                node_activation = activation[hist.node_id]
                try:
                    node_activation = float(node_activation.item())
                except RuntimeError:
                    raise ValueError(f"Impossible to parse activation level {node_activation} to float")

                hist_prob = hist.compute_hist_prob(node_activation)
                likelihood = likelihood - math.log(hist_prob)

            likelihoods.append(LikelihoodScore(input_id=index, score=likelihood))

        return likelihoods
