from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader

from src.model.classificator import Classificator
from src.preprocessing.dataset import IndexDataset


class ActivationFilter(ABC):
    @abstractmethod
    def get_mask(self,
                 indexes: torch.Tensor,
                 inputs: torch.Tensor,
                 targets: torch.Tensor,
                 predictions: torch.Tensor,
                 activations: torch.Tensor) -> torch.Tensor:
        pass


class ClassificationFilter(ActivationFilter):
    """Filter by correct/incorrect classification."""

    def __init__(self, keep_correct: bool = True):
        self.keep_correct = keep_correct

    def get_mask(self,
                 indexes: torch.Tensor,
                 inputs: torch.Tensor,
                 targets: torch.Tensor,
                 predictions: torch.Tensor,
                 activations: torch.Tensor) -> torch.Tensor:
        is_correct = predictions == targets.long()
        return is_correct if self.keep_correct else ~is_correct


class FeatureFilter(ActivationFilter):
    """Filter by feature column value."""

    def __init__(self, column_name: str, value: int, dataset: IndexDataset):
        self.column_name = column_name
        self.value = value
        self.dataset = dataset

    def get_mask(self,
                 indexes: torch.Tensor,
                 inputs: torch.Tensor,
                 targets: torch.Tensor,
                 predictions: torch.Tensor,
                 activations: torch.Tensor) -> torch.Tensor:

        col_idx = self.dataset.get_index_col(self.column_name)
        return inputs[:, col_idx] == self.value


class FilterChain:
    """Compose multiple filters."""

    def __init__(self, filters: list[ActivationFilter]):
        self.filters = filters

    def apply(self,
              indexes: torch.Tensor,
              inputs: torch.Tensor,
              targets: torch.Tensor,
              predictions: torch.Tensor,
              activations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = None
        for f in self.filters:
            f_mask = f.get_mask(indexes, inputs, targets, predictions, activations)
            mask = f_mask if mask is None else mask & f_mask

        return indexes[mask], activations[mask]


class ActivationGetter:
    """Class to query activation levels based on their node or on their inputs."""

    def __init__(self, indexes: torch.Tensor, activations: torch.Tensor):
        self.activations = activations
        self.indexes = indexes
        self.index_map = {int(idx): i for i, idx in enumerate(indexes)}

    def get_by_index(self, index: int) -> torch.Tensor:
        row = self.index_map[index]
        return self.activations[row]

    def iterate_indexes(self):
        for index in self.indexes:
            yield index, self.get_by_index(index)

    def get_by_node(self, node_id: int) -> torch.Tensor:
        return self.activations[:, node_id]

    def num_inputs(self) -> int:
        return self.activations.shape[0]

    def num_nodes(self) -> int:
        return self.activations.shape[1]


class ActivationExtractor:
    """Extract activation levels for all neurons, for all inputs."""

    def __init__(self, classificator: Classificator):
        self.classificator = classificator

    def extract(
            self,
            dataloader: DataLoader,
            filters: list[ActivationFilter] | None = None) -> ActivationGetter:
        """Extract activations with optional filters.

        Args:
            dataloader: DataLoader providing batches of (index, inputs, targets)
            filters: List of filters to apply. If None, returns all activations.

        Returns:
            ActivationGetter with filtered activations and indexes.
        """
        self.classificator.eval()

        all_indexes = []
        all_inputs = []
        all_targets = []
        all_predictions = []
        all_activations = []

        with torch.no_grad():
            for index, inputs, targets in dataloader:
                outputs, hidden = self.classificator(inputs)
                predictions = torch.argmax(outputs, dim=1)

                all_indexes.append(index)
                all_inputs.append(inputs)
                all_targets.append(targets)
                all_predictions.append(predictions)
                all_activations.append(hidden)

        indexes = torch.cat(all_indexes)
        inputs = torch.cat(all_inputs)
        targets = torch.cat(all_targets)
        predictions = torch.cat(all_predictions)
        activations = torch.cat(all_activations)

        if filters:
            filter_chain = FilterChain(filters)
            indexes, activations = filter_chain.apply(
                indexes, inputs, targets, predictions, activations
            )

        return ActivationGetter(activations=activations, indexes=indexes)
