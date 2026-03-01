import torch
from torch.utils.data import DataLoader

from src.model.classificator import Classificator


class ActivationGetter:
    """Class to query activation levels based on their node or on their inputs"""

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
    """Extract activation levels for all neurons, for all inputs"""

    def __init__(self, classificator: Classificator):
        self.classificator = classificator

    def extract(self, dataloader: DataLoader):
        self.classificator.eval()

        activations = []
        indexes = []

        with torch.no_grad():
            for batch in dataloader:
                index, inputs, targets = batch

                _, hidden = self.classificator(inputs)

                activations.append(hidden)
                indexes.append(index)

        index_tensor = torch.cat(indexes, dim=0)
        activation_tensor = torch.cat(activations, dim=0)

        return ActivationGetter(activations=activation_tensor, indexes=index_tensor)
