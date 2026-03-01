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

    def extract(self, dataloader: DataLoader, filter_correct: bool = True):
        self.classificator.eval()

        all_activations = []
        all_indexes = []
        all_correct_mask = []

        with torch.no_grad():
            for batch in dataloader:
                index, inputs, targets = batch

                outputs, hidden = self.classificator(inputs)
                predictions = torch.argmax(outputs, dim=1)
                true_labels = targets.long()

                is_correct = (predictions == true_labels)
                all_correct_mask.append(is_correct)
                all_activations.append(hidden)
                all_indexes.append(index)

        index_tensor = torch.cat(all_indexes, dim=0)
        activation_tensor = torch.cat(all_activations, dim=0)
        correct_mask = torch.cat(all_correct_mask, dim=0)

        if filter_correct:
            index_tensor = index_tensor[correct_mask]
            activation_tensor = activation_tensor[correct_mask]

        return ActivationGetter(activations=activation_tensor, indexes=index_tensor)
