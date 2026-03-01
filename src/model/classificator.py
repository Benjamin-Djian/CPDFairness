import torch
from torch import nn

from src.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(name=__name__)


class Classificator(nn.Module):
    def __init__(self, input_dim: int,
                 hidden_dims: list[int],
                 num_classes: int,
                 negative_slope=0.2,
                 dropout: float = 0.0,
                 seed=42):

        super().__init__()
        torch.manual_seed(seed)
        layers = []
        prev_dim = input_dim
        if not hidden_dims:
            raise ValueError("Model architecture must include at least one hidden layer")
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope))

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        self.last_hidden_dim = hidden_dims[-1]
        self.seq = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_data = self.seq(x)
        return self.output(hidden_data), hidden_data
