import torch
from torch import nn
from torch.nn import NLLLoss
from torch.optim import Adam

from src.model.binary_classificator import BinaryClassificator
from src.model.trainer import Trainer, TrainHistory


class TestTrainHistory:
    """Tests for TrainHistory class."""

    def test_init(self):
        """Test TrainHistory initialization."""
        history = TrainHistory(
            train_loss=[0.5, 0.3],
            train_acc=[0.8, 0.9],
            val_loss=[0.6, 0.4],
            val_acc=[0.75, 0.85]
        )

        assert history.train_loss == [0.5, 0.3]
        assert history.train_acc == [0.8, 0.9]
        assert history.val_loss == [0.6, 0.4]
        assert history.val_acc == [0.75, 0.85]


class TestTrainer:
    """Tests for Trainer class."""

    def test_init(self):
        """Test Trainer initialization."""
        model = BinaryClassificator(input_dim=10, hidden_dims=[5], activation_fctn=nn.ReLU(), seed=42)
        trainer = Trainer(model=model, loss_fctn=NLLLoss(), optimizer=Adam(model.parameters()))

        assert trainer.model is model
        assert isinstance(trainer.optimizer, torch.optim.Adam)
        assert isinstance(trainer.criterion, nn.NLLLoss)
