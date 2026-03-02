import pytest
import torch
from torch import nn

from src.model.classificator import Classificator
from src.model.trainer import Trainer, TrainHistory
from torch.utils.data import DataLoader, TensorDataset


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
        model = Classificator(input_dim=10, hidden_dims=[5], num_classes=2, seed=42)
        trainer = Trainer(model=model, learning_rate=0.001)
        
        assert trainer.model is model
        assert isinstance(trainer.optimizer, torch.optim.Adam)
        assert isinstance(trainer.criterion, nn.NLLLoss)
