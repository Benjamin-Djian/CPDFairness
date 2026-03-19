from pathlib import Path

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import src.utils.env as e
from src.model.binary_classificator import BinaryClassificator
from src.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(name=__name__)


class TrainHistory:
    """Tracks training and validation loss/accuracy across epochs."""

    def __init__(self, train_loss: list[float], train_acc: list[float], val_loss: list[float], val_acc: list[float]):
        self.nbr_epochs = len(train_loss)
        self.train_loss = train_loss
        self.train_acc = train_acc
        self.val_loss = val_loss
        self.val_acc = val_acc
        self.validate_history()

    def validate_history(self):
        if self.train_loss == [] or self.train_acc == [] or self.val_loss == [] or self.val_acc == []:
            raise ValueError("TrainHistory: cannot consider empty train history")
        if len(self.train_loss) != len(self.train_acc):
            raise ValueError("TrainHistory: train loss and accuracy does not have the same size")
        if len(self.val_loss) != len(self.val_acc):
            raise ValueError("TrainHistory: validation loss and validation does not have the same size")

    def __repr__(self):
        return (f"Train loss : {self.train_loss}\n "
                f"Train acc : {self.train_acc}\n"
                f"Validation loss : {self.val_loss}\n "
                f"Validation acc : {self.val_acc}\n")

    def save_fig(self, save_path: Path):
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=e.DEFAULT_FIGURE_SIZE)
        ax1.plot([x for x in range(1, self.nbr_epochs + 1)], self.train_loss, label="Training Loss")
        ax1.plot([x for x in range(1, self.nbr_epochs + 1)], self.val_loss, label="Validation Loss")
        ax1.set_title("Training Loss")
        ax1.legend()
        ax1.set_xlabel('Number of epochs')
        ax2.plot([x for x in range(1, self.nbr_epochs + 1)], self.train_acc, label="Training Accuracy")
        ax2.plot([x for x in range(1, self.nbr_epochs + 1)], self.val_acc, label="Validation Accuracy")
        ax2.set_title("Accuracy")
        ax2.legend()
        ax2.set_xlabel('Number of epochs')

        save_path_fig = Path(save_path).with_suffix('.pdf')
        fig.savefig(save_path_fig, dpi=e.DEFAULT_DPI, format='pdf')


class Trainer:
    """Trainer for BinaryClassificator with evaluation and training methods."""

    def __init__(self, model: BinaryClassificator, loss_fctn: nn.Module, optimizer: Optimizer):
        self.model = model
        self.optimizer = optimizer
        self.criterion = loss_fctn

    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        running_acc = 0
        total_samples = 0

        with torch.no_grad():
            for batch in loader:
                _, inputs, labels = batch
                logits = self.model(inputs)[0]
                loss = self.criterion(logits, labels.long())

                running_loss += loss.item()
                running_acc += torch.eq(logits.argmax(1), labels.long()).sum().item()
                total_samples += labels.size(0)

        total_loss = running_loss / total_samples
        total_accuracy = running_acc / total_samples

        return total_loss, total_accuracy

    def train_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        running_acc = 0
        total_samples = 0

        for batch in train_loader:
            _, inputs, labels = batch
            logits, _ = self.model(inputs)
            loss = self.criterion(logits, labels.long())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            running_acc += torch.eq(logits.argmax(1), labels.long()).sum().item()
            total_samples += labels.size(0)

        train_loss = running_loss / total_samples
        train_accuracy = running_acc / total_samples

        return train_loss, train_accuracy

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> TrainHistory:
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []
        for epoch in range(epochs):
            logger.info(f"Epochs {epoch + 1}/{epochs}")
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)

        return TrainHistory(train_loss_list, train_acc_list, val_loss_list, val_acc_list)
