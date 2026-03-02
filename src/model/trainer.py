import torch
from torch import nn
from torch.utils.data import DataLoader

from src.model.classificator import Classificator
from src.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(name=__name__)


class TrainHistory:
    def __init__(self, train_loss: list[float], train_acc: list[float], val_loss: list[float], val_acc: list[float]):
        self.train_loss = train_loss
        self.train_acc = train_acc
        self.val_loss = val_loss
        self.val_acc = val_acc

    def __repr__(self):
        return (f"Train loss : {self.train_loss}\n "
                f"Train acc : {self.train_acc}\n"
                f"Validation loss : {self.val_loss}\n "
                f"Validation acc : {self.val_acc}\n")


class Trainer:
    def __init__(self, model: Classificator, learning_rate: float):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.NLLLoss()

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
