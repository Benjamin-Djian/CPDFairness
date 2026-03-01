import torch
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        running_acc = 0
        with torch.no_grad():
            for inputs, labels in loader:

                logits = self.model(inputs)
                loss = self.criterion(logits, labels)

                running_loss += loss.item()

                running_acc += torch.eq(logits.argmax(1), labels).sum().item()

        total_loss = running_loss / len(loader)
        total_accuracy = running_acc / len(loader)

        return total_loss, total_accuracy

    def train_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        running_acc = 0

        for inputs, labels in train_loader:
            self.optimizer.zero_grad()

            logits, _ = self.model(inputs)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            running_acc += torch.eq(logits.argmax(1), labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = running_acc / len(train_loader)

        return train_loss, train_accuracy

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs=10):
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
