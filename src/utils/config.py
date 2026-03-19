from pathlib import Path
from typing import List, Literal, ClassVar

import yaml
from pydantic import BaseModel, Field
from torch import nn

from src.preprocessing.data_preparator import DataPreparator, AdultDataPreparator, GermanDataPreparator, \
    LawDataPreparator
from src.preprocessing.prepro_operations import PreprocessingOperation, DownSampling, UpSampling, \
    CorrelationRemoverPrepro, DisparateImpactRemoverPrepro


# -------- Experiment --------
class SaveConfig(BaseModel):
    """Config for saving specific data of the experiment in the experiment dir
    :param dataset: Save the preprocessed train, test, and valid set (in CSV format).
    :param histogram: Save the histograms of activation level of neurones (in CSV format).
    :param likelihood: Save the likelihood score (in CSV format).
    :param model: Save the Pytorch model state (in .pth format)
    """
    dataset: bool
    histogram: bool
    likelihood: bool
    model: bool


class ExperimentConfig(BaseModel):
    """Config for experiment settings
    :param seed: Random seed for reproducibility.
    :param save: SaveConfig for saving experiment data (dataset, histogram, likelihood, model).
    """
    seed: int
    save: SaveConfig


# -------- Data --------
class DataConfig(BaseModel):
    """Config for dataset and data loading settings
    :param dataset: Name of the dataset to use (adult, german, or law).
    :param sens_attr: Name of the sensitive attribute column for fairness analysis, must be a binary column.
    :param train_split: Proportion of data to use for training (between 0 and 1).
    :param valid_split: Proportion of data to use for validation (between 0 and 1).
    :param batch_size: Number of samples per batch during training (must be greater than 0).
    """
    dataset: str
    sens_attr: str
    train_split: float = Field(..., ge=0, le=1)
    valid_split: float = Field(..., ge=0, le=1)
    batch_size: int = Field(..., gt=0)

    preparators: ClassVar[dict[str, type]] = {
        "adult": AdultDataPreparator,
        "german": GermanDataPreparator,
        "law": LawDataPreparator,
    }

    def build_preparator(self, additional_steps) -> DataPreparator:
        try:
            return self.preparators[self.dataset](sens_attr_name=self.sens_attr, additional_steps=additional_steps)
        except KeyError:
            raise ValueError(f"Unknown dataset name: {self.dataset}")


# -------- Preprocessing --------
class ClassBalanceConfig(BaseModel):
    """Config for class balancing operations to handle imbalanced datasets
    :param downsampling: Whether to apply random undersampling to the majority class.
    :param upsampling: Whether to apply random oversampling to the minority class.
    """
    downsampling: bool
    upsampling: bool

    def build(self, seed: int) -> List[PreprocessingOperation]:
        steps = []
        if self.downsampling and self.upsampling:
            raise ValueError("Can not use downsampling AND upsampling at the same time. Please modify config options")
        if self.downsampling:
            steps.append(DownSampling(seed))
        if self.upsampling:
            steps.append(UpSampling(seed))
        return steps


class FairnessConfig(BaseModel):
    """Config for fairness-related preprocessing operations
    :param correlation_remover: Whether to apply Correlation Remover algorithm from FairLearn.
    :param disparate_impact_remover: Whether to apply Disparate Impact Remover algorithm from AIF360.
    """
    correlation_remover: bool
    disparate_impact_remover: bool

    def build(self, sens_attr: str) -> List[PreprocessingOperation]:
        steps = []
        if self.correlation_remover:
            steps.append(CorrelationRemoverPrepro(sens_attr_name=sens_attr))
        if self.disparate_impact_remover:
            steps.append(DisparateImpactRemoverPrepro(sens_attr_name=sens_attr))
        return steps


class PreprocessingConfig(BaseModel):
    """Config for all preprocessing operations
     :param class_balance: ClassBalanceConfig for handling imbalanced datasets.
     :param fairness: FairnessConfig for fairness-related preprocessing.
     """
    class_balance: ClassBalanceConfig
    fairness: FairnessConfig

    def build(self, seed: int, sens_attr: str) -> list[PreprocessingOperation]:
        steps = []
        steps.extend(self.class_balance.build(seed))
        steps.extend(self.fairness.build(sens_attr))
        return steps


# -------- Model --------
class LayersConfig(BaseModel):
    """Config for neural network hidden layer architecture
        :param hidden_dims: List of integers specifying the number of units in each hidden layer.
        :param dropout: Dropout rate applied after each hidden layer (between 0 and 1). Can be set to 0 for no dropout.
    """
    hidden_dims: List[int]
    dropout: float = Field(..., ge=0, le=1)


class ActivationConfig(BaseModel):
    """Config for neural network activation function
        :param type: Type of activation function (relu, leaky_relu, tanh, or sigmoid).
        :param neg_slope: Negative slope for leaky_relu activation (required when type is leaky_relu).
    """
    type: Literal["relu", "leaky_relu", "tanh", "sigmoid"]
    neg_slope: float | None = None

    def model_post_init(self, __context):
        if self.type == "leaky_relu" and self.neg_slope is None:
            raise ValueError("neg_slope must be provided for leaky_relu")

    def build(self) -> nn.Module:
        if self.type == "relu":
            return nn.ReLU()
        elif self.type == "leaky_relu":
            return nn.LeakyReLU(negative_slope=self.neg_slope)
        elif self.type == "tanh":
            return nn.Tanh()
        elif self.type == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Activation function {self.type} not implemented")


class ModelConfig(BaseModel):
    """Config for the neural network model architecture
        :param input_dim: Number of input features (must be greater than 0).
        :param layers: LayersConfig for hidden layer dimensions and dropout.
        :param activation: ActivationConfig for the activation function.
    """
    input_dim: int = Field(..., gt=0)
    layers: LayersConfig
    activation: ActivationConfig


# -------- Training --------
class TrainingConfig(BaseModel):
    """Config for model training settings
        :param learning_rate: Learning rate for the optimizer (must be greater than 0).
        :param epochs: Number of training epochs (must be greater than 0).
        :param use_class_weight: Whether to use weighted loss function to handle imbalanced data.
    """
    learning_rate: float = Field(..., gt=0)
    epochs: int = Field(..., gt=0)
    use_class_weight: bool


# -------- Root Config --------
class Config(BaseModel):
    """Root configuration class for the entire experiment pipeline
        :param experiment: ExperimentConfig for experiment settings (seed, save options).
        :param data: DataConfig for dataset and data loading settings.
        :param preprocessing: PreprocessingConfig for preprocessing operations.
        :param model: ModelConfig for neural network architecture.
        :param training: TrainingConfig for training hyperparameters.
    """
    experiment: ExperimentConfig
    data: DataConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    training: TrainingConfig

    model_config = {
        "extra": "forbid",  # Raise error if unknown keys
        "frozen": True  # Avoid config modification at runtime
    }

    def build_preparator(self) -> DataPreparator:
        steps = self.preprocessing.build(
            seed=self.experiment.seed,
            sens_attr=self.data.sens_attr
        )
        return self.data.build_preparator(additional_steps=steps)


def load_config(path: Path) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return Config(**raw)
