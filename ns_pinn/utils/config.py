from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class MLOpsConfig:
    workspace: str
    project: str


@dataclass
class NNConfig:
    layers: list[int]
    activation: str
    dropout: float


@dataclass
class OptimizerConfig:
    learning_rate: float
    learning_rate_decay: float
    weight_decay: float


@dataclass
class ModelConfig:
    nn: NNConfig
    optimizer: OptimizerConfig
    data_loss_coef: float
    physics_loss_coef: float
    rho: float
    mu: float
    checkpoint_path: str


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int


@dataclass
class NSPINNConfig:
    dataset: str
    dataset_size: int
    validation_interval: int
    seed: int
    num_workers: int
    prod_mode: bool
    mlops: MLOpsConfig
    model: ModelConfig
    training: TrainingConfig
