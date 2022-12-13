from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class MLOpsConfig:
    platform: str
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
    weight_decay: float


@dataclass
class ModelConfig:
    nn: NNConfig
    optimizer: OptimizerConfig
    data_loss_coef: float
    physics_loss_coef: float
    rho: float
    mu: float


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int


@dataclass
class NSPINNConfig:
    dataset: str
    seed: int
    num_workers: int
    prod_mode: bool
    mlops: MLOpsConfig
    model: ModelConfig
    training: TrainingConfig
