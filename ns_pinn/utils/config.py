from dataclasses import dataclass


@dataclass
class NeptuneConfig:
    workspace: str
    project: str
    prefix: str


@dataclass
class NNConfig:
    layers: list[int]
    activation: str
    dropout: float


@dataclass
class OptimizerConfig:
    learning_rate: float


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
    neptune: NeptuneConfig
    model: ModelConfig
    training: TrainingConfig
