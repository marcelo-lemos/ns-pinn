import logging

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger

from models.components.mlp import MLP
from models.navier_stokes_pinn import NavierStokes2DPINN
from utils.config import NSPINNConfig
from utils.dataset import NavierStokes2DDataset


@hydra.main(version_base=None, config_path='../config', config_name='config.yaml')
def main(cfg: NSPINNConfig) -> None:
    logger = logging.getLogger(__name__)

    logger.info('Instantiating neptune logger...')
    neptune_logger = NeptuneLogger(
        project=f'{cfg.neptune.workspace}/{cfg.neptune.project}',
        prefix=cfg.neptune.prefix
    )
    logger.info('Logging hyperparameters...')
    neptune_logger.experiment['parameters'] = cfg

    logger.info('Configuring metrics...')
    val_metrics = torchmetrics.MetricCollection({
        'MAE': torchmetrics.MeanAbsoluteError(),
        'MAPE': torchmetrics.MeanAbsolutePercentageError(),
        'MSE': torchmetrics.MeanSquaredError()
    }, prefix='validation/')

    logger.info('Instantiating model...')
    match cfg.model.nn.activation:
        case 'hardswish':
            activation = nn.Hardswish
        case 'relu':
            activation = nn.ReLU
        case 'silu':
            activation = nn.SiLU
        case 'tanh':
            activation = nn.Tanh
        case _:
            logger.error(
                f'Unsuported activation {cfg.model.nn.activation}. Defaulting to Tanh.')
            activation = nn.Tanh
    net = MLP(cfg.model.nn.layers, activation, cfg.model.nn.dropout)

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.model.optimizer.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.training.epochs)

    ns_2d = NavierStokes2DPINN(
        net=net,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        val_metrics=val_metrics,
        data_loss_coef=cfg.model.data_loss_coef,
        physics_loss_coef=cfg.model.physics_loss_coef,
        rho=cfg.model.rho,
        mu=cfg.model.mu
    )

    logger.info('Loading dataset...')
    data_path = hydra.utils.to_absolute_path(cfg.dataset)
    dataset = NavierStokes2DDataset(data_path)
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.num_workers
    )

    logger.info('Starting training...')
    trainer = pl.Trainer(
        logger=neptune_logger,
        max_epochs=cfg.training.epochs,
        accelerator='auto'
    )
    trainer.fit(ns_2d, train_dataloader, val_dataloader)
    logger.info('Finished!')


if __name__ == '__main__':
    cs = ConfigStore.instance()
    cs.store(name='ns_pinn_config', node=NSPINNConfig)
    main()
