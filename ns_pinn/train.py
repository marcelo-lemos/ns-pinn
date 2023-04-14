import logging
import os

import numpy as np
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.loggers.wandb import WandbLogger

from models.components.mlp import MLP
from models.navier_stokes_pinn import NavierStokes2DPINN
from utils.config import NSPINNConfig
from utils.dataset import NavierStokes2DDataset


@hydra.main(version_base=None, config_path='../config', config_name='config.yaml')
def main(cfg: NSPINNConfig) -> None:
    logger = logging.getLogger(__name__)

    logger.info('Starting pytorch lightning logger...')
    if cfg.prod_mode:
        match cfg.mlops.platform:
            case 'neptune':
                logger.info('Instantiating neptune logger...')
                pl_logger = NeptuneLogger(
                    project=f'{cfg.mlops.workspace}/{cfg.mlops.project}',
                    prefix=''
                )
                logger.info('Logging hyperparameters...')
                pl_logger.experiment['parameters'] = OmegaConf.to_container(
                    cfg, resolve=True, throw_on_missing=True)
            case 'wandb':
                logger.info('Instantiating wandb logger...')
                pl_logger = WandbLogger(
                    project=cfg.mlops.project, entity=cfg.mlops.workspace)
                pl_logger.experiment.config.update(OmegaConf.to_container(
                    cfg, resolve=True, throw_on_missing=True))
            case _:
                logger.critical(
                    f'Unsupported MLOps platform: {cfg.mlops.platform}')
                exit()
    else:
        logger.info('Instantiating CSV logger...')
        pl_logger = CSVLogger(os.getcwd())

    logger.info('Instantiating model...')
    ns_2d = NavierStokes2DPINN(
        layers=cfg.model.nn.layers,
        activation=cfg.model.nn.activation,
        dropout=cfg.model.nn.dropout,
        learning_rate=cfg.model.optimizer.learning_rate,
        weight_decay=cfg.model.optimizer.weight_decay,
        training_epochs=cfg.training.epochs,
        data_loss_coef=cfg.model.data_loss_coef,
        physics_loss_coef=cfg.model.physics_loss_coef,
        rho=cfg.model.rho,
        mu=cfg.model.mu
    )

    logger.info('Loading dataset...')
    data_path = hydra.utils.to_absolute_path(cfg.dataset)
    dataset = NavierStokes2DDataset(data_path)
    train_dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )
    val_dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False
    )

    logger.info('Starting training...')
    trainer = pl.Trainer(
        logger=pl_logger,
        max_epochs=cfg.training.epochs,
        accelerator='gpu',
        devices=1
    )
    trainer.fit(ns_2d, train_dataloader, val_dataloader)
    logger.info('Finished training.')


if __name__ == '__main__':
    cs = ConfigStore.instance()
    cs.store(name='ns_pinn_config', node=NSPINNConfig)
    main()
