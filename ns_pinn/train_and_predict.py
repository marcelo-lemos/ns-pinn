import logging
import os

import numpy as np
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from models.components.mlp import MLP
from models.navier_stokes_pinn import NavierStokes2DPINN
from utils.config import NSPINNConfig
from utils.data import NavierStokes2DDataModule
from utils.csv_prediction_writer import CSVPredictionWriter


@hydra.main(version_base=None, config_path='../config', config_name='config.yaml')
def main(cfg: NSPINNConfig) -> None:
    logger = logging.getLogger(__name__)

    logger.info('Starting lightning logger...')
    if cfg.prod_mode:
        logger.info('Instantiating wandb logger...')
        lightning_logger = WandbLogger(
            project=cfg.mlops.project, entity=cfg.mlops.workspace)
        lightning_logger.experiment.config.update(OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True))
    else:
        logger.info('Instantiating CSV logger...')
        lightning_logger = CSVLogger(os.getcwd())

    logger.info('Instantiating model...')
    ns_2d = NavierStokes2DPINN(
        layers=cfg.model.nn.layers,
        activation=cfg.model.nn.activation,
        dropout=cfg.model.nn.dropout,
        learning_rate=cfg.model.optimizer.learning_rate,
        lr_decay=cfg.model.optimizer.learning_rate_decay,
        weight_decay=cfg.model.optimizer.weight_decay,
        data_loss_coef=cfg.model.data_loss_coef,
        physics_loss_coef=cfg.model.physics_loss_coef,
        rho=cfg.model.rho,
        mu=cfg.model.mu
    )

    logger.info('Configuring datasets...')
    data_path = hydra.utils.to_absolute_path(cfg.dataset)
    datamodule = NavierStokes2DDataModule(data_path, cfg.training.batch_size, cfg.num_workers)

    logger.info('Creating callbacks...')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    path_no_ext = os.path.splitext(data_path)[0]
    prediction_writer = CSVPredictionWriter(f'{path_no_ext}-predictions.csv', 'epoch')

    logger.info('Starting training...')
    
    trainer = L.Trainer(
        accelerator='gpu',
        logger=lightning_logger,
        callbacks=[lr_monitor, prediction_writer],
        max_epochs=cfg.training.epochs,
        check_val_every_n_epoch=cfg.validation_interval,
        enable_progress_bar=False,
        # profiler='simple',
    )
    trainer.fit(ns_2d, datamodule=datamodule)
    logger.info('Finished training.')

    logger.info('Starting predicting...')
    trainer.predict(ns_2d, datamodule=datamodule, return_predictions=False)
    logger.info('Finished predicting.')


if __name__ == '__main__':
    cs = ConfigStore.instance()
    cs.store(name='ns_pinn_config', node=NSPINNConfig)
    main()
