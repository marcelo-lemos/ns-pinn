import logging
import os

import numpy as np
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from models.components.mlp import MLP
from models.navier_stokes_pinn import NavierStokes2DPINN
from utils.config import NSPINNConfig
from utils.dataset import NavierStokes2DDataset


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
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False
    )

    logger.info('Starting training...')
    trainer = L.Trainer(
        logger=lightning_logger,
        max_epochs=cfg.training.epochs,
        accelerator='gpu',
        enable_progress_bar=False
    )
    trainer.fit(ns_2d, train_dataloader, val_dataloader)
    logger.info('Finished training.')

    logger.info('Starting predicting...')
    predictions = trainer.predict(ns_2d, val_dataloader)
    predictions = torch.cat(predictions)
    predictions = predictions.numpy(force=True)
    path_no_ext = os.path.splitext(data_path)[0]
    np.savetxt(f'{path_no_ext}-predictions.csv', predictions, delimiter=",")
    logger.info('Finished predicting.')


if __name__ == '__main__':
    cs = ConfigStore.instance()
    cs.store(name='ns_pinn_config', node=NSPINNConfig)
    main()
