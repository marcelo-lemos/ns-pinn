import logging
import os

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
            case 'wandb':
                logger.info('Instantiating wandb logger...')
                pl_logger = WandbLogger(project=cfg.mlops.project)
            case _:
                logger.critical(
                    f'Unsupported MLOps platform: {cfg.mlops.platform}')
                exit()
    else:
        logger.info('Instantiating CSV logger...')
        pl_logger = CSVLogger(os.getcwd())

    logger.info('Instantiating model...')
    ns_2d = NavierStokes2DPINN.load_from_checkpoint('')
    ns_2d.eval()

    logger.info('Loading dataset...')
    data_path = hydra.utils.to_absolute_path(cfg.dataset)
    dataset = NavierStokes2DDataset(data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.num_workers
    )

    for batch_idx, batch in enumerate(dataloader):
        X, Y = batch
        predictions = ns_2d(X)
        print(predictions)


if __name__ == '__main__':
    cs = ConfigStore.instance()
    cs.store(name='ns_pinn_config', node=NSPINNConfig)
    main()
