from typing import Any, Literal, Sequence

import numpy as np
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter


class CSVPredictionWriter(BasePredictionWriter):
    def __init__(self, output_file: str, write_interval: Literal['batch', 'epoch', 'batch_and_epoch'] = "batch") -> None:
        super().__init__(write_interval)
        self.output_file = output_file

    def write_on_epoch_end(self, trainer: Trainer, pl_module: LightningModule, predictions: Sequence[Any], batch_indices: Sequence[Any]) -> None:
        predictions = torch.cat(predictions)
        predictions = predictions.numpy(force=True)
        np.savetxt(self.output_file, predictions, delimiter=',')
