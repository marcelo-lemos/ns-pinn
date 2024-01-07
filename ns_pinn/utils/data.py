import numpy as np
from numpy.typing import NDArray
import polars as pl
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


class NavierStokes2DDataset(Dataset):
    def __init__(self, csv_file: str) -> None:
        df = pl.read_csv(csv_file, has_header=False)
        self.data = df.to_numpy().astype(np.float32)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> list[NDArray]:
        row = self.data[index]
        return tuple(np.hsplit(row, row.shape[0]))


class NavierStokes2DDataModule(L.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int, num_workers: int) -> None:
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        self.dataset = NavierStokes2DDataset(self.data_path)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
