import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

# TODO: receive np.ndarray as input and separate into x, y, t, u, v
class NavierStokes2DDataset(Dataset):
    def __init__(self, csv_file: str, size:int,  batch_size: int) -> None:
        self.df = pl.scan_csv(csv_file, has_header=False, new_columns=['x','y','t','u','v'])
        self.size = size
        self.batch_size = batch_size

    def __len__(self) -> int:
        return (self.size // self.batch_size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        offset = index * self.batch_size
        batch = self.df.slice(offset, self.batch_size).select(['x','y','t','u','v']).collect().to_numpy().astype(np.float32)
        x, y, t, u, v = np.hsplit(batch, 5)

        return x, y, t, u, v
