import polars as pl
import torch
from torch.utils.data import Dataset

# TODO: receive np.ndarray as input and separate into x, y, t, u, v
class NavierStokes2DDataset(Dataset):
    def __init__(self, csv_file: str) -> None:
        df = pl.read_csv(csv_file, has_header=False, dtypes=[pl.Float32]*5, new_columns=['x','y','t','u','v'])
        self.data = df.select(['x', 'y', 't'])
        self.targets = df.select(['u', 'v'])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, t = [torch.tensor([value], dtype=torch.float32) for value in self.data.row(index)]
        u, v = [torch.tensor([value], dtype=torch.float32) for value in self.targets.row(index)]
        return x, y, t, u, v
