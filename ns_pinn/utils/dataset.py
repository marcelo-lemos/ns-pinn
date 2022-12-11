from typing import Tuple

import numpy as np
from torch.utils.data import Dataset

# TODO: receive np.ndarray as input and separate into x, y, t, u, v, p
class NavierStokes2DDataset(Dataset):
    def __init__(self, csv_file: str) -> None:
        data = np.genfromtxt(csv_file, delimiter=',')
        self.X, self.Y = np.array_split(data, [3], axis=1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.X[index], self.Y[index]
