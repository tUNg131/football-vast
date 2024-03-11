import random
from typing import Tuple, List, Callable, Union, Optional

import torch
import h5py
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset


class SetOriginToCenterOfMassAndAlignXY:

    def __call__(self, sample: Tensor) -> Tensor:
        # Rotation matrix
        direction = self.direction(sample)
        sin = direction[1]
        cos = direction[0]
        rot = torch.tensor([[sin + cos, cos - sin],
                            [sin - cos, cos + sin]])

        # Translate to set the orgin to center of mass
        sample = sample - self.centroids_mean(sample)

        # Rotate to the x-y line
        sample = torch.matmul(sample, rot)
        return sample

    @staticmethod
    def centroids_mean(sample: Tensor) -> Tensor: 
        return sample - sample[:, -1].mean(dim=0)

    @staticmethod
    def direction(sample: Tensor) -> Tensor:
        first = sample[0, -1]
        last = sample[-1, -1]

        direction = last - first
        normalised_direction = direction / torch.norm(direction)
        return normalised_direction


class AddRandomNoise:

    def __init__(self, std: float) -> None:
        self.std = std

    def __call__(self, sample: Tensor) -> Tensor:
        noise = self.std * torch.randn_like(sample)
        return noise + sample


class DropRandomUniform:
    # Set a random fraction of elements along the first dimension to NaN

    def __init__(self, ratio: int) -> None:
        self.ratio = ratio / 100

    def __call__(self, sample: Tensor) -> Tensor:
        random_mask = torch.rand(sample.size(0)) < self.ratio
        sample[random_mask] = float('nan')
        return sample


class DropRandomChunk:

    def __init__(self, chunk_size: int) -> None:
        self.chunk_size = chunk_size

    def get_chunk_size(self) -> int:
        return self.chunk_size

    def __call__(self, sample: Tensor) -> Tensor:
        chunk_size = self.get_chunk_size()
        start_index = random.randint(1, sample.size(0) - chunk_size - 2)
        sample[start_index:start_index + chunk_size] = float('nan')
        return sample


class DropRandomChunkVariableSize(DropRandomChunk):
    def get_chunk_size(self) -> int:
        return random.randint(1, self.chunk_size)


class HumanPoseDataset(Dataset):
    __included_joint_indices__ = [0, 3, 6, 7, 9, 12, 13, 14, 15, 16, 19, 22, 23, 25, 28]

    def __init__(self,
                 path: str,
                 n_timesteps: int,
                 transform: Optional[Union[Callable, List[Callable]]] = []) -> None:
        self.path = path
        self.n_timesteps = n_timesteps
        self.transform = transform

        with h5py.File(path, "r") as f:
            ds = f['data']
            self._data = np.empty(ds.shape, dtype=ds.dtype)
            ds.read_direct(self._data)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        raw = self._data[index]
        assert len(raw) >= self.n_timesteps

        sample = torch.from_numpy(raw[:self.n_timesteps+1])

        # Only keep main joints
        sample = sample[:, self.__included_joint_indices__]
        target = sample.clone()

        if callable(self.transform):
            sample = self.transform(sample)
        else:
            for tsfrm in self.transform:
                sample = tsfrm(sample)

        return sample, target

    def __len__(self) -> int:
        return len(self._data)
