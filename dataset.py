import random
from typing import Tuple, List, Callable, Union, Optional

import torch
import h5py
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset


class AddGeometricInvariantFeatures:
    
    def __call__(self, sample: Tensor) -> Tensor:
        velocity = sample[1:] - sample[:-1]

        velocity_magnitude = torch.norm(velocity[1:] + velocity[:-1], dim=2, keepdim=True) / 2
        omega = torch.sum(velocity[1:] * velocity[:-1], dim=2, keepdim=True)

        return torch.cat((sample[1:-1], velocity_magnitude, omega), dim=2)


class SetOriginToJoint:

    def __init__(self, joint: int) -> None:
        self.joint = joint

    def __call__(self, sample: Tensor) -> Tensor:
        data_without_joint = torch.cat((sample[:, :self.joint], sample[:, self.joint+1:]),
                                       dim=1)
        joint = sample[:, self.joint]
        return data_without_joint - joint.unsqueeze(1)


class FilterJoints:

    def __init__(self, included: List[int]) -> None:
        self.included = included
    
    def __call__(self, sample: Tensor) -> Tensor:
        return sample[:, self.included]


class SetOriginToCenterOfMassAndAlignXY:

    def __call__(self, sample: Tensor) -> Tensor:
        # Rotation matrix
        direction = self.direction(sample)
        sin = direction[1]
        cos = direction[0]
        rot = torch.tensor([[sin + cos, cos - sin, 0],
                            [sin - cos, cos + sin, 0],
                            [0, 0, 1]])

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
                 preprocessing: Optional[Union[Callable, List[Callable]]] = [],
                 transform: Optional[Union[Callable, List[Callable]]] = []) -> None:
        self.path = path
        self.preprocessing = preprocessing
        self.transform = transform

        with h5py.File(path, "r") as f:
            ds = f['data']
            self._data = np.empty(ds.shape, dtype=ds.dtype)
            ds.read_direct(self._data)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        raw = torch.from_numpy(self._data[index])

        if callable(self.preprocessing):
            clean = self.preprocessing(raw)
        else:
            for p in self.preprocessing:
                raw = p(raw)
            clean = raw
        target = clean.clone()

        if callable(self.transform):
            sample = self.transform(clean)
        else:
            for t in self.transform:
                clean = t(clean)
            sample = clean

        return sample, target

    def __len__(self) -> int:
        return len(self._data)
