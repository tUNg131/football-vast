import random
from typing import Tuple, List, Callable, Optional

import torch
import h5py
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

DEFAULT_JOINT_INDICES = [0, 3, 6, 7, 9, 12, 14, 15, 16, 19, 22, 23, 25, 28]
MIDHIP_INDEX = 13


class DropRandomUniform:
    # Set a random fraction of elements along the first dimension to NaN

    def __init__(self, ratio: int) -> None:
        self.ratio = int(ratio) / 100

    def __call__(self, sample: Tensor) -> Tensor:
        random_mask = torch.rand(sample.size(0)) < self.ratio
        sample[random_mask] = float('nan')
        return sample


class DropRandomChunk:

    def __init__(self, chunk_size: int) -> None:
        self.chunk_size = int(chunk_size)

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
    def __init__(self,
                 path: str,
                 drop: Callable[[Tensor], Tensor],
                 noise: Optional[float] = 0.0,
                 joints: Optional[List[int]] = DEFAULT_JOINT_INDICES):
        
        self.path = path
        self.noise = noise
        self.joints = joints
        self.drop = drop

        with h5py.File(path, "r") as f:
            ds = f['data']
            self._data = np.empty(ds.shape, dtype=ds.dtype)
            ds.read_direct(self._data)

    def __len__(self) -> int:
        return len(self._data)
    

class HumanPoseMidHipDataset(HumanPoseDataset):

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        raw = torch.from_numpy(self._data[index])

        # Excluded joints & subtract midhip
        sample = raw[:, self.joints] - raw[:, MIDHIP_INDEX].unsqueeze(1)

        target = sample.clone()
        
        # Add noise to sample
        sample = sample + self.noise * torch.randn_like(sample)

        # Randomly masking
        sample = self.drop(sample)

        return sample, target


class HumanPoseMidHipDatasetWithGeometricInvariantFeatures(HumanPoseDataset):

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        full = torch.from_numpy(self._data[index])
        raw = full[:, self.joints]

        velocity = raw[1:] - raw[:-1]
        velocity_magnitude = torch.norm(
            (velocity[1:] + velocity[:-1]) / 2,
            dim=2, # (n_timesteps, n_joints, d_joint)
            keepdim=True
        )
        omega = (
            torch.sum(velocity[1:] * velocity[:-1], dim=2, keepdim=True)
            / torch.norm(velocity[1:], dim=2, keepdim=True)
            / torch.norm(velocity[:-1], dim=2, keepdim=True)
        )

        # Excluded joints & subtract midhip
        joints = (raw - full[:, MIDHIP_INDEX].unsqueeze(1))[1:-1]

        # Concat features
        sample = torch.cat(
            (joints, velocity_magnitude, omega),
            dim=2
        )

        target = joints.clone()
        
        # Add noise to sample
        sample = sample + self.noise * torch.randn_like(sample)

        # Randomly masking
        sample = self.drop(sample)

        return sample, target