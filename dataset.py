import os
import random

import torch
import h5py
import numpy as np


class HumanPoseDataset(torch.utils.data.Dataset):

    def __init__(self, path, drop_type='chunk15', device=None):
        if not os.path.isfile(path):
            raise

        self.path = path
        self.device = device
        self.drop_type = drop_type

        with h5py.File(path, "r") as f:
            ds = f['data']
            self._data_array = np.empty(ds.shape, dtype=np.float32)
            ds.read_direct(self._data_array)

    def preprocessing(self, data):
        # Subtract mid-hip
        data = data - data[..., 13, np.newaxis, :]

        # Remove joints
        removed_indexes = [1, 2, 4, 5, 8, 10, 11, 13, 17, 18, 20, 21, 24, 26, 27]
        data = np.delete(data, removed_indexes, axis=1)

        return data

    def __getitem__(self, idx):
        clean = self.preprocessing(self._data_array[idx])

        # data = torch.tensor(clean, dtype=torch.float, device=self.device)
        data = torch.tensor(clean, device=self.device)
        target = data.clone()

        data = self.drop(data)

        return data, target

    def __len__(self):
        return len(self._data_array) * 2

    @staticmethod
    def drop_chunk(data, gap_size):
        sequence_length = data.size(dim=0)

        assert sequence_length - gap_size - 2 >= 1

        start_index = random.randint(1, sequence_length - gap_size - 2)
        data[start_index:start_index + gap_size] = float('nan')
        return data

    @staticmethod
    def drop_random(data, ratio):
        # Set a random fraction of elements along the first dimension to NaN
        random_mask = (torch.rand(data.size(0)) < ratio).unsqueeze(1)

        # Broadcast the mask along the second dimension
        random_mask = random_mask.expand_as(data)

        data[random_mask] = float('nan')
        return data

    def drop(self, data):
        if self.drop_type.startswith("chunk"):
            gap_size = int(self.drop_type[5:])
            return self.drop_chunk(data, gap_size=gap_size)
        elif self.drop_type.startswith("random"):
            ratio = float(self.drop[6:]) / 100
            return self.drop_random(data, ratio=ratio)
