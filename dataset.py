import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

from preprocessing import (REGION_KEYS, DEFAULT_NOISE_SIGMA, load_stats,
                           normalize_stokes, region_ids)


class MultimodalSolarDataset(Dataset):
    def __init__(self, master_h5_file, models_h5_file, good_profiles_file,
                 stats_file='normalization_stats.npz',
                 noise_sigma=DEFAULT_NOISE_SIGMA):
        self.master_path = master_h5_file
        self.models_path = models_h5_file
        self.noise_sigma = noise_sigma

        self.indices = np.load(good_profiles_file)
        self.indices = np.sort(self.indices)

        self.physical_indices = [1, 3, 4, 5, 6, 7]

        self.stats = load_stats(stats_file) if stats_file else None
        valid_stats_idx = [0, 2, 3, 4, 5, 6]
        if self.stats is not None:
            self.m_mean = torch.from_numpy(
                self.stats['models_mean'][valid_stats_idx]).float()
            self.m_std = torch.from_numpy(
                self.stats['models_std'][valid_stats_idx]).float()

        with h5py.File(self.master_path, 'r') as f:
            self.region_lengths = [f[k].shape[2] for k in REGION_KEYS]
        self.region_tensor = torch.from_numpy(region_ids(self.region_lengths))

        # HDF5 handles are not picklable, so each DataLoader worker opens its own
        # on first read and reuses them.
        self._master = None
        self._models = None

    def _open(self):
        if self._master is None:
            self._master = h5py.File(self.master_path, 'r', swmr=True, libver='latest')
            self._models = h5py.File(self.models_path, 'r', swmr=True, libver='latest')
        return self._master, self._models

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        f_master, fm = self._open()

        stokes_concat = np.concatenate(
            [f_master[k][real_idx] for k in REGION_KEYS], axis=1)

        full_model = torch.from_numpy(fm['model'][real_idx, :, :]).float()
        params = full_model[:, self.physical_indices]

        if self.stats is not None:
            stokes_concat = normalize_stokes(stokes_concat, self.stats,
                                             noise_sigma=self.noise_sigma)
            params_tensor = (params - self.m_mean) / (self.m_std + 1e-6)
        else:
            params_tensor = params

        stokes_tensor = torch.from_numpy(np.ascontiguousarray(stokes_concat)).float()
        lengths = torch.tensor(self.region_lengths)

        return stokes_tensor, params_tensor, self.region_tensor, lengths
