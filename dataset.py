import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class MultimodalSolarDataset(Dataset):
    def __init__(self, master_h5_file, models_h5_file, good_profiles_file, stats_file=None):
        self.master_path = master_h5_file
        self.models_path = models_h5_file
        
        self.indices = np.load(good_profiles_file)
        self.indices = np.sort(self.indices)
        
        self.physical_indices = [1, 3, 4, 5, 6, 7]
        
        self.s_mean, self.s_std = None, None
        self.m_mean, self.m_std = None, None
        if stats_file:
            stats = np.load(stats_file)
            self.s_mean = torch.from_numpy(stats['stokes_mean']).float()
            self.s_std = torch.from_numpy(stats['stokes_std']).float()
            
            valid_stats_idx = [0, 2, 3, 4, 5, 6]
            
            self.m_mean = torch.from_numpy(stats['models_mean'][valid_stats_idx]).float()
            self.m_std = torch.from_numpy(stats['models_std'][valid_stats_idx]).float()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        with h5py.File(self.master_path, 'r', swmr=True, libver='latest') as f_master, \
             h5py.File(self.models_path, 'r', swmr=True, libver='latest') as fm:
            
            s_630 = f_master['stokes_630'][real_idx]
            s_caii = f_master['stokes_caii'][real_idx]
            s_caiik = f_master['stokes_caii_k'][real_idx]
            
            stokes_concat = np.concatenate([s_630, s_caii, s_caiik], axis=1)
            stokes_tensor = torch.from_numpy(stokes_concat).float()
            
            stokes_tensor = stokes_tensor + torch.randn_like(stokes_tensor) * 1e-3
            
            L_630 = s_630.shape[1]
            L_caii = s_caii.shape[1]
            L_caiik = s_caiik.shape[1]
            
            region_ids = np.concatenate([
                np.zeros(L_630, dtype=np.int64),
                np.ones(L_caii, dtype=np.int64),
                np.full(L_caiik, 2, dtype=np.int64)
            ])
            region_tensor = torch.from_numpy(region_ids)
            
            full_model = torch.from_numpy(fm['model'][real_idx, :, :]).float()
            params = full_model[:, self.physical_indices]
            
        if self.s_mean is not None:
            stokes_tensor = (stokes_tensor - self.s_mean.view(4, 1)) / (self.s_std.view(4, 1) + 1e-6)
            params_tensor = (params - self.m_mean) / (self.m_std + 1e-6)
        else:
            params_tensor = params

        lengths = torch.tensor([L_630, L_caii, L_caiik])

        return stokes_tensor, params_tensor, region_tensor, lengths