import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path('./dataset')

FILE_STOKES_MERGED = DATA_DIR / 'multimodal_stokes_training.h5'
FILE_MODELS = DATA_DIR / 'database_models/models_training.h5'
FILE_GOOD_PROFILES = DATA_DIR / 'database_630/good_profiles_training.npy'

OUTPUT_STATS = Path('normalization_stats.npz')
BATCH_SIZE = 10_000

def compute_global_stats():
    print("Loading valid profile indices...")
    good_indices = np.sort(np.load(FILE_GOOD_PROFILES))
    total_samples = len(good_indices)
    print(f"Total available samples: {total_samples}")

    stokes_sum = np.zeros(4, dtype=np.float64)
    stokes_sq_sum = np.zeros(4, dtype=np.float64)
    total_stokes_pixels = 0

    params_sum = np.zeros(7, dtype=np.float64)
    params_sq_sum = np.zeros(7, dtype=np.float64)
    total_model_points = 0

    with h5py.File(FILE_STOKES_MERGED, 'r') as f_stokes, h5py.File(FILE_MODELS, 'r') as f_models:
        models_dset = f_models['model']

        print("Computing Stokes statistics (3 regiones unificadas)...")
        for i in tqdm(range(0, total_samples, BATCH_SIZE)):
            idx_batch = good_indices[i : i + BATCH_SIZE]
            
            s1 = f_stokes['stokes_630'][idx_batch, :, :]
            s2 = f_stokes['stokes_caii'][idx_batch, :, :]
            s3 = f_stokes['stokes_caii_k'][idx_batch, :, :]
            
            s_batch = np.concatenate([s1, s2, s3], axis=2)
            
            s_flat = np.transpose(s_batch, (1, 0, 2)).reshape(4, -1)
            
            stokes_sum += np.sum(s_flat, axis=1, dtype=np.float64)
            stokes_sq_sum += np.sum(s_flat**2, axis=1, dtype=np.float64)
            total_stokes_pixels += s_flat.shape[1]

        stokes_mean = stokes_sum / total_stokes_pixels
        stokes_std = np.sqrt((stokes_sq_sum / total_stokes_pixels) - (stokes_mean**2))

        print("Computing atmospheric model statistics...")
        for i in tqdm(range(0, total_samples, BATCH_SIZE)):
            idx_batch = good_indices[i : i + BATCH_SIZE]
            
            m_batch = models_dset[idx_batch, :, :]
            m_phys = m_batch[:, :, 1:].reshape(-1, 7)
            
            params_sum += np.sum(m_phys, axis=0, dtype=np.float64)
            params_sq_sum += np.sum(m_phys**2, axis=0, dtype=np.float64)
            total_model_points += m_phys.shape[0]

        models_mean = params_sum / total_model_points
        models_std = np.sqrt((params_sq_sum / total_model_points) - (models_mean**2))

    print(f"Stokes Mean: {stokes_mean}")
    print(f"Stokes Std:  {stokes_std}")
    print(f"Models Mean: {models_mean}")
    print(f"Models Std:  {models_std}")

    np.savez(
        OUTPUT_STATS, 
        stokes_mean=stokes_mean, 
        stokes_std=stokes_std,
        models_mean=models_mean, 
        models_std=models_std
    )
    print(f"Statistics successfully saved to {OUTPUT_STATS}")

if __name__ == "__main__":
    compute_global_stats()