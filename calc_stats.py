import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

from preprocessing import REGION_KEYS, continuum_level

DATA_DIR = Path('./dataset')

FILE_STOKES_MERGED = DATA_DIR / 'multimodal_stokes_training.h5'
FILE_MODELS = DATA_DIR / 'database_models/models_training.h5'
FILE_GOOD_PROFILES = DATA_DIR / 'database_630/good_profiles_training.npy'

OUTPUT_STATS = Path('normalization_stats.npz')
BATCH_SIZE = 10_000
CONTINUUM_SAMPLES = 20_000


def compute_continuum(f_stokes, good_indices):
    """Factor bringing each region into the continuum units of Fe I 630 nm.

    The absolute median continuum depends on which subset of atmospheres is
    measured, since the intensity varies widely across samples. The ratio
    between regions is stable, so the scale is anchored on 630 nm, which is
    delivered already in continuum units and is the region the 1e-3 instrumental
    noise level was defined against.
    """
    subset = good_indices[np.linspace(
        0, len(good_indices) - 1,
        min(CONTINUUM_SAMPLES, len(good_indices))).astype(int)]
    subset = np.unique(subset)

    continuum = np.zeros(len(REGION_KEYS), dtype=np.float64)
    for r, key in enumerate(REGION_KEYS):
        continuum[r] = continuum_level(f_stokes[key][subset, 0, :])

    continuum /= continuum[0]
    for r, key in enumerate(REGION_KEYS):
        print(f"  {key:14s} Ic / Ic_630 = {continuum[r]:.6g}")
    return continuum



def compute_global_stats():
    print("Loading valid profile indices...")
    good_indices = np.sort(np.load(FILE_GOOD_PROFILES))
    total_samples = len(good_indices)
    print(f"Total available samples: {total_samples}")

    n_regions = len(REGION_KEYS)
    stokes_sum = np.zeros((n_regions, 4), dtype=np.float64)
    stokes_sq_sum = np.zeros((n_regions, 4), dtype=np.float64)
    stokes_points = np.zeros(n_regions, dtype=np.int64)

    params_sum = np.zeros(7, dtype=np.float64)
    params_sq_sum = np.zeros(7, dtype=np.float64)
    total_model_points = 0

    with h5py.File(FILE_STOKES_MERGED, 'r') as f_stokes, h5py.File(FILE_MODELS, 'r') as f_models:
        models_dset = f_models['model']
        region_lengths = np.array([f_stokes[k].shape[2] for k in REGION_KEYS],
                                  dtype=np.int64)

        print("Estimating the continuum level of each spectral region...")
        continuum = compute_continuum(f_stokes, good_indices)

        print("Computing Stokes statistics (per region, continuum-normalized)...")
        for i in tqdm(range(0, total_samples, BATCH_SIZE)):
            idx_batch = good_indices[i: i + BATCH_SIZE]

            for r, key in enumerate(REGION_KEYS):
                s_batch = f_stokes[key][idx_batch, :, :] / continuum[r]
                s_flat = np.transpose(s_batch, (1, 0, 2)).reshape(4, -1)

                stokes_sum[r] += np.sum(s_flat, axis=1, dtype=np.float64)
                stokes_sq_sum[r] += np.sum(s_flat ** 2, axis=1, dtype=np.float64)
                stokes_points[r] += s_flat.shape[1]

        stokes_mean = stokes_sum / stokes_points[:, None]
        stokes_std = np.sqrt((stokes_sq_sum / stokes_points[:, None]) - stokes_mean ** 2)

        print("Computing atmospheric model statistics...")
        for i in tqdm(range(0, total_samples, BATCH_SIZE)):
            idx_batch = good_indices[i: i + BATCH_SIZE]

            m_batch = models_dset[idx_batch, :, :]
            m_phys = m_batch[:, :, 1:].reshape(-1, 7)

            params_sum += np.sum(m_phys, axis=0, dtype=np.float64)
            params_sq_sum += np.sum(m_phys ** 2, axis=0, dtype=np.float64)
            total_model_points += m_phys.shape[0]

        models_mean = params_sum / total_model_points
        models_std = np.sqrt((params_sq_sum / total_model_points) - models_mean ** 2)

    for r, key in enumerate(REGION_KEYS):
        print(f"{key:14s} mean: {np.array2string(stokes_mean[r], precision=4)}")
        print(f"{'':14s} std:  {np.array2string(stokes_std[r], precision=4)}")
    print(f"Models Mean: {models_mean}")
    print(f"Models Std:  {models_std}")

    np.savez(
        OUTPUT_STATS,
        continuum=continuum,
        region_lengths=region_lengths,
        stokes_mean=stokes_mean,
        stokes_std=stokes_std,
        models_mean=models_mean,
        models_std=models_std
    )
    print(f"Statistics successfully saved to {OUTPUT_STATS}")


if __name__ == "__main__":
    compute_global_stats()
