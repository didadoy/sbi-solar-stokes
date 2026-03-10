import h5py
import numpy as np
from astropy.io import fits
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path('./dataset')
FILE_630 = DATA_DIR / 'database_630/stokes_testing.h5'
FILE_CAII = DATA_DIR / 'database_caii/synthetic_CaII_8542_testing.fits'
FILE_CAII_K = DATA_DIR / 'database_caii_k/synthetic_CaII_K_testing.fits'

OUTPUT_FILE = DATA_DIR / 'multimodal_stokes_testing.h5'

def normalize_batch(data):
    """
    Asegura que el batch salga SIEMPRE con forma (Batch, 4, Longitud).
    """
    if data.ndim == 4 and data.shape[1] == 1:
        data = data[:, 0, :, :]
        
    if data.ndim == 3 and data.shape[2] == 4:
        data = np.transpose(data, (0, 2, 1))
        
    return data

def build_multimodal_dataset():
    hdul_caii = fits.open(FILE_CAII, memmap=True)
    data_caii = hdul_caii[0].data
    
    hdul_caii_k = fits.open(FILE_CAII_K, memmap=True)
    data_caii_k = hdul_caii_k[0].data

    with h5py.File(FILE_630, 'r') as f_630:
        data_630 = f_630['spec1']['stokes']
        N_samples = data_630.shape[0]
        
        L_630 = [dim for dim in data_630.shape[1:] if dim not in (1, 4)][0]
        L_caii = [dim for dim in data_caii.shape[1:] if dim not in (1, 4)][0]
        L_caii_k = [dim for dim in data_caii_k.shape[1:] if dim not in (1, 4)][0]

        with h5py.File(OUTPUT_FILE, 'w') as f_out:
            dset_630 = f_out.create_dataset('stokes_630', shape=(N_samples, 4, L_630), dtype='float32', chunks=(1024, 4, L_630))
            dset_caii = f_out.create_dataset('stokes_caii', shape=(N_samples, 4, L_caii), dtype='float32', chunks=(1024, 4, L_caii))
            dset_caii_k = f_out.create_dataset('stokes_caii_k', shape=(N_samples, 4, L_caii_k), dtype='float32', chunks=(1024, 4, L_caii_k))

            batch_size = 5000
            
            for i in tqdm(range(0, N_samples, batch_size)):
                end = min(i + batch_size, N_samples)
                
                dset_630[i:end] = normalize_batch(data_630[i:end])
                dset_caii[i:end] = normalize_batch(data_caii[i:end])
                dset_caii_k[i:end] = normalize_batch(data_caii_k[i:end])
                
    hdul_caii.close()
    hdul_caii_k.close()

if __name__ == "__main__":
    build_multimodal_dataset()