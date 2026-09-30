import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.signal import savgol_filter
from models import SolarFlowModel
from preprocessing import (REGION_KEYS, load_stats, normalize_stokes,
                           region_bounds, region_ids)

CHECKPOINT_PATH = './checkpoints_region_norm/multimodal_ep50.pth'
STATS_FILE = 'normalization_stats.npz'

DATA_MASTER = './dataset/multimodal_stokes_testing.h5' 
DATA_MODELS = './dataset/database_models/models_testing.h5'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_SAMPLES = 50 

USE_630 = True
USE_CAII = True
USE_CAII_K = True

# Cosmetic smoothing only. Off by default: it narrows the spread of the sampled
# trajectories and therefore misrepresents the posterior.
SMOOTH_OUTPUT = False

CONFIG = {
    'context_dim': 64,
    'physical_dim': 6, 
    'depth_points': 80
}

def get_multimodal_sample(idx, stats):
    with h5py.File(DATA_MASTER, 'r') as f_master, h5py.File(DATA_MODELS, 'r') as fm:
        stokes_concat = np.concatenate([f_master[k][idx] for k in REGION_KEYS], axis=1)
        full_model = fm['model'][idx, :, :]

    logtau = full_model[:, 0]
    physical = full_model[:, [1, 3, 4, 5, 6, 7]]

    lengths = stats['region_lengths']
    return stokes_concat, physical, logtau, region_ids(lengths), lengths

@torch.no_grad()
def generate_multiple_atmospheres(model, stokes_input, ids, lengths, stats, device, steps=50):
    model.eval()
    keep = [0, 2, 3, 4, 5, 6]
    m_mean = stats['models_mean'][keep].reshape(1, 6)
    m_std = stats['models_std'][keep].reshape(1, 6)

    stokes_norm = normalize_stokes(stokes_input, stats)
    stokes_tensor = torch.from_numpy(stokes_norm).float().to(device).unsqueeze(0)
    region_tensor = torch.from_numpy(ids).to(device).unsqueeze(0)

    bounds = region_bounds(lengths)
    mask = torch.zeros((1, bounds[-1][1]), dtype=torch.bool).to(device)
    for use, (i0, i1) in zip([USE_630, USE_CAII, USE_CAII_K], bounds):
        if not use:
            mask[0, i0:i1] = True

    context_single = model.encoder(stokes_tensor, region_tensor, padding_mask=mask)
    context_batch = context_single.repeat(NUM_SAMPLES, 1)
    
    x_t = torch.randn(NUM_SAMPLES, 80, 6).to(device)
    dt = 1.0 / steps
    
    for i in range(steps):
        t_tensor = torch.full((NUM_SAMPLES, 1), i / steps, device=device)
        velocity = model.vector_field(t_tensor, x_t, context_batch)
        x_t = x_t + velocity * dt
        
    x_final = x_t.cpu().numpy()
    
    if SMOOTH_OUTPUT:
        for n in range(NUM_SAMPLES):
            for i in range(6):
                x_final[n, :, i] = savgol_filter(x_final[n, :, i], window_length=15, polyorder=3)

    return x_final * m_std + m_mean

def plot_results(real_stokes, real_phys, preds_phys, logtau, sample_idx):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    modes = f"630:{USE_630} | CaII:{USE_CAII} | CaII_K:{USE_CAII_K}"
    fig.suptitle(f'Multimodal inference (Sample ID: {sample_idx}) - {NUM_SAMPLES} trajectories\n{modes}', fontsize=16)
    
    stokes_labels = ['Stokes I', 'Stokes Q', 'Stokes U', 'Stokes V']
    for i in range(4):
        axes[0, i].plot(real_stokes[i], 'k-', linewidth=1.5)
        axes[0, i].set_title(stokes_labels[i])
        axes[0, i].grid(True, alpha=0.3)
        
    phys_labels = ['Temperature (T)', 'Velocity (v)', 'Bx', 'Bz']
    plot_indices = [0, 2, 3, 5] 
    
    preds_mean = preds_phys.mean(axis=0)
    
    for i, p_idx in enumerate(plot_indices):
        ax = axes[1, i]
        
        for n in range(NUM_SAMPLES):
            ax.plot(logtau, preds_phys[n, :, p_idx], 'r-', alpha=0.15)
            
        ax.plot(logtau, preds_mean[:, p_idx], 'r--', linewidth=2, label='AI Mean')
        ax.plot(logtau, real_phys[:, p_idx], 'k-', linewidth=2, label='Real')
        
        ax.set_title(phys_labels[i])
        ax.set_xlabel('log(tau)')
        ax.grid(True, alpha=0.3)
        
        p_min = np.percentile(preds_phys[:, :, p_idx], 2)
        p_max = np.percentile(preds_phys[:, :, p_idx], 98)
        
        r_min = real_phys[:, p_idx].min()
        r_max = real_phys[:, p_idx].max()

        y_min = min(p_min, r_min)
        y_max = max(p_max, r_max)
        
        rango = y_max - y_min
        if rango == 0: rango = 1.0
        
        ax.set_ylim(y_min - (rango * 0.1), y_max + (rango * 0.1))

        ax.set_xticks([1, 0, -1, -2, -3, -4, -5, -6, -7])
        if not ax.xaxis_inverted():
            ax.invert_xaxis()
        
        if i == 0: ax.legend()
        
    plt.tight_layout()
    plt.show()

def main():
    try:
        model = SolarFlowModel(CONFIG).to(DEVICE)
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
    except FileNotFoundError:
        print(f"Model not found at {CHECKPOINT_PATH}")
        return

    idx = 550
    stats = load_stats(STATS_FILE)
    
    stokes, real_phys, logtau, ids, lengths = get_multimodal_sample(idx, stats)
    
    preds_phys = generate_multiple_atmospheres(model, stokes, ids, lengths, stats, DEVICE)
    
    plot_results(stokes, real_phys, preds_phys, logtau, idx)

if __name__ == "__main__":
    main()
