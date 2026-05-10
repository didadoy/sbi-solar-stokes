import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.signal import savgol_filter
from models import SolarFlowModel

CHECKPOINT_PATH = './checkpoints_physical_noise/multimodal_ep50.pth'
STATS_FILE = 'normalization_stats.npz'

DATA_MASTER = './dataset/multimodal_stokes_testing.h5' 
DATA_MODELS = './dataset/database_models/models_testing.h5'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_SAMPLES = 50 

USE_630 = True
USE_CAII = True
USE_CAII_K = True

CONFIG = {
    'context_dim': 64,
    'physical_dim': 6, 
    'depth_points': 80
}

def load_stats():
    stats = np.load(STATS_FILE)
    keep_indices = [0, 2, 3, 4, 5, 6]
    
    s_mean = stats['stokes_mean'].reshape(4, 1)
    s_std = stats['stokes_std'].reshape(4, 1)
    
    m_mean = stats['models_mean'][keep_indices].reshape(1, 6)
    m_std = stats['models_std'][keep_indices].reshape(1, 6)
    
    return s_mean, s_std, m_mean, m_std

def get_multimodal_sample(idx):
    with h5py.File(DATA_MASTER, 'r') as f_master, h5py.File(DATA_MODELS, 'r') as fm:
        s_630 = f_master['stokes_630'][idx]
        s_caii = f_master['stokes_caii'][idx]
        s_caiik = f_master['stokes_caii_k'][idx]
        
        stokes_concat = np.concatenate([s_630, s_caii, s_caiik], axis=1)
        
        L0, L1, L2 = s_630.shape[1], s_caii.shape[1], s_caiik.shape[1]
        
        region_ids = np.concatenate([
            np.zeros(L0, dtype=np.int64),
            np.ones(L1, dtype=np.int64),
            np.full(L2, 2, dtype=np.int64)
        ])
        
        full_model = fm['model'][idx, :, :]
        logtau = full_model[:, 0]
        physical = full_model[:, [1, 3, 4, 5, 6, 7]] 
        
    return stokes_concat, physical, logtau, region_ids, (L0, L1, L2)

@torch.no_grad()
def generate_multiple_atmospheres(model, stokes_input, region_ids, lengths, device, steps=50):
    model.eval()
    s_mean, s_std, m_mean, m_std = load_stats()
    
    stokes_tensor = torch.from_numpy(stokes_input).float().to(device).unsqueeze(0)
    region_tensor = torch.from_numpy(region_ids).to(device).unsqueeze(0)
    
    s_mean_t = torch.from_numpy(s_mean).float().to(device)
    s_std_t = torch.from_numpy(s_std).float().to(device)
    stokes_norm = (stokes_tensor - s_mean_t) / (s_std_t + 1e-6)
    
    L0, L1, L2 = lengths
    mask = torch.zeros((1, L0 + L1 + L2), dtype=torch.bool).to(device)
    if not USE_630: mask[0, 0:L0] = True
    if not USE_CAII: mask[0, L0:L0+L1] = True
    if not USE_CAII_K: mask[0, L0+L1:] = True

    context_single = model.encoder(stokes_norm, region_tensor, padding_mask=mask)
    
    context_batch = context_single.repeat(NUM_SAMPLES, 1)
    
    x_t = torch.randn(NUM_SAMPLES, 80, 6).to(device)
    dt = 1.0 / steps
    
    for i in range(steps):
        t_tensor = torch.full((NUM_SAMPLES, 1), i / steps, device=device)
        velocity = model.vector_field(t_tensor, x_t, context_batch)
        x_t = x_t + velocity * dt
        
    x_final = x_t.cpu().numpy()
    
    try:
        for n in range(NUM_SAMPLES):
            for i in range(6):
                x_final[n, :, i] = savgol_filter(x_final[n, :, i], window_length=15, polyorder=3)
    except Exception:
        pass

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
        
    phys_labels = ['Temperatura (T)', 'Velocidad (v)', 'Bx', 'Bz']
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
        
        # --- NUEVA SOLUCIÓN: Percentiles ---
        # Cogemos el 96% central de las predicciones para ignorar picos locos
        p_min = np.percentile(preds_phys[:, :, p_idx], 2)
        p_max = np.percentile(preds_phys[:, :, p_idx], 98)
        
        # También miramos dónde está la realidad
        r_min = real_phys[:, p_idx].min()
        r_max = real_phys[:, p_idx].max()
        
        # El límite final abarca la realidad y el 96% de la IA
        y_min = min(p_min, r_min)
        y_max = max(p_max, r_max)
        
        rango = y_max - y_min
        if rango == 0: rango = 1.0
        
        # Le damos un pelín de aire (10%) por arriba y por abajo
        ax.set_ylim(y_min - (rango * 0.1), y_max + (rango * 0.1))
        # -----------------------------------

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
        print(f"No se encontró el modelo en {CHECKPOINT_PATH}")
        return

    idx = 550
    
    stokes, real_phys, logtau, region_ids, lengths = get_multimodal_sample(idx)
    
    preds_phys = generate_multiple_atmospheres(model, stokes, region_ids, lengths, DEVICE)
    
    plot_results(stokes, real_phys, preds_phys, logtau, idx)

if __name__ == "__main__":
    main()