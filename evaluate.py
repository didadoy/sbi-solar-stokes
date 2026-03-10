import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from models import SolarFlowModel
from tqdm import tqdm

CHECKPOINT_PATH = './checkpoints_multimodal/multimodal_ep50.pth'
STATS_FILE = 'normalization_stats.npz'
DATA_MASTER = './dataset/multimodal_stokes_testing.h5'
DATA_MODELS = './dataset/database_models/models_testing.h5'
PLOTS_DIR = './plots'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_EVAL_SAMPLES = 500 
LOGTAU_MIN = -3.0
LOGTAU_MAX = 0.0

CONFIG = {
    'context_dim': 64,
    'physical_dim': 6, 
    'depth_points': 80
}

PHYSICAL_LABELS = ['Temperature (T)', 'V. Micro (vmic)', 'Velocity (v)', 'Bx', 'By', 'Bz']

def setup():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    stats = np.load(STATS_FILE)
    keep_indices = [0, 2, 3, 4, 5, 6]
    s_mean = stats['stokes_mean'].reshape(4, 1)
    s_std = stats['stokes_std'].reshape(4, 1)
    m_mean = stats['models_mean'][keep_indices].reshape(1, 1, 6)
    m_std = stats['models_std'][keep_indices].reshape(1, 1, 6)
    return s_mean, s_std, m_mean, m_std

def get_eval_batch(num_samples):
    with h5py.File(DATA_MASTER, 'r') as f_master, h5py.File(DATA_MODELS, 'r') as fm:
        s_630 = f_master['stokes_630'][:num_samples]
        s_caii = f_master['stokes_caii'][:num_samples]
        s_caiik = f_master['stokes_caii_k'][:num_samples]
        
        stokes_concat = np.concatenate([s_630, s_caii, s_caiik], axis=2)
        
        L0, L1, L2 = s_630.shape[2], s_caii.shape[2], s_caiik.shape[2]
        
        region_ids = np.concatenate([
            np.zeros(L0, dtype=np.int64),
            np.ones(L1, dtype=np.int64),
            np.full(L2, 2, dtype=np.int64)
        ])
        
        region_batch = np.tile(region_ids, (num_samples, 1))
        
        full_model = fm['model'][:num_samples, :, :]
        logtau = full_model[0, :, 0]
        physical = full_model[:, :, [1, 3, 4, 5, 6, 7]] 
        
    return stokes_concat, physical, logtau, region_batch, (L0, L1, L2)

@torch.no_grad()
def run_evaluation(model, stokes_batch, region_batch, s_mean, s_std, m_mean, m_std, steps=30):
    model.eval()
    
    stokes_t = torch.from_numpy(stokes_batch).float().to(DEVICE)
    region_t = torch.from_numpy(region_batch).to(DEVICE)
    s_mean_t = torch.from_numpy(s_mean).float().to(DEVICE).unsqueeze(0)
    s_std_t = torch.from_numpy(s_std).float().to(DEVICE).unsqueeze(0)
    
    stokes_norm = (stokes_t - s_mean_t) / (s_std_t + 1e-6)
    
    B, L = region_batch.shape
    mask = torch.zeros((B, L), dtype=torch.bool).to(DEVICE)
    
    context = model.encoder(stokes_norm, region_t, padding_mask=mask)
    x_t = torch.randn(B, 80, 6).to(DEVICE)
    dt = 1.0 / steps
    
    for i in tqdm(range(steps), desc="ODE Solving"):
        t_tensor = torch.full((B, 1), i / steps, device=DEVICE)
        velocity = model.vector_field(t_tensor, x_t, context)
        x_t = x_t + velocity * dt
        
    preds_norm = x_t.cpu().numpy()
    preds_denorm = preds_norm * m_std + m_mean
    
    return preds_denorm

def calculate_and_save_metrics(reals, preds, logtau):
    valid_indices = np.where((logtau >= LOGTAU_MIN) & (logtau <= LOGTAU_MAX))[0]
    
    reals_global_flat = reals.reshape(-1, 6)
    preds_global_flat = preds.reshape(-1, 6)
    
    reals_filtered = reals[:, valid_indices, :]
    preds_filtered = preds[:, valid_indices, :]
    reals_filt_flat = reals_filtered.reshape(-1, 6)
    preds_filt_flat = preds_filtered.reshape(-1, 6)
    
    report = f"=== MULTIMODAL EVALUATION REPORT ===\n"
    report += f"Total samples: {NUM_EVAL_SAMPLES}\n"
    report += f"Filtered region: log(tau) [{LOGTAU_MIN}, {LOGTAU_MAX}]\n\n"
    
    for i, label in enumerate(PHYSICAL_LABELS):
        rmse_g = np.sqrt(mean_squared_error(reals_global_flat[:, i], preds_global_flat[:, i]))
        mae_g = mean_absolute_error(reals_global_flat[:, i], preds_global_flat[:, i])
        r2_g = r2_score(reals_global_flat[:, i], preds_global_flat[:, i])
        
        rmse_f = np.sqrt(mean_squared_error(reals_filt_flat[:, i], preds_filt_flat[:, i]))
        mae_f = mean_absolute_error(reals_filt_flat[:, i], preds_filt_flat[:, i])
        r2_f = r2_score(reals_filt_flat[:, i], preds_filt_flat[:, i])
        
        report += f"[{label}]\n"
        report += f"  Global   -> RMSE: {rmse_g:.4f} | MAE: {mae_g:.4f} | R2: {r2_g:.4f}\n"
        report += f"  Filtered -> RMSE: {rmse_f:.4f} | MAE: {mae_f:.4f} | R2: {r2_f:.4f}\n\n"
        
    with open(f"{PLOTS_DIR}/metrics_report.txt", "w") as f:
        f.write(report)

def plot_scatter_density(reals, preds, logtau):
    valid_indices = np.where((logtau >= LOGTAU_MIN) & (logtau <= LOGTAU_MAX))[0]
    reals_filtered = reals[:, valid_indices, :]
    preds_filtered = preds[:, valid_indices, :]
    
    reals_flat = reals_filtered.reshape(-1, 6)
    preds_flat = preds_filtered.reshape(-1, 6)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Prediction Density vs Real (Filtered log(tau) [{LOGTAU_MIN}, {LOGTAU_MAX}])', fontsize=16)
    
    for i, ax in enumerate(axes.flatten()):
        h = ax.hist2d(reals_flat[:, i], preds_flat[:, i], bins=100, cmap='inferno', cmin=1)
        
        min_val = min(reals_flat[:, i].min(), preds_flat[:, i].min())
        max_val = max(reals_flat[:, i].max(), preds_flat[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'w--', alpha=0.8, label='Ideal')
        
        ax.set_title(PHYSICAL_LABELS[i])
        ax.set_xlabel('Real Value')
        ax.set_ylabel('AI Prediction')
        fig.colorbar(h[3], ax=ax)
        
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/scatter_density_filtered.png", dpi=300)
    plt.close()

def plot_error_vs_depth(reals, preds, logtau):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Mean Absolute Error (MAE) vs Atmospheric Depth log(tau)', fontsize=16)
    
    mae_per_depth = np.mean(np.abs(reals - preds), axis=0)
    
    for i, ax in enumerate(axes.flatten()):
        ax.plot(logtau, mae_per_depth[:, i], 'b-', linewidth=2)
        ax.axvspan(LOGTAU_MIN, LOGTAU_MAX, color='green', alpha=0.1, label='Eval Region')
        
        ax.set_title(f"MAE: {PHYSICAL_LABELS[i]}")
        ax.set_xlabel('log(tau)')
        ax.set_ylabel('Absolute Error')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        if i == 0: ax.legend()
        
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/error_vs_depth.png", dpi=300)
    plt.close()

def main():
    try:
        model = SolarFlowModel(CONFIG).to(DEVICE)
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
    except FileNotFoundError:
        return

    s_mean, s_std, m_mean, m_std = setup()
    stokes, reals, logtau, region_ids, lengths = get_eval_batch(NUM_EVAL_SAMPLES)
    preds = run_evaluation(model, stokes, region_ids, s_mean, s_std, m_mean, m_std, steps=30)
    
    calculate_and_save_metrics(reals, preds, logtau)
    plot_scatter_density(reals, preds, logtau)
    plot_error_vs_depth(reals, preds, logtau)

if __name__ == "__main__":
    main()