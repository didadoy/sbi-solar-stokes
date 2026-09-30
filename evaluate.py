import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from models import SolarFlowModel
from preprocessing import (REGION_KEYS, load_stats, normalize_stokes,
                           region_bounds, region_ids)
from tqdm import tqdm

CHECKPOINT_PATH = './checkpoints_region_norm/multimodal_ep50.pth'
STATS_FILE = 'normalization_stats.npz'
DATA_MASTER = './dataset/multimodal_stokes_testing.h5'
DATA_MODELS = './dataset/database_models/models_testing.h5'
GOOD_PROFILES = './dataset/database_630/good_profiles_testing.npy'
PLOTS_DIR = './plots'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_EVAL_SAMPLES = 500 
NUM_ENSEMBLES = 50
PLOT_SAMPLE = 0
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
    stats = load_stats(STATS_FILE)
    keep_indices = [0, 2, 3, 4, 5, 6]
    m_mean = stats['models_mean'][keep_indices].reshape(1, 1, 6)
    m_std = stats['models_std'][keep_indices].reshape(1, 1, 6)
    return stats, m_mean, m_std

def get_eval_batch(num_samples, stats):
    # Only valid profiles are evaluated, the same ones the model sees during
    # training. Taking the first num_samples rows of the file instead pulls in
    # 37% discarded profiles and severely degrades the metrics.
    good = np.sort(np.load(GOOD_PROFILES))[:num_samples]

    with h5py.File(DATA_MASTER, 'r') as f_master, h5py.File(DATA_MODELS, 'r') as fm:
        stokes_raw = np.concatenate([f_master[k][good] for k in REGION_KEYS], axis=2)
        full_model = fm['model'][good, :, :]

    logtau = full_model[0, :, 0]
    physical = full_model[:, :, [1, 3, 4, 5, 6, 7]]

    lengths = stats['region_lengths']
    region_batch = np.tile(region_ids(lengths), (len(good), 1))
    stokes_norm = normalize_stokes(stokes_raw, stats)

    return stokes_raw, stokes_norm, physical, logtau, region_batch, lengths

def chromospheric_mask(batch_size, lengths):
    """Mask switching off Ca II 8542 and Ca II K, leaving only Fe I 630 nm."""
    bounds = region_bounds(lengths)
    total = bounds[-1][1]
    mask = torch.zeros((batch_size, total), dtype=torch.bool)
    for i0, i1 in bounds[1:]:
        mask[:, i0:i1] = True
    return mask.to(DEVICE)

@torch.no_grad()
def run_evaluation(model, stokes_norm, region_batch, m_mean, m_std, steps=30, num_ensembles=50):
    model.eval()
    
    stokes_t = torch.from_numpy(stokes_norm).float().to(DEVICE)
    region_t = torch.from_numpy(region_batch).to(DEVICE)
    
    B, L = region_batch.shape
    mask = torch.zeros((B, L), dtype=torch.bool).to(DEVICE)
    
    context = model.encoder(stokes_t, region_t, padding_mask=mask)
    
    ensemble_preds = np.zeros((num_ensembles, B, 80, 6), dtype=np.float32)
    
    for n in tqdm(range(num_ensembles), desc="Ensemble Sampling"):
        x_t = torch.randn(B, 80, 6).to(DEVICE)
        dt = 1.0 / steps
        
        for i in range(steps):
            t_tensor = torch.full((B, 1), i / steps, device=DEVICE)
            velocity = model.vector_field(t_tensor, x_t, context)
            x_t = x_t + velocity * dt
            
        ensemble_preds[n] = x_t.cpu().numpy()
        
    preds_mean_norm = np.mean(ensemble_preds, axis=0)
    preds_denorm = preds_mean_norm * m_std + m_mean
    
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
    report += f"Total samples: {NUM_EVAL_SAMPLES} (valid profiles only)\n"
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
    print(report)

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
        
        ax.set_xticks([1, 0, -1, -2, -3, -4, -5, -6, -7])
        if not ax.xaxis_inverted():
            ax.invert_xaxis()
            
        if i == 0: ax.legend()
        
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/error_vs_depth.png", dpi=300)
    plt.close()

@torch.no_grad()
def sample_trajectories(model, context, num_trajectories, m_mean, m_std, steps=30, desc=""):
    trajectories = []
    for _ in tqdm(range(num_trajectories), desc=desc):
        x_t = torch.randn(1, 80, 6).to(DEVICE)
        dt = 1.0 / steps
        for i in range(steps):
            t_tensor = torch.full((1, 1), i / steps, device=DEVICE)
            velocity = model.vector_field(t_tensor, x_t, context)
            x_t = x_t + velocity * dt
        trajectories.append(x_t.cpu().numpy()[0] * m_std[0, 0, :] + m_mean[0, 0, :])
    return np.array(trajectories)

def single_sample_context(model, stokes_norm, region_batch, sample, mask):
    stokes_t = torch.from_numpy(stokes_norm[sample:sample + 1]).float().to(DEVICE)
    region_t = torch.from_numpy(region_batch[sample:sample + 1]).to(DEVICE)
    return model.encoder(stokes_t, region_t, padding_mask=mask)

@torch.no_grad()
def generate_dropout_figure(model, stokes_norm, region_batch, reals, logtau, m_mean, m_std,
                            lengths, sample=PLOT_SAMPLE, num_trajectories=50):
    model.eval()
    real_atmos = reals[sample]

    mask = chromospheric_mask(1, lengths)
    context = single_sample_context(model, stokes_norm, region_batch, sample, mask)

    trajectories = sample_trajectories(model, context, num_trajectories, m_mean, m_std,
                                       desc="Sampling Trajectories")
    mean_pred = np.mean(trajectories, axis=0)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Inference under Modality Dropout (Only Fe I 630 nm) - Sample ID {sample}, '
                 f'{num_trajectories} trajectories', fontsize=16)
    
    for i, ax in enumerate(axes.flatten()):
        for j in range(num_trajectories):
            ax.plot(logtau, trajectories[j, :, i], 'r-', alpha=0.1)
            
        ax.plot(logtau, mean_pred[:, i], 'r--', linewidth=2, label='AI Mean')
        ax.plot(logtau, real_atmos[:, i], 'k-', linewidth=2, label='Real')
        
        ax.set_title(PHYSICAL_LABELS[i])
        ax.set_xlabel('log(tau)')
        ax.grid(True, alpha=0.3)
        
        ax.set_xticks([1, 0, -1, -2, -3, -4, -5, -6, -7])
        if not ax.xaxis_inverted():
            ax.invert_xaxis()
            
        if i == 0: ax.legend()
        
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/multimodal_inference_dropout_{sample}.png", dpi=300)
    plt.close()

@torch.no_grad()
def generate_percentile_uncertainty_figure(model, stokes_norm, region_batch, reals, logtau,
                                           m_mean, m_std, sample=PLOT_SAMPLE,
                                           num_trajectories=100):
    """Uncertainty quantification plot based on 1- and 2-sigma percentile bands."""
    model.eval()
    real_atmos = reals[sample]

    mask = torch.zeros((1, region_batch.shape[1]), dtype=torch.bool).to(DEVICE)
    context = single_sample_context(model, stokes_norm, region_batch, sample, mask)

    trajectories = sample_trajectories(model, context, num_trajectories, m_mean, m_std,
                                       desc="Sampling Percentile Trajectories")
    
    p2_5 = np.percentile(trajectories, 2.5, axis=0)   # 50 - 95/2 (Limite inferior 2-sigma)
    p16 = np.percentile(trajectories, 16.0, axis=0)   # 50 - 68/2 (Limite inferior 1-sigma)
    p50 = np.percentile(trajectories, 50.0, axis=0)   # Mediana (Percentil 50)
    p84 = np.percentile(trajectories, 84.0, axis=0)   # 50 + 68/2 (Limite superior 1-sigma)
    p97_5 = np.percentile(trajectories, 97.5, axis=0) # 50 + 95/2 (Limite superior 2-sigma)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Bayesian Uncertainty Quantification via Stratified Atmospheric Percentiles '
                 f'(Sample ID {sample})', fontsize=16)
    
    for i, ax in enumerate(axes.flatten()):
        ax.fill_between(logtau, p2_5[:, i], p97_5[:, i], color='red', alpha=0.15, label=r'$2\sigma$ Interval (95\%)')
        
        ax.fill_between(logtau, p16[:, i], p84[:, i], color='red', alpha=0.35, label=r'$1\sigma$ Interval (68\%)')
        
        ax.plot(logtau, p50[:, i], 'r--', linewidth=2, label='AI Median ($\mu_{p50}$)')
        
        ax.plot(logtau, real_atmos[:, i], 'k-', linewidth=2, label='Ground Truth')
        
        ax.set_title(PHYSICAL_LABELS[i])
        ax.set_xlabel(r'$\log(\tau)$')
        ax.grid(True, alpha=0.3)
        
        ax.set_xticks([1, 0, -1, -2, -3, -4, -5, -6, -7])
        if not ax.xaxis_inverted():
            ax.invert_xaxis()
            
        if i == 0: ax.legend(fontsize=10)
        
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/multimodal_inference_percentiles_{sample}.png", dpi=300)
    plt.close()

def main():
    try:
        model = SolarFlowModel(CONFIG).to(DEVICE)
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
    except FileNotFoundError:
        print(f"Checkpoint not found at {CHECKPOINT_PATH}")
        return

    stats, m_mean, m_std = setup()
    _, stokes_norm, reals, logtau, region_ids_batch, lengths = get_eval_batch(
        NUM_EVAL_SAMPLES, stats)
    
    preds = run_evaluation(model, stokes_norm, region_ids_batch, m_mean, m_std,
                           steps=30, num_ensembles=NUM_ENSEMBLES)
    calculate_and_save_metrics(reals, preds, logtau)
    plot_scatter_density(reals, preds, logtau)
    plot_error_vs_depth(reals, preds, logtau)
    
    generate_dropout_figure(model, stokes_norm, region_ids_batch, reals, logtau,
                            m_mean, m_std, lengths)
    generate_percentile_uncertainty_figure(model, stokes_norm, region_ids_batch, reals,
                                           logtau, m_mean, m_std)
    
if __name__ == "__main__":
    main()
