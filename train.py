import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np
import os
from tqdm import tqdm

from dataset import MultimodalSolarDataset
from models import SolarFlowModel
from preprocessing import DEFAULT_NOISE_SIGMA

CONFIG = {
    'batch_size': 128,
    'lr': 5e-4,
    'epochs': 50,
    'context_dim': 64,
    'physical_dim': 6,
    'depth_points': 80,
    'noise_sigma': DEFAULT_NOISE_SIGMA,
    'seed': 0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': './checkpoints_region_norm'
}

def create_modality_dropout_mask(batch_lengths, dropout_prob=0.3):
    B = batch_lengths.shape[0]
    total_L = batch_lengths[0].sum().item()
    mask = torch.zeros((B, total_L), dtype=torch.bool)
    
    L0, L1, L2 = batch_lengths[0].tolist()
    
    for i in range(B):
        drop = torch.rand(3) < dropout_prob
        
        if drop.all():
            drop[torch.randint(0, 3, (1,))] = False
            
        if drop[0]: mask[i, 0 : L0] = True
        if drop[1]: mask[i, L0 : L0+L1] = True
        if drop[2]: mask[i, L0+L1 : total_L] = True
            
    return mask

def train():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    print(f"--- Starting Multimodal Training on {CONFIG['device']} ---")
    
    dataset = MultimodalSolarDataset(
        master_h5_file='./dataset/multimodal_stokes_training.h5',
        models_h5_file='./dataset/database_models/models_training.h5',
        good_profiles_file='./dataset/database_630/good_profiles_training.npy',
        stats_file='normalization_stats.npz',
        noise_sigma=CONFIG['noise_sigma']
    )
    
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True,
                        num_workers=8, pin_memory=True, persistent_workers=True,
                        prefetch_factor=4)
    
    model = SolarFlowModel(CONFIG).to(CONFIG['device'])
    optimizer = AdamW(model.parameters(), lr=CONFIG['lr'])
    
    print(f"Model created. Trainable parameters: {sum(p.numel() for p in model.parameters())}")
    
    loss_history = []
    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for stokes, atmospheres, region_ids, lengths in pbar:
            stokes = stokes.to(CONFIG['device'])
            x_1 = atmospheres.to(CONFIG['device'])
            region_ids = region_ids.to(CONFIG['device'])
            
            batch_size = x_1.shape[0]
            
            padding_mask = create_modality_dropout_mask(lengths, dropout_prob=0.3).to(CONFIG['device'])
            
            t = torch.rand(batch_size, 1, 1, device=CONFIG['device'])
            
            x_0 = torch.randn_like(x_1)
            
            x_t = t * x_1 + (1 - t) * x_0
            target_v = x_1 - x_0
            
            predicted_v = model(t.view(batch_size, 1), x_t, stokes, region_ids, padding_mask)
            
            loss = torch.mean((predicted_v - target_v)**2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} Final Loss: {avg_loss:.6f}")
        np.save(f"{CONFIG['save_dir']}/loss_history.npy", np.array(loss_history))
        
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{CONFIG['save_dir']}/multimodal_ep{epoch+1}.pth")

if __name__ == "__main__":
    train()