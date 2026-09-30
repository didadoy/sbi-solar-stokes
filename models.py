import torch
import torch.nn as nn


class MultimodalStokesEmbedding(nn.Module):
    def __init__(self, input_channels=4, max_seq_length=500, d_model=128, nhead=4, num_layers=3, output_dim=64, num_regions=3):
        super().__init__()
        
        self.input_proj = nn.Conv1d(input_channels, d_model, kernel_size=1)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_length, d_model))
        
        self.region_embedding = nn.Embedding(num_regions, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_head = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )
        self.output_dim = output_dim

    def forward(self, stokes, region_ids, padding_mask=None):
        """
        stokes: (Batch, 4, L) -> the Stokes profiles
        region_ids: (Batch, L) -> 0s, 1s and 2s marking the spectral region of each point
        padding_mask: (Batch, L) -> True means "ignore this point" (modality dropout)
        """
        B, _, L = stokes.shape
        
        x = self.input_proj(stokes)
        x = x.transpose(1, 2)
        
        x = x + self.pos_embedding[:, :L, :]
        
        r_emb = self.region_embedding(region_ids) 
        x = x + r_emb
        
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        if padding_mask is not None:
            valid_mask = (~padding_mask).unsqueeze(-1).float()
            x = x * valid_mask
            
            sum_x = x.sum(dim=1)
            valid_tokens = valid_mask.sum(dim=1).clamp(min=1.0)
            pooled = sum_x / valid_tokens
        else:
            pooled = x.mean(dim=1)
            
        context = self.output_head(pooled)
        return context

class VectorFieldNetwork(nn.Module):
    """Predicts the vector field v(x, t|cond) transporting Gaussian noise to atmospheres."""
    def __init__(self, physical_dim=6, depth_points=80, context_dim=128, hidden_dim=512):
        super().__init__()
        
        self.flat_dim = physical_dim * depth_points
        
        input_size = self.flat_dim + context_dim + 1 
        
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.flat_dim)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, t, x, context):
        """
        t: (Batch, 1) or float
        x: (Batch, 80, 6) -> current atmosphere
        context: (Batch, context_dim) -> Stokes embedding
        """
        x_flat = x.view(x.shape[0], -1)
        
        if isinstance(t, float) or t.ndim == 0:
            t = torch.full((x.shape[0], 1), t, device=x.device)
        elif t.ndim == 1:
            t = t.unsqueeze(1)
            
        inp = torch.cat([x_flat, t, context], dim=1)
        
        v_flat = self.net(inp)
        
        return v_flat.view_as(x)

class SolarFlowModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.encoder = MultimodalStokesEmbedding(
            input_channels=4,
            max_seq_length=500,
            output_dim=config['context_dim']
        )
        
        self.vector_field = VectorFieldNetwork(
            physical_dim=config['physical_dim'],
            depth_points=config['depth_points'],
            context_dim=config['context_dim']
        )
        
    def forward(self, t, x_t, stokes, region_ids, padding_mask=None):
        context = self.encoder(stokes, region_ids, padding_mask)
        
        v_pred = self.vector_field(t, x_t, context)
        return v_pred