"""Measure how much information each spectral region actually contributes.

For each region its spectral block is shuffled across samples, which destroys the
correspondence with the atmosphere without changing the marginal distribution of
the input. If the region carries information the metrics drop; if it does not,
they stay put.

Usage:
    python ablation_regions.py [path/to/checkpoint.pth]
"""
import sys

import h5py
import numpy as np
import torch
from sklearn.metrics import r2_score

from models import SolarFlowModel
from preprocessing import REGION_KEYS, load_stats, normalize_stokes, region_bounds, region_ids

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else './checkpoints_region_norm/multimodal_ep50.pth'
DATA_MASTER = './dataset/multimodal_stokes_testing.h5'
DATA_MODELS = './dataset/database_models/models_testing.h5'
GOOD_PROFILES = './dataset/database_630/good_profiles_testing.npy'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N = 400
NUM_ENSEMBLES = 30
SHUFFLE_OFFSET = 137
CONFIG = {'context_dim': 64, 'physical_dim': 6, 'depth_points': 80}
LABELS = ['T', 'vmic', 'v', 'Bx', 'By', 'Bz']

stats = load_stats()
keep = [0, 2, 3, 4, 5, 6]
m_mean = stats['models_mean'][keep].reshape(1, 1, 6)
m_std = stats['models_std'][keep].reshape(1, 1, 6)
bounds = region_bounds(stats['region_lengths'])
total_len = bounds[-1][1]

good = np.sort(np.load(GOOD_PROFILES))[:N]
with h5py.File(DATA_MASTER, 'r') as f, h5py.File(DATA_MODELS, 'r') as fm:
    raw = np.concatenate([f[k][good] for k in REGION_KEYS], axis=2)
    full = fm['model'][good]
logtau = full[0, :, 0]
reals = full[:, :, [1, 3, 4, 5, 6, 7]]
region_t = torch.from_numpy(np.tile(region_ids(stats['region_lengths']), (N, 1))).to(DEVICE)

model = SolarFlowModel(CONFIG).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True))
model.eval()
print(f'Checkpoint: {CHECKPOINT}\nSamples: {N} (valid profiles only)\n')


def context(x, mask_regions=()):
    xt = torch.from_numpy(normalize_stokes(x, stats)).float().to(DEVICE)
    mask = torch.zeros((N, total_len), dtype=torch.bool, device=DEVICE)
    for r in mask_regions:
        mask[:, bounds[r][0]:bounds[r][1]] = True
    with torch.no_grad():
        return model.encoder(xt, region_t, padding_mask=mask)


@torch.no_grad()
def predict(ctx, steps=30):
    torch.manual_seed(0)
    acc = torch.zeros(N, 80, 6, device=DEVICE)
    for _ in range(NUM_ENSEMBLES):
        x = torch.randn(N, 80, 6, device=DEVICE)
        for i in range(steps):
            t = torch.full((N, 1), i / steps, device=DEVICE)
            x = x + model.vector_field(t, x, ctx) * (1.0 / steps)
        acc += x
    return (acc / NUM_ENSEMBLES).cpu().numpy() * m_std + m_mean


def shuffled(regions):
    out = raw.copy()
    roll = np.roll(np.arange(N), SHUFFLE_OFFSET)
    for r in regions:
        i0, i1 = bounds[r]
        out[:, :, i0:i1] = raw[roll][:, :, i0:i1]
    return out


variants = {
    'intact': (raw, ()),
    'shuffle 630': (shuffled([0]), ()),
    'shuffle CaII': (shuffled([1]), ()),
    'shuffle CaII_K': (shuffled([2]), ()),
    'shuffle both CaII': (shuffled([1, 2]), ()),
    'mask both CaII': (raw, (1, 2)),
}

base_ctx = context(raw)
scale = base_ctx.std().item()
sel = (logtau >= -3.0) & (logtau <= 0.0)
high = logtau < -3.0

print(f"{'variant':22s} {'dctx':>8s} " + ' '.join(f'{l:>7s}' for l in LABELS)
      + f" | {'R2 T high':>10s}")
for name, (x, mask_regions) in variants.items():
    ctx = context(x, mask_regions)
    d = (ctx - base_ctx).abs().mean().item() / scale * 100
    p = predict(ctx)
    r2 = [r2_score(reals[:, sel, i].ravel(), p[:, sel, i].ravel()) for i in range(6)]
    r2_high = r2_score(reals[:, high, 0].ravel(), p[:, high, 0].ravel())
    print(f'{name:22s} {d:7.3f}% ' + ' '.join(f'{v:7.4f}' for v in r2)
          + f" | {r2_high:10.4f}")

print('\ndctx = mean change of the context vector relative to the intact case.')
print('R2 T high = temperature R2 for log(tau) < -3, where only Ca II can inform.')
