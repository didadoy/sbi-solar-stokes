import numpy as np

REGION_KEYS = ('stokes_630', 'stokes_caii', 'stokes_caii_k')

CONTINUUM_WINDOW = 10

DEFAULT_NOISE_SIGMA = 1e-3


def continuum_level(stokes_i, window=CONTINUUM_WINDOW):
    """Continuum intensity of one spectral region, in its original units.

    The median is used instead of the mean because the continuum distribution is
    strongly skewed in Ca II K, where the UV intensity depends exponentially on
    temperature. On the Fe I 630 nm region, which the IAC delivers already
    normalized to the quiet-Sun continuum, this statistic returns 1.05; that
    agreement is what justifies applying it to the other two regions.
    """
    wings = np.concatenate([stokes_i[:, :window], stokes_i[:, -window:]], axis=1)
    return float(np.median(wings.mean(axis=1)))


def region_bounds(lengths):
    bounds, start = [], 0
    for length in lengths:
        bounds.append((start, start + int(length)))
        start += int(length)
    return bounds


def load_stats(path='normalization_stats.npz'):
    stats = np.load(path)
    if 'continuum' not in stats:
        raise ValueError(
            f"{path} uses the old global-statistics format. "
            "Re-run calc_stats.py to regenerate it per region."
        )
    return {
        'continuum': stats['continuum'],
        'stokes_mean': stats['stokes_mean'],
        'stokes_std': stats['stokes_std'],
        'region_lengths': stats['region_lengths'],
        'models_mean': stats['models_mean'],
        'models_std': stats['models_std'],
    }


def normalize_stokes(stokes, stats, noise_sigma=0.0, rng=None):
    """Preprocess concatenated Stokes profiles: continuum, noise and Z-score.

    stokes: array of shape (..., 4, L_total) in the raw units of each source file.

    The three regions arrive in different units: 630 nm is normalized to the
    continuum while Ca II and Ca II K come in absolute CGS units, about five
    orders of magnitude lower. Each region is first divided by its own continuum
    so that a 1e-3 noise level means the same thing in all three, and only then
    standardized with its own statistics. A single global Z-score across the
    concatenated sequence collapses the Ca II tokens to a constant.
    """
    out = np.array(stokes, dtype=np.float32, copy=True)
    gen = rng if rng is not None else np.random
    for r, (i0, i1) in enumerate(region_bounds(stats['region_lengths'])):
        block = out[..., i0:i1] / stats['continuum'][r]
        if noise_sigma > 0.0:
            block = block + gen.normal(0.0, noise_sigma, size=block.shape)
        mean = stats['stokes_mean'][r].reshape(4, 1)
        std = stats['stokes_std'][r].reshape(4, 1)
        out[..., i0:i1] = (block - mean) / (std + 1e-6)
    return out


def region_ids(lengths):
    return np.concatenate([
        np.full(int(length), r, dtype=np.int64)
        for r, length in enumerate(lengths)
    ])
