"""Flow-matching timestep schedule and dense FLOPs estimation for Krea 2."""

from __future__ import annotations

import math

import numpy as np

from tpu_jax.config import Krea2DiTConfig


def roundup(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def compute_timesteps(
    *,
    width: int,
    height: int,
    steps: int,
    compression: int = 8,
    patch: int = 2,
    minres: int = 256,
    maxres: int = 1280,
    y1: float = 0.5,
    y2: float = 1.15,
    sigma: float = 1.0,
    mu: float | None = None,
) -> np.ndarray:
    """Exact NumPy equivalent of ``sampling.timesteps`` (t: 1.0 -> 0.0, length steps + 1)."""
    align = compression * patch
    w_aligned = roundup(width, align)
    h_aligned = roundup(height, align)
    seq_len = (h_aligned // align) * (w_aligned // align)

    if mu is None:
        x1 = (minres // align) ** 2
        x2 = (maxres // align) ** 2
        slope = (y2 - y1) / (x2 - x1)
        mu = slope * seq_len + (y1 - slope * x1)

    exp_mu = math.exp(mu)
    grid = np.linspace(1.0, 0.0, steps + 1, dtype=np.float64)
    out = np.empty_like(grid, dtype=np.float32)
    out[0] = 1.0
    out[-1] = 0.0
    if steps > 1:
        mid = grid[1:-1]
        out[1:-1] = (exp_mu / (exp_mu + (1.0 / mid - 1.0) ** sigma)).astype(np.float32)
    return out


def estimate_step_tflops(
    config: Krea2DiTConfig,
    *,
    width: int,
    height: int,
    txt_len: int = 512,
    pad_multiple: int = 256,
    attn_q_len: int | None = None,
    attn_kv_len: int | None = None,
    batch_size: int = 1,
    do_cfg: bool = False,
) -> float:
    """Estimate dense FLOPs (TFLOPs) for one Krea 2 denoising step (main DiT blocks)."""
    align = 8 * config.patch
    h_tok = roundup(height, align) // align
    w_tok = roundup(width, align) // align
    raw_seq = txt_len + h_tok * w_tok
    seq_len = roundup(raw_seq, pad_multiple)
    q_len = seq_len if attn_q_len is None else int(attn_q_len)
    kv_len = seq_len if attn_kv_len is None else int(attn_kv_len)

    d = config.features
    kv_d = config.kvheads * config.head_dim
    mlp_d = config.mlp_dim

    # Projections per token (2 * in * out MACs->FLOPs):
    # wq (d->d), wk (d->kv_d), wv (d->kv_d), gate (d->d), wo (d->d)
    proj_flops = 2.0 * seq_len * d * (3 * d + 2 * kv_d)
    # SwiGLU MLP: gate (d->mlp_d), up (d->mlp_d), down (mlp_d->d)
    mlp_flops = 6.0 * seq_len * d * mlp_d
    # Attention QK^T + AV: 4 * q_len * kv_len * d
    attn_flops = 4.0 * q_len * kv_len * d

    per_pass = config.layers * (proj_flops + mlp_flops + attn_flops)
    passes = batch_size * (2 if do_cfg else 1)
    return (per_pass * passes) / 1e12
