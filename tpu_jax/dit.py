"""Pure JAX/XLA implementation of Krea 2 SingleStreamDiT (12B GQA) + denoising loop.

Performance optimizations tailored for Google Cloud TPU (v6e / v5e / v5p):
1. Hoists ``txtfusion`` (4 transformer blocks), ``txtmlp``, sequence padding, and
   3D axial RoPE (cos/sin) ONCE outside the denoising loop (official PyTorch
   recomputes them on every step).
2. Full-graph ``jax.jit`` across the entire N-step Euler ODE loop via
   ``jax.lax.fori_loop`` with buffer donation (zero host syncs, zero Python
   dispatch bubbles).
3. ``jax.lax.scan`` over all 28 stacked ``SingleStreamBlock`` layers (compiles
   1 block body instead of 28 unrolled copies; faster compile + better XLA
   instruction/VMEM scheduling).
4. Load-time fused & pre-transposed MXU GEMMs:
   - Single ``qkvgate_w`` ``[6144, 15360]`` replacing 4 separate projections.
   - Single ``mlp_gate_up_w`` ``[6144, 32768]`` replacing 2 separate projections.
   - Pre-folded ``scale + 1.0`` RMSNorm weights in FP32.
5. Zero-copy 4:1 GQA attention backends:
   - ``splash``: Native TPU Pallas SplashAttention (native 48Q/12KV GQA, zero
     KV expansion in HBM, ``SegmentIds`` masking, zero O(S^2) materialization,
     critical for 2048x2048).
   - ``flash``: Native TPU Pallas FlashAttention.
   - ``xla``: ``jax.nn.dot_product_attention(..., implementation="xla")`` (native GQA).
   - ``einsum``: Exact FP32 reference for CPU/TPU parity testing.
6. Pre-pads the joint sequence ONCE before ``lax.scan`` to a multiple of
   ``max(256, block_size)`` so per-block attention incurs zero pad/slice overhead
   (note 1024x1024 -> 4608 = 9*512 and 2048x2048 -> 16896 = 33*512 need zero
   extra padding for ``block_size=512``).
7. Optional 1D Tensor Parallelism via ``jax.sharding.Mesh`` across TPU chips.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from tpu_jax.config import Krea2DiTConfig
from tpu_jax.scheduler import roundup

VALID_ATTN_IMPLS = ("splash", "flash", "xla", "einsum")


def _rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    """Exact match to mmdit.RMSNorm where ``weight`` already has ``+ 1.0`` folded in."""
    x_f32 = x.astype(jnp.float32)
    inv_rms = lax.rsqrt(jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True) + jnp.float32(eps))
    return (x_f32 * inv_rms * weight.astype(jnp.float32)).astype(x.dtype)


def _matmul(x: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    return lax.dot_general(
        x,
        w,
        (((x.ndim - 1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(x.dtype)


def precompute_rope_omegas(
    axes_dims: tuple[int, int, int],
    theta: float = 1000.0,
    ntk: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact float64->float32 inverse frequency tables matching ``mmdit.rope``."""
    omegas = []
    for d in axes_dims:
        scale = np.arange(0, d, 2, dtype=np.float64) / float(d)
        omega = 1.0 / np.power(float(theta) * float(ntk), scale)
        omegas.append(omega.astype(np.float32))
    return tuple(omegas)  # type: ignore[return-value]


def compute_rope_cos_sin(
    pos: jnp.ndarray,
    omegas: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build FP32 ``(cos, sin)`` of shape ``[B, S, head_dim // 2]`` matching ``PositionalEncoding``."""
    pos_f32 = pos.astype(jnp.float32)
    cos_parts = []
    sin_parts = []
    for axis_idx, omega in enumerate(omegas):
        angles = pos_f32[..., axis_idx : axis_idx + 1] * omega[None, None, :]
        cos_parts.append(jnp.cos(angles))
        sin_parts.append(jnp.sin(angles))
    return jnp.concatenate(cos_parts, axis=-1), jnp.concatenate(sin_parts, axis=-1)


def _apply_rope(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """Apply interleaved pair RoPE to ``x [B, S, H, D]`` using FP32 ``cos, sin [B, S, D//2]``."""
    orig_dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    x0 = x_f32[..., 0::2]
    x1 = x_f32[..., 1::2]
    c = cos[:, :, None, :]
    s = sin[:, :, None, :]
    y0 = x0 * c - x1 * s
    y1 = x0 * s + x1 * c
    return jnp.stack([y0, y1], axis=-1).reshape(x.shape).astype(orig_dtype)


def _temb(
    t: jnp.ndarray,
    dim: int,
    time_freqs: jnp.ndarray,
    *,
    tfactor: float = 1000.0,
    out_dtype: Any = jnp.bfloat16,
) -> jnp.ndarray:
    """Exact match to ``mmdit.temb``: returns ``[B, 1, dim]`` with ``cat(cos, sin)``."""
    args = (t.astype(jnp.float32) * jnp.float32(tfactor))[:, None, None] * time_freqs[None, None, :]
    return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1).astype(out_dtype)


def _splash_attention_single(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    valid_mask_1d: jnp.ndarray | None,
    *,
    block_size: int = 512,
) -> jnp.ndarray:
    """Single-batch native GQA Pallas SplashAttention on ``[S, H_q, D]``, ``[S, H_kv, D]``."""
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel,
        splash_attention_mask,
    )

    seq_len, num_q_heads, head_dim = q.shape
    eff_block = min(block_size, max(128, ((seq_len + 127) // 128) * 128))
    padded_len = ((seq_len + eff_block - 1) // eff_block) * eff_block
    pad = padded_len - seq_len

    # SplashAttention expects [H, S, D] and does NOT scale Q internally.
    scale = jnp.float32(1.0 / math.sqrt(head_dim))
    q_h = (jnp.transpose(q, (1, 0, 2)).astype(jnp.float32) * scale).astype(q.dtype)
    k_h = jnp.transpose(k, (1, 0, 2))
    v_h = jnp.transpose(v, (1, 0, 2))

    if pad > 0:
        q_h = jnp.pad(q_h, ((0, 0), (0, pad), (0, 0)))
        k_h = jnp.pad(k_h, ((0, 0), (0, pad), (0, 0)))
        v_h = jnp.pad(v_h, ((0, 0), (0, pad), (0, 0)))
        if valid_mask_1d is not None:
            valid_padded = jnp.pad(valid_mask_1d, ((0, pad),), constant_values=False)
        else:
            valid_padded = jnp.pad(
                jnp.ones((seq_len,), dtype=jnp.bool_), ((0, pad),), constant_values=False
            )
    else:
        valid_padded = valid_mask_1d

    if valid_padded is not None:
        seg_val = valid_padded.astype(jnp.int32)
        seg_ids = splash_attention_kernel.SegmentIds(q=seg_val, kv=seg_val)
    else:
        seg_ids = None

    mask = splash_attention_mask.MultiHeadMask(
        [splash_attention_mask.FullMask((padded_len, padded_len)) for _ in range(num_q_heads)]
    )
    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=min(eff_block, padded_len),
        block_kv=min(eff_block, padded_len),
        block_kv_compute=min(eff_block, padded_len),
    )
    kernel = splash_attention_kernel.make_splash_mha(
        mask=mask,
        head_shards=1,
        q_seq_shards=1,
        block_sizes=block_sizes,
    )
    out_h = kernel(q_h, k_h, v_h, segment_ids=seg_ids)
    if pad > 0:
        out_h = out_h[:, :seq_len, :]
    return jnp.transpose(out_h, (1, 0, 2)).astype(v.dtype)


def _pallas_flash_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    valid_mask: jnp.ndarray | None,
    *,
    block_size: int = 512,
) -> jnp.ndarray:
    """TPU Pallas FlashAttention on ``[B, S, H_q, D]``, ``[B, S, H_kv, D]``."""
    from jax.experimental.pallas.ops.tpu import flash_attention as pallas_fa

    batch, seq_len, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    eff_block = min(block_size, max(128, ((seq_len + 127) // 128) * 128))
    padded_len = ((seq_len + eff_block - 1) // eff_block) * eff_block
    pad = padded_len - seq_len

    # Pallas flash_attention expects [B, H, S, D] with matching H across Q/K/V.
    q_bhsd = jnp.transpose(q, (0, 2, 1, 3))
    k_bhsd = jnp.transpose(k, (0, 2, 1, 3))
    v_bhsd = jnp.transpose(v, (0, 2, 1, 3))
    if num_kv_heads != num_q_heads:
        rep = num_q_heads // num_kv_heads
        k_bhsd = jnp.repeat(k_bhsd, rep, axis=1)
        v_bhsd = jnp.repeat(v_bhsd, rep, axis=1)

    if pad > 0:
        q_bhsd = jnp.pad(q_bhsd, ((0, 0), (0, 0), (0, pad), (0, 0)))
        k_bhsd = jnp.pad(k_bhsd, ((0, 0), (0, 0), (0, pad), (0, 0)))
        v_bhsd = jnp.pad(v_bhsd, ((0, 0), (0, 0), (0, pad), (0, 0)))
        if valid_mask is not None:
            valid_padded = jnp.pad(valid_mask, ((0, 0), (0, pad)), constant_values=False)
        else:
            valid_padded = jnp.pad(
                jnp.ones((batch, seq_len), dtype=jnp.bool_), ((0, 0), (0, pad)), constant_values=False
            )
    else:
        valid_padded = valid_mask

    if valid_padded is not None:
        seg_val = valid_padded.astype(jnp.int32)
        seg_ids = pallas_fa.SegmentIds(q=seg_val, kv=seg_val)
    else:
        seg_ids = None

    block_sizes = pallas_fa.BlockSizes(
        block_q=eff_block,
        block_k_major=eff_block,
        block_k=eff_block,
        block_b=1,
    )
    out = pallas_fa.flash_attention(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        segment_ids=seg_ids,
        sm_scale=1.0 / math.sqrt(head_dim),
        block_sizes=block_sizes,
    )
    if pad > 0:
        out = out[:, :, :seq_len, :]
    return jnp.transpose(out, (0, 2, 1, 3)).astype(v.dtype)


def _gqa_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    valid_mask: jnp.ndarray | None,
    *,
    attn_impl: str = "splash",
    splash_block_size: int = 512,
) -> jnp.ndarray:
    """Multi-head / Grouped-Query Attention: ``[B, S, H_q, D]`` -> ``[B, S, H_q * D]``."""
    batch, seq_len, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    out_dim = num_q_heads * head_dim

    if attn_impl == "splash":
        if batch == 1:
            vm0 = valid_mask[0] if valid_mask is not None else None
            out = _splash_attention_single(q[0], k[0], v[0], vm0, block_size=splash_block_size)[None, ...]
        else:
            outs = [
                _splash_attention_single(
                    q[b],
                    k[b],
                    v[b],
                    valid_mask[b] if valid_mask is not None else None,
                    block_size=splash_block_size,
                )
                for b in range(batch)
            ]
            out = jnp.stack(outs, axis=0)
        return out.reshape(batch, seq_len, out_dim)

    if attn_impl == "flash":
        out = _pallas_flash_attention(q, k, v, valid_mask, block_size=splash_block_size)
        return out.reshape(batch, seq_len, out_dim)

    if attn_impl == "xla":
        mask_4d = valid_mask[:, None, None, :] if valid_mask is not None else None
        out = jax.nn.dot_product_attention(q, k, v, mask=mask_4d, implementation="xla")
        return out.astype(v.dtype).reshape(batch, seq_len, out_dim)

    if attn_impl == "einsum":
        group_size = num_q_heads // num_kv_heads
        scale = jnp.float32(1.0 / math.sqrt(head_dim))
        q_g = q.astype(jnp.float32).reshape(batch, seq_len, num_kv_heads, group_size, head_dim)
        k_f = k.astype(jnp.float32)
        v_f = v.astype(jnp.float32)
        logits = jnp.einsum("bskgd,btkd->bkgst", q_g, k_f) * scale
        if valid_mask is not None:
            logits = jnp.where(valid_mask[:, None, None, None, :], logits, jnp.float32(-1e30))
        weights = jax.nn.softmax(logits, axis=-1)
        out = jnp.einsum("bkgst,btkd->bskgd", weights, v_f).astype(v.dtype)
        return out.reshape(batch, seq_len, out_dim)

    raise ValueError(f"Unknown attn_impl={attn_impl!r}; expected one of {VALID_ATTN_IMPLS}.")


def _attention_block_op(
    y: jnp.ndarray,
    block: dict[str, jnp.ndarray],
    cos: jnp.ndarray | None,
    sin: jnp.ndarray | None,
    valid_mask: jnp.ndarray | None,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    norm_eps: float,
    attn_impl: str,
    splash_block_size: int,
) -> jnp.ndarray:
    """Fused KV-grouped QKVGate -> QK-Norm -> RoPE -> GQA -> Sigmoid-Gate -> Output Proj.

    ``block["qkvgate_w"]`` has shape ``[dim, kv_heads * (2 * group + 2) * head_dim]``
    organized per KV-head group as ``[Q_group, K, V, Gate_group]``. This runs as a
    single MXU GEMM on 1 chip AND shards cleanly along ``kv_heads`` for 1D TP.
    """
    batch, seq_len, _ = y.shape
    group_size = num_heads // num_kv_heads
    q_g = group_size * head_dim
    per_kv = 2 * q_g + 2 * head_dim

    qkvg = _matmul(y, block["qkvgate_w"]).reshape(batch, seq_len, num_kv_heads, per_kv)
    q = qkvg[..., :q_g].reshape(batch, seq_len, num_heads, head_dim)
    k = qkvg[..., q_g : q_g + head_dim]
    v = qkvg[..., q_g + head_dim : q_g + 2 * head_dim]
    gate = qkvg[..., q_g + 2 * head_dim :].reshape(batch, seq_len, num_heads * head_dim)

    q = _rms_norm(q, block["qnorm"], norm_eps)
    k = _rms_norm(k, block["knorm"], norm_eps)
    if cos is not None and sin is not None:
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

    attn_out = _gqa_attention(
        q,
        k,
        v,
        valid_mask,
        attn_impl=attn_impl,
        splash_block_size=splash_block_size,
    )
    gated = attn_out * jax.nn.sigmoid(gate)
    return _matmul(gated, block["wo_w"])


def _swiglu_op(y: jnp.ndarray, block: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Fused Gate+Up SwiGLU MLP (``mlp_gate_up_w`` shape ``[dim, 2 * mlp_dim]`` paired by col)."""
    gu = _matmul(y, block["mlp_gate_up_w"])
    gu_paired = gu.reshape(*y.shape[:-1], -1, 2)
    gate = gu_paired[..., 0]
    up = gu_paired[..., 1]
    return _matmul(jax.nn.silu(gate) * up, block["mlp_down_w"])


def _text_fusion_block_forward(
    x: jnp.ndarray,
    block: dict[str, jnp.ndarray],
    valid_mask: jnp.ndarray | None,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    norm_eps: float,
    attn_impl: str,
) -> jnp.ndarray:
    # TextFusion runs once per prompt on tiny S (12 or 512); xla/einsum avoids Pallas pad overhead.
    eff_impl = "xla" if attn_impl in ("splash", "flash") else attn_impl
    h = _rms_norm(x, block["prenorm"], norm_eps)
    x = x + _attention_block_op(
        h,
        block,
        None,
        None,
        valid_mask,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        norm_eps=norm_eps,
        attn_impl=eff_impl,
        splash_block_size=128,
    )
    h = _rms_norm(x, block["postnorm"], norm_eps)
    return x + _swiglu_op(h, block)


def _single_stream_block_forward(
    x: jnp.ndarray,
    block: dict[str, jnp.ndarray],
    tvec: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    valid_mask: jnp.ndarray,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    norm_eps: float,
    attn_impl: str,
    splash_block_size: int,
) -> jnp.ndarray:
    """Single Krea 2 ``SingleStreamBlock`` forward (``jax.lax.scan`` body)."""
    w_dtype = block["qkvgate_w"].dtype
    mod_out = tvec + block["mod_lin"][None, None, :]
    prescale, preshift, pregate, postscale, postshift, postgate = jnp.split(mod_out, 6, axis=-1)

    y = ((1.0 + prescale) * _rms_norm(x, block["prenorm"], norm_eps) + preshift).astype(w_dtype)
    attn_out = _attention_block_op(
        y,
        block,
        cos,
        sin,
        valid_mask,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        norm_eps=norm_eps,
        attn_impl=attn_impl,
        splash_block_size=splash_block_size,
    )
    x = x + pregate * attn_out

    y = ((1.0 + postscale) * _rms_norm(x, block["postnorm"], norm_eps) + postshift).astype(w_dtype)
    mlp_out = _swiglu_op(y, block)
    return x + postgate * mlp_out


def build_positions_and_mask(
    batch: int,
    latent_h: int,
    latent_w: int,
    txtmask: jnp.ndarray,
    *,
    patch: int = 2,
    pad_multiple: int = 256,
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """Build padded ``(pos, valid_mask, pad_len)`` matching ``sampling.prepare`` + ``mmdit.py``."""
    h_tok = latent_h // patch
    w_tok = latent_w // patch
    img_len = h_tok * w_tok
    txt_len = txtmask.shape[1]

    rows = jnp.arange(h_tok, dtype=jnp.float32)[:, None]
    cols = jnp.arange(w_tok, dtype=jnp.float32)[None, :]
    zeros_hw = jnp.zeros((h_tok, w_tok), dtype=jnp.float32)
    img_ids = jnp.stack(
        [zeros_hw, jnp.broadcast_to(rows, (h_tok, w_tok)), jnp.broadcast_to(cols, (h_tok, w_tok))],
        axis=-1,
    ).reshape(1, img_len, 3)
    img_pos = jnp.broadcast_to(img_ids, (batch, img_len, 3))
    txt_pos = jnp.zeros((batch, txt_len, 3), dtype=jnp.float32)
    pos = jnp.concatenate([txt_pos, img_pos], axis=1)

    img_mask = jnp.ones((batch, img_len), dtype=jnp.bool_)
    valid_mask = jnp.concatenate([txtmask.astype(jnp.bool_), img_mask], axis=1)

    full_len = txt_len + img_len
    pad_len = (-full_len) % pad_multiple
    if pad_len > 0:
        pos = jnp.pad(pos, ((0, 0), (0, pad_len), (0, 0)))
        valid_mask = jnp.pad(valid_mask, ((0, 0), (0, pad_len)), constant_values=False)
    return pos, valid_mask, pad_len


def prepare_text_conditioning(
    params: dict[str, Any],
    config: Krea2DiTConfig,
    context: jnp.ndarray,
    txtmask: jnp.ndarray,
    *,
    latent_h: int,
    latent_w: int,
    attn_impl: str = "splash",
    splash_block_size: int = 512,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """Run TextFusion + TextMLP + RoPE ONCE before the denoising loop.

    Returns ``(text_tokens, rope_cos, rope_sin, joint_valid_mask, pad_len)``.
    """
    w_dtype = params["first_w"].dtype
    b, l_txt, n_layers, d_txt = context.shape
    txtmask_bool = txtmask.astype(jnp.bool_)

    # 1. Layerwise attention across the N=12 selected Qwen3-VL layers per token.
    x = context.astype(w_dtype).reshape(b * l_txt, n_layers, d_txt)
    for blk in params["txtfusion_layerwise"]:
        x = _text_fusion_block_forward(
            x,
            blk,
            None,
            num_heads=config.txtheads,
            num_kv_heads=config.txtkvheads,
            head_dim=config.text_head_dim,
            norm_eps=config.norm_eps,
            attn_impl=attn_impl,
        )

    # 2. Project N=12 -> 1 along layer axis: (b*l, n, d) -> (b, l, d, n) @ (n, 1) -> (b, l, d)
    x = jnp.transpose(x.reshape(b, l_txt, n_layers, d_txt), (0, 1, 3, 2))
    x = _matmul(x, params["txtfusion_proj_w"]).squeeze(-1)

    # 3. Token-level refiner blocks with text mask.
    for blk in params["txtfusion_refiner"]:
        x = _text_fusion_block_forward(
            x,
            blk,
            txtmask_bool,
            num_heads=config.txtheads,
            num_kv_heads=config.txtkvheads,
            head_dim=config.text_head_dim,
            norm_eps=config.norm_eps,
            attn_impl=attn_impl,
        )

    # 4. Text MLP: RMSNorm -> Linear -> GELU(tanh) -> Linear
    x = _rms_norm(x, params["txtmlp_norm"], config.norm_eps)
    x = _matmul(x, params["txtmlp_fc1_w"]) + params["txtmlp_fc1_b"]
    x = jax.nn.gelu(x, approximate=True)
    text_tokens = _matmul(x, params["txtmlp_fc2_w"]) + params["txtmlp_fc2_b"]

    # 5. Pre-pad once to a multiple of max(256, splash_block_size) for zero per-block padding.
    pad_multiple = max(256, splash_block_size) if attn_impl in ("splash", "flash") else 256
    pos, valid_mask, pad_len = build_positions_and_mask(
        b, latent_h, latent_w, txtmask_bool, patch=config.patch, pad_multiple=pad_multiple
    )
    rope_cos, rope_sin = compute_rope_cos_sin(pos, params["rope_omegas"])
    return text_tokens, rope_cos, rope_sin, valid_mask, pad_len


def dit_forward_precomputed(
    params: dict[str, Any],
    config: Krea2DiTConfig,
    img_tokens: jnp.ndarray,
    timestep: jnp.ndarray,
    text_tokens: jnp.ndarray,
    rope_cos: jnp.ndarray,
    rope_sin: jnp.ndarray,
    valid_mask: jnp.ndarray,
    *,
    pad_len: int = 0,
    attn_impl: str = "splash",
    splash_block_size: int = 512,
) -> jnp.ndarray:
    """Single Krea 2 DiT forward pass using precomputed text conditioning and RoPE.

    Args:
        img_tokens: Patchified latent ``[B, img_len, C * patch^2]``.
        timestep: Scalar or ``[B]`` float timestep in ``[0, 1]``.

    Returns:
        Predicted velocity ``[B, img_len, C * patch^2]`` in model dtype.
    """
    w_dtype = params["first_w"].dtype
    batch, img_len, _ = img_tokens.shape
    txt_len = text_tokens.shape[1]

    # Timestep embedding -> tmlp -> tproj
    t_b = jnp.broadcast_to(jnp.asarray(timestep, dtype=jnp.float32).reshape(-1), (batch,))
    t_in = _temb(t_b, config.tdim, params["time_freqs"], out_dtype=w_dtype)
    t_h = jax.nn.gelu(_matmul(t_in, params["tmlp_fc1_w"]) + params["tmlp_fc1_b"], approximate=True)
    t_emb = _matmul(t_h, params["tmlp_fc2_w"]) + params["tmlp_fc2_b"]
    tvec = _matmul(jax.nn.gelu(t_emb, approximate=True), params["tproj_w"]) + params["tproj_b"]

    img_emb = _matmul(img_tokens.astype(w_dtype), params["first_w"]) + params["first_b"]
    combined = jnp.concatenate([text_tokens, img_emb], axis=1)
    if pad_len > 0:
        combined = jnp.pad(combined, ((0, 0), (0, pad_len), (0, 0)))

    block_fn = partial(
        _single_stream_block_forward,
        num_heads=config.heads,
        num_kv_heads=config.kvheads,
        head_dim=config.head_dim,
        norm_eps=config.norm_eps,
        attn_impl=attn_impl,
        splash_block_size=splash_block_size,
    )

    def _scan_body(carry: jnp.ndarray, blk: dict[str, jnp.ndarray]):
        return block_fn(carry, blk, tvec, rope_cos, rope_sin, valid_mask), None

    combined, _ = lax.scan(_scan_body, combined, params["blocks"])

    # Slice image tokens BEFORE LastLayer so we skip redundant final norm+linear on text+pad!
    img_out = combined[:, txt_len : txt_len + img_len, :]
    mod_last = t_emb + params["last_mod_lin"][None, :, :]
    scale, shift = jnp.split(mod_last, 2, axis=1)
    img_out = (1.0 + scale) * _rms_norm(img_out, params["last_norm"], config.norm_eps) + shift
    return _matmul(img_out, params["last_linear_w"]) + params["last_linear_b"]


def patchify_latents(latents: jnp.ndarray, patch: int = 2) -> jnp.ndarray:
    """``[B, C, H, W]`` -> ``[B, (H/p)*(W/p), C*p*p]`` matching ``sampling.prepare``."""
    b, c, h, w = latents.shape
    hp, wp = h // patch, w // patch
    x = latents.reshape(b, c, hp, patch, wp, patch)
    x = jnp.transpose(x, (0, 2, 4, 1, 3, 5))
    return x.reshape(b, hp * wp, c * patch * patch)


def unpatchify_latents(tokens: jnp.ndarray, latent_h: int, latent_w: int, channels: int = 16, patch: int = 2) -> jnp.ndarray:
    """``[B, (H/p)*(W/p), C*p*p]`` -> ``[B, C, H, W]`` matching ``sampling.sample``."""
    b = tokens.shape[0]
    hp, wp = latent_h // patch, latent_w // patch
    x = tokens.reshape(b, hp, wp, channels, patch, patch)
    x = jnp.transpose(x, (0, 3, 1, 4, 2, 5))
    return x.reshape(b, channels, latent_h, latent_w)


def make_denoise_fn(
    config: Krea2DiTConfig,
    *,
    width: int,
    height: int,
    guidance_scale: float = 0.0,
    attn_impl: str = "splash",
    splash_block_size: int = 512,
    latent_dtype: Any = jnp.float32,
    donate_latents: bool = True,
):
    """Build a single ``jax.jit``-compiled full N-step Euler denoising function.

    - ``guidance_scale == 0.0`` (Turbo default): single forward pass per step,
      zero unconditional conditioning work.
    - ``guidance_scale > 0.0`` (Raw / CFG mode): runs cond + uncond with
      ``v = cond + guidance_scale * (cond - uncond)`` (exact match to ``sampling.py``).
    """
    align = 8 * config.patch
    w_aligned = roundup(width, align)
    h_aligned = roundup(height, align)
    latent_h = h_aligned // 8
    latent_w = w_aligned // 8
    do_cfg = guidance_scale > 0.0
    donate = (1,) if donate_latents else ()

    @partial(jax.jit, donate_argnums=donate)
    def _denoise(
        params: dict[str, Any],
        init_latents: jnp.ndarray,
        pos_context: jnp.ndarray,
        pos_txtmask: jnp.ndarray,
        neg_context: jnp.ndarray | None,
        neg_txtmask: jnp.ndarray | None,
        ts: jnp.ndarray,
    ) -> jnp.ndarray:
        num_steps = ts.shape[0] - 1
        img_tokens = patchify_latents(init_latents.astype(latent_dtype), patch=config.patch)

        pos_txt, pos_cos, pos_sin, pos_mask, pos_pad = prepare_text_conditioning(
            params,
            config,
            pos_context,
            pos_txtmask,
            latent_h=latent_h,
            latent_w=latent_w,
            attn_impl=attn_impl,
            splash_block_size=splash_block_size,
        )
        if do_cfg:
            assert neg_context is not None and neg_txtmask is not None
            neg_txt, neg_cos, neg_sin, neg_mask, neg_pad = prepare_text_conditioning(
                params,
                config,
                neg_context,
                neg_txtmask,
                latent_h=latent_h,
                latent_w=latent_w,
                attn_impl=attn_impl,
                splash_block_size=splash_block_size,
            )
        else:
            neg_txt = neg_cos = neg_sin = neg_mask = None
            neg_pad = 0

        def _step(step_idx: int, cur_tokens: jnp.ndarray) -> jnp.ndarray:
            t_curr = ts[step_idx]
            t_prev = ts[step_idx + 1]
            dt = (t_prev - t_curr).astype(latent_dtype)

            cond = dit_forward_precomputed(
                params,
                config,
                cur_tokens,
                t_curr,
                pos_txt,
                pos_cos,
                pos_sin,
                pos_mask,
                pad_len=pos_pad,
                attn_impl=attn_impl,
                splash_block_size=splash_block_size,
            ).astype(latent_dtype)

            if do_cfg:
                uncond = dit_forward_precomputed(
                    params,
                    config,
                    cur_tokens,
                    t_curr,
                    neg_txt,
                    neg_cos,
                    neg_sin,
                    neg_mask,
                    pad_len=neg_pad,
                    attn_impl=attn_impl,
                    splash_block_size=splash_block_size,
                ).astype(latent_dtype)
                v = cond + jnp.asarray(guidance_scale, dtype=latent_dtype) * (cond - uncond)
            else:
                v = cond

            return cur_tokens + dt * v

        final_tokens = lax.fori_loop(0, num_steps, _step, img_tokens)
        return unpatchify_latents(
            final_tokens,
            latent_h=latent_h,
            latent_w=latent_w,
            channels=config.channels,
            patch=config.patch,
        )

    return _denoise
