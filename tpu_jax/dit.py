"""Pure JAX/XLA implementation of Krea 2 SingleStreamDiT (12B GQA) + denoising loop.

Performance optimizations tailored for Google Cloud TPU (v6e / v5e / v5p):
1. Hoists ``txtfusion`` (4 transformer blocks), ``txtmlp``, sequence padding, and
   3D axial RoPE (cos/sin) ONCE outside the denoising loop (and broadcasts across
   ``batch > 1`` for multi-seed Midjourney-style batch generation).
2. Full-graph ``jax.jit`` across the entire N-step Euler ODE loop via
   ``jax.lax.fori_loop`` with buffer donation (zero host syncs, zero Python
   dispatch bubbles).
3. ``jax.lax.scan`` over all 28 stacked ``SingleStreamBlock`` layers with
   configurable ``unroll`` to let XLA Latency Hiding Scheduler overlap weight
   DMA prefetching with compute.
4. TPU-aligned load-time fused & pre-transposed MXU GEMMs (zero stride-2 lane shuffles):
   - Contiguous per-shard ``qkvgate_w`` (``[Q | K | V | Gate]`` on 1 chip).
   - Contiguous per-shard ``mlp_gate_up_w`` (``[Gate | Up]``, eliminating minor-dim
     stride-2 de-interleave).
   - Pre-permuted ``[even | odd]`` Q/K head channels for contiguous half-split RoPE.
   - Pre-folded ``scale + 1.0`` RMSNorm and ``1 / sqrt(head_dim)`` folded into ``qnorm``.
   - Fused FP32 ``_modulated_rms_norm`` avoiding redundant BF16 roundtrips.
5. Text token compaction + image-first ``[img | valid_txt | pad]`` layout:
   - Compacts valid text tokens into small buckets (e.g. 64/128/256/512), cutting
     dead text tokens from all 28 layers' GEMMs and attention.
   - Places aligned image tokens first so 94-97% of SplashAttention tiles are
     compile-time ``FULL`` (zero VPU ``SegmentIds`` mask overhead when using static
     masks).
6. Decoupled GEMM alignment (multiple of 256 for v6e 256x256 MXU) and asymmetric
   Pallas SplashAttention tiling (``block_q``, ``block_kv``, ``block_kv_compute``)
   with single-launch merged-head batching for shared-prompt ``batch > 1``.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from tpu_jax.config import Krea2DiTConfig
from tpu_jax.scheduler import roundup

VALID_ATTN_IMPLS = ("splash", "flash", "xla", "einsum", "pallas_gqa")
_LOG2_E = np.float32(np.log2(np.e))


def _rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    """Exact match to mmdit.RMSNorm where ``weight`` already has ``+ 1.0`` folded in."""
    x_f32 = x.astype(jnp.float32)
    inv_rms = lax.rsqrt(jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True) + jnp.float32(eps))
    return (x_f32 * inv_rms * weight.astype(jnp.float32)).astype(x.dtype)


def _modulated_rms_norm(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    scale: jnp.ndarray,
    shift: jnp.ndarray,
    eps: float = 1e-5,
    out_dtype: Any | None = None,
) -> jnp.ndarray:
    """Fused FP32 RMSNorm + AdaLN modulation ``(1 + scale) * rms_norm(x) + shift``."""
    x_f32 = x.astype(jnp.float32)
    inv_rms = lax.rsqrt(jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True) + jnp.float32(eps))
    normed = x_f32 * inv_rms * weight.astype(jnp.float32)
    out = (jnp.float32(1.0) + scale.astype(jnp.float32)) * normed + shift.astype(jnp.float32)
    return out.astype(x.dtype if out_dtype is None else out_dtype)


def _matmul(x: jnp.ndarray, w: jnp.ndarray, scale: jnp.ndarray | None = None) -> jnp.ndarray:
    w_eff = w.astype(x.dtype) if w.dtype != x.dtype else w
    out = lax.dot_general(
        x,
        w_eff,
        (((x.ndim - 1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    if scale is not None:
        out = out * scale.astype(jnp.float32)
    return out.astype(x.dtype)


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


def _apply_rope(
    x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray, *, head_first: bool = False
) -> jnp.ndarray:
    """Apply contiguous half-split RoPE to BSHD or BHSD (weights pre-permuted even/odd)."""
    orig_dtype = x.dtype
    half = x.shape[-1] // 2
    x_f32 = x.astype(jnp.float32)
    x0 = x_f32[..., :half]
    x1 = x_f32[..., half:]
    head_axis = 1 if head_first else 2
    c = jnp.expand_dims(cos, head_axis)
    s = jnp.expand_dims(sin, head_axis)
    y0 = x0 * c - x1 * s
    y1 = x0 * s + x1 * c
    return jnp.concatenate([y0, y1], axis=-1).astype(orig_dtype)


def _temb(
    t: jnp.ndarray,
    dim: int,
    time_freqs: jnp.ndarray,
    *,
    tfactor: float = 1000.0,
    out_dtype: Any = jnp.bfloat16,
) -> jnp.ndarray:
    """Exact match to ``mmdit.temb``: returns ``[B_t, 1, dim]`` with ``cat(cos, sin)``."""
    args = (t.astype(jnp.float32) * jnp.float32(tfactor))[:, None, None] * time_freqs[None, None, :]
    return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1).astype(out_dtype)


def compact_text_embeddings(
    context: np.ndarray,
    txtmask: np.ndarray,
    *,
    buckets: tuple[int, ...] = (64, 128, 256, 512),
    exact: bool = False,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Compact valid text tokens to the front and pad to a small static bucket length.

    Valid for Krea 2 because:
    - ``TextFusionTransformer`` has no positional encoding or causal mask (only ``txtmask``).
    - All text tokens in ``SingleStreamDiT`` share identical 3D RoPE position ``(0, 0, 0)``.

    Returns:
        ``(ctx_compact, mask_compact, common_valid_len)`` where ``common_valid_len`` is the
        exact number of valid text tokens if every batch item has the same count, else ``None``.
    """
    mask_bool = np.asarray(txtmask, dtype=np.bool_)
    batch, orig_len = mask_bool.shape
    valid_indices = [np.flatnonzero(mask_bool[b]) for b in range(batch)]
    counts = [int(len(idx)) for idx in valid_indices]
    max_valid = max(1, max(counts))

    if exact:
        target_len = min(orig_len, max_valid)
    else:
        sorted_buckets = sorted(int(b) for b in buckets if int(b) > 0)
        target_len = next((b for b in sorted_buckets if b >= max_valid), orig_len)
        target_len = min(orig_len, target_len)

    ctx_out = np.zeros((batch, target_len, *context.shape[2:]), dtype=context.dtype)
    mask_out = np.zeros((batch, target_len), dtype=np.bool_)
    for b, idx in enumerate(valid_indices):
        n = min(len(idx), target_len)
        if n > 0:
            ctx_out[b, :n] = context[b, idx[:n]]
            mask_out[b, :n] = True

    common_valid_len = counts[0] if (len(set(counts)) == 1 and counts[0] <= target_len) else None
    return ctx_out, mask_out, common_valid_len


def _resolve_splash_block_sizes(
    seq_len: int,
    block_q: int,
    block_kv: int,
    block_kv_compute: int,
) -> tuple[int, int, int, int, int]:
    """Resolve TPU-valid (128-aligned) asymmetric SplashAttention block sizes and padded lengths."""
    max_blk = max(128, roundup(seq_len, 128))
    eff_bq = min(max(128, (int(block_q) // 128) * 128), max_blk)
    eff_bkv = min(max(128, (int(block_kv) // 128) * 128), max_blk)
    eff_bkvc = min(max(128, (int(block_kv_compute) // 128) * 128), eff_bkv)
    while eff_bkv % eff_bkvc != 0 and eff_bkvc > 128:
        eff_bkvc -= 128
    q_padded = roundup(seq_len, eff_bq)
    kv_padded = roundup(seq_len, eff_bkv)
    return eff_bq, eff_bkv, eff_bkvc, q_padded, kv_padded


def _make_prefix_valid_mask(
    q_padded: int,
    kv_padded: int,
    valid_len: int,
    splash_attention_mask: Any,
) -> Any:
    """Build a lazy O(1)-memory compile-time prefix mask for ``kv < valid_len``.

    Avoids materializing or repeatedly hashing a dense ``(q_padded, kv_padded)`` boolean
    array (which is ~289 MiB at 2048x2048 and takes tens of seconds in ``NumpyMask``).
    Every block with ``kv_end <= valid_len`` returns all-True (``FULL``, zero VPU mask ops);
    every block with ``kv_start >= valid_len`` returns all-False (``EMPTY``, skipped); and
    allowing all query rows to attend to ``kv < valid_len`` avoids all-False softmax rows.
    """
    if valid_len >= kv_padded:
        return splash_attention_mask.FullMask((q_padded, kv_padded))

    base_cls = getattr(splash_attention_mask, "Mask", object)
    shape = (int(q_padded), int(kv_padded))
    vlen = int(valid_len)

    # Cache the dynamically bound subclass per splash_attention_mask module so __eq__/__hash__ match.
    cls = getattr(_make_prefix_valid_mask, "_cls_cache", {}).get(base_cls)
    if cls is None:

        class _PrefixValidMask(base_cls):
            def __init__(self, mask_shape: tuple[int, int], prefix_len: int):
                self._shape = (int(mask_shape[0]), int(mask_shape[1]))
                self._valid_len = int(prefix_len)

            @property
            def shape(self) -> tuple[int, int]:
                return self._shape

            def __getitem__(self, idx) -> np.ndarray:
                if len(idx) != 2:
                    raise NotImplementedError(f"Unsupported slice: {idx}")
                q_slice, kv_slice = idx
                if not isinstance(q_slice, slice) or not isinstance(kv_slice, slice):
                    raise NotImplementedError(f"Unsupported slice: {idx}")
                q_start, q_stop, q_step = q_slice.indices(self._shape[0])
                kv_start, kv_stop, kv_step = kv_slice.indices(self._shape[1])
                if q_step != 1 or kv_step != 1:
                    raise NotImplementedError(f"Unsupported stepped slice: {idx}")
                q_len = max(0, q_stop - q_start)
                kv_len = max(0, kv_stop - kv_start)
                if kv_stop <= self._valid_len:
                    return np.ones((q_len, kv_len), dtype=np.bool_)
                if kv_start >= self._valid_len:
                    return np.zeros((q_len, kv_len), dtype=np.bool_)
                out = np.zeros((q_len, kv_len), dtype=np.bool_)
                out[:, : self._valid_len - kv_start] = True
                return out

            def __eq__(self, other: object) -> bool:
                if not isinstance(other, _PrefixValidMask):
                    return NotImplemented
                return self._shape == other._shape and self._valid_len == other._valid_len

            def __hash__(self) -> int:
                return hash((_PrefixValidMask, self._shape, self._valid_len))

        cache = getattr(_make_prefix_valid_mask, "_cls_cache", None)
        if cache is None:
            cache = {}
            setattr(_make_prefix_valid_mask, "_cls_cache", cache)
        cache[base_cls] = _PrefixValidMask
        cls = _PrefixValidMask

    return cls(shape, vlen)


def _build_static_splash_mask(
    q_padded: int,
    kv_padded: int,
    valid_len: int,
    splash_attention_mask: Any,
) -> Any:
    """Build compile-time lazy static mask for contiguous valid prefix ``[0, valid_len)``."""
    return _make_prefix_valid_mask(q_padded, kv_padded, valid_len, splash_attention_mask)


def _splash_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    valid_mask: jnp.ndarray | None,
    *,
    block_q: int = 512,
    block_kv: int = 1024,
    block_kv_compute: int = 512,
    static_valid_len: int | None = None,
) -> jnp.ndarray:
    """Native GQA Pallas SplashAttention on ``[B, S, H_q, D]``, ``[B, S, H_kv, D]``.

    Note: ``q`` already has ``1 / sqrt(head_dim)`` pre-folded into ``qnorm``.
    Supports:
    - Asymmetric ``(block_q, block_kv, block_kv_compute)`` independent of GEMM sequence padding.
    - Compile-time static ``NumpyMask`` (``segment_ids=None``) when ``static_valid_len`` is given,
      eliminating per-tile VPU ``SegmentIds`` masking across 94-97% of image-to-image tiles.
    - Single-launch head-merged batching (``[B * H_q, S, D]``) when batch elements share a mask
      (e.g., Midjourney-style multi-seed batch generation), and ``jax.vmap`` otherwise.
    """
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel,
        splash_attention_mask,
    )

    batch, seq_len, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    eff_bq, eff_bkv, eff_bkvc, q_padded, kv_padded = _resolve_splash_block_sizes(
        seq_len, block_q, block_kv, block_kv_compute
    )
    q_pad = q_padded - seq_len
    kv_pad = kv_padded - seq_len

    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=eff_bq,
        block_kv=eff_bkv,
        block_kv_compute=eff_bkvc,
    )

    use_static_mask = (static_valid_len is not None) or (valid_mask is None)
    shared_mask = (batch == 1) or use_static_mask or (valid_mask is not None and valid_mask.shape[0] == 1)

    if use_static_mask:
        eff_valid = int(static_valid_len) if static_valid_len is not None else seq_len
        base_mask = _build_static_splash_mask(q_padded, kv_padded, eff_valid, splash_attention_mask)
    else:
        base_mask = splash_attention_mask.FullMask((q_padded, kv_padded))

    if shared_mask:
        # Merge batch into head dimension: (b * H_q + hq) // G == b * H_kv + (hq // G), exact 4:1 GQA!
        q_bh = jnp.transpose(q, (0, 2, 1, 3)).reshape(batch * num_q_heads, seq_len, head_dim)
        k_bh = jnp.transpose(k, (0, 2, 1, 3)).reshape(batch * num_kv_heads, seq_len, head_dim)
        v_bh = jnp.transpose(v, (0, 2, 1, 3)).reshape(batch * num_kv_heads, seq_len, head_dim)
        if q_pad > 0:
            q_bh = jnp.pad(q_bh, ((0, 0), (0, q_pad), (0, 0)))
        if kv_pad > 0:
            k_bh = jnp.pad(k_bh, ((0, 0), (0, kv_pad), (0, 0)))
            v_bh = jnp.pad(v_bh, ((0, 0), (0, kv_pad), (0, 0)))

        if use_static_mask:
            seg_ids = None
        else:
            assert valid_mask is not None
            vm = valid_mask[0]
            q_seg = (jnp.pad(vm, ((0, q_pad),), constant_values=False) if q_pad > 0 else vm).astype(jnp.int32)
            kv_seg = (jnp.pad(vm, ((0, kv_pad),), constant_values=False) if kv_pad > 0 else vm).astype(jnp.int32)
            seg_ids = splash_attention_kernel.SegmentIds(q=q_seg, kv=kv_seg)

        mask = splash_attention_mask.MultiHeadMask([base_mask for _ in range(batch * num_q_heads)])
        kernel = splash_attention_kernel.make_splash_mha(
            mask=mask,
            head_shards=1,
            q_seq_shards=1,
            block_sizes=block_sizes,
        )
        out_bh = kernel(q_bh, k_bh, v_bh, segment_ids=seg_ids)
        if q_pad > 0:
            out_bh = out_bh[:, :seq_len, :]
        out = out_bh.reshape(batch, num_q_heads, seq_len, head_dim)
        return jnp.transpose(out, (0, 2, 1, 3)).astype(v.dtype)

    # Heterogeneous masks across batch elements -> vmap a single kernel over batch.
    assert valid_mask is not None
    q_bhsd = jnp.transpose(q, (0, 2, 1, 3))
    k_bhsd = jnp.transpose(k, (0, 2, 1, 3))
    v_bhsd = jnp.transpose(v, (0, 2, 1, 3))
    if q_pad > 0:
        q_bhsd = jnp.pad(q_bhsd, ((0, 0), (0, 0), (0, q_pad), (0, 0)))
    if kv_pad > 0:
        k_bhsd = jnp.pad(k_bhsd, ((0, 0), (0, 0), (0, kv_pad), (0, 0)))
        v_bhsd = jnp.pad(v_bhsd, ((0, 0), (0, 0), (0, kv_pad), (0, 0)))

    q_seg_2d = (
        jnp.pad(valid_mask, ((0, 0), (0, q_pad)), constant_values=False) if q_pad > 0 else valid_mask
    ).astype(jnp.int32)
    kv_seg_2d = (
        jnp.pad(valid_mask, ((0, 0), (0, kv_pad)), constant_values=False) if kv_pad > 0 else valid_mask
    ).astype(jnp.int32)
    seg_ids_2d = splash_attention_kernel.SegmentIds(q=q_seg_2d, kv=kv_seg_2d)

    mask = splash_attention_mask.MultiHeadMask([base_mask for _ in range(num_q_heads)])
    kernel = splash_attention_kernel.make_splash_mha(
        mask=mask,
        head_shards=1,
        q_seq_shards=1,
        block_sizes=block_sizes,
    )
    out_bhsd = jax.vmap(
        kernel,
        in_axes=(0, 0, 0, splash_attention_kernel.SegmentIds(0, 0)),
    )(q_bhsd, k_bhsd, v_bhsd, seg_ids_2d)
    if q_pad > 0:
        out_bhsd = out_bhsd[:, :, :seq_len, :]
    return jnp.transpose(out_bhsd, (0, 2, 1, 3)).astype(v.dtype)


def _pallas_flash_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    valid_mask: jnp.ndarray | None,
    *,
    block_q: int = 512,
    block_kv: int = 512,
) -> jnp.ndarray:
    """TPU Pallas FlashAttention on ``[B, S, H_q, D]``, ``[B, S, H_kv, D]`` (Q pre-scaled)."""
    from jax.experimental.pallas.ops.tpu import flash_attention as pallas_fa

    batch, seq_len, num_q_heads, _ = q.shape
    num_kv_heads = k.shape[2]
    eff_bq, eff_bkv, _, q_padded, kv_padded = _resolve_splash_block_sizes(seq_len, block_q, block_kv, block_kv)
    eff_block = min(eff_bq, eff_bkv)
    padded_len = max(q_padded, kv_padded)
    padded_len = roundup(padded_len, eff_block)
    pad = padded_len - seq_len

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
            vm = jnp.broadcast_to(valid_mask, (batch, seq_len))
            valid_padded = jnp.pad(vm, ((0, 0), (0, pad)), constant_values=False)
        else:
            valid_padded = jnp.pad(
                jnp.ones((batch, seq_len), dtype=jnp.bool_), ((0, 0), (0, pad)), constant_values=False
            )
    else:
        valid_padded = jnp.broadcast_to(valid_mask, (batch, seq_len)) if valid_mask is not None else None

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
        sm_scale=1.0,
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
    splash_block_q: int = 512,
    splash_block_kv: int = 1024,
    splash_block_kv_compute: int = 512,
    static_valid_len: int | None = None,
) -> jnp.ndarray:
    """Multi-head / Grouped-Query Attention: ``[B, S, H_q, D]`` -> ``[B, S, H_q * D]`` (Q pre-scaled)."""
    batch, seq_len, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    out_dim = num_q_heads * head_dim

    if attn_impl == "pallas_gqa":
        return _pallas_gqa(
            (q.astype(jnp.float32) * _LOG2_E).astype(q.dtype).transpose(0, 2, 1, 3),
            k.transpose(0, 2, 1, 3),
            v.transpose(0, 2, 1, 3),
            valid_mask,
            splash_block_q,
            splash_block_kv,
            splash_block_kv_compute,
            static_valid_len,
        )

    if attn_impl == "splash":
        out = _splash_attention(
            q,
            k,
            v,
            valid_mask,
            block_q=splash_block_q,
            block_kv=splash_block_kv,
            block_kv_compute=splash_block_kv_compute,
            static_valid_len=static_valid_len,
        )
        return out.reshape(batch, seq_len, out_dim)

    if attn_impl == "flash":
        out = _pallas_flash_attention(
            q,
            k,
            v,
            valid_mask,
            block_q=splash_block_q,
            block_kv=splash_block_kv,
        )
        return out.reshape(batch, seq_len, out_dim)

    if attn_impl == "xla":
        mask_4d = valid_mask[:, None, None, :] if valid_mask is not None else None
        out = jax.nn.dot_product_attention(q, k, v, mask=mask_4d, scale=1.0, implementation="xla")
        return out.astype(v.dtype).reshape(batch, seq_len, out_dim)

    if attn_impl == "einsum":
        group_size = num_q_heads // num_kv_heads
        q_g = q.astype(jnp.float32).reshape(batch, seq_len, num_kv_heads, group_size, head_dim)
        k_f = k.astype(jnp.float32)
        v_f = v.astype(jnp.float32)
        logits = jnp.einsum("bskgd,btkd->bkgst", q_g, k_f)
        if valid_mask is not None:
            logits = jnp.where(valid_mask[:, None, None, None, :], logits, jnp.float32(-1e30))
        weights = jax.nn.softmax(logits, axis=-1)
        out = jnp.einsum("bkgst,btkd->bskgd", weights, v_f).astype(v.dtype)
        return out.reshape(batch, seq_len, out_dim)

    raise ValueError(f"Unknown attn_impl={attn_impl!r}; expected one of {VALID_ATTN_IMPLS}.")


def _pallas_gqa(q, k, v, valid_mask, block_q, block_kv, block_kv_compute, static_valid_len):
    from tpu_jax.pallas_gqa_attn import pallas_gqa_attention

    bq, bkv, bkvc, _, _ = _resolve_splash_block_sizes(q.shape[2], block_q, block_kv, block_kv_compute)
    return pallas_gqa_attention(
        q, k, v, valid_mask, block_q=bq, block_kv=bkv, block_kv_compute=bkvc, static_valid_len=static_valid_len
    )


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
    splash_block_q: int,
    splash_block_kv: int,
    splash_block_kv_compute: int,
    static_valid_len: int | None = None,
    tp_size: int = 1,
) -> jnp.ndarray:
    """Fused contiguous QKVGate -> QK-Norm -> half-split RoPE -> GQA -> Sigmoid-Gate -> Wo."""
    batch, seq_len, _ = y.shape
    qkvg = _matmul(y, block["qkvgate_w"], block.get("qkvgate_s"))
    if tp_size <= 1:
        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        q = qkvg[..., :q_dim].reshape(batch, seq_len, num_heads, head_dim)
        k = qkvg[..., q_dim : q_dim + kv_dim].reshape(batch, seq_len, num_kv_heads, head_dim)
        v = qkvg[..., q_dim + kv_dim : q_dim + 2 * kv_dim].reshape(batch, seq_len, num_kv_heads, head_dim)
        gate = qkvg[..., q_dim + 2 * kv_dim :]
    else:
        q_s = (num_heads // tp_size) * head_dim
        kv_s = (num_kv_heads // tp_size) * head_dim
        per_s = 2 * q_s + 2 * kv_s
        qkvg_s = qkvg.reshape(batch, seq_len, tp_size, per_s)
        q = qkvg_s[..., :q_s].reshape(batch, seq_len, num_heads, head_dim)
        k = qkvg_s[..., q_s : q_s + kv_s].reshape(batch, seq_len, num_kv_heads, head_dim)
        v = qkvg_s[..., q_s + kv_s : q_s + 2 * kv_s].reshape(batch, seq_len, num_kv_heads, head_dim)
        gate = qkvg_s[..., q_s + 2 * kv_s :].reshape(batch, seq_len, num_heads * head_dim)

    grouped = attn_impl == "pallas_gqa"
    qnorm = block["qnorm"]
    if grouped:
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        qnorm = block["qnorm_log2"] if "qnorm_log2" in block else qnorm * _LOG2_E
    q = _rms_norm(q, qnorm, norm_eps)
    k = _rms_norm(k, block["knorm"], norm_eps)
    if cos is not None and sin is not None:
        q = _apply_rope(q, cos, sin, head_first=grouped)
        k = _apply_rope(k, cos, sin, head_first=grouped)

    if grouped:
        attn_out = _pallas_gqa(
            q, k, v, valid_mask, splash_block_q, splash_block_kv, splash_block_kv_compute, static_valid_len
        )
    else:
        attn_out = _gqa_attention(
            q,
            k,
            v,
            valid_mask,
            attn_impl=attn_impl,
            splash_block_q=splash_block_q,
            splash_block_kv=splash_block_kv,
            splash_block_kv_compute=splash_block_kv_compute,
            static_valid_len=static_valid_len,
        )
    gated = attn_out * jax.nn.sigmoid(gate)
    return _matmul(gated, block["wo_w"], block.get("wo_s"))


def _swiglu_op(y: jnp.ndarray, block: dict[str, jnp.ndarray], *, tp_size: int = 1) -> jnp.ndarray:
    """Fused contiguous Gate+Up SwiGLU MLP (zero stride-2 minor-dim shuffle)."""
    gu = _matmul(y, block["mlp_gate_up_w"], block.get("mlp_gate_up_s"))
    if tp_size <= 1:
        gate, up = jnp.split(gu, 2, axis=-1)
    else:
        gu_s = gu.reshape(*y.shape[:-1], tp_size, 2, -1)
        gate = gu_s[..., 0, :].reshape(*y.shape[:-1], -1)
        up = gu_s[..., 1, :].reshape(*y.shape[:-1], -1)
    return _matmul(jax.nn.silu(gate) * up, block["mlp_down_w"], block.get("mlp_down_s"))


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
    # TextFusion runs once per prompt on tiny S (12 or <=512); xla/einsum avoids Pallas pad overhead.
    eff_impl = "xla" if attn_impl in ("splash", "flash", "pallas_gqa") else attn_impl
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
        splash_block_q=128,
        splash_block_kv=128,
        splash_block_kv_compute=128,
        static_valid_len=None,
        tp_size=1,
    )
    h = _rms_norm(x, block["postnorm"], norm_eps)
    return x + _swiglu_op(h, block, tp_size=1)


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
    splash_block_q: int,
    splash_block_kv: int,
    splash_block_kv_compute: int,
    static_valid_len: int | None = None,
    tp_size: int = 1,
) -> jnp.ndarray:
    """Single Krea 2 ``SingleStreamBlock`` forward (``jax.lax.scan`` body)."""
    w_dtype = block["mod_lin"].dtype
    mod_out = tvec + block["mod_lin"][None, None, :]
    prescale, preshift, pregate, postscale, postshift, postgate = jnp.split(mod_out, 6, axis=-1)

    y = _modulated_rms_norm(x, block["prenorm"], prescale, preshift, norm_eps, out_dtype=w_dtype)
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
        splash_block_q=splash_block_q,
        splash_block_kv=splash_block_kv,
        splash_block_kv_compute=splash_block_kv_compute,
        static_valid_len=static_valid_len,
        tp_size=tp_size,
    )
    x = x + pregate * attn_out

    y = _modulated_rms_norm(x, block["postnorm"], postscale, postshift, norm_eps, out_dtype=w_dtype)
    mlp_out = _swiglu_op(y, block, tp_size=tp_size)
    return x + postgate * mlp_out


def build_positions_and_mask(
    batch: int,
    latent_h: int,
    latent_w: int,
    txtmask: jnp.ndarray,
    *,
    patch: int = 2,
    pad_multiple: int = 256,
    img_first: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """Build padded ``(pos, valid_mask, pad_len)`` matching ``sampling.prepare`` + ``mmdit.py``.

    When ``img_first=True`` (default), places aligned image tokens at ``[0, img_len)``
    followed by text tokens at ``[img_len, img_len + txt_len)`` so the entire image
    region forms a contiguous valid prefix for static SplashAttention block skipping.
    """
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

    img_mask = jnp.ones((batch, img_len), dtype=jnp.bool_)
    txt_mask_b = txtmask.astype(jnp.bool_)
    if img_first:
        pos = jnp.concatenate([img_pos, txt_pos], axis=1)
        valid_mask = jnp.concatenate([img_mask, txt_mask_b], axis=1)
    else:
        pos = jnp.concatenate([txt_pos, img_pos], axis=1)
        valid_mask = jnp.concatenate([txt_mask_b, img_mask], axis=1)

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
    gemm_pad_multiple: int = 256,
    img_first: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """Run TextFusion + TextMLP + RoPE ONCE before the denoising loop.

    Decouples GEMM sequence alignment (``gemm_pad_multiple=256`` for v6e 256x256 MXU)
    from SplashAttention tile sizes so large attention tiles never waste GEMM FLOPs.

    Returns ``(text_tokens, rope_cos, rope_sin, joint_valid_mask, pad_len)``.
    """
    del splash_block_size  # Kept for API compatibility; GEMM alignment is decoupled.
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

    # 5. Pad GEMM sequence only to 256-aligned MXU boundary (independent of Splash tile size).
    pos, valid_mask, pad_len = build_positions_and_mask(
        b,
        latent_h,
        latent_w,
        txtmask_bool,
        patch=config.patch,
        pad_multiple=gemm_pad_multiple,
        img_first=img_first,
    )
    rope_cos, rope_sin = compute_rope_cos_sin(pos, params["rope_omegas"])
    return text_tokens, rope_cos, rope_sin, valid_mask, pad_len


def _axis_shard_count(arr: Any, axis: int) -> int | None:
    """Infer how many shards partition ``arr`` along ``axis`` from its JAX sharding."""
    sharding = getattr(arr, "sharding", None)
    if sharding is None:
        aval = getattr(arr, "aval", None)
        sharding = getattr(aval, "sharding", None)
    if sharding is None:
        return None
    shape = getattr(arr, "shape", None)
    if shape is None:
        return None
    ax = axis if axis >= 0 else len(shape) + axis
    shard_shape_fn = getattr(sharding, "shard_shape", None)
    if callable(shard_shape_fn):
        try:
            s_shape = shard_shape_fn(tuple(shape))
            if s_shape[ax] > 0:
                return int(shape[ax] // s_shape[ax])
        except Exception:
            pass
    mesh = getattr(sharding, "mesh", None)
    spec = getattr(sharding, "spec", None)
    if mesh is not None and spec is not None and ax < len(spec):
        ax_spec = spec[ax]
        if ax_spec is None:
            return 1
        names = (ax_spec,) if isinstance(ax_spec, str) else tuple(ax_spec)
        prod = 1
        for name in names:
            if name is not None:
                prod *= int(mesh.shape[name])
        return prod
    return None


def infer_and_validate_tp_size(params: dict[str, Any], tp_size: int | None = None) -> int:
    """Infer and validate TP layout shard count from sharded DiT block weights.

    Prevents silent numerical corruption when weights are loaded for a multi-chip TP
    mesh (packing fused ``qkvgate_w`` / ``mlp_gate_up_w`` per TP shard) while a caller
    leaves ``tp_size`` defaulted to 1.
    """
    blocks = params.get("blocks")
    inferred: dict[str, int] = {}
    if isinstance(blocks, dict):
        for key, ax in (
            ("qkvgate_w", -1),
            ("mlp_gate_up_w", -1),
            ("wo_w", -2),
            ("mlp_down_w", -2),
        ):
            if key in blocks:
                cnt = _axis_shard_count(blocks[key], ax)
                if cnt is not None:
                    inferred[key] = cnt

    unique_counts = set(inferred.values())
    if len(unique_counts) > 1:
        raise ValueError(
            f"Inconsistent TP sharding across DiT block weights: {inferred}. "
            "Ensure all fused projections are sharded with the same 1D TP mesh."
        )
    sharded_tp = next(iter(unique_counts)) if unique_counts else 1

    if tp_size is None:
        return sharded_tp

    req_tp = int(tp_size)
    if req_tp < 1:
        raise ValueError(f"tp_size must be >= 1, got {req_tp}.")
    if sharded_tp > 1 and req_tp != sharded_tp:
        raise ValueError(
            f"DiT weights are sharded for tp_size={sharded_tp} ({inferred}), "
            f"which does not match requested tp_size={req_tp}. "
            "Pass tp_size=None (auto) or the matching mesh TP size."
        )
    return req_tp


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
    splash_block_q: int | None = None,
    splash_block_kv: int | None = None,
    splash_block_kv_compute: int | None = None,
    static_valid_len: int | None = None,
    scan_unroll: int = 1,
    img_first: bool = True,
    tp_size: int | None = None,
) -> jnp.ndarray:
    """Single Krea 2 DiT forward pass using precomputed text conditioning and RoPE.

    Supports broadcasting ``text_tokens`` / ``rope_cos`` / ``rope_sin`` / ``valid_mask``
    from batch=1 across ``batch > 1`` ``img_tokens`` (multi-seed batch generation).
    When ``tp_size=None`` (default), infers TP layout automatically from ``params`` sharding.
    """
    eff_tp_size = infer_and_validate_tp_size(params, tp_size)
    w_dtype = params["first_w"].dtype
    batch, img_len, _ = img_tokens.shape
    txt_len = text_tokens.shape[1]

    bq = splash_block_size if splash_block_q is None else splash_block_q
    bkv = splash_block_size if splash_block_kv is None else splash_block_kv
    bkvc = min(bq, bkv) if splash_block_kv_compute is None else splash_block_kv_compute

    # Timestep embedding -> tmlp -> tproj (stays [1, 1, *] when timestep is scalar -> broadcasts over batch)
    t_arr = jnp.asarray(timestep, dtype=jnp.float32).reshape(-1)
    t_b = t_arr if t_arr.shape[0] in (1, batch) else jnp.broadcast_to(t_arr, (batch,))
    t_in = _temb(t_b, config.tdim, params["time_freqs"], out_dtype=w_dtype)
    t_h = jax.nn.gelu(_matmul(t_in, params["tmlp_fc1_w"]) + params["tmlp_fc1_b"], approximate=True)
    t_emb = _matmul(t_h, params["tmlp_fc2_w"]) + params["tmlp_fc2_b"]
    tvec = _matmul(jax.nn.gelu(t_emb, approximate=True), params["tproj_w"]) + params["tproj_b"]

    img_emb = _matmul(img_tokens.astype(w_dtype), params["first_w"]) + params["first_b"]
    if text_tokens.shape[0] == 1 and batch > 1:
        txt_b = jnp.broadcast_to(text_tokens, (batch, txt_len, text_tokens.shape[-1]))
    else:
        txt_b = text_tokens

    if img_first:
        combined = jnp.concatenate([img_emb, txt_b], axis=1)
    else:
        combined = jnp.concatenate([txt_b, img_emb], axis=1)

    if pad_len > 0:
        combined = jnp.pad(combined, ((0, 0), (0, pad_len), (0, 0)))

    block_fn = partial(
        _single_stream_block_forward,
        num_heads=config.heads,
        num_kv_heads=config.kvheads,
        head_dim=config.head_dim,
        norm_eps=config.norm_eps,
        attn_impl=attn_impl,
        splash_block_q=bq,
        splash_block_kv=bkv,
        splash_block_kv_compute=bkvc,
        static_valid_len=static_valid_len if img_first else None,
        tp_size=eff_tp_size,
    )

    def _scan_body(carry: jnp.ndarray, blk: dict[str, jnp.ndarray]):
        return block_fn(carry, blk, tvec, rope_cos, rope_sin, valid_mask), None

    combined, _ = lax.scan(_scan_body, combined, params["blocks"], unroll=max(1, int(scan_unroll)))

    # Slice image tokens BEFORE LastLayer so we skip redundant final norm+linear on text+pad!
    img_out = combined[:, :img_len, :] if img_first else combined[:, txt_len : txt_len + img_len, :]
    mod_last = t_emb + params["last_mod_lin"][None, :, :]
    scale, shift = jnp.split(mod_last, 2, axis=1)
    img_out = _modulated_rms_norm(img_out, params["last_norm"], scale, shift, config.norm_eps, out_dtype=w_dtype)
    return _matmul(img_out, params["last_linear_w"]) + params["last_linear_b"]


def patchify_latents(latents: jnp.ndarray, patch: int = 2) -> jnp.ndarray:
    """``[B, C, H, W]`` -> ``[B, (H/p)*(W/p), C*p*p]`` matching ``sampling.prepare``."""
    b, c, h, w = latents.shape
    hp, wp = h // patch, w // patch
    x = latents.reshape(b, c, hp, patch, wp, patch)
    x = jnp.transpose(x, (0, 2, 4, 1, 3, 5))
    return x.reshape(b, hp * wp, c * patch * patch)


def unpatchify_latents(
    tokens: jnp.ndarray,
    latent_h: int,
    latent_w: int,
    channels: int = 16,
    patch: int = 2,
) -> jnp.ndarray:
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
    splash_block_q: int | None = None,
    splash_block_kv: int | None = None,
    splash_block_kv_compute: int | None = None,
    pos_static_valid_len: int | None = None,
    neg_static_valid_len: int | None = None,
    scan_unroll: int = 1,
    gemm_pad_multiple: int = 256,
    img_first: bool = True,
    tp_size: int | None = None,
    latent_dtype: Any = jnp.float32,
    donate_latents: bool = True,
):
    """Build a single ``jax.jit``-compiled full N-step Euler denoising function.

    - ``guidance_scale == 0.0`` (Turbo default): single forward pass per step,
      zero unconditional conditioning work.
    - ``guidance_scale > 0.0`` (Raw / CFG mode): runs cond + uncond with
      ``v = cond + guidance_scale * (cond - uncond)`` (exact match to ``sampling.py``).
    - Supports ``init_latents [B, C, H, W]`` with ``pos_context [1, L, 12, D]``
      (runs TextFusion once and broadcasts across all ``B`` seeds!).
    - When ``tp_size=None`` (default), infers and validates the TP weight layout
      from ``params`` sharding before execution.
    """
    align = 8 * config.patch
    w_aligned = roundup(width, align)
    h_aligned = roundup(height, align)
    latent_h = h_aligned // 8
    latent_w = w_aligned // 8
    do_cfg = guidance_scale > 0.0
    donate = (1,) if donate_latents else ()
    compiled_by_tp: dict[int, Any] = {}

    def _get_jitted(resolved_tp: int):
        fn = compiled_by_tp.get(resolved_tp)
        if fn is not None:
            return fn

        @partial(jax.jit, donate_argnums=donate)
        def _denoise_inner(
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
                gemm_pad_multiple=gemm_pad_multiple,
                img_first=img_first,
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
                    gemm_pad_multiple=gemm_pad_multiple,
                    img_first=img_first,
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
                    splash_block_q=splash_block_q,
                    splash_block_kv=splash_block_kv,
                    splash_block_kv_compute=splash_block_kv_compute,
                    static_valid_len=pos_static_valid_len,
                    scan_unroll=scan_unroll,
                    img_first=img_first,
                    tp_size=resolved_tp,
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
                        splash_block_q=splash_block_q,
                        splash_block_kv=splash_block_kv,
                        splash_block_kv_compute=splash_block_kv_compute,
                        static_valid_len=neg_static_valid_len,
                        scan_unroll=scan_unroll,
                        img_first=img_first,
                        tp_size=resolved_tp,
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

        compiled_by_tp[resolved_tp] = _denoise_inner
        return _denoise_inner

    def _denoise(
        params: dict[str, Any],
        init_latents: jnp.ndarray,
        pos_context: jnp.ndarray,
        pos_txtmask: jnp.ndarray,
        neg_context: jnp.ndarray | None,
        neg_txtmask: jnp.ndarray | None,
        ts: jnp.ndarray,
    ) -> jnp.ndarray:
        resolved_tp = infer_and_validate_tp_size(params, tp_size)
        return _get_jitted(resolved_tp)(
            params, init_latents, pos_context, pos_txtmask, neg_context, neg_txtmask, ts
        )

    return _denoise
