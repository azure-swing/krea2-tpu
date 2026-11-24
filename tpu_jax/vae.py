"""Pure JAX/XLA implementation of AutoencoderKLQwenImage decoder (T=1 2D-reduced).

Mathematical exactness for single-image (T=1) decoding:
- Every ``QwenImageCausalConv3d`` pads temporally by ``(2, 0)`` on ``T=1``, giving
  ``[0, 0, x_0]``. Convolving a ``[out, in, 3, 3, 3]`` kernel over ``[0, 0, x_0]``
  is mathematically identical to a 2D Conv3x3 using ONLY the last temporal slice
  ``weight[:, :, -1, :, :]``.
- In ``QwenImageResample(mode="upsample3d")``, on the first frame (``i == 0``),
  ``feat_cache[idx]`` is ``None`` so ``time_conv`` is skipped; only 2x nearest
  spatial upsample + 2D Conv3x3 executes.
- All convolutions run natively in TPU-optimal **NHWC** layout.
- Mid-block single-head self-attention uses chunked online softmax via
  ``jax.lax.scan`` over query blocks (default chunk 1024), avoiding the 16 GB
  ``65536 x 65536`` dense attention matrix at 2048x2048 resolution!
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from tpu_jax.config import QwenImageVAEConfig


def _vae_rms_norm(x: jnp.ndarray, gamma: jnp.ndarray) -> jnp.ndarray:
    """Exact match to ``QwenImageRMS_norm`` in NHWC layout.

    PyTorch reference: ``F.normalize(x, dim=1) * sqrt(C) * gamma`` (L2 eps=1e-12).
    """
    orig_dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    c = x.shape[-1]
    l2 = jnp.sqrt(jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True))
    normed = x_f32 / jnp.maximum(l2, jnp.float32(1e-12))
    out = normed * jnp.float32(math.sqrt(c)) * gamma.astype(jnp.float32)
    return out.astype(orig_dtype)


def _conv2d_nhwc(
    x: jnp.ndarray,
    weight_hwio: jnp.ndarray,
    bias: jnp.ndarray,
    *,
    padding: str = "SAME",
) -> jnp.ndarray:
    out = lax.conv_general_dilated(
        lhs=x,
        rhs=weight_hwio,
        window_strides=(1, 1),
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        preferred_element_type=jnp.float32,
    ).astype(x.dtype)
    return out + bias.astype(x.dtype)


def _nearest_upsample_2x_nhwc(x: jnp.ndarray) -> jnp.ndarray:
    """Exact 2x nearest-neighbor spatial upsample in NHWC (matches ``nearest-exact``)."""
    b, h, w, c = x.shape
    x = jnp.broadcast_to(x[:, :, None, :, None, :], (b, h, 2, w, 2, c))
    return x.reshape(b, h * 2, w * 2, c)


def _chunked_single_head_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    *,
    q_chunk_size: int = 1024,
) -> jnp.ndarray:
    """Memory-efficient exact single-head attention on ``[B, S, C]`` via ``lax.scan``."""
    b, s, c = q.shape
    scale = jnp.float32(1.0 / math.sqrt(c))

    if s <= q_chunk_size:
        q_f = q.astype(jnp.float32) * scale
        k_f = k.astype(jnp.float32)
        v_f = v.astype(jnp.float32)
        logits = jnp.einsum("bqc,bkc->bqk", q_f, k_f)
        weights = jax.nn.softmax(logits, axis=-1)
        return jnp.einsum("bqk,bkc->bqc", weights, v_f).astype(v.dtype)

    pad = (-s) % q_chunk_size
    if pad > 0:
        q_padded = jnp.pad(q, ((0, 0), (0, pad), (0, 0)))
    else:
        q_padded = q
    num_chunks = q_padded.shape[1] // q_chunk_size
    q_chunks = q_padded.reshape(b, num_chunks, q_chunk_size, c)
    q_chunks = jnp.transpose(q_chunks, (1, 0, 2, 3))  # [num_chunks, B, chunk, C]

    k_f = k.astype(jnp.float32)
    v_f = v.astype(jnp.float32)

    def _chunk_body(_carry, q_blk):
        q_f = q_blk.astype(jnp.float32) * scale
        logits = jnp.einsum("bqc,bkc->bqk", q_f, k_f)
        weights = jax.nn.softmax(logits, axis=-1)
        out_blk = jnp.einsum("bqk,bkc->bqc", weights, v_f).astype(v.dtype)
        return None, out_blk

    _, out_chunks = lax.scan(_chunk_body, None, q_chunks)
    out = jnp.transpose(out_chunks, (1, 0, 2, 3)).reshape(b, num_chunks * q_chunk_size, c)
    if pad > 0:
        out = out[:, :s, :]
    return out


def _resblock_forward(x: jnp.ndarray, blk: dict[str, jnp.ndarray]) -> jnp.ndarray:
    h = x
    if "shortcut_w" in blk:
        h = _conv2d_nhwc(h, blk["shortcut_w"], blk["shortcut_b"], padding="VALID")

    x = jax.nn.silu(_vae_rms_norm(x, blk["norm1"]))
    x = _conv2d_nhwc(x, blk["conv1_w"], blk["conv1_b"], padding="SAME")
    x = jax.nn.silu(_vae_rms_norm(x, blk["norm2"]))
    x = _conv2d_nhwc(x, blk["conv2_w"], blk["conv2_b"], padding="SAME")
    return x + h


def _mid_attention_forward(
    x: jnp.ndarray,
    attn: dict[str, jnp.ndarray],
    *,
    q_chunk_size: int = 1024,
) -> jnp.ndarray:
    b, h, w, c = x.shape
    identity = x
    y = _vae_rms_norm(x, attn["norm"])
    qkv = _conv2d_nhwc(y, attn["to_qkv_w"], attn["to_qkv_b"], padding="VALID")
    qkv = qkv.reshape(b, h * w, 3 * c)
    q, k, v = jnp.split(qkv, 3, axis=-1)
    out = _chunked_single_head_attention(q, k, v, q_chunk_size=q_chunk_size)
    out = out.reshape(b, h, w, c)
    out = _conv2d_nhwc(out, attn["proj_w"], attn["proj_b"], padding="VALID")
    return identity + out


def vae_decode_nhwc(
    params: dict[str, Any],
    config: QwenImageVAEConfig,
    latents_nchw: jnp.ndarray,
    *,
    q_chunk_size: int = 1024,
) -> jnp.ndarray:
    """Decode ``latents [B, 16, H, W]`` to clamped ``[-1, 1]`` pixels ``[B, H*8, W*8, 3]``."""
    w_dtype = params["conv_in_w"].dtype
    mean = jnp.asarray(config.latents_mean, dtype=jnp.float32).reshape(1, 1, 1, -1)
    std = jnp.asarray(config.latents_std, dtype=jnp.float32).reshape(1, 1, 1, -1)

    # NCHW -> NHWC and denormalize (matching autoencoder.py: (x * std) + mean)
    x = jnp.transpose(latents_nchw.astype(jnp.float32), (0, 2, 3, 1))
    x = (x * std + mean).astype(w_dtype)

    # post_quant_conv (1x1)
    x = _conv2d_nhwc(x, params["post_quant_conv_w"], params["post_quant_conv_b"], padding="VALID")
    # decoder.conv_in (3x3)
    x = _conv2d_nhwc(x, params["conv_in_w"], params["conv_in_b"], padding="SAME")

    # mid_block: resnet0 -> attn0 -> resnet1
    x = _resblock_forward(x, params["mid_resnets"][0])
    x = _mid_attention_forward(x, params["mid_attn"], q_chunk_size=q_chunk_size)
    x = _resblock_forward(x, params["mid_resnets"][1])

    # up_blocks (4 stages)
    for up_blk in params["up_blocks"]:
        for resnet in up_blk["resnets"]:
            x = _resblock_forward(x, resnet)
        if "upsample_w" in up_blk:
            x = _nearest_upsample_2x_nhwc(x)
            x = _conv2d_nhwc(x, up_blk["upsample_w"], up_blk["upsample_b"], padding="SAME")

    # head: norm_out -> silu -> conv_out -> clamp(-1, 1)
    x = jax.nn.silu(_vae_rms_norm(x, params["norm_out"]))
    x = _conv2d_nhwc(x, params["conv_out_w"], params["conv_out_b"], padding="SAME")
    return jnp.clip(x, -1.0, 1.0)


def make_vae_decode_fn(config: QwenImageVAEConfig, *, q_chunk_size: int = 1024):
    """Return a ``jax.jit``-compiled VAE decoder returning uint8-ready ``[B, H*8, W*8, 3]`` in ``[-1, 1]``."""

    @jax.jit
    def _decode(params: dict[str, Any], latents_nchw: jnp.ndarray) -> jnp.ndarray:
        return vae_decode_nhwc(params, config, latents_nchw, q_chunk_size=q_chunk_size)

    return _decode
