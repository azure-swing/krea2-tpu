"""Forward-only grouped attention for the Krea 2 DiT on TPU v6e."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_jax.scheduler import roundup


_HEAD_DIM = 128
_GROUP_SIZE = 4
_DOT_NT = (((1,), (1,)), ((), ()))


def _gqa_kernel(
    q_ref, k_ref, v_ref, mask_ref, out_ref, m_ref, l_ref, o_ref, *, compute_kv, tail, dynamic_mask
):
    kv_step = pl.program_id(2)
    last = kv_step == pl.num_programs(2) - 1
    q = q_ref[...]
    q_rows = q.shape[0]
    kv_tile = k_ref.shape[0]
    lane_repeats = compute_kv // _HEAD_DIM
    precision = lax.Precision.HIGHEST if q.dtype == jnp.float32 else None

    @pl.when(kv_step == 0)
    def initialize():
        m_ref[...] = jnp.full(m_ref.shape, -jnp.inf, dtype=jnp.float32)
        l_ref[...] = jnp.zeros(l_ref.shape, dtype=jnp.float32)
        o_ref[...] = jnp.zeros(o_ref.shape, dtype=jnp.float32)

    def logits_at(start, valid):
        tile = pl.ds(start, compute_kv)
        logits = lax.dot_general(
            q, k_ref[tile, :], _DOT_NT, precision=precision, preferred_element_type=jnp.float32
        )
        if dynamic_mask:
            logits = jnp.where(mask_ref[:, tile] != 0, logits, -jnp.inf)
        elif valid is not None and start + compute_kv > valid:
            k_index = lax.broadcasted_iota(jnp.int32, logits.shape, 1)
            logits = jnp.where(k_index < valid - start, logits, -jnp.inf)
        return logits

    def prepare(logits, m, denom):
        tile_max = jnp.max(logits, axis=1, keepdims=True)
        next_m = jnp.maximum(m, jnp.broadcast_to(tile_max, (q_rows, _HEAD_DIM)))
        # A dynamic mask may leave a row empty across several subtiles.
        safe_m = jnp.where(next_m == -jnp.inf, 0.0, next_m)
        alpha = jnp.where(m == next_m, 1.0, jnp.exp2(m - safe_m))
        p = jnp.exp2(logits - jnp.tile(safe_m, (1, lane_repeats)))
        tile_sum = jnp.sum(p, axis=1, keepdims=True)
        next_denom = denom * alpha + jnp.broadcast_to(tile_sum, (q_rows, _HEAD_DIM))
        return next_m, next_denom, p.astype(v_ref.dtype), alpha

    def project(p, start):
        # V arrives as [D, KV]. Both dot operands contract along their minor axis.
        return lax.dot_general(
            p,
            v_ref[:, pl.ds(start, compute_kv)],
            _DOT_NT,
            precision=precision,
            preferred_element_type=jnp.float32,
        )

    def accumulate(valid):
        n_subtiles = kv_tile // compute_kv if valid is None else (valid + compute_kv - 1) // compute_kv
        m, denom, p, alpha = prepare(logits_at(0, valid), m_ref[...], l_ref[...])
        numerator = o_ref[...]
        for subtile in range(1, n_subtiles):
            start = subtile * compute_kv
            # Alternate the pending P/alpha and the next QK/softmax. These two
            # matmuls are independent of the next vector update until it consumes QK.
            logits = logits_at(start, valid)
            pv = project(p, start - compute_kv)
            m, denom, p, next_alpha = prepare(logits, m, denom)
            numerator = alpha * numerator + pv
            alpha = next_alpha
        pv = project(p, (n_subtiles - 1) * compute_kv)
        return m, denom, alpha * numerator + pv

    if tail:
        m, denom, numerator = lax.cond(last, lambda: accumulate(tail), lambda: accumulate(None))
    else:
        m, denom, numerator = accumulate(None)

    @pl.when(last)
    def finish():
        out_ref[...] = (numerator / jnp.where(denom > 0, denom, 1.0)).astype(out_ref.dtype)

    @pl.when(~last)
    def save():
        m_ref[...] = m
        l_ref[...] = denom
        o_ref[...] = numerator


def pallas_gqa_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    valid_mask: jax.Array | None,
    *,
    block_q: int,
    block_kv: int,
    block_kv_compute: int,
    static_valid_len: int | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Return [B, S, Hq*D] from base-2 prescaled Q and unscaled K/V, all [B, H, S, D]."""
    batch, q_heads, seq_len, head_dim = q.shape
    kv_heads = k.shape[1]
    if head_dim != _HEAD_DIM or q_heads != _GROUP_SIZE * kv_heads:
        raise ValueError("pallas_gqa requires head_dim=128 and four query heads per KV head.")

    # block_q is in tokens per head. Four aligned query tiles share each K/V DMA.
    q_rows = block_q * _GROUP_SIZE
    q_padded = roundup(seq_len, block_q)
    valid_len = seq_len if static_valid_len is None else static_valid_len
    if not 0 <= valid_len <= seq_len:
        raise ValueError("static_valid_len must be within the input sequence.")
    if valid_len == 0:
        return jnp.zeros((batch, seq_len, q_heads * head_dim), dtype=v.dtype)
    dynamic_mask = static_valid_len is None and valid_mask is not None
    kv_padded = roundup(valid_len, block_kv)
    flat_heads = batch * kv_heads

    q = q.reshape(batch, kv_heads, _GROUP_SIZE, seq_len, head_dim).transpose(0, 1, 3, 2, 4)
    q = q.reshape(flat_heads, seq_len * _GROUP_SIZE, head_dim)
    q = jnp.pad(q, ((0, 0), (0, (q_padded - seq_len) * _GROUP_SIZE), (0, 0)))
    k = k[:, :, :valid_len, :].reshape(flat_heads, valid_len, head_dim)
    k = jnp.pad(k, ((0, 0), (0, kv_padded - valid_len), (0, 0)))
    v = v[:, :, :valid_len, :].reshape(flat_heads, valid_len, head_dim).transpose(0, 2, 1)
    v = jnp.pad(v, ((0, 0), (0, 0), (0, kv_padded - valid_len)))

    mask_spec = None
    mask = None
    if dynamic_mask:
        mask = jnp.pad(valid_mask[:, None, :].astype(jnp.int32), ((0, 0), (0, 0), (0, kv_padded - valid_len)))
        masks_per_batch = mask.shape[0] > 1
        mask_spec = pl.BlockSpec(
            (None, 1, block_kv), lambda head, _, j: (head // kv_heads if masks_per_batch else 0, 0, j)
        )

    q_spec = pl.BlockSpec((None, q_rows, head_dim), lambda head, i, _: (head, i, 0))
    k_spec = pl.BlockSpec((None, block_kv, head_dim), lambda head, _, j: (head, j, 0))
    v_spec = pl.BlockSpec((None, head_dim, block_kv), lambda head, _, j: (head, 0, j))
    stats = pltpu.VMEM((q_rows, head_dim), jnp.float32)
    out = pl.pallas_call(
        partial(
            _gqa_kernel,
            compute_kv=block_kv_compute,
            tail=0 if dynamic_mask else valid_len % block_kv,
            dynamic_mask=dynamic_mask,
        ),
        name="krea2_pallas_gqa",
        out_shape=jax.ShapeDtypeStruct(q.shape, v.dtype),
        grid=(flat_heads, q_padded // block_q, kv_padded // block_kv),
        in_specs=(q_spec, k_spec, v_spec, mask_spec),
        out_specs=q_spec,
        scratch_shapes=(stats, stats, stats),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
        interpret=interpret,
    )(q, k, v, mask)

    out = out[:, :seq_len * _GROUP_SIZE, :].reshape(batch, kv_heads, seq_len, _GROUP_SIZE, head_dim)
    return out.transpose(0, 2, 1, 3, 4).reshape(batch, seq_len, q_heads * head_dim)
