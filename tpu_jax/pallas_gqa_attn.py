"""Forward-only grouped attention for the Krea 2 DiT on TPU v6e."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_jax.config import PallasGqaOptions
from tpu_jax.scheduler import roundup


_HEAD_DIM = 128
_GROUP_SIZE = 4
_DOT_NT = (((1,), (1,)), ((), ()))


def _approx_exp2(x, *, non_positive: bool = False):
    """Endpoint-constrained cubic, about 1.1e-4 relative error on normal results."""
    # Clamping at -127 maps underflow/masked entries to (-127 + 127) << 23 == +0.0f,
    # avoiding a separate zero-masking select.
    bounded = jnp.maximum(x, -127.0) if non_positive else jnp.clip(x, -127.0, 127.0)
    whole = jnp.floor(bounded)
    fraction = bounded - whole
    # The fractional polynomial is exactly 1 and 2 at the respective endpoints.
    poly = ((0.07826797 * fraction + 0.22630768) * fraction + 0.69542435) * fraction + 1.0
    scale = lax.bitcast_convert_type((whole.astype(jnp.int32) + 127) << 23, jnp.float32)
    value = poly * scale
    return value if non_positive else jnp.where(x > 127.0, jnp.inf, value)


def _dot(lhs, rhs):
    precision = lax.Precision.HIGHEST if lhs.dtype == jnp.float32 else None
    return lax.dot_general(lhs, rhs, _DOT_NT, precision=precision, preferred_element_type=jnp.float32)


def _attention_logits(q, k, mask, valid):
    logits = _dot(q, k)
    if mask is not None:
        logits = jnp.where(mask != 0, logits, -jnp.inf)
    elif valid is not None and valid < logits.shape[1]:
        columns = lax.broadcasted_iota(jnp.int32, logits.shape, 1)
        logits = jnp.where(columns < valid, logits, -jnp.inf)
    return logits


def _softmax_update(logits, m, denom, dtype, *, dynamic_mask, options):
    q_rows, kv_cols = logits.shape
    lane_replicated = m.shape[1] == _HEAD_DIM
    lane_repeats = kv_cols // _HEAD_DIM

    def _expand_stat(x):
        return jnp.broadcast_to(x, (q_rows, _HEAD_DIM)) if lane_replicated else x

    def _tile_stat(x):
        return jnp.tile(x, (1, lane_repeats)) if lane_replicated else x

    def online():
        next_m = jnp.maximum(m, _expand_stat(jnp.max(logits, axis=1, keepdims=True)))
        safe_m = jnp.where(next_m == -jnp.inf, 0.0, next_m) if dynamic_mask else next_m
        alpha = jnp.exp2(m - safe_m)
        shifted = logits - _tile_stat(safe_m)
        p = _approx_exp2(shifted, non_positive=True) if options.fast_exp2 else jnp.exp2(shifted)
        denom_next = denom * alpha + _expand_stat(jnp.sum(p, axis=1, keepdims=True))
        return next_m, denom_next, p.astype(dtype), alpha, jnp.bool_(True)

    if not options.lazy_rescale:
        return online()

    # A bounded sum also bounds all individual weights. Nonfinite sums (including
    # an uninitialized anchor) and outliers replay the tile with a fresh maximum.
    shifted = logits - _tile_stat(m)
    p = _approx_exp2(shifted, non_positive=False) if options.fast_exp2 else jnp.exp2(shifted)
    tile_sum_1 = jnp.sum(p, axis=1, keepdims=True)
    return lax.cond(
        jnp.all(tile_sum_1 <= 65536.0),
        lambda: (m, denom + _expand_stat(tile_sum_1), p.astype(dtype), jnp.ones_like(m), jnp.bool_(False)),
        online,
    )


def _update_numerator(numerator, pv, alpha, rescale, *, lazy_rescale):
    if lazy_rescale:
        return lax.cond(rescale, lambda: alpha * numerator + pv, lambda: numerator + pv)
    return alpha * numerator + pv


def _gqa_kernel(
    q_ref, k_ref, v_ref, mask_ref, out_ref, m_ref, l_ref, o_ref, *, compute_kv, tail, dynamic_mask, options
):
    kv_step = pl.program_id(2)
    last = kv_step == pl.num_programs(2) - 1
    q = q_ref[...]
    kv_tile = k_ref.shape[0]

    @pl.when(kv_step == 0)
    def initialize():
        m_ref[...] = jnp.full(m_ref.shape, -jnp.inf, dtype=jnp.float32)
        l_ref[...] = jnp.zeros(l_ref.shape, dtype=jnp.float32)
        o_ref[...] = jnp.zeros(o_ref.shape, dtype=jnp.float32)

    def logits_at(start, valid):
        tile = pl.ds(start, compute_kv)
        mask = mask_ref[:, tile] if dynamic_mask else None
        return _attention_logits(q, k_ref[tile, :], mask, None if valid is None else valid - start)

    def project(p, start):
        # V arrives as [D, KV]. Both dot operands contract along their minor axis.
        return _dot(p, v_ref[:, pl.ds(start, compute_kv)])

    def prepare(logits, m, denom):
        return _softmax_update(logits, m, denom, v_ref.dtype, dynamic_mask=dynamic_mask, options=options)

    update = partial(_update_numerator, lazy_rescale=options.lazy_rescale)

    def accumulate(valid):
        n_subtiles = kv_tile // compute_kv if valid is None else (valid + compute_kv - 1) // compute_kv
        m, denom, p, alpha, rescale = prepare(logits_at(0, valid), m_ref[...], l_ref[...])
        numerator = o_ref[...]
        for subtile in range(1, n_subtiles):
            start = subtile * compute_kv
            # Alternate the pending P/alpha and the next QK/softmax. These two
            # matmuls are independent of the next vector update until it consumes QK.
            logits = logits_at(start, valid)
            pv = project(p, start - compute_kv)
            m, denom, p, next_alpha, next_rescale = prepare(logits, m, denom)
            numerator = update(numerator, pv, alpha, rescale)
            alpha, rescale = next_alpha, next_rescale
        pv = project(p, (n_subtiles - 1) * compute_kv)
        return m, denom, update(numerator, pv, alpha, rescale)

    if tail:
        m, denom, numerator = lax.cond(last, lambda: accumulate(tail), lambda: accumulate(None))
    else:
        m, denom, numerator = accumulate(None)

    @pl.when(last)
    def finish():
        divisor = jnp.where(denom > 0, denom, 1.0) if dynamic_mask else denom
        out_ref[...] = (numerator / divisor).astype(out_ref.dtype)

    @pl.when(~last)
    def save():
        m_ref[...] = m
        l_ref[...] = denom
        o_ref[...] = numerator


def _continuous_kernel(
    q_ref, k_hbm, v_hbm, mask_hbm, out_ref, k_buffer, v_buffer, k_sem, v_sem,
    mask_buffer=None, mask_sem=None, *, kv_heads, block_kv, compute_kv, tail, options,
):
    """Double-buffer KV pages and carry the last pending PV across each page boundary."""
    head = pl.program_id(0)
    pages = k_hbm.shape[1] // block_kv
    q = q_ref[...]
    dynamic_mask = mask_hbm is not None
    subtile_count = block_kv // compute_kv

    def copies(page, slot):
        region = pl.ds(page * block_kv, block_kv)
        transfers = [
            pltpu.make_async_copy(k_hbm.at[head, region, :], k_buffer.at[slot], k_sem),
            pltpu.make_async_copy(v_hbm.at[head, :, region], v_buffer.at[slot], v_sem),
        ]
        if dynamic_mask:
            batch = lax.div(head, jnp.int32(kv_heads)) if mask_hbm.shape[0] > 1 else 0
            transfers.append(
                pltpu.make_async_copy(mask_hbm.at[batch, :, region], mask_buffer.at[slot], mask_sem)
            )
        return transfers

    def logits_at(slot, subtile, valid):
        tile = pl.ds(subtile * compute_kv, compute_kv)
        mask = mask_buffer[slot, :, tile] if dynamic_mask else None
        return _attention_logits(q, k_buffer[slot, tile, :], mask, valid)

    def project(p, slot, subtile):
        return _dot(p, v_buffer[slot, :, pl.ds(subtile * compute_kv, compute_kv)])

    def prepare(logits, m, denom):
        return _softmax_update(logits, m, denom, v_buffer.dtype, dynamic_mask=dynamic_mask, options=options)

    update = partial(_update_numerator, lazy_rescale=options.lazy_rescale)

    def advance(state, slot, subtile, next_slot, next_subtile, next_valid, transfers=()):
        m, denom, numerator, p, alpha, rescale = state
        pv = project(p, slot, subtile)
        for transfer in transfers:
            transfer.wait()
        logits = logits_at(next_slot, next_subtile, next_valid)
        m, denom, next_p, next_alpha, next_rescale = prepare(logits, m, denom)
        return m, denom, update(numerator, pv, alpha, rescale), next_p, next_alpha, next_rescale

    first_transfers = copies(0, 0)
    for transfer in first_transfers:
        transfer.start()
    for transfer in first_transfers:
        transfer.wait()
    low = jnp.full((q.shape[0], _HEAD_DIM), -jnp.inf, dtype=jnp.float32)
    zero = jnp.zeros_like(low)
    m, denom, p, alpha, rescale = prepare(logits_at(0, 0, tail if pages == 1 and tail else None), low, zero)
    state = (m, denom, jnp.zeros(q.shape, dtype=jnp.float32), p, alpha, rescale)

    def process_page(page, state):
        slot = page % 2
        next_slot = 1 - slot
        transfers = copies(page + 1, next_slot)
        for transfer in transfers:
            transfer.start()
        for sub in range(1, subtile_count):
            state = advance(state, slot, sub - 1, slot, sub, None)
        # When the final page starts with a partial tile, complete the DMA wait
        # outside lax.cond and only branch on the masked vs unmasked logits tile.
        if tail and tail < compute_kv:
            m, denom, numerator, p, alpha, rescale = state
            pv = project(p, slot, subtile_count - 1)
            for transfer in transfers:
                transfer.wait()
            logits = lax.cond(
                page == pages - 2,
                lambda: logits_at(next_slot, 0, tail),
                lambda: logits_at(next_slot, 0, None),
            )
            m, denom, next_p, next_alpha, next_rescale = prepare(logits, m, denom)
            return m, denom, update(numerator, pv, alpha, rescale), next_p, next_alpha, next_rescale
        return advance(state, slot, subtile_count - 1, next_slot, 0, None, transfers)

    state = lax.fori_loop(0, pages - 1, process_page, state)
    slot = (pages - 1) % 2
    final_subtiles = (tail + compute_kv - 1) // compute_kv if tail else subtile_count
    for sub in range(1, final_subtiles):
        state = advance(state, slot, sub - 1, slot, sub, tail - sub * compute_kv if tail else None)
    _, denom, numerator, p, alpha, rescale = state
    numerator = update(numerator, project(p, slot, final_subtiles - 1), alpha, rescale)
    divisor = jnp.where(denom > 0, denom, 1.0) if dynamic_mask else denom
    out_ref[...] = (numerator / divisor).astype(out_ref.dtype)


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
    options: PallasGqaOptions = PallasGqaOptions(),
    grouped_q: bool = False,
) -> jax.Array:
    """Return BSH from base-2 prescaled Q and BHSD K/V. Grouped Q is B, Hkv, S, G, D."""
    kv_heads = k.shape[1]
    if grouped_q:
        batch, q_groups, seq_len, group_size, head_dim = q.shape
        q_heads = q_groups * group_size
        layout_valid = q_groups == kv_heads and group_size == _GROUP_SIZE
    else:
        batch, q_heads, seq_len, head_dim = q.shape
        layout_valid = q_heads == _GROUP_SIZE * kv_heads
    if head_dim != _HEAD_DIM or not layout_valid:
        raise ValueError("pallas_gqa requires head_dim=128 and four query heads per KV head.")

    # block_q is in tokens per head. Four aligned query tiles share each K/V DMA.
    q_rows = block_q * _GROUP_SIZE
    if q_rows <= 0 or q_rows % 8:
        raise ValueError("block_q must be positive and produce a grouped Q tile divisible by 8 rows.")
    if (
        block_kv <= 0
        or block_kv_compute <= 0
        or block_kv_compute % _HEAD_DIM
        or block_kv % block_kv_compute
    ):
        raise ValueError(
            "block_kv and block_kv_compute must be positive multiples of 128, "
            "and block_kv must be divisible by block_kv_compute."
        )
    q_padded = roundup(seq_len, block_q)
    valid_len = seq_len if static_valid_len is None else static_valid_len
    if not 0 <= valid_len <= seq_len:
        raise ValueError("static_valid_len must be within the input sequence.")
    if valid_len == 0:
        return jnp.zeros((batch, seq_len, q_heads * head_dim), dtype=v.dtype)
    dynamic_mask = static_valid_len is None and valid_mask is not None
    kv_padded = roundup(valid_len, block_kv)
    flat_heads = batch * kv_heads

    if not grouped_q:
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

    q_spec = pl.BlockSpec((None, q_rows, head_dim), lambda head, i, *unused: (head, i, 0))
    grid = (flat_heads, q_padded // block_q)
    tail = 0 if dynamic_mask else valid_len % block_kv
    if options.continuous_kv:
        hbm = pl.BlockSpec(memory_space=pltpu.HBM)
        kernel = partial(
            _continuous_kernel, kv_heads=kv_heads, block_kv=block_kv, compute_kv=block_kv_compute,
            tail=tail, options=options,
        )
        in_specs = (q_spec, hbm, hbm, hbm if dynamic_mask else None)
        scratch = (
            pltpu.VMEM((2, block_kv, head_dim), k.dtype),
            pltpu.VMEM((2, head_dim, block_kv), v.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        )
        if dynamic_mask:
            scratch += (pltpu.VMEM((2, 1, block_kv), mask.dtype), pltpu.SemaphoreType.DMA)
        semantics = ("parallel", "parallel")
        if interpret is True:
            interpret = pltpu.InterpretParams(detect_races=True)
    else:
        kernel = partial(
            _gqa_kernel, compute_kv=block_kv_compute, tail=tail, dynamic_mask=dynamic_mask, options=options
        )
        in_specs = (
            q_spec,
            pl.BlockSpec((None, block_kv, head_dim), lambda head, _, j: (head, j, 0)),
            pl.BlockSpec((None, head_dim, block_kv), lambda head, _, j: (head, 0, j)),
            mask_spec,
        )
        stats = pltpu.VMEM((q_rows, head_dim), jnp.float32)
        scratch = (stats, stats, stats)
        grid += (kv_padded // block_kv,)
        semantics = ("parallel", "parallel", "arbitrary")
    out = pl.pallas_call(
        kernel,
        name="krea2_pallas_gqa",
        out_shape=jax.ShapeDtypeStruct(q.shape, v.dtype),
        grid=grid,
        in_specs=in_specs,
        out_specs=q_spec,
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=semantics, vmem_limit_bytes=options.vmem_limit_bytes
        ),
        interpret=interpret,
    )(q, k, v, mask)

    out = out[:, :seq_len * _GROUP_SIZE, :].reshape(batch, kv_heads, seq_len, _GROUP_SIZE, head_dim)
    return out.transpose(0, 2, 1, 3, 4).reshape(batch, seq_len, q_heads * head_dim)
