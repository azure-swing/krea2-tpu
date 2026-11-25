"""Zero-PyTorch parallel streaming safetensors weight loaders for Krea 2 (JAX/TPU).

Loading optimizations (inspired by MOSS-Image-TPU + native JAX/XLA):
1. Uses ``safetensors`` ``framework="numpy"`` (returning native ``ml_dtypes.bfloat16``
   arrays directly with zero JAX CPU dispatch overhead per tensor).
2. Pre-allocates ONE contiguous NumPy buffer per stacked field (``[L, ...]``) and
   populates all blocks in parallel with ``ThreadPoolExecutor`` (eliminating the
   2x peak RAM spike of ``jnp.stack`` and overlapping GIL-free Rust mmap reads +
   NumPy transposes across CPU cores).
3. Pipelines field-by-field **async H2D** (``jax.device_put``) directly to TPU
   HBM (or 1D TPU ``Mesh`` sharding) as each field completes, immediately freeing
   its CPU buffer while the next field is assembled on CPU threads.
   -> Peak host working buffer during 25 GB DiT load is <= 11.3 GB (1 field)
      instead of ~50 GB!
4. Explicit ``release_tpu_pytree()`` calls ``arr.delete()`` on every TPU array
   leaf for deterministic, zero-GC-lag HBM reclamation during stage switches.
"""

from __future__ import annotations

import gc
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import ml_dtypes
import numpy as np

from tpu_jax.config import Krea2DiTConfig, Qwen3TextConfig, QwenImageVAEConfig
from tpu_jax.dit import precompute_rope_omegas


def _default_num_workers(num_workers: int | None = None) -> int:
    if num_workers is not None:
        return max(1, int(num_workers))
    env = os.environ.get("KREA2_LOAD_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, min(8, os.cpu_count() or 4))


def _resolve_np_dtype(dtype: Any) -> np.dtype:
    if isinstance(dtype, np.dtype):
        return dtype
    if isinstance(dtype, str):
        key = dtype.lower()
        if key in ("bf16", "bfloat16"):
            return np.dtype(ml_dtypes.bfloat16)
        if key in ("fp32", "float32"):
            return np.dtype(np.float32)
        if key in ("fp16", "float16"):
            return np.dtype(np.float16)
        raise ValueError(f"Unsupported dtype {dtype!r}.")
    # Handle jax.numpy dtype objects
    name = getattr(dtype, "name", str(dtype)).lower()
    if "bfloat16" in name:
        return np.dtype(ml_dtypes.bfloat16)
    if "float32" in name:
        return np.dtype(np.float32)
    if "float16" in name:
        return np.dtype(np.float16)
    return np.dtype(dtype)


def _as_np(val: Any, target_dtype: np.dtype | None = None) -> np.ndarray:
    arr = np.asarray(val)
    if target_dtype is not None and arr.dtype != target_dtype:
        arr = arr.astype(target_dtype, copy=False)
    return arr


def _make_put_fn(
    *,
    device: Any | None = None,
    mesh: Any | None = None,
    tp_axis: str = "tp",
    use_cpu_jax: bool = True,
) -> Callable[[np.ndarray, Any | None], Any]:
    """Build a function that dispatches each assembled NumPy buffer directly to target device/sharding."""
    import jax

    if mesh is not None:
        from jax.sharding import NamedSharding, PartitionSpec as P

        rep = NamedSharding(mesh, P())

        def _put_sharded(arr: np.ndarray, spec: Any | None = None):
            sharding = NamedSharding(mesh, spec) if spec is not None else rep
            return jax.device_put(arr, sharding)

        return _put_sharded

    if device is not None:

        def _put_device(arr: np.ndarray, _spec: Any | None = None):
            return jax.device_put(arr, device)

        return _put_device

    if use_cpu_jax:
        cpu_devs = jax.devices("cpu")
        cpu_dev = cpu_devs[0] if cpu_devs else None

        def _put_cpu(arr: np.ndarray, _spec: Any | None = None):
            return jax.device_put(arr, cpu_dev) if cpu_dev is not None else jax.numpy.asarray(arr)

        return _put_cpu

    return lambda arr, _spec=None: arr


def release_tpu_pytree(params: Any) -> tuple[float, int]:
    """Deterministically free TPU HBM buffers via ``Array.delete()`` (zero GC lag)."""
    import jax

    t0 = time.perf_counter()
    freed = 0

    def _delete_leaf(leaf: Any):
        nonlocal freed
        if isinstance(leaf, jax.Array):
            try:
                if not leaf.is_deleted():
                    leaf.delete()
                    freed += 1
            except Exception:
                pass
        return None

    jax.tree_util.tree_map(_delete_leaf, params)
    gc.collect()
    return time.perf_counter() - t0, freed


def transfer_pytree_to_device(
    params: dict[str, Any],
    *,
    device: Any | None = None,
    mesh: Any | None = None,
    tp_axis: str = "tp",
    kind: str = "generic",
) -> dict[str, Any]:
    """Async H2D transfer of a pre-fused CPU PyTree to TPU device or 1D Mesh."""
    import jax

    if mesh is not None and kind == "dit":
        return shard_dit_params(params, mesh, tp_axis=tp_axis)
    if device is None:
        return params
    return jax.device_put(params, device)


class _LazySafetensorsReader:
    """Lazy multi-shard safetensors reader via ``framework='numpy'`` (native ``ml_dtypes.bfloat16``)."""

    def __init__(self, files: list[Path]):
        from safetensors import safe_open

        self._handles = [safe_open(str(p), framework="numpy", device="cpu") for p in files]
        self._key_to_handle: dict[str, Any] = {}
        for h in self._handles:
            for k in h.keys():
                self._key_to_handle[k] = h

    def keys(self) -> set[str]:
        return set(self._key_to_handle.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._key_to_handle

    def get(self, key: str) -> np.ndarray:
        if key not in self._key_to_handle:
            raise KeyError(f"Weight key not found in safetensors: {key}")
        return self._key_to_handle[key].get_tensor(key)

    def close(self) -> None:
        self._key_to_handle.clear()
        self._handles.clear()


def _rope_half_split_perm(head_dim: int) -> np.ndarray:
    """Permutation ``[0, 2, ..., D-2, 1, 3, ..., D-1]`` for contiguous half-split RoPE."""
    return np.concatenate([np.arange(0, head_dim, 2), np.arange(1, head_dim, 2)])


def _permute_head_dim(w_t: np.ndarray, num_heads: int, head_dim: int, perm: np.ndarray) -> np.ndarray:
    """Permute output head-dim channels of ``w_t [in_dim, num_heads * head_dim]``."""
    in_dim = w_t.shape[0]
    return w_t.reshape(in_dim, num_heads, head_dim)[:, :, perm].reshape(in_dim, num_heads * head_dim)


def _fuse_qkvgate_into(
    out: np.ndarray,
    wq_t: np.ndarray,
    wk_t: np.ndarray,
    wv_t: np.ndarray,
    wgate_t: np.ndarray,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    tp_size: int = 1,
) -> None:
    """Write per-TP-shard contiguous ``[Q_s, K_s, V_s, Gate_s]`` directly into ``out``.

    For ``tp_size == 1``, this produces a single contiguous ``[Q | K | V | Gate]`` layout
    so TPU slicing avoids strided KV-group gathers. For ``tp_size > 1``, each TP shard
    holds its own contiguous ``[Q_s | K_s | V_s | Gate_s]`` slice along the sharded axis.
    """
    dim = wq_t.shape[0]
    if tp_size <= 1:
        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        out[:, :q_dim] = wq_t
        out[:, q_dim : q_dim + kv_dim] = wk_t
        out[:, q_dim + kv_dim : q_dim + 2 * kv_dim] = wv_t
        out[:, q_dim + 2 * kv_dim :] = wgate_t
        return

    if num_kv_heads % tp_size != 0 or num_heads % tp_size != 0:
        raise ValueError(
            f"tp_size={tp_size} must divide num_heads={num_heads} and num_kv_heads={num_kv_heads}."
        )
    q_s = (num_heads // tp_size) * head_dim
    kv_s = (num_kv_heads // tp_size) * head_dim
    per_s = 2 * q_s + 2 * kv_s
    view = out.reshape(dim, tp_size, per_s)
    view[:, :, :q_s] = wq_t.reshape(dim, tp_size, q_s)
    view[:, :, q_s : q_s + kv_s] = wk_t.reshape(dim, tp_size, kv_s)
    view[:, :, q_s + kv_s : q_s + 2 * kv_s] = wv_t.reshape(dim, tp_size, kv_s)
    view[:, :, q_s + 2 * kv_s :] = wgate_t.reshape(dim, tp_size, q_s)


def _fuse_qkvgate_np(
    wq_t: np.ndarray,
    wk_t: np.ndarray,
    wv_t: np.ndarray,
    wgate_t: np.ndarray,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    tp_size: int = 1,
) -> np.ndarray:
    dim = wq_t.shape[0]
    total_out = 2 * num_heads * head_dim + 2 * num_kv_heads * head_dim
    out = np.empty((dim, total_out), dtype=wq_t.dtype)
    _fuse_qkvgate_into(
        out,
        wq_t,
        wk_t,
        wv_t,
        wgate_t,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tp_size=tp_size,
    )
    return out


def _fuse_mlp_gate_up_into(
    out: np.ndarray,
    wgate_t: np.ndarray,
    wup_t: np.ndarray,
    *,
    tp_size: int = 1,
) -> None:
    """Write per-TP-shard contiguous ``[Gate_s | Up_s]`` directly into ``out [dim, 2 * mlp_dim]``.

    Avoids stride-2 minor-dim interleaving (``0::2`` / ``1::2``), which causes costly
    128-lane VPU shuffles on TPU.
    """
    dim, mlp_dim = wgate_t.shape
    if tp_size <= 1:
        out[:, :mlp_dim] = wgate_t
        out[:, mlp_dim:] = wup_t
        return

    if mlp_dim % tp_size != 0:
        raise ValueError(f"tp_size={tp_size} must divide mlp_dim={mlp_dim}.")
    m_s = mlp_dim // tp_size
    view = out.reshape(dim, tp_size, 2, m_s)
    view[:, :, 0, :] = wgate_t.reshape(dim, tp_size, m_s)
    view[:, :, 1, :] = wup_t.reshape(dim, tp_size, m_s)


def _fuse_mlp_gate_up_np(
    wgate_t: np.ndarray,
    wup_t: np.ndarray,
    *,
    tp_size: int = 1,
) -> np.ndarray:
    dim, mlp_dim = wgate_t.shape
    out = np.empty((dim, 2 * mlp_dim), dtype=wgate_t.dtype)
    _fuse_mlp_gate_up_into(out, wgate_t, wup_t, tp_size=tp_size)
    return out


# ---------------------------------------------------------------------------
# 1. Krea 2 SingleStreamDiT Weight Builder & Parallel Streaming Loader
# ---------------------------------------------------------------------------


def _quantize_int8_per_channel(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-output-channel INT8 weight quantization (along axis=-2)."""
    w_f32 = w.astype(np.float32, copy=False)
    absmax = np.max(np.abs(w_f32), axis=-2, keepdims=True)
    scale = np.maximum(absmax / 127.0, 1e-12).astype(ml_dtypes.bfloat16)
    w_q = np.clip(np.rint(w_f32 / scale.astype(np.float32)), -127, 127).astype(np.int8)
    return np.ascontiguousarray(w_q), np.ascontiguousarray(np.squeeze(scale, axis=-2))


def _build_dit_params(
    get_tensor: Callable[[str], Any],
    all_keys: set[str],
    config: Krea2DiTConfig,
    *,
    np_dtype: np.dtype,
    put_fn: Callable[[np.ndarray, Any | None], Any],
    num_workers: int = 1,
    tp_axis: str = "tp",
    tp_size: int = 1,
    quant: str = "none",
) -> dict[str, Any]:
    from jax.sharding import PartitionSpec as P

    if quant not in ("none", "int8"):
        raise ValueError(f"Unsupported quant={quant!r}; expected 'none' or 'int8'.")
    use_int8 = quant == "int8"
    bf16_np = np.dtype(ml_dtypes.bfloat16)
    is_diffusers = "transformer_blocks.0.attn.to_q.weight" in all_keys
    f32 = np.dtype(np.float32)
    rope_perm = _rope_half_split_perm(config.head_dim)
    q_inv_sqrt = np.float32(1.0 / math.sqrt(config.head_dim))
    txt_q_inv_sqrt = np.float32(1.0 / math.sqrt(config.text_head_dim))

    def _w_np(key: str, *, transpose: bool = False, dtype: np.dtype = np_dtype) -> np.ndarray:
        arr = _as_np(get_tensor(key), dtype)
        if transpose:
            arr = np.swapaxes(arr, -1, -2)
        return arr

    def _norm_np(key: str) -> np.ndarray:
        # Krea 2 RMSNorm uses (scale.float() + 1.0); both native & diffusers store raw scale.
        return _as_np(get_tensor(key), f32) + np.float32(1.0)

    def _load_text_fusion_block(prefix: str, *, diffusers: bool) -> dict[str, Any]:
        if diffusers:
            wq = _w_np(f"{prefix}.attn.to_q.weight", transpose=True)
            wk = _w_np(f"{prefix}.attn.to_k.weight", transpose=True)
            wv = _w_np(f"{prefix}.attn.to_v.weight", transpose=True)
            wg = _w_np(f"{prefix}.attn.to_gate.weight", transpose=True)
            wo = _w_np(f"{prefix}.attn.to_out.0.weight", transpose=True)
            qn = _norm_np(f"{prefix}.attn.norm_q.weight") * txt_q_inv_sqrt
            kn = _norm_np(f"{prefix}.attn.norm_k.weight")
            mg = _w_np(f"{prefix}.ff.gate.weight", transpose=True)
            mu = _w_np(f"{prefix}.ff.up.weight", transpose=True)
            md = _w_np(f"{prefix}.ff.down.weight", transpose=True)
            pn = _norm_np(f"{prefix}.norm1.weight")
            pon = _norm_np(f"{prefix}.norm2.weight")
        else:
            wq = _w_np(f"{prefix}.attn.wq.weight", transpose=True)
            wk = _w_np(f"{prefix}.attn.wk.weight", transpose=True)
            wv = _w_np(f"{prefix}.attn.wv.weight", transpose=True)
            wg = _w_np(f"{prefix}.attn.gate.weight", transpose=True)
            wo = _w_np(f"{prefix}.attn.wo.weight", transpose=True)
            qn = _norm_np(f"{prefix}.attn.qknorm.qnorm.scale") * txt_q_inv_sqrt
            kn = _norm_np(f"{prefix}.attn.qknorm.knorm.scale")
            mg = _w_np(f"{prefix}.mlp.gate.weight", transpose=True)
            mu = _w_np(f"{prefix}.mlp.up.weight", transpose=True)
            md = _w_np(f"{prefix}.mlp.down.weight", transpose=True)
            pn = _norm_np(f"{prefix}.prenorm.scale")
            pon = _norm_np(f"{prefix}.postnorm.scale")

        qkvg = _fuse_qkvgate_np(
            wq,
            wk,
            wv,
            wg,
            num_heads=config.txtheads,
            num_kv_heads=config.txtkvheads,
            head_dim=config.text_head_dim,
            tp_size=1,
        )
        mgu = _fuse_mlp_gate_up_np(mg, mu, tp_size=1)
        blk_out: dict[str, Any] = {
            "prenorm": put_fn(pn, None),
            "postnorm": put_fn(pon, None),
            "qnorm": put_fn(qn, None),
            "knorm": put_fn(kn, None),
        }
        for name, arr in (
            ("qkvgate", qkvg),
            ("wo", np.ascontiguousarray(wo)),
            ("mlp_gate_up", mgu),
            ("mlp_down", np.ascontiguousarray(md)),
        ):
            if use_int8:
                wq_i8, ws_bf16 = _quantize_int8_per_channel(arr)
                blk_out[f"{name}_w"] = put_fn(wq_i8, None)
                blk_out[f"{name}_s"] = put_fn(ws_bf16, None)
            else:
                blk_out[f"{name}_w"] = put_fn(arr, None)
        return blk_out

    # TextFusion blocks (2 layerwise + 2 refiner)
    tf_prefix = "text_fusion" if is_diffusers else "txtfusion"
    layerwise_blocks = [
        _load_text_fusion_block(f"{tf_prefix}.layerwise_blocks.{i}", diffusers=is_diffusers)
        for i in range(config.num_layerwise_text_blocks)
    ]
    refiner_blocks = [
        _load_text_fusion_block(f"{tf_prefix}.refiner_blocks.{i}", diffusers=is_diffusers)
        for i in range(config.num_refiner_text_blocks)
    ]
    txtfusion_proj_w = put_fn(np.ascontiguousarray(_w_np(f"{tf_prefix}.projector.weight", transpose=True)), None)

    # Parallel pre-allocated field-by-field assembly + immediate async H2D dispatch.
    blk_prefix = "transformer_blocks" if is_diffusers else "blocks"
    n_layers = config.layers
    dim = config.features
    hd = config.head_dim
    kv_heads = config.kvheads
    qkvg_out = 2 * config.heads * hd + 2 * kv_heads * hd
    mlp_dim = config.mlp_dim

    def _run_parallel(fn: Callable[[int], None]) -> None:
        if num_workers <= 1 or n_layers <= 1:
            for i in range(n_layers):
                fn(i)
            return
        with ThreadPoolExecutor(max_workers=min(num_workers, n_layers)) as ex:
            list(ex.map(fn, range(n_layers)))

    def _stream_1d_norm(
        suffix_diff: str,
        suffix_nat: str,
        length: int,
        *,
        perm: np.ndarray | None = None,
        scale_factor: np.float32 | None = None,
    ):
        buf = np.empty((n_layers, length), dtype=f32)
        suf = suffix_diff if is_diffusers else suffix_nat

        def _fill(i: int):
            v = _norm_np(f"{blk_prefix}.{i}.{suf}")
            if perm is not None:
                v = v[perm]
            if scale_factor is not None:
                v = v * scale_factor
            buf[i] = v

        _run_parallel(_fill)
        res = put_fn(buf, None)
        del buf
        return res

    def _stream_2d_linear(
        suffix_diff: str, suffix_nat: str, shape_2d: tuple[int, int], spec: Any, scale_spec: Any
    ):
        suf = suffix_diff if is_diffusers else suffix_nat
        if use_int8:
            buf_q = np.empty((n_layers, shape_2d[0], shape_2d[1]), dtype=np.int8)
            buf_s = np.empty((n_layers, shape_2d[1]), dtype=bf16_np)

            def _fill_q(i: int):
                wq_i8, ws = _quantize_int8_per_channel(_w_np(f"{blk_prefix}.{i}.{suf}", transpose=True))
                buf_q[i] = wq_i8
                buf_s[i] = ws

            _run_parallel(_fill_q)
            res_w = put_fn(buf_q, spec)
            res_s = put_fn(buf_s, scale_spec)
            del buf_q, buf_s
            return res_w, res_s

        buf = np.empty((n_layers, shape_2d[0], shape_2d[1]), dtype=np_dtype)

        def _fill(i: int):
            buf[i] = _w_np(f"{blk_prefix}.{i}.{suf}", transpose=True)

        _run_parallel(_fill)
        res = put_fn(buf, spec)
        del buf
        return res, None

    def _stream_mod_lin():
        buf = np.empty((n_layers, 6 * dim), dtype=np_dtype)
        suf = "scale_shift_table" if is_diffusers else "mod.lin"

        def _fill(i: int):
            buf[i] = _w_np(f"{blk_prefix}.{i}.{suf}").reshape(-1)

        _run_parallel(_fill)
        res = put_fn(buf, None)
        del buf
        return res

    def _stream_qkvgate():
        def _load_layer_fused(i: int, dst: np.ndarray) -> None:
            p = f"{blk_prefix}.{i}"
            if is_diffusers:
                wq = _w_np(f"{p}.attn.to_q.weight", transpose=True)
                wk = _w_np(f"{p}.attn.to_k.weight", transpose=True)
                wv = _w_np(f"{p}.attn.to_v.weight", transpose=True)
                wg = _w_np(f"{p}.attn.to_gate.weight", transpose=True)
            else:
                wq = _w_np(f"{p}.attn.wq.weight", transpose=True)
                wk = _w_np(f"{p}.attn.wk.weight", transpose=True)
                wv = _w_np(f"{p}.attn.wv.weight", transpose=True)
                wg = _w_np(f"{p}.attn.gate.weight", transpose=True)
            # Pre-permute Q/K head-dim channels to [even|odd] for contiguous half-split RoPE.
            wq = _permute_head_dim(wq, config.heads, hd, rope_perm)
            wk = _permute_head_dim(wk, kv_heads, hd, rope_perm)
            _fuse_qkvgate_into(
                dst,
                wq,
                wk,
                wv,
                wg,
                num_heads=config.heads,
                num_kv_heads=kv_heads,
                head_dim=hd,
                tp_size=tp_size,
            )

        if use_int8:
            buf_q = np.empty((n_layers, dim, qkvg_out), dtype=np.int8)
            buf_s = np.empty((n_layers, qkvg_out), dtype=bf16_np)

            def _fill_q(i: int):
                tmp = np.empty((dim, qkvg_out), dtype=np_dtype)
                _load_layer_fused(i, tmp)
                wq_i8, ws = _quantize_int8_per_channel(tmp)
                buf_q[i] = wq_i8
                buf_s[i] = ws

            _run_parallel(_fill_q)
            res_w = put_fn(buf_q, P(None, None, tp_axis))
            res_s = put_fn(buf_s, P(None, tp_axis))
            del buf_q, buf_s
            return res_w, res_s

        buf = np.empty((n_layers, dim, qkvg_out), dtype=np_dtype)
        _run_parallel(lambda i: _load_layer_fused(i, buf[i]))
        res = put_fn(buf, P(None, None, tp_axis))
        del buf
        return res, None

    def _stream_mlp_gate_up():
        def _load_layer_fused(i: int, dst: np.ndarray) -> None:
            p = f"{blk_prefix}.{i}"
            if is_diffusers:
                mg = _w_np(f"{p}.ff.gate.weight", transpose=True)
                mu = _w_np(f"{p}.ff.up.weight", transpose=True)
            else:
                mg = _w_np(f"{p}.mlp.gate.weight", transpose=True)
                mu = _w_np(f"{p}.mlp.up.weight", transpose=True)
            _fuse_mlp_gate_up_into(dst, mg, mu, tp_size=tp_size)

        if use_int8:
            buf_q = np.empty((n_layers, dim, 2 * mlp_dim), dtype=np.int8)
            buf_s = np.empty((n_layers, 2 * mlp_dim), dtype=bf16_np)

            def _fill_q(i: int):
                tmp = np.empty((dim, 2 * mlp_dim), dtype=np_dtype)
                _load_layer_fused(i, tmp)
                wq_i8, ws = _quantize_int8_per_channel(tmp)
                buf_q[i] = wq_i8
                buf_s[i] = ws

            _run_parallel(_fill_q)
            res_w = put_fn(buf_q, P(None, None, tp_axis))
            res_s = put_fn(buf_s, P(None, tp_axis))
            del buf_q, buf_s
            return res_w, res_s

        buf = np.empty((n_layers, dim, 2 * mlp_dim), dtype=np_dtype)
        _run_parallel(lambda i: _load_layer_fused(i, buf[i]))
        res = put_fn(buf, P(None, None, tp_axis))
        del buf
        return res, None

    # Order large fields so async H2D DMA overlaps cleanly with CPU thread-pool assembly!
    qkvgate_w, qkvgate_s = _stream_qkvgate()
    wo_w, wo_s = _stream_2d_linear(
        "attn.to_out.0.weight", "attn.wo.weight", (dim, dim), P(None, tp_axis, None), P()
    )
    mlp_gate_up_w, mlp_gate_up_s = _stream_mlp_gate_up()
    mlp_down_w, mlp_down_s = _stream_2d_linear(
        "ff.down.weight", "mlp.down.weight", (mlp_dim, dim), P(None, tp_axis, None), P()
    )
    blocks = {
        "prenorm": _stream_1d_norm("norm1.weight", "prenorm.scale", dim),
        "postnorm": _stream_1d_norm("norm2.weight", "postnorm.scale", dim),
        "qnorm": _stream_1d_norm(
            "attn.norm_q.weight",
            "attn.qknorm.qnorm.scale",
            hd,
            perm=rope_perm,
            scale_factor=q_inv_sqrt,
        ),
        "knorm": _stream_1d_norm(
            "attn.norm_k.weight",
            "attn.qknorm.knorm.scale",
            hd,
            perm=rope_perm,
        ),
        "mod_lin": _stream_mod_lin(),
        "qkvgate_w": qkvgate_w,
        "wo_w": wo_w,
        "mlp_gate_up_w": mlp_gate_up_w,
        "mlp_down_w": mlp_down_w,
    }
    if use_int8:
        blocks["qkvgate_s"] = qkvgate_s
        blocks["wo_s"] = wo_s
        blocks["mlp_gate_up_s"] = mlp_gate_up_s
        blocks["mlp_down_s"] = mlp_down_s

    # Precomputed FP32 sinusoidal time frequencies and 3D axial RoPE omegas.
    half = config.tdim // 2
    time_freqs = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float32) / float(half)).astype(np.float32)
    rope_omegas = tuple(
        put_fn(o, None) for o in precompute_rope_omegas(config.axes_dims, config.theta, config.ntk)
    )

    def _put_lin(key: str, *, transpose: bool = False):
        arr = _w_np(key, transpose=transpose)
        return put_fn(np.ascontiguousarray(arr), None)

    if is_diffusers:
        first_w = _put_lin("img_in.weight", transpose=True)
        first_b = _put_lin("img_in.bias")
        tmlp_fc1_w = _put_lin("time_embed.linear_1.weight", transpose=True)
        tmlp_fc1_b = _put_lin("time_embed.linear_1.bias")
        tmlp_fc2_w = _put_lin("time_embed.linear_2.weight", transpose=True)
        tmlp_fc2_b = _put_lin("time_embed.linear_2.bias")
        tproj_w = _put_lin("time_mod_proj.weight", transpose=True)
        tproj_b = _put_lin("time_mod_proj.bias")
        txtmlp_norm = put_fn(_norm_np("txt_in.norm.weight"), None)
        txtmlp_fc1_w = _put_lin("txt_in.linear_1.weight", transpose=True)
        txtmlp_fc1_b = _put_lin("txt_in.linear_1.bias")
        txtmlp_fc2_w = _put_lin("txt_in.linear_2.weight", transpose=True)
        txtmlp_fc2_b = _put_lin("txt_in.linear_2.bias")
        last_norm = put_fn(_norm_np("final_layer.norm.weight"), None)
        last_mod_lin = put_fn(_w_np("final_layer.scale_shift_table").reshape(2, dim), None)
        last_linear_w = _put_lin("final_layer.linear.weight", transpose=True)
        last_linear_b = _put_lin("final_layer.linear.bias")
    else:
        first_w = _put_lin("first.weight", transpose=True)
        first_b = _put_lin("first.bias")
        tmlp_fc1_w = _put_lin("tmlp.0.weight", transpose=True)
        tmlp_fc1_b = _put_lin("tmlp.0.bias")
        tmlp_fc2_w = _put_lin("tmlp.2.weight", transpose=True)
        tmlp_fc2_b = _put_lin("tmlp.2.bias")
        tproj_w = _put_lin("tproj.1.weight", transpose=True)
        tproj_b = _put_lin("tproj.1.bias")
        txtmlp_norm = put_fn(_norm_np("txtmlp.0.scale"), None)
        txtmlp_fc1_w = _put_lin("txtmlp.1.weight", transpose=True)
        txtmlp_fc1_b = _put_lin("txtmlp.1.bias")
        txtmlp_fc2_w = _put_lin("txtmlp.3.weight", transpose=True)
        txtmlp_fc2_b = _put_lin("txtmlp.3.bias")
        last_norm = put_fn(_norm_np("last.norm.scale"), None)
        last_mod_lin = put_fn(_w_np("last.modulation.lin").reshape(2, dim), None)
        last_linear_w = _put_lin("last.linear.weight", transpose=True)
        last_linear_b = _put_lin("last.linear.bias")

    return {
        "first_w": first_w,
        "first_b": first_b,
        "tmlp_fc1_w": tmlp_fc1_w,
        "tmlp_fc1_b": tmlp_fc1_b,
        "tmlp_fc2_w": tmlp_fc2_w,
        "tmlp_fc2_b": tmlp_fc2_b,
        "tproj_w": tproj_w,
        "tproj_b": tproj_b,
        "txtfusion_layerwise": layerwise_blocks,
        "txtfusion_proj_w": txtfusion_proj_w,
        "txtfusion_refiner": refiner_blocks,
        "txtmlp_norm": txtmlp_norm,
        "txtmlp_fc1_w": txtmlp_fc1_w,
        "txtmlp_fc1_b": txtmlp_fc1_b,
        "txtmlp_fc2_w": txtmlp_fc2_w,
        "txtmlp_fc2_b": txtmlp_fc2_b,
        "blocks": blocks,
        "last_norm": last_norm,
        "last_mod_lin": last_mod_lin,
        "last_linear_w": last_linear_w,
        "last_linear_b": last_linear_b,
        "time_freqs": put_fn(time_freqs, None),
        "rope_omegas": rope_omegas,
    }


def build_dit_params_from_state_dict(
    state_dict: dict[str, Any],
    config: Krea2DiTConfig,
    *,
    dtype: Any = "bfloat16",
    consume: bool = False,
    device: Any | None = None,
    mesh: Any | None = None,
    num_workers: int = 1,
    quant: str = "none",
) -> dict[str, Any]:
    """Convert an in-memory PyTorch/NumPy state dict into scan-ready JAX DiT params."""
    np_dtype = _resolve_np_dtype(dtype)
    all_keys = set(state_dict.keys())
    put_fn = _make_put_fn(device=device, mesh=mesh)
    tp_size = int(mesh.shape["tp"]) if mesh is not None else 1

    def _get(k: str):
        return state_dict.pop(k) if consume else state_dict[k]

    return _build_dit_params(
        _get,
        all_keys,
        config,
        np_dtype=np_dtype,
        put_fn=put_fn,
        num_workers=1 if consume else num_workers,
        tp_size=tp_size,
        quant=quant,
    )


def _resolve_dit_files(model_path: str | Path) -> tuple[list[Path], Path | None]:
    p = Path(model_path)
    if p.is_file():
        return [p], None
    for single_name in ("turbo.safetensors", "raw.safetensors"):
        cand = p / single_name
        if cand.is_file():
            cfg = p / "transformer" / "config.json"
            return [cand], (cfg if cfg.is_file() else None)
    tdir = p / "transformer" if (p / "transformer").is_dir() else p
    cfg = tdir / "config.json"
    idx_file = tdir / "diffusion_pytorch_model.safetensors.index.json"
    if idx_file.is_file():
        wmap = json.loads(idx_file.read_text(encoding="utf-8")).get("weight_map", {})
        files = [tdir / name for name in sorted(set(wmap.values()))]
        return files, (cfg if cfg.is_file() else None)
    shards = sorted(tdir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No DiT .safetensors checkpoint found under {p}")
    return shards, (cfg if cfg.is_file() else None)


def shard_dit_params(params: dict[str, Any], mesh: Any, tp_axis: str = "tp") -> dict[str, Any]:
    """Shard DiT params across a 1D TPU ``Mesh`` (replicated for 1-device mesh)."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P

    rep = NamedSharding(mesh, P())
    sharded: dict[str, Any] = {}
    for k, v in params.items():
        if k == "blocks":
            blk_out = {}
            for bk, bv in v.items():
                if bk in ("qkvgate_w", "mlp_gate_up_w"):
                    spec = P(None, None, tp_axis)
                elif bk in ("qkvgate_s", "mlp_gate_up_s"):
                    spec = P(None, tp_axis)
                elif bk in ("wo_w", "mlp_down_w"):
                    spec = P(None, tp_axis, None)
                else:
                    spec = P()
                blk_out[bk] = jax.device_put(bv, NamedSharding(mesh, spec))
            sharded[k] = blk_out
        else:
            sharded[k] = jax.tree_util.tree_map(lambda arr: jax.device_put(arr, rep), v)
    return sharded


def load_dit_weights(
    model_path: str | Path,
    *,
    config: Krea2DiTConfig | None = None,
    dtype: str = "bfloat16",
    device: Any | None = None,
    mesh: Any | None = None,
    num_workers: int | None = None,
    quant: str = "none",
) -> tuple[Krea2DiTConfig, dict[str, Any]]:
    """Parallel-stream Krea 2 DiT safetensors directly into TPU HBM (zero PyTorch)."""
    files, cfg_path = _resolve_dit_files(model_path)
    if config is None:
        config = Krea2DiTConfig.from_diffusers_json(cfg_path) if cfg_path is not None else Krea2DiTConfig()

    np_dtype = _resolve_np_dtype(dtype)
    workers = _default_num_workers(num_workers)
    put_fn = _make_put_fn(device=device, mesh=mesh)
    tp_size = int(mesh.shape["tp"]) if mesh is not None else 1

    reader = _LazySafetensorsReader(files)
    try:
        params = _build_dit_params(
            reader.get,
            reader.keys(),
            config,
            np_dtype=np_dtype,
            put_fn=put_fn,
            num_workers=workers,
            tp_size=tp_size,
            quant=quant,
        )
    finally:
        reader.close()
        gc.collect()
    return config, params


# ---------------------------------------------------------------------------
# 2. Qwen3-VL-4B Text Tower Weight Builder & Parallel Streaming Loader
# ---------------------------------------------------------------------------


def _build_qwen3_text_params(
    get_tensor: Callable[[str], Any],
    all_keys: set[str],
    config: Qwen3TextConfig,
    *,
    np_dtype: np.dtype,
    put_fn: Callable[[np.ndarray, Any | None], Any],
    num_workers: int = 1,
) -> dict[str, Any]:
    if "model.language_model.embed_tokens.weight" in all_keys:
        prefix = "model.language_model"
    elif "language_model.embed_tokens.weight" in all_keys:
        prefix = "language_model"
    else:
        prefix = ""

    def _p(suffix: str) -> str:
        return f"{prefix}.{suffix}" if prefix else suffix

    def _w_np(key: str, *, transpose: bool = False) -> np.ndarray:
        arr = _as_np(get_tensor(key), np_dtype)
        if transpose:
            arr = np.swapaxes(arr, -1, -2)
        return arr

    num_active = config.num_active_layers
    h_dim = config.hidden_size
    q_dim = config.num_attention_heads * config.head_dim
    kv_dim = config.num_key_value_heads * config.head_dim
    inter = config.intermediate_size

    def _run_parallel(fn: Callable[[int], None]) -> None:
        if num_workers <= 1 or num_active <= 1:
            for i in range(num_active):
                fn(i)
            return
        with ThreadPoolExecutor(max_workers=min(num_workers, num_active)) as ex:
            list(ex.map(fn, range(num_active)))

    def _stream_1d(suffix: str, length: int):
        buf = np.empty((num_active, length), dtype=np_dtype)

        def _fill(i: int):
            buf[i] = _w_np(_p(f"layers.{i}.{suffix}"))

        _run_parallel(_fill)
        res = put_fn(buf, None)
        del buf
        return res

    def _stream_qkv():
        buf = np.empty((num_active, h_dim, q_dim + 2 * kv_dim), dtype=np_dtype)

        def _fill(i: int):
            lp = _p(f"layers.{i}.self_attn")
            buf[i, :, :q_dim] = _w_np(f"{lp}.q_proj.weight", transpose=True)
            buf[i, :, q_dim : q_dim + kv_dim] = _w_np(f"{lp}.k_proj.weight", transpose=True)
            buf[i, :, q_dim + kv_dim :] = _w_np(f"{lp}.v_proj.weight", transpose=True)

        _run_parallel(_fill)
        res = put_fn(buf, None)
        del buf
        return res

    def _stream_2d(suffix: str, shape_2d: tuple[int, int]):
        buf = np.empty((num_active, shape_2d[0], shape_2d[1]), dtype=np_dtype)

        def _fill(i: int):
            buf[i] = _w_np(_p(f"layers.{i}.{suffix}"), transpose=True)

        _run_parallel(_fill)
        res = put_fn(buf, None)
        del buf
        return res

    def _stream_mlp_gu():
        buf = np.empty((num_active, h_dim, 2 * inter), dtype=np_dtype)

        def _fill(i: int):
            lp = _p(f"layers.{i}.mlp")
            buf[i, :, :inter] = _w_np(f"{lp}.gate_proj.weight", transpose=True)
            buf[i, :, inter:] = _w_np(f"{lp}.up_proj.weight", transpose=True)

        _run_parallel(_fill)
        res = put_fn(buf, None)
        del buf
        return res

    layers = {
        "input_layernorm": _stream_1d("input_layernorm.weight", h_dim),
        "post_attention_layernorm": _stream_1d("post_attention_layernorm.weight", h_dim),
        "q_norm": _stream_1d("self_attn.q_norm.weight", config.head_dim),
        "k_norm": _stream_1d("self_attn.k_norm.weight", config.head_dim),
        "qkv_w": _stream_qkv(),
        "o_w": _stream_2d("self_attn.o_proj.weight", (q_dim, h_dim)),
        "mlp_gate_up_w": _stream_mlp_gu(),
        "mlp_down_w": _stream_2d("mlp.down_proj.weight", (inter, h_dim)),
    }

    inv_freq = (
        1.0
        / np.power(
            np.float64(config.rope_theta),
            np.arange(0, config.head_dim, 2, dtype=np.float64) / float(config.head_dim),
        )
    ).astype(np.float32)

    params: dict[str, Any] = {
        "embed_tokens": put_fn(np.ascontiguousarray(_w_np(_p("embed_tokens.weight"))), None),
        "layers": layers,
        "inv_freq": put_fn(inv_freq, None),
    }
    if max(config.select_layers) >= config.num_hidden_layers and _p("norm.weight") in all_keys:
        params["final_norm"] = put_fn(np.ascontiguousarray(_w_np(_p("norm.weight"))), None)
    return params


def build_qwen3_text_params_from_state_dict(
    state_dict: dict[str, Any],
    config: Qwen3TextConfig,
    *,
    dtype: Any = "bfloat16",
    consume: bool = False,
    device: Any | None = None,
    num_workers: int = 1,
) -> dict[str, Any]:
    np_dtype = _resolve_np_dtype(dtype)
    all_keys = set(state_dict.keys())
    put_fn = _make_put_fn(device=device)

    def _get(k: str):
        return state_dict.pop(k) if consume else state_dict[k]

    return _build_qwen3_text_params(
        _get, all_keys, config, np_dtype=np_dtype, put_fn=put_fn, num_workers=1 if consume else num_workers
    )


def load_qwen3_text_weights(
    model_path: str | Path,
    *,
    config: Qwen3TextConfig | None = None,
    dtype: str = "bfloat16",
    device: Any | None = None,
    num_workers: int | None = None,
) -> tuple[Qwen3TextConfig, dict[str, Any]]:
    """Parallel-stream Qwen3-VL text tower safetensors (skipping vision tower & unused layer 35)."""
    p = Path(model_path)
    te_dir = p / "text_encoder" if (p / "text_encoder").is_dir() else p
    cfg_path = te_dir / "config.json"
    if config is None:
        config = Qwen3TextConfig.from_json(cfg_path) if cfg_path.is_file() else Qwen3TextConfig()

    idx_file = te_dir / "model.safetensors.index.json"
    if idx_file.is_file():
        wmap = json.loads(idx_file.read_text(encoding="utf-8")).get("weight_map", {})
        files = [te_dir / name for name in sorted(set(wmap.values()))]
    else:
        files = sorted(te_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No text_encoder .safetensors found under {te_dir}")

    np_dtype = _resolve_np_dtype(dtype)
    workers = _default_num_workers(num_workers)
    put_fn = _make_put_fn(device=device)

    reader = _LazySafetensorsReader(files)
    try:
        params = _build_qwen3_text_params(
            reader.get,
            reader.keys(),
            config,
            np_dtype=np_dtype,
            put_fn=put_fn,
            num_workers=workers,
        )
    finally:
        reader.close()
        gc.collect()
    return config, params


# ---------------------------------------------------------------------------
# 3. AutoencoderKLQwenImage Decoder Weight Builder & Loader (T=1 2D-reduced)
# ---------------------------------------------------------------------------


def _build_vae_decoder_params(
    get_tensor: Callable[[str], Any],
    all_keys: set[str],
    config: QwenImageVAEConfig,
    *,
    np_dtype: np.dtype,
    put_fn: Callable[[np.ndarray, Any | None], Any],
) -> dict[str, Any]:
    f32 = np.dtype(np.float32)

    def _conv_2d_hwio(key: str):
        arr = _as_np(get_tensor(key), np_dtype)
        # CausalConv3d [O, I, kT, kH, kW] -> last temporal slice [:, :, -1, :, :] for T=1.
        if arr.ndim == 5:
            arr = arr[:, :, -1, :, :]
        return put_fn(np.ascontiguousarray(np.transpose(arr, (2, 3, 1, 0))), None)

    def _vec(key: str, *, is_f32: bool = False):
        arr = _as_np(get_tensor(key), f32 if is_f32 else np_dtype).reshape(-1)
        return put_fn(np.ascontiguousarray(arr), None)

    def _load_resnet(prefix: str) -> dict[str, Any]:
        d = {
            "norm1": _vec(f"{prefix}.norm1.gamma", is_f32=True),
            "conv1_w": _conv_2d_hwio(f"{prefix}.conv1.weight"),
            "conv1_b": _vec(f"{prefix}.conv1.bias"),
            "norm2": _vec(f"{prefix}.norm2.gamma", is_f32=True),
            "conv2_w": _conv_2d_hwio(f"{prefix}.conv2.weight"),
            "conv2_b": _vec(f"{prefix}.conv2.bias"),
        }
        sc_key = f"{prefix}.conv_shortcut.weight"
        if sc_key in all_keys:
            d["shortcut_w"] = _conv_2d_hwio(sc_key)
            d["shortcut_b"] = _vec(f"{prefix}.conv_shortcut.bias")
        return d

    up_blocks = []
    num_stages = len(config.dim_mult)
    for i in range(num_stages):
        resnets = [
            _load_resnet(f"decoder.up_blocks.{i}.resnets.{j}")
            for j in range(config.num_res_blocks + 1)
        ]
        stage: dict[str, Any] = {"resnets": resnets}
        up_w_key = f"decoder.up_blocks.{i}.upsamplers.0.resample.1.weight"
        if up_w_key in all_keys:
            stage["upsample_w"] = _conv_2d_hwio(up_w_key)
            stage["upsample_b"] = _vec(f"decoder.up_blocks.{i}.upsamplers.0.resample.1.bias")
        up_blocks.append(stage)

    return {
        "post_quant_conv_w": _conv_2d_hwio("post_quant_conv.weight"),
        "post_quant_conv_b": _vec("post_quant_conv.bias"),
        "conv_in_w": _conv_2d_hwio("decoder.conv_in.weight"),
        "conv_in_b": _vec("decoder.conv_in.bias"),
        "mid_resnets": [
            _load_resnet("decoder.mid_block.resnets.0"),
            _load_resnet("decoder.mid_block.resnets.1"),
        ],
        "mid_attn": {
            "norm": _vec("decoder.mid_block.attentions.0.norm.gamma", is_f32=True),
            "to_qkv_w": _conv_2d_hwio("decoder.mid_block.attentions.0.to_qkv.weight"),
            "to_qkv_b": _vec("decoder.mid_block.attentions.0.to_qkv.bias"),
            "proj_w": _conv_2d_hwio("decoder.mid_block.attentions.0.proj.weight"),
            "proj_b": _vec("decoder.mid_block.attentions.0.proj.bias"),
        },
        "up_blocks": up_blocks,
        "norm_out": _vec("decoder.norm_out.gamma", is_f32=True),
        "conv_out_w": _conv_2d_hwio("decoder.conv_out.weight"),
        "conv_out_b": _vec("decoder.conv_out.bias"),
    }


def build_vae_decoder_params_from_state_dict(
    state_dict: dict[str, Any],
    config: QwenImageVAEConfig,
    *,
    dtype: Any = "bfloat16",
    consume: bool = False,
    device: Any | None = None,
) -> dict[str, Any]:
    np_dtype = _resolve_np_dtype(dtype)
    all_keys = set(state_dict.keys())
    put_fn = _make_put_fn(device=device)

    def _get(k: str):
        return state_dict.pop(k) if consume else state_dict[k]

    return _build_vae_decoder_params(_get, all_keys, config, np_dtype=np_dtype, put_fn=put_fn)


def load_vae_decoder_weights(
    model_path: str | Path,
    *,
    config: QwenImageVAEConfig | None = None,
    dtype: str = "bfloat16",
    device: Any | None = None,
) -> tuple[QwenImageVAEConfig, dict[str, Any]]:
    """Stream AutoencoderKLQwenImage decoder weights directly to TPU (T=1 2D-reduced)."""
    p = Path(model_path)
    vae_dir = p / "vae" if (p / "vae").is_dir() else p
    cfg_path = vae_dir / "config.json"
    if config is None:
        config = QwenImageVAEConfig.from_json(cfg_path) if cfg_path.is_file() else QwenImageVAEConfig()

    files = sorted(vae_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No VAE .safetensors found under {vae_dir}")

    np_dtype = _resolve_np_dtype(dtype)
    put_fn = _make_put_fn(device=device)

    reader = _LazySafetensorsReader(files)
    try:
        params = _build_vae_decoder_params(
            reader.get, reader.keys(), config, np_dtype=np_dtype, put_fn=put_fn
        )
    finally:
        reader.close()
        gc.collect()
    return config, params
