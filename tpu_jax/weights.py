"""Zero-PyTorch streaming safetensors weight loaders for Krea 2 (JAX/TPU).

Supports:
1. DiT (``SingleStreamDiT``, 12B):
   - Official native checkpoint (``turbo.safetensors`` / ``raw.safetensors``)
   - HuggingFace Diffusers sharded ``transformer/`` directory
   Streams keys lazily from safetensors on CPU, casts linear weights to target
   dtype (``bfloat16``), keeps RMSNorm scales in ``float32`` with ``+ 1.0``
   pre-folded, fuses QKVGate (KV-head grouped) and Gate+Up SwiGLU projections,
   and stacks all 28 blocks along axis 0 for ``jax.lax.scan``.
2. Qwen3-VL-4B Text Encoder (``text_encoder/model.safetensors``):
   - Reads ONLY ``language_model.*`` active layers ``0..(max(select_layers)-1)``
     (skips ~1 GB vision tower and unused layer 35 + final norm).
   - Fuses QKV and Gate+Up projections and stacks active layers for ``lax.scan``.
3. Qwen-Image VAE Decoder (``vae/diffusion_pytorch_model.safetensors``):
   - Reads ONLY ``post_quant_conv.*`` and ``decoder.*`` (skips encoder and
     unused ``time_conv`` for T=1).
   - Slices causal 3D conv kernels at ``t=-1`` and transposes ``OIHW -> HWIO``
     for TPU-native NHWC 2D convolutions.
4. Optional 1D Tensor Parallel sharding across multi-chip TPU meshes.
"""

from __future__ import annotations

import gc
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tpu_jax.config import Krea2DiTConfig, Qwen3TextConfig, QwenImageVAEConfig
from tpu_jax.dit import precompute_rope_omegas


def _resolve_jnp_dtype(dtype: Any):
    import jax.numpy as jnp

    if not isinstance(dtype, str):
        return dtype
    mapping = {
        "bf16": jnp.bfloat16,
        "bfloat16": jnp.bfloat16,
        "fp32": jnp.float32,
        "float32": jnp.float32,
        "fp16": jnp.float16,
        "float16": jnp.float16,
    }
    key = dtype.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype {dtype!r}; expected one of {tuple(mapping)}.")
    return mapping[key]


class _LazySafetensorsReader:
    """Lazy multi-shard safetensors reader via ``framework='flax'`` on CPU (zero PyTorch)."""

    def __init__(self, files: list[Path]):
        from safetensors import safe_open

        self._handles = [safe_open(str(p), framework="flax", device="cpu") for p in files]
        self._key_to_handle: dict[str, Any] = {}
        for h in self._handles:
            for k in h.keys():
                self._key_to_handle[k] = h

    def keys(self) -> set[str]:
        return set(self._key_to_handle.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._key_to_handle

    def get(self, key: str):
        if key not in self._key_to_handle:
            raise KeyError(f"Weight key not found in safetensors: {key}")
        return self._key_to_handle[key].get_tensor(key)

    def close(self) -> None:
        self._key_to_handle.clear()
        self._handles.clear()


def _fuse_qkvgate_kv_grouped(
    wq_t,
    wk_t,
    wv_t,
    wgate_t,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
):
    """Pack ``[wq^T, wk^T, wv^T, wgate^T]`` into ``[dim, kv_heads * (2*g + 2) * head_dim]``."""
    import jax.numpy as jnp

    dim = wq_t.shape[0]
    group = num_heads // num_kv_heads
    q_g = wq_t.reshape(dim, num_kv_heads, group * head_dim)
    k_g = wk_t.reshape(dim, num_kv_heads, head_dim)
    v_g = wv_t.reshape(dim, num_kv_heads, head_dim)
    g_g = wgate_t.reshape(dim, num_kv_heads, group * head_dim)
    packed = jnp.concatenate([q_g, k_g, v_g, g_g], axis=-1)
    return packed.reshape(dim, -1)


def _fuse_mlp_gate_up_paired(wgate_t, wup_t):
    """Pack ``[wgate^T, wup^T]`` into ``[dim, 2 * mlp_dim]`` interleaved as ``(gate_j, up_j)``."""
    import jax.numpy as jnp

    dim, mlp_dim = wgate_t.shape
    return jnp.stack([wgate_t, wup_t], axis=-1).reshape(dim, 2 * mlp_dim)


# ---------------------------------------------------------------------------
# 1. Krea 2 SingleStreamDiT Weight Builder & Loader
# ---------------------------------------------------------------------------


def _build_dit_params(
    get_tensor: Callable[[str], Any],
    all_keys: set[str],
    config: Krea2DiTConfig,
    *,
    dtype: Any,
) -> dict[str, Any]:
    import jax.numpy as jnp

    is_diffusers = "transformer_blocks.0.attn.to_q.weight" in all_keys

    def _w(key: str, *, transpose: bool = False, cast_dtype: Any = dtype):
        arr = jnp.asarray(get_tensor(key))
        if cast_dtype is not None and arr.dtype != cast_dtype:
            arr = arr.astype(cast_dtype)
        if transpose:
            arr = jnp.swapaxes(arr, -1, -2)
        return arr

    def _norm(key: str):
        # Krea 2 RMSNorm uses (scale.float() + 1.0); both native & diffusers store raw scale.
        arr = jnp.asarray(get_tensor(key), dtype=jnp.float32)
        return arr + jnp.float32(1.0)

    def _load_text_fusion_block(prefix: str, *, diffusers: bool) -> dict[str, Any]:
        if diffusers:
            wq = _w(f"{prefix}.attn.to_q.weight", transpose=True)
            wk = _w(f"{prefix}.attn.to_k.weight", transpose=True)
            wv = _w(f"{prefix}.attn.to_v.weight", transpose=True)
            wg = _w(f"{prefix}.attn.to_gate.weight", transpose=True)
            wo = _w(f"{prefix}.attn.to_out.0.weight", transpose=True)
            qn = _norm(f"{prefix}.attn.norm_q.weight")
            kn = _norm(f"{prefix}.attn.norm_k.weight")
            mg = _w(f"{prefix}.ff.gate.weight", transpose=True)
            mu = _w(f"{prefix}.ff.up.weight", transpose=True)
            md = _w(f"{prefix}.ff.down.weight", transpose=True)
            pn = _norm(f"{prefix}.norm1.weight")
            pon = _norm(f"{prefix}.norm2.weight")
        else:
            wq = _w(f"{prefix}.attn.wq.weight", transpose=True)
            wk = _w(f"{prefix}.attn.wk.weight", transpose=True)
            wv = _w(f"{prefix}.attn.wv.weight", transpose=True)
            wg = _w(f"{prefix}.attn.gate.weight", transpose=True)
            wo = _w(f"{prefix}.attn.wo.weight", transpose=True)
            qn = _norm(f"{prefix}.attn.qknorm.qnorm.scale")
            kn = _norm(f"{prefix}.attn.qknorm.knorm.scale")
            mg = _w(f"{prefix}.mlp.gate.weight", transpose=True)
            mu = _w(f"{prefix}.mlp.up.weight", transpose=True)
            md = _w(f"{prefix}.mlp.down.weight", transpose=True)
            pn = _norm(f"{prefix}.prenorm.scale")
            pon = _norm(f"{prefix}.postnorm.scale")

        return {
            "prenorm": pn,
            "postnorm": pon,
            "qnorm": qn,
            "knorm": kn,
            "qkvgate_w": _fuse_qkvgate_kv_grouped(
                wq,
                wk,
                wv,
                wg,
                num_heads=config.txtheads,
                num_kv_heads=config.txtkvheads,
                head_dim=config.text_head_dim,
            ),
            "wo_w": wo,
            "mlp_gate_up_w": _fuse_mlp_gate_up_paired(mg, mu),
            "mlp_down_w": md,
        }

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
    txtfusion_proj_w = _w(f"{tf_prefix}.projector.weight", transpose=True)

    # Stack 28 main SingleStreamBlocks field-by-field to minimize peak host RAM.
    blk_prefix = "transformer_blocks" if is_diffusers else "blocks"

    def _stack_list(fn):
        items = [fn(i) for i in range(config.layers)]
        stacked = jnp.stack(items, axis=0)
        items.clear()
        return stacked

    def _load_qkvg(i: int):
        p = f"{blk_prefix}.{i}"
        if is_diffusers:
            wq = _w(f"{p}.attn.to_q.weight", transpose=True)
            wk = _w(f"{p}.attn.to_k.weight", transpose=True)
            wv = _w(f"{p}.attn.to_v.weight", transpose=True)
            wg = _w(f"{p}.attn.to_gate.weight", transpose=True)
        else:
            wq = _w(f"{p}.attn.wq.weight", transpose=True)
            wk = _w(f"{p}.attn.wk.weight", transpose=True)
            wv = _w(f"{p}.attn.wv.weight", transpose=True)
            wg = _w(f"{p}.attn.gate.weight", transpose=True)
        return _fuse_qkvgate_kv_grouped(
            wq, wk, wv, wg, num_heads=config.heads, num_kv_heads=config.kvheads, head_dim=config.head_dim
        )

    def _load_mlp_gu(i: int):
        p = f"{blk_prefix}.{i}"
        if is_diffusers:
            mg = _w(f"{p}.ff.gate.weight", transpose=True)
            mu = _w(f"{p}.ff.up.weight", transpose=True)
        else:
            mg = _w(f"{p}.mlp.gate.weight", transpose=True)
            mu = _w(f"{p}.mlp.up.weight", transpose=True)
        return _fuse_mlp_gate_up_paired(mg, mu)

    blocks = {
        "prenorm": _stack_list(
            lambda i: _norm(f"{blk_prefix}.{i}.norm1.weight" if is_diffusers else f"{blk_prefix}.{i}.prenorm.scale")
        ),
        "postnorm": _stack_list(
            lambda i: _norm(f"{blk_prefix}.{i}.norm2.weight" if is_diffusers else f"{blk_prefix}.{i}.postnorm.scale")
        ),
        "qnorm": _stack_list(
            lambda i: _norm(
                f"{blk_prefix}.{i}.attn.norm_q.weight"
                if is_diffusers
                else f"{blk_prefix}.{i}.attn.qknorm.qnorm.scale"
            )
        ),
        "knorm": _stack_list(
            lambda i: _norm(
                f"{blk_prefix}.{i}.attn.norm_k.weight"
                if is_diffusers
                else f"{blk_prefix}.{i}.attn.qknorm.knorm.scale"
            )
        ),
        "mod_lin": _stack_list(
            lambda i: _w(
                f"{blk_prefix}.{i}.scale_shift_table" if is_diffusers else f"{blk_prefix}.{i}.mod.lin"
            ).reshape(-1)
        ),
        "qkvgate_w": _stack_list(_load_qkvg),
        "wo_w": _stack_list(
            lambda i: _w(
                f"{blk_prefix}.{i}.attn.to_out.0.weight" if is_diffusers else f"{blk_prefix}.{i}.attn.wo.weight",
                transpose=True,
            )
        ),
        "mlp_gate_up_w": _stack_list(_load_mlp_gu),
        "mlp_down_w": _stack_list(
            lambda i: _w(
                f"{blk_prefix}.{i}.ff.down.weight" if is_diffusers else f"{blk_prefix}.{i}.mlp.down.weight",
                transpose=True,
            )
        ),
    }

    # Precomputed FP32 sinusoidal time frequencies and 3D axial RoPE omegas.
    half = config.tdim // 2
    time_freqs = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float32) / float(half)).astype(np.float32)
    rope_omegas = tuple(jnp.asarray(o, dtype=jnp.float32) for o in precompute_rope_omegas(config.axes_dims, config.theta, config.ntk))

    if is_diffusers:
        first_w = _w("img_in.weight", transpose=True)
        first_b = _w("img_in.bias")
        tmlp_fc1_w = _w("time_embed.linear_1.weight", transpose=True)
        tmlp_fc1_b = _w("time_embed.linear_1.bias")
        tmlp_fc2_w = _w("time_embed.linear_2.weight", transpose=True)
        tmlp_fc2_b = _w("time_embed.linear_2.bias")
        tproj_w = _w("time_mod_proj.weight", transpose=True)
        tproj_b = _w("time_mod_proj.bias")
        txtmlp_norm = _norm("txt_in.norm.weight")
        txtmlp_fc1_w = _w("txt_in.linear_1.weight", transpose=True)
        txtmlp_fc1_b = _w("txt_in.linear_1.bias")
        txtmlp_fc2_w = _w("txt_in.linear_2.weight", transpose=True)
        txtmlp_fc2_b = _w("txt_in.linear_2.bias")
        last_norm = _norm("final_layer.norm.weight")
        last_mod_lin = _w("final_layer.scale_shift_table").reshape(2, config.features)
        last_linear_w = _w("final_layer.linear.weight", transpose=True)
        last_linear_b = _w("final_layer.linear.bias")
    else:
        first_w = _w("first.weight", transpose=True)
        first_b = _w("first.bias")
        tmlp_fc1_w = _w("tmlp.0.weight", transpose=True)
        tmlp_fc1_b = _w("tmlp.0.bias")
        tmlp_fc2_w = _w("tmlp.2.weight", transpose=True)
        tmlp_fc2_b = _w("tmlp.2.bias")
        tproj_w = _w("tproj.1.weight", transpose=True)
        tproj_b = _w("tproj.1.bias")
        txtmlp_norm = _norm("txtmlp.0.scale")
        txtmlp_fc1_w = _w("txtmlp.1.weight", transpose=True)
        txtmlp_fc1_b = _w("txtmlp.1.bias")
        txtmlp_fc2_w = _w("txtmlp.3.weight", transpose=True)
        txtmlp_fc2_b = _w("txtmlp.3.bias")
        last_norm = _norm("last.norm.scale")
        last_mod_lin = _w("last.modulation.lin").reshape(2, config.features)
        last_linear_w = _w("last.linear.weight", transpose=True)
        last_linear_b = _w("last.linear.bias")

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
        "time_freqs": jnp.asarray(time_freqs, dtype=jnp.float32),
        "rope_omegas": rope_omegas,
    }


def build_dit_params_from_state_dict(
    state_dict: dict[str, Any],
    config: Krea2DiTConfig,
    *,
    dtype: Any = "bfloat16",
    consume: bool = False,
) -> dict[str, Any]:
    """Convert an in-memory PyTorch/NumPy state dict into scan-ready JAX DiT params."""
    jnp_dtype = _resolve_jnp_dtype(dtype)
    all_keys = set(state_dict.keys())

    def _get(k: str):
        return state_dict.pop(k) if consume else state_dict[k]

    return _build_dit_params(_get, all_keys, config, dtype=jnp_dtype)


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
) -> tuple[Krea2DiTConfig, dict[str, Any]]:
    """Stream Krea 2 DiT safetensors directly to CPU -> TPU (zero PyTorch)."""
    import jax

    files, cfg_path = _resolve_dit_files(model_path)
    if config is None:
        config = Krea2DiTConfig.from_diffusers_json(cfg_path) if cfg_path is not None else Krea2DiTConfig()

    jnp_dtype = _resolve_jnp_dtype(dtype)
    cpu_devices = jax.devices("cpu")
    cpu_ctx = jax.default_device(cpu_devices[0]) if cpu_devices else nullcontext()

    reader = _LazySafetensorsReader(files)
    try:
        with cpu_ctx:
            params = _build_dit_params(reader.get, reader.keys(), config, dtype=jnp_dtype)
    finally:
        reader.close()
        gc.collect()

    if mesh is not None:
        params = shard_dit_params(params, mesh)
        gc.collect()
    elif device is not None:
        params = jax.device_put(params, device)
        gc.collect()
    return config, params


# ---------------------------------------------------------------------------
# 2. Qwen3-VL-4B Text Tower Weight Builder & Loader
# ---------------------------------------------------------------------------


def _build_qwen3_text_params(
    get_tensor: Callable[[str], Any],
    all_keys: set[str],
    config: Qwen3TextConfig,
    *,
    dtype: Any,
) -> dict[str, Any]:
    import jax.numpy as jnp

    # Support both "language_model.*" (text_encoder/model.safetensors) and
    # "model.language_model.*" (Qwen3VLForConditionalGeneration state_dict).
    if "model.language_model.embed_tokens.weight" in all_keys:
        prefix = "model.language_model"
    elif "language_model.embed_tokens.weight" in all_keys:
        prefix = "language_model"
    else:
        prefix = ""

    def _p(suffix: str) -> str:
        return f"{prefix}.{suffix}" if prefix else suffix

    def _w(key: str, *, transpose: bool = False):
        arr = jnp.asarray(get_tensor(key))
        if dtype is not None and arr.dtype != dtype:
            arr = arr.astype(dtype)
        if transpose:
            arr = jnp.swapaxes(arr, -1, -2)
        return arr

    num_active = config.num_active_layers

    def _stack(fn):
        items = [fn(i) for i in range(num_active)]
        out = jnp.stack(items, axis=0)
        items.clear()
        return out

    def _load_qkv(i: int):
        lp = _p(f"layers.{i}.self_attn")
        wq = _w(f"{lp}.q_proj.weight", transpose=True)
        wk = _w(f"{lp}.k_proj.weight", transpose=True)
        wv = _w(f"{lp}.v_proj.weight", transpose=True)
        return jnp.concatenate([wq, wk, wv], axis=-1)

    def _load_mlp_gu(i: int):
        lp = _p(f"layers.{i}.mlp")
        wg = _w(f"{lp}.gate_proj.weight", transpose=True)
        wu = _w(f"{lp}.up_proj.weight", transpose=True)
        return jnp.concatenate([wg, wu], axis=-1)

    layers = {
        "input_layernorm": _stack(lambda i: _w(_p(f"layers.{i}.input_layernorm.weight"))),
        "post_attention_layernorm": _stack(lambda i: _w(_p(f"layers.{i}.post_attention_layernorm.weight"))),
        "q_norm": _stack(lambda i: _w(_p(f"layers.{i}.self_attn.q_norm.weight"))),
        "k_norm": _stack(lambda i: _w(_p(f"layers.{i}.self_attn.k_norm.weight"))),
        "qkv_w": _stack(_load_qkv),
        "o_w": _stack(lambda i: _w(_p(f"layers.{i}.self_attn.o_proj.weight"), transpose=True)),
        "mlp_gate_up_w": _stack(_load_mlp_gu),
        "mlp_down_w": _stack(lambda i: _w(_p(f"layers.{i}.mlp.down_proj.weight"), transpose=True)),
    }

    inv_freq = (
        1.0
        / np.power(
            np.float64(config.rope_theta),
            np.arange(0, config.head_dim, 2, dtype=np.float64) / float(config.head_dim),
        )
    ).astype(np.float32)

    params: dict[str, Any] = {
        "embed_tokens": _w(_p("embed_tokens.weight")),
        "layers": layers,
        "inv_freq": jnp.asarray(inv_freq, dtype=jnp.float32),
    }
    if max(config.select_layers) >= config.num_hidden_layers and _p("norm.weight") in all_keys:
        params["final_norm"] = _w(_p("norm.weight"))
    return params


def build_qwen3_text_params_from_state_dict(
    state_dict: dict[str, Any],
    config: Qwen3TextConfig,
    *,
    dtype: Any = "bfloat16",
    consume: bool = False,
) -> dict[str, Any]:
    jnp_dtype = _resolve_jnp_dtype(dtype)
    all_keys = set(state_dict.keys())

    def _get(k: str):
        return state_dict.pop(k) if consume else state_dict[k]

    return _build_qwen3_text_params(_get, all_keys, config, dtype=jnp_dtype)


def load_qwen3_text_weights(
    model_path: str | Path,
    *,
    config: Qwen3TextConfig | None = None,
    dtype: str = "bfloat16",
    device: Any | None = None,
) -> tuple[Qwen3TextConfig, dict[str, Any]]:
    """Stream Qwen3-VL text tower safetensors (skipping vision tower & unused layer 35)."""
    import jax

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

    jnp_dtype = _resolve_jnp_dtype(dtype)
    cpu_devices = jax.devices("cpu")
    cpu_ctx = jax.default_device(cpu_devices[0]) if cpu_devices else nullcontext()

    reader = _LazySafetensorsReader(files)
    try:
        with cpu_ctx:
            params = _build_qwen3_text_params(reader.get, reader.keys(), config, dtype=jnp_dtype)
    finally:
        reader.close()
        gc.collect()

    if device is not None:
        params = jax.device_put(params, device)
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
    dtype: Any,
) -> dict[str, Any]:
    import jax.numpy as jnp

    def _conv_2d_hwio(key: str):
        arr = jnp.asarray(get_tensor(key))
        if dtype is not None and arr.dtype != dtype:
            arr = arr.astype(dtype)
        # CausalConv3d [O, I, kT, kH, kW] -> take last temporal slice [:, :, -1, :, :] for T=1.
        if arr.ndim == 5:
            arr = arr[:, :, -1, :, :]
        # OIHW -> HWIO for NHWC conv_general_dilated
        return jnp.transpose(arr, (2, 3, 1, 0))

    def _vec(key: str, *, f32: bool = False):
        arr = jnp.asarray(get_tensor(key)).reshape(-1)
        target = jnp.float32 if f32 else dtype
        if target is not None and arr.dtype != target:
            arr = arr.astype(target)
        return arr

    def _load_resnet(prefix: str) -> dict[str, Any]:
        d = {
            "norm1": _vec(f"{prefix}.norm1.gamma", f32=True),
            "conv1_w": _conv_2d_hwio(f"{prefix}.conv1.weight"),
            "conv1_b": _vec(f"{prefix}.conv1.bias"),
            "norm2": _vec(f"{prefix}.norm2.gamma", f32=True),
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
            "norm": _vec("decoder.mid_block.attentions.0.norm.gamma", f32=True),
            "to_qkv_w": _conv_2d_hwio("decoder.mid_block.attentions.0.to_qkv.weight"),
            "to_qkv_b": _vec("decoder.mid_block.attentions.0.to_qkv.bias"),
            "proj_w": _conv_2d_hwio("decoder.mid_block.attentions.0.proj.weight"),
            "proj_b": _vec("decoder.mid_block.attentions.0.proj.bias"),
        },
        "up_blocks": up_blocks,
        "norm_out": _vec("decoder.norm_out.gamma", f32=True),
        "conv_out_w": _conv_2d_hwio("decoder.conv_out.weight"),
        "conv_out_b": _vec("decoder.conv_out.bias"),
    }


def build_vae_decoder_params_from_state_dict(
    state_dict: dict[str, Any],
    config: QwenImageVAEConfig,
    *,
    dtype: Any = "bfloat16",
    consume: bool = False,
) -> dict[str, Any]:
    jnp_dtype = _resolve_jnp_dtype(dtype)
    all_keys = set(state_dict.keys())

    def _get(k: str):
        return state_dict.pop(k) if consume else state_dict[k]

    return _build_vae_decoder_params(_get, all_keys, config, dtype=jnp_dtype)


def load_vae_decoder_weights(
    model_path: str | Path,
    *,
    config: QwenImageVAEConfig | None = None,
    dtype: str = "bfloat16",
    device: Any | None = None,
) -> tuple[QwenImageVAEConfig, dict[str, Any]]:
    """Stream AutoencoderKLQwenImage decoder weights (T=1 2D-reduced, zero PyTorch)."""
    import jax

    p = Path(model_path)
    vae_dir = p / "vae" if (p / "vae").is_dir() else p
    cfg_path = vae_dir / "config.json"
    if config is None:
        config = QwenImageVAEConfig.from_json(cfg_path) if cfg_path.is_file() else QwenImageVAEConfig()

    files = sorted(vae_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No VAE .safetensors found under {vae_dir}")

    jnp_dtype = _resolve_jnp_dtype(dtype)
    cpu_devices = jax.devices("cpu")
    cpu_ctx = jax.default_device(cpu_devices[0]) if cpu_devices else nullcontext()

    reader = _LazySafetensorsReader(files)
    try:
        with cpu_ctx:
            params = _build_vae_decoder_params(reader.get, reader.keys(), config, dtype=jnp_dtype)
    finally:
        reader.close()
        gc.collect()

    if device is not None:
        params = jax.device_put(params, device)
        gc.collect()
    return config, params
