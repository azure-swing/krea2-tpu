#!/usr/bin/env python3
"""Native JAX/XLA TPU inference runner & benchmark suite for Krea 2 (Turbo / Raw).

Zero PyTorch runtime dependency. Performance & utilization stack:
- RAM-backed ``tmpfs`` downloads by default on Linux TPU (auto-mounts up to
  64 GiB tmpfs on Colab, skipping duplicate 26 GB diffusers shards -> ~35.7 GiB).
- Parallel pre-allocated NumPy/bfloat16 safetensors reads pipelined with async
  field-by-field H2D ``jax.device_put`` with TPU-aligned contiguous layouts
  (zero stride-2 SwiGLU / RoPE shuffles, pre-folded Q-scale & RMSNorm+1).
- Text token compaction into static buckets + image-first ``[img | txt | pad]``
  layout enabling compile-time static SplashAttention masks (zero VPU
  ``SegmentIds`` overhead across 94-97% of image-to-image tiles).
- Decoupled 256-aligned GEMM sequences + asymmetric SplashAttention tiling
  ``(block_q, block_kv, block_kv_compute)`` + configurable ``lax.scan`` unroll.
- Single-launch head-merged batched SplashAttention + prompt deduplication for
  Midjourney-style multi-seed batch generation (``--num-images 4 --save-grid``).
- Explicit ``Array.delete()`` via ``release_tpu_pytree()`` & resident VAE.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _tpu_peak_bf16_tflops(device_kind: str, num_chips: int = 1) -> float | None:
    kind = device_kind.lower()
    per_chip: float | None = None
    if "v6" in kind or "trillium" in kind:
        per_chip = 918.0
    elif "v5e" in kind or "v5 lite" in kind:
        per_chip = 197.0
    elif "v5p" in kind:
        per_chip = 459.0
    elif "v4" in kind:
        per_chip = 275.0
    return per_chip * num_chips if per_chip is not None else None


def _memory_stats_gib(device) -> dict[str, float | None]:
    try:
        stats = device.memory_stats()
    except Exception:
        return {"bytes_in_use_gib": None, "peak_bytes_in_use_gib": None, "bytes_limit_gib": None}
    if not stats:
        return {"bytes_in_use_gib": None, "peak_bytes_in_use_gib": None, "bytes_limit_gib": None}
    g = 1024.0**3
    return {
        "bytes_in_use_gib": round(stats["bytes_in_use"] / g, 3) if "bytes_in_use" in stats else None,
        "peak_bytes_in_use_gib": (
            round(stats["peak_bytes_in_use"] / g, 3) if "peak_bytes_in_use" in stats else None
        ),
        "bytes_limit_gib": round(stats["bytes_limit"] / g, 3) if "bytes_limit" in stats else None,
    }


def _should_keep_vae_resident(mode: str, device: Any) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    limit = _memory_stats_gib(device).get("bytes_limit_gib")
    return bool(limit is not None and limit >= 20.0)


def _resolve_splash_config(
    args,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int]:
    """Resolve ``(block_q, block_kv, block_kv_compute)`` from CLI flags and resolution.

    TPU v6e-1 empirical defaults (from microbench_1k/2k on JAX 0.12.1):
    - 1K (~1 MP): 1024x1024x512
    - 2K (>=2.25 MP): 1024x2048x512
    """
    is_hi_res = (width * height) >= (1536 * 1536)
    base = args.splash_tile
    if base is not None:
        bq = args.splash_block_q if args.splash_block_q is not None else base
        bkv = args.splash_block_kv if args.splash_block_kv is not None else base
        bkvc = args.splash_block_kv_compute if args.splash_block_kv_compute is not None else min(bq, bkv)
    else:
        bq = args.splash_block_q if args.splash_block_q is not None else 1024
        bkv = args.splash_block_kv if args.splash_block_kv is not None else (2048 if is_hi_res else 1024)
        bkvc = args.splash_block_kv_compute if args.splash_block_kv_compute is not None else 512
    return int(bq), int(bkv), int(bkvc)


def _parse_splash_configs(spec: str | None, fallback_tiles: str) -> list[tuple[int, int, int]]:
    if spec:
        out: list[tuple[int, int, int]] = []
        for item in spec.split(","):
            item = item.strip().lower()
            if not item:
                continue
            parts = [int(p) for p in item.split("x")]
            if len(parts) == 1:
                out.append((parts[0], parts[0], parts[0]))
            elif len(parts) == 2:
                out.append((parts[0], parts[1], min(parts[0], parts[1])))
            elif len(parts) == 3:
                out.append((parts[0], parts[1], parts[2]))
            else:
                raise ValueError(f"Invalid splash config {item!r}; use BQ or BQxBKV or BQxBKVxBKVC.")
        return out
    return [(int(x.strip()), int(x.strip()), int(x.strip())) for x in fallback_tiles.split(",") if x.strip()]


def _parse_int_list(spec: str) -> list[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def _configure_jax(args) -> tuple[Any, Any | None, list[Any]]:
    import jax
    from jax.sharding import Mesh

    if getattr(args, "compilation_cache_dir", None):
        jax.config.update("jax_compilation_cache_dir", str(args.compilation_cache_dir))

    devices = jax.devices()
    dev0 = devices[0]
    tp = getattr(args, "tp", 1)
    if tp is None or tp <= 0:
        tp = len(devices)
    mesh = None
    if tp > 1:
        if tp > len(devices):
            raise ValueError(f"--tp={tp} exceeds available JAX devices ({len(devices)}).")
        mesh = Mesh(np.array(devices[:tp]), axis_names=("tp",))
    return dev0, mesh, devices[:tp]


def _resolve_model_root(
    args,
    *,
    components: tuple[str, ...] = ("dit", "text_encoder", "tokenizer", "vae"),
) -> tuple[Path, dict[str, Any]]:
    from tpu_jax.model_cache import host_memory_stats_gib, resolve_model_bundle

    t0 = time.perf_counter()
    if args.model:
        root, backend = Path(args.model), "local"
    elif args.hf_repo:
        token = args.hf_token
        if not token and args.hf_token_file:
            token = Path(args.hf_token_file).read_text(encoding="utf-8").strip()
        root, backend = resolve_model_bundle(
            args.hf_repo,
            components=components,
            token=token,
            local_files_only=args.local_files_only,
            disk_cache_dir=args.download_dir,
            prefer_ram=not args.disk_cache,
        )
    else:
        raise ValueError("Provide either --model <local_dir> or --hf-repo <repo_id>.")

    info = {
        "model_root": str(root),
        "cache_backend": backend,
        "resolve_seconds": round(time.perf_counter() - t0, 3),
        "host_mem_after_resolve": host_memory_stats_gib(),
    }
    print(f"[model_cache] {json.dumps(info)}", flush=True)
    return root, info


def _make_init_latents(
    *,
    batch: int,
    width: int,
    height: int,
    channels: int = 16,
    compression: int = 8,
    patch: int = 2,
    seed: int = 0,
    init_latents_path: str | None = None,
) -> np.ndarray:
    from tpu_jax.scheduler import roundup

    align = compression * patch
    w_a = roundup(width, align)
    h_a = roundup(height, align)
    shape = (batch, channels, h_a // compression, w_a // compression)
    if init_latents_path:
        arr = np.load(init_latents_path).astype(np.float32)
        if tuple(arr.shape) != shape:
            raise ValueError(f"init_latents shape {arr.shape} != expected {shape}")
        return arr
    slices = []
    for i in range(batch):
        rng = np.random.default_rng(seed + i)
        slices.append(rng.standard_normal((1, *shape[1:])).astype(np.float32))
    return np.concatenate(slices, axis=0)


def _pixels_to_u8(pixels_nhwc: np.ndarray) -> np.ndarray:
    arr = np.clip(pixels_nhwc, -1.0, 1.0) * 0.5 + 0.5
    return np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)


def _save_images(
    pixels_nhwc: np.ndarray,
    output_prefix: str | Path,
    *,
    start_idx: int = 0,
    save_grid: bool = False,
) -> list[str]:
    from PIL import Image

    u8 = _pixels_to_u8(pixels_nhwc)
    saved = []
    prefix = str(output_prefix)
    stem = prefix[:-4] if prefix.endswith(".png") else prefix
    for i in range(u8.shape[0]):
        idx = start_idx + i
        if u8.shape[0] == 1 and start_idx == 0 and prefix.endswith(".png"):
            out_path = prefix
        else:
            out_path = f"{stem}_{idx}.png"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(u8[i]).save(out_path)
        saved.append(out_path)

    if save_grid and u8.shape[0] > 1:
        n, h, w, c = u8.shape
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        canvas = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
        for i in range(n):
            r, col = divmod(i, cols)
            canvas[r * h : (r + 1) * h, col * w : (col + 1) * w] = u8[i]
        grid_path = f"{stem}_grid.png" if start_idx == 0 else f"{stem}_grid_{start_idx}.png"
        Path(grid_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(canvas).save(grid_path)
        saved.append(grid_path)
    return saved


def _encode_prompts_stage1(
    model_root: Path,
    prompt_groups: list[list[str]],
    negative_prompt_groups: list[list[str]] | None,
    *,
    dtype: str,
    device: Any,
    num_workers: int | None = None,
    text_attn_impl: str = "xla",
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    import jax
    import jax.numpy as jnp

    from tpu_jax.model_cache import host_memory_stats_gib
    from tpu_jax.text_encoder import Krea2Tokenizer, make_text_encode_fn
    from tpu_jax.weights import load_qwen3_text_weights, release_tpu_pytree

    host_before = host_memory_stats_gib()
    hbm_before = _memory_stats_gib(device)

    t0 = time.perf_counter()
    te_cfg, te_params = load_qwen3_text_weights(
        model_root, dtype=dtype, device=device, num_workers=num_workers
    )
    jax.block_until_ready(te_params)
    load_sec = time.perf_counter() - t0

    tokenizer = Krea2Tokenizer(model_root, config=te_cfg)
    encode_fn = make_text_encode_fn(te_cfg, attn_impl=text_attn_impl)

    results: list[dict[str, np.ndarray]] = []
    encode_pos_times = []
    encode_neg_times = []
    for idx, prompts in enumerate(prompt_groups):
        pos_ids, pos_mask = tokenizer.tokenize(prompts)
        t1 = time.perf_counter()
        pos_h, pos_m = encode_fn(
            te_params,
            jax.device_put(jnp.asarray(pos_ids, dtype=jnp.int32), device),
            jax.device_put(jnp.asarray(pos_mask, dtype=jnp.bool_), device),
        )
        pos_h.block_until_ready()
        encode_pos_times.append(round(time.perf_counter() - t1, 4))
        entry: dict[str, np.ndarray] = {
            "pos_context": np.asarray(pos_h, dtype=np.float32),
            "pos_txtmask": np.asarray(pos_m, dtype=np.bool_),
        }
        release_tpu_pytree((pos_h, pos_m))

        if negative_prompt_groups is not None:
            neg_ids, neg_mask = tokenizer.tokenize(negative_prompt_groups[idx])
            t2 = time.perf_counter()
            neg_h, neg_m = encode_fn(
                te_params,
                jax.device_put(jnp.asarray(neg_ids, dtype=jnp.int32), device),
                jax.device_put(jnp.asarray(neg_mask, dtype=jnp.bool_), device),
            )
            neg_h.block_until_ready()
            encode_neg_times.append(round(time.perf_counter() - t2, 4))
            entry["neg_context"] = np.asarray(neg_h, dtype=np.float32)
            entry["neg_txtmask"] = np.asarray(neg_m, dtype=np.bool_)
            release_tpu_pytree((neg_h, neg_m))
        results.append(entry)

    hbm_peak = _memory_stats_gib(device)
    drop_sec, freed_leaves = release_tpu_pytree(te_params)
    del te_params
    gc.collect()
    hbm_after_drop = _memory_stats_gib(device)
    host_after = host_memory_stats_gib()

    metrics = {
        "load_and_h2d_seconds": round(load_sec, 3),
        "encode_pos_seconds": encode_pos_times,
        "encode_neg_seconds": encode_neg_times,
        "drop_tpu_seconds": round(drop_sec, 4),
        "freed_leaves": freed_leaves,
        "hbm_before": hbm_before,
        "hbm_peak": hbm_peak,
        "hbm_after_drop": hbm_after_drop,
        "host_mem_before": host_before,
        "host_mem_after": host_after,
    }
    return results, metrics


def _load_cached_prompt_embeds(
    data: Any,
    *,
    num_images: int,
    do_cfg: bool,
) -> dict[str, np.ndarray]:
    """Load and validate cached ``.npz`` prompt embeddings without silently dropping rows.

    Accepts either a singleton batch (``batch == 1``, broadcast across ``num_images``) or
    per-image embeddings matching ``num_images`` (``batch == num_images``).
    """
    files = set(data.files) if hasattr(data, "files") else set(data.keys())
    required = ["pos_context", "pos_txtmask"]
    if do_cfg:
        required.extend(["neg_context", "neg_txtmask"])
    missing = [k for k in required if k not in files]
    if missing:
        raise ValueError(f"Cached prompt_embeds is missing required keys: {missing} (found {sorted(files)}).")

    def _validate_pair(prefix: str) -> tuple[np.ndarray, np.ndarray]:
        ctx = np.asarray(data[f"{prefix}_context"])
        mask = np.asarray(data[f"{prefix}_txtmask"], dtype=np.bool_)
        if ctx.ndim != 4:
            raise ValueError(
                f"Cached {prefix}_context must be 4D [B, L, N_layers, D], got shape {ctx.shape}."
            )
        if mask.ndim != 2:
            raise ValueError(f"Cached {prefix}_txtmask must be 2D [B, L], got shape {mask.shape}.")
        if ctx.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Cached {prefix}_context shape {ctx.shape} does not match {prefix}_txtmask shape {mask.shape}."
            )
        b = int(ctx.shape[0])
        if b not in (1, num_images):
            raise ValueError(
                f"Cached {prefix}_context batch size {b} is incompatible with --num-images={num_images} "
                f"(expected 1 or {num_images})."
            )
        return ctx, mask

    pos_ctx, pos_mask = _validate_pair("pos")
    out: dict[str, np.ndarray] = {
        "pos_context": pos_ctx,
        "pos_txtmask": pos_mask,
    }
    if do_cfg:
        neg_ctx, neg_mask = _validate_pair("neg")
        out["neg_context"] = neg_ctx
        out["neg_txtmask"] = neg_mask
    return out


def _prepare_embeds_for_dit(
    embeds: dict[str, np.ndarray],
    *,
    do_cfg: bool,
    compact_text: bool,
    text_buckets: tuple[int, ...],
) -> tuple[dict[str, np.ndarray], int | None, int | None]:
    """Optionally compact valid text tokens into static buckets and return exact valid counts."""
    from tpu_jax.dit import compact_text_embeddings

    out = dict(embeds)
    pos_ctx, pos_mask = out["pos_context"], out["pos_txtmask"]
    if compact_text:
        pos_ctx, pos_mask, pos_common = compact_text_embeddings(pos_ctx, pos_mask, buckets=text_buckets)
        out["pos_context"] = pos_ctx
        out["pos_txtmask"] = pos_mask
    else:
        pos_common = None

    neg_common = None
    if do_cfg and "neg_context" in out and "neg_txtmask" in out:
        neg_ctx, neg_mask = out["neg_context"], out["neg_txtmask"]
        if compact_text:
            neg_ctx, neg_mask, neg_common = compact_text_embeddings(neg_ctx, neg_mask, buckets=text_buckets)
            out["neg_context"] = neg_ctx
            out["neg_txtmask"] = neg_mask
    return out, pos_common, neg_common


def run_probe(args) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from tpu_jax.model_cache import host_memory_stats_gib

    dev0, _mesh, active_devs = _configure_jax(args)
    device_kind = getattr(dev0, "device_kind", str(dev0))
    info: dict[str, Any] = {
        "jax_version": jax.__version__,
        "default_backend": jax.default_backend(),
        "device_count": len(jax.devices()),
        "active_tp_devices": len(active_devs),
        "device_kind": device_kind,
        "peak_bf16_tflops_est": _tpu_peak_bf16_tflops(device_kind, len(active_devs)),
        "platform": dev0.platform,
        "host_memory": host_memory_stats_gib(),
        "memory_stats_before": _memory_stats_gib(dev0),
        "resident_vae_auto": _should_keep_vae_resident("auto", dev0),
    }

    a = jnp.ones((1024, 1024), dtype=jnp.bfloat16)
    b = jnp.ones((1024, 1024), dtype=jnp.bfloat16)
    fn = jax.jit(lambda x, y: x @ y)
    t0 = time.perf_counter()
    c = fn(a, b)
    c.block_until_ready()
    info["matmul_1024_cold_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    t1 = time.perf_counter()
    c = fn(a, b)
    c.block_until_ready()
    info["matmul_1024_warm_ms"] = round((time.perf_counter() - t1) * 1000.0, 2)

    for name, mod_path in (
        ("splash_attention", "jax.experimental.pallas.ops.tpu.splash_attention"),
        ("flash_attention", "jax.experimental.pallas.ops.tpu.flash_attention"),
    ):
        try:
            __import__(mod_path)
            info[f"{name}_available"] = True
        except Exception as exc:
            info[f"{name}_available"] = False
            info[f"{name}_error"] = str(exc)

    info["memory_stats_after"] = _memory_stats_gib(dev0)
    print(json.dumps(info, indent=2), flush=True)
    return info


def run_generate(args) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from tpu_jax.dit import make_denoise_fn
    from tpu_jax.model_cache import host_memory_stats_gib
    from tpu_jax.scheduler import compute_timesteps, estimate_step_tflops, roundup
    from tpu_jax.vae import make_vae_decode_fn
    from tpu_jax.weights import (
        load_dit_weights,
        load_vae_decoder_weights,
        release_tpu_pytree,
    )

    dev0, mesh, active_devs = _configure_jax(args)
    tp_size = len(active_devs)
    device_kind = getattr(dev0, "device_kind", str(dev0))
    peak_tflops = _tpu_peak_bf16_tflops(device_kind, tp_size)

    if args.prompts_file:
        prompt_list = [
            line.strip()
            for line in Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        prompt_list = [args.prompt]

    components: tuple[str, ...] = (
        ("dit", "vae") if args.prompt_embeds else ("dit", "text_encoder", "tokenizer", "vae")
    )
    model_root, cache_info = _resolve_model_root(args, components=components)
    do_cfg = args.cfg > 0.0

    # --- Optional Resident VAE (~125 MB BF16 on >= 20 GiB HBM) ---
    resident_vae = _should_keep_vae_resident(args.resident_vae, dev0)
    vae_cfg = None
    vae_params = None
    vae_load_sec = 0.0
    if resident_vae:
        t_v0 = time.perf_counter()
        vae_cfg, vae_params = load_vae_decoder_weights(model_root, dtype=args.vae_dtype, device=dev0)
        jax.block_until_ready(vae_params)
        vae_load_sec = time.perf_counter() - t_v0
        print(
            f"[vae:resident] loaded in {vae_load_sec:.3f}s, hbm={_memory_stats_gib(dev0)}",
            flush=True,
        )

    # --- Stage 1: Encode each unique prompt ONCE (batch=1; DiT broadcasts over --num-images!) ---
    if args.prompt_embeds:
        with np.load(args.prompt_embeds, allow_pickle=False) as data:
            cached_entry = _load_cached_prompt_embeds(data, num_images=args.num_images, do_cfg=do_cfg)
        embeds_list = [cached_entry]
        stage1_metrics: dict[str, Any] = {
            "loaded_from_cache": str(args.prompt_embeds),
            "cached_pos_batch": int(cached_entry["pos_context"].shape[0]),
            "cached_neg_batch": int(cached_entry["neg_context"].shape[0]) if do_cfg else None,
        }
    else:
        prompt_groups = [[p] for p in prompt_list]
        neg_groups = [[args.negative_prompt] for _ in prompt_list] if do_cfg else None
        embeds_list, stage1_metrics = _encode_prompts_stage1(
            model_root,
            prompt_groups,
            neg_groups,
            dtype=args.dtype,
            device=dev0,
            num_workers=args.load_workers,
            text_attn_impl="xla" if dev0.platform == "tpu" else "einsum",
        )
        if args.save_embeds:
            Path(args.save_embeds).parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.save_embeds, **embeds_list[0])
    print(f"[stage1:text_encoder] {json.dumps(stage1_metrics)}", flush=True)

    # --- Stage 2: Parallel-stream DiT once & denoise ALL prompts ---
    dit_host_before = host_memory_stats_gib()
    dit_hbm_before = _memory_stats_gib(dev0)
    t_load0 = time.perf_counter()
    dit_path = Path(args.dit_path) if args.dit_path else model_root
    dit_cfg, dit_params = load_dit_weights(
        dit_path,
        dtype=args.dtype,
        device=None if mesh is not None else dev0,
        mesh=mesh,
        num_workers=args.load_workers,
        quant=args.quant,
    )
    jax.block_until_ready(dit_params)
    dit_load_sec = time.perf_counter() - t_load0

    ts_np = compute_timesteps(
        width=args.width,
        height=args.height,
        steps=args.steps,
        y1=args.y1,
        y2=args.y2,
        mu=args.mu,
    )
    ts_j = jnp.asarray(ts_np, dtype=jnp.float32)
    if mesh is None:
        ts_j = jax.device_put(ts_j, dev0)

    attn_impl = args.attn_impl
    if attn_impl == "auto":
        attn_impl = "splash" if (dev0.platform == "tpu" and mesh is None) else "xla"

    bq, bkv, bkvc = _resolve_splash_config(args, width=args.width, height=args.height)
    compact_text = args.compact_text != "off"
    text_buckets = tuple(_parse_int_list(args.text_buckets))
    align = 8 * dit_cfg.patch
    img_len = (roundup(args.height, align) // align) * (roundup(args.width, align) // align)

    jnp_dtype = jnp.bfloat16 if args.dtype in ("bf16", "bfloat16") else jnp.float32
    all_run_times: list[float] = []
    run_records: list[dict[str, Any]] = []
    final_latents_list: list[np.ndarray] = []
    denoise_fn_cache: dict[tuple[Any, ...], Any] = {}
    executed_compile_keys: set[tuple[Any, ...]] = set()
    used_txt_len = 512
    used_neg_txt_len: int | None = None
    used_static_mask = False

    for p_idx, raw_embeds in enumerate(embeds_list):
        embeds, pos_common_txt, neg_common_txt = _prepare_embeds_for_dit(
            raw_embeds,
            do_cfg=do_cfg,
            compact_text=compact_text,
            text_buckets=text_buckets,
        )
        pos_txt_len = int(embeds["pos_txtmask"].shape[1])
        neg_txt_len = int(embeds["neg_txtmask"].shape[1]) if do_cfg else None
        pos_batch = int(embeds["pos_txtmask"].shape[0])
        neg_batch = int(embeds["neg_txtmask"].shape[0]) if do_cfg else None
        used_txt_len = pos_txt_len
        used_neg_txt_len = neg_txt_len

        # Decide static lazy PrefixMask vs dynamic SegmentIds.
        use_static = False
        if attn_impl == "splash" and compact_text and pos_common_txt is not None:
            if args.splash_mask_mode == "static":
                use_static = True
            elif args.splash_mask_mode == "auto":
                # Single prompt (incl. MJ batch-N): static mask eliminates VPU SegmentIds across all image tiles.
                # Multi-prompt file: segment mode avoids recompiling per exact prompt token count.
                use_static = (len(embeds_list) == 1) or (pos_common_txt == pos_txt_len)
        used_static_mask = use_static
        pos_static_len = (img_len + pos_common_txt) if use_static and pos_common_txt is not None else None
        neg_static_len = (
            (img_len + neg_common_txt) if (use_static and do_cfg and neg_common_txt is not None) else None
        )

        cache_key = (pos_txt_len, neg_txt_len, pos_batch, neg_batch, pos_static_len, neg_static_len)
        if cache_key not in denoise_fn_cache:
            denoise_fn_cache[cache_key] = make_denoise_fn(
                dit_cfg,
                width=args.width,
                height=args.height,
                guidance_scale=args.cfg,
                attn_impl=attn_impl,
                splash_block_q=bq,
                splash_block_kv=bkv,
                splash_block_kv_compute=bkvc,
                pos_static_valid_len=pos_static_len,
                neg_static_valid_len=neg_static_len,
                scan_unroll=args.scan_unroll,
                tp_size=tp_size,
                latent_dtype=jnp.float32,
                donate_latents=True,
            )
        denoise_fn = denoise_fn_cache[cache_key]

        pos_step_tf = estimate_step_tflops(
            dit_cfg,
            width=args.width,
            height=args.height,
            txt_len=pos_txt_len,
            pad_multiple=256,
            batch_size=args.num_images,
            do_cfg=False,
        )
        neg_step_tf = (
            estimate_step_tflops(
                dit_cfg,
                width=args.width,
                height=args.height,
                txt_len=neg_txt_len,
                pad_multiple=256,
                batch_size=args.num_images,
                do_cfg=False,
            )
            if (do_cfg and neg_txt_len is not None)
            else 0.0
        )
        run_step_tflops = pos_step_tf + neg_step_tf

        init_lat_np = _make_init_latents(
            batch=args.num_images,
            width=args.width,
            height=args.height,
            channels=dit_cfg.channels,
            patch=dit_cfg.patch,
            seed=args.seed + p_idx * args.num_images,
            init_latents_path=args.init_latents if p_idx == 0 else None,
        )
        pos_ctx = jnp.asarray(embeds["pos_context"], dtype=jnp_dtype)
        pos_mask = jnp.asarray(embeds["pos_txtmask"], dtype=jnp.bool_)
        neg_ctx = jnp.asarray(embeds["neg_context"], dtype=jnp_dtype) if do_cfg else None
        neg_mask = jnp.asarray(embeds["neg_txtmask"], dtype=jnp.bool_) if do_cfg else None
        if mesh is None:
            pos_ctx, pos_mask = jax.device_put((pos_ctx, pos_mask), dev0)
            if do_cfg:
                neg_ctx, neg_mask = jax.device_put((neg_ctx, neg_mask), dev0)

        n_runs = args.runs if len(embeds_list) == 1 else 1
        saved_lat = None
        for r_idx in range(n_runs):
            lat_in = jnp.asarray(init_lat_np, dtype=jnp.float32)
            if mesh is None:
                lat_in = jax.device_put(lat_in, dev0)
            is_compiled_cold = cache_key not in executed_compile_keys
            t0 = time.perf_counter()
            out_lat = denoise_fn(dit_params, lat_in, pos_ctx, pos_mask, neg_ctx, neg_mask, ts_j)
            out_lat.block_until_ready()
            dt = time.perf_counter() - t0
            executed_compile_keys.add(cache_key)
            all_run_times.append(round(dt, 4))
            run_ach_tf = (run_step_tflops * args.steps) / dt
            run_records.append(
                {
                    "prompt_index": p_idx,
                    "run_index": r_idx,
                    "compiled": is_compiled_cold,
                    "seconds": round(dt, 4),
                    "pos_txt_len": pos_txt_len,
                    "neg_txt_len": neg_txt_len,
                    "pos_static_valid_len": pos_static_len,
                    "neg_static_valid_len": neg_static_len,
                    "step_tflops_est": round(run_step_tflops, 2),
                    "achieved_tflops": round(run_ach_tf, 2),
                    "mfu_pct_bf16": round(100.0 * run_ach_tf / peak_tflops, 2) if peak_tflops else None,
                }
            )
            if r_idx == 0:
                saved_lat = np.asarray(out_lat, dtype=np.float32)
            release_tpu_pytree(out_lat)
        assert saved_lat is not None
        final_latents_list.append(saved_lat)
        release_tpu_pytree((pos_ctx, pos_mask, neg_ctx, neg_mask))

    dit_hbm_peak = _memory_stats_gib(dev0)
    dit_drop_sec, dit_freed = release_tpu_pytree((dit_params, ts_j))
    del dit_params, ts_j
    gc.collect()
    dit_hbm_after = _memory_stats_gib(dev0)
    dit_host_after = host_memory_stats_gib()

    warm_records = [r for r in run_records if not r["compiled"]]
    agg_records = warm_records if warm_records else run_records
    warm_median = float(statistics.median(r["seconds"] for r in agg_records))
    total_agg_tflops = sum(r["step_tflops_est"] * args.steps for r in agg_records)
    total_agg_seconds = sum(r["seconds"] for r in agg_records)
    step_tflops = float(statistics.median(r["step_tflops_est"] for r in agg_records))
    achieved_tflops = total_agg_tflops / total_agg_seconds if total_agg_seconds > 0 else 0.0
    mfu_pct = round(100.0 * achieved_tflops / peak_tflops, 2) if peak_tflops else None

    stage2_metrics = {
        "attn_impl": attn_impl,
        "splash_blocks": f"{bq}x{bkv}x{bkvc}" if attn_impl in ("splash", "flash") else None,
        "splash_mask_mode": "static" if used_static_mask else "segment",
        "compact_text": compact_text,
        "effective_txt_len": used_txt_len,
        "effective_neg_txt_len": used_neg_txt_len,
        "scan_unroll": args.scan_unroll,
        "batch_size": args.num_images,
        "load_and_h2d_seconds": round(dit_load_sec, 3),
        "drop_tpu_seconds": round(dit_drop_sec, 4),
        "freed_leaves": dit_freed,
        "run_seconds": all_run_times,
        "run_records": run_records,
        "has_warm_runs": bool(warm_records),
        "num_compiles": sum(1 for r in run_records if r["compiled"]),
        "cold_seconds": all_run_times[0],
        "warm_median_seconds": round(warm_median, 4),
        "warm_sec_per_image": round(warm_median / max(1, args.num_images), 4),
        "warm_ms_per_step": round(warm_median * 1000.0 / args.steps, 2),
        "warm_ms_per_step_per_image": round(warm_median * 1000.0 / (args.steps * max(1, args.num_images)), 2),
        "step_tflops_est": round(step_tflops, 2),
        "achieved_tflops": round(achieved_tflops, 2),
        "mfu_pct_bf16": mfu_pct,
        "latents_has_nan": any(bool(np.isnan(x).any()) for x in final_latents_list),
        "latents_mean": round(float(np.mean(final_latents_list[0])), 5),
        "latents_std": round(float(np.std(final_latents_list[0])), 5),
        "hbm_before": dit_hbm_before,
        "hbm_peak": dit_hbm_peak,
        "hbm_after_drop": dit_hbm_after,
        "host_mem_before": dit_host_before,
        "host_mem_after": dit_host_after,
    }
    print(f"[stage2:dit] {json.dumps(stage2_metrics)}", flush=True)

    # --- Stage 3: VAE Decode (micro-batched at 2K to avoid peak pixel buffer OOM) ---
    if vae_params is None:
        t_vae0 = time.perf_counter()
        vae_cfg, vae_params = load_vae_decoder_weights(model_root, dtype=args.vae_dtype, device=dev0)
        jax.block_until_ready(vae_params)
        vae_load_sec = time.perf_counter() - t_vae0

    decode_fn = make_vae_decode_fn(vae_cfg, q_chunk_size=args.vae_attn_chunk)
    vae_bs = args.vae_batch_size
    if vae_bs <= 0:
        vae_bs = 1 if (args.width * args.height >= 2_000_000) else max(1, args.num_images)

    saved_paths: list[str] = []
    decode_times: list[float] = []
    has_nan_pix = False
    img_counter = 0
    for lat_np in final_latents_list:
        prompt_pixels: list[np.ndarray] = []
        t_dec_total = 0.0
        for s_idx in range(0, lat_np.shape[0], vae_bs):
            sub_lat = lat_np[s_idx : s_idx + vae_bs]
            lat_j = jax.device_put(jnp.asarray(sub_lat, dtype=jnp.float32), dev0)
            t_dec0 = time.perf_counter()
            pixels_j = decode_fn(vae_params, lat_j)
            pixels_j.block_until_ready()
            t_dec_total += time.perf_counter() - t_dec0
            pix_np = np.asarray(pixels_j, dtype=np.float32)
            has_nan_pix = has_nan_pix or bool(np.isnan(pix_np).any())
            prompt_pixels.append(pix_np)
            release_tpu_pytree((pixels_j, lat_j))
        decode_times.append(round(t_dec_total, 4))
        full_pix = np.concatenate(prompt_pixels, axis=0)
        saved_paths.extend(
            _save_images(full_pix, args.output, start_idx=img_counter, save_grid=args.save_grid)
        )
        img_counter += full_pix.shape[0]

    vae_mem = _memory_stats_gib(dev0)
    release_tpu_pytree(vae_params)
    del vae_params
    gc.collect()

    stage3_metrics = {
        "vae_dtype": args.vae_dtype,
        "resident_vae": resident_vae,
        "vae_batch_size": vae_bs,
        "load_seconds": round(vae_load_sec, 3),
        "decode_seconds": decode_times,
        "pixels_has_nan": has_nan_pix,
        "saved_images": saved_paths,
        "memory_stats_peak": vae_mem,
    }
    print(f"[stage3:vae] {json.dumps(stage3_metrics)}", flush=True)

    report = {
        "jax_version": jax.__version__,
        "device_kind": device_kind,
        "active_tp_devices": tp_size,
        "peak_bf16_tflops": peak_tflops,
        "prompts": prompt_list,
        "width": args.width,
        "height": args.height,
        "num_images": args.num_images,
        "steps": args.steps,
        "cfg": args.cfg,
        "mu": args.mu,
        "seed": args.seed,
        "dtype": args.dtype,
        "model_cache": cache_info,
        "stage1_text_encoder": stage1_metrics,
        "stage2_dit": stage2_metrics,
        "stage3_vae": stage3_metrics,
    }
    if args.save_report:
        rp = Path(args.save_report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_bench(args) -> dict[str, Any]:
    """Matrix benchmark sweeping resolutions, batch sizes, attention configs, mask modes & scan unroll."""
    import jax
    import jax.numpy as jnp

    from tpu_jax.dit import compact_text_embeddings, make_denoise_fn
    from tpu_jax.scheduler import compute_timesteps, estimate_step_tflops, roundup
    from tpu_jax.weights import load_dit_weights, release_tpu_pytree

    dev0, mesh, active_devs = _configure_jax(args)
    tp_size = len(active_devs)
    device_kind = getattr(dev0, "device_kind", str(dev0))
    peak_tflops = _tpu_peak_bf16_tflops(device_kind, tp_size)

    need_te = not (args.prompt_embeds or args.synthetic_embeds)
    comps: tuple[str, ...] = ("dit", "text_encoder", "tokenizer") if need_te else ("dit",)
    model_root, cache_info = _resolve_model_root(args, components=comps)

    batch_sizes = _parse_int_list(args.batch_sizes)

    if args.prompt_embeds:
        with np.load(args.prompt_embeds, allow_pickle=False) as data:
            expected_bs = batch_sizes[0] if len(batch_sizes) == 1 else 1
            cached = _load_cached_prompt_embeds(data, num_images=expected_bs, do_cfg=False)
        pos_ctx_raw = cached["pos_context"]
        pos_mask_raw = cached["pos_txtmask"]
    elif args.synthetic_embeds:
        rng = np.random.default_rng(0)
        pos_ctx_raw = rng.standard_normal((1, 512, 12, 2560)).astype(np.float32) * 0.1
        pos_mask_raw = np.zeros((1, 512), dtype=bool)
        pos_mask_raw[:, :48] = True
        pos_mask_raw[:, -5:] = True
    else:
        embeds_list, _ = _encode_prompts_stage1(
            model_root,
            [[args.prompt]],
            None,
            dtype=args.dtype,
            device=dev0,
            num_workers=args.load_workers,
            text_attn_impl="xla" if dev0.platform == "tpu" else "einsum",
        )
        pos_ctx_raw = embeds_list[0]["pos_context"][:1]
        pos_mask_raw = embeds_list[0]["pos_txtmask"][:1]

    compact_text = args.compact_text != "off"
    text_buckets = tuple(_parse_int_list(args.text_buckets))
    if compact_text:
        pos_ctx_np, pos_mask_np, common_valid_txt = compact_text_embeddings(
            pos_ctx_raw, pos_mask_raw, buckets=text_buckets
        )
    else:
        pos_ctx_np, pos_mask_np, common_valid_txt = pos_ctx_raw, pos_mask_raw, None

    t_load0 = time.perf_counter()
    dit_path = Path(args.dit_path) if args.dit_path else model_root
    dit_cfg, dit_params = load_dit_weights(
        dit_path,
        dtype=args.dtype,
        device=None if mesh is not None else dev0,
        mesh=mesh,
        num_workers=args.load_workers,
        quant=args.quant,
    )
    jax.block_until_ready(dit_params)
    load_sec = time.perf_counter() - t_load0

    jnp_dtype = jnp.bfloat16 if args.dtype in ("bf16", "bfloat16") else jnp.float32
    pos_ctx = jnp.asarray(pos_ctx_np, dtype=jnp_dtype)
    pos_mask = jnp.asarray(pos_mask_np, dtype=jnp.bool_)
    if mesh is None:
        pos_ctx, pos_mask = jax.device_put((pos_ctx, pos_mask), dev0)

    attn_impls = [x.strip() for x in args.attn_impls.split(",") if x.strip()]
    splash_cfgs = _parse_splash_configs(args.splash_configs, args.splash_tiles)
    mask_modes = [x.strip() for x in args.splash_mask_modes.split(",") if x.strip()]
    scan_unrolls = _parse_int_list(args.scan_unrolls)
    resolutions = []
    for item in args.resolutions.split(","):
        item = item.strip().lower()
        if not item:
            continue
        w_s, h_s = item.split("x")
        resolutions.append((int(w_s), int(h_s)))

    align = 8 * dit_cfg.patch
    txt_len = int(pos_mask_np.shape[1])
    cases = []
    for width, height in resolutions:
        img_len = (roundup(height, align) // align) * (roundup(width, align) // align)
        ts_np = compute_timesteps(width=width, height=height, steps=args.steps, mu=args.mu)
        ts_j = jnp.asarray(ts_np, dtype=jnp.float32)
        if mesh is None:
            ts_j = jax.device_put(ts_j, dev0)

        for batch_size in batch_sizes:
            init_lat_np = _make_init_latents(
                batch=batch_size,
                width=width,
                height=height,
                channels=dit_cfg.channels,
                patch=dit_cfg.patch,
                seed=args.seed,
            )
            for attn_impl in attn_impls:
                cfgs = splash_cfgs if attn_impl in ("splash", "flash") else [(512, 512, 512)]
                m_modes = mask_modes if attn_impl == "splash" else ["segment"]
                for bq, bkv, bkvc in cfgs:
                    for m_mode in m_modes:
                        for unroll in scan_unrolls:
                            use_static = m_mode == "static" and compact_text and (common_valid_txt is not None)
                            static_vlen = (img_len + common_valid_txt) if use_static else None
                            case_name = (
                                f"{width}x{height}_b{batch_size}_{attn_impl}"
                                + (f"_{bq}x{bkv}x{bkvc}_{m_mode}" if attn_impl == "splash" else "")
                                + (f"_u{unroll}" if unroll > 1 else "")
                            )
                            denoise_fn = make_denoise_fn(
                                dit_cfg,
                                width=width,
                                height=height,
                                guidance_scale=args.cfg,
                                attn_impl=attn_impl,
                                splash_block_q=bq,
                                splash_block_kv=bkv,
                                splash_block_kv_compute=bkvc,
                                pos_static_valid_len=static_vlen,
                                scan_unroll=unroll,
                                tp_size=tp_size,
                                latent_dtype=jnp.float32,
                                donate_latents=True,
                            )
                            run_times = []
                            err = None
                            last_lat = None
                            try:
                                for r_idx in range(args.runs):
                                    lat_in = jnp.asarray(init_lat_np, dtype=jnp.float32)
                                    if mesh is None:
                                        lat_in = jax.device_put(lat_in, dev0)
                                    t0 = time.perf_counter()
                                    out = denoise_fn(dit_params, lat_in, pos_ctx, pos_mask, None, None, ts_j)
                                    out.block_until_ready()
                                    run_times.append(round(time.perf_counter() - t0, 4))
                                    if r_idx == 0:
                                        last_lat = np.asarray(out, dtype=np.float32)
                                    release_tpu_pytree(out)
                            except Exception as exc:
                                err = f"{type(exc).__name__}: {exc}"

                            if err is not None:
                                entry = {
                                    "case": case_name,
                                    "width": width,
                                    "height": height,
                                    "batch_size": batch_size,
                                    "attn_impl": attn_impl,
                                    "splash_blocks": f"{bq}x{bkv}x{bkvc}" if attn_impl in ("splash", "flash") else None,
                                    "splash_mask_mode": m_mode if attn_impl == "splash" else None,
                                    "scan_unroll": unroll,
                                    "error": err,
                                    "memory_stats": _memory_stats_gib(dev0),
                                }
                            else:
                                warm_times = run_times[1:] if len(run_times) > 1 else run_times
                                warm_med = float(statistics.median(warm_times))
                                step_tf = estimate_step_tflops(
                                    dit_cfg,
                                    width=width,
                                    height=height,
                                    txt_len=txt_len,
                                    pad_multiple=256,
                                    batch_size=batch_size,
                                    do_cfg=args.cfg > 0.0,
                                )
                                ach_tf = (step_tf * args.steps) / warm_med
                                entry = {
                                    "case": case_name,
                                    "width": width,
                                    "height": height,
                                    "batch_size": batch_size,
                                    "attn_impl": attn_impl,
                                    "splash_blocks": f"{bq}x{bkv}x{bkvc}" if attn_impl in ("splash", "flash") else None,
                                    "splash_mask_mode": ("static" if use_static else "segment") if attn_impl == "splash" else None,
                                    "compact_text": compact_text,
                                    "effective_txt_len": txt_len,
                                    "scan_unroll": unroll,
                                    "steps": args.steps,
                                    "run_seconds": run_times,
                                    "cold_seconds": run_times[0],
                                    "warm_median_seconds": round(warm_med, 4),
                                    "warm_sec_per_image": round(warm_med / batch_size, 4),
                                    "warm_ms_per_step": round(warm_med * 1000.0 / args.steps, 2),
                                    "warm_ms_per_step_per_image": round(
                                        warm_med * 1000.0 / (args.steps * batch_size), 2
                                    ),
                                    "step_tflops_est": round(step_tf, 2),
                                    "achieved_tflops": round(ach_tf, 2),
                                    "mfu_pct_bf16": round(100.0 * ach_tf / peak_tflops, 2) if peak_tflops else None,
                                    "latents_has_nan": bool(np.isnan(last_lat).any()),
                                    "latents_std": round(float(np.std(last_lat)), 5),
                                    "memory_stats": _memory_stats_gib(dev0),
                                }
                            cases.append(entry)
                            print(json.dumps(entry, indent=2), flush=True)

    summary = {
        "jax_version": jax.__version__,
        "device_kind": device_kind,
        "active_tp_devices": tp_size,
        "peak_bf16_tflops": peak_tflops,
        "model_cache": cache_info,
        "load_and_h2d_seconds": round(load_sec, 3),
        "cases": cases,
    }
    if args.save_report:
        rp = Path(args.save_report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_microbench(args) -> dict[str, Any]:
    """Synthetic single-block breakdown benchmark (no checkpoint download required!).

    Measures isolated millisecond timings & TFLOP/s for every component inside one
    Krea 2 SingleStreamBlock (4 MXU GEMMs, QKNorm+RoPE, SplashAttention across tile
    & mask configs, and full 1-block forward) across ``--batch-sizes``.
    """
    import jax
    import jax.numpy as jnp

    from tpu_jax.config import Krea2DiTConfig
    from tpu_jax.dit import (
        _apply_rope,
        _gqa_attention,
        _matmul,
        _rms_norm,
        _single_stream_block_forward,
        _swiglu_op,
    )
    from tpu_jax.scheduler import roundup
    from tpu_jax.weights import release_tpu_pytree

    dev0, _mesh, _active_devs = _configure_jax(args)
    device_kind = getattr(dev0, "device_kind", str(dev0))
    # All microbench tensors are placed on a single chip (dev0), so MFU uses 1-chip peak.
    peak_tflops = _tpu_peak_bf16_tflops(device_kind, 1)
    cfg = Krea2DiTConfig()

    d = cfg.features
    hd = cfg.head_dim
    hq = cfg.heads
    hkv = cfg.kvheads
    qkvg_out = 2 * hq * hd + 2 * hkv * hd
    mlp_d = cfg.mlp_dim
    dtype = jnp.bfloat16 if args.dtype in ("bf16", "bfloat16") else jnp.float32

    align = 8 * cfg.patch
    img_len = (roundup(args.height, align) // align) * (roundup(args.width, align) // align)
    txt_len = 64 if args.compact_text != "off" else 512
    seq_len = roundup(img_len + txt_len, 256)
    valid_len = img_len + min(53, txt_len)

    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    block = {
        "prenorm": jax.device_put(jnp.ones((d,), dtype=jnp.float32), dev0),
        "postnorm": jax.device_put(jnp.ones((d,), dtype=jnp.float32), dev0),
        "qnorm": jax.device_put(jnp.full((hd,), 1.0 / math.sqrt(hd), dtype=jnp.float32), dev0),
        "knorm": jax.device_put(jnp.ones((hd,), dtype=jnp.float32), dev0),
        "mod_lin": jax.device_put(jnp.zeros((6 * d,), dtype=dtype), dev0),
        "qkvgate_w": jax.device_put(jax.random.normal(k1, (d, qkvg_out), dtype=dtype) * 0.02, dev0),
        "wo_w": jax.device_put(jax.random.normal(k2, (d, d), dtype=dtype) * 0.02, dev0),
        "mlp_gate_up_w": jax.device_put(jax.random.normal(k3, (d, 2 * mlp_d), dtype=dtype) * 0.02, dev0),
        "mlp_down_w": jax.device_put(jax.random.normal(k4, (mlp_d, d), dtype=dtype) * 0.02, dev0),
    }
    jax.block_until_ready(block)

    def _time_fn(fn, *fn_args, repeats: int = 10) -> float:
        # 2 warmup + median of `repeats`
        for _ in range(2):
            out = fn(*fn_args)
            jax.block_until_ready(out)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = fn(*fn_args)
            jax.block_until_ready(out)
            times.append((time.perf_counter() - t0) * 1000.0)
        return float(statistics.median(times))

    splash_cfgs = _parse_splash_configs(args.splash_configs, args.splash_tiles)
    mask_modes = [x.strip() for x in args.splash_mask_modes.split(",") if x.strip()]
    batch_sizes = _parse_int_list(args.batch_sizes)

    results: list[dict[str, Any]] = []
    for bs in batch_sizes:
        x = jax.device_put(jnp.ones((bs, seq_len, d), dtype=dtype) * 0.1, dev0)
        tvec = jax.device_put(jnp.zeros((1, 1, 6 * d), dtype=dtype), dev0)
        cos = jax.device_put(jnp.ones((1, seq_len, hd // 2), dtype=jnp.float32), dev0)
        sin = jax.device_put(jnp.zeros((1, seq_len, hd // 2), dtype=jnp.float32), dev0)
        vm_np = np.zeros((1, seq_len), dtype=np.bool_)
        vm_np[:, :valid_len] = True
        vm = jax.device_put(jnp.asarray(vm_np), dev0)
        q_in = jax.device_put(jnp.ones((bs, seq_len, hq, hd), dtype=dtype) * 0.05, dev0)
        k_in = jax.device_put(jnp.ones((bs, seq_len, hkv, hd), dtype=dtype) * 0.05, dev0)
        v_in = jax.device_put(jnp.ones((bs, seq_len, hkv, hd), dtype=dtype) * 0.05, dev0)

        def _rec(name: str, ms: float, flops: float) -> dict[str, Any]:
            tf = (flops / 1e12) / (ms / 1000.0) if ms > 0 else 0.0
            mfu = round(100.0 * tf / peak_tflops, 2) if peak_tflops else None
            return {
                "op": name,
                "ms_1layer": round(ms, 3),
                "ms_28layers_est": round(ms * cfg.layers, 2),
                "tflops": round(tf, 2),
                "mfu_pct": mfu,
            }

        ops = []
        fn_qkvg = jax.jit(lambda a, w: _matmul(a, w))
        ops.append(_rec("gemm_qkvgate", _time_fn(fn_qkvg, x, block["qkvgate_w"]), 2.0 * bs * seq_len * d * qkvg_out))

        fn_wo = jax.jit(lambda a, w: _matmul(a, w))
        ops.append(_rec("gemm_wo", _time_fn(fn_wo, x, block["wo_w"]), 2.0 * bs * seq_len * d * d))

        fn_mgu = jax.jit(lambda a, w: _matmul(a, w))
        ops.append(_rec("gemm_mlp_gate_up", _time_fn(fn_mgu, x, block["mlp_gate_up_w"]), 2.0 * bs * seq_len * d * (2 * mlp_d)))

        fn_swiglu = jax.jit(lambda a, blk: _swiglu_op(a, blk))
        ops.append(_rec("swiglu_full", _time_fn(fn_swiglu, x, block), 6.0 * bs * seq_len * d * mlp_d))

        fn_qkr = jax.jit(
            lambda qq, kk, qn, kn, c, s: (
                _apply_rope(_rms_norm(qq, qn, cfg.norm_eps), c, s),
                _apply_rope(_rms_norm(kk, kn, cfg.norm_eps), c, s),
            )
        )
        ops.append(_rec("qknorm_plus_rope", _time_fn(fn_qkr, q_in, k_in, block["qnorm"], block["knorm"], cos, sin), 0.0))

        attn_flops = 4.0 * bs * seq_len * seq_len * d
        for bq, bkv, bkvc in splash_cfgs:
            for m_mode in mask_modes:
                svl = valid_len if m_mode == "static" else None
                fn_sp = jax.jit(
                    lambda qq, kk, vv, m, _bq=bq, _bkv=bkv, _bkvc=bkvc, _svl=svl: _gqa_attention(
                        qq,
                        kk,
                        vv,
                        m,
                        attn_impl="splash",
                        splash_block_q=_bq,
                        splash_block_kv=_bkv,
                        splash_block_kv_compute=_bkvc,
                        static_valid_len=_svl,
                    )
                )
                try:
                    ms_sp = _time_fn(fn_sp, q_in, k_in, v_in, vm)
                    ops.append(_rec(f"splash_{bq}x{bkv}x{bkvc}_{m_mode}", ms_sp, attn_flops))
                except Exception as exc:
                    ops.append({"op": f"splash_{bq}x{bkv}x{bkvc}_{m_mode}", "error": f"{type(exc).__name__}: {exc}"})

        # Full single block with first splash config + static mask
        bq0, bkv0, bkvc0 = splash_cfgs[0]
        svl0 = valid_len if ("static" in mask_modes) else None
        fn_blk = jax.jit(
            lambda a, blk, tv, c, s, m: _single_stream_block_forward(
                a,
                blk,
                tv,
                c,
                s,
                m,
                num_heads=hq,
                num_kv_heads=hkv,
                head_dim=hd,
                norm_eps=cfg.norm_eps,
                attn_impl="splash",
                splash_block_q=bq0,
                splash_block_kv=bkv0,
                splash_block_kv_compute=bkvc0,
                static_valid_len=svl0,
            )
        )
        total_blk_flops = (
            2.0 * bs * seq_len * d * (3 * d + 2 * hkv * hd)
            + 6.0 * bs * seq_len * d * mlp_d
            + attn_flops
        )
        ops.append(_rec(f"full_block_{bq0}x{bkv0}x{bkvc0}", _time_fn(fn_blk, x, block, tvec, cos, sin, vm), total_blk_flops))

        entry = {
            "width": args.width,
            "height": args.height,
            "seq_len": seq_len,
            "valid_len": valid_len,
            "batch_size": bs,
            "ops": ops,
        }
        results.append(entry)
        print(json.dumps(entry, indent=2), flush=True)
        release_tpu_pytree((x, tvec, cos, sin, vm, q_in, k_in, v_in))

    release_tpu_pytree(block)
    summary = {
        "jax_version": jax.__version__,
        "device_kind": device_kind,
        "microbench_devices": 1,
        "peak_bf16_tflops": peak_tflops,
        "results": results,
    }
    if args.save_report:
        rp = Path(args.save_report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_parity(args) -> dict[str, Any]:
    """Compare single DiT forward across attention backends on real weights."""
    import jax
    import jax.numpy as jnp

    from tpu_jax.dit import (
        compact_text_embeddings,
        dit_forward_precomputed,
        patchify_latents,
        prepare_text_conditioning,
    )
    from tpu_jax.weights import load_dit_weights

    dev0, mesh, active_devs = _configure_jax(args)
    tp_size = len(active_devs)
    model_root, cache_info = _resolve_model_root(args, components=("dit",))
    dit_path = Path(args.dit_path) if args.dit_path else model_root

    t0 = time.perf_counter()
    dit_cfg, dit_params = load_dit_weights(
        dit_path,
        dtype=args.dtype,
        device=None if mesh is not None else dev0,
        mesh=mesh,
        num_workers=args.load_workers,
        quant=args.quant,
    )
    jax.block_until_ready(dit_params)
    load_sec = time.perf_counter() - t0

    rng = np.random.default_rng(args.seed)
    jnp_dtype = jnp.bfloat16 if args.dtype in ("bf16", "bfloat16") else jnp.float32
    pos_ctx_raw = rng.standard_normal((1, 512, 12, 2560)).astype(np.float32) * 0.1
    pos_mask_raw = np.zeros((1, 512), dtype=bool)
    pos_mask_raw[:, :48] = True
    pos_mask_raw[:, -5:] = True
    if args.compact_text != "off":
        pos_ctx_np, pos_mask_np, common_valid = compact_text_embeddings(
            pos_ctx_raw, pos_mask_raw, buckets=tuple(_parse_int_list(args.text_buckets))
        )
    else:
        pos_ctx_np, pos_mask_np, common_valid = pos_ctx_raw, pos_mask_raw, None

    pos_ctx = jnp.asarray(pos_ctx_np, dtype=jnp_dtype)
    pos_mask = jnp.asarray(pos_mask_np, dtype=jnp.bool_)
    lat = jnp.asarray(
        _make_init_latents(batch=args.num_images, width=args.width, height=args.height, seed=args.seed),
        dtype=jnp_dtype,
    )
    tok = patchify_latents(lat, patch=dit_cfg.patch)
    t_val = jnp.asarray([1.0], dtype=jnp.float32)
    if mesh is None:
        pos_ctx, pos_mask, tok, t_val = jax.device_put((pos_ctx, pos_mask, tok, t_val), dev0)

    bq, bkv, bkvc = _resolve_splash_config(args, width=args.width, height=args.height)
    impls = [x.strip() for x in args.attn_impls.split(",") if x.strip()]
    lat_h, lat_w = args.height // 8, args.width // 8
    img_len = (lat_h // dit_cfg.patch) * (lat_w // dit_cfg.patch)
    static_vlen = (img_len + common_valid) if (common_valid is not None and args.splash_mask_mode != "segment") else None
    outputs = {}
    timings_ms = {}
    for impl in impls:
        @jax.jit
        def _single(p, x, t, c, m, _impl=impl):
            txt, cos, sin, jmask, pad_len = prepare_text_conditioning(
                p, dit_cfg, c, m, latent_h=lat_h, latent_w=lat_w, attn_impl=_impl
            )
            return dit_forward_precomputed(
                p,
                dit_cfg,
                x,
                t,
                txt,
                cos,
                sin,
                jmask,
                pad_len=pad_len,
                attn_impl=_impl,
                splash_block_q=bq,
                splash_block_kv=bkv,
                splash_block_kv_compute=bkvc,
                static_valid_len=static_vlen if _impl == "splash" else None,
                scan_unroll=args.scan_unroll,
                tp_size=tp_size,
            )

        out = _single(dit_params, tok, t_val, pos_ctx, pos_mask)
        out.block_until_ready()
        t_s = time.perf_counter()
        out = _single(dit_params, tok, t_val, pos_ctx, pos_mask)
        out.block_until_ready()
        timings_ms[impl] = round((time.perf_counter() - t_s) * 1000.0, 2)
        outputs[impl] = np.asarray(out, dtype=np.float64)

    ref_name = impls[0]
    ref = outputs[ref_name]
    comparisons = {}
    for name in impls[1:]:
        cur = outputs[name]
        diff = np.abs(cur - ref)
        cos = float(np.sum(cur * ref) / (np.linalg.norm(cur) * np.linalg.norm(ref) + 1e-12))
        comparisons[f"{name}_vs_{ref_name}"] = {
            "max_abs_diff": float(np.max(diff)),
            "mean_abs_diff": float(np.mean(diff)),
            "cosine_similarity": round(cos, 8),
            "has_nan": bool(np.isnan(cur).any()),
        }

    report = {
        "device_kind": getattr(dev0, "device_kind", str(dev0)),
        "model_cache": cache_info,
        "load_and_h2d_seconds": round(load_sec, 3),
        "single_forward_warm_ms": timings_ms,
        "comparisons": comparisons,
        "memory_stats": _memory_stats_gib(dev0),
    }
    print(json.dumps(report, indent=2), flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Native JAX/XLA TPU inference runner for Krea 2")
    p.add_argument(
        "--mode",
        choices=("probe", "generate", "bench", "microbench", "parity"),
        default="probe",
    )
    p.add_argument("--model", type=str, default=None, help="Local model snapshot directory")
    p.add_argument("--dit-path", type=str, default=None, help="Optional explicit turbo.safetensors / transformer/ path")
    p.add_argument(
        "--hf-repo", default="krea/Krea-2-Turbo", help="Hugging Face repo ID (default: %(default)s)"
    )
    p.add_argument("--download-dir", type=str, default=None, help="Fallback disk dir when RAM tmpfs is unavailable")
    p.add_argument("--disk-cache", action="store_true", help="Disable RAM tmpfs cache and download to disk")
    p.add_argument("--local-files-only", action="store_true", help="Offline mode (reuse cached snapshot only)")
    p.add_argument("--hf-token", type=str, default=None)
    p.add_argument("--hf-token-file", type=str, default=None)
    p.add_argument("--prompt", type=str, default="A red fox sitting in fresh snow, golden hour, photorealistic")
    p.add_argument("--prompts-file", type=str, default=None, help="File with one prompt per line (single stage load)")
    p.add_argument("--negative-prompt", type=str, default="")
    p.add_argument("--prompt-embeds", type=str, default=None, help="Optional cached .npz prompt embeddings")
    p.add_argument("--save-embeds", type=str, default=None, help="Save Stage 1 embeddings to .npz")
    p.add_argument("--synthetic-embeds", action="store_true", help="Use synthetic embeds for pure DiT bench")
    p.add_argument("--init-latents", type=str, default=None, help="Optional .npy FP32 initial noise")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--steps", type=int, default=8, help="8 for Turbo, 52 for Raw")
    p.add_argument("--cfg", type=float, default=0.0, help="0.0 for Turbo, 3.5 for Raw")
    p.add_argument("--mu", type=float, default=1.15, help="Fixed 1.15 for Turbo; omit/None for dynamic")
    p.add_argument("--y1", type=float, default=0.5)
    p.add_argument("--y2", type=float, default=1.15)
    p.add_argument("--num-images", type=int, default=1, help="Batch size per prompt (e.g. 4 for MJ 4-up)")
    p.add_argument("--save-grid", action="store_true", help="Also save a contact-sheet grid PNG when --num-images > 1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", choices=("bf16", "bfloat16", "fp32", "float32"), default="bf16")
    p.add_argument("--quant", choices=("none", "int8"), default="none", help="Weight-only quantization for DiT")
    p.add_argument("--vae-dtype", choices=("bf16", "bfloat16", "fp32", "float32"), default="bf16")
    p.add_argument("--vae-batch-size", type=int, default=0, help="VAE decode micro-batch (0=auto)")
    p.add_argument("--resident-vae", choices=("auto", "on", "off"), default="auto")
    p.add_argument("--vae-attn-chunk", type=int, default=1024)
    p.add_argument("--load-workers", type=int, default=None, help="Parallel CPU threads for safetensors streaming")
    p.add_argument("--attn-impl", choices=("auto", "splash", "flash", "xla", "einsum"), default="auto")
    p.add_argument("--attn-impls", type=str, default="splash")
    p.add_argument("--splash-tile", type=int, default=None, help="Symmetric Splash tile shorthand (overrides auto)")
    p.add_argument("--splash-block-q", type=int, default=None, help="SplashAttention block_q (default: 1024)")
    p.add_argument("--splash-block-kv", type=int, default=None, help="SplashAttention block_kv (default: 1024@1K, 2048@2K)")
    p.add_argument("--splash-block-kv-compute", type=int, default=None, help="SplashAttention block_kv_compute (default: 512)")
    p.add_argument("--splash-tiles", type=str, default="512,1024", help="Fallback symmetric tiles for bench")
    p.add_argument(
        "--splash-configs",
        type=str,
        default="512x1024x512,1024x1024x512,1024x2048x512",
        help="Comma-separated BQxBKVxBKVC configs for bench/microbench",
    )
    p.add_argument(
        "--splash-mask-mode",
        choices=("auto", "static", "segment"),
        default="auto",
        help="Static compile-time NumpyMask (zero VPU SegmentIds) vs dynamic SegmentIds",
    )
    p.add_argument("--splash-mask-modes", type=str, default="static,segment", help="Mask modes to sweep in bench")
    p.add_argument("--compact-text", choices=("auto", "on", "off"), default="auto", help="Compact valid text tokens")
    p.add_argument("--text-buckets", type=str, default="64,128,256,512", help="Static text bucket lengths")
    p.add_argument("--scan-unroll", type=int, default=1, help="lax.scan unroll factor across the 28 DiT layers")
    p.add_argument("--scan-unrolls", type=str, default="1", help="Unroll factors to sweep in bench (e.g. 1,2,4)")
    p.add_argument("--batch-sizes", type=str, default="1,2,4", help="Batch sizes to sweep in bench/microbench")
    p.add_argument("--resolutions", type=str, default="1024x1024,2048x2048")
    p.add_argument("--tp", type=int, default=1, help="1D tensor parallel chips (1 for single v6e-1)")
    p.add_argument("--runs", type=int, default=2, help="1 cold + (runs-1) warm runs")
    p.add_argument("--compilation-cache-dir", type=str, default=None)
    p.add_argument("--output", type=str, default="krea2_jax_out")
    p.add_argument("--save-report", type=str, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "probe":
        run_probe(args)
        return 0
    if args.mode == "generate":
        run_generate(args)
        return 0
    if args.mode == "bench":
        run_bench(args)
        return 0
    if args.mode == "microbench":
        run_microbench(args)
        return 0
    if args.mode == "parity":
        run_parity(args)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
