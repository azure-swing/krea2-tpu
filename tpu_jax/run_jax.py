#!/usr/bin/env python3
"""Native JAX/XLA TPU inference runner & benchmark suite for Krea 2 (Turbo / Raw).

Zero PyTorch runtime dependency. Orchestrates all three stages natively in JAX
with strict HBM lifecycle management so 12B BF16 Krea 2 fits cleanly on single
TPU v6e-1 (32 GiB HBM) and scales to multi-chip TPU meshes via 1D TP:

  Stage 1: Qwen3-VL-4B text tower (~7.5 GB BF16) -> free HBM
  Stage 2: SingleStreamDiT 12B (~25.0 GB BF16)   -> free HBM
  Stage 3: Qwen-Image VAE decoder (~0.13 GB BF16) -> PNG
"""

from __future__ import annotations

import argparse
import gc
import json
import os
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
    if "v6e" in kind or "trillium" in kind:
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


def download_krea2_snapshot(
    repo_id: str,
    local_dir: str | Path,
    *,
    token: str | None = None,
    checkpoint_file: str = "turbo.safetensors",
) -> Path:
    """Download ONLY the minimal files needed for native JAX inference (~35.7 GB vs ~62 GB).

    Skips duplicate diffusers ``transformer/`` shards when downloading single-file
    ``turbo.safetensors`` / ``raw.safetensors``.
    """
    from huggingface_hub import snapshot_download

    allow_patterns = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        checkpoint_file,
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "text_encoder/model.safetensors.index.json",
        "text_encoder/model-*.safetensors",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ]
    out = snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        token=token or os.environ.get("HF_TOKEN"),
    )
    return Path(out)


def _resolve_model_root(args) -> Path:
    if args.model:
        return Path(args.model)
    if args.hf_repo:
        target = Path(args.download_dir or "/dev/shm/krea-2-turbo")
        ckpt = "raw.safetensors" if "raw" in args.hf_repo.lower() else "turbo.safetensors"
        token = args.hf_token
        if not token and args.hf_token_file:
            token = Path(args.hf_token_file).read_text(encoding="utf-8").strip()
        print(f"[download] Fetching minimal JAX files from {args.hf_repo} -> {target}", flush=True)
        t0 = time.perf_counter()
        root = download_krea2_snapshot(args.hf_repo, target, token=token, checkpoint_file=ckpt)
        print(f"[download] Done in {time.perf_counter() - t0:.1f}s", flush=True)
        return root
    raise ValueError("Provide either --model <local_dir> or --hf-repo <repo_id>.")


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
    # Per-image deterministic normal noise (matching seed + i convention)
    slices = []
    for i in range(batch):
        rng = np.random.default_rng(seed + i)
        slices.append(rng.standard_normal((1, *shape[1:])).astype(np.float32))
    return np.concatenate(slices, axis=0)


def _save_images(pixels_nhwc: np.ndarray, output_prefix: str | Path) -> list[str]:
    from PIL import Image

    arr = np.clip(pixels_nhwc, -1.0, 1.0) * 0.5 + 0.5
    u8 = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
    saved = []
    prefix = str(output_prefix)
    for i in range(u8.shape[0]):
        out_path = f"{prefix}_{i}.png" if u8.shape[0] > 1 or not prefix.endswith(".png") else prefix
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(u8[i]).save(out_path)
        saved.append(out_path)
    return saved


def _encode_prompts_stage1(
    model_root: Path,
    prompts: list[str],
    negative_prompts: list[str] | None,
    *,
    dtype: str,
    device: Any,
    text_attn_impl: str = "xla",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import jax
    import jax.numpy as jnp

    from tpu_jax.text_encoder import Krea2Tokenizer, make_text_encode_fn
    from tpu_jax.weights import load_qwen3_text_weights

    t0 = time.perf_counter()
    te_cfg, te_params = load_qwen3_text_weights(model_root, dtype=dtype, device=device)
    jax.block_until_ready(te_params)
    load_sec = time.perf_counter() - t0

    tokenizer = Krea2Tokenizer(model_root, config=te_cfg)
    encode_fn = make_text_encode_fn(te_cfg, attn_impl=text_attn_impl)

    pos_ids, pos_mask = tokenizer.tokenize(prompts)
    t1 = time.perf_counter()
    pos_h, pos_m = encode_fn(
        te_params,
        jax.device_put(jnp.asarray(pos_ids, dtype=jnp.int32), device),
        jax.device_put(jnp.asarray(pos_mask, dtype=jnp.bool_), device),
    )
    pos_h.block_until_ready()
    encode_pos_sec = time.perf_counter() - t1

    result: dict[str, np.ndarray] = {
        "pos_context": np.asarray(pos_h, dtype=np.float32),
        "pos_txtmask": np.asarray(pos_m, dtype=np.bool_),
    }
    encode_neg_sec = 0.0
    if negative_prompts is not None:
        neg_ids, neg_mask = tokenizer.tokenize(negative_prompts)
        t2 = time.perf_counter()
        neg_h, neg_m = encode_fn(
            te_params,
            jax.device_put(jnp.asarray(neg_ids, dtype=jnp.int32), device),
            jax.device_put(jnp.asarray(neg_mask, dtype=jnp.bool_), device),
        )
        neg_h.block_until_ready()
        encode_neg_sec = time.perf_counter() - t2
        result["neg_context"] = np.asarray(neg_h, dtype=np.float32)
        result["neg_txtmask"] = np.asarray(neg_m, dtype=np.bool_)

    mem = _memory_stats_gib(device)
    del te_params, pos_h, pos_m
    gc.collect()

    metrics = {
        "load_seconds": round(load_sec, 3),
        "encode_pos_seconds": round(encode_pos_sec, 3),
        "encode_neg_seconds": round(encode_neg_sec, 3),
        "memory_stats_peak": mem,
    }
    return result, metrics


def run_probe(args) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

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
        "memory_stats_before": _memory_stats_gib(dev0),
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
    from tpu_jax.scheduler import compute_timesteps, estimate_step_tflops
    from tpu_jax.vae import make_vae_decode_fn
    from tpu_jax.weights import load_dit_weights, load_vae_decoder_weights

    dev0, mesh, active_devs = _configure_jax(args)
    device_kind = getattr(dev0, "device_kind", str(dev0))
    peak_tflops = _tpu_peak_bf16_tflops(device_kind, len(active_devs))
    model_root = _resolve_model_root(args)

    prompts = [args.prompt] * args.num_images
    do_cfg = args.cfg > 0.0
    neg_prompts = [args.negative_prompt] * args.num_images if do_cfg else None

    # --- Stage 1: Text Encoding (or load cached embeds) ---
    if args.prompt_embeds:
        data = np.load(args.prompt_embeds, allow_pickle=False)
        embeds = {k: data[k] for k in data.files}
        stage1_metrics: dict[str, Any] = {"loaded_from_cache": str(args.prompt_embeds)}
    else:
        embeds, stage1_metrics = _encode_prompts_stage1(
            model_root,
            prompts,
            neg_prompts,
            dtype=args.dtype,
            device=dev0,
            text_attn_impl="xla" if dev0.platform == "tpu" else "einsum",
        )
        if args.save_embeds:
            Path(args.save_embeds).parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.save_embeds, **embeds)
    print(f"[stage1:text_encoder] {json.dumps(stage1_metrics)}", flush=True)

    # --- Stage 2: DiT Denoising ---
    t_load0 = time.perf_counter()
    dit_path = Path(args.dit_path) if args.dit_path else model_root
    dit_cfg, dit_params = load_dit_weights(
        dit_path,
        dtype=args.dtype,
        device=None if mesh is not None else dev0,
        mesh=mesh,
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
    init_lat_np = _make_init_latents(
        batch=args.num_images,
        width=args.width,
        height=args.height,
        channels=dit_cfg.channels,
        patch=dit_cfg.patch,
        seed=args.seed,
        init_latents_path=args.init_latents,
    )

    jnp_dtype = jnp.bfloat16 if args.dtype in ("bf16", "bfloat16") else jnp.float32
    pos_ctx = jnp.asarray(embeds["pos_context"], dtype=jnp_dtype)
    pos_mask = jnp.asarray(embeds["pos_txtmask"], dtype=jnp.bool_)
    neg_ctx = jnp.asarray(embeds["neg_context"], dtype=jnp_dtype) if do_cfg else None
    neg_mask = jnp.asarray(embeds["neg_txtmask"], dtype=jnp.bool_) if do_cfg else None
    ts_j = jnp.asarray(ts_np, dtype=jnp.float32)

    if mesh is None:
        pos_ctx, pos_mask, ts_j = jax.device_put((pos_ctx, pos_mask, ts_j), dev0)
        if do_cfg:
            neg_ctx, neg_mask = jax.device_put((neg_ctx, neg_mask), dev0)

    attn_impl = args.attn_impl
    if attn_impl == "auto":
        attn_impl = "splash" if (dev0.platform == "tpu" and mesh is None) else "xla"

    denoise_fn = make_denoise_fn(
        dit_cfg,
        width=args.width,
        height=args.height,
        guidance_scale=args.cfg,
        attn_impl=attn_impl,
        splash_block_size=args.splash_tile,
        latent_dtype=jnp.float32,
        donate_latents=True,
    )

    run_times = []
    final_latents_np = None
    for run_idx in range(args.runs):
        lat_in = jnp.asarray(init_lat_np, dtype=jnp.float32)
        if mesh is None:
            lat_in = jax.device_put(lat_in, dev0)
        t0 = time.perf_counter()
        out_lat = denoise_fn(dit_params, lat_in, pos_ctx, pos_mask, neg_ctx, neg_mask, ts_j)
        out_lat.block_until_ready()
        dt = time.perf_counter() - t0
        run_times.append(round(dt, 4))
        if run_idx == 0:
            final_latents_np = np.asarray(out_lat, dtype=np.float32)
        del out_lat

    dit_mem = _memory_stats_gib(dev0)
    del dit_params, pos_ctx, pos_mask, neg_ctx, neg_mask, ts_j
    gc.collect()

    warm_times = run_times[1:] if len(run_times) > 1 else run_times
    warm_median = float(statistics.median(warm_times))
    step_tflops = estimate_step_tflops(
        dit_cfg,
        width=args.width,
        height=args.height,
        txt_len=int(embeds["pos_txtmask"].shape[1]),
        pad_multiple=max(256, args.splash_tile) if attn_impl in ("splash", "flash") else 256,
        batch_size=args.num_images,
        do_cfg=do_cfg,
    )
    achieved_tflops = (step_tflops * args.steps) / warm_median
    mfu_pct = round(100.0 * achieved_tflops / peak_tflops, 2) if peak_tflops else None

    stage2_metrics = {
        "attn_impl": attn_impl,
        "splash_tile": args.splash_tile if attn_impl in ("splash", "flash") else None,
        "load_seconds": round(dit_load_sec, 3),
        "run_seconds": run_times,
        "cold_seconds": run_times[0],
        "warm_median_seconds": round(warm_median, 4),
        "warm_ms_per_step": round(warm_median * 1000.0 / args.steps, 2),
        "step_tflops_est": round(step_tflops, 2),
        "achieved_tflops": round(achieved_tflops, 2),
        "mfu_pct_bf16": mfu_pct,
        "latents_has_nan": bool(np.isnan(final_latents_np).any()),
        "latents_mean": round(float(np.mean(final_latents_np)), 5),
        "latents_std": round(float(np.std(final_latents_np)), 5),
        "memory_stats_peak": dit_mem,
    }
    print(f"[stage2:dit] {json.dumps(stage2_metrics)}", flush=True)

    # --- Stage 3: VAE Decode ---
    t_vae0 = time.perf_counter()
    vae_cfg, vae_params = load_vae_decoder_weights(model_root, dtype=args.vae_dtype, device=dev0)
    jax.block_until_ready(vae_params)
    vae_load_sec = time.perf_counter() - t_vae0

    decode_fn = make_vae_decode_fn(vae_cfg, q_chunk_size=args.vae_attn_chunk)
    lat_j = jax.device_put(jnp.asarray(final_latents_np, dtype=jnp.float32), dev0)
    t_dec0 = time.perf_counter()
    pixels_j = decode_fn(vae_params, lat_j)
    pixels_j.block_until_ready()
    vae_decode_sec = time.perf_counter() - t_dec0
    pixels_np = np.asarray(pixels_j, dtype=np.float32)

    saved_paths = _save_images(pixels_np, args.output)
    vae_mem = _memory_stats_gib(dev0)
    del vae_params, pixels_j, lat_j
    gc.collect()

    stage3_metrics = {
        "vae_dtype": args.vae_dtype,
        "load_seconds": round(vae_load_sec, 3),
        "decode_seconds": round(vae_decode_sec, 3),
        "pixels_has_nan": bool(np.isnan(pixels_np).any()),
        "saved_images": saved_paths,
        "memory_stats_peak": vae_mem,
    }
    print(f"[stage3:vae] {json.dumps(stage3_metrics)}", flush=True)

    report = {
        "jax_version": jax.__version__,
        "device_kind": device_kind,
        "active_tp_devices": len(active_devs),
        "peak_bf16_tflops": peak_tflops,
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "cfg": args.cfg,
        "mu": args.mu,
        "seed": args.seed,
        "dtype": args.dtype,
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
    """Matrix benchmark sweeping attention backends, tile sizes, and resolutions on DiT."""
    import jax
    import jax.numpy as jnp

    from tpu_jax.dit import make_denoise_fn
    from tpu_jax.scheduler import compute_timesteps, estimate_step_tflops
    from tpu_jax.weights import load_dit_weights

    dev0, mesh, active_devs = _configure_jax(args)
    device_kind = getattr(dev0, "device_kind", str(dev0))
    peak_tflops = _tpu_peak_bf16_tflops(device_kind, len(active_devs))
    model_root = _resolve_model_root(args)

    # Load or synthesize embeddings (no need to burn Stage 1 time if benchmarking pure DiT speed)
    if args.prompt_embeds:
        data = np.load(args.prompt_embeds, allow_pickle=False)
        pos_ctx_np = data["pos_context"]
        pos_mask_np = data["pos_txtmask"].astype(bool)
    elif args.synthetic_embeds:
        rng = np.random.default_rng(0)
        pos_ctx_np = rng.standard_normal((1, 512, 12, 2560)).astype(np.float32) * 0.1
        pos_mask_np = np.zeros((1, 512), dtype=bool)
        pos_mask_np[:, :48] = True
        pos_mask_np[:, -5:] = True
    else:
        embeds, _ = _encode_prompts_stage1(
            model_root,
            [args.prompt],
            None,
            dtype=args.dtype,
            device=dev0,
            text_attn_impl="xla" if dev0.platform == "tpu" else "einsum",
        )
        pos_ctx_np = embeds["pos_context"]
        pos_mask_np = embeds["pos_txtmask"]

    t_load0 = time.perf_counter()
    dit_path = Path(args.dit_path) if args.dit_path else model_root
    dit_cfg, dit_params = load_dit_weights(
        dit_path,
        dtype=args.dtype,
        device=None if mesh is not None else dev0,
        mesh=mesh,
    )
    jax.block_until_ready(dit_params)
    load_sec = time.perf_counter() - t_load0

    jnp_dtype = jnp.bfloat16 if args.dtype in ("bf16", "bfloat16") else jnp.float32
    pos_ctx = jnp.asarray(pos_ctx_np, dtype=jnp_dtype)
    pos_mask = jnp.asarray(pos_mask_np, dtype=jnp.bool_)
    if mesh is None:
        pos_ctx, pos_mask = jax.device_put((pos_ctx, pos_mask), dev0)

    attn_impls = [x.strip() for x in args.attn_impls.split(",") if x.strip()]
    splash_tiles = [int(x.strip()) for x in args.splash_tiles.split(",") if x.strip()]
    resolutions = []
    for item in args.resolutions.split(","):
        item = item.strip().lower()
        if not item:
            continue
        w_s, h_s = item.split("x")
        resolutions.append((int(w_s), int(h_s)))

    cases = []
    for width, height in resolutions:
        ts_np = compute_timesteps(width=width, height=height, steps=args.steps, mu=args.mu)
        ts_j = jnp.asarray(ts_np, dtype=jnp.float32)
        if mesh is None:
            ts_j = jax.device_put(ts_j, dev0)
        init_lat_np = _make_init_latents(
            batch=1, width=width, height=height, channels=dit_cfg.channels, patch=dit_cfg.patch, seed=args.seed
        )

        for attn_impl in attn_impls:
            tiles = splash_tiles if attn_impl in ("splash", "flash") else [args.splash_tile]
            for tile in tiles:
                case_name = (
                    f"{width}x{height}_{attn_impl}"
                    + (f"_tile{tile}" if attn_impl in ("splash", "flash") else "")
                )
                denoise_fn = make_denoise_fn(
                    dit_cfg,
                    width=width,
                    height=height,
                    guidance_scale=args.cfg,
                    attn_impl=attn_impl,
                    splash_block_size=tile,
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
                        del out
                except Exception as exc:
                    err = f"{type(exc).__name__}: {exc}"

                if err is not None:
                    entry = {
                        "case": case_name,
                        "width": width,
                        "height": height,
                        "attn_impl": attn_impl,
                        "splash_tile": tile if attn_impl in ("splash", "flash") else None,
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
                        txt_len=int(pos_mask_np.shape[1]),
                        pad_multiple=max(256, tile) if attn_impl in ("splash", "flash") else 256,
                    )
                    ach_tf = (step_tf * args.steps) / warm_med
                    entry = {
                        "case": case_name,
                        "width": width,
                        "height": height,
                        "attn_impl": attn_impl,
                        "splash_tile": tile if attn_impl in ("splash", "flash") else None,
                        "steps": args.steps,
                        "run_seconds": run_times,
                        "cold_seconds": run_times[0],
                        "warm_median_seconds": round(warm_med, 4),
                        "warm_ms_per_step": round(warm_med * 1000.0 / args.steps, 2),
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
        "active_tp_devices": len(active_devs),
        "peak_bf16_tflops": peak_tflops,
        "load_seconds": round(load_sec, 3),
        "cases": cases,
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

    from tpu_jax.dit import dit_forward_precomputed, patchify_latents, prepare_text_conditioning
    from tpu_jax.weights import load_dit_weights

    dev0, mesh, _ = _configure_jax(args)
    model_root = _resolve_model_root(args)
    dit_path = Path(args.dit_path) if args.dit_path else model_root

    t0 = time.perf_counter()
    dit_cfg, dit_params = load_dit_weights(
        dit_path, dtype=args.dtype, device=None if mesh is not None else dev0, mesh=mesh
    )
    jax.block_until_ready(dit_params)
    load_sec = time.perf_counter() - t0

    rng = np.random.default_rng(args.seed)
    jnp_dtype = jnp.bfloat16 if args.dtype in ("bf16", "bfloat16") else jnp.float32
    pos_ctx = jnp.asarray(rng.standard_normal((1, 512, 12, 2560)).astype(np.float32) * 0.1, dtype=jnp_dtype)
    pos_mask_np = np.zeros((1, 512), dtype=bool)
    pos_mask_np[:, :48] = True
    pos_mask_np[:, -5:] = True
    pos_mask = jnp.asarray(pos_mask_np, dtype=jnp.bool_)
    lat = jnp.asarray(
        _make_init_latents(batch=1, width=args.width, height=args.height, seed=args.seed),
        dtype=jnp_dtype,
    )
    tok = patchify_latents(lat, patch=dit_cfg.patch)
    t_val = jnp.asarray([1.0], dtype=jnp.float32)
    if mesh is None:
        pos_ctx, pos_mask, tok, t_val = jax.device_put((pos_ctx, pos_mask, tok, t_val), dev0)

    impls = [x.strip() for x in args.attn_impls.split(",") if x.strip()]
    lat_h, lat_w = args.height // 8, args.width // 8
    outputs = {}
    timings_ms = {}
    for impl in impls:
        @jax.jit
        def _single(p, x, t, c, m, _impl=impl):
            txt, cos, sin, jmask, pad_len = prepare_text_conditioning(
                p, dit_cfg, c, m, latent_h=lat_h, latent_w=lat_w, attn_impl=_impl, splash_block_size=args.splash_tile
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
                splash_block_size=args.splash_tile,
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
        "load_seconds": round(load_sec, 3),
        "single_forward_warm_ms": timings_ms,
        "comparisons": comparisons,
        "memory_stats": _memory_stats_gib(dev0),
    }
    print(json.dumps(report, indent=2), flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Native JAX/XLA TPU inference runner for Krea 2")
    p.add_argument("--mode", choices=("probe", "generate", "bench", "parity"), default="probe")
    p.add_argument("--model", type=str, default=None, help="Local model snapshot directory")
    p.add_argument("--dit-path", type=str, default=None, help="Optional explicit turbo.safetensors / transformer/ path")
    p.add_argument("--hf-repo", type=str, default=None, help="HuggingFace repo id (e.g. krea/Krea-2-Turbo)")
    p.add_argument("--download-dir", type=str, default="/dev/shm/krea-2-turbo", help="Target dir for --hf-repo")
    p.add_argument("--hf-token", type=str, default=None)
    p.add_argument("--hf-token-file", type=str, default=None)
    p.add_argument("--prompt", type=str, default="A red fox sitting in fresh snow, golden hour, photorealistic")
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
    p.add_argument("--num-images", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", choices=("bf16", "bfloat16", "fp32", "float32"), default="bf16")
    p.add_argument("--vae-dtype", choices=("bf16", "bfloat16", "fp32", "float32"), default="bf16")
    p.add_argument("--vae-attn-chunk", type=int, default=1024)
    p.add_argument("--attn-impl", choices=("auto", "splash", "flash", "xla", "einsum"), default="auto")
    p.add_argument("--attn-impls", type=str, default="splash,flash,xla")
    p.add_argument("--splash-tile", type=int, default=512)
    p.add_argument("--splash-tiles", type=str, default="512,1024")
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
    if args.mode == "parity":
        run_parity(args)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
