# Krea 2 Native JAX / XLA TPU Inference Stack (`tpu_jax`)

Pure JAX/XLA end-to-end inference stack for **Krea 2 (Turbo & Raw)** on Google Cloud TPU (v6e-1, v5e, v5p) with **zero PyTorch runtime dependency**.

## Architecture & Performance Optimizations

| Component | Pure JAX Implementation | Key Optimizations |
|---|---|---|
| **Stage 1: Qwen3-VL-4B Text Encoder** (`text_encoder.py`) | 35 active decoder layers (`lax.scan`), Rust `tokenizers` | Skips ~1 GB vision tower and unused layer 35 + final norm; fused QKV & SwiGLU GEMMs; exact `cumsum(mask)-1` M-RoPE position IDs; freed from HBM (~7.5 GB) immediately after encoding |
| **Stage 2: 12B SingleStreamDiT** (`dit.py`) | 28 stacked GQA blocks (`lax.scan`), full N-step `lax.fori_loop` + `donate_argnums` | Hoists `txtfusion` (4 blocks), `txtmlp`, padding, and 3D RoPE ONCE outside the denoising loop; KV-head-grouped fused `qkvgate_w` (`[6144, 15360]`) and paired `mlp_gate_up_w` (`[6144, 32768]`); pre-folded RMSNorm `scale + 1.0`; native 48Q/12KV GQA **Pallas SplashAttention** (zero KV expansion, zero O(S²) mask, pre-padded once outside scan); optional 1D TPU Mesh TP (`--tp`) |
| **Stage 3: Qwen-Image VAE Decoder** (`vae.py`) | T=1 2D-reduced NHWC decoder | Slices causal 3D conv kernels at `t=-1` and skips unused `time_conv`; runs native NHWC 2D convs; chunked `lax.scan` mid-block attention (avoids 16 GB OOM at 2048×2048) |
| **RAM Cache Downloader** (`model_cache.py`) | RAM-backed `tmpfs` snapshot resolver | Auto-mounts dynamic `tmpfs` (up to 64 GiB on Colab) with exact missing-byte sizing via `HfApi`, `FileLock`, component markers, and automatic disk fallback (`KREA2_TPU_DISK_CACHE=1`); skips duplicate 26 GB diffusers shards (~35.7 GiB total vs ~62 GiB) |
| **Parallel Streaming Loader** (`weights.py`) | `safetensors` (`framework="numpy"`, native `ml_dtypes.bfloat16`) | Pre-allocates one contiguous buffer per stacked field (eliminating 2× `jnp.stack` RAM spike), fills 28/35 layers in parallel with `ThreadPoolExecutor`, pipelines async field-by-field H2D `jax.device_put` (peak host buffer <= 11.3 GiB), and deterministically frees HBM via `Array.delete()` |
| **Resident VAE + Multi-Prompt Stages** (`run_jax.py`) | Auto resident VAE (`>= 20 GiB` HBM) + `--prompts-file` | Keeps 125 MB BF16 VAE resident on TPU v6e-1 across stages; encodes all prompts once → drops Text Encoder → denoises all prompts once → decodes all latents |

## BF16 HBM Budget (Single TPU v6e-1 = 32 GiB HBM)

- **Stage 1 (Text Encoder)**: ~7.5 GB BF16 → freed before Stage 2
- **Stage 2 (DiT 12B)**: ~25.0 GB (23.3 GiB) BF16 → ~8.5 GiB headroom for activations & XLA buffers → freed before Stage 3
- **Stage 3 (VAE Decoder)**: ~0.13 GB BF16 (or ~0.25 GB FP32)

## Quick Start on TPU

The JAX runner defaults to `krea/Krea-2-Turbo` with 8 steps, CFG 0, and fixed `mu=1.15`. Use `--model` to select a local snapshot or `--hf-repo` to select another repository.

```bash
bash tpu_jax/setup_tpu.sh

# 1. Hardware & Pallas probe
python3 tpu_jax/run_jax.py --mode probe

# 2. End-to-end Turbo 1024x1024 (8 steps, BF16, SplashAttention)
export HF_TOKEN=hf_...
python3 tpu_jax/run_jax.py --mode generate \
  --download-dir /dev/shm/krea-2-turbo \
  --prompt "A red fox sitting in fresh snow, golden hour, photorealistic" \
  --width 1024 --height 1024 --steps 8 --cfg 0.0 --mu 1.15 \
  --attn-impl splash --splash-tile 512 --runs 2 \
  --compilation-cache-dir ~/.cache/krea2_jax_xla \
  --output krea2_1k --save-report report_1k.json

# 3. 2048x2048 Turbo
python3 tpu_jax/run_jax.py --mode generate \
  --model /dev/shm/krea-2-turbo \
  --prompt "A red fox sitting in fresh snow, golden hour, photorealistic" \
  --width 2048 --height 2048 --steps 8 --cfg 0.0 --mu 1.15 \
  --attn-impl splash --splash-tile 512 --runs 2 \
  --compilation-cache-dir ~/.cache/krea2_jax_xla \
  --output krea2_2k --save-report report_2k.json

# 4. Attention & tile matrix benchmark (splash vs flash vs xla @ 1K & 2K)
python3 tpu_jax/run_jax.py --mode bench \
  --model /dev/shm/krea-2-turbo \
  --synthetic-embeds \
  --attn-impls splash,flash,xla \
  --splash-tiles 512,1024 \
  --resolutions 1024x1024,2048x2048 \
  --steps 8 --cfg 0.0 --mu 1.15 --runs 3 \
  --save-report bench_report.json
```

## Grouped Pallas Attention on TPU v6e

Use `--attn-impl pallas_gqa` for SingleStreamDiT inference on v6e. The kernel
loads a K/V tile once for each four-head query group, folds `log2(e)` into
QNorm at load time, and stages adjacent KV subtiles to overlap matrix and
softmax work. It consumes sequence-minor V and skips empty tiles for static
prefix masks. BF16 Q scaling and the reordered computation can change rounding.

Run the synthetic batch-1 benchmark on v6e without downloading weights.

```bash
python3 tpu_jax/run_jax.py --mode microbench --attn-impl pallas_gqa \
  --width 848 --height 1200 --batch-sizes 1 \
  --splash-configs 1024x1024x512 --splash-mask-modes static

python3 tpu_jax/run_jax.py --mode microbench --attn-impl pallas_gqa \
  --width 1696 --height 2400 --batch-sizes 1 \
  --splash-configs 1024x2048x512 --splash-mask-modes static
```

The first tile size is the number of Q tokens per head. Each grouped Q tile
has four times as many rows.

## Local Unit & PyTorch-Parity Tests

```bash
python -m unittest tpu_jax/tests/test_jax_parity.py -v
python -m unittest tpu_jax.tests.test_pallas_gqa_attn -v
```
