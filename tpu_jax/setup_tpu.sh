#!/usr/bin/env bash
set -euo pipefail

# Native JAX/XLA TPU setup for Krea 2 (zero PyTorch runtime required).
# Designed for Google Colab TPU v6e-1 (32 GiB HBM) and Cloud TPU VMs.

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
"${PYTHON_BIN}" -m pip install -U safetensors tokenizers pillow numpy huggingface_hub hf-xet filelock

cat <<'EOF'
Krea 2 Native JAX TPU environment setup complete.

Recommended environment variables:
  export HF_XET_HIGH_PERFORMANCE=1
  export HF_XET_NUM_CONCURRENT_RANGE_GETS=16
  export JAX_COMPILATION_CACHE_DIR="${HOME}/.cache/krea2_jax_xla"

Quick hardware probe & single-block MFU breakdown (no model download needed):
  python3 tpu_jax/run_jax.py --mode probe
  python3 tpu_jax/run_jax.py --mode microbench --width 1024 --height 1024 --batch-sizes 1,2,4

End-to-end Turbo generation (1024x1024, 4-up MJ batch, 8 steps, BF16, SplashAttention):
  python3 tpu_jax/run_jax.py --mode generate \
    --hf-repo krea/Krea-2-Turbo \
    --download-dir /dev/shm/krea-2-turbo \
    --prompt "A red fox sitting in fresh snow, golden hour, photorealistic" \
    --width 1024 --height 1024 --num-images 4 --save-grid --steps 8 --cfg 0.0 --mu 1.15 \
    --attn-impl splash --runs 2 \
    --compilation-cache-dir "${HOME}/.cache/krea2_jax_xla" \
    --output krea2_turbo_1k --save-report report_1k.json
EOF
