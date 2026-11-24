#!/usr/bin/env bash
set -euo pipefail

# Native JAX/XLA TPU setup for Krea 2 (zero PyTorch runtime required).
# Designed for Google Colab TPU v6e-1 (32 GiB HBM) and Cloud TPU VMs.

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
"${PYTHON_BIN}" -m pip install -U safetensors tokenizers pillow numpy huggingface_hub hf_transfer

cat <<'EOF'
Krea 2 Native JAX TPU environment setup complete.

Recommended environment variables:
  export HF_HUB_ENABLE_HF_TRANSFER=1
  export JAX_COMPILATION_CACHE_DIR="${HOME}/.cache/krea2_jax_xla"

Quick hardware probe:
  python3 tpu_jax/run_jax.py --mode probe

End-to-end Turbo generation (1024x1024, 8 steps, BF16, Pallas SplashAttention):
  python3 tpu_jax/run_jax.py --mode generate \
    --hf-repo krea/Krea-2-Turbo \
    --download-dir /dev/shm/krea-2-turbo \
    --prompt "A red fox sitting in fresh snow, golden hour, photorealistic" \
    --width 1024 --height 1024 --steps 8 --cfg 0.0 --mu 1.15 \
    --attn-impl splash --splash-tile 512 --runs 2 \
    --compilation-cache-dir "${HOME}/.cache/krea2_jax_xla" \
    --output krea2_turbo_1k --save-report report_1k.json
EOF
