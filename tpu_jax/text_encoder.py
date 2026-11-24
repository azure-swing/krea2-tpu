"""Pure JAX/XLA implementation of Qwen3-VL-4B text tower for Krea 2 conditioning.

Key features:
- Zero PyTorch runtime dependency: tokenizes directly with Rust ``tokenizers``
  (from ``tokenizer/tokenizer.json``) or falls back to ``transformers.AutoTokenizer``.
- Skips the vision encoder (~1 GB) and unused layer 35 + final norm (since
  Krea 2 selects ``hidden_states`` indices ``(2, 5, ..., 35)``, i.e. outputs of
  decoder layers ``1, 4, ..., 34``).
- Stacks all 35 active decoder layers for ``jax.lax.scan`` with fused
  ``qkv_w`` and ``mlp_gate_up_w`` MXU GEMMs.
- Exact match to ``encoder.Qwen3VLConditioner`` + ``Qwen3VLModel``:
  - Right-pads prompt to ``max_length + prefix_idx - suffix_len`` then appends
    the 5 suffix tokens (creating an interior padding gap).
  - Computes ``position_ids = where(mask, cumsum(mask) - 1, 1)`` matching
    ``Qwen3VLModel.get_rope_index``.
  - Applies causal + key-padding attention mask and ``theta=5_000_000`` RoPE.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from tpu_jax.config import Qwen3TextConfig


def _rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    x_f32 = x.astype(jnp.float32)
    inv_rms = lax.rsqrt(jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True) + jnp.float32(eps))
    return (x_f32 * inv_rms).astype(x.dtype) * weight.astype(x.dtype)


def _matmul(x: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    return lax.dot_general(
        x,
        w,
        (((x.ndim - 1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(x.dtype)


def _rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _compute_qwen3_rope(
    position_ids: jnp.ndarray,
    inv_freq: jnp.ndarray,
    dtype: Any,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute ``(cos, sin)`` of shape ``[B, S, head_dim]`` matching ``Qwen3VLTextRotaryEmbedding``."""
    # For text-only inputs all 3 M-RoPE axes share the exact same position_ids, so
    # apply_interleaved_mrope reduces identically to 1D RoPE.
    freqs = position_ids.astype(jnp.float32)[..., None] * inv_freq[None, None, :]
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def _qwen3_layer_forward(
    hidden_states: jnp.ndarray,
    layer: dict[str, jnp.ndarray],
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    attn_mask_4d: jnp.ndarray,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rms_norm_eps: float,
    attn_impl: str,
) -> jnp.ndarray:
    batch, seq_len, _ = hidden_states.shape
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    residual = hidden_states
    h = _rms_norm(hidden_states, layer["input_layernorm"], rms_norm_eps)

    qkv = _matmul(h, layer["qkv_w"])
    q = qkv[..., :q_dim].reshape(batch, seq_len, num_heads, head_dim)
    k = qkv[..., q_dim : q_dim + kv_dim].reshape(batch, seq_len, num_kv_heads, head_dim)
    v = qkv[..., q_dim + kv_dim :].reshape(batch, seq_len, num_kv_heads, head_dim)

    q = _rms_norm(q, layer["q_norm"], rms_norm_eps)
    k = _rms_norm(k, layer["k_norm"], rms_norm_eps)

    # Apply RoPE in [B, S, H, D] layout
    c = cos[:, :, None, :]
    s = sin[:, :, None, :]
    q = q * c + _rotate_half(q) * s
    k = k * c + _rotate_half(k) * s

    if attn_impl == "xla":
        attn_out = jax.nn.dot_product_attention(
            q, k, v, mask=attn_mask_4d, implementation="xla"
        ).astype(v.dtype)
    elif attn_impl == "einsum":
        group_size = num_heads // num_kv_heads
        scale = jnp.float32(1.0 / math.sqrt(head_dim))
        q_g = q.astype(jnp.float32).reshape(batch, seq_len, num_kv_heads, group_size, head_dim)
        k_f = k.astype(jnp.float32)
        v_f = v.astype(jnp.float32)
        logits = jnp.einsum("bskgd,btkd->bkgst", q_g, k_f) * scale
        logits = jnp.where(attn_mask_4d[:, :, None, :, :], logits, jnp.float32(-1e30))
        weights = jax.nn.softmax(logits, axis=-1)
        attn_out = jnp.einsum("bkgst,btkd->bskgd", weights, v_f).astype(v.dtype)
    else:
        raise ValueError(f"Unsupported text encoder attn_impl={attn_impl!r}; use 'xla' or 'einsum'.")

    attn_out = attn_out.reshape(batch, seq_len, q_dim)
    hidden_states = residual + _matmul(attn_out, layer["o_w"])

    residual = hidden_states
    h = _rms_norm(hidden_states, layer["post_attention_layernorm"], rms_norm_eps)
    gu = _matmul(h, layer["mlp_gate_up_w"])
    gate, up = jnp.split(gu, 2, axis=-1)
    mlp_out = _matmul(jax.nn.silu(gate) * up, layer["mlp_down_w"])
    return residual + mlp_out


def qwen3_encode_tokens(
    params: dict[str, Any],
    config: Qwen3TextConfig,
    input_ids: jnp.ndarray,
    attention_mask: jnp.ndarray,
    *,
    attn_impl: str = "xla",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run Qwen3-VL text tower and extract Krea 2 multi-layer hidden states.

    Args:
        input_ids: ``[B, S_total]`` int32 token IDs (prefix + padded prompt + suffix).
        attention_mask: ``[B, S_total]`` bool mask.

    Returns:
        ``(hiddens, mask)`` sliced from ``prefix_start_idx`` onward:
        - ``hiddens``: ``[B, S_total - prefix_start_idx, len(select_layers), hidden_size]``
        - ``mask``: ``[B, S_total - prefix_start_idx]`` bool
    """
    mask_bool = attention_mask.astype(jnp.bool_)
    _, seq_len = input_ids.shape

    # Exact Qwen3VLModel.get_rope_index position_ids for text-only inputs.
    pos_ids = jnp.cumsum(mask_bool.astype(jnp.int32), axis=-1) - 1
    pos_ids = jnp.where(mask_bool, pos_ids, jnp.int32(1))

    h0 = params["embed_tokens"][input_ids]
    cos, sin = _compute_qwen3_rope(pos_ids, params["inv_freq"], h0.dtype)

    # Causal mask [1, 1, S, S] combined with key-padding mask [B, 1, 1, S].
    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))[None, None, :, :]
    attn_mask_4d = causal & mask_bool[:, None, None, :]

    def _scan_body(carry: jnp.ndarray, layer: dict[str, jnp.ndarray]):
        nxt = _qwen3_layer_forward(
            carry,
            layer,
            cos,
            sin,
            attn_mask_4d,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            attn_impl=attn_impl,
        )
        return nxt, nxt

    final_h, layer_outs = lax.scan(_scan_body, h0, params["layers"])

    # HuggingFace hidden_states indexing:
    #   index 0 -> embed_tokens (h0)
    #   index k (1 <= k < num_hidden_layers) -> output of layers[k - 1]
    #   index num_hidden_layers -> final_norm(layers[-1])
    selected = []
    for idx in config.select_layers:
        if idx == 0:
            selected.append(h0)
        elif idx < config.num_hidden_layers:
            selected.append(layer_outs[idx - 1])
        else:
            if "final_norm" not in params:
                raise KeyError(
                    f"select_layers includes final normalized layer {idx}, but final_norm was not loaded."
                )
            selected.append(_rms_norm(final_h, params["final_norm"], config.rms_norm_eps))

    # Stack to [B, S_total, N_layers, D] and drop prompt template prefix.
    hiddens = jnp.stack(selected, axis=2)
    start = config.prefix_start_idx
    return hiddens[:, start:, :, :], mask_bool[:, start:]


def make_text_encode_fn(config: Qwen3TextConfig, *, attn_impl: str = "xla"):
    """Return a ``jax.jit``-compiled Qwen3-VL text encoder function."""

    @jax.jit
    def _encode(
        params: dict[str, Any],
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return qwen3_encode_tokens(params, config, input_ids, attention_mask, attn_impl=attn_impl)

    return _encode


class Krea2Tokenizer:
    """Zero-PyTorch prompt tokenizer matching ``encoder.Qwen3VLConditioner`` exactly."""

    def __init__(self, tokenizer_dir: str | Path, config: Qwen3TextConfig | None = None):
        self.config = config or Qwen3TextConfig()
        tdir = Path(tokenizer_dir)
        if (tdir / "tokenizer").is_dir():
            tdir = tdir / "tokenizer"

        tok_json = tdir / "tokenizer.json"
        tok_cfg_path = tdir / "tokenizer_config.json"
        self._backend = None
        self._pad_id = 151643  # Qwen2/Qwen3 <|endoftext|> default

        if tok_cfg_path.is_file():
            tcfg = json.loads(tok_cfg_path.read_text(encoding="utf-8"))
            pad_tok = tcfg.get("pad_token", "<|endoftext|>")
        else:
            pad_tok = "<|endoftext|>"

        try:
            from tokenizers import Tokenizer

            if tok_json.is_file():
                tok = Tokenizer.from_file(str(tok_json))
                pid = tok.token_to_id(pad_tok)
                if pid is not None:
                    self._pad_id = int(pid)
                self._backend = ("tokenizers", tok)
        except ImportError:
            pass

        if self._backend is None:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(str(tdir))
            if tok.pad_token_id is not None:
                self._pad_id = int(tok.pad_token_id)
            self._backend = ("transformers", tok)

    def _encode_raw(self, text: str) -> list[int]:
        kind, tok = self._backend
        if kind == "tokenizers":
            return list(tok.encode(text, add_special_tokens=True).ids)
        return list(tok.encode(text, add_special_tokens=True))

    def tokenize(self, prompts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(input_ids [B, S_total] int32, attention_mask [B, S_total] bool)``."""
        cfg = self.config
        suffix_ids = self._encode_raw(cfg.prompt_suffix)
        target_body_len = cfg.max_length + cfg.prefix_start_idx - cfg.suffix_token_count
        total_len = target_body_len + len(suffix_ids)

        ids_batch = np.full((len(prompts), total_len), self._pad_id, dtype=np.int32)
        mask_batch = np.zeros((len(prompts), total_len), dtype=np.bool_)

        for i, prompt in enumerate(prompts):
            body = self._encode_raw(cfg.prompt_prefix + prompt)[:target_body_len]
            n_body = len(body)
            ids_batch[i, :n_body] = body
            mask_batch[i, :n_body] = True
            ids_batch[i, target_body_len:] = suffix_ids
            mask_batch[i, target_body_len:] = True

        return ids_batch, mask_batch
