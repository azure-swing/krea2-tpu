"""Grouped Pallas attention parity on CPU (interpret mode) or TPU."""

import unittest
from functools import partial
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from tpu_jax.dit import _attention_block_op, _gqa_attention
from tpu_jax.pallas_gqa_attn import pallas_gqa_attention


class TestPallasGqaAttention(unittest.TestCase):
    def test_masked_attention_against_einsum(self):
        seq_len = 397
        positions = jnp.arange(seq_len)[None, :]
        keys = jax.random.split(jax.random.key(13), 3)
        q = jax.random.normal(keys[0], (1, seq_len, 8, 128)) / np.sqrt(128)
        k = jax.random.normal(keys[1], (1, seq_len, 2, 128))
        v = jax.random.normal(keys[2], (1, seq_len, 2, 128)) * 0.2
        gaps = positions % 5 != 0
        cases = {
            "gaps": (gaps, None),
            "masked_first_tile": (gaps.at[:, :256].set(False), None),
            "all_masked": (jnp.zeros_like(gaps), None),
            "partial_prefix": (None, 385),
            "aligned_prefix": (None, 256),
            "unmasked": (None, None),
        }
        for dtype in (jnp.float32, jnp.bfloat16):
            q_in, k_in, v_in = (a.astype(dtype) for a in (q, k, v))
            q_log2 = (q_in.astype(jnp.float32) * np.log2(np.e)).astype(dtype)
            for name, (mask, prefix) in cases.items():
                with self.subTest(dtype=dtype, mask=name):
                    pallas = jax.jit(
                        partial(
                            pallas_gqa_attention,
                            block_q=128,
                            block_kv=256,
                            block_kv_compute=128,
                            static_valid_len=prefix,
                            interpret=jax.default_backend() != "tpu",
                        )
                    )
                    got = pallas(*(a.transpose(0, 2, 1, 3) for a in (q_log2, k_in, v_in)), mask)
                    ref_mask = positions < prefix if prefix is not None else mask
                    expected = _gqa_attention(q_in, k_in, v_in, ref_mask, attn_impl="einsum")
                    if ref_mask is not None:
                        expected = jnp.where(jnp.any(ref_mask, axis=1)[:, None, None], expected, 0)
                    tol = 1e-4 if dtype == jnp.float32 else 1e-3
                    np.testing.assert_allclose(
                        np.asarray(got, dtype=np.float32),
                        np.asarray(expected, dtype=np.float32),
                        rtol=tol,
                        atol=tol,
                    )

    def test_attention_block_uses_folded_qnorm_and_grouped_rope(self):
        seq_len, features = 259, 512
        keys = jax.random.split(jax.random.key(23), 4)
        x = jax.random.normal(keys[0], (1, seq_len, features))
        block = {
            "qkvgate_w": jax.random.normal(keys[1], (features, 2 * features + 256)) / np.sqrt(features),
            "qnorm": jnp.linspace(0.8, 1.2, 128) / np.sqrt(128),
            "knorm": jnp.linspace(1.2, 0.8, 128),
            "wo_w": jax.random.normal(keys[2], (features, features)) / np.sqrt(features),
        }
        block["qnorm_log2"] = block["qnorm"] * np.log2(np.e)
        angles = jax.random.normal(keys[3], (1, seq_len, 64))
        mask = jnp.arange(seq_len)[None, :] % 3 != 1
        attention = partial(
            _attention_block_op,
            num_heads=4,
            num_kv_heads=1,
            head_dim=128,
            norm_eps=1e-5,
            splash_block_q=128,
            splash_block_kv=256,
            splash_block_kv_compute=128,
        )
        args = x, block, jnp.cos(angles), jnp.sin(angles), mask
        expected = jax.jit(partial(attention, attn_impl="einsum"))(*args)
        with patch(
            "tpu_jax.pallas_gqa_attn.pallas_gqa_attention",
            partial(pallas_gqa_attention, interpret=jax.default_backend() != "tpu"),
        ):
            actual = jax.jit(partial(attention, attn_impl="pallas_gqa"))(*args)
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-4, atol=1e-4)

    def test_dynamic_mask_is_shared_or_per_batch_item(self):
        seq_len = 257
        keys = jax.random.split(jax.random.key(19), 3)
        q = jax.random.normal(keys[0], (2, seq_len, 8, 128)) / np.sqrt(128)
        k, v = (jax.random.normal(key, (2, seq_len, 2, 128)) for key in keys[1:])
        positions = jnp.arange(seq_len)[None, :]
        mask = (positions + jnp.arange(2)[:, None]) % 3 != 0
        q_log2 = q * np.float32(np.log2(np.e))
        qkv = tuple(x.transpose(0, 2, 1, 3) for x in (q_log2, k, v))
        pallas = jax.jit(
            partial(
                pallas_gqa_attention,
                block_q=128,
                block_kv=256,
                block_kv_compute=128,
                interpret=jax.default_backend() != "tpu",
            )
        )
        for valid in (mask, mask[:1]):
            with self.subTest(mask_rows=valid.shape[0]):
                got = pallas(*qkv, valid)
                expected = _gqa_attention(q, k, v, valid, attn_impl="einsum")
                np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-4, atol=1e-4)

    def test_rejects_invalid_tile_sizes(self):
        q = jax.ShapeDtypeStruct((1, 4, 383, 128), jnp.float32)
        kv = jax.ShapeDtypeStruct((1, 1, 383, 128), jnp.float32)
        fn = jax.jit(
            pallas_gqa_attention,
            static_argnames=("block_q", "block_kv", "block_kv_compute"),
        )
        for bq, bkv, bkvc, invalid in (
            (0, 256, 128, "block_q"),
            (127, 256, 128, "block_q"),
            (128, 0, 128, "block_kv"),
            (128, 256, 0, "block_kv"),
            (128, 256, 64, "block_kv"),
            (128, 384, 256, "block_kv"),
        ):
            with self.subTest(tile=(bq, bkv, bkvc)), self.assertRaisesRegex(ValueError, invalid):
                fn.trace(q, kv, kv, None, block_q=bq, block_kv=bkv, block_kv_compute=bkvc)

    def test_lowering_for_v6e_at_issue_resolutions(self):
        mesh = AbstractMesh((), (), abstract_device=AbstractDevice("TPU v6 lite", 1, "tpu"))
        for seq_len, block_kv_compute in ((4096, 512), (16128, 1024)):
            for batch, mask_rows, grouped_q in (
                (1, 0, True),
                (1, 1, True),
                (2, 1, False),
                (2, 2, False),
            ):
                with self.subTest(
                    seq_len=seq_len, batch=batch, mask_rows=mask_rows, grouped_q=grouped_q
                ), use_abstract_mesh(mesh):
                    q_shape = (batch, 12, seq_len, 4, 128) if grouped_q else (batch, 48, seq_len, 128)
                    q = jax.ShapeDtypeStruct(q_shape, jnp.bfloat16)
                    kv = jax.ShapeDtypeStruct((batch, 12, seq_len, 128), jnp.bfloat16)
                    mask = jax.ShapeDtypeStruct((mask_rows, seq_len), jnp.bool_) if mask_rows else None
                    fn = jax.jit(
                        partial(
                            pallas_gqa_attention,
                            block_q=256,
                            block_kv=2048,
                            block_kv_compute=block_kv_compute,
                            static_valid_len=None if mask_rows else seq_len - 17,
                            grouped_q=grouped_q,
                        )
                    )
                    lowered = fn.trace(q, kv, kv, mask).lower(lowering_platforms=("tpu",))
                    self.assertIn("stablehlo.custom_call", lowered.as_text())


if __name__ == "__main__":
    unittest.main()
