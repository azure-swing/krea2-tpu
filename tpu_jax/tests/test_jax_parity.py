"""Numerical parity unit tests: native JAX stack vs upstream PyTorch reference.

Verifies every component of ``tpu_jax`` against upstream PyTorch modules:
1. ``scheduler.compute_timesteps`` vs ``sampling.timesteps`` (Turbo & Raw)
2. ``dit`` (native + diffusers weight formats, GQA, 3D RoPE, TextFusion,
   hoisted conditioning, full ``lax.fori_loop`` Euler denoising for Turbo & CFG)
   vs ``mmdit.SingleStreamDiT`` + ``sampling``
3. ``text_encoder.qwen3_encode_tokens`` (with interior padding gap & M-RoPE
   ``cumsum(mask) - 1`` position IDs) vs ``transformers.Qwen3VLModel``
4. ``vae.vae_decode_nhwc`` (T=1 2D-reduced CausalConv3d + RMSNorm + chunked
   mid-block attention) vs ``diffusers.AutoencoderKLQwenImage``
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

# Disable torch.compile before importing torch/mmdit so upstream @torch.compile(fullgraph=True)
# runs eagerly on Windows CPU across PyTorch 2.6-2.11+ without MSVC/Triton.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import numpy as np
import torch

# Make upstream @torch.compile(fullgraph=True) in mmdit.py a no-op on CPU test runners
# (PyTorch 2.11+ raises if fullgraph=True runs with dynamo disabled).
torch._dynamo.config.disable = True
torch.compile = lambda fn=None, *_a, **_kw: fn if fn is not None else (lambda f: f)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tpu_jax.config import Krea2DiTConfig, Qwen3TextConfig, QwenImageVAEConfig  # noqa: E402
from tpu_jax.scheduler import compute_timesteps, estimate_step_tflops  # noqa: E402

HAS_JAX = importlib.util.find_spec("jax") is not None


class TestSchedulerParity(unittest.TestCase):
    def test_timesteps_match_sampling_reference(self):
        import sampling

        cases = [
            # Turbo fixed mu=1.15 at 1K and 2K
            dict(width=1024, height=1024, steps=8, mu=1.15),
            dict(width=2048, height=2048, steps=8, mu=1.15),
            # Raw dynamic mu across resolutions (including non-multiple-of-16)
            dict(width=1024, height=1024, steps=28, mu=None),
            dict(width=1280, height=768, steps=52, mu=None),
            dict(width=1000, height=750, steps=12, mu=None),
        ]
        for kw in cases:
            w = sampling.roundup(kw["width"], 16, "width")
            h = sampling.roundup(kw["height"], 16, "height")
            seq_len = (h // 16) * (w // 16)
            x1 = (256 // 16) ** 2
            x2 = (1280 // 16) ** 2
            ref = np.asarray(
                sampling.timesteps(seq_len, kw["steps"], x1, x2, y1=0.5, y2=1.15, mu=kw["mu"]),
                dtype=np.float32,
            )
            got = compute_timesteps(
                width=kw["width"],
                height=kw["height"],
                steps=kw["steps"],
                mu=kw["mu"],
            )
            np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)

    def test_tflops_estimator_positive(self):
        cfg = Krea2DiTConfig()
        tf_1k = estimate_step_tflops(cfg, width=1024, height=1024)
        tf_2k = estimate_step_tflops(cfg, width=2048, height=2048)
        self.assertGreater(tf_1k, 50.0)
        self.assertGreater(tf_2k, tf_1k * 3.5)


def _native_to_diffusers_state_dict(sd: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Remap native ``mmdit.SingleStreamDiT`` keys to HuggingFace Diffusers keys."""
    out: dict[str, np.ndarray] = {}
    simple_map = {
        "first.weight": "img_in.weight",
        "first.bias": "img_in.bias",
        "tmlp.0.weight": "time_embed.linear_1.weight",
        "tmlp.0.bias": "time_embed.linear_1.bias",
        "tmlp.2.weight": "time_embed.linear_2.weight",
        "tmlp.2.bias": "time_embed.linear_2.bias",
        "tproj.1.weight": "time_mod_proj.weight",
        "tproj.1.bias": "time_mod_proj.bias",
        "txtfusion.projector.weight": "text_fusion.projector.weight",
        "txtmlp.0.scale": "txt_in.norm.weight",
        "txtmlp.1.weight": "txt_in.linear_1.weight",
        "txtmlp.1.bias": "txt_in.linear_1.bias",
        "txtmlp.3.weight": "txt_in.linear_2.weight",
        "txtmlp.3.bias": "txt_in.linear_2.bias",
        "last.norm.scale": "final_layer.norm.weight",
        "last.modulation.lin": "final_layer.scale_shift_table",
        "last.linear.weight": "final_layer.linear.weight",
        "last.linear.bias": "final_layer.linear.bias",
    }

    def _map_block_suffix(rest: str) -> str:
        table = {
            "prenorm.scale": "norm1.weight",
            "postnorm.scale": "norm2.weight",
            "mod.lin": "scale_shift_table",
            "attn.wq.weight": "attn.to_q.weight",
            "attn.wk.weight": "attn.to_k.weight",
            "attn.wv.weight": "attn.to_v.weight",
            "attn.gate.weight": "attn.to_gate.weight",
            "attn.wo.weight": "attn.to_out.0.weight",
            "attn.qknorm.qnorm.scale": "attn.norm_q.weight",
            "attn.qknorm.knorm.scale": "attn.norm_k.weight",
            "mlp.gate.weight": "ff.gate.weight",
            "mlp.up.weight": "ff.up.weight",
            "mlp.down.weight": "ff.down.weight",
        }
        return table[rest]

    for k, v in sd.items():
        if k in simple_map:
            out[simple_map[k]] = v
        elif k.startswith("blocks."):
            parts = k.split(".", 2)
            out[f"transformer_blocks.{parts[1]}.{_map_block_suffix(parts[2])}"] = v
        elif k.startswith("txtfusion.layerwise_blocks.") or k.startswith("txtfusion.refiner_blocks."):
            parts = k.split(".", 3)
            out[f"text_fusion.{parts[1]}.{parts[2]}.{_map_block_suffix(parts[3])}"] = v
        else:
            raise KeyError(f"Unmapped key: {k}")
    return out


@unittest.skipUnless(HAS_JAX, "JAX is not installed in this environment")
class TestJaxDiTParity(unittest.TestCase):
    def _make_tiny_models(self):
        import jax.numpy as jnp
        import mmdit
        from tpu_jax.weights import build_dit_params_from_state_dict

        torch.manual_seed(42)
        pt_cfg = mmdit.SingleMMDiTConfig(
            features=128,
            tdim=32,
            txtdim=64,
            heads=8,
            kvheads=2,
            multiplier=2,
            layers=3,
            patch=2,
            channels=4,
            bias=False,
            theta=1000.0,
            txtlayers=4,
            txtheads=4,
            txtkvheads=4,
        )
        jax_cfg = Krea2DiTConfig(
            features=128,
            tdim=32,
            txtdim=64,
            heads=8,
            kvheads=2,
            multiplier=2,
            layers=3,
            patch=2,
            channels=4,
            bias=False,
            theta=1000.0,
            txtlayers=4,
            txtheads=4,
            txtkvheads=4,
        )
        pt_model = mmdit.SingleStreamDiT(pt_cfg).eval()
        # Non-zero init for scale/modulation params so RMSNorm(+1) and mod.lin are exercised!
        with torch.no_grad():
            for name, p in pt_model.named_parameters():
                if name.endswith(".scale") or name.endswith(".lin"):
                    p.normal_(mean=0.0, std=0.2)

        raw_sd = {k: v.detach().cpu().float().numpy() for k, v in pt_model.state_dict().items()}
        jax_params_native = build_dit_params_from_state_dict(raw_sd, jax_cfg, dtype=jnp.float32)
        diffusers_sd = _native_to_diffusers_state_dict(raw_sd)
        jax_params_diffusers = build_dit_params_from_state_dict(diffusers_sd, jax_cfg, dtype=jnp.float32)
        return pt_model, jax_cfg, jax_params_native, jax_params_diffusers

    def test_single_forward_matches_pytorch_native_and_diffusers(self):
        import jax.numpy as jnp
        import mmdit
        import sampling
        from tpu_jax.dit import (
            dit_forward_precomputed,
            patchify_latents,
            prepare_text_conditioning,
        )

        pt_model, jax_cfg, jax_native, jax_diffusers = self._make_tiny_models()

        torch.manual_seed(7)
        b, c, h, w = 2, 4, 8, 6  # patch=2 -> h'=4, w'=3 -> img_len=12
        txt_len = 10
        lat_pt = torch.randn(b, c, h, w, dtype=torch.float32)
        ctx_pt = torch.randn(b, txt_len, jax_cfg.txtlayers, jax_cfg.txtdim, dtype=torch.float32)
        # Interior padding gap + valid suffix (matches encoder.py pattern!)
        txtmask_pt = torch.tensor(
            [
                [1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
            ],
            dtype=torch.bool,
        )
        t_pt = torch.tensor([0.75, 0.25], dtype=torch.float32)

        img_tok_pt, pos_pt, mask_pt = sampling.prepare(lat_pt, txt_len, jax_cfg.patch, txtmask_pt)
        with torch.no_grad(), patch.object(mmdit, "sdpa_kernel", lambda *_a, **_kw: nullcontext()):
            ref_out = pt_model(img_tok_pt, ctx_pt, t_pt, pos_pt, mask_pt).numpy()

        lat_j = jnp.asarray(lat_pt.numpy())
        ctx_j = jnp.asarray(ctx_pt.numpy())
        mask_j = jnp.asarray(txtmask_pt.numpy())
        t_j = jnp.asarray(t_pt.numpy())
        tok_j = patchify_latents(lat_j, patch=jax_cfg.patch)

        for label, params in (("native", jax_native), ("diffusers", jax_diffusers)):
            for impl in ("einsum", "xla"):
                txt, cos, sin, jmask, pad_len = prepare_text_conditioning(
                    params, jax_cfg, ctx_j, mask_j, latent_h=h, latent_w=w, attn_impl=impl
                )
                got = np.asarray(
                    dit_forward_precomputed(
                        params,
                        jax_cfg,
                        tok_j,
                        t_j,
                        txt,
                        cos,
                        sin,
                        jmask,
                        pad_len=pad_len,
                        attn_impl=impl,
                    )
                )
                np.testing.assert_allclose(
                    got,
                    ref_out,
                    rtol=1e-4,
                    atol=2e-4,
                    err_msg=f"Mismatch for format={label}, attn_impl={impl}",
                )

    def test_full_euler_denoise_loop_turbo_and_cfg(self):
        import jax.numpy as jnp
        import mmdit
        import sampling
        from tpu_jax.dit import make_denoise_fn

        pt_model, jax_cfg, jax_native, _ = self._make_tiny_models()

        torch.manual_seed(11)
        b, c, lat_h, lat_w = 1, 4, 8, 8
        width, height = lat_w * 8, lat_h * 8
        txt_len = 8
        lat_pt = torch.randn(b, c, lat_h, lat_w, dtype=torch.float32)
        pos_ctx = torch.randn(b, txt_len, jax_cfg.txtlayers, jax_cfg.txtdim, dtype=torch.float32)
        pos_mask = torch.tensor([[1, 1, 1, 0, 0, 1, 1, 1]], dtype=torch.bool)
        neg_ctx = torch.randn(b, txt_len, jax_cfg.txtlayers, jax_cfg.txtdim, dtype=torch.float32)
        neg_mask = torch.tensor([[1, 0, 0, 0, 0, 1, 1, 1]], dtype=torch.bool)

        ts_np = compute_timesteps(width=width, height=height, steps=4, mu=1.15)

        for guidance in (0.0, 3.5):
            # Reference PyTorch Euler loop matching sampling.sample
            x_pt, pos_pt, mask_pt = sampling.prepare(lat_pt, txt_len, jax_cfg.patch, pos_mask)
            _, unpos_pt, unmask_pt = sampling.prepare(lat_pt, txt_len, jax_cfg.patch, neg_mask)
            cur = x_pt.clone()
            with torch.no_grad(), patch.object(mmdit, "sdpa_kernel", lambda *_a, **_kw: nullcontext()):
                for tcurr, tprev in zip(ts_np[:-1], ts_np[1:]):
                    t_tensor = torch.full((b,), float(tcurr), dtype=torch.float32)
                    cond = pt_model(cur, pos_ctx, t_tensor, pos_pt, mask_pt)
                    if guidance > 0.0:
                        uncond = pt_model(cur, neg_ctx, t_tensor, unpos_pt, unmask_pt)
                        v = cond + guidance * (cond - uncond)
                    else:
                        v = cond
                    cur = cur + float(tprev - tcurr) * v
            ref_lat = (
                cur.reshape(b, lat_h // 2, lat_w // 2, c, 2, 2)
                .permute(0, 3, 1, 4, 2, 5)
                .reshape(b, c, lat_h, lat_w)
                .numpy()
            )

            denoise_fn = make_denoise_fn(
                jax_cfg,
                width=width,
                height=height,
                guidance_scale=guidance,
                attn_impl="xla",
                latent_dtype=jnp.float32,
                donate_latents=False,
            )
            got_lat = np.asarray(
                denoise_fn(
                    jax_native,
                    jnp.asarray(lat_pt.numpy()),
                    jnp.asarray(pos_ctx.numpy()),
                    jnp.asarray(pos_mask.numpy()),
                    jnp.asarray(neg_ctx.numpy()) if guidance > 0.0 else None,
                    jnp.asarray(neg_mask.numpy()) if guidance > 0.0 else None,
                    jnp.asarray(ts_np),
                )
            )
            np.testing.assert_allclose(
                got_lat,
                ref_lat,
                rtol=1e-4,
                atol=3e-4,
                err_msg=f"Euler loop mismatch for guidance={guidance}",
            )

    def test_text_compaction_and_mj_batch4_broadcast_and_unroll_match_pytorch(self):
        import jax.numpy as jnp
        import mmdit
        import sampling
        from tpu_jax.dit import compact_text_embeddings, make_denoise_fn

        pt_model, jax_cfg, jax_native, _ = self._make_tiny_models()

        torch.manual_seed(29)
        b, c, lat_h, lat_w = 4, 4, 8, 8
        width, height = lat_w * 8, lat_h * 8
        txt_len = 12
        lat_pt = torch.randn(b, c, lat_h, lat_w, dtype=torch.float32)
        # Single prompt with interior padding gap + valid suffix (exact Krea 2 pattern)
        pos_ctx_1 = torch.randn(1, txt_len, jax_cfg.txtlayers, jax_cfg.txtdim, dtype=torch.float32)
        pos_mask_1 = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]], dtype=torch.bool)
        pos_ctx_b = pos_ctx_1.expand(b, -1, -1, -1).contiguous()
        pos_mask_b = pos_mask_1.expand(b, -1).contiguous()

        ts_np = compute_timesteps(width=width, height=height, steps=3, mu=1.15)
        x_pt, pos_pt, mask_pt = sampling.prepare(lat_pt, txt_len, jax_cfg.patch, pos_mask_b)
        cur = x_pt.clone()
        with torch.no_grad(), patch.object(mmdit, "sdpa_kernel", lambda *_a, **_kw: nullcontext()):
            for tcurr, tprev in zip(ts_np[:-1], ts_np[1:]):
                t_tensor = torch.full((b,), float(tcurr), dtype=torch.float32)
                v = pt_model(cur, pos_ctx_b, t_tensor, pos_pt, mask_pt)
                cur = cur + float(tprev - tcurr) * v
        ref_lat = (
            cur.reshape(b, lat_h // 2, lat_w // 2, c, 2, 2)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(b, c, lat_h, lat_w)
            .numpy()
        )

        # Compact 12 -> 8 bucket (6 valid tokens) & broadcast batch=1 prompt across batch=4 seeds!
        ctx_c, mask_c, common_valid = compact_text_embeddings(
            pos_ctx_1.numpy(), pos_mask_1.numpy(), buckets=(8, 12)
        )
        self.assertEqual(ctx_c.shape[1], 8)
        self.assertEqual(common_valid, 6)

        for unroll in (1, 2):
            denoise_fn = make_denoise_fn(
                jax_cfg,
                width=width,
                height=height,
                guidance_scale=0.0,
                attn_impl="xla",
                scan_unroll=unroll,
                latent_dtype=jnp.float32,
                donate_latents=False,
            )
            got_lat = np.asarray(
                denoise_fn(
                    jax_native,
                    jnp.asarray(lat_pt.numpy()),
                    jnp.asarray(ctx_c),
                    jnp.asarray(mask_c),
                    None,
                    None,
                    jnp.asarray(ts_np),
                )
            )
            self.assertEqual(got_lat.shape, (b, c, lat_h, lat_w))
            np.testing.assert_allclose(
                got_lat,
                ref_lat,
                rtol=1e-4,
                atol=3e-4,
                err_msg=f"Compacted MJ batch=4 mismatch (unroll={unroll})",
            )

    def test_tp2_packed_layout_matches_pytorch(self):
        import jax.numpy as jnp
        import mmdit
        import sampling
        from tpu_jax.dit import dit_forward_precomputed, patchify_latents, prepare_text_conditioning
        from tpu_jax.weights import _build_dit_params, _make_put_fn, _resolve_np_dtype

        pt_model, jax_cfg, _, _ = self._make_tiny_models()
        raw_sd = {k: v.detach().cpu().float().numpy() for k, v in pt_model.state_dict().items()}
        tp2_params = _build_dit_params(
            raw_sd.__getitem__,
            set(raw_sd.keys()),
            jax_cfg,
            np_dtype=_resolve_np_dtype("float32"),
            put_fn=_make_put_fn(device=None, mesh=None),
            num_workers=1,
            tp_size=2,
        )

        torch.manual_seed(17)
        b, c, h, w = 2, 4, 8, 6
        txt_len = 8
        lat_pt = torch.randn(b, c, h, w, dtype=torch.float32)
        ctx_pt = torch.randn(b, txt_len, jax_cfg.txtlayers, jax_cfg.txtdim, dtype=torch.float32)
        txtmask_pt = torch.tensor([[1, 1, 1, 0, 0, 1, 1, 1], [1, 1, 0, 0, 0, 1, 1, 1]], dtype=torch.bool)
        t_pt = torch.tensor([0.5, 0.8], dtype=torch.float32)

        img_tok_pt, pos_pt, mask_pt = sampling.prepare(lat_pt, txt_len, jax_cfg.patch, txtmask_pt)
        with torch.no_grad(), patch.object(mmdit, "sdpa_kernel", lambda *_a, **_kw: nullcontext()):
            ref_out = pt_model(img_tok_pt, ctx_pt, t_pt, pos_pt, mask_pt).numpy()

        tok_j = patchify_latents(jnp.asarray(lat_pt.numpy()), patch=jax_cfg.patch)
        txt, cos, sin, jmask, pad_len = prepare_text_conditioning(
            tp2_params, jax_cfg, jnp.asarray(ctx_pt.numpy()), jnp.asarray(txtmask_pt.numpy()),
            latent_h=h, latent_w=w, attn_impl="xla",
        )
        got = np.asarray(
            dit_forward_precomputed(
                tp2_params, jax_cfg, tok_j, jnp.asarray(t_pt.numpy()),
                txt, cos, sin, jmask, pad_len=pad_len, attn_impl="xla", tp_size=2,
            )
        )
        np.testing.assert_allclose(got, ref_out, rtol=1e-4, atol=2e-4)


class TestRunnerAndSplashHelpers(unittest.TestCase):
    def test_v6_lite_peak_tflops_and_splash_block_resolution(self):
        from types import SimpleNamespace
        from tpu_jax.run_jax import _parse_splash_configs, _resolve_splash_config, _tpu_peak_bf16_tflops

        self.assertEqual(_tpu_peak_bf16_tflops("TPU v6 lite", 1), 918.0)
        self.assertEqual(_tpu_peak_bf16_tflops("TPU v6e", 4), 3672.0)
        self.assertEqual(_tpu_peak_bf16_tflops("TPU v5 lite", 1), 197.0)

        args_auto = SimpleNamespace(
            splash_tile=None, splash_block_q=None, splash_block_kv=None, splash_block_kv_compute=None
        )
        self.assertEqual(_resolve_splash_config(args_auto, width=1024, height=1024), (1024, 1024, 512))
        self.assertEqual(_resolve_splash_config(args_auto, width=2048, height=2048), (1024, 2048, 512))
        self.assertEqual(
            _parse_splash_configs("512x1024x512,1024,1024x2048", "512"),
            [(512, 1024, 512), (1024, 1024, 1024), (1024, 2048, 1024)],
        )

    def test_save_grid_creates_contact_sheet(self):
        import tempfile
        from PIL import Image
        from tpu_jax.run_jax import _save_images

        with tempfile.TemporaryDirectory() as tmp:
            pix = np.zeros((4, 32, 48, 3), dtype=np.float32)
            saved = _save_images(pix, Path(tmp) / "mj_out", save_grid=True)
            self.assertEqual(len(saved), 5)
            with Image.open(saved[-1]) as grid_img:
                self.assertEqual(grid_img.size, (96, 64))

    @unittest.skipUnless(HAS_JAX, "JAX is not installed in this environment")
    def test_splash_helpers_and_static_mask_builder(self):
        import time
        from jax.experimental.pallas.ops.tpu.splash_attention import (
            splash_attention_kernel,
            splash_attention_mask,
        )
        from tpu_jax.dit import _build_static_splash_mask, _resolve_splash_block_sizes

        self.assertEqual(_resolve_splash_block_sizes(4352, 512, 1024, 512), (512, 1024, 512, 4608, 5120))
        self.assertEqual(_resolve_splash_block_sizes(16640, 1024, 2048, 512), (1024, 2048, 512, 17408, 18432))

        mask = _build_static_splash_mask(512, 1024, 420, splash_attention_mask)
        self.assertEqual(mask.shape, (512, 1024))
        self.assertTrue(bool(mask[0:512, 0:420].all()))
        self.assertFalse(bool(mask[0:512, 420:1024].any()))
        partial_blk = mask[0:128, 384:512]
        self.assertEqual(partial_blk.shape, (128, 128))
        self.assertTrue(bool(partial_blk[:, :36].all()))
        self.assertFalse(bool(partial_blk[:, 36:].any()))
        self.assertEqual(
            hash(mask),
            hash(_build_static_splash_mask(512, 1024, 420, splash_attention_mask)),
        )
        self.assertEqual(
            _build_static_splash_mask(512, 512, 512, splash_attention_mask),
            splash_attention_mask.FullMask((512, 512)),
        )

        # Regression test (Issue #1 P2): 2048x2048 batch-4 (192 heads, 17408x17408) lazy prefix
        # mask + SplashAttention kernel construction must be fast (was ~83s with dense NumpyMask).
        t0 = time.perf_counter()
        big_mask = _build_static_splash_mask(17408, 17408, 16437, splash_attention_mask)
        mh_mask = splash_attention_mask.MultiHeadMask([big_mask for _ in range(192)])
        block_sizes = splash_attention_kernel.BlockSizes(
            block_q=1024, block_kv=1024, block_kv_compute=512
        )
        _kernel = splash_attention_kernel.make_splash_mha(
            mask=mh_mask, head_shards=1, q_seq_shards=1, block_sizes=block_sizes
        )
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 5.0, f"Lazy static Splash mask construction too slow: {elapsed:.2f}s")

    def test_cached_prompt_embeds_preserves_per_image_rows_and_validates(self):
        from tpu_jax.run_jax import _load_cached_prompt_embeds

        pos_ctx = np.arange(2 * 8 * 4 * 16, dtype=np.float32).reshape(2, 8, 4, 16)
        pos_mask = np.array([[1, 1, 1, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 0, 1, 1]], dtype=np.bool_)
        neg_ctx = np.zeros((1, 8, 4, 16), dtype=np.float32)
        neg_mask = np.array([[1, 0, 0, 0, 0, 1, 1, 1]], dtype=np.bool_)
        payload = {
            "pos_context": pos_ctx,
            "pos_txtmask": pos_mask,
            "neg_context": neg_ctx,
            "neg_txtmask": neg_mask,
        }

        # Both distinct rows preserved for num_images=2 (and singleton neg allowed for CFG)
        loaded = _load_cached_prompt_embeds(payload, num_images=2, do_cfg=True)
        np.testing.assert_array_equal(loaded["pos_context"], pos_ctx)
        np.testing.assert_array_equal(loaded["pos_txtmask"], pos_mask)
        self.assertEqual(loaded["neg_context"].shape[0], 1)

        # Mismatched batch (2 cached rows with num_images=4) must raise instead of silently slicing
        with self.assertRaises(ValueError):
            _load_cached_prompt_embeds(payload, num_images=4, do_cfg=False)

        # Missing negative keys when do_cfg=True must raise
        with self.assertRaises(ValueError):
            _load_cached_prompt_embeds(
                {"pos_context": pos_ctx[:1], "pos_txtmask": pos_mask[:1]}, num_images=2, do_cfg=True
            )

    @unittest.skipUnless(HAS_JAX, "JAX is not installed in this environment")
    def test_infer_and_validate_tp_size_from_sharding(self):
        from types import SimpleNamespace
        from tpu_jax.dit import infer_and_validate_tp_size

        class _FakeSharding:
            def __init__(self, shards_by_axis):
                self._s = shards_by_axis

            def shard_shape(self, shape):
                return tuple(dim // self._s.get(i, 1) for i, dim in enumerate(shape))

        fake_arr = lambda shape, sh: SimpleNamespace(shape=shape, sharding=_FakeSharding(sh))
        tp2_params = {
            "blocks": {
                "qkvgate_w": fake_arr((2, 64, 192), {2: 2}),
                "mlp_gate_up_w": fake_arr((2, 64, 256), {2: 2}),
                "wo_w": fake_arr((2, 64, 64), {1: 2}),
                "mlp_down_w": fake_arr((2, 128, 64), {1: 2}),
            }
        }
        # Auto-inference returns 2 when tp_size=None
        self.assertEqual(infer_and_validate_tp_size(tp2_params, None), 2)
        self.assertEqual(infer_and_validate_tp_size(tp2_params, 2), 2)
        # Explicit mismatch raises instead of silently corrupting outputs
        with self.assertRaises(ValueError):
            infer_and_validate_tp_size(tp2_params, 1)

        # Inconsistent sharding across weights raises
        bad_params = {
            "blocks": {
                "qkvgate_w": fake_arr((2, 64, 192), {2: 2}),
                "wo_w": fake_arr((2, 64, 64), {}),
            }
        }
        with self.assertRaises(ValueError):
            infer_and_validate_tp_size(bad_params, None)


@unittest.skipUnless(HAS_JAX, "JAX is not installed in this environment")
class TestJaxQwen3TextEncoderParity(unittest.TestCase):
    def test_tiny_qwen3_vl_text_tower_matches_transformers(self):
        import jax.numpy as jnp
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
        from tpu_jax.text_encoder import qwen3_encode_tokens
        from tpu_jax.weights import build_qwen3_text_params_from_state_dict

        torch.manual_seed(123)
        hf_cfg = Qwen3VLConfig(
            text_config={
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 6,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "rms_norm_eps": 1e-6,
                "rope_theta": 5000000.0,
                "rope_scaling": {"rope_type": "default", "mrope_section": [4, 2, 2], "mrope_interleaved": True},
                "hidden_act": "silu",
                "attention_bias": False,
            },
            vision_config={
                "depth": 1,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_heads": 2,
                "out_hidden_size": 64,
                "deepstack_visual_indexes": [0],
            },
        )
        pt_model = Qwen3VLModel(hf_cfg).eval()
        # Select layers (1, 3, 5) out of 6 total -> exercises skipping layer 5 + final norm!
        jax_cfg = Qwen3TextConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            rms_norm_eps=1e-6,
            rope_theta=5000000.0,
            select_layers=(1, 3, 5),
            prefix_start_idx=4,
        )
        raw_sd = {k: v.detach().cpu().float().numpy() for k, v in pt_model.state_dict().items()}
        jax_params = build_qwen3_text_params_from_state_dict(raw_sd, jax_cfg, dtype=jnp.float32)

        # Prefix (4 tokens) + prompt + interior padding gap + 3 suffix tokens (exact encoder.py layout!)
        input_ids_np = np.array(
            [
                [10, 11, 12, 13, 20, 21, 22, 0, 0, 0, 30, 31, 32],
                [10, 11, 12, 13, 40, 41, 42, 43, 44, 0, 30, 31, 32],
            ],
            dtype=np.int64,
        )
        attn_mask_np = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
            ],
            dtype=np.bool_,
        )

        with torch.no_grad():
            pt_out = pt_model(
                input_ids=torch.from_numpy(input_ids_np),
                attention_mask=torch.from_numpy(attn_mask_np),
                output_hidden_states=True,
                use_cache=False,
            )
            ref_h = torch.stack([pt_out.hidden_states[i] for i in jax_cfg.select_layers], dim=2)
            ref_h = ref_h[:, jax_cfg.prefix_start_idx :].numpy()
            ref_m = attn_mask_np[:, jax_cfg.prefix_start_idx :]

        for impl in ("einsum", "xla"):
            got_h, got_m = qwen3_encode_tokens(
                jax_params,
                jax_cfg,
                jnp.asarray(input_ids_np, dtype=jnp.int32),
                jnp.asarray(attn_mask_np, dtype=jnp.bool_),
                attn_impl=impl,
            )
            got_h_np = np.asarray(got_h)
            got_m_np = np.asarray(got_m)
            np.testing.assert_array_equal(got_m_np, ref_m)
            # Compare valid (non-padded) token positions strictly
            valid_idx = np.where(ref_m)
            np.testing.assert_allclose(
                got_h_np[valid_idx],
                ref_h[valid_idx],
                rtol=1e-4,
                atol=2e-4,
                err_msg=f"Qwen3 text encoder mismatch for attn_impl={impl}",
            )


@unittest.skipUnless(HAS_JAX, "JAX is not installed in this environment")
class TestJaxVAEDecoderParity(unittest.TestCase):
    def test_tiny_qwen_image_vae_decoder_matches_diffusers(self):
        import jax.numpy as jnp
        from diffusers import AutoencoderKLQwenImage
        from tpu_jax.vae import vae_decode_nhwc
        from tpu_jax.weights import build_vae_decoder_params_from_state_dict

        torch.manual_seed(99)
        mean = [0.1, -0.2, 0.3, -0.4]
        std = [1.2, 0.8, 1.5, 0.9]
        pt_vae = AutoencoderKLQwenImage(
            base_dim=16,
            z_dim=4,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=1,
            attn_scales=[],
            temperal_downsample=[False, True, True],
            latents_mean=mean,
            latents_std=std,
        ).eval()

        jax_cfg = QwenImageVAEConfig(
            base_dim=16,
            z_dim=4,
            dim_mult=(1, 2, 4, 4),
            num_res_blocks=1,
            compression=8,
            latents_mean=tuple(mean),
            latents_std=tuple(std),
        )
        raw_sd = {k: v.detach().cpu().float().numpy() for k, v in pt_vae.state_dict().items()}
        jax_params = build_vae_decoder_params_from_state_dict(raw_sd, jax_cfg, dtype=jnp.float32)

        lat_pt = torch.randn(1, 4, 6, 8, dtype=torch.float32)
        with torch.no_grad():
            mean_t = torch.tensor(mean).view(1, 4, 1, 1, 1)
            std_t = torch.tensor(std).view(1, 4, 1, 1, 1)
            x5 = lat_pt.unsqueeze(2) * std_t + mean_t
            ref_nchw = pt_vae.decode(x5).sample[:, :, 0, :, :]
            ref_nhwc = ref_nchw.permute(0, 2, 3, 1).numpy()

        # Test both unchunked (q_chunk_size=1024) and chunked lax.scan (q_chunk_size=16) paths!
        for q_chunk in (1024, 16):
            got = np.asarray(
                vae_decode_nhwc(jax_params, jax_cfg, jnp.asarray(lat_pt.numpy()), q_chunk_size=q_chunk)
            )
            np.testing.assert_allclose(
                got,
                ref_nhwc,
                rtol=1e-4,
                atol=2e-4,
                err_msg=f"VAE decoder mismatch for q_chunk_size={q_chunk}",
            )


@unittest.skipUnless(HAS_JAX, "JAX is not installed in this environment")
class TestSafetensorsStreamingRoundTrip(unittest.TestCase):
    def test_lazy_safetensors_loaders_for_all_three_stages(self):
        import tempfile
        import jax.numpy as jnp
        import mmdit
        from diffusers import AutoencoderKLQwenImage
        from safetensors.numpy import save_file
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
        from tpu_jax.weights import (
            load_dit_weights,
            load_qwen3_text_weights,
            load_vae_decoder_weights,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 1. Native DiT turbo.safetensors
            pt_cfg = mmdit.SingleMMDiTConfig(
                features=64, tdim=16, txtdim=32, heads=4, kvheads=2, multiplier=2,
                layers=2, patch=2, channels=4, txtlayers=2, txtheads=2, txtkvheads=2,
            )
            jax_dit_cfg = Krea2DiTConfig(
                features=64, tdim=16, txtdim=32, heads=4, kvheads=2, multiplier=2,
                layers=2, patch=2, channels=4, txtlayers=2, txtheads=2, txtkvheads=2,
            )
            pt_dit = mmdit.SingleStreamDiT(pt_cfg).eval()
            save_file(
                {k: v.detach().cpu().float().numpy() for k, v in pt_dit.state_dict().items()},
                str(root / "turbo.safetensors"),
            )
            _, dit_params = load_dit_weights(root, config=jax_dit_cfg, dtype="bfloat16")
            self.assertEqual(dit_params["blocks"]["qkvgate_w"].dtype, jnp.bfloat16)
            self.assertEqual(dit_params["blocks"]["prenorm"].dtype, jnp.float32)
            self.assertEqual(dit_params["blocks"]["qkvgate_w"].shape, (2, 64, 192))

            # 2. Text encoder text_encoder/model.safetensors
            (root / "text_encoder").mkdir()
            hf_cfg = Qwen3VLConfig(
                text_config={
                    "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
                    "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 2,
                    "head_dim": 16, "rms_norm_eps": 1e-6, "rope_theta": 5000000.0,
                    "rope_scaling": {"rope_type": "default", "mrope_section": [4, 2, 2], "mrope_interleaved": True},
                    "hidden_act": "silu", "attention_bias": False,
                },
                vision_config={
                    "depth": 1, "hidden_size": 16, "intermediate_size": 32,
                    "num_heads": 2, "out_hidden_size": 32, "deepstack_visual_indexes": [0],
                },
            )
            pt_te = Qwen3VLModel(hf_cfg).eval()
            save_file(
                {k: v.detach().cpu().float().numpy() for k, v in pt_te.state_dict().items()},
                str(root / "text_encoder" / "model.safetensors"),
            )
            te_cfg = Qwen3TextConfig(
                vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                num_attention_heads=2, num_key_value_heads=2, head_dim=16, select_layers=(1, 3),
            )
            _, te_params = load_qwen3_text_weights(root, config=te_cfg, dtype="bfloat16")
            self.assertEqual(te_params["layers"]["qkv_w"].shape[0], 3)  # skips layer 3!
            self.assertEqual(te_params["embed_tokens"].dtype, jnp.bfloat16)

            # 3. VAE vae/diffusion_pytorch_model.safetensors
            (root / "vae").mkdir()
            pt_vae = AutoencoderKLQwenImage(
                base_dim=16, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=1,
                attn_scales=[], temperal_downsample=[False, True, True],
                latents_mean=[0.0] * 4, latents_std=[1.0] * 4,
            ).eval()
            save_file(
                {k: v.detach().cpu().float().numpy() for k, v in pt_vae.state_dict().items()},
                str(root / "vae" / "diffusion_pytorch_model.safetensors"),
            )
            vae_cfg = QwenImageVAEConfig(
                base_dim=16, z_dim=4, dim_mult=(1, 2, 4, 4), num_res_blocks=1,
                latents_mean=(0.0,) * 4, latents_std=(1.0,) * 4,
            )
            _, vae_params = load_vae_decoder_weights(root, config=vae_cfg, dtype="bfloat16")
            self.assertEqual(vae_params["conv_in_w"].dtype, jnp.bfloat16)
            self.assertEqual(vae_params["conv_in_w"].shape, (3, 3, 4, 64))


if __name__ == "__main__":
    unittest.main()
