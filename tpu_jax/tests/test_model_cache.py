"""Unit tests for RAM-backed model cache and parallel streaming weight loading."""

from __future__ import annotations

import errno
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tpu_jax import model_cache  # noqa: E402

HAS_JAX = importlib.util.find_spec("jax") is not None


class TestModelCache(unittest.TestCase):
    repo = "krea/Krea-2-Turbo"
    components = ("dit", "vae")
    patterns = model_cache.minimal_krea2_patterns(
        checkpoint_file="turbo.safetensors", components=("dit", "vae")
    )

    def test_minimal_patterns_skip_duplicate_diffusers_transformer(self):
        pats = model_cache.minimal_krea2_patterns(checkpoint_file="turbo.safetensors")
        self.assertIn("turbo.safetensors", pats)
        self.assertFalse(any("transformer" in p for p in pats))
        self.assertIn("text_encoder/model.safetensors", pats)
        self.assertIn("vae/diffusion_pytorch_model.safetensors", pats)

    def test_dynamic_tmpfs_sizing_respects_headroom_and_cap(self):
        # 170 GiB Colab v6e-1 VM -> capped at 64 GiB
        self.assertEqual(model_cache._desired_tmpfs_size_gib(170.0), 64)
        # 60 GiB VM -> 60 - 16 = 44 GiB
        self.assertEqual(model_cache._desired_tmpfs_size_gib(60.0), 44)
        # Low RAM VM -> floor 16 GiB (and download_to_ram will still reject if too small)
        self.assertEqual(model_cache._desired_tmpfs_size_gib(20.0), 16)

    def test_resolve_prefers_ram_tmpfs_when_available(self):
        with (
            patch.object(model_cache, "host_memory_stats_gib", return_value={"available_gib": 120.0}),
            patch.object(model_cache, "download_to_ram", return_value=Path("ram-snapshot")) as ram,
            patch.object(model_cache, "snapshot_download") as disk,
        ):
            path, backend = model_cache.resolve_model_bundle(self.repo)
            self.assertEqual(path, Path("ram-snapshot"))
            self.assertEqual(backend, "ram_tmpfs")
            disk.assert_not_called()
            self.assertIn("turbo.safetensors", ram.call_args.args[1])

    def test_offline_reuses_only_complete_ram_downloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model_index.json").touch()
            with (
                patch.object(model_cache.sys, "platform", "linux"),
                patch.dict(model_cache.os.environ, {"KREA2_TPU_DISK_CACHE": "0"}),
                patch.object(model_cache, "_ram_cache", return_value=root) as cache,
                patch.object(model_cache, "snapshot_download", return_value=root) as download,
                patch.object(model_cache, "HfApi") as api,
            ):
                # Missing component markers -> rejected
                self.assertIsNone(
                    model_cache.download_to_ram(
                        self.repo, self.patterns, self.components, 100.0, local_files_only=True
                    )
                )
                cache.assert_called_once_with(create=False, available_gib=100.0)

                # Create both component markers -> accepted offline without HfApi call
                markers = root / model_cache._COMPONENT_MARKERS
                markers.mkdir()
                (markers / "dit").touch()
                (markers / "vae").touch()
                got = model_cache.download_to_ram(
                    self.repo, self.patterns, self.components, 100.0, local_files_only=True
                )
                self.assertEqual(got, root)
                self.assertTrue(download.call_args.kwargs["local_files_only"])
                api.assert_not_called()

    def test_ram_download_counts_only_missing_bytes_and_handles_enospc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model_index.json").touch()
            (root / "turbo.safetensors").touch()
            (root / "vae").mkdir()
            (root / "vae/diffusion_pytorch_model.safetensors").touch()

            files = [
                SimpleNamespace(rfilename="model_index.json", size=512),
                SimpleNamespace(rfilename="turbo.safetensors", size=26 * 1024**3),
                SimpleNamespace(rfilename="vae/diffusion_pytorch_model.safetensors", size=500 * 1024**2),
                SimpleNamespace(rfilename="unused.bin", size=None),
            ]
            usage_huge = SimpleNamespace(total=80 * 1024**3, used=0, free=80 * 1024**3)
            with (
                patch.object(model_cache.sys, "platform", "linux"),
                patch.dict(model_cache.os.environ, {"KREA2_TPU_DISK_CACHE": "0"}),
                patch.object(model_cache, "HF_HUB_OFFLINE", False),
                patch.object(model_cache, "_ram_cache", return_value=root),
                patch.object(model_cache.shutil, "disk_usage", return_value=usage_huge),
                patch.object(model_cache, "HfApi") as api,
                patch.object(model_cache, "try_to_load_from_cache", return_value=None) as cached,
                patch.object(model_cache, "snapshot_download", return_value=root) as download,
            ):
                api.return_value.model_info.return_value = SimpleNamespace(siblings=files, sha="abc")
                # 30 GiB available - 16 GiB headroom = 14 GiB < 26.5 GiB needed -> fallback None
                self.assertIsNone(
                    model_cache.download_to_ram(
                        self.repo, self.patterns, self.components, 30.0, local_files_only=False
                    )
                )
                download.assert_not_called()

                # When already cached (missing_bytes == 0), 30 GiB RAM suffices!
                cached.return_value = str(root)
                got = model_cache.download_to_ram(
                    self.repo, self.patterns, self.components, 30.0, local_files_only=False
                )
                self.assertEqual(got, root)
                self.assertTrue((root / model_cache._COMPONENT_MARKERS / "dit").is_file())
                self.assertTrue((root / model_cache._COMPONENT_MARKERS / "vae").is_file())

                # ENOSPC during download falls back cleanly to None
                download.side_effect = OSError(errno.ENOSPC, "No space left on device")
                self.assertIsNone(
                    model_cache.download_to_ram(
                        self.repo, self.patterns, self.components, 100.0, local_files_only=False
                    )
                )


@unittest.skipUnless(HAS_JAX, "JAX is not installed in this environment")
class TestParallelStreamingAndRelease(unittest.TestCase):
    def test_parallel_loader_matches_single_thread_and_release_deletes_leaves(self):
        import jax
        import torch
        from safetensors.numpy import save_file

        torch._dynamo.config.disable = True
        import mmdit
        from tpu_jax.config import Krea2DiTConfig
        from tpu_jax.weights import load_dit_weights, release_tpu_pytree

        torch.manual_seed(5)
        pt_cfg = mmdit.SingleMMDiTConfig(
            features=64, tdim=16, txtdim=32, heads=4, kvheads=2, multiplier=2,
            layers=4, patch=2, channels=4, txtlayers=2, txtheads=2, txtkvheads=2,
        )
        jax_cfg = Krea2DiTConfig(
            features=64, tdim=16, txtdim=32, heads=4, kvheads=2, multiplier=2,
            layers=4, patch=2, channels=4, txtlayers=2, txtheads=2, txtkvheads=2,
        )
        pt_dit = mmdit.SingleStreamDiT(pt_cfg).eval()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_file(
                {k: v.detach().cpu().float().numpy() for k, v in pt_dit.state_dict().items()},
                str(root / "turbo.safetensors"),
            )
            dev = jax.devices()[0]
            _, p_seq = load_dit_weights(root, config=jax_cfg, dtype="bfloat16", device=dev, num_workers=1)
            _, p_par = load_dit_weights(root, config=jax_cfg, dtype="bfloat16", device=dev, num_workers=4)

            seq_leaves = jax.tree_util.tree_leaves(p_seq)
            par_leaves = jax.tree_util.tree_leaves(p_par)
            self.assertEqual(len(seq_leaves), len(par_leaves))
            for a, b in zip(seq_leaves, par_leaves):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

            drop_sec, freed = release_tpu_pytree(p_par)
            self.assertGreaterEqual(drop_sec, 0.0)
            self.assertEqual(freed, len(par_leaves))
            self.assertTrue(all(leaf.is_deleted() for leaf in par_leaves))
            release_tpu_pytree(p_seq)


if __name__ == "__main__":
    unittest.main()
