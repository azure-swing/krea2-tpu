"""RAM-backed tmpfs checkpoint downloads for Krea 2 on Google Cloud TPU.

Adapted from MOSS-Image-TPU's ``tpu/model_cache.py`` and sized for Krea 2's
~35.7 GiB minimal checkpoint bundle (``turbo.safetensors`` + ``text_encoder``
+ ``tokenizer`` + ``vae``).

Key behaviors on Linux TPU VMs (Colab TPU v6e-1 / GCP Cloud TPU):
1. On Colab (``COLAB_RELEASE_TAG`` as root), dynamically mounts a dedicated
   ``tmpfs`` at ``/content/.krea2-tpu-ram`` sized up to
   ``min(available_ram - HOST_HEADROOM, MAX_TMPFS_GIB)`` so the full ~35.7 GiB
   Krea 2 bundle fits in RAM (Colab's default ``/dev/shm`` is typically too small).
2. On non-Colab Linux VMs, reuses ``/dev/shm`` when it is a ``tmpfs`` with
   sufficient free space.
3. Queries ``HfApi().model_info(files_metadata=True)`` to count ONLY missing
   bytes (already-cached files are not double-counted), reserves host RAM
   headroom for parallel safetensors loading, and falls back cleanly to the
   standard disk cache when RAM is insufficient or ``KREA2_TPU_DISK_CACHE=1``.
4. Writes per-component completion markers (``.krea2-components/<name>``) so
   offline / repeated runs reuse the RAM snapshot safely without network calls.
"""

from __future__ import annotations

import errno
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from filelock import FileLock
from huggingface_hub import HfApi, snapshot_download, try_to_load_from_cache
from huggingface_hub.constants import HF_HUB_OFFLINE
from huggingface_hub.utils import LocalEntryNotFoundError, filter_repo_objects

_LOG = logging.getLogger(__name__)
_COLAB_RAM = Path("/content/.krea2-tpu-ram")
_SHARED_RAM = Path("/dev/shm")
_GIB = 1024**3
# Reserve headroom for JAX/XLA runtime + single streaming field working buffer (~11.3 GiB max).
_HOST_HEADROOM = 16 * _GIB
_MAX_TMPFS_GIB = 64
_COMPONENT_MARKERS = ".krea2-components"


def host_memory_stats_gib() -> dict[str, float | None]:
    """Read Linux ``/proc/meminfo`` + process RSS without requiring ``psutil``."""
    out: dict[str, float | None] = {
        "total_gib": None,
        "available_gib": None,
        "process_rss_gib": None,
    }
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        vals: dict[str, int] = {}
        for line in meminfo.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].endswith(":"):
                vals[parts[0][:-1]] = int(parts[1]) * 1024
        if "MemTotal" in vals:
            out["total_gib"] = round(vals["MemTotal"] / _GIB, 3)
        if "MemAvailable" in vals:
            out["available_gib"] = round(vals["MemAvailable"] / _GIB, 3)
    except OSError:
        pass
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                out["process_rss_gib"] = round((int(line.split()[1]) * 1024) / _GIB, 3)
                break
    except OSError:
        pass
    return out


def minimal_krea2_patterns(
    *,
    checkpoint_file: str = "turbo.safetensors",
    components: tuple[str, ...] = ("dit", "text_encoder", "tokenizer", "vae"),
) -> list[str]:
    """Minimal HF allow_patterns skipping duplicate 26 GB diffusers transformer shards."""
    patterns = ["model_index.json", "scheduler/scheduler_config.json"]
    if "dit" in components:
        patterns.append(checkpoint_file)
    if "text_encoder" in components:
        patterns.extend(
            [
                "text_encoder/config.json",
                "text_encoder/model.safetensors",
                "text_encoder/model.safetensors.index.json",
                "text_encoder/model-*.safetensors",
            ]
        )
    if "tokenizer" in components:
        patterns.extend(["tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"])
    if "vae" in components:
        patterns.extend(["vae/config.json", "vae/diffusion_pytorch_model.safetensors"])
    return patterns


def _is_tmpfs(path: Path) -> bool:
    try:
        for entry in Path("/proc/self/mounts").read_text(encoding="utf-8").splitlines():
            parts = entry.split()
            if len(parts) >= 3 and parts[1] == str(path) and parts[2] == "tmpfs":
                return True
    except OSError:
        return False
    return False


def _desired_tmpfs_size_gib(available_gib: float | None) -> int:
    if available_gib is None:
        return 48
    headroom_gib = _HOST_HEADROOM / _GIB
    usable = int(max(0.0, available_gib - headroom_gib))
    return max(16, min(_MAX_TMPFS_GIB, usable))


def _ram_cache(create: bool, available_gib: float | None = None) -> Path | None:
    try:
        euid = os.geteuid()
        if (
            create
            and os.getenv("COLAB_RELEASE_TAG")
            and euid == 0
            and _COLAB_RAM.parent.is_dir()
        ):
            with FileLock(str(_COLAB_RAM) + ".lock"):
                if not (_COLAB_RAM.is_mount() or _COLAB_RAM.is_symlink()):
                    _COLAB_RAM.mkdir(mode=0o700, exist_ok=True)
                    if not any(_COLAB_RAM.iterdir()):
                        size_gib = _desired_tmpfs_size_gib(available_gib)
                        subprocess.run(
                            [
                                "mount",
                                "-t",
                                "tmpfs",
                                "-o",
                                f"size={size_gib}G,nosuid,nodev,noexec,mode=0700",
                                "tmpfs",
                                str(_COLAB_RAM),
                            ],
                            check=False,
                            capture_output=True,
                        )
        for mount in (_COLAB_RAM, _SHARED_RAM):
            if not _is_tmpfs(mount):
                continue
            cache = mount / f"krea2-hub-{euid}"
            if create:
                cache.mkdir(mode=0o700, exist_ok=True)
            if cache.is_dir() and not cache.is_symlink() and cache.stat().st_uid == euid:
                return cache
    except (OSError, AttributeError) as exc:
        _LOG.debug("TPU RAM cache unavailable: %s", exc)
    return None


def download_to_ram(
    source: str,
    patterns: list[str],
    components: tuple[str, ...],
    available_gib: float | None,
    *,
    local_files_only: bool = False,
    token: str | None = None,
) -> Path | None:
    """Download minimal Krea 2 checkpoint snapshot into RAM-backed ``tmpfs`` when feasible."""
    if sys.platform != "linux" or os.getenv("KREA2_TPU_DISK_CACHE") == "1":
        return None
    offline = local_files_only or bool(HF_HUB_OFFLINE)
    cache = _ram_cache(create=not offline, available_gib=available_gib)
    if cache is None:
        return None

    options: dict[str, Any] = {"cache_dir": cache, "allow_patterns": patterns}
    if token:
        options["token"] = token

    if offline:
        try:
            path = Path(snapshot_download(source, local_files_only=True, **options))
        except LocalEntryNotFoundError:
            return None
        if (path / "model_index.json").is_file() and all(
            (path / _COMPONENT_MARKERS / name).is_file() for name in components
        ):
            return path
        return None

    with FileLock(str(cache / ".krea2-download.lock")):
        try:
            info = HfApi(token=token).model_info(source, files_metadata=True, timeout=15)
        except Exception as exc:
            _LOG.debug("Could not size Krea 2 TPU RAM download: %s", exc)
            return None

        files = {
            file.rfilename: file.size
            for file in filter_repo_objects(
                info.siblings or (),
                allow_patterns=patterns,
                key=lambda file: file.rfilename,
            )
        }
        if not files or any(size is None for size in files.values()):
            return None

        missing_bytes = sum(
            size
            for name, size in files.items()
            if not isinstance(
                try_to_load_from_cache(source, name, cache_dir=cache, revision=info.sha),
                str,
            )
        )
        free_ram = max(0, int((available_gib or 0.0) * _GIB) - _HOST_HEADROOM)
        free_tmpfs = shutil.disk_usage(cache).free
        if missing_bytes > min(free_ram, free_tmpfs):
            _LOG.warning(
                "Insufficient RAM/tmpfs for Krea 2 download (need %.2f GiB, free_ram=%.2f GiB, "
                "free_tmpfs=%.2f GiB); falling back to disk cache.",
                missing_bytes / _GIB,
                free_ram / _GIB,
                free_tmpfs / _GIB,
            )
            return None

        try:
            path = Path(snapshot_download(source, **options))
        except (OSError, RuntimeError) as exc:
            if (
                getattr(exc, "errno", None) not in (errno.ENOSPC, errno.EDQUOT)
                and "No space left on device" not in str(exc)
            ):
                raise
            _LOG.warning("TPU RAM cache filled during download; falling back to disk cache.")
            return None

        if not all((path / name).is_file() for name in files):
            return None

        markers = path / _COMPONENT_MARKERS
        markers.mkdir(exist_ok=True)
        for name in components:
            (markers / name).touch()
        return path


def resolve_model_bundle(
    source: str | Path,
    *,
    checkpoint_file: str | None = None,
    components: tuple[str, ...] = ("dit", "text_encoder", "tokenizer", "vae"),
    token: str | None = None,
    local_files_only: bool = False,
    disk_cache_dir: str | Path | None = None,
    prefer_ram: bool = True,
) -> tuple[Path, str]:
    """Resolve local path or HF repo to ``(snapshot_path, backend)``.

    ``backend`` is one of ``"local"``, ``"ram_tmpfs"``, or ``"disk"``.
    """
    src_path = Path(source)
    if src_path.exists():
        return src_path, "local"

    repo_id = str(source)
    ckpt = checkpoint_file or ("raw.safetensors" if "raw" in repo_id.lower() else "turbo.safetensors")
    patterns = minimal_krea2_patterns(checkpoint_file=ckpt, components=components)
    eff_token = token or os.environ.get("HF_TOKEN")

    if prefer_ram:
        avail_gib = host_memory_stats_gib()["available_gib"]
        ram_path = download_to_ram(
            repo_id,
            patterns,
            components,
            avail_gib,
            local_files_only=local_files_only,
            token=eff_token,
        )
        if ram_path is not None:
            return ram_path, "ram_tmpfs"

    dl_kwargs: dict[str, Any] = {
        "allow_patterns": patterns,
        "local_files_only": local_files_only,
    }
    if eff_token:
        dl_kwargs["token"] = eff_token
    if disk_cache_dir:
        dl_kwargs["local_dir"] = str(disk_cache_dir)
    path = Path(snapshot_download(repo_id, **dl_kwargs))
    return path, "disk"
