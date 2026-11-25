"""Model configurations for Krea 2 native JAX inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Krea2DiTConfig:
    """Configuration for Krea 2 SingleStreamDiT (12B GQA DiT)."""

    features: int = 6144
    tdim: int = 256
    txtdim: int = 2560
    heads: int = 48
    kvheads: int = 12
    multiplier: int = 4
    layers: int = 28
    patch: int = 2
    channels: int = 16
    bias: bool = False
    theta: float = 1000.0
    ntk: float = 1.0
    norm_eps: float = 1e-5
    txtlayers: int = 12
    txtheads: int = 20
    txtkvheads: int = 20
    num_layerwise_text_blocks: int = 2
    num_refiner_text_blocks: int = 2
    intermediate_size: int | None = None
    text_intermediate_size: int | None = None

    @property
    def head_dim(self) -> int:
        return self.features // self.heads

    @property
    def text_head_dim(self) -> int:
        return self.txtdim // self.txtheads

    @property
    def mlp_dim(self) -> int:
        if self.intermediate_size is not None:
            return self.intermediate_size
        raw = int(2 * self.features / 3) * self.multiplier
        return 128 * ((raw + 127) // 128)

    @property
    def text_mlp_dim(self) -> int:
        if self.text_intermediate_size is not None:
            return self.text_intermediate_size
        raw = int(2 * self.txtdim / 3) * self.multiplier
        return 128 * ((raw + 127) // 128)

    @property
    def axes_dims(self) -> tuple[int, int, int]:
        hd = self.head_dim
        axes = (
            hd - 12 * (hd // 16),
            6 * (hd // 16),
            6 * (hd // 16),
        )
        assert sum(axes) == hd, f"sum(axes)={sum(axes)} != head_dim={hd}"
        assert all(a % 2 == 0 for a in axes), f"axes must be even, got {axes}"
        return axes

    @classmethod
    def from_diffusers_json(cls, path: str | Path) -> "Krea2DiTConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        heads = int(raw.get("num_attention_heads", 48))
        head_dim = int(raw.get("attention_head_dim", 128))
        features = heads * head_dim
        patch = 2
        in_channels = int(raw.get("in_channels", 64))
        channels = in_channels // (patch * patch)
        return cls(
            features=features,
            tdim=int(raw.get("timestep_embed_dim", 256)),
            txtdim=int(raw.get("text_hidden_dim", 2560)),
            heads=heads,
            kvheads=int(raw.get("num_key_value_heads", 12)),
            layers=int(raw.get("num_layers", 28)),
            patch=patch,
            channels=channels,
            theta=float(raw.get("rope_theta", 1000.0)),
            norm_eps=float(raw.get("norm_eps", 1e-5)),
            txtlayers=int(raw.get("num_text_layers", 12)),
            txtheads=int(raw.get("text_num_attention_heads", 20)),
            txtkvheads=int(raw.get("text_num_key_value_heads", 20)),
            num_layerwise_text_blocks=int(raw.get("num_layerwise_text_blocks", 2)),
            num_refiner_text_blocks=int(raw.get("num_refiner_text_blocks", 2)),
            intermediate_size=raw.get("intermediate_size"),
            text_intermediate_size=raw.get("text_intermediate_size"),
        )


@dataclass(frozen=True)
class Qwen3TextConfig:
    """Configuration for the Qwen3-VL-4B text tower used by Krea 2."""

    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    max_length: int = 512
    select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
    prompt_prefix: str = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
        "<|im_end|>\n<|im_start|>user\n"
    )
    prompt_suffix: str = "<|im_end|>\n<|im_start|>assistant\n"
    prefix_start_idx: int = 34
    suffix_token_count: int = 5

    @property
    def num_active_layers(self) -> int:
        """Number of decoder layers required to reach max(select_layers).

        In HuggingFace transformers, hidden_states[0] is embed_tokens output and
        hidden_states[k] (1 <= k <= num_hidden_layers - 1) is the output of layer k - 1
        (un-normalized). hidden_states[num_hidden_layers] is final norm(layers[-1]).
        """
        max_sel = max(self.select_layers)
        return min(max_sel, self.num_hidden_layers)

    @classmethod
    def from_json(cls, path: str | Path, **overrides) -> "Qwen3TextConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        tc = raw.get("text_config", raw)
        rope_params = tc.get("rope_parameters") or {}
        rope_theta = float(rope_params.get("rope_theta", tc.get("rope_theta", 5000000.0)))
        kwargs = dict(
            vocab_size=int(tc.get("vocab_size", 151936)),
            hidden_size=int(tc.get("hidden_size", 2560)),
            intermediate_size=int(tc.get("intermediate_size", 9728)),
            num_hidden_layers=int(tc.get("num_hidden_layers", 36)),
            num_attention_heads=int(tc.get("num_attention_heads", 32)),
            num_key_value_heads=int(tc.get("num_key_value_heads", 8)),
            head_dim=int(tc.get("head_dim", 128)),
            rms_norm_eps=float(tc.get("rms_norm_eps", 1e-6)),
            rope_theta=rope_theta,
        )
        kwargs.update(overrides)
        return cls(**kwargs)


@dataclass(frozen=True)
class QwenImageVAEConfig:
    """Configuration for AutoencoderKLQwenImage (f8, 16 latent channels)."""

    base_dim: int = 96
    z_dim: int = 16
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    compression: int = 8
    latents_mean: tuple[float, ...] = (
        -0.7571, -0.7089, -0.9113, 0.1075,
        -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632,
        -0.1922, -0.9497, 0.2503, -0.2921,
    )
    latents_std: tuple[float, ...] = (
        2.8184, 1.4541, 2.3275, 2.6558,
        1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579,
        1.6382, 1.1253, 2.8251, 1.9160,
    )

    @classmethod
    def from_json(cls, path: str | Path) -> "QwenImageVAEConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        td = raw.get("temperal_downsample", [False, True, True])
        return cls(
            base_dim=int(raw.get("base_dim", 96)),
            z_dim=int(raw.get("z_dim", 16)),
            dim_mult=tuple(int(x) for x in raw.get("dim_mult", (1, 2, 4, 4))),
            num_res_blocks=int(raw.get("num_res_blocks", 2)),
            compression=2 ** len(td),
            latents_mean=tuple(float(x) for x in raw.get("latents_mean", cls.latents_mean)),
            latents_std=tuple(float(x) for x in raw.get("latents_std", cls.latents_std)),
        )


KREA2_DEFAULT_CONFIG = Krea2DiTConfig()
