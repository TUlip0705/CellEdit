from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """
    timesteps: (B,) in [0,1]
    returns: (B, dim)
    """
    if timesteps.ndim != 1:
        timesteps = timesteps.view(-1)
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, device=device, dtype=torch.float32) / half
    ).to(dtype=timesteps.dtype)
    args = timesteps[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)  # [2, H, W]
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class PatchEmbed(nn.Module):
    def __init__(self, input_size: int, patch_size: int, in_chans: int, embed_dim: int, bias: bool = True):
        super().__init__()
        if input_size % patch_size != 0:
            raise ValueError(f"input_size must be divisible by patch_size: {input_size=} {patch_size=}")
        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.grid_size = self.input_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        x = self.proj(x)  # [B,D,H',W']
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B,T,D]
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=0.0, bias=True, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.mha(x, x, x, need_weights=False)
        return y


class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class AdaLNBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, mlp_ratio=mlp_ratio)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


def build_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


@dataclass
class LatentSiTConfig:
    input_size: int = 64
    patch_size: int = 2
    in_channels: int = 24
    hidden_size: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    encoder_depth: int = 4

    time_dim: int = 128
    cond_dim: int = 384

    teacher_dim: int = 384
    projector_type: Literal["mlp", "conv", "none"] = "mlp"
    projector_dim: int = 2048
    conv_kernel: int = 3

    use_input_bn: bool = False
    bn_momentum: float = 0.1
    bn_eps: float = 1e-4


class LatentSiT(nn.Module):
    """
    SiT-style transformer backbone in 24ch VAE latent space.

    Inputs:
      z_t:  (B,C,H,W)
      t:    (B,) in [0,1]
      cond: (B,cond_dim)

    Output:
      v:    (B,C,H,W) velocity field

    Optional:
      return_proj=True returns projected tokens (B,T,teacher_dim) for REPA alignment.
    """

    def __init__(self, cfg: LatentSiTConfig):
        super().__init__()
        self.cfg = cfg
        self.in_channels = int(cfg.in_channels)
        self.out_channels = int(cfg.in_channels)
        self.patch_size = int(cfg.patch_size)
        self.hidden_size = int(cfg.hidden_size)

        self.input_bn: nn.BatchNorm2d | None = None
        if bool(getattr(cfg, "use_input_bn", False)):
            self.input_bn = nn.BatchNorm2d(
                int(cfg.in_channels),
                eps=float(getattr(cfg, "bn_eps", 1e-4)),
                momentum=float(getattr(cfg, "bn_momentum", 0.1)),
                affine=False,
                track_running_stats=True,
            )
            self.input_bn.reset_running_stats()

        self.x_embedder = PatchEmbed(
            input_size=int(cfg.input_size),
            patch_size=int(cfg.patch_size),
            in_chans=int(cfg.in_channels),
            embed_dim=int(cfg.hidden_size),
            bias=True,
        )
        num_patches = int(self.x_embedder.num_patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, int(cfg.hidden_size)), requires_grad=False)

        self.time_mlp = nn.Sequential(
            nn.Linear(int(cfg.time_dim), int(cfg.hidden_size)),
            nn.SiLU(),
            nn.Linear(int(cfg.hidden_size), int(cfg.hidden_size)),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(int(cfg.cond_dim), int(cfg.hidden_size)),
            nn.SiLU(),
            nn.Linear(int(cfg.hidden_size), int(cfg.hidden_size)),
        )

        self.blocks = nn.ModuleList(
            [
                AdaLNBlock(int(cfg.hidden_size), int(cfg.num_heads), mlp_ratio=float(cfg.mlp_ratio))
                for _ in range(int(cfg.depth))
            ]
        )
        self.final_layer = FinalLayer(int(cfg.hidden_size), int(cfg.patch_size), int(self.out_channels))

        # projector for REPA
        self.projector_type = str(getattr(cfg, "projector_type", "mlp")).lower()
        if self.projector_type == "none":
            self.proj = None
        elif self.projector_type == "mlp":
            self.proj = build_mlp(int(cfg.hidden_size), int(cfg.projector_dim), int(cfg.teacher_dim))
        elif self.projector_type == "conv":
            k = int(getattr(cfg, "conv_kernel", 3))
            pad = k // 2
            self.proj = nn.Conv2d(int(cfg.hidden_size), int(cfg.teacher_dim), kernel_size=k, padding=pad)
        else:
            raise ValueError(f"Unsupported projector_type: {self.projector_type}")

        self.initialize_weights()

    def initialize_weights(self) -> None:
        # sin-cos pos embed
        grid_size = int(self.x_embedder.grid_size)
        pos = get_2d_sincos_pos_embed(int(self.pos_embed.shape[-1]), grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        # patch embed init like linear
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.x_embedder.proj.bias is not None:
            nn.init.constant_(self.x_embedder.proj.bias, 0.0)

        # time/cond mlp
        nn.init.normal_(self.time_mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_mlp[2].weight, std=0.02)
        nn.init.normal_(self.cond_mlp[0].weight, std=0.02)
        nn.init.normal_(self.cond_mlp[2].weight, std=0.02)
        if self.time_mlp[0].bias is not None:
            nn.init.constant_(self.time_mlp[0].bias, 0.0)
        if self.time_mlp[2].bias is not None:
            nn.init.constant_(self.time_mlp[2].bias, 0.0)
        if self.cond_mlp[0].bias is not None:
            nn.init.constant_(self.cond_mlp[0].bias, 0.0)
        if self.cond_mlp[2].bias is not None:
            nn.init.constant_(self.cond_mlp[2].bias, 0.0)

        # adaLN-zero
        for blk in self.blocks:
            nn.init.constant_(blk.adaLN_modulation[-1].weight, 0.0)
            nn.init.constant_(blk.adaLN_modulation[-1].bias, 0.0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0.0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0.0)
        nn.init.constant_(self.final_layer.linear.weight, 0.0)
        nn.init.constant_(self.final_layer.linear.bias, 0.0)

        # projector init
        if self.proj is not None:
            for m in self.proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, p^2 * C)
        return: (B, C, H, W)
        """
        b, t, _ = x.shape
        c = int(self.out_channels)
        p = int(self.patch_size)
        h = w = int(math.isqrt(t))
        if h * w != t:
            raise ValueError(f"T must be a square, got {t}")
        x = x.view(b, h, w, p, p, c)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(b, c, h * p, w * p)

    def _encode_tokens(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        *,
        align_only: bool = False,
        apply_input_bn: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.input_bn is not None and bool(apply_input_bn):
            z_t = self.input_bn(z_t)

        x = self.x_embedder(z_t) + self.pos_embed  # (B,T,D)

        t_emb = sinusoidal_time_embedding(t, int(self.cfg.time_dim))
        t_emb = self.time_mlp(t_emb)
        c_emb = self.cond_mlp(cond)
        c = t_emb + c_emb

        proj_tokens: torch.Tensor | None = None
        enc_depth = int(getattr(self.cfg, "encoder_depth", 0))
        if enc_depth <= 0 or enc_depth > len(self.blocks):
            enc_depth = len(self.blocks)

        for i, blk in enumerate(self.blocks):
            x = blk(x, c)
            if (i + 1) == enc_depth:
                if self.proj is not None:
                    proj_tokens = x
                if align_only:
                    break
        return x, c, proj_tokens

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        *,
        return_proj: bool = False,
        align_only: bool = False,
        apply_input_bn: bool = True,
    ):
        if t.ndim != 1:
            t = t.view(-1)

        x, c, proj_tokens = self._encode_tokens(
            z_t, t, cond, align_only=bool(align_only), apply_input_bn=bool(apply_input_bn)
        )

        proj_out = None
        if return_proj and self.proj is not None:
            src = proj_tokens if proj_tokens is not None else x
            if self.projector_type in ("mlp",):
                b, tt, d = src.shape
                flat = src.reshape(b * tt, d)
                out = self.proj(flat).reshape(b, tt, -1)
                proj_out = out
            elif self.projector_type == "conv":
                b, tt, d = src.shape
                h = w = int(math.isqrt(tt))
                if h * w != tt:
                    raise ValueError(f"conv projector expects square grid tokens, got T={tt}")
                feat = src.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()  # (B,D,H,W)
                y = self.proj(feat)  # (B,teacher_dim,H,W)
                proj_out = y.permute(0, 2, 3, 1).reshape(b, tt, -1).contiguous()

        if align_only:
            if return_proj and proj_out is not None:
                return proj_out
            raise ValueError("align_only requires return_proj=True and a non-null projector.")

        x = self.final_layer(x, c)
        v = self.unpatchify(x)
        if return_proj and proj_out is not None:
            return v, proj_out
        return v
