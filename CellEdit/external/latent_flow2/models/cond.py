from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math
import torch
import torch.nn as nn


class _KPGTProjector(nn.Module):
    """
    Project a fixed KPGT vector (e.g. 2304-d) into model hidden size.

    Designed so that a zero input stays near zero (biases initialized to 0).
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError(f"Invalid dims for KPGT projector: in_dim={in_dim}, out_dim={out_dim}")

        self.block1 = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=True),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Dropout(p=float(dropout)),
        )
        self.block2_ln = nn.LayerNorm(out_dim)
        self.block2 = nn.Sequential(
            nn.Linear(out_dim, out_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(p=float(dropout)),
        )
        self.out = nn.Linear(out_dim, out_dim, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.block1(x)
        h2 = self.block2(self.block2_ln(h1))
        h = h1 + h2
        return self.out(h)


def canonicalize_drug_mode(value: str | None) -> str:
    mode = str(value or "kpgt").strip().lower()
    aliases = {
        "kpgt": "kpgt",
        "morgan": "morgan",
        "morgan_fp": "morgan",
        "morganfp": "morgan",
        "fp": "morgan",
        "fingerprint": "morgan",
        "treatment_id": "treatment_id",
        "treat_id": "treatment_id",
        "discrete": "treatment_id",
        "disc": "treatment_id",
    }
    if mode not in aliases:
        raise ValueError(f"Unsupported cond.drug_mode: {value}")
    return aliases[mode]


def canonicalize_dose_mode(value: str | None) -> str:
    mode = str(value or "bins").strip().lower()
    aliases = {
        "bins": "bins",
        "discrete": "bins",
        "dose_id": "bins",
        "cont_film": "cont_film",
        "log10_film": "cont_film",
        "film": "cont_film",
    }
    if mode not in aliases:
        raise ValueError(f"Unsupported cond.dose_mode: {value}")
    return aliases[mode]


@dataclass(frozen=True)
class KPGTCondConfig:
    kpgt_dim: int = 2304
    hidden_dim: int = 384
    dose_bins: int = 9
    cond_drop_prob: float = 0.1
    keep_empty_id: bool = True
    kpgt_dropout: float = 0.1


class KPGTConditionEncoder(nn.Module):
    """
    Sum-fusion condition encoder:

      cond = LN( proj_kpgt(fp) + emb_dose(dose_id) + emb_empty(empty_id) )

    Classifier-free guidance:
      - learnable `uncond` vector
      - during training, with prob `cond_drop_prob`, replace cond with uncond
    """

    def __init__(self, cfg: KPGTCondConfig):
        super().__init__()
        self.cfg = cfg

        self.kpgt_proj = _KPGTProjector(int(cfg.kpgt_dim), int(cfg.hidden_dim), dropout=float(cfg.kpgt_dropout))
        self.emb_dose = nn.Embedding(int(cfg.dose_bins), int(cfg.hidden_dim))

        self.emb_empty: nn.Embedding | None
        if bool(cfg.keep_empty_id):
            self.emb_empty = nn.Embedding(2, int(cfg.hidden_dim))
        else:
            self.emb_empty = None

        self.out_norm = nn.LayerNorm(int(cfg.hidden_dim), elementwise_affine=False, eps=1e-6)

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

        self.uncond = nn.Parameter(torch.zeros(int(cfg.hidden_dim)))
        nn.init.normal_(self.uncond, std=0.02)

    def uncond_batch(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.uncond.to(device=device, dtype=dtype).unsqueeze(0).expand(int(batch_size), -1)

    def forward(
        self,
        *,
        kpgt_fp: torch.Tensor,   # (B, kpgt_dim)
        dose_id: torch.Tensor,   # (B,)
        empty_id: torch.Tensor,  # (B,)
        apply_cfg_dropout: bool = True,
    ) -> torch.Tensor:
        device = kpgt_fp.device
        dtype = kpgt_fp.dtype
        h = self.kpgt_proj(kpgt_fp.to(device=device, dtype=dtype))
        h = h + self.emb_dose(dose_id.to(device=device))
        if self.emb_empty is not None:
            h = h + self.emb_empty(empty_id.to(device=device))
        h = self.out_norm(h)

        if apply_cfg_dropout and float(self.cfg.cond_drop_prob) > 0.0 and self.training:
            drop = torch.rand((h.shape[0], 1), device=device) < float(self.cfg.cond_drop_prob)
            uncond = self.uncond_batch(h.shape[0], device=device, dtype=h.dtype)
            h = torch.where(drop, uncond, h)

        return h


@dataclass(frozen=True)
class TreatmentDoseCondConfig:
    num_treatments: int
    hidden_dim: int = 384
    dose_bins: int = 9
    cond_drop_prob: float = 0.1
    keep_empty_id: bool = True


class TreatmentDoseConditionEncoder(nn.Module):
    """
    Discrete treatment + discrete dose encoder:

      cond = LN( emb_treatment(treatment_id) + emb_dose(dose_id) + emb_empty(empty_id) )
    """

    def __init__(self, cfg: TreatmentDoseCondConfig):
        super().__init__()
        self.cfg = cfg
        if int(cfg.num_treatments) <= 0:
            raise ValueError(f"num_treatments must be >0, got {cfg.num_treatments}")
        if int(cfg.hidden_dim) <= 0:
            raise ValueError(f"hidden_dim must be >0, got {cfg.hidden_dim}")
        if int(cfg.dose_bins) <= 0:
            raise ValueError(f"dose_bins must be >0, got {cfg.dose_bins}")

        self.emb_treatment = nn.Embedding(int(cfg.num_treatments), int(cfg.hidden_dim))
        self.emb_dose = nn.Embedding(int(cfg.dose_bins), int(cfg.hidden_dim))

        self.emb_empty: nn.Embedding | None
        if bool(cfg.keep_empty_id):
            self.emb_empty = nn.Embedding(2, int(cfg.hidden_dim))
        else:
            self.emb_empty = None

        self.out_norm = nn.LayerNorm(int(cfg.hidden_dim), elementwise_affine=False, eps=1e-6)

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

        self.uncond = nn.Parameter(torch.zeros(int(cfg.hidden_dim)))
        nn.init.normal_(self.uncond, std=0.02)

    def uncond_batch(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.uncond.to(device=device, dtype=dtype).unsqueeze(0).expand(int(batch_size), -1)

    def forward(
        self,
        *,
        treatment_id: torch.Tensor,
        dose_id: torch.Tensor,
        empty_id: torch.Tensor,
        apply_cfg_dropout: bool = True,
    ) -> torch.Tensor:
        device = treatment_id.device
        h = self.emb_treatment(treatment_id.to(device=device))
        h = h + self.emb_dose(dose_id.to(device=device))
        if self.emb_empty is not None:
            h = h + self.emb_empty(empty_id.to(device=device))
        h = self.out_norm(h)

        if apply_cfg_dropout and float(self.cfg.cond_drop_prob) > 0.0 and self.training:
            drop = torch.rand((h.shape[0], 1), device=device) < float(self.cfg.cond_drop_prob)
            uncond = self.uncond_batch(h.shape[0], device=device, dtype=h.dtype)
            h = torch.where(drop, uncond, h)

        return h


class _DoseFourierEncoder(nn.Module):
    def __init__(self, *, fourier_k: int, out_dim: int, mlp_dim: int | None = None):
        super().__init__()
        kk = int(fourier_k)
        if kk <= 0:
            raise ValueError(f"fourier_k must be >0, got {fourier_k}")
        self.fourier_k = kk

        out_dim = int(out_dim)
        if out_dim <= 0:
            raise ValueError(f"out_dim must be >0, got {out_dim}")
        mlp_dim = int(mlp_dim) if mlp_dim is not None else out_dim
        if mlp_dim <= 0:
            raise ValueError(f"mlp_dim must be >0, got {mlp_dim}")

        self.mlp = nn.Sequential(
            nn.Linear(2 * kk, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, out_dim),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, dose_cont: torch.Tensor) -> torch.Tensor:
        """
        dose_cont: (B,) typically scaled to ~[-1,1]
        returns: (B, out_dim)
        """
        if dose_cont.ndim != 1:
            dose_cont = dose_cont.view(-1)

        device = dose_cont.device
        out_dtype = dose_cont.dtype

        # NOTE: sin/cos Fourier features are periodic. If we feed dose_cont in [-1,1]
        # directly with integer frequencies, values that differ by 1 become identical.
        # Map to an open interval (eps, 1-eps) to avoid wrap-around collisions.
        # Keep the Fourier path in float32: with fourier_k=16, the largest frequency is
        # 32768, which can overflow to inf under fp16 before sin/cos are applied.
        dose_f = dose_cont.to(device=device, dtype=torch.float32)
        eps = torch.tensor(1e-3, device=device, dtype=torch.float32)
        dose_u = 0.5 * (dose_f + 1.0)  # [-1,1] -> [0,1]
        dose_u = torch.clamp(dose_u, 0.0, 1.0)
        dose_u = dose_u * (1.0 - 2.0 * eps) + eps  # -> [eps, 1-eps]

        freqs = 2.0 ** torch.arange(self.fourier_k, device=device, dtype=torch.float32)
        args = (2.0 * math.pi) * dose_u[:, None] * freqs[None]
        phi = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        mlp_in = phi.to(dtype=self.mlp[0].weight.dtype)
        if device.type == "cuda":
            with torch.cuda.amp.autocast(enabled=False):
                out = self.mlp(mlp_in)
        else:
            out = self.mlp(mlp_in)
        return out.to(dtype=out_dtype)


@dataclass(frozen=True)
class KPGTFiLMCondConfig:
    kpgt_dim: int = 2304
    hidden_dim: int = 384
    cond_drop_prob: float = 0.1
    keep_empty_id: bool = True
    gate_empty_dose: bool = True
    kpgt_dropout: float = 0.1

    dose_fourier_k: int = 16
    dose_mlp_dim: int = 384
    dose_alpha_init: float = 0.1


class KPGTFiLMConditionEncoder(nn.Module):
    """
    Continuous-dose conditioning (KPGT "continuous version"):

      k  = proj_kpgt(fp)
      u  = DoseEncoder(dose_cont)              # scalar -> vector
      [s,b] = Linear(u)
      k' = k * (1+s) + b
      cond = LN( k + alpha*(k'-k) + emb_empty(empty_id) )

    Notes:
      - dose_cont is expected to be a continuous scalar (e.g. scaled log10 concentration).
      - EMPTY samples must set empty_id=1; by default we hard-gate dose path by multiplying by (1-empty_id).
      - gate_empty_dose controls whether EMPTY samples receive the continuous dose path.
      - Classifier-free guidance dropout matches KPGTConditionEncoder.
    """

    def __init__(self, cfg: KPGTFiLMCondConfig):
        super().__init__()
        self.cfg = cfg

        self.kpgt_proj = _KPGTProjector(int(cfg.kpgt_dim), int(cfg.hidden_dim), dropout=float(cfg.kpgt_dropout))
        self.dose_enc = _DoseFourierEncoder(
            fourier_k=int(cfg.dose_fourier_k),
            out_dim=int(cfg.hidden_dim),
            mlp_dim=int(cfg.dose_mlp_dim),
        )
        self.film = nn.Linear(int(cfg.hidden_dim), 2 * int(cfg.hidden_dim), bias=True)
        nn.init.normal_(self.film.weight, std=0.02)
        if self.film.bias is not None:
            nn.init.constant_(self.film.bias, 0.0)

        self.alpha = nn.Parameter(torch.tensor(float(cfg.dose_alpha_init), dtype=torch.float32))

        self.emb_empty: nn.Embedding | None
        if bool(cfg.keep_empty_id):
            self.emb_empty = nn.Embedding(2, int(cfg.hidden_dim))
            nn.init.normal_(self.emb_empty.weight, std=0.02)
        else:
            self.emb_empty = None

        self.out_norm = nn.LayerNorm(int(cfg.hidden_dim), elementwise_affine=False, eps=1e-6)

        self.uncond = nn.Parameter(torch.zeros(int(cfg.hidden_dim)))
        nn.init.normal_(self.uncond, std=0.02)

    def uncond_batch(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.uncond.to(device=device, dtype=dtype).unsqueeze(0).expand(int(batch_size), -1)

    def forward(
        self,
        *,
        kpgt_fp: torch.Tensor,     # (B, kpgt_dim)
        dose_cont: torch.Tensor,   # (B,)
        empty_id: torch.Tensor,    # (B,)
        apply_cfg_dropout: bool = True,
    ) -> torch.Tensor:
        device = kpgt_fp.device
        dtype = kpgt_fp.dtype

        k = self.kpgt_proj(kpgt_fp.to(device=device, dtype=dtype))

        # Hard gate dose path for EMPTY samples.
        if bool(self.cfg.gate_empty_dose):
            m = (1.0 - empty_id.to(device=device, dtype=dtype)).view(-1, 1)
        else:
            m = torch.ones((kpgt_fp.shape[0], 1), device=device, dtype=dtype)
        u = self.dose_enc(dose_cont.to(device=device, dtype=dtype)) * m
        s, b = self.film(u).chunk(2, dim=-1)

        k_mod = k * (1.0 + s) + b
        # Gate again after FiLM to ensure EMPTY samples cannot be affected by FiLM biases.
        delta = (k_mod - k) * m
        h = k + self.alpha.to(device=device, dtype=dtype) * delta

        if self.emb_empty is not None:
            h = h + self.emb_empty(empty_id.to(device=device))
        h = self.out_norm(h)

        if apply_cfg_dropout and float(self.cfg.cond_drop_prob) > 0.0 and self.training:
            drop = torch.rand((h.shape[0], 1), device=device) < float(self.cfg.cond_drop_prob)
            uncond = self.uncond_batch(h.shape[0], device=device, dtype=h.dtype)
            h = torch.where(drop, uncond, h)

        return h


def build_condition_encoder(
    cond_cfg: dict[str, Any],
    *,
    cond_drop_prob: float | None = None,
) -> tuple[nn.Module, str, str]:
    drug_mode = canonicalize_drug_mode(cond_cfg.get("drug_mode", "kpgt"))
    dose_mode = canonicalize_dose_mode(cond_cfg.get("dose_mode", "bins"))
    hidden_dim = int(cond_cfg.get("hidden_dim", 384))
    cond_drop = float(cond_cfg.get("cond_drop_prob", 0.1) if cond_drop_prob is None else cond_drop_prob)
    keep_empty_id = bool(cond_cfg.get("keep_empty_id", True))

    if drug_mode == "treatment_id":
        if dose_mode != "bins":
            raise ValueError("drug_mode=treatment_id currently only supports dose_mode=bins")
        cond = TreatmentDoseConditionEncoder(
            TreatmentDoseCondConfig(
                num_treatments=int(cond_cfg.get("num_treatments", 0)),
                hidden_dim=hidden_dim,
                dose_bins=int(cond_cfg.get("dose_bins", 9)),
                cond_drop_prob=cond_drop,
                keep_empty_id=keep_empty_id,
            )
        )
        return cond, drug_mode, dose_mode

    fp_dim = int(cond_cfg.get("fp_dim", cond_cfg.get("kpgt_dim", 2304)))
    if dose_mode == "bins":
        cond = KPGTConditionEncoder(
            KPGTCondConfig(
                kpgt_dim=fp_dim,
                hidden_dim=hidden_dim,
                dose_bins=int(cond_cfg.get("dose_bins", 9)),
                cond_drop_prob=cond_drop,
                keep_empty_id=keep_empty_id,
                kpgt_dropout=float(cond_cfg.get("kpgt_dropout", 0.1)),
            )
        )
    else:
        cond = KPGTFiLMConditionEncoder(
            KPGTFiLMCondConfig(
                kpgt_dim=fp_dim,
                hidden_dim=hidden_dim,
                cond_drop_prob=cond_drop,
                keep_empty_id=keep_empty_id,
                gate_empty_dose=bool(cond_cfg.get("gate_empty_dose", True)),
                kpgt_dropout=float(cond_cfg.get("kpgt_dropout", 0.1)),
                dose_fourier_k=int(cond_cfg.get("dose_fourier_k", 16)),
                dose_mlp_dim=int(cond_cfg.get("dose_mlp_dim", hidden_dim)),
                dose_alpha_init=float(cond_cfg.get("dose_alpha_init", 0.1)),
            )
        )
    return cond, drug_mode, dose_mode


def condition_forward(
    cond: nn.Module,
    *,
    drug_mode: str,
    dose_mode: str,
    drug_fp: torch.Tensor | None = None,
    treatment_id: torch.Tensor | None = None,
    dose_id: torch.Tensor | None = None,
    dose_cont: torch.Tensor | None = None,
    empty_id: torch.Tensor,
    apply_cfg_dropout: bool = True,
) -> torch.Tensor:
    drug_mode = canonicalize_drug_mode(drug_mode)
    dose_mode = canonicalize_dose_mode(dose_mode)

    if drug_mode == "treatment_id":
        if treatment_id is None or dose_id is None:
            raise ValueError("treatment_id and dose_id are required for drug_mode=treatment_id")
        return cond(
            treatment_id=treatment_id,
            dose_id=dose_id,
            empty_id=empty_id,
            apply_cfg_dropout=apply_cfg_dropout,
        )

    if drug_fp is None:
        raise ValueError("drug_fp is required for fingerprint-based conditioning")
    if dose_mode == "bins":
        if dose_id is None:
            raise ValueError("dose_id is required for dose_mode=bins")
        return cond(
            kpgt_fp=drug_fp,
            dose_id=dose_id,
            empty_id=empty_id,
            apply_cfg_dropout=apply_cfg_dropout,
        )
    if dose_cont is None:
        raise ValueError("dose_cont is required for dose_mode=cont_film")
    return cond(
        kpgt_fp=drug_fp,
        dose_cont=dose_cont,
        empty_id=empty_id,
        apply_cfg_dropout=apply_cfg_dropout,
    )
