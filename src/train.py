# src/train.py
"""Model definition and training-specific utilities.
All model-level code from the original monolithic script is
collected here so that anything dealing with forward passes,
weight freezing, rank prediction, … lives in **one** place.  No
experiment-orchestration logic is allowed inside this file – that is
handled by src.main.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline


class MLPRankPredictor(nn.Module):
    """Four-layer MLP producing Tucker ranks (r_s,r_c,r_t,r_m).
    Output is Softplus-clamped to the interval [4,64] as required by
    SAFE-HASTE.  The layer dimensions are kept identical to the paper
    to guarantee equivalence with the reference implementation.
    """

    def __init__(self, in_dim: int = 128, hidden: List[int] | None = None):
        super().__init__()
        if hidden is None:
            hidden = [256, 64, 16]
        dims = [in_dim] + hidden + [4]
        layers: List[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.SiLU()]
        layers.pop()  # remove activation of the output layer
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,in_dim)
        r = self.softplus(self.net(x)) + 4.0  # lower-bound
        return torch.clamp(r, max=64.0)


class AVSDBase(nn.Module):
    """Audio-Visual Stable-Diffusion (AVSD-base 12-step).

    The backbone weights are loaded from
    ``stabilityai/stable-diffusion-xl-base-1.0`` and all parameters
    except the lightweight audio & depth adapters are **frozen**.  A
    rank predictor can be attached via ``enable_tucker``; the internal
    UNet of the diffusers pipeline is then monkey-patched with
    ``set_tucker_ranks`` at run-time.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()

        # ------------------------------------------------------------------
        # Mixed-precision only when CUDA is available.  On CPU use fp32
        # because many kernels are not implemented for fp16.
        # ------------------------------------------------------------------
        dtype = torch.float16 if device.startswith("cuda") else torch.float32

        # NOTE: from_pretrained can throw if models are not cached and the
        # host has no internet connection.  We purposefully *do not*
        # silence this error – fail-fast is preferable to silently running
        # with an uninitialised model.
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=dtype,
        ).to(device)

        # Freeze UNet – we only fine-tune adapters / rank predictor
        for p in self.pipe.unet.parameters():
            p.requires_grad_(False)

        # 1×1 convolution adapters for audio/depth modalities
        self.audio_adapter = nn.Conv2d(128, 128, 1)
        self.depth_adapter = nn.Conv2d(128, 128, 1)
        self.rank_predictor: MLPRankPredictor | None = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def enable_tucker(self, rank_predictor: MLPRankPredictor):
        """Attach a Tucker-rank predictor (SAFE-HASTE)."""
        self.rank_predictor = rank_predictor

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        rgb_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        depth_latent: torch.Tensor,
        prompt_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one 12-step AVSD forward pass.

        Args
        ----
        rgb_latent : latent feature map from RGB frames
        audio_latent : 128×T mel spectrogram encoded features
        depth_latent : latent from the depth adapter
        prompt_emb : optional text embedding (unused here)
        """
        x = (
            rgb_latent
            + self.audio_adapter(audio_latent)
            + self.depth_adapter(depth_latent)
        )

        # Adaptive Tucker sparsity if requested
        if self.rank_predictor is not None:
            ranks = self.rank_predictor(x.mean(dim=(2, 3)))  # (B,4)
            # ``set_tucker_ranks`` is provided by diffusers ≥0.20.0
            self.pipe.unet.set_tucker_ranks(tuple(ranks.int().tolist()))

        # Diffusers handles the latent scheduler internally
        return self.pipe(prompt_emb, latents=x, num_inference_steps=12).images