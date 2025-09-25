# src/evaluate.py
"""Evaluation helpers – metrics, statistical aggregation and plotting.
Anything that **reads** model outputs without changing parameters
belongs here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional.audio import frechet_audio_distance
from torchmetrics.image.kid import KernelInceptionDistance  # noqa – may be used later

sns.set_style("whitegrid")


# ---------------------------------------------------------------------
# Metrics wrappers
# ---------------------------------------------------------------------
class FID(torch.nn.Module):
    """Simple wrapper around torchmetrics' Frechet Inception Distance."""

    def __init__(self, **kwargs):
        super().__init__()
        self._fid = FrechetInceptionDistance(**kwargs)

    @torch.no_grad()
    def update(self, real: torch.Tensor, fake: torch.Tensor):
        self._fid.update(real, real=True)
        self._fid.update(fake, real=False)

    def compute(self) -> float:
        return float(self._fid.compute())


class AudioFD(torch.nn.Module):
    """Frechet Audio Distance aggregation helper."""

    def __init__(self):
        super().__init__()
        self._vals: List[float] = []

    @torch.no_grad()
    def update(self, real: torch.Tensor, fake: torch.Tensor):
        self._vals.append(float(frechet_audio_distance(real, fake)))

    def compute(self) -> float:
        return float(torch.tensor(self._vals).mean())


# ---------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------

def line_plot(
    xs: List[float],
    ys_dict: Dict[str, List[float]],
    title: str,
    xlabel: str,
    ylabel: str,
    fname: Path | str,
):
    """Draw a labelled line plot and save as a **tight** PDF."""
    plt.figure(figsize=(6, 4))
    for label, ys in ys_dict.items():
        plt.plot(xs, ys, marker="o", label=label)
        for x, y in zip(xs, ys):
            plt.text(x, y, f"{y:.3f}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
