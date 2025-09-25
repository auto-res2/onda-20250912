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

# ---------------------------------------------------------------------
# TorchMetrics – robust Frechet Audio Distance import
# ---------------------------------------------------------------------
# The functional helper `frechet_audio_distance` moved across several
# TorchMetrics releases.  To stay version-agnostic we *first* try the
# canonical import; if that fails we fall back to a minimal re-
# implementation that wraps the underlying metric class.
# ---------------------------------------------------------------------
try:
    # Newer TorchMetrics versions (⩾0.11) expose the function here
    from torchmetrics.functional.audio import frechet_audio_distance as _frechet_audio_distance
except (ImportError, ModuleNotFoundError):

    # Older releases – construct the metric manually each call.
    def _frechet_audio_distance(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Compute Frechet Audio Distance for a pair of batches.

        We instantiate `torchmetrics.audio.FrechetAudioDistance` on the
        fly to avoid a hard dependency on the *functional* helper that
        may be missing in the installed TorchMetrics version.
        """
        from torchmetrics.audio import FrechetAudioDistance  # local import keeps import cost negligible

        fad_metric = FrechetAudioDistance()
        fad_metric.update(real, real=True)
        fad_metric.update(fake, real=False)
        return fad_metric.compute()

# Expose a *single* public symbol – this avoids the "Name already
# defined" static-analysis error that arises when the same name is
# rebound in different branches.
frechet_audio_distance = _frechet_audio_distance

# Optional KID import retained for future use (plotting, reports, …)
from torchmetrics.image.kid import KernelInceptionDistance  # noqa: F401 – used in extended analyses

sns.set_style("whitegrid")


# ---------------------------------------------------------------------
# Metrics wrappers
# ---------------------------------------------------------------------
class FID(torch.nn.Module):
    """Simple wrapper around TorchMetrics' Frechet-Inception-Distance."""

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
        # Each call returns a scalar tensor; converting to float keeps
        # the memory footprint negligible while preserving precision.
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
    """Draw a labelled line plot and save as a *tight* PDF figure."""
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
