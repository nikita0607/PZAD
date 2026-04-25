"""Visualization helpers for PINN/MLP heat experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap(
    field: np.ndarray,
    title: str = "Temperature Field",
    cmap: str = "inferno",
    save_path: str | Path | None = None,
) -> None:
    plt.figure(figsize=(6, 5))
    img = plt.imshow(field, origin="lower", cmap=cmap, aspect="auto")
    plt.title(title)
    plt.xlabel("x-index")
    plt.ylabel("y-index")
    plt.colorbar(img, label="Temperature")
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_comparison(
    fdm_field: np.ndarray,
    pred_field: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    err = np.abs(fdm_field - pred_field)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    a0 = axes[0].imshow(fdm_field, origin="lower", cmap="inferno", aspect="auto")
    axes[0].set_title("FDM (Ground Truth)")
    plt.colorbar(a0, ax=axes[0], fraction=0.046, pad=0.04)

    a1 = axes[1].imshow(pred_field, origin="lower", cmap="inferno", aspect="auto")
    axes[1].set_title("Model Prediction")
    plt.colorbar(a1, ax=axes[1], fraction=0.046, pad=0.04)

    a2 = axes[2].imshow(err, origin="lower", cmap="magma", aspect="auto")
    axes[2].set_title("|Error|")
    plt.colorbar(a2, ax=axes[2], fraction=0.046, pad=0.04)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_loss_curves(
    curves: Sequence[Iterable[float]],
    labels: Sequence[str],
    title: str = "Training Curves",
    save_path: str | Path | None = None,
) -> None:
    plt.figure(figsize=(8, 5))
    for curve, label in zip(curves, labels):
        plt.plot(list(curve), label=label)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()
