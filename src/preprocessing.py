"""Data preprocessing utilities for PINN/MLP heat equation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Dict

import numpy as np
from scipy.stats.qmc import LatinHypercube


@dataclass
class DomainConfig:
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0
    t_cold: float = 20.0
    t_hot: float = 60.0


def normalize(values: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    """Min-max normalization into [0, 1]."""
    values = np.asarray(values, dtype=np.float64)
    return (values - min_value) / (max_value - min_value)


def denormalize(values: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    """Inverse min-max transform."""
    values = np.asarray(values, dtype=np.float64)
    return values * (max_value - min_value) + min_value


def sample_domain_points(
    n_points: int,
    cfg: DomainConfig,
    seed: int = 42,
) -> np.ndarray:
    """Latin Hypercube sampling for interior collocation points."""
    sampler = LatinHypercube(d=3, seed=seed)
    pts = sampler.random(n_points)
    x = cfg.x_min + (cfg.x_max - cfg.x_min) * pts[:, 0]
    y = cfg.y_min + (cfg.y_max - cfg.y_min) * pts[:, 1]
    t = cfg.t_min + (cfg.t_max - cfg.t_min) * pts[:, 2]
    return np.column_stack([x, y, t])


def sample_initial_points(
    n_points: int,
    cfg: DomainConfig,
    seed: int = 42,
) -> np.ndarray:
    """Sample initial condition points (t = t_min)."""
    sampler = LatinHypercube(d=2, seed=seed)
    pts = sampler.random(n_points)
    x = cfg.x_min + (cfg.x_max - cfg.x_min) * pts[:, 0]
    y = cfg.y_min + (cfg.y_max - cfg.y_min) * pts[:, 1]
    t = np.full_like(x, cfg.t_min)
    return np.column_stack([x, y, t])


def sample_boundary_points(
    n_points_per_side: int,
    cfg: DomainConfig,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate boundary samples for all four sides of the 2D plate."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(cfg.t_min, cfg.t_max, size=n_points_per_side)
    x = rng.uniform(cfg.x_min, cfg.x_max, size=n_points_per_side)
    y = rng.uniform(cfg.y_min, cfg.y_max, size=n_points_per_side)

    left = np.column_stack([np.full(n_points_per_side, cfg.x_min), y, t])
    right = np.column_stack([np.full(n_points_per_side, cfg.x_max), y, t])
    bottom = np.column_stack([x, np.full(n_points_per_side, cfg.y_min), t])
    top = np.column_stack([x, np.full(n_points_per_side, cfg.y_max), t])

    return {"left": left, "right": right, "bottom": bottom, "top": top}


def solve_fdm_steady(
    nx: int = 51,
    ny: int = 51,
    max_iter: int = 20_000,
    tolerance: float = 1e-6,
    omega: float = 1.6,
    cfg: DomainConfig | None = None,
) -> np.ndarray:
    """
    Solve steady-state 2D heat equation with Dirichlet BC using SOR:
    d2T/dx2 + d2T/dy2 = 0.
    BC: top/left/right = T_cold, bottom = T_hot.
    """
    cfg = cfg or DomainConfig()
    temp = np.full((ny, nx), cfg.t_cold, dtype=np.float64)
    temp[-1, :] = cfg.t_hot

    for _ in range(max_iter):
        max_delta = 0.0
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                old = temp[j, i]
                avg = 0.25 * (
                    temp[j + 1, i]
                    + temp[j - 1, i]
                    + temp[j, i + 1]
                    + temp[j, i - 1]
                )
                new = (1.0 - omega) * old + omega * avg
                delta = abs(new - old)
                if delta > max_delta:
                    max_delta = delta
                temp[j, i] = new
        if max_delta < tolerance:
            break

    return temp


def build_processed_dataset(
    output_dir: str | Path = "data/processed",
    n_collocation: int = 25_000,
    n_initial: int = 2_000,
    n_boundary_per_side: int = 2_000,
    fdm_nx: int = 51,
    fdm_ny: int = 51,
    seed: int = 42,
) -> Path:
    """
    Create and save processed data:
    - collocation points
    - IC/BC points
    - FDM ground truth field
    - normalization metadata
    """
    cfg = DomainConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    collocation = sample_domain_points(n_collocation, cfg, seed=seed)
    initial = sample_initial_points(n_initial, cfg, seed=seed + 1)
    boundaries = sample_boundary_points(n_boundary_per_side, cfg, seed=seed + 2)
    fdm = solve_fdm_steady(nx=fdm_nx, ny=fdm_ny, cfg=cfg)

    x_grid = np.linspace(cfg.x_min, cfg.x_max, fdm_nx)
    y_grid = np.linspace(cfg.y_min, cfg.y_max, fdm_ny)
    xx, yy = np.meshgrid(x_grid, y_grid)

    # Normalize coordinates and temperature into [0, 1].
    collocation_norm = np.column_stack(
        [
            normalize(collocation[:, 0], cfg.x_min, cfg.x_max),
            normalize(collocation[:, 1], cfg.y_min, cfg.y_max),
            normalize(collocation[:, 2], cfg.t_min, cfg.t_max),
        ]
    )
    initial_norm = np.column_stack(
        [
            normalize(initial[:, 0], cfg.x_min, cfg.x_max),
            normalize(initial[:, 1], cfg.y_min, cfg.y_max),
            normalize(initial[:, 2], cfg.t_min, cfg.t_max),
        ]
    )
    boundaries_norm = {
        k: np.column_stack(
            [
                normalize(v[:, 0], cfg.x_min, cfg.x_max),
                normalize(v[:, 1], cfg.y_min, cfg.y_max),
                normalize(v[:, 2], cfg.t_min, cfg.t_max),
            ]
        )
        for k, v in boundaries.items()
    }
    fdm_norm = normalize(fdm, cfg.t_cold, cfg.t_hot)

    npz_path = output_dir / "pinn_dataset.npz"
    np.savez_compressed(
        npz_path,
        collocation=collocation,
        collocation_norm=collocation_norm,
        initial=initial,
        initial_norm=initial_norm,
        boundary_left=boundaries["left"],
        boundary_right=boundaries["right"],
        boundary_top=boundaries["top"],
        boundary_bottom=boundaries["bottom"],
        boundary_left_norm=boundaries_norm["left"],
        boundary_right_norm=boundaries_norm["right"],
        boundary_top_norm=boundaries_norm["top"],
        boundary_bottom_norm=boundaries_norm["bottom"],
        fdm_field=fdm,
        fdm_field_norm=fdm_norm,
        fdm_xx=xx,
        fdm_yy=yy,
    )

    meta = {
        "normalization": {
            "x": [cfg.x_min, cfg.x_max],
            "y": [cfg.y_min, cfg.y_max],
            "t": [cfg.t_min, cfg.t_max],
            "temperature": [cfg.t_cold, cfg.t_hot],
        },
        "counts": {
            "collocation": n_collocation,
            "initial": n_initial,
            "boundary_per_side": n_boundary_per_side,
        },
        "fdm_grid": {"nx": fdm_nx, "ny": fdm_ny},
    }
    (output_dir / "pinn_dataset_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    return npz_path


if __name__ == "__main__":
    path = build_processed_dataset()
    print(f"Saved processed dataset to: {path}")
