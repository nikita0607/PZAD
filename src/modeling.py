"""Modeling utilities: baseline MLP, PINN model, losses, and training loops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_hidden_layers: int = 4,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        activation = activation or nn.Tanh()

        layers: List[nn.Module] = [nn.Linear(input_dim, hidden_dim), activation]
        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), activation])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        return self.net(xyt)


class PINN(MLP):
    """Same backbone as MLP; optimization objective differs."""


@dataclass
class LossWeights:
    lambda_pde: float = 1.0
    lambda_bc: float = 1_000.0
    lambda_ic: float = 100.0
    lambda_data: float = 0.0


def _grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def pde_residual(
    model: nn.Module,
    xyt: torch.Tensor,
    alpha: float = 0.05,
) -> torch.Tensor:
    xyt = xyt.requires_grad_(True)
    u = model(xyt)
    grads = _grad(u, xyt)
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]
    u_t = grads[:, 2:3]

    u_xx = _grad(u_x, xyt)[:, 0:1]
    u_yy = _grad(u_y, xyt)[:, 1:2]
    return u_t - alpha * (u_xx + u_yy)


def pinn_loss(
    model: nn.Module,
    collocation: torch.Tensor,
    bc: Dict[str, torch.Tensor],
    ic: torch.Tensor,
    weights: LossWeights,
    alpha: float = 0.05,
) -> Dict[str, torch.Tensor]:
    mse = nn.MSELoss()
    residual = pde_residual(model, collocation, alpha=alpha)
    loss_pde = mse(residual, torch.zeros_like(residual))

    # Dirichlet boundaries in normalized temperature [0, 1]:
    # top/left/right -> 0.0, bottom -> 1.0
    pred_left = model(bc["left"])
    pred_right = model(bc["right"])
    pred_top = model(bc["top"])
    pred_bottom = model(bc["bottom"])

    loss_bc = (
        mse(pred_left, torch.zeros_like(pred_left))
        + mse(pred_right, torch.zeros_like(pred_right))
        + mse(pred_top, torch.zeros_like(pred_top))
        + mse(pred_bottom, torch.ones_like(pred_bottom))
    )

    # Initial condition: normalized temperature starts from 0.0
    pred_ic = model(ic)
    loss_ic = mse(pred_ic, torch.zeros_like(pred_ic))

    total = (
        weights.lambda_pde * loss_pde
        + weights.lambda_bc * loss_bc
        + weights.lambda_ic * loss_ic
    )
    return {"total": total, "pde": loss_pde, "bc": loss_bc, "ic": loss_ic}


def mlp_supervised_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return nn.MSELoss()(model(inputs), targets)


def train_model(
    model: nn.Module,
    loss_fn: Callable[[], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    epochs: int = 2_000,
    log_every: int = 100,
) -> List[float]:
    history: List[float] = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu().item()))
        if epoch % log_every == 0:
            print(f"[{epoch:5d}] loss={history[-1]:.6e}")
    return history


def run_ablation_study(
    train_once: Callable[[Dict[str, float]], float],
    configs: List[Dict[str, float]],
) -> List[Dict[str, float]]:
    """
    Generic helper for ablation experiments.
    `train_once` should return validation MSE for each config.
    """
    results: List[Dict[str, float]] = []
    for cfg in configs:
        val_mse = float(train_once(cfg))
        row = dict(cfg)
        row["val_mse"] = val_mse
        results.append(row)
    return sorted(results, key=lambda x: x["val_mse"])


def save_model_weights(model: nn.Module, path: str | Path) -> Path:
    """Save model state_dict to disk."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target)
    return target


def load_model_weights(
    model: nn.Module,
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> nn.Module:
    """Load model state_dict from disk into an existing model instance."""
    state = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(state)
    model.eval()
    return model
