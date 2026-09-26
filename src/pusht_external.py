from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import random

import numpy as np
import torch
from torch import nn

from .model import MLPConfig
from .policies import matched_covariance_actions


def matched_local_offsets(
    n: int,
    epsilon: Fraction,
    *,
    scale: float,
    seed: int,
    block_size: int = 64,
) -> np.ndarray:
    """Create blockwise-balanced local PushT action offsets.

    Each block has exactly zero mean and covariance ``scale**2 * I`` while
    Var(dx * dy) = epsilon * scale**4.  The blockwise construction reduces
    accidental state-action confounding when each offset is paired with a
    different anchor state.
    """
    if n % block_size != 0:
        raise ValueError("n must be divisible by block_size")
    blocks = []
    for b in range(n // block_size):
        # Base construction has Cov(a)=I. Multiplying by scale preserves the
        # exact matched-moment construction at the desired physical magnitude.
        block = matched_covariance_actions(
            block_size,
            epsilon,
            seed=seed + 104729 * b,
        )
        blocks.append(block * float(scale))
    return np.concatenate(blocks, axis=0)


def offset_moments(offsets: np.ndarray) -> dict[str, np.ndarray | float]:
    x = np.asarray(offsets, dtype=np.float64)
    mean = x.mean(axis=0)
    centered = x - mean
    cov = centered.T @ centered / len(x)
    phi = x[:, 0] * x[:, 1]
    return {
        "mean": mean,
        "cov": cov,
        "phi_mean": float(phi.mean()),
        "phi_var": float(phi.var()),
    }


def anchor_passes_geometry(
    obs: np.ndarray,
    *,
    agent_margin: float = 80.0,
    min_agent_block_distance: float = 45.0,
    max_agent_block_distance: float = 160.0,
) -> bool:
    """Pre-action anchor filter defined only from reset geometry.

    This deliberately does not inspect any transition outcome or contact count.
    It keeps target actions in bounds and focuses evaluation on states where
    block interaction is geometrically plausible.
    """
    obs = np.asarray(obs, dtype=np.float64)
    if obs.shape != (5,):
        raise ValueError("PushT state observation must have shape (5,)")
    ax, ay, bx, by, _ = obs
    if not (agent_margin <= ax <= 512.0 - agent_margin):
        return False
    if not (agent_margin <= ay <= 512.0 - agent_margin):
        return False
    dist = float(np.hypot(ax - bx, ay - by))
    return min_agent_block_distance <= dist <= max_agent_block_distance


def wrap_angle(x: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


def encode_input(obs: np.ndarray, offset: np.ndarray, *, offset_scale: float) -> np.ndarray:
    """Encode one PushT state plus local target-position offset."""
    obs = np.asarray(obs, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    theta = float(obs[4])
    denom = np.sqrt(2.0) * float(offset_scale)
    return np.array(
        [
            obs[0] / 512.0,
            obs[1] / 512.0,
            obs[2] / 512.0,
            obs[3] / 512.0,
            np.sin(theta),
            np.cos(theta),
            offset[0] / denom,
            offset[1] / denom,
        ],
        dtype=np.float64,
    )


def encode_block_transition(obs: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
    """Decision-relevant external dynamics target: normalized block motion."""
    obs = np.asarray(obs, dtype=np.float64)
    next_obs = np.asarray(next_obs, dtype=np.float64)
    dtheta = float(wrap_angle(next_obs[4] - obs[4]))
    return np.array(
        [
            (next_obs[2] - obs[2]) / 512.0,
            (next_obs[3] - obs[3]) / 512.0,
            dtheta / (2.0 * np.pi),
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class PushTTransitionDataset:
    x: np.ndarray
    y: np.ndarray
    anchor_seeds: np.ndarray
    offsets: np.ndarray


class VectorTransitionMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int, depth: int):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(dim, width), nn.ReLU()])
            dim = width
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def train_vector_mlp(
    x: np.ndarray,
    y: np.ndarray,
    config: MLPConfig,
    *,
    seed: int,
) -> VectorTransitionMLP:
    _seed_everything(seed)
    x_t = torch.as_tensor(np.asarray(x, dtype=np.float32))
    y_t = torch.as_tensor(np.asarray(y, dtype=np.float32))
    if x_t.ndim != 2 or y_t.ndim != 2 or len(x_t) != len(y_t):
        raise ValueError("x and y must be two-dimensional with equal row counts")

    model = VectorTransitionMLP(
        input_dim=x_t.shape[1],
        output_dim=y_t.shape[1],
        width=config.width,
        depth=config.depth,
    )
    opt = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 1_000_003)
    n = len(x_t)

    model.train()
    for _ in range(config.steps):
        idx = torch.randint(0, n, (config.batch_size,), generator=g)
        loss = loss_fn(model(x_t[idx]), y_t[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


def predict_vector(model: VectorTransitionMLP, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model(torch.as_tensor(np.asarray(x, dtype=np.float32)))
    return out.cpu().numpy().astype(np.float64)


def vector_mse(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.mean((pred - target) ** 2))
