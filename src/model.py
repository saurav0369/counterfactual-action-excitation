from dataclasses import dataclass
import random
import numpy as np
import torch
from torch import nn

# Small CPU MLPs are faster and reproducible with one intra-op thread.
torch.set_num_threads(1)


@dataclass(frozen=True)
class MLPConfig:
    width: int
    depth: int
    learning_rate: float
    weight_decay: float = 0.0
    batch_size: int = 256
    steps: int = 1000


class TransitionMLP(nn.Module):
    def __init__(self, width: int, depth: int):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers = []
        in_dim = 3
        for _ in range(depth):
            layers.extend([nn.Linear(in_dim, width), nn.ReLU()])
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_mlp(x: np.ndarray, y: np.ndarray, config: MLPConfig, *, seed: int) -> TransitionMLP:
    _seed_everything(seed)
    x_t = torch.as_tensor(np.asarray(x, dtype=np.float32))
    y_t = torch.as_tensor(np.asarray(y, dtype=np.float32).reshape(-1))
    if len(x_t) != len(y_t):
        raise ValueError("x and y must have the same number of rows")

    model = TransitionMLP(config.width, config.depth)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()

    # Dedicated CPU generator makes minibatch order deterministic without
    # coupling it to parameter initialization or data generation.
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 1_000_003)

    n = len(x_t)
    model.train()
    for _ in range(config.steps):
        idx = torch.randint(0, n, (config.batch_size,), generator=g)
        pred = model(x_t[idx])
        loss = loss_fn(pred, y_t[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return model


def predict(model: TransitionMLP, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model(torch.as_tensor(np.asarray(x, dtype=np.float32)))
    return out.cpu().numpy().astype(np.float64)
