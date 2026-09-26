import numpy as np


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if pred.shape != target.shape:
        raise ValueError("pred and target must have identical shapes")
    return float(np.mean((pred - target) ** 2))


def safe_log_ratio(numerator: float, denominator: float, eps: float = 1e-15) -> float:
    if numerator < 0 or denominator < 0:
        raise ValueError("errors must be non-negative")
    return float(np.log(max(numerator, eps) / max(denominator, eps)))


def paired_bootstrap_mean_ci(values: np.ndarray, *, n_boot: int = 10_000, ci: float = 0.95, seed: int = 314159):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) < 2:
        raise ValueError("at least two paired values are required")
    if not (0.0 < ci < 1.0):
        raise ValueError("ci must be in (0, 1)")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(values.mean()), float(lo), float(hi)
