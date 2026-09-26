import numpy as np


def conditional_mean(states: np.ndarray, actions: np.ndarray, alpha: float = 0.8, beta: float = 1.0) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64).reshape(-1)
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError("actions must have shape [N, 2]")
    if len(states) != len(actions):
        raise ValueError("states and actions must have the same length")
    return alpha * states + beta * actions[:, 0] * actions[:, 1]


def sample_transitions(states: np.ndarray, actions: np.ndarray, noise_std: float, seed: int, alpha: float = 0.8, beta: float = 1.0):
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    mu = conditional_mean(states, actions, alpha=alpha, beta=beta)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_std, size=len(mu))
    return mu + noise, mu


def sample_states(n: int, seed: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive")
    return np.random.default_rng(seed).normal(0.0, 1.0, size=n)
