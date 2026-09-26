from fractions import Fraction
import numpy as np

CORNERS = np.array([
    [1.0, 1.0],
    [1.0, -1.0],
    [-1.0, 1.0],
    [-1.0, -1.0],
], dtype=np.float64)

_s2 = np.sqrt(2.0)
AXES = np.array([
    [_s2, 0.0],
    [-_s2, 0.0],
    [0.0, _s2],
    [0.0, -_s2],
], dtype=np.float64)


def matched_covariance_actions(n: int, epsilon: Fraction, seed: int) -> np.ndarray:
    """Balanced finite-sample construction.

    Returns actions with empirical E[a]=0 and population-form empirical Cov(a)=I
    (up to floating-point precision), while Var(a1*a2)=epsilon.

    Requirements:
      * 0 <= epsilon <= 1
      * n*epsilon and n*(1-epsilon) are integers divisible by 4
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if epsilon < 0 or epsilon > 1:
        raise ValueError("epsilon must lie in [0, 1]")

    num = n * epsilon.numerator
    if num % epsilon.denominator != 0:
        raise ValueError("n * epsilon must be an integer")
    n_corner = num // epsilon.denominator
    n_axis = n - n_corner

    if n_corner % 4 != 0:
        raise ValueError("corner sample count must be divisible by 4")
    if n_axis % 4 != 0:
        raise ValueError("axis sample count must be divisible by 4")

    blocks = []
    if n_corner:
        blocks.append(np.repeat(CORNERS, n_corner // 4, axis=0))
    if n_axis:
        blocks.append(np.repeat(AXES, n_axis // 4, axis=0))
    actions = np.concatenate(blocks, axis=0)

    rng = np.random.default_rng(seed)
    return actions[rng.permutation(n)]


def nonlinear_feature(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError("actions must have shape [N, 2]")
    return actions[:, 0] * actions[:, 1]


def empirical_moments(actions: np.ndarray) -> dict:
    actions = np.asarray(actions, dtype=np.float64)
    mean = actions.mean(axis=0)
    centered = actions - mean
    cov = centered.T @ centered / len(actions)
    phi = nonlinear_feature(actions)
    return {
        "mean": mean,
        "cov": cov,
        "phi_mean": float(phi.mean()),
        "phi_var": float(phi.var()),
    }
