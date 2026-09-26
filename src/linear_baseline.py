import numpy as np


def interaction_features(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64).reshape(-1)
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError("actions must have shape [N, 2]")
    phi = actions[:, 0] * actions[:, 1]
    return np.column_stack([np.ones(len(states)), states, phi])


def fit_interaction_ols(states: np.ndarray, actions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    x = interaction_features(states, actions)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return coef


def predict_interaction_ols(coef: np.ndarray, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return interaction_features(states, actions) @ np.asarray(coef, dtype=np.float64)
