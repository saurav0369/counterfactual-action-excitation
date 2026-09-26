from dataclasses import dataclass
from fractions import Fraction
import numpy as np

from .dynamics import conditional_mean, sample_states
from .policies import matched_covariance_actions


@dataclass(frozen=True)
class TransitionDataset:
    states: np.ndarray
    actions: np.ndarray
    targets_noisy: np.ndarray
    targets_mean: np.ndarray

    @property
    def x(self) -> np.ndarray:
        return np.column_stack([self.states, self.actions])


def make_training_dataset(
    n: int,
    epsilon: Fraction,
    *,
    state_seed: int,
    action_seed: int,
    noise_seed: int,
    noise_std: float = 0.1,
    alpha: float = 0.8,
    beta: float = 1.0,
) -> TransitionDataset:
    states = sample_states(n, state_seed)
    actions = matched_covariance_actions(n, epsilon, action_seed)
    mean = conditional_mean(states, actions, alpha=alpha, beta=beta)
    rng = np.random.default_rng(noise_seed)
    noisy = mean + rng.normal(0.0, noise_std, size=n)
    return TransitionDataset(states, actions, noisy, mean)


def make_factual_eval_dataset(
    n: int,
    epsilon: Fraction,
    *,
    state_seed: int,
    action_seed: int,
    alpha: float = 0.8,
    beta: float = 1.0,
) -> TransitionDataset:
    states = sample_states(n, state_seed)
    actions = matched_covariance_actions(n, epsilon, action_seed)
    mean = conditional_mean(states, actions, alpha=alpha, beta=beta)
    return TransitionDataset(states, actions, mean.copy(), mean)


def make_counterfactual_eval_dataset(
    n: int,
    *,
    state_seed: int,
    action_seed: int,
    alpha: float = 0.8,
    beta: float = 1.0,
) -> TransitionDataset:
    # epsilon=1 is the exactly balanced four-corner intervention distribution.
    return make_factual_eval_dataset(
        n,
        Fraction(1, 1),
        state_seed=state_seed,
        action_seed=action_seed,
        alpha=alpha,
        beta=beta,
    )
