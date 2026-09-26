import numpy as np
from fractions import Fraction

from src.data import make_training_dataset
from src.linear_baseline import interaction_features, fit_interaction_ols, predict_interaction_ols


def test_interaction_features_shape_and_values():
    s = np.array([0.0, 1.0])
    a = np.array([[1.0, -1.0], [np.sqrt(2.0), 0.0]])
    x = interaction_features(s, a)
    assert x.shape == (2, 3)
    np.testing.assert_allclose(x[:, 0], 1.0)
    np.testing.assert_allclose(x[:, 1], s)
    np.testing.assert_allclose(x[:, 2], [-1.0, 0.0])


def test_ols_recovers_noiseless_coefficients():
    d = make_training_dataset(
        4096, Fraction(1, 1), state_seed=91, action_seed=92, noise_seed=93,
        noise_std=0.0, alpha=0.8, beta=1.25)
    coef = fit_interaction_ols(d.states, d.actions, d.targets_noisy)
    np.testing.assert_allclose(coef, [0.0, 0.8, 1.25], atol=1e-10)
    pred = predict_interaction_ols(coef, d.states, d.actions)
    np.testing.assert_allclose(pred, d.targets_mean, atol=1e-10)
