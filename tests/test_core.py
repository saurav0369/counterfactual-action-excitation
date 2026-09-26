from fractions import Fraction
import numpy as np

from src.policies import matched_covariance_actions, empirical_moments
from src.dynamics import conditional_mean, sample_transitions
from src.metrics import mse, safe_log_ratio, paired_bootstrap_mean_ci


def test_exact_low_excitation_moments():
    eps = Fraction(1, 64)
    a = matched_covariance_actions(4096, eps, seed=123)
    m = empirical_moments(a)
    np.testing.assert_allclose(m["mean"], np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(m["cov"], np.eye(2), atol=1e-12)
    np.testing.assert_allclose(m["phi_mean"], 0.0, atol=1e-12)
    np.testing.assert_allclose(m["phi_var"], float(eps), atol=1e-12)


def test_exact_high_excitation_moments():
    a = matched_covariance_actions(4096, Fraction(1, 1), seed=456)
    m = empirical_moments(a)
    np.testing.assert_allclose(m["mean"], np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(m["cov"], np.eye(2), atol=1e-12)
    np.testing.assert_allclose(m["phi_mean"], 0.0, atol=1e-12)
    np.testing.assert_allclose(m["phi_var"], 1.0, atol=1e-12)


def test_invalid_unbalanced_sample_rejected():
    try:
        matched_covariance_actions(1000, Fraction(1, 64), seed=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected invalid balanced-support size to be rejected")


def test_noiseless_conditional_mean_is_evaluation_target():
    s = np.array([0.0, 1.0, -1.0])
    a = np.array([[1.0, 1.0], [1.0, -1.0], [np.sqrt(2.0), 0.0]])
    mu = conditional_mean(s, a)
    expected = np.array([1.0, -0.2, -0.8])
    np.testing.assert_allclose(mu, expected, atol=1e-12)
    y, mu2 = sample_transitions(s, a, noise_std=0.1, seed=7)
    np.testing.assert_allclose(mu2, mu, atol=1e-12)
    assert not np.allclose(y, mu)


def test_metrics_smoke():
    assert mse(np.array([1., 2.]), np.array([1., 4.])) == 2.0
    assert np.isclose(safe_log_ratio(2.0, 1.0), np.log(2.0))
    vals = np.array([0.1, 0.2, 0.3, 0.4])
    mean, lo, hi = paired_bootstrap_mean_ci(vals, n_boot=1000, seed=11)
    assert lo <= mean <= hi


def test_counterfactual_eval_is_balanced_corner_distribution():
    from src.data import make_counterfactual_eval_dataset
    from src.policies import empirical_moments
    ds = make_counterfactual_eval_dataset(16384, state_seed=9, action_seed=10)
    m = empirical_moments(ds.actions)
    np.testing.assert_allclose(m["mean"], np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(m["cov"], np.eye(2), atol=1e-12)
    np.testing.assert_allclose(m["phi_var"], 1.0, atol=1e-12)


def test_low_and_high_share_exact_raw_covariance():
    low = matched_covariance_actions(4096, Fraction(1, 64), seed=1)
    high = matched_covariance_actions(4096, Fraction(1, 1), seed=2)
    ml = empirical_moments(low)
    mh = empirical_moments(high)
    np.testing.assert_allclose(ml["mean"], mh["mean"], atol=1e-12)
    np.testing.assert_allclose(ml["cov"], mh["cov"], atol=1e-12)
    assert np.isclose(mh["phi_var"] / ml["phi_var"], 64.0)
