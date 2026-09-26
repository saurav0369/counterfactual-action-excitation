from fractions import Fraction

import numpy as np

from src.pusht_external import (
    anchor_passes_geometry,
    encode_block_transition,
    encode_input,
    matched_local_offsets,
    offset_moments,
)


def test_external_low_high_offsets_match_raw_moments_blockwise():
    scale = 40.0
    low = matched_local_offsets(64, Fraction(1, 16), scale=scale, seed=1)
    high = matched_local_offsets(64, Fraction(1, 1), scale=scale, seed=2)
    ml = offset_moments(low)
    mh = offset_moments(high)
    expected_cov = scale ** 2 * np.eye(2)
    np.testing.assert_allclose(ml["mean"], 0.0, atol=1e-12)
    np.testing.assert_allclose(mh["mean"], 0.0, atol=1e-12)
    np.testing.assert_allclose(ml["cov"], expected_cov, atol=1e-10)
    np.testing.assert_allclose(mh["cov"], expected_cov, atol=1e-10)
    assert np.isclose(mh["phi_var"] / ml["phi_var"], 16.0)


def test_external_multi_block_schedule_keeps_moments_in_each_block():
    low = matched_local_offsets(256, Fraction(1, 16), scale=40.0, seed=3)
    for start in range(0, 256, 64):
        m = offset_moments(low[start:start + 64])
        np.testing.assert_allclose(m["mean"], 0.0, atol=1e-12)
        np.testing.assert_allclose(m["cov"], 1600.0 * np.eye(2), atol=1e-10)


def test_anchor_filter_is_geometry_only():
    assert anchor_passes_geometry(np.array([256.0, 256.0, 320.0, 256.0, 0.5]))
    assert not anchor_passes_geometry(np.array([50.0, 256.0, 150.0, 256.0, 0.5]))
    assert not anchor_passes_geometry(np.array([256.0, 256.0, 500.0, 500.0, 0.5]))


def test_external_encoding_shapes_and_angle_wrap():
    s = np.array([256.0, 256.0, 300.0, 300.0, 2 * np.pi - 0.02])
    ns = np.array([260.0, 250.0, 304.0, 297.0, 0.01])
    off = np.array([40.0, -40.0])
    x = encode_input(s, off, offset_scale=40.0)
    y = encode_block_transition(s, ns)
    assert x.shape == (8,)
    assert y.shape == (3,)
    np.testing.assert_allclose(y[:2], [4.0 / 512.0, -3.0 / 512.0], atol=1e-12)
    np.testing.assert_allclose(y[2], 0.03 / (2 * np.pi), atol=1e-12)
