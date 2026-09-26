from __future__ import annotations

import json
import os
from fractions import Fraction
from pathlib import Path

import numpy as np

from src.model import MLPConfig
from src.pusht_external import (
    PushTTransitionDataset,
    anchor_passes_geometry,
    encode_block_transition,
    encode_input,
    matched_local_offsets,
    offset_moments,
    predict_vector,
    train_vector_mlp,
    vector_mse,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "protocol_external_pusht_v0.1.json"
RAW_DIR = ROOT / "results" / "raw" / "pusht_external"


def load_protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text())


def make_env(protocol: dict):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    try:
        import gymnasium as gym
        import gym_pusht  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "PushT external validation requires the optional dependency: "
            "pip install -r requirements-pusht.txt"
        ) from exc
    e = protocol["environment"]
    return gym.make(e["env_id"], obs_type=e["obs_type"], render_mode=e["render_mode"])


def reset_obs(env, seed: int) -> np.ndarray:
    obs, _ = env.reset(seed=int(seed))
    obs = np.asarray(obs, dtype=np.float64)
    if obs.shape != (5,):
        raise RuntimeError(f"unexpected PushT state shape: {obs.shape}")
    return obs


def verify_reset_determinism(env, seed: int) -> None:
    a = reset_obs(env, seed)
    b = reset_obs(env, seed)
    np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-12)


def select_anchor_seeds(env, *, start_seed: int, n: int, protocol: dict) -> tuple[np.ndarray, int]:
    f = protocol["state_control"]["anchor_filter"]
    selected: list[int] = []
    max_candidates = 100_000
    for seed in range(start_seed, start_seed + max_candidates):
        obs = reset_obs(env, seed)
        if anchor_passes_geometry(
            obs,
            agent_margin=f["agent_margin_pixels"],
            min_agent_block_distance=f["min_agent_block_distance_pixels"],
            max_agent_block_distance=f["max_agent_block_distance_pixels"],
        ):
            selected.append(seed)
            if len(selected) == n:
                return np.asarray(selected, dtype=np.int64), seed - start_seed + 1
    raise RuntimeError(
        f"failed to find {n} eligible anchors within {max_candidates} candidates "
        f"starting at seed {start_seed}"
    )


def collect_dataset(env, anchor_seeds: np.ndarray, offsets: np.ndarray, *, offset_scale: float) -> PushTTransitionDataset:
    if len(anchor_seeds) != len(offsets):
        raise ValueError("anchor_seeds and offsets must have equal length")
    xs = np.empty((len(anchor_seeds), 8), dtype=np.float64)
    ys = np.empty((len(anchor_seeds), 3), dtype=np.float64)

    for i, (seed, offset) in enumerate(zip(anchor_seeds, offsets)):
        obs = reset_obs(env, int(seed))
        action = obs[:2] + offset
        if np.any(action < 0.0) or np.any(action > 512.0):
            raise RuntimeError("absolute PushT action left [0,512]^2; clipping is forbidden")
        next_obs, _, _, _, _ = env.step(np.asarray(action, dtype=np.float32))
        next_obs = np.asarray(next_obs, dtype=np.float64)
        xs[i] = encode_input(obs, offset, offset_scale=offset_scale)
        ys[i] = encode_block_transition(obs, next_obs)

    return PushTTransitionDataset(
        x=xs,
        y=ys,
        anchor_seeds=np.asarray(anchor_seeds, dtype=np.int64),
        offsets=np.asarray(offsets, dtype=np.float64),
    )


def collect_counterfactual_dataset(env, anchor_seeds: np.ndarray, *, offset_scale: float) -> PushTTransitionDataset:
    corners = offset_scale * np.array(
        [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]],
        dtype=np.float64,
    )
    seeds = np.repeat(np.asarray(anchor_seeds, dtype=np.int64), 4)
    offsets = np.tile(corners, (len(anchor_seeds), 1))
    return collect_dataset(env, seeds, offsets, offset_scale=offset_scale)


def assert_blockwise_moment_match(low: np.ndarray, high: np.ndarray, *, block_size: int, scale: float) -> None:
    expected_cov = (scale ** 2) * np.eye(2)
    for start in range(0, len(low), block_size):
        ml = offset_moments(low[start:start + block_size])
        mh = offset_moments(high[start:start + block_size])
        np.testing.assert_allclose(ml["mean"], np.zeros(2), atol=1e-10)
        np.testing.assert_allclose(mh["mean"], np.zeros(2), atol=1e-10)
        np.testing.assert_allclose(ml["cov"], expected_cov, atol=1e-8)
        np.testing.assert_allclose(mh["cov"], expected_cov, atol=1e-8)


def run_replicate(r: int, protocol: dict) -> dict:
    env = make_env(protocol)
    try:
        base_seed = 1_000_000 + 100_000 * r
        n_train = int(protocol["n_train_anchors"])
        n_fact = int(protocol["n_factual_anchors"])
        n_cf = int(protocol["n_counterfactual_anchors"])
        total = n_train + n_fact + n_cf

        anchors, candidates_scanned = select_anchor_seeds(
            env, start_seed=base_seed, n=total, protocol=protocol
        )
        verify_reset_determinism(env, int(anchors[0]))

        train_anchors = anchors[:n_train]
        fact_anchors = anchors[n_train:n_train + n_fact]
        cf_anchors = anchors[n_train + n_fact:]

        scale = float(protocol["action_parameterization"]["offset_scale_pixels"])
        block_size = int(protocol["action_parameterization"]["block_size"])
        eps_low = Fraction(1, 16)
        eps_high = Fraction(1, 1)

        train_low_offsets = matched_local_offsets(
            n_train, eps_low, scale=scale, seed=50001 + r, block_size=block_size
        )
        train_high_offsets = matched_local_offsets(
            n_train, eps_high, scale=scale, seed=70001 + r, block_size=block_size
        )
        fact_low_offsets = matched_local_offsets(
            n_fact, eps_low, scale=scale, seed=80001 + r, block_size=block_size
        )
        fact_high_offsets = matched_local_offsets(
            n_fact, eps_high, scale=scale, seed=90001 + r, block_size=block_size
        )

        assert_blockwise_moment_match(
            train_low_offsets, train_high_offsets, block_size=block_size, scale=scale
        )
        assert_blockwise_moment_match(
            fact_low_offsets, fact_high_offsets, block_size=block_size, scale=scale
        )

        train_low = collect_dataset(env, train_anchors, train_low_offsets, offset_scale=scale)
        train_high = collect_dataset(env, train_anchors, train_high_offsets, offset_scale=scale)
        fact_low = collect_dataset(env, fact_anchors, fact_low_offsets, offset_scale=scale)
        fact_high = collect_dataset(env, fact_anchors, fact_high_offsets, offset_scale=scale)
        cf = collect_counterfactual_dataset(env, cf_anchors, offset_scale=scale)

        np.testing.assert_array_equal(train_low.anchor_seeds, train_high.anchor_seeds)
        np.testing.assert_array_equal(fact_low.anchor_seeds, fact_high.anchor_seeds)

        m = protocol["model"]
        cfg = MLPConfig(
            width=int(m["hidden_width"]),
            depth=int(m["hidden_layers"]),
            learning_rate=float(m["learning_rate"]),
            weight_decay=float(m["weight_decay"]),
            batch_size=int(m["batch_size"]),
            steps=int(m["steps"]),
        )
        model_seed = 60001 + r
        low_model = train_vector_mlp(train_low.x, train_low.y, cfg, seed=model_seed)
        high_model = train_vector_mlp(train_high.x, train_high.y, cfg, seed=model_seed)

        factual_low = vector_mse(predict_vector(low_model, fact_low.x), fact_low.y)
        factual_high = vector_mse(predict_vector(high_model, fact_high.x), fact_high.y)
        cf_low = vector_mse(predict_vector(low_model, cf.x), cf.y)
        cf_high = vector_mse(predict_vector(high_model, cf.x), cf.y)

        floor = float(protocol["evaluation"]["ratio_floor"])
        factual_ratio = max(factual_low, floor) / max(factual_high, floor)
        cf_ratio = max(cf_low, floor) / max(cf_high, floor)
        amplification = cf_ratio / factual_ratio

        record = {
            "replicate": r + 1,
            "anchor_seed_start": base_seed,
            "candidates_scanned": int(candidates_scanned),
            "anchor_seed_first": int(anchors[0]),
            "anchor_seed_last": int(anchors[-1]),
            "model_seed": model_seed,
            "factual_mse_low": factual_low,
            "factual_mse_high": factual_high,
            "counterfactual_mse_low": cf_low,
            "counterfactual_mse_high": cf_high,
            "factual_ratio": factual_ratio,
            "counterfactual_ratio": cf_ratio,
            "amplification_ratio": amplification,
            "train_low_phi_var": offset_moments(train_low_offsets)["phi_var"],
            "train_high_phi_var": offset_moments(train_high_offsets)["phi_var"],
        }

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            RAW_DIR / f"replicate_{r + 1:02d}_anchors.npz",
            train_anchors=train_anchors,
            factual_anchors=fact_anchors,
            counterfactual_anchors=cf_anchors,
            train_low_offsets=train_low_offsets,
            train_high_offsets=train_high_offsets,
            factual_low_offsets=fact_low_offsets,
            factual_high_offsets=fact_high_offsets,
        )
        (RAW_DIR / f"replicate_{r + 1:02d}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        return record
    finally:
        env.close()


def main() -> None:
    protocol = load_protocol()
    records = []
    for r in range(int(protocol["replicates"])):
        print(f"PushT external replicate {r + 1}/{protocol['replicates']}", flush=True)
        records.append(run_replicate(r, protocol))
    (RAW_DIR / "all_replicates.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"Wrote {RAW_DIR / 'all_replicates.json'}")


if __name__ == "__main__":
    main()
