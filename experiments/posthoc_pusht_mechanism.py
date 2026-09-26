from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from src.metrics import paired_bootstrap_mean_ci
from src.model import MLPConfig
from src.pusht_external import (
    PushTTransitionDataset,
    encode_block_transition,
    encode_input,
    predict_vector,
    train_vector_mlp,
    vector_mse,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs" / "posthoc_pusht_mechanism_plan_v0.1.json"
PROTOCOL_PATH = ROOT / "configs" / "protocol_external_pusht_v0.1.json"
RAW_DIR = ROOT / "results" / "raw" / "pusht_external"
OUT_PATH = ROOT / "results" / "derived" / "pusht_mechanism_posthoc.json"


def make_env(protocol: dict):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import gymnasium as gym
    import gym_pusht  # noqa: F401

    e = protocol["environment"]
    return gym.make(e["env_id"], obs_type=e["obs_type"], render_mode=e["render_mode"])


def reset_obs(env, seed: int) -> np.ndarray:
    obs, _ = env.reset(seed=int(seed))
    return np.asarray(obs, dtype=np.float64)


def collect_dataset(env, anchor_seeds: np.ndarray, offsets: np.ndarray, *, offset_scale: float) -> PushTTransitionDataset:
    xs = np.empty((len(anchor_seeds), 8), dtype=np.float64)
    ys = np.empty((len(anchor_seeds), 3), dtype=np.float64)
    for i, (seed, offset) in enumerate(zip(anchor_seeds, offsets)):
        obs = reset_obs(env, int(seed))
        action = obs[:2] + offset
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


def collect_counterfactual_with_diagnostics(env, anchor_seeds: np.ndarray, *, offset_scale: float):
    corners = offset_scale * np.array(
        [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]],
        dtype=np.float64,
    )
    n = len(anchor_seeds) * 4
    xs = np.empty((n, 8), dtype=np.float64)
    ys = np.empty((n, 3), dtype=np.float64)
    contacts = np.empty(n, dtype=bool)
    block_translation = np.empty(n, dtype=np.float64)
    initial_distance = np.empty(n, dtype=np.float64)
    alignment = np.empty(n, dtype=np.float64)
    seeds_out = np.repeat(np.asarray(anchor_seeds, dtype=np.int64), 4)
    offsets_out = np.tile(corners, (len(anchor_seeds), 1))

    for i, (seed, offset) in enumerate(zip(seeds_out, offsets_out)):
        obs = reset_obs(env, int(seed))
        action = obs[:2] + offset
        next_obs, _, _, _, info = env.step(np.asarray(action, dtype=np.float32))
        next_obs = np.asarray(next_obs, dtype=np.float64)

        xs[i] = encode_input(obs, offset, offset_scale=offset_scale)
        ys[i] = encode_block_transition(obs, next_obs)
        contacts[i] = int(info.get("n_contacts", 0)) > 0
        block_translation[i] = float(np.hypot(next_obs[2] - obs[2], next_obs[3] - obs[3]))
        vec = obs[2:4] - obs[:2]
        dist = float(np.linalg.norm(vec))
        initial_distance[i] = dist
        denom = max(dist * float(np.linalg.norm(offset)), 1e-15)
        alignment[i] = float(np.dot(vec, offset) / denom)

    ds = PushTTransitionDataset(
        x=xs,
        y=ys,
        anchor_seeds=seeds_out,
        offsets=offsets_out,
    )
    diagnostics = {
        "contact": contacts,
        "block_translation": block_translation,
        "initial_distance": initial_distance,
        "alignment": alignment,
    }
    return ds, diagnostics


def _mse_per_sample(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean((np.asarray(pred) - np.asarray(target)) ** 2, axis=1)


def _safe_ratio(a: float, b: float, floor: float = 1e-15) -> float:
    return max(float(a), floor) / max(float(b), floor)


def _group_record(low_err: np.ndarray, high_err: np.ndarray, mask: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "low_mse": None, "high_mse": None, "ratio": None, "difference": None}
    low = float(low_err[mask].mean())
    high = float(high_err[mask].mean())
    return {
        "n": n,
        "low_mse": low,
        "high_mse": high,
        "ratio": _safe_ratio(low, high),
        "difference": low - high,
    }


def _summary_from_replicates(replicates: list[dict], group_path: tuple[str, ...], *, seed: int) -> dict:
    vals = []
    ns = []
    diffs = []
    for rec in replicates:
        cur = rec
        for key in group_path:
            cur = cur[key]
        if cur["ratio"] is not None and cur["n"] > 0:
            vals.append(np.log(cur["ratio"]))
            ns.append(cur["n"])
            diffs.append(cur["difference"])
    if len(vals) < 2:
        return {"n_replicates": len(vals), "geometric_mean_ratio": None, "ci95_ratio": None}
    mean, lo, hi = paired_bootstrap_mean_ci(np.asarray(vals), n_boot=10000, seed=seed)
    return {
        "n_replicates": len(vals),
        "mean_samples_per_replicate": float(np.mean(ns)),
        "geometric_mean_ratio": float(np.exp(mean)),
        "ci95_ratio": [float(np.exp(lo)), float(np.exp(hi))],
        "mean_absolute_mse_difference": float(np.mean(diffs)),
    }


def main() -> None:
    plan = json.loads(PLAN_PATH.read_text())
    protocol = json.loads(PROTOCOL_PATH.read_text())
    original_records = {r["replicate"]: r for r in json.loads((RAW_DIR / "all_replicates.json").read_text())}
    scale = float(protocol["action_parameterization"]["offset_scale_pixels"])
    m = protocol["model"]
    cfg = MLPConfig(
        width=int(m["hidden_width"]),
        depth=int(m["hidden_layers"]),
        learning_rate=float(m["learning_rate"]),
        weight_decay=float(m["weight_decay"]),
        batch_size=int(m["batch_size"]),
        steps=int(m["steps"]),
    )

    replicate_outputs = []
    max_rel_reproduction_error = 0.0
    env = make_env(protocol)
    try:
        for r in range(1, int(protocol["replicates"]) + 1):
            print(f"PushT mechanism diagnostic replicate {r}/{protocol['replicates']}", flush=True)
            z = np.load(RAW_DIR / f"replicate_{r:02d}_anchors.npz")
            train_low = collect_dataset(env, z["train_anchors"], z["train_low_offsets"], offset_scale=scale)
            train_high = collect_dataset(env, z["train_anchors"], z["train_high_offsets"], offset_scale=scale)
            fact_low = collect_dataset(env, z["factual_anchors"], z["factual_low_offsets"], offset_scale=scale)
            fact_high = collect_dataset(env, z["factual_anchors"], z["factual_high_offsets"], offset_scale=scale)
            cf, diag = collect_counterfactual_with_diagnostics(env, z["counterfactual_anchors"], offset_scale=scale)

            seed = int(original_records[r]["model_seed"])
            low_model = train_vector_mlp(train_low.x, train_low.y, cfg, seed=seed)
            high_model = train_vector_mlp(train_high.x, train_high.y, cfg, seed=seed)

            pred_fact_low = predict_vector(low_model, fact_low.x)
            pred_fact_high = predict_vector(high_model, fact_high.x)
            pred_cf_low = predict_vector(low_model, cf.x)
            pred_cf_high = predict_vector(high_model, cf.x)

            actual = {
                "factual_mse_low": vector_mse(pred_fact_low, fact_low.y),
                "factual_mse_high": vector_mse(pred_fact_high, fact_high.y),
                "counterfactual_mse_low": vector_mse(pred_cf_low, cf.y),
                "counterfactual_mse_high": vector_mse(pred_cf_high, cf.y),
            }
            expected = original_records[r]
            reproduction = {}
            for key, value in actual.items():
                exp = float(expected[key])
                rel = abs(value - exp) / max(abs(exp), 1e-15)
                max_rel_reproduction_error = max(max_rel_reproduction_error, rel)
                reproduction[key] = {"rerun": value, "original": exp, "relative_error": rel}
                if not np.isclose(value, exp, rtol=1e-5, atol=1e-12):
                    raise RuntimeError(f"replicate {r} failed aggregate reproduction for {key}: {value} vs {exp}")

            low_err = _mse_per_sample(pred_cf_low, cf.y)
            high_err = _mse_per_sample(pred_cf_high, cf.y)
            contact = diag["contact"]
            moving = diag["block_translation"] > 1e-4
            distance = diag["initial_distance"]
            alignment = diag["alignment"]

            distance_groups = {}
            for lo, hi in plan["secondary_descriptive_splits"]["initial_center_distance_pixels"]["bins"]:
                label = f"{lo:g}_{hi:g}"
                distance_groups[label] = _group_record(low_err, high_err, (distance >= lo) & (distance < hi))

            align_spec = plan["secondary_descriptive_splits"]["action_alignment"]
            alignment_groups = {}
            for (lo, hi), label in zip(align_spec["bins"], align_spec["labels"]):
                alignment_groups[label] = _group_record(low_err, high_err, (alignment >= lo) & (alignment < hi))

            rep = {
                "replicate": r,
                "reproduction": reproduction,
                "overall_cf": _group_record(low_err, high_err, np.ones(len(low_err), dtype=bool)),
                "contact": {
                    "yes": _group_record(low_err, high_err, contact),
                    "no": _group_record(low_err, high_err, ~contact),
                    "fraction": float(contact.mean()),
                },
                "block_motion": {
                    "yes": _group_record(low_err, high_err, moving),
                    "no": _group_record(low_err, high_err, ~moving),
                    "fraction": float(moving.mean()),
                },
                "distance_bins": distance_groups,
                "alignment_bins": alignment_groups,
                "descriptive_physics": {
                    "mean_initial_distance": float(distance.mean()),
                    "median_initial_distance": float(np.median(distance)),
                    "mean_block_translation_pixels": float(diag["block_translation"].mean()),
                    "median_block_translation_pixels": float(np.median(diag["block_translation"])),
                },
            }
            replicate_outputs.append(rep)
    finally:
        env.close()

    seed0 = int(plan["statistics"]["bootstrap_seed"])
    summaries = {
        "overall_cf": _summary_from_replicates(replicate_outputs, ("overall_cf",), seed=seed0),
        "contact_yes": _summary_from_replicates(replicate_outputs, ("contact", "yes"), seed=seed0 + 1),
        "contact_no": _summary_from_replicates(replicate_outputs, ("contact", "no"), seed=seed0 + 2),
        "block_motion_yes": _summary_from_replicates(replicate_outputs, ("block_motion", "yes"), seed=seed0 + 3),
        "block_motion_no": _summary_from_replicates(replicate_outputs, ("block_motion", "no"), seed=seed0 + 4),
        "distance_bins": {},
        "alignment_bins": {},
    }
    for j, (lo, hi) in enumerate(plan["secondary_descriptive_splits"]["initial_center_distance_pixels"]["bins"]):
        label = f"{lo:g}_{hi:g}"
        summaries["distance_bins"][label] = _summary_from_replicates(
            replicate_outputs, ("distance_bins", label), seed=seed0 + 10 + j
        )
    for j, label in enumerate(plan["secondary_descriptive_splits"]["action_alignment"]["labels"]):
        summaries["alignment_bins"][label] = _summary_from_replicates(
            replicate_outputs, ("alignment_bins", label), seed=seed0 + 20 + j
        )

    out = {
        "analysis": plan["analysis"],
        "status": "POSTHOC_EXPLORATORY",
        "cannot_change_external_pass_fail": True,
        "aggregate_reproduction_max_relative_error": max_rel_reproduction_error,
        "mean_contact_fraction": float(np.mean([r["contact"]["fraction"] for r in replicate_outputs])),
        "mean_block_motion_fraction": float(np.mean([r["block_motion"]["fraction"] for r in replicate_outputs])),
        "summaries": summaries,
        "replicates": replicate_outputs,
        "interpretation_rule": plan["statistics"]["interpretation"],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ["analysis", "aggregate_reproduction_max_relative_error", "mean_contact_fraction", "mean_block_motion_fraction", "summaries"]}, indent=2))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
