from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.metrics import paired_bootstrap_mean_ci

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "configs" / "protocol_external_pusht_v0.1.json").read_text())
RAW = ROOT / "results" / "raw" / "pusht_external" / "all_replicates.json"
OUT = ROOT / "results" / "derived" / "pusht_external_analysis.json"


def summarize_log(values: np.ndarray, *, seed: int) -> dict:
    mean, lo, hi = paired_bootstrap_mean_ci(
        values,
        n_boot=int(PROTOCOL["bootstrap"]["replicates"]),
        seed=seed,
    )
    return {
        "geometric_mean_ratio": float(np.exp(mean)),
        "ci95_ratio": [float(np.exp(lo)), float(np.exp(hi))],
        "mean_log_ratio": mean,
        "ci95_log": [lo, hi],
    }


def main() -> None:
    records = json.loads(RAW.read_text())
    factual = np.log([r["factual_ratio"] for r in records])
    cf = np.log([r["counterfactual_ratio"] for r in records])
    amp = np.log([r["amplification_ratio"] for r in records])
    seed = int(PROTOCOL["bootstrap"]["seed"])

    factual_s = summarize_log(factual, seed=seed)
    cf_s = summarize_log(cf, seed=seed + 1)
    amp_s = summarize_log(amp, seed=seed + 2)

    primary_threshold = float(PROTOCOL["primary_success_threshold_ratio"])
    secondary_threshold = float(PROTOCOL["secondary_counterfactual_threshold_ratio"])
    primary_pass = amp_s["ci95_ratio"][0] > primary_threshold
    secondary_pass = cf_s["ci95_ratio"][0] > secondary_threshold

    out = {
        "protocol": PROTOCOL["protocol"],
        "n_replicates": len(records),
        "factual_low_high_ratio": factual_s,
        "counterfactual_low_high_ratio": cf_s,
        "amplification_ratio": amp_s,
        "primary_threshold": primary_threshold,
        "primary_pass": bool(primary_pass),
        "secondary_threshold": secondary_threshold,
        "secondary_pass": bool(secondary_pass),
        "individual_amplification_ratios": [r["amplification_ratio"] for r in records],
        "result_policy": (
            "External support only if the frozen primary rule passes. A failure remains a reported "
            "boundary on the controlled v0.3 finding."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
