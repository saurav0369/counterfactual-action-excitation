from fractions import Fraction
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.policies import matched_covariance_actions, empirical_moments

OUT = ROOT / "results" / "derived"
FIG = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def serialize(m):
    return {
        "mean": np.asarray(m["mean"]).tolist(),
        "cov": np.asarray(m["cov"]).tolist(),
        "phi_mean": m["phi_mean"],
        "phi_var": m["phi_var"],
    }


def main():
    n = 4096
    low = matched_covariance_actions(n, Fraction(1, 64), seed=123)
    high = matched_covariance_actions(n, Fraction(1, 1), seed=456)
    result = {"low": serialize(empirical_moments(low)), "high": serialize(empirical_moments(high))}
    (OUT / "e1_moments.json").write_text(json.dumps(result, indent=2))

    # One concept-first figure. No style/color customization by design.
    fig, ax = plt.subplots()
    ax.scatter(low[:, 0], low[:, 1], alpha=0.12, label="low excitation")
    ax.scatter(high[:, 0], high[:, 1], alpha=0.12, label="high excitation")
    ax.set_xlabel("a1")
    ax.set_ylabel("a2")
    ax.set_title("Matched covariance, different nonlinear excitation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "e1_action_support.png", dpi=180)
    plt.close(fig)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
