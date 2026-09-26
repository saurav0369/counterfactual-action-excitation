from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
A = json.loads((ROOT / 'results/derived/v03_confirmatory_analysis.json').read_text())

# Figure 1: primary ratios with 95% CIs.
labels = ['On-policy factual', 'Counterfactual', 'CF / factual\namplification']
keys = ['factual', 'counterfactual', 'amplification']
vals = [A['mlp'][k]['geometric_mean_ratio'] for k in keys]
los = [A['mlp'][k]['ci95_ratio'][0] for k in keys]
his = [A['mlp'][k]['ci95_ratio'][1] for k in keys]
err = np.array([[v-l for v,l in zip(vals,los)], [h-v for v,h in zip(vals,his)]])

fig, ax = plt.subplots(figsize=(7.4, 4.4))
x = np.arange(len(labels))
ax.errorbar(x, vals, yerr=err, fmt='o', capsize=5)
ax.set_yscale('log')
ax.set_xticks(x, labels)
ax.set_ylabel('Low-excitation / high-excitation error ratio (log scale)')
ax.set_title('Prospective v0.3 neural confirmation')
ax.axhline(1.0, linewidth=1)
ax.grid(axis='y', alpha=0.25)
fig.tight_layout()
fig.savefig(ROOT / 'figures/v03_primary_ratios.png', dpi=220)
plt.close(fig)

# Figure 2: run-level amplification across held-out beta values.
rows = A['mlp']['run_level']
betas = sorted(set(r['beta'] for r in rows))
fig, ax = plt.subplots(figsize=(7.4, 4.4))
for i, b in enumerate(betas):
    ys = [r['amplification_ratio'] for r in rows if r['beta'] == b]
    jitter = np.linspace(-0.08, 0.08, len(ys))
    ax.scatter(np.full(len(ys), i) + jitter, ys)
    gm = float(np.exp(np.mean(np.log(ys))))
    ax.hlines(gm, i-0.18, i+0.18, linewidth=2)
ax.set_xticks(range(len(betas)), [str(b) for b in betas])
ax.set_xlabel('Held-out interaction strength β')
ax.set_ylabel('Intervention-generalization amplification ratio')
ax.set_title('Amplification is consistent across held-out dynamics strengths')
ax.axhline(2.0, linewidth=1, linestyle='--')
ax.grid(axis='y', alpha=0.25)
fig.tight_layout()
fig.savefig(ROOT / 'figures/v03_by_beta.png', dpi=220)
plt.close(fig)
