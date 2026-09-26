# Matched Action Covariance Can Hide Counterfactual Weakness

This repository studies a simple failure mode in action-conditioned dynamics models: ordinary action statistics can look well covered while transition-relevant nonlinear action features are not.

The construction matches the action mean and covariance exactly across two behavior policies, while changing the excitation of the nonlinear feature `a1*a2`.

![Matched raw statistics, different nonlinear excitation](figures/fig1_matched_statistics.svg)

## Main result

The initial preregistered hypothesis required near-equivalent factual accuracy together with substantially worse counterfactual accuracy under weak excitation. That hypothesis failed: factual error also increased.

A subsequent prospective study therefore tested a narrower claim on fresh seeds and held-out interaction strengths: whether counterfactual degradation is disproportionately larger than factual degradation.

Across 20 paired runs:

| Quantity | Low / high excitation | 95% CI |
|---|---:|---:|
| Factual error | 1.90× | [1.55, 2.32] |
| Counterfactual error | 50.93× | [41.98, 61.42] |
| Counterfactual-vs-factual amplification | **26.76×** | **[23.62, 30.21]** |

The preregistered primary criterion passed: the lower bound of the 95% CI for amplification exceeded 2×. All 20 individual paired-run amplification ratios were also above 2×.

![Prospective controlled result](figures/fig2_prospective_result.svg)

## External boundary: PushT

The external test used the same matched-moment idea in PushT, with matched anchor states and the 2×128 MLP capacity and optimizer settings inherited from the controlled study.

| Quantity | Low / high excitation | 95% CI |
|---|---:|---:|
| Factual error | 1.001× | [0.975, 1.029] |
| Counterfactual error | 1.325× | [1.298, 1.350] |
| Amplification | 1.324× | [1.281, 1.363] |

The preregistered external criteria did **not** pass: the lower confidence bound did not exceed either the 2× amplification threshold or the 1.5× counterfactual-error threshold. PushT is therefore reported as a boundary on the controlled result, not as external confirmation.

![Controlled result versus PushT](figures/fig3_external_boundary.svg)

## What the study supports

In the tested controlled nonlinear dynamics, matching raw action mean and covariance did not guarantee comparable excitation of a transition-relevant nonlinear action feature. Weak excitation produced substantially greater degradation under alternative interventions than under behavior-weighted factual evaluation.

The study does **not** establish a universal failure of world models, a new theory of persistent excitation, or broad external validity.

## What failed

- **v0.2.1:** the preregistered factual-equivalence criterion failed; factual error degraded by about 1.94×.
- **PushT:** both preregistered external magnitude criteria failed.
- **Post-hoc mechanism audit:** the proposed contact/block-motion explanation was not supported; relative separation was smaller in contact and block-motion states.

The failed tests remain part of the research record.

## Repository map

- `src/` — dynamics, data, models, metrics, and PushT utilities
- `experiments/` — controlled, confirmatory, external, and diagnostic runners
- `configs/` — frozen protocols, claim boundaries, and hash manifests
- `results/derived/` — aggregate analyses used in the figures
- `results/raw/` — compact raw run summaries for the main studies
- `results/deviations/` — runtime fixes and protocol notes
- `tests/` — construction and external-protocol checks
- [`PROVENANCE.md`](PROVENANCE.md) — archival and provenance notes

## Reproduce

Install the core dependencies from the repository root:

```bash
python -m pip install -e .
```

### Verify the stored analyses

These commands recompute the reported aggregate statistics from the stored raw outputs without rerunning model training:

```bash
python -m pytest -q
python -m experiments.analyze_v03
python -m experiments.analyze_pusht_external
```

### Rerun the controlled study
```bash
python -m experiments.e1_diagnostic
python -m experiments.run_v03_confirmatory
python -m experiments.analyze_v03
```

### Rerun PushT

The frozen external run used `gym-pusht==0.1.6` with `pymunk==6.10.0`; the compatible runtime is recorded separately because the original frozen dependency file is retained for hash integrity.
```bash
python -m pip install -r requirements-pusht-runtime.txt
python -m experiments.run_pusht_external
python -m experiments.analyze_pusht_external
```

## Preprint

Technical preprint in preparation.

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff).

## License

Code is released under the MIT License.