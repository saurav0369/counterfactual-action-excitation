from pathlib import Path
import json
import math
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def stratified_bootstrap(rows, field, beta_values, reps, seed):
    rng = np.random.default_rng(seed)
    groups = {b: [r for r in rows if r['beta'] == b] for b in beta_values}
    point = float(np.mean([r[field] for r in rows]))
    boots = np.empty(reps, dtype=np.float64)
    for k in range(reps):
        vals = []
        for b in beta_values:
            g = groups[b]
            idx = rng.integers(0, len(g), size=len(g))
            vals.extend(g[j][field] for j in idx)
        boots[k] = np.mean(vals)
    ci = np.quantile(boots, [0.025, 0.975])
    return point, [float(ci[0]), float(ci[1])]


def summarize_model(raw, model_key, protocol):
    rows = []
    for r in raw:
        low = r['conditions']['low'][model_key]
        high = r['conditions']['high'][model_key]
        fact_log = math.log(low['factual_mse'] / high['factual_mse'])
        cf_log = math.log(low['counterfactual_mse'] / high['counterfactual_mse'])
        rows.append({
            'pair_index': r['pair_index'], 'beta': r['beta'],
            'factual_log_ratio': fact_log,
            'counterfactual_log_ratio': cf_log,
            'amplification_log_ratio': cf_log - fact_log,
            'factual_ratio': math.exp(fact_log),
            'counterfactual_ratio': math.exp(cf_log),
            'amplification_ratio': math.exp(cf_log - fact_log),
        })

    beta_values = sorted(set(protocol['dynamics']['beta_schedule']))
    reps = protocol['bootstrap']['replicates']
    seed = protocol['bootstrap']['seed']
    fact_mean, fact_ci = stratified_bootstrap(rows, 'factual_log_ratio', beta_values, reps, seed)
    cf_mean, cf_ci = stratified_bootstrap(rows, 'counterfactual_log_ratio', beta_values, reps, seed + 1)
    amp_mean, amp_ci = stratified_bootstrap(rows, 'amplification_log_ratio', beta_values, reps, seed + 2)

    return {
        'factual': {
            'mean_log_ratio': fact_mean,
            'geometric_mean_ratio': math.exp(fact_mean),
            'ci95_log': fact_ci,
            'ci95_ratio': [math.exp(fact_ci[0]), math.exp(fact_ci[1])],
        },
        'counterfactual': {
            'mean_log_ratio': cf_mean,
            'geometric_mean_ratio': math.exp(cf_mean),
            'ci95_log': cf_ci,
            'ci95_ratio': [math.exp(cf_ci[0]), math.exp(cf_ci[1])],
            'secondary_pass': cf_ci[0] > math.log(protocol['secondary_counterfactual_threshold_ratio']),
        },
        'amplification': {
            'mean_log_ratio': amp_mean,
            'geometric_mean_ratio': math.exp(amp_mean),
            'ci95_log': amp_ci,
            'ci95_ratio': [math.exp(amp_ci[0]), math.exp(amp_ci[1])],
            'threshold_ratio': protocol['primary_success_threshold_ratio'],
            'primary_pass': amp_ci[0] > math.log(protocol['primary_success_threshold_ratio']),
        },
        'run_level': rows,
    }


def main():
    protocol = json.loads((ROOT / 'configs/protocol_v0.3.json').read_text())
    raw = json.loads((ROOT / 'results/raw/v03_confirmatory/all_pairs.json').read_text())
    out = {
        'status': 'V0.3_CONFIRMATORY_ANALYSIS',
        'n_pairs': len(raw),
        'beta_values': sorted(set(protocol['dynamics']['beta_schedule'])),
        'mlp': summarize_model(raw, 'mlp', protocol),
        'ols_calibration': summarize_model(raw, 'ols', protocol),
    }
    out['primary_hypothesis_supported'] = out['mlp']['amplification']['primary_pass']
    out['secondary_counterfactual_supported'] = out['mlp']['counterfactual']['secondary_pass']
    p = ROOT / 'results/derived/v03_confirmatory_analysis.json'
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        'mlp_factual_ratio': out['mlp']['factual']['geometric_mean_ratio'],
        'mlp_cf_ratio': out['mlp']['counterfactual']['geometric_mean_ratio'],
        'mlp_amp_ratio': out['mlp']['amplification']['geometric_mean_ratio'],
        'mlp_amp_ci': out['mlp']['amplification']['ci95_ratio'],
        'primary_pass': out['primary_hypothesis_supported'],
        'ols_factual_ratio': out['ols_calibration']['factual']['geometric_mean_ratio'],
        'ols_cf_ratio': out['ols_calibration']['counterfactual']['geometric_mean_ratio'],
        'ols_amp_ratio': out['ols_calibration']['amplification']['geometric_mean_ratio'],
    }, indent=2))


if __name__ == '__main__':
    main()
