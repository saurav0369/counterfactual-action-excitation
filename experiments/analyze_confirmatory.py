from pathlib import Path
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from src.metrics import safe_log_ratio, paired_bootstrap_mean_ci


def main():
    manifest=json.loads((ROOT/'configs/confirmatory_manifest.json').read_text())
    rows=json.loads((ROOT/'results/raw/confirmatory/all_pairs.json').read_text())
    if len(rows)!=20:
        raise RuntimeError(f"expected 20 complete pairs, found {len(rows)}")

    factual=np.array([safe_log_ratio(r['conditions']['low']['factual_mse'],r['conditions']['high']['factual_mse']) for r in rows])
    cf=np.array([safe_log_ratio(r['conditions']['low']['counterfactual_mse'],r['conditions']['high']['counterfactual_mse']) for r in rows])
    b=manifest['bootstrap']
    f_mean,f_lo,f_hi=paired_bootstrap_mean_ci(factual,n_boot=b['replicates'],seed=b['seed'])
    c_mean,c_lo,c_hi=paired_bootstrap_mean_ci(cf,n_boot=b['replicates'],seed=b['seed'])
    eq_lo,eq_hi=manifest['success_criteria']['factual_log_ratio_ci_must_lie_inside']
    cf_thr=manifest['success_criteria']['counterfactual_mean_log_ratio_ci_lower_must_exceed']
    factual_pass=(f_lo>eq_lo and f_hi<eq_hi)
    cf_pass=(c_lo>cf_thr)

    result={
      'n_pairs':len(rows),
      'factual_log_ratio':{'mean':f_mean,'ci95':[f_lo,f_hi],'geometric_mean_ratio_low_over_high':float(np.exp(f_mean)),'ci95_ratio':[float(np.exp(f_lo)),float(np.exp(f_hi))],'equivalence_margin_ratio':[float(np.exp(eq_lo)),float(np.exp(eq_hi))],'pass':factual_pass},
      'counterfactual_log_ratio':{'mean':c_mean,'ci95':[c_lo,c_hi],'geometric_mean_ratio_low_over_high':float(np.exp(c_mean)),'ci95_ratio':[float(np.exp(c_lo)),float(np.exp(c_hi))],'minimum_required_ratio':float(np.exp(cf_thr)),'pass':cf_pass},
      'primary_hypothesis_supported':bool(factual_pass and cf_pass),
      'run_level':[
        {'pair_index':r['pair_index'],'factual_ratio_low_over_high':float(np.exp(factual[i])),'counterfactual_ratio_low_over_high':float(np.exp(cf[i])),
         'low_factual_mse':r['conditions']['low']['factual_mse'],'high_factual_mse':r['conditions']['high']['factual_mse'],
         'low_counterfactual_mse':r['conditions']['low']['counterfactual_mse'],'high_counterfactual_mse':r['conditions']['high']['counterfactual_mse']}
        for i,r in enumerate(rows)]
    }
    out=ROOT/'results/derived/confirmatory_analysis.json'; out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
