from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path
import json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))


def one(i, data_seed, model_seed, manifest):
    from src.data import make_training_dataset, make_factual_eval_dataset
    from src.metrics import mse
    from src.model import MLPConfig, train_mlp, predict
    from src.dynamics import conditional_mean, sample_states
    from src.policies import AXES, CORNERS
    dyn=manifest['primary_dynamics']; mc=manifest['primary_model']
    cfg=MLPConfig(width=mc['hidden_width'],depth=mc['hidden_layers'],learning_rate=mc['learning_rate'],weight_decay=mc['weight_decay'],batch_size=mc['batch_size'],steps=mc['steps'])
    train=make_training_dataset(manifest['n_train'],Fraction(1,64),state_seed=data_seed,action_seed=data_seed+101,noise_seed=data_seed+303,noise_std=dyn['noise_std_train'],alpha=dyn['alpha'],beta=dyn['beta'])
    model=train_mlp(train.x,train.targets_noisy,cfg,seed=model_seed)
    n=16384; states=sample_states(n,data_seed+30000)
    # exactly balanced repeated supports
    axes=np.repeat(AXES,n//4,axis=0)
    corners=np.repeat(CORNERS,n//4,axis=0)
    rng=np.random.default_rng(data_seed+30001); axes=axes[rng.permutation(n)]
    rng=np.random.default_rng(data_seed+30002); corners=corners[rng.permutation(n)]
    x_axis=np.column_stack([states,axes]); y_axis=conditional_mean(states,axes,alpha=dyn['alpha'],beta=dyn['beta'])
    x_corner=np.column_stack([states,corners]); y_corner=conditional_mean(states,corners,alpha=dyn['alpha'],beta=dyn['beta'])
    return {'pair_index':i,'axis_mse':mse(predict(model,x_axis),y_axis),'corner_mse':mse(predict(model,x_corner),y_corner)}


def main():
    m=json.loads((ROOT/'configs/confirmatory_manifest.json').read_text())
    tasks=list(enumerate(zip(m['data_seeds'],m['model_seeds']),start=1)); rows=[]
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs=[ex.submit(one,i,ds,ms,m) for i,(ds,ms) in tasks]
        for f in as_completed(futs): rows.append(f.result())
    rows.sort(key=lambda x:x['pair_index'])
    out={
      'status':'POSTHOC_EXPLORATORY_NOT_CONFIRMATORY',
      'mean_axis_mse':float(np.mean([r['axis_mse'] for r in rows])),
      'mean_corner_mse':float(np.mean([r['corner_mse'] for r in rows])),
      'expected_low_factual_from_mixture':float((63/64)*np.mean([r['axis_mse'] for r in rows])+(1/64)*np.mean([r['corner_mse'] for r in rows])),
      'rows':rows
    }
    (ROOT/'results/derived/posthoc_factual_diagnostic.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
