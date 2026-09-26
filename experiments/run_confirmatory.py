from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_pair(pair_index: int, data_seed: int, model_seed: int, manifest: dict):
    from src.data import make_training_dataset, make_factual_eval_dataset, make_counterfactual_eval_dataset
    from src.metrics import mse
    from src.model import MLPConfig, train_mlp, predict

    dyn = manifest['primary_dynamics']
    model_cfg = manifest['primary_model']
    cfg = MLPConfig(
        width=model_cfg['hidden_width'], depth=model_cfg['hidden_layers'],
        learning_rate=model_cfg['learning_rate'], weight_decay=model_cfg['weight_decay'],
        batch_size=model_cfg['batch_size'], steps=model_cfg['steps'])

    state_seed_train = data_seed
    noise_seed_train = data_seed + 303
    factual_state_seed = data_seed + 10_000
    cf_state_seed = data_seed + 20_000
    cf_action_seed = data_seed + 20_101

    cf = make_counterfactual_eval_dataset(
        manifest['n_counterfactual_test'], state_seed=cf_state_seed,
        action_seed=cf_action_seed, alpha=dyn['alpha'], beta=dyn['beta'])

    out = {'pair_index':pair_index,'data_seed':data_seed,'model_seed':model_seed,'conditions':{}}
    for cname, eps, action_offset in [
        ('low', Fraction(1,64), 101),
        ('high', Fraction(1,1), 202),
    ]:
        train = make_training_dataset(
            manifest['n_train'], eps, state_seed=state_seed_train,
            action_seed=data_seed + action_offset, noise_seed=noise_seed_train,
            noise_std=dyn['noise_std_train'], alpha=dyn['alpha'], beta=dyn['beta'])
        factual = make_factual_eval_dataset(
            manifest['n_factual_test'], eps, state_seed=factual_state_seed,
            action_seed=data_seed + 10_000 + action_offset,
            alpha=dyn['alpha'], beta=dyn['beta'])
        model = train_mlp(train.x, train.targets_noisy, cfg, seed=model_seed)
        out['conditions'][cname] = {
            'factual_mse':mse(predict(model, factual.x), factual.targets_mean),
            'counterfactual_mse':mse(predict(model, cf.x), cf.targets_mean),
        }
    return out


def main():
    manifest=json.loads((ROOT/'configs/confirmatory_manifest.json').read_text())
    raw_dir=ROOT/'results/raw/confirmatory'; raw_dir.mkdir(parents=True,exist_ok=True)
    tasks=list(enumerate(zip(manifest['data_seeds'],manifest['model_seeds']),start=1))
    results=[]
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs={ex.submit(run_pair,i,ds,ms,manifest):i for i,(ds,ms) in tasks}
        for fut in as_completed(futs):
            r=fut.result(); results.append(r)
            p=raw_dir/f"pair_{r['pair_index']:02d}.json"; p.write_text(json.dumps(r,indent=2))
            print(f"completed pair {r['pair_index']:02d}", flush=True)
    results.sort(key=lambda r:r['pair_index'])
    (raw_dir/'all_pairs.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__': main()
