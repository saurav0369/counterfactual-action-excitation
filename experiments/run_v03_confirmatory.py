from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_pair(pair_index: int, data_seed: int, model_seed: int, beta: float, protocol: dict):
    from src.data import make_training_dataset, make_factual_eval_dataset, make_counterfactual_eval_dataset
    from src.linear_baseline import fit_interaction_ols, predict_interaction_ols
    from src.metrics import mse
    from src.model import MLPConfig, train_mlp, predict

    alpha = protocol['dynamics']['alpha']
    noise_std = protocol['dynamics']['noise_std_train']
    m = protocol['primary_model']
    cfg = MLPConfig(
        width=m['hidden_width'], depth=m['hidden_layers'],
        learning_rate=m['learning_rate'], weight_decay=m['weight_decay'],
        batch_size=m['batch_size'], steps=m['steps'])

    train_state_seed = data_seed
    train_noise_seed = data_seed + 303
    factual_state_seed = data_seed + 10_000
    cf_state_seed = data_seed + 20_000
    cf_action_seed = data_seed + 20_101

    cf = make_counterfactual_eval_dataset(
        protocol['n_counterfactual_test'], state_seed=cf_state_seed,
        action_seed=cf_action_seed, alpha=alpha, beta=beta)

    out = {
        'pair_index': pair_index,
        'data_seed': data_seed,
        'model_seed': model_seed,
        'beta': beta,
        'conditions': {}
    }

    for cname, eps, action_offset in [
        ('low', Fraction(1, 64), 101),
        ('high', Fraction(1, 1), 202),
    ]:
        train = make_training_dataset(
            protocol['n_train'], eps,
            state_seed=train_state_seed,
            action_seed=data_seed + action_offset,
            noise_seed=train_noise_seed,
            noise_std=noise_std, alpha=alpha, beta=beta)
        factual = make_factual_eval_dataset(
            protocol['n_factual_test'], eps,
            state_seed=factual_state_seed,
            action_seed=data_seed + 10_000 + action_offset,
            alpha=alpha, beta=beta)

        model = train_mlp(train.x, train.targets_noisy, cfg, seed=model_seed)
        mlp_fact = mse(predict(model, factual.x), factual.targets_mean)
        mlp_cf = mse(predict(model, cf.x), cf.targets_mean)

        ols_coef = fit_interaction_ols(train.states, train.actions, train.targets_noisy)
        ols_fact = mse(predict_interaction_ols(ols_coef, factual.states, factual.actions), factual.targets_mean)
        ols_cf = mse(predict_interaction_ols(ols_coef, cf.states, cf.actions), cf.targets_mean)

        out['conditions'][cname] = {
            'mlp': {'factual_mse': mlp_fact, 'counterfactual_mse': mlp_cf},
            'ols': {
                'factual_mse': ols_fact,
                'counterfactual_mse': ols_cf,
                'coef': ols_coef.tolist(),
            }
        }
    return out


def main():
    protocol = json.loads((ROOT / 'configs/protocol_v0.3.json').read_text())
    raw_dir = ROOT / 'results/raw/v03_confirmatory'
    raw_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for i, (ds, ms, beta) in enumerate(zip(
        protocol['data_seeds'], protocol['model_seeds'], protocol['dynamics']['beta_schedule']
    ), start=1):
        tasks.append((i, ds, ms, beta))

    results = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {
            ex.submit(run_pair, i, ds, ms, beta, protocol): i
            for i, ds, ms, beta in tasks
        }
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            (raw_dir / f"pair_{r['pair_index']:02d}.json").write_text(json.dumps(r, indent=2))
            print(f"completed v0.3 pair {r['pair_index']:02d} beta={r['beta']}", flush=True)

    results.sort(key=lambda r: r['pair_index'])
    (raw_dir / 'all_pairs.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
