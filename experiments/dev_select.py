from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from itertools import product
from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parameter_count(width: int, depth: int) -> int:
    total = 0
    in_dim = 3
    for _ in range(depth):
        total += in_dim * width + width
        in_dim = width
    total += in_dim + 1
    return total


def eval_config(args):
    width, depth, lr, grid, protocol = args
    from src.data import make_training_dataset, make_factual_eval_dataset
    from src.metrics import mse
    from src.model import MLPConfig, train_mlp, predict

    alpha = protocol["primary_dynamics"]["alpha"]
    beta = protocol["primary_dynamics"]["beta"]
    noise_std = protocol["primary_dynamics"]["noise_std_train"]
    n_train = protocol["n_train"]
    n_val = protocol["n_validation"]
    cfg = MLPConfig(width=width, depth=depth, learning_rate=lr,
                    weight_decay=grid["weight_decay"],
                    batch_size=grid["batch_size"], steps=grid["steps"])
    conditions = [("low", Fraction(1,64)), ("high", Fraction(1,1))]
    losses, details = [], []
    for dseed, mseed in zip(grid["dev_data_seeds"], grid["dev_model_seeds"]):
        for cname, eps in conditions:
            train = make_training_dataset(
                n_train, eps, state_seed=dseed,
                action_seed=dseed + (101 if cname == "low" else 202),
                noise_seed=dseed + 303, noise_std=noise_std,
                alpha=alpha, beta=beta)
            val = make_factual_eval_dataset(
                n_val, eps, state_seed=dseed + 10_000,
                action_seed=dseed + (10_101 if cname == "low" else 10_202),
                alpha=alpha, beta=beta)
            model = train_mlp(train.x, train.targets_noisy, cfg, seed=mseed)
            v = mse(predict(model, val.x), val.targets_mean)
            losses.append(v)
            details.append({"data_seed":dseed,"model_seed":mseed,"condition":cname,"val_mse":v})
    return {
        "width":width,"depth":depth,"learning_rate":lr,
        "weight_decay":grid["weight_decay"],"batch_size":grid["batch_size"],"steps":grid["steps"],
        "parameter_count":parameter_count(width, depth),
        "mean_factual_val_mse":sum(losses)/len(losses),"details":details
    }


def main():
    grid = json.loads((ROOT/'configs/dev_grid.json').read_text())
    protocol = json.loads((ROOT/'configs/protocol_v0.2.1.json').read_text())
    combos = list(product(grid['widths'], grid['depths'], grid['learning_rates']))
    tasks = [(w,d,lr,grid,protocol) for w,d,lr in combos]
    t0=time.time(); rows=[]
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs=[ex.submit(eval_config,t) for t in tasks]
        for fut in as_completed(futs):
            r=fut.result(); rows.append(r)
            print(r['width'],r['depth'],r['learning_rate'],r['mean_factual_val_mse'],flush=True)
    rows.sort(key=lambda r:(r['mean_factual_val_mse'],r['parameter_count'],r['learning_rate']))
    out={"winner":rows[0],"all_configs":rows,"elapsed_seconds":time.time()-t0,"runner":"parallel_semantics_equivalent"}
    p=ROOT/'results/derived/dev_selection.json'; p.write_text(json.dumps(out,indent=2))
    print('WINNER',json.dumps(rows[0],indent=2),flush=True)

if __name__=='__main__': main()
