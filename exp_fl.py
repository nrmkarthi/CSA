"""RQ4: privacy-preserving distributed learning -- accuracy, communication, DP noise, poisoning."""
import numpy as np, sys, os, time, json, joblib
from common import *
from fl import train_federated, make_sites
from exp_robust import det_metrics

WHAT = sys.argv[1].split(',') if len(sys.argv) > 1 else ['base', 'dp', 'poison']
SEEDS_ = [int(x) for x in os.environ['SEEDS_ONLY'].split(',')] if os.environ.get('SEEDS_ONLY') else SEEDS
R = 50


def score_all(D, det):
    return det_metrics(D, [det.score(X).sum(1) for X, L in D['tests']])


if __name__ == '__main__':
    D = dataset()
    out = {}
    try:
        out = loadres('fl')
    except Exception:
        pass
    for seed in SEEDS_:
        r = out.get(str(seed), {})
        t0 = time.time()
        if 'base' in WHAT:
            for tag in ['central', 'fedavg'] + [f'local{i}' for i in range(5)]:
                p = f'{CACHE}/run_hai-21.03/det_{tag}_s{seed if not tag.startswith("local") else int(tag[5:])}.pkl'
                if tag.startswith('local') and int(tag[5:]) != seed:
                    continue
                if os.path.exists(p):
                    r['local' if tag.startswith('local') else tag] = score_all(D, joblib.load(p))
            c = json.load(open(f'{CACHE}/run_hai-21.03/comm_fedavg_s{seed}.json'))
            r['comm'] = c
        if 'dp' in WHAT:
            for z in (1.0,):
                det, comm = train_federated(D, seed, rounds=R, dp_sigma=z, clip=None)
                r[f'dp_z{z}'] = score_all(D, det)
        if 'poison' in WHAT:
            for agg in ('fedavg', 'trimmed'):
                det, comm = train_federated(D, seed, rounds=R, agg=agg, poison=dict(site=2, kind='flip', scale=5.0))
                r[f'poison_{agg}'] = score_all(D, det)
            det, comm = train_federated(D, seed, rounds=R, agg='trimmed')
            r['clean_trimmed'] = score_all(D, det)
        out[str(seed)] = r
        save('fl', out)
        print('fl', seed, WHAT, round(time.time() - t0), 's', flush=True)
