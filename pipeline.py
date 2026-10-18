"""Runs Layers 2-4 for each seed and caches out-of-fold outputs (leave-one-test-file-out)."""
import numpy as np, joblib, os, sys, time
from common import *
from detect import build_detector
from sa import frame_features, SituationClassifier

VER = sys.argv[1] if len(sys.argv) > 1 else 'hai-21.03'
OUT = f'{CACHE}/run_{VER}'
os.makedirs(OUT, exist_ok=True)


def run_seed(seed, D, tag='central', det=None):
    t0 = time.time()
    if det is None:
        det = build_detector(D, seed)
    joblib.dump(det, f'{OUT}/det_{tag}_s{seed}.pkl')
    phis, Fs, Ls = [], [], []
    tsc = time.time()
    for X, L in D['tests']:
        phi = det.score(X); phis.append(phi); Fs.append(frame_features(phi, det.proc_of)); Ls.append(L)
    t_score = time.time() - tsc
    nfeat = len(det.feats)
    K = len(Fs)
    for k in range(K):
        tr = [j for j in range(K) if j != k]
        clf = SituationClassifier(seed).fit([Fs[j] for j in tr], [Ls[j] for j in tr], nfeat)
        ph, tg = clf.predict(Fs[k], nfeat)
        np.savez_compressed(f'{OUT}/oof_{tag}_s{seed}_f{k}.npz', phi=phis[k].astype(np.float32), ph=ph.astype(np.float32),
                            tg=tg.astype(np.float32), L=Ls[k])
    print(VER, tag, 'seed', seed, 'done in', round(time.time() - t0, 1), 's; scoring', round(t_score, 1), flush=True)


if __name__ == '__main__':
    D = dataset(VER)
    mode = sys.argv[2] if len(sys.argv) > 2 else 'central'
    seeds = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else SEEDS
    import json
    for s in seeds:
        if mode == 'central':
            run_seed(s, D)
        elif mode == 'fedavg':
            from fl import train_federated
            R = int(os.environ.get('FL_ROUNDS', '40'))
            t = time.time(); det, comm = train_federated(D, s, rounds=R); comm['train_s'] = time.time() - t
            json.dump(comm, open(f'{OUT}/comm_fedavg_s{s}.json', 'w'))
            run_seed(s, D, tag='fedavg', det=det)
        elif mode.startswith('local'):
            from fl import train_local, make_sites
            site = int(mode[5:])
            det = train_local(D, s, make_sites(D)[site])
            run_seed(s, D, tag=mode, det=det)
