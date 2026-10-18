"""Detection baselines on identical splits: Isolation Forest, PCA, GCR-only, AE-only, fused (ours)."""
import numpy as np, sys, os, joblib
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from common import *
from detect import ALPHA, Q

VER = sys.argv[1] if len(sys.argv) > 1 else 'hai-21.03'
TAG = sys.argv[2] if len(sys.argv) > 2 else 'central'
SEEDS_ = [int(x) for x in os.environ['SEEDS_ONLY'].split(',')] if os.environ.get('SEEDS_ONLY') else SEEDS


def evaluate(D, score_fn, cal_score):
    thr = np.quantile(cal_score, Q)
    ys, ss = [], []
    for X, L in D['tests']:
        ss.append(score_fn(X) / thr); ys.append(L[:, 0])
    s = np.concatenate(ss); y = np.concatenate(ys)
    m = point_metrics(y, s > 1, s)
    det = []; off = 0
    for X, L in D['tests']:
        d, l = event_metrics(L[:, 0], s[off:off + len(L)] > 1); det += list(d); off += len(L)
    m['event_recall'] = float(np.mean(det))
    from etapr_util import etapr_scores
    off = 0; ps = []; yl = []
    for X, L in D['tests']:
        ps.append(s[off:off + len(L)] > 1); yl.append(L[:, 0]); off += len(L)
    m.update(etapr_scores(yl, ps))
    return m


if __name__ == '__main__':
    D = dataset(VER)
    out = {}
    for seed in SEEDS_:
        det = joblib.load(f'{CACHE}/run_{VER}/det_{TAG}_s{seed}.pkl')
        nm = det.norm
        Afit = np.vstack([nm.aug(c)[::20] for c in D['clients']])
        r = {}
        ifo = IsolationForest(n_estimators=200, random_state=seed, n_jobs=1).fit(Afit)
        f = lambda X: ewma(-ifo.score_samples(nm.aug(X)), ALPHA)
        r['isolation_forest'] = evaluate(D, f, f(D['cal']))
        pca = PCA(0.99, random_state=seed).fit(Afit)
        def fp(X):
            A = nm.aug(X); R = A - pca.inverse_transform(pca.transform(A)); return ewma((R[:, :det.p] ** 2).mean(1), ALPHA)
        r['pca'] = evaluate(D, fp, fp(D['cal']))
        for comp in ('gcr', 'ae'):
            det.disabled = {'gcr', 'ae', 'rule'} - {comp}
            g = lambda X: det.score(X).sum(1)
            r[f'{comp}_only'] = evaluate(D, g, g(D['cal']))
        det.disabled = set()
        g = lambda X: det.score(X).sum(1)
        r['fused_ours'] = evaluate(D, g, g(D['cal']))
        out[seed] = r
        print('baselines', VER, seed, {k: round(v['f1'], 3) for k, v in r.items()}, flush=True)
        save(f'baselines_{TAG}_{VER}', out)
