"""Shared utilities for the XAI-CSA experiments (HAI 21.03 / 20.07)."""
import numpy as np, os, json, time
from scipy import stats

# All paths are relative to the repository root unless overridden by environment variables.
ROOT = os.environ.get('XAICSA_ROOT', os.path.dirname(os.path.abspath(__file__)))
CACHE = os.environ.get('XAICSA_CACHE', os.path.join(ROOT, 'cache'))
RES = os.environ.get('XAICSA_RESULTS', os.path.join(ROOT, 'results'))
os.makedirs(RES, exist_ok=True)
PROCS = ['P1', 'P2', 'P3', 'P4']
SEEDS = [0, 1, 2, 3, 4]


def load(ver, name):
    d = np.load(f'{CACHE}/{ver}_{name}.npz', allow_pickle=True)
    return d['X'].astype(np.float32), d['L'].astype(np.int8), [str(f) for f in d['feats']]


def dataset(ver='hai-21.03'):
    """Returns dict with fit/cal normal data and the labelled test files."""
    if ver == 'hai-21.03':
        trn = [load(ver, f'train{i}') for i in (1, 2, 3)]
        tst = [load(ver, f'test{i}') for i in (1, 2, 3, 4, 5)]
    else:  # hai-20.07: train2 contains two attack episodes -> drop those rows
        trn = [load(ver, 'train1'), load(ver, 'train2')]
        X, L, f = trn[1]
        ok = L[:, 0] == 0
        trn[1] = (X[ok], L[ok], f)
        tst = [load(ver, 'test1'), load(ver, 'test2')]
    feats = trn[0][2]
    # last 20 % of the last normal file is held out for threshold calibration
    Xl = trn[-1][0]
    n = int(len(Xl) * 0.8)
    clients = [t[0] for t in trn[:-1]] + [Xl[:n]]
    return dict(feats=feats, clients=clients, cal=Xl[n:],
                tests=[(t[0], t[1]) for t in tst],
                proc_of=np.array([PROCS.index(f.split('_')[0]) for f in feats]))


def ewma(x, a=0.05):
    """Causal exponentially weighted moving average along axis 0 (works for 1-D or 2-D)."""
    from scipy.signal import lfilter
    x = np.asarray(x, dtype=np.float64)
    zi = (1 - a) * x[0:1]          # initialise the filter state at the first value
    y, _ = lfilter([a], [1, -(1 - a)], x, axis=0, zi=zi)
    return y


def segments(mask):
    m = np.asarray(mask).astype(np.int8)
    d = np.diff(np.r_[0, m, 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def ci95(vals):
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return float(v.mean()) if len(v) else float('nan'), 0.0
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(h)


def boot_ci(vals, B=2000, seed=0):
    v = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    bs = [rng.choice(v, len(v)).mean() for _ in range(B)]
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def save(name, obj):
    with open(f'{RES}/{name}.json', 'w') as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))


def loadres(name):
    with open(f'{RES}/{name}.json') as f:
        return json.load(f)


def point_metrics(y, pred, score=None):
    from sklearn.metrics import average_precision_score
    y = np.asarray(y).astype(bool); pred = np.asarray(pred).astype(bool)
    tp = (y & pred).sum(); fp = (~y & pred).sum(); fn = (y & ~pred).sum()
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    F = 2 * P * R / max(P + R, 1e-12)
    out = dict(precision=P, recall=R, f1=F, fpr=fp / max((~y).sum(), 1))
    if score is not None:
        from sklearn.metrics import precision_recall_curve
        out['pr_auc'] = average_precision_score(y, score)
        pr, rc, _ = precision_recall_curve(y, score)
        out['f1_oracle'] = float(np.max(2 * pr * rc / np.maximum(pr + rc, 1e-12)))   # best F1 over thresholds (uses labels)
    return {k: float(v) for k, v in out.items()}


def event_metrics(y, pred, tol=0):
    """Event-level recall and detection latency (s) for each attack episode."""
    eps = segments(y)
    det, lat = [], []
    for s, e in eps:
        hit = np.where(pred[s:e + tol])[0]
        det.append(len(hit) > 0)
        lat.append(float(hit[0]) if len(hit) else np.nan)
    return np.array(det), np.array(lat)
