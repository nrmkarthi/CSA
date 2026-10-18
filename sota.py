"""Re-implementations of SOTA multivariate time-series anomaly detectors, evaluated under the identical
protocol of XAI-CSA (same normal training data, same held-out calibration, same causal EWMA smoothing,
same 0.999-quantile threshold, same test files and metrics):
  * USAD  (Audibert et al., KDD 2020)  - adversarially trained twin-decoder auto-encoder on windows
  * GDN   (Deng & Hooi, AAAI 2021)     - graph-attention forecasting with learned sensor embeddings
  * Deep SVDD (Ruff et al., ICML 2018) - one-class hypersphere on window embeddings
Architectures follow the original papers; they are numpy re-implementations (see nn.py), not the
authors' official code."""
import numpy as np, sys, os, time, joblib
from nn import T, param, Adam, mlp, make_mlp, cat
from common import *
from detect import Q, ALPHA
from etapr_util import etapr_scores

SEEDS_ = [int(x) for x in os.environ['SEEDS_ONLY'].split(',')] if os.environ.get('SEEDS_ONLY') else [0, 1, 2]
WHICH = sys.argv[1].split(',') if len(sys.argv) > 1 else ['usad', 'gdn', 'dsvdd']


def windows(Z, w, stride=1):
    from numpy.lib.stride_tricks import sliding_window_view
    V = sliding_window_view(Z, w, axis=0)                      # (T-w+1, n, w) view, no copy
    idx = np.arange(w - 1, len(Z), stride)
    return np.ascontiguousarray(np.swapaxes(V[idx - w + 1], 1, 2)), idx


def win_scores(Z, w, fn, bs=8192):
    """Score all windows ending at t = w-1..T-1 in batches (avoids materialising every window)."""
    from numpy.lib.stride_tricks import sliding_window_view
    V = sliding_window_view(Z, w, axis=0)
    out = [fn(np.ascontiguousarray(np.swapaxes(V[i:i + bs], 1, 2)).reshape(min(bs, len(V) - i), -1)) for i in range(0, len(V), bs)]
    return np.concatenate(out), np.arange(w - 1, len(Z))


def pad_scores(s, idx, T_):
    out = np.empty(T_, np.float64); out[:idx[0]] = s[0]; out[idx] = s
    # carry forward for strided gaps (stride 1 at test time => no gaps)
    return out


# ------------------------------------------------------------------------------------------- USAD
class USAD:
    def __init__(self, dim, z=1000, seed=0):
        r = np.random.default_rng(seed)
        h1, h2 = dim // 2, dim // 4
        self.E = make_mlp([dim, h1, h2, z], r); self.D1 = make_mlp([z, h2, h1, dim], r); self.D2 = make_mlp([z, h2, h1, dim], r)
        self.pE = [p for l in self.E for p in l]; self.p1 = [p for l in self.D1 for p in l]; self.p2 = [p for l in self.D2 for p in l]

    def fit(self, X, epochs=10, bs=512, seed=0):
        o1 = Adam(self.pE + self.p1); o2 = Adam(self.pE + self.p2)
        rng = np.random.default_rng(seed)
        for n in range(1, epochs + 1):
            for b in np.array_split(rng.permutation(len(X)), max(1, len(X) // bs)):
                x = T(X[b])
                # phase 1: AE1 = D1(E(x)); AE2(AE1)
                z = mlp(x, self.E).relu(); w1 = mlp(z, self.D1, 'sigmoid'); w3 = mlp(mlp(w1, self.E).relu(), self.D2, 'sigmoid')
                l1 = (x - w1).square().mean() * (1 / n) + (x - w3).square().mean() * (1 - 1 / n)
                l1.backward(); o1.step(); o2.zero()
                z = mlp(x, self.E).relu(); w1 = mlp(z, self.D1, 'sigmoid'); w2 = mlp(z, self.D2, 'sigmoid')
                w3 = mlp(mlp(w1, self.E).relu(), self.D2, 'sigmoid')
                l2 = (x - w2).square().mean() * (1 / n) - (x - w3).square().mean() * (1 - 1 / n)
                l2.backward(); o2.step(); o1.zero()
        return self

    def score(self, X, a=0.5, b=0.5, bs=4096):
        out = []
        for i in range(0, len(X), bs):
            x = T(X[i:i + bs]); z = mlp(x, self.E).relu(); w1 = mlp(z, self.D1, 'sigmoid')
            w3 = mlp(mlp(w1, self.E).relu(), self.D2, 'sigmoid')
            out.append(a * ((x.v - w1.v) ** 2).mean(1) + b * ((x.v - w3.v) ** 2).mean(1))
        return np.concatenate(out)


# ------------------------------------------------------------------------------------------- Deep SVDD
class DeepSVDD:
    def __init__(self, dim, seed=0):
        r = np.random.default_rng(seed)
        self.L = make_mlp([dim, 256, 64, 32], r, bias=False)
        self.p = [W for W, b in self.L]

    def fit(self, X, epochs=10, bs=512, seed=0):
        self.c = mlp(T(X[:20000]), self.L).v.mean(0, keepdims=True)
        self.c[np.abs(self.c) < 0.1] = 0.1 * np.sign(self.c[np.abs(self.c) < 0.1] + 1e-9)
        opt = Adam(self.p, lr=1e-4, wd=1e-6); rng = np.random.default_rng(seed)
        for n in range(epochs):
            for b in np.array_split(rng.permutation(len(X)), max(1, len(X) // bs)):
                l = (mlp(T(X[b]), self.L) - T(self.c)).square().sum(1).mean(); l.backward(); opt.step()
        return self

    def score(self, X, bs=8192):
        return np.concatenate([((mlp(T(X[i:i + bs]), self.L).v - self.c) ** 2).sum(1) for i in range(0, len(X), bs)])


# ------------------------------------------------------------------------------------------- GDN
class GDN:
    def __init__(self, N, w=5, d=64, k=15, seed=0):
        r = np.random.default_rng(seed)
        self.N, self.w, self.d, self.k = N, w, d, k
        self.emb = param((N, d), r, 0.1); self.W = param((w, d), r)
        # attention vector a = [a1h, a1e, a2h, a2e] on (W x_i (+) v_i) (+) (W x_j (+) v_j), split for efficiency
        self.a1h, self.a1e, self.a2h, self.a2e = [param((d, 1), r, 0.1) for _ in range(4)]
        self.out = make_mlp([d, d, 1], r)
        self.p = [self.emb, self.W, self.a1h, self.a1e, self.a2h, self.a2e] + [p for l in self.out for p in l]

    def graph(self):
        e = self.emb.v / (np.linalg.norm(self.emb.v, axis=1, keepdims=True) + 1e-9)
        S = e @ e.T; np.fill_diagonal(S, -np.inf)
        nb = np.argsort(-S, 1)[:, :self.k]
        return np.c_[np.arange(self.N), nb]                     # include self-loop (k+1 neighbours)

    def forward(self, Xw, idx):                                  # Xw: (B, N, w)
        B = Xw.shape[0]; K = idx.shape[1]; d = self.d
        h = T(Xw) @ self.W                                       # B,N,d
        si = (h @ self.a1h + self.emb @ self.a1e).reshape(B, self.N, 1)
        sn = (h @ self.a2h + self.emb @ self.a2e).reshape(B, self.N).take(idx, 1)   # B,N,K
        att = (sn + si).leaky(0.2).softmax(-1)                   # GAT attention over the K neighbours
        hj = h.take(idx, 1)                                      # B,N,K,d
        z = (att.reshape(B, self.N, K, 1) * hj).sum(2).relu()   # B,N,d
        y = mlp((z * self.emb).reshape(B * self.N, d), self.out).reshape(B, self.N)
        return y

    def fit(self, Xw, Y, epochs=10, bs=64, seed=0):
        opt = Adam(self.p, lr=1e-3, wd=0.0); rng = np.random.default_rng(seed)
        for n in range(epochs):
            idx = self.graph()
            for b in np.array_split(rng.permutation(len(Xw)), max(1, len(Xw) // bs)):
                l = (self.forward(Xw[b], idx) - T(Y[b])).square().mean(); l.backward(); opt.step()
        return self

    def errors(self, Xw, Y, bs=512):
        idx = self.graph()
        return np.concatenate([np.abs(self.forward(Xw[i:i + bs], idx).v - Y[i:i + bs]) for i in range(0, len(Xw), bs)])


# ------------------------------------------------------------------------------------------- protocol
def evaluate(D, score_fn):
    cal = ewma(score_fn(D['cal']), ALPHA); thr = np.quantile(cal, Q)
    ss, ys = [], []
    for X, L in D['tests']:
        ss.append(ewma(score_fn(X), ALPHA) / thr); ys.append(L[:, 0])
    s = np.concatenate(ss); y = np.concatenate(ys)
    m = point_metrics(y, s > 1, s)
    dets = []; off = 0
    for X, L in D['tests']:
        d, _ = event_metrics(L[:, 0], s[off:off + len(L)] > 1); dets += list(d); off += len(L)
    m['event_recall'] = float(np.mean(dets))
    m.update(etapr_scores(ys, [x > 1 for x in ss]))
    return m


if __name__ == '__main__':
    D = dataset()
    out = {}
    try:
        out = loadres('sota')
    except Exception:
        pass
    for seed in SEEDS_:
        det = joblib.load(f'{CACHE}/run_hai-21.03/det_central_s{seed}.pkl')
        nm = det.norm
        Zc = [nm.z(c) for c in D['clients']]
        r = out.get(str(seed), {})
        if 'usad' in WHICH:
            t0 = time.time(); w = 10
            lo = np.min([z.min(0) for z in Zc], 0); hi = np.max([z.max(0) for z in Zc], 0)
            mm = lambda Z: np.clip((Z - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32)
            Xtr = np.vstack([windows(mm(z), w, 10)[0].reshape(-1, w * nm.keep.sum()) for z in Zc])
            m = USAD(Xtr.shape[1], z=w * 100, seed=seed).fit(Xtr, epochs=10, bs=1024, seed=seed)   # official z = window x 100
            del Xtr
            def fn(X):
                s_, idx = win_scores(mm(nm.z(X)), w, m.score); return pad_scores(s_, idx, len(X))
            r['usad'] = evaluate(D, fn); r['usad']['train_s'] = time.time() - t0
            print('usad', seed, {k: round(v, 3) for k, v in r['usad'].items()}, flush=True)
        if 'dsvdd' in WHICH:
            t0 = time.time(); w = 10
            Xtr = np.vstack([windows(z.astype(np.float32), w, 5)[0].reshape(-1, w * nm.keep.sum()) for z in Zc])
            m = DeepSVDD(Xtr.shape[1], seed=seed).fit(Xtr, epochs=10, seed=seed)
            del Xtr
            def fn(X):
                s_, idx = win_scores(nm.z(X).astype(np.float32), w, m.score); return pad_scores(s_, idx, len(X))
            r['deep_svdd'] = evaluate(D, fn); r['deep_svdd']['train_s'] = time.time() - t0
            print('dsvdd', seed, {k: round(v, 3) for k, v in r['deep_svdd'].items()}, flush=True)
        if 'gdn' in WHICH:
            t0 = time.time(); w = 5
            def wins(Z, stride=1):
                Z = Z.astype(np.float32); idx = np.arange(w, len(Z), stride)
                Xw = np.stack([Z[i - w:i].T for i in idx]); return Xw, Z[idx], idx
            parts = [wins(z, 16) for z in Zc]
            Xw = np.vstack([p[0] for p in parts]); Y = np.vstack([p[1] for p in parts])
            m = GDN(Xw.shape[1], seed=seed).fit(Xw, Y, epochs=5, seed=seed)
            Xc, Yc, _ = wins(nm.z(D['cal']), 1); E = m.errors(Xc, Yc)
            med = np.median(E, 0); iqr = np.subtract(*np.percentile(E, [75, 25], 0)) + 1e-6
            def fn(X):
                Xw_, Y_, idx = wins(nm.z(X), 1); s = (np.abs(m.errors(Xw_, Y_) - med) / iqr).max(1)
                return pad_scores(s, idx, len(X))
            r['gdn'] = evaluate(D, fn); r['gdn']['train_s'] = time.time() - t0
            print('gdn', seed, {k: round(v, 3) for k, v in r['gdn'].items()}, flush=True)
        out[str(seed)] = r
        save('sota', out)
