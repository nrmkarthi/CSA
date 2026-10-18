"""Layer 2/3 of XAI-CSA: normalisation, hybrid anomaly detection, attack classification.

Anomaly detector  = ensemble of
   (i)  Gaussian conditional residual model (GCR): residual of each tag given all other tags
        (closed form via the precision matrix; the precision matrix doubles as the
         data-driven asset-dependency graph),
   (ii) MLP auto-encoder (AE) reconstruction residual,
   (iii) invariant rules for tags that are constant during normal operation.
Every component score is a sum of per-tag contributions, so the fused score decomposes
exactly into per-tag attributions phi_i(t) (used by the XAI module).
"""
import numpy as np, warnings
from sklearn.neural_network import MLPRegressor
from common import ewma

warnings.filterwarnings('ignore')
LAG = 10
Q = 0.999          # calibration quantile on held-out normal data (target FPR 0.1 %)
ALPHA = 0.05       # EWMA smoothing factor
W_RULE = 5.0


def sum_stats(X):
    X = X.astype(np.float64)
    return len(X), X.sum(0), (X ** 2).sum(0)


def norm_from_stats(stats_list):
    n = sum(s[0] for s in stats_list); s1 = sum(s[1] for s in stats_list); s2 = sum(s[2] for s in stats_list)
    mu = s1 / n; sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0))
    return mu, sd


class Normalizer:
    def __init__(self, mu, sd):
        self.mu, self.sd = mu, sd
        self.keep = sd > 1e-6
        self.const = ~self.keep

    def z(self, X):
        return ((X - self.mu) / np.where(self.keep, self.sd, 1.0))[:, self.keep]

    def aug(self, X):
        Z = self.z(X)
        L = np.vstack([np.repeat(Z[:1], LAG, 0), Z])[:len(Z)]
        return np.hstack([Z, Z - L]).astype(np.float32)

    def rule_viol(self, X):
        c = self.mu[self.const]
        return (np.abs(X[:, self.const] - c) > 1e-3 * np.maximum(1, np.abs(c))).astype(np.float32)


class GCR:
    """Gaussian conditional residuals r_i = (Lambda z)_i / sqrt(Lambda_ii)."""
    def __init__(self, cov, p, shrink=1e-3):
        d = cov.shape[0]
        S = cov + shrink * np.eye(d) * np.trace(cov) / d
        self.P = np.linalg.inv(S)
        self.p = p
        dg = np.diag(self.P)
        self.scale = (1 / np.sqrt(dg))[:p]
        self.Pc = self.P[:, :p]

    def resid(self, A):
        return (A @ self.Pc) * self.scale[None, :]

    def partial_corr(self):
        P = self.P[:self.p, :self.p]
        d = np.sqrt(np.diag(P))
        return -P / np.outer(d, d)


def aug_cov_stats(A):
    A = A.astype(np.float64)
    return len(A), A.sum(0), A.T @ A


def cov_from_stats(st):
    n = sum(s[0] for s in st); s1 = sum(s[1] for s in st); S = sum(s[2] for s in st)
    m = s1 / n
    return S / n - np.outer(m, m)


def new_ae(seed, hidden=(128, 32, 128)):
    return MLPRegressor(hidden_layer_sizes=hidden, batch_size=512, learning_rate_init=1e-3,
                        random_state=seed, max_iter=1, warm_start=False)


def ae_train_central(A, p, seed, epochs=10, sub=4):
    m = new_ae(seed)
    rng = np.random.default_rng(seed)
    Asub = A[::sub]
    for ep in range(epochs):
        idx = rng.permutation(len(Asub))
        m.partial_fit(Asub[idx], Asub[idx, :p])
    return m


class Detector:
    def __init__(self, norm, gcr, ae, feats, proc_of):
        self.norm, self.gcr, self.ae = norm, gcr, ae
        self.feats = feats; self.proc_of = proc_of
        self.p = int(norm.keep.sum())
        # index mapping of per-tag attributions back to the original 79 tags
        self.idx_keep = np.where(norm.keep)[0]; self.idx_const = np.where(norm.const)[0]
        self.rule_on = np.ones(len(self.idx_const), bool)
        self.bias_gcr = np.zeros(self.p); self.bias_ae = np.zeros(self.p)  # drift-compensation offsets
        self.disabled = set()  # component failure simulation: {'gcr','ae','rule'}

    # raw per-tag squared residuals (unsmoothed)
    def raw_contrib(self, X, A=None):
        if A is None:
            A = self.norm.aug(X)
        out = {}
        if self.gcr is not None:
            r = self.gcr.resid(A) - self.bias_gcr
            out['gcr'] = r ** 2
        if self.ae is not None:
            e = (A[:, :self.p] - self.ae.predict(A)) / self.ae_sd - self.bias_ae
            out['ae'] = e ** 2
        out['rule'] = self.norm.rule_viol(X) * self.rule_on[None, :]
        return out

    def calibrate(self, Xcal):
        A = self.norm.aug(Xcal)
        if self.ae is not None:
            self.ae_sd = (A[:, :self.p] - self.ae.predict(A)).std(0) + 1e-6
        rc = self.raw_contrib(Xcal, A)
        self.theta = {}
        for k in ('gcr', 'ae'):
            if k in rc:
                s = ewma(rc[k].mean(1), ALPHA)
                self.theta[k] = np.quantile(s, Q)
        # rules that fire during held-out normal operation are disabled (governance)
        self.rule_on = ~(rc['rule'].max(0) > 0)
        phi = self._phi(rc, smooth=True)
        self.theta_f = np.quantile(phi.sum(1), Q)
        return self

    def _phi(self, rc, smooth=True):
        T = len(rc['rule'])
        phi = np.zeros((T, len(self.feats)))
        comps = [k for k in ('gcr', 'ae') if k in rc and k not in self.disabled]
        for k in comps:
            c = rc[k] / (self.p * self.theta[k]) / len(comps)
            phi[:, self.idx_keep] += c
        if 'rule' not in self.disabled:
            phi[:, self.idx_const] += W_RULE * rc['rule']
        if smooth:
            phi = ewma(phi, ALPHA)
        return phi

    def score(self, X):
        rc = self.raw_contrib(X)
        phi = self._phi(rc) / self.theta_f
        return phi.astype(np.float32)          # alert <=> phi.sum(1) > 1


def build_detector(D, seed, ae_model=None, clients=None, epochs=10):
    """Centralised training on the given clients' normal data (default: all clients)."""
    clients = D['clients'] if clients is None else clients
    mu, sd = norm_from_stats([sum_stats(c) for c in clients])
    nm = Normalizer(mu, sd)
    As = [nm.aug(c) for c in clients]
    p = int(nm.keep.sum())
    cov = cov_from_stats([aug_cov_stats(a[::5]) for a in As])
    gcr = GCR(cov, p)
    if ae_model is None:
        ae_model = ae_train_central(np.vstack(As), p, seed, epochs=epochs)
    det = Detector(nm, gcr, ae_model, D['feats'], D['proc_of'])
    return det.calibrate(D['cal'])
