"""Explainability module (M5) of XAI-CSA and the explanation-quality metrics.

* residual attribution  : exact additive decomposition of the fused anomaly score into per-tag terms
* Shapley sampling       : model-agnostic baseline (permutation sampling, mean baseline)
* situation-grounded     : attribution accumulated over the incident window, diffused over the
                           asset graph and fused with the attack classifier's target estimate
Metrics: deletion fidelity, temporal & perturbation stability, cross-method consistency,
process localisation.
"""
import numpy as np
from common import ewma


def raw_score_and_phi(det, X):
    """Unsmoothed fused score and per-tag attribution for each row of X (rows are full time series
    so that lag features are valid)."""
    rc = det.raw_contrib(X)
    phi = det._phi(rc, smooth=False) / det.theta_f
    return phi.sum(1), phi


def _cond_impute_aug(det, A, S_tags):
    """Gaussian-conditional imputation in the augmented space of the tags in S_tags (kept-tag indices)."""
    p = det.p
    dims = np.r_[S_tags, S_tags + p]
    P = det.gcr.P
    rest = np.setdiff1d(np.arange(P.shape[0]), dims)
    # E[a_S | a_rest] = -P_SS^{-1} P_S,rest a_rest
    M = -np.linalg.solve(P[np.ix_(dims, dims)], P[np.ix_(dims, rest)])
    A2 = A.copy()
    A2[:, dims] = A[:, rest] @ M.T
    return A2


def score_from_aug(det, X, A):
    rc = {}
    r = det.gcr.resid(A) - det.bias_gcr; rc['gcr'] = r ** 2
    e = (A[:, :det.p] - det.ae.predict(A)) / det.ae_sd - det.bias_ae; rc['ae'] = e ** 2
    rc['rule'] = det.norm.rule_viol(X) * det.rule_on[None, :]
    return (det._phi(rc, smooth=False) / det.theta_f).sum(1)


def deletion_fidelity(det, X, rows, attr, k=5, mode='cond', rng=None):
    """Relative drop of the raw fused score after 'removing' the top-k tags of `attr` at `rows`.
    attr: (len(rows), n_tags).  Removal = conditional-Gaussian imputation (mode='cond') or
    replacement by the normal mean (mode='mean'); constant tags are reset to their nominal value."""
    A = det.norm.aug(X)
    base = score_from_aug(det, X[rows], A[rows])
    keep_pos = -np.ones(len(det.feats), int); keep_pos[det.idx_keep] = np.arange(det.p)
    drops = []
    for j, r in enumerate(rows):
        top = np.argsort(-attr[j])[:k]
        Xr = X[r:r + 1].copy(); Ar = A[r:r + 1].copy()
        for t in top:
            if det.norm.const[t]:
                Xr[0, t] = det.norm.mu[t]
        kt = np.array([keep_pos[t] for t in top if keep_pos[t] >= 0], int)
        if len(kt):
            if mode == 'cond':
                Ar = _cond_impute_aug(det, Ar, kt)
            else:
                Ar[:, kt] = 0; Ar[:, kt + det.p] = 0
        s2 = score_from_aug(det, Xr, Ar)[0]
        drops.append((base[j] - s2) / (base[j] + 1e-9))
    return np.array(drops)


def shapley_sampling(det, X, rows, M=16, seed=0):
    """Permutation-sampling Shapley values (Strumbelj & Kononenko) of the raw fused score over tags,
    with absent tags set to their normal mean (z = 0, constant tags nominal)."""
    rng = np.random.default_rng(seed)
    A = det.norm.aug(X)
    nf = len(det.feats); p = det.p
    keep_pos = -np.ones(nf, int); keep_pos[det.idx_keep] = np.arange(p)
    out = np.zeros((len(rows), nf))
    for j, r in enumerate(rows):
        for m in range(M):
            perm = rng.permutation(nf)
            # build the chain of coalitions: start from baseline (all absent), add tags in perm order
            Ab = np.zeros((nf + 1, A.shape[1]), np.float32)
            Xb = np.repeat(det.norm.mu[None, :].astype(np.float32), nf + 1, 0)
            for i, t in enumerate(perm):
                Ab[i + 1] = Ab[i]; Xb[i + 1] = Xb[i]
                if keep_pos[t] >= 0:
                    Ab[i + 1, keep_pos[t]] = A[r, keep_pos[t]]; Ab[i + 1, keep_pos[t] + p] = A[r, keep_pos[t] + p]
                else:
                    Xb[i + 1, t] = X[r, t]
            v = score_from_aug(det, Xb, Ab)
            out[j, perm] += np.diff(v)
    return out / M


def topk(a, k=5):
    return set(np.argsort(-a)[:k].tolist())


def jacc(a, b):
    return len(a & b) / max(len(a | b), 1)


def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b).correlation
    return float(r) if np.isfinite(r) else 0.0


def grounded_attr(phi, open_id, A_graph, tau=0.005):
    """Situation-grounded attribution: per-tag attribution accumulated (slow EWMA) inside the current
    incident, restarted when a new incident opens, then diffused one hop over the asset graph."""
    T, n = phi.shape
    g = np.zeros_like(phi); acc = np.zeros(n); cur = -1
    for t in range(T):
        if open_id[t] != cur:
            acc[:] = 0; cur = open_id[t]
        if cur >= 0:
            acc = (1 - tau) * acc + phi[t]
        g[t] = acc
    g = 0.5 * g + 0.5 * g @ A_graph.T
    return g
