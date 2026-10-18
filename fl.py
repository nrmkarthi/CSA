"""Privacy-preserving distributed learning (M7) for XAI-CSA.

Sites hold disjoint normal-operation periods of the plant (non-IID operating regimes).
 * normaliser and Gaussian conditional model: exact aggregation of additive sufficient statistics
   (compatible with secure aggregation - only sums leave a site)
 * auto-encoder: FedAvg (or Byzantine-robust median / trimmed mean) over local Adam epochs,
   optional update clipping + Gaussian noise (central DP-style), optional poisoning client.
"""
import numpy as np
from detect import (Normalizer, GCR, Detector, sum_stats, norm_from_stats, aug_cov_stats,
                    cov_from_stats, new_ae)


def make_sites(D, n_sites=6):
    """Split each normal-operation client file into halves -> 6 sites with different regimes."""
    sites = []
    for c in D['clients']:
        h = len(c) // 2
        sites += [c[:h], c[h:]]
    return sites[:n_sites]


def _flat(m):
    return np.concatenate([w.ravel() for w in m.coefs_ + m.intercepts_])


def _set(m, v):
    i = 0
    for w in m.coefs_ + m.intercepts_:
        n = w.size; w[...] = v[i:i + n].reshape(w.shape); i += n


def train_federated(D, seed, rounds=10, local_epochs=1, agg='fedavg', dp_sigma=0.0, clip=None,
                    poison=None, sites=None, sub=4, trim=1, snap_every=None):
    rng = np.random.default_rng(seed)
    sites = make_sites(D) if sites is None else sites
    comm = dict(bytes_up=0, bytes_down=0)
    # --- round 0: secure aggregation of moments -> global normaliser
    st = [sum_stats(s) for s in sites]
    comm['bytes_up'] += sum(8 * (1 + 2 * s[1].size) for s in st)
    mu, sd = norm_from_stats(st)
    nm = Normalizer(mu, sd)
    As = [nm.aug(s) for s in sites]
    p = int(nm.keep.sum())
    cst = [aug_cov_stats(a[::5]) for a in As]
    comm['bytes_up'] += sum(8 * (1 + c[1].size + c[2].size) for c in cst)
    gcr = GCR(cov_from_stats(cst), p)
    # --- auto-encoder FedAvg
    models = [new_ae(seed) for _ in sites]
    Asub = [a[::sub] for a in As]
    for m, a in zip(models, Asub):
        m.partial_fit(a[:512], a[:512, :p])        # initialise shapes
    g = _flat(models[0]).copy()
    for m in models:
        _set(m, g)
    nparam = g.size
    w = np.array([len(a) for a in Asub], float); w /= w.sum()
    snaps = []
    for r in range(rounds):
        upd = []
        for i, (m, a) in enumerate(zip(models, Asub)):
            _set(m, g)
            for e in range(local_epochs):
                idx = rng.permutation(len(a))
                m.partial_fit(a[idx], a[idx, :p])
            d = _flat(m) - g
            if clip is not None:
                d = d * min(1.0, clip / (np.linalg.norm(d) + 1e-12))
            if poison is not None and i == poison['site']:
                if poison['kind'] == 'flip':
                    d = -poison['scale'] * d
                elif poison['kind'] == 'noise':
                    d = rng.normal(0, poison['scale'] * np.std(d), d.shape)
            upd.append(d)
        comm['bytes_down'] += 4 * nparam * len(sites); comm['bytes_up'] += 4 * nparam * len(sites)
        U = np.stack(upd)
        if agg == 'fedavg':
            delta = (w[:, None] * U).sum(0)
        elif agg == 'median':
            delta = np.median(U, 0)
        elif agg == 'trimmed':
            S = np.sort(U, 0); delta = S[trim:len(U) - trim].mean(0)
        if dp_sigma > 0:
            C = clip if clip is not None else np.median(np.linalg.norm(U, axis=1))
            delta = delta + rng.normal(0, dp_sigma * C / len(sites), delta.shape)
        g = g + delta
        if snap_every and (r + 1) % snap_every == 0:
            import copy
            mm = copy.deepcopy(models[0]); _set(mm, g)
            snaps.append((r + 1, Detector(nm, gcr, mm, D['feats'], D['proc_of']).calibrate(D['cal'])))
    final = models[0]; _set(final, g)
    det = Detector(nm, gcr, final, D['feats'], D['proc_of']).calibrate(D['cal'])
    comm['n_params_ae'] = int(nparam)
    comm['raw_bytes_central'] = int(sum(4 * s.size for s in sites))
    if snap_every:
        return det, comm, snaps
    return det, comm


def train_local(D, seed, site_X, cal_frac=0.2, sub=4, epochs=10):
    """Local-only training: a site trains on its own data and calibrates on its own held-out tail."""
    from detect import ae_train_central
    n = int(len(site_X) * (1 - cal_frac))
    fit, cal = site_X[:n], site_X[n:]
    mu, sd = norm_from_stats([sum_stats(fit)])
    nm = Normalizer(mu, sd)
    A = nm.aug(fit); p = int(nm.keep.sum())
    gcr = GCR(cov_from_stats([aug_cov_stats(A[::5])]), p)
    ae = ae_train_central(A, p, seed, epochs=epochs, sub=sub)
    # feature set must match the global one (constant tags may differ per site)
    return Detector(nm, gcr, ae, D['feats'], D['proc_of']).calibrate(cal)
